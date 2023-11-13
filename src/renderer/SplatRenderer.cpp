/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include "../../../../src/cs-utils/filesystem.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <rasterizer.h>
#include <thread>

#define CUDA_SAFE_CALL_ALWAYS(A)                                                                   \
  A;                                                                                               \
  cudaDeviceSynchronize();                                                                         \
  if (cudaPeekAtLastError() != cudaSuccess)                                                        \
    logger().error(cudaGetErrorString(cudaGetLastError()));

#if DEBUG || _DEBUG
#define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
#define CUDA_SAFE_CALL(A) A
#endif

namespace csp::gaussiansplatting {

typedef	Eigen::Matrix<float, 4, 4, Eigen::DontAlign, 4, 4>	Matrix4f;

SplatRenderer::SplatRenderer() {
 // Create space for view parameters
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mViewCuda, sizeof(Matrix4f)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mProjCuda, sizeof(Matrix4f)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mCamPosCuda, 3 * sizeof(float)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mBackgroundCuda, 3 * sizeof(float)));
  
  // The background color is actually not used.
  float bg[3] = {0.f, 0.f, 0.f};
  CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(mBackgroundCuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

  auto resizeFunctional = [](void** ptr, size_t& S) {
    auto lambda = [ptr, &S](size_t N) {
      if (N > S) {
        if (*ptr)
          CUDA_SAFE_CALL(cudaFree(*ptr));
        CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
        S = 2 * N;
      }
      return reinterpret_cast<char*>(*ptr);
    };
    return lambda;
  };

  mGeomBufferFunc    = resizeFunctional(&mGeomPtr, mAllocdGeom);
  mBinningBufferFunc = resizeFunctional(&mBinningPtr, mAllocdBinning);
  mImgBufferFunc     = resizeFunctional(&mImgPtr, mAllocdImg);

  mCopyShader.InitVertexShaderFromString(cs::utils::filesystem::loadToString("../share/resources/shaders/copy.vert"));
  mCopyShader.InitFragmentShaderFromString(cs::utils::filesystem::loadToString("../share/resources/shaders/copy.frag"));
  mCopyShader.Link();

  mUniforms.mWidth = glGetUniformLocation(mCopyShader.GetProgram(), "width");
  mUniforms.mHeight = glGetUniformLocation(mCopyShader.GetProgram(), "height");
}

SplatRenderer::~SplatRenderer() {
  cudaFree(mViewCuda);
  cudaFree(mProjCuda);
  cudaFree(mCamPosCuda);
  cudaFree(mBackgroundCuda);

  if (!mInteropFailed) {
    cudaGraphicsUnregisterResource(mImageBufferCuda);
  } else {
    cudaFree(mFallbackBufferCuda);
  }
  glDeleteBuffers(1, &mImageBuffer);

  if (mGeomPtr)
    cudaFree(mGeomPtr);
  if (mBinningPtr)
    cudaFree(mBinningPtr);
  if (mImgPtr)
    cudaFree(mImgPtr);
}

void SplatRenderer::draw(float scale, int count,
     const GaussianData& mesh,
     glm::vec3 const& camPos, glm::mat4 matMV, glm::mat4 matP) {
    
     std::array<GLint, 4> current_viewport{};
  glGetIntegerv(GL_VIEWPORT, current_viewport.data());
  createBuffers(current_viewport[2], current_viewport[3]);  

  float tan_fovy = 1.f / matP[1][1];
  float tan_fovx = tan_fovy * (1.f * mWidth / mHeight);

  // Convert view and projection to target coordinate system
  matP = matP * matMV;
  matMV = glm::row(matMV, 1, -1.f * glm::row(matMV, 1));
  matMV = glm::row(matMV, 2, -1.f * glm::row(matMV, 2));
  matP = glm::row(matP, 1, -1.f * glm::row(matP, 1));

  // Copy frame-dependent data to GPU
  CUDA_SAFE_CALL(
      cudaMemcpy(mViewCuda, glm::value_ptr(matMV), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(mProjCuda, glm::value_ptr(matP), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(mCamPosCuda, glm::value_ptr(camPos), sizeof(float) * 3, cudaMemcpyHostToDevice));
  

  float* image_cuda = nullptr;
  if (!mInteropFailed) {
    // Map OpenGL buffer resource for use with CUDA
    size_t bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &mImageBufferCuda));
    CUDA_SAFE_CALL(
        cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, mImageBufferCuda));
  } else {
    image_cuda = mFallbackBufferCuda;
  }

  // Rasterize
  bool _fastCulling = true;
  int*   rects  = _fastCulling ? mesh.mRectCuda : nullptr;

  // Can be used for cropping
  float* boxmin = nullptr;
  float* boxmax = nullptr;
  
  CudaRasterizer::Rasterizer::forward(mGeomBufferFunc, mBinningBufferFunc, mImgBufferFunc, count,
      3, 16, mBackgroundCuda, mWidth, mHeight, mesh.mPosCuda, mesh.mShsCuda,
      nullptr, mesh.mOpacityCuda, mesh.mScaleCuda, scale, mesh.mRotCuda, nullptr, mViewCuda,
      mProjCuda, mCamPosCuda, tan_fovx, tan_fovy, false, image_cuda);//, nullptr, rects, boxmin, boxmax);


  if (!mInteropFailed) {
    // Unmap OpenGL resource for use with OpenGL
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &mImageBufferCuda));
  } else {
    CUDA_SAFE_CALL(cudaMemcpy(mFallbackBytes.data(), mFallbackBufferCuda, mFallbackBytes.size(),
        cudaMemcpyDeviceToHost));
    glNamedBufferSubData(mImageBuffer, 0, mFallbackBytes.size(), mFallbackBytes.data());
  }
  

  // Copy image contents to framebuffer
  mCopyShader.Bind();

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);

  glUniform1i(mUniforms.mWidth, mWidth);
  glUniform1i(mUniforms.mHeight, mHeight);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mImageBuffer);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);

  mCopyShader.Release();

  if (cudaPeekAtLastError() != cudaSuccess) {
    logger().error("A CUDA error occurred during rendering: {}", cudaGetErrorString(cudaGetLastError()));
  }
 
}

void SplatRenderer::createBuffers(uint32_t width, uint32_t height) {
  if (mWidth != width || mHeight != height) {

    // Clean any previous buffer
    if (mImageBufferCuda && !mInteropFailed) {
      cudaGraphicsUnregisterResource(mImageBufferCuda);
    } else if (mFallbackBufferCuda) {
      cudaFree(mFallbackBufferCuda);
    }

    if (mImageBuffer) {
      glDeleteBuffers(1, &mImageBuffer);
    }

    // Create GL buffer ready for CUDA/GL interop
    bool useInterop = true;

    glCreateBuffers(1, &mImageBuffer);
    glNamedBufferStorage(
        mImageBuffer, width * height * 4 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

    if (useInterop) {
      if (cudaPeekAtLastError() != cudaSuccess) {
        logger().error("A CUDA error occurred in setup: {}", cudaGetErrorString(cudaGetLastError()));
      }
      cudaGraphicsGLRegisterBuffer(
          &mImageBufferCuda, mImageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
      useInterop &= (cudaGetLastError() == cudaSuccess);
    }

    if (!useInterop) {
      mFallbackBytes.resize(width * height * 4 * sizeof(float));
      cudaMalloc(&mFallbackBufferCuda, mFallbackBytes.size());
      mInteropFailed = true;
    }

    mWidth = width;
    mHeight = height;
  }

}

}
