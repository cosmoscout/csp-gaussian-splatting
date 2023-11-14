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

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
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

  for (auto const& viewportData: mViewportData) {
    if (!viewportData.second.mInteropFailed) {
      cudaGraphicsUnregisterResource(viewportData.second.mImageBufferCuda);
    } else {
      cudaFree(viewportData.second.mFallbackBufferCuda);
    }
    glDeleteBuffers(1, &viewportData.second.mImageBuffer);
  }

  if (mGeomPtr)
    cudaFree(mGeomPtr);
  if (mBinningPtr)
    cudaFree(mBinningPtr);
  if (mImgPtr)
    cudaFree(mImgPtr);
}

void SplatRenderer::draw(float scale, int count, bool doFade,
     const GaussianData& mesh,
     glm::vec3 const& camPos, glm::mat4 matMV, glm::mat4 matP) {

  auto& viewportData = getCurrentViewportData();  

  float tan_fovy = 1.f / matP[1][1];
  float tan_fovx = tan_fovy * (1.f * viewportData.mWidth / viewportData.mHeight);

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
  if (!viewportData.mInteropFailed) {
    // Map OpenGL buffer resource for use with CUDA
    size_t bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &viewportData.mImageBufferCuda));
    CUDA_SAFE_CALL(
        cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, viewportData.mImageBufferCuda));
  } else {
    image_cuda = viewportData.mFallbackBufferCuda;
  }

  CudaRasterizer::Rasterizer::forward(mGeomBufferFunc, mBinningBufferFunc, mImgBufferFunc, count,
      3, 16, mBackgroundCuda, viewportData.mWidth, viewportData.mHeight, mesh.mPosCuda, mesh.mShsCuda,
      nullptr, mesh.mOpacityCuda, mesh.mScaleCuda, scale, mesh.mRotCuda, nullptr, mViewCuda,
      mProjCuda, mCamPosCuda, tan_fovx, tan_fovy, false, doFade, image_cuda);


  if (!viewportData.mInteropFailed) {
    // Unmap OpenGL resource for use with OpenGL
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &viewportData.mImageBufferCuda));
  } else {
    CUDA_SAFE_CALL(cudaMemcpy(viewportData.mFallbackBytes.data(), viewportData.mFallbackBufferCuda, viewportData.mFallbackBytes.size(),
        cudaMemcpyDeviceToHost));
    glNamedBufferSubData(viewportData.mImageBuffer, 0, viewportData.mFallbackBytes.size(), viewportData.mFallbackBytes.data());
  }
  

  // Copy image contents to framebuffer
  mCopyShader.Bind();

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);

  glUniform1i(mUniforms.mWidth, viewportData.mWidth);
  glUniform1i(mUniforms.mHeight, viewportData.mHeight);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, viewportData.mImageBuffer);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);

  mCopyShader.Release();

  if (cudaPeekAtLastError() != cudaSuccess) {
    logger().error("A CUDA error occurred during rendering: {}", cudaGetErrorString(cudaGetLastError()));
  }
 
}

SplatRenderer::ViewportData& SplatRenderer::getCurrentViewportData() {

  std::array<GLint, 4> viewportExtent{};
  glGetIntegerv(GL_VIEWPORT, viewportExtent.data());
  GLint width = viewportExtent.at(2);
  GLint height = viewportExtent.at(3);

  auto vistaViewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  auto viewportData  = mViewportData.find(vistaViewport);
  bool needsRecreation = false;

  if (viewportData == mViewportData.end()) {
    needsRecreation = true;

  } else if (viewportData->second.mWidth != width || viewportData->second.mHeight != height) {

    // Clean any previous buffer
    if (!viewportData->second.mInteropFailed) {
      cudaGraphicsUnregisterResource(viewportData->second.mImageBufferCuda);
    } else {
      cudaFree(viewportData->second.mFallbackBufferCuda);
    }
    glDeleteBuffers(1, &viewportData->second.mImageBuffer);

    needsRecreation = true;
  }


  if (needsRecreation) {

    std::cout << "#####################################" << std::endl;

    ViewportData data;

    // Create GL buffer ready for CUDA/GL interop
    bool useInterop = true;

    glCreateBuffers(1, &data.mImageBuffer);
    glNamedBufferStorage(
        data.mImageBuffer, width * height * 4 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

    if (useInterop) {
      if (cudaPeekAtLastError() != cudaSuccess) {
        logger().error("A CUDA error occurred in setup: {}", cudaGetErrorString(cudaGetLastError()));
      }
      cudaGraphicsGLRegisterBuffer(
          &data.mImageBufferCuda, data.mImageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
      useInterop &= (cudaGetLastError() == cudaSuccess);
    }

    if (!useInterop) {
      data.mFallbackBytes.resize(width * height * 4 * sizeof(float));
      cudaMalloc(&data.mFallbackBufferCuda, data.mFallbackBytes.size());
      data.mInteropFailed = true;
    }

    data.mWidth = width;
    data.mHeight = height;

    mViewportData[vistaViewport] = data;
  }

  return mViewportData[vistaViewport];
}

}
