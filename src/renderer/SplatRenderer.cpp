////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#include "../../../../src/cs-utils/filesystem.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <rasterizer.h>
#include <thread>

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::gaussiansplatting {

////////////////////////////////////////////////////////////////////////////////////////////////////

typedef Eigen::Matrix<float, 4, 4, Eigen::DontAlign, 4, 4> Matrix4f;

////////////////////////////////////////////////////////////////////////////////////////////////////

SplatRenderer::SplatRenderer() {
  // Create space for view parameters
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc(reinterpret_cast<void**>(&mViewCuda), sizeof(Matrix4f)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc(reinterpret_cast<void**>(&mProjCuda), sizeof(Matrix4f)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc(reinterpret_cast<void**>(&mCamPosCuda), 3 * sizeof(float)));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc(reinterpret_cast<void**>(&mBackgroundCuda), 3 * sizeof(float)));

  // The background color is actually not used in our fork of diff-gaussian-rasterization.
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
      return static_cast<char*>(*ptr);
    };
    return lambda;
  };

  mGeomBufferFunc    = resizeFunctional(&mGeomPtr, mAllocdGeom);
  mBinningBufferFunc = resizeFunctional(&mBinningPtr, mAllocdBinning);
  mImgBufferFunc     = resizeFunctional(&mImgPtr, mAllocdImg);

  mCopyShader.InitVertexShaderFromString(
      cs::utils::filesystem::loadToString("../share/resources/shaders/copy.vert"));
  mCopyShader.InitFragmentShaderFromString(
      cs::utils::filesystem::loadToString("../share/resources/shaders/copy.frag"));
  mCopyShader.Link();

  mUniforms.mWidth  = mCopyShader.GetUniformLocation("width");
  mUniforms.mHeight = mCopyShader.GetUniformLocation("height");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SplatRenderer::~SplatRenderer() {
  cudaFree(mViewCuda);
  cudaFree(mProjCuda);
  cudaFree(mCamPosCuda);
  cudaFree(mBackgroundCuda);

  for (auto const& [viewport, data] : mViewportData) {
    if (!data.mInteropFailed) {
      cudaGraphicsUnregisterResource(data.mImageBufferCuda);
    } else {
      cudaFree(data.mFallbackBufferCuda);
    }
    glDeleteBuffers(1, &data.mImageBuffer);
  }

  if (mGeomPtr) {
    cudaFree(mGeomPtr);
  }

  if (mBinningPtr) {
    cudaFree(mBinningPtr);
  }

  if (mImgPtr) {
    cudaFree(mImgPtr);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SplatRenderer::draw(float scale, int count, bool doFade, const GaussianData& mesh,
    glm::vec3 const& camPos, glm::mat4 matMV, glm::mat4 matP) {

  auto& viewportData = getCurrentViewportData();

  float tanFoVY = 1.f / matP[1][1];
  float tanFoVX = tanFoVY * (1.f * viewportData.mWidth / viewportData.mHeight);

  // Convert view and projection to target coordinate system. This is done in the same manner as in
  // the original implementation:
  // https://gitlab.inria.fr/sibr/sibr_core/-/blob/fossa_compatibility/src/projects/gaussianviewer/renderer/GaussianView.cpp#L469
  matP  = matP * matMV;
  matMV = glm::row(matMV, 1, -1.f * glm::row(matMV, 1));
  matMV = glm::row(matMV, 2, -1.f * glm::row(matMV, 2));
  matP  = glm::row(matP, 1, -1.f * glm::row(matP, 1));

  // Copy frame-dependent data to GPU.
  CUDA_SAFE_CALL(
      cudaMemcpy(mViewCuda, glm::value_ptr(matMV), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(mProjCuda, glm::value_ptr(matP), sizeof(glm::mat4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(mCamPosCuda, glm::value_ptr(camPos), sizeof(float) * 3, cudaMemcpyHostToDevice));

  // Map the OpenGL buffer for use with CUDA or use the fallback buffer if interop failed.
  float* image_cuda = nullptr;
  if (!viewportData.mInteropFailed) {
    size_t bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &viewportData.mImageBufferCuda));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&image_cuda), &bytes, viewportData.mImageBufferCuda));
  } else {
    image_cuda = viewportData.mFallbackBufferCuda;
  }

  // Draw the radiance field!
  CudaRasterizer::Rasterizer::forward(mGeomBufferFunc, mBinningBufferFunc, mImgBufferFunc, count, 3,
      16, mBackgroundCuda, viewportData.mWidth, viewportData.mHeight, mesh.mPosCuda, mesh.mShsCuda,
      nullptr, mesh.mOpacityCuda, mesh.mScaleCuda, scale, mesh.mRotCuda, nullptr, mViewCuda,
      mProjCuda, mCamPosCuda, tanFoVX, tanFoVY, false, doFade, image_cuda);

  // Unmap the OpenGL resource or manually copy the data.
  if (!viewportData.mInteropFailed) {
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &viewportData.mImageBufferCuda));
  } else {
    CUDA_SAFE_CALL(cudaMemcpy(viewportData.mFallbackBytes.data(), viewportData.mFallbackBufferCuda,
        viewportData.mFallbackBytes.size(), cudaMemcpyDeviceToHost));
    glNamedBufferSubData(viewportData.mImageBuffer, 0, viewportData.mFallbackBytes.size(),
        viewportData.mFallbackBytes.data());
  }

  // Copy image contents to frame buffer. For this, we disable the depth test, depth writing, and
  // perform pre-multiplied alpha blending.
  mCopyShader.Bind();

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);

  glUniform1i(mUniforms.mWidth, viewportData.mWidth);
  glUniform1i(mUniforms.mHeight, viewportData.mHeight);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, viewportData.mImageBuffer);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);

  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);

  mCopyShader.Release();

  if (cudaPeekAtLastError() != cudaSuccess) {
    logger().error(
        "A CUDA error occurred during rendering: {}", cudaGetErrorString(cudaGetLastError()));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SplatRenderer::ViewportData& SplatRenderer::getCurrentViewportData() {

  std::array<GLint, 4> viewportExtent{};
  glGetIntegerv(GL_VIEWPORT, viewportExtent.data());
  GLint width  = viewportExtent.at(2);
  GLint height = viewportExtent.at(3);

  auto vistaViewport   = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  auto viewportData    = mViewportData.find(vistaViewport);
  bool needsRecreation = false;

  // If we haven't seen this viewport before or it changed size, we need to create new resources for
  // it.
  if (viewportData == mViewportData.end()) {
    needsRecreation = true;

  } else if (viewportData->second.mWidth != width || viewportData->second.mHeight != height) {

    // If the viewport changed size, we have to free any previously created resources first.
    if (!viewportData->second.mInteropFailed) {
      cudaGraphicsUnregisterResource(viewportData->second.mImageBufferCuda);
    } else {
      cudaFree(viewportData->second.mFallbackBufferCuda);
    }
    glDeleteBuffers(1, &viewportData->second.mImageBuffer);

    needsRecreation = true;
  }

  // Create OpenGL buffer ready for CUDA/GL interop.
  if (needsRecreation) {

    ViewportData data;

    bool useInterop = true;

    glCreateBuffers(1, &data.mImageBuffer);
    glNamedBufferStorage(
        data.mImageBuffer, width * height * 4 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

    if (useInterop) {
      if (cudaPeekAtLastError() != cudaSuccess) {
        logger().error(
            "A CUDA error occurred in setup: {}", cudaGetErrorString(cudaGetLastError()));
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

    data.mWidth  = width;
    data.mHeight = height;

    // Store the new viewport data so that we can reuse it next time.
    mViewportData[vistaViewport] = data;
  }

  return mViewportData[vistaViewport];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::gaussiansplatting
