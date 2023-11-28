////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#include "GaussianData.hpp"

#include <cuda_runtime.h>

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

GaussianData::GaussianData(std::vector<Pos> const& pos, std::vector<Rot> const& rot,
    std::vector<Scale> const& scale, std::vector<float> const& alpha,
    std::vector<SHs<3>> const& color) {
  auto count = pos.size();

  glCreateBuffers(1, &mPosOpenGL);
  glCreateBuffers(1, &mRotOpenGL);
  glCreateBuffers(1, &mScaleOpenGL);
  glCreateBuffers(1, &mAlphaOpenGL);
  glCreateBuffers(1, &mColorOpenGL);
  glNamedBufferStorage(mPosOpenGL, count * 3 * sizeof(float), pos.data(), 0);
  glNamedBufferStorage(mRotOpenGL, count * 4 * sizeof(float), rot.data(), 0);
  glNamedBufferStorage(mScaleOpenGL, count * 3 * sizeof(float), scale.data(), 0);
  glNamedBufferStorage(mAlphaOpenGL, count * sizeof(float), alpha.data(), 0);
  glNamedBufferStorage(mColorOpenGL, count * sizeof(float) * 48, color.data(), 0);

  // Allocate and fill the GPU data
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc(reinterpret_cast<void**>(&mPosCuda), sizeof(Pos) * count));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(mPosCuda, pos.data(), sizeof(Pos) * count, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc(reinterpret_cast<void**>(&mRotCuda), sizeof(Rot) * count));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(mRotCuda, rot.data(), sizeof(Rot) * count, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc(reinterpret_cast<void**>(&mShsCuda), sizeof(SHs<3>) * count));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(mShsCuda, color.data(), sizeof(SHs<3>) * count, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc(reinterpret_cast<void**>(&mOpacityCuda), sizeof(float) * count));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(mOpacityCuda, alpha.data(), sizeof(float) * count, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL_ALWAYS(cudaMalloc(reinterpret_cast<void**>(&mScaleCuda), sizeof(Scale) * count));
  CUDA_SAFE_CALL_ALWAYS(
      cudaMemcpy(mScaleCuda, scale.data(), sizeof(Scale) * count, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GaussianData::~GaussianData() {
  glDeleteBuffers(1, &mPosOpenGL);
  glDeleteBuffers(1, &mRotOpenGL);
  glDeleteBuffers(1, &mScaleOpenGL);
  glDeleteBuffers(1, &mAlphaOpenGL);
  glDeleteBuffers(1, &mColorOpenGL);

  cudaFree(mPosCuda);
  cudaFree(mRotCuda);
  cudaFree(mScaleCuda);
  cudaFree(mOpacityCuda);
  cudaFree(mShsCuda);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::gaussiansplatting
