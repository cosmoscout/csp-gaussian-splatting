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

#ifndef CSP_GAUSSIAN_SPLATTING_GAUSSIAN_VIEW_HPP
#define CSP_GAUSSIAN_SPLATTING_GAUSSIAN_VIEW_HPP

#include <VistaOGLExt/VistaGLSLShader.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <functional>
#include <memory>

class VistaViewport;

namespace CudaRasterizer {
class Rasterizer;
}

namespace csp::gaussiansplatting {

class BufferCopyRenderer;

class SplatRenderer {

 public:
  SplatRenderer();
  virtual ~SplatRenderer();

  void draw(float scale, int count, bool doFade,
     const GaussianData& mesh,
      glm::vec3 const& camPos,  glm::mat4  matMV,  glm::mat4  matP);

 private:
 struct ViewportData {
    uint32_t mWidth = 0;
    uint32_t mHeight = 0;

    GLuint                 mImageBuffer = 0;
    cudaGraphicsResource_t mImageBufferCuda;

    bool              mInteropFailed = false;
    std::vector<char> mFallbackBytes;
    float*            mFallbackBufferCuda = nullptr;
  };

  std::unordered_map<VistaViewport*, ViewportData> mViewportData;

  ViewportData& getCurrentViewportData();

  float* mViewCuda = nullptr;
  float* mProjCuda = nullptr;
  float* mCamPosCuda = nullptr;
  float* mBackgroundCuda = nullptr;

  size_t                         mAllocdGeom = 0, mAllocdBinning = 0, mAllocdImg = 0;
  void *                         mGeomPtr = nullptr, *mBinningPtr = nullptr, *mImgPtr = nullptr;
  std::function<char*(size_t N)> mGeomBufferFunc, mBinningBufferFunc, mImgBufferFunc;

  struct {
    uint32_t mWidth    = 0;
    uint32_t mHeight = 0;
  } mUniforms;

  VistaGLSLShader mCopyShader;
};

} // namespace csp::gaussiansplatting

#endif