////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#ifndef CSP_GAUSSIAN_SPLATTING_GAUSSIAN_VIEW_HPP
#define CSP_GAUSSIAN_SPLATTING_GAUSSIAN_VIEW_HPP

#include <VistaOGLExt/VistaGLSLShader.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <functional>
#include <glm/glm.hpp>
#include <memory>

class VistaViewport;

namespace csp::gaussiansplatting {

/// This renderer uses the diff-gaussian-rasterization submodule to render the radiance
/// field. The individual gaussian splats are first ray-casted with CUDA and then
/// resulting image is then drawn on top of the OpenGL frame buffer.
class SplatRenderer {

 public:
  SplatRenderer();
  virtual ~SplatRenderer();

  void draw(float scale, int count, bool doFade, const GaussianData& mesh, glm::vec3 const& camPos,
      glm::mat4 matMV, glm::mat4 matP);

 private:
  /// This stores the CUDA and OpenGL resources required for rendering. As the draw method above can
  /// be called for different viewports with different resolutions, we have to store such a struct
  /// for each viewport.
  struct ViewportData {
    uint32_t mWidth  = 0;
    uint32_t mHeight = 0;

    GLuint                 mImageBuffer = 0;
    cudaGraphicsResource_t mImageBufferCuda;

    bool              mInteropFailed = false;
    std::vector<char> mFallbackBytes;
    float*            mFallbackBufferCuda = nullptr;
  };

  /// Returns the ViewportData struct for the viewport which is currently rendered.
  ViewportData& getCurrentViewportData();

  std::unordered_map<VistaViewport*, ViewportData> mViewportData;

  /// CUDA buffers which store the view and projection parameters.
  float* mViewCuda       = nullptr;
  float* mProjCuda       = nullptr;
  float* mCamPosCuda     = nullptr;
  float* mBackgroundCuda = nullptr;

  size_t mAllocdGeom    = 0;
  size_t mAllocdBinning = 0;
  size_t mAllocdImg     = 0;

  void* mGeomPtr    = nullptr;
  void* mBinningPtr = nullptr;
  void* mImgPtr     = nullptr;

  std::function<char*(size_t N)> mGeomBufferFunc;
  std::function<char*(size_t N)> mBinningBufferFunc;
  std::function<char*(size_t N)> mImgBufferFunc;

  struct {
    uint32_t mWidth  = 0;
    uint32_t mHeight = 0;
  } mUniforms;

  VistaGLSLShader mCopyShader;
};

} // namespace csp::gaussiansplatting

#endif