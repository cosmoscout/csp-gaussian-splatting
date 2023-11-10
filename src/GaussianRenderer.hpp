////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_GAUSSIAN_SPLATTING_GAUSSIAN_RENDERER_HPP
#define CSP_GAUSSIAN_SPLATTING_GAUSSIAN_RENDERER_HPP

#include "Plugin.hpp"
#include "Eigen/Eigen"
#include "renderer/GaussianData.hpp"

//#include "renderer/SplatRenderer.hpp"
#include "renderer/SurfaceRenderer.hpp"

#include "../../../src/cs-core/EclipseShadowReceiver.hpp"
#include "../../../src/cs-scene/CelestialObject.hpp"

#include <cuda_runtime.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

namespace cs::core {
class SolarSystem;
} // namespace cs::core

namespace csp::gaussiansplatting {

class GaussianRenderer : public cs::scene::CelestialObject, public IVistaOpenGLDraw {
 public:
  typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Vector3f;
  typedef Eigen::Matrix<int, 3, 1, Eigen::DontAlign>   Vector3i;
  typedef	Eigen::Matrix<float, 4, 4, Eigen::DontAlign, 4, 4>	Matrix4f;
  
  GaussianRenderer(std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string objectName);

  GaussianRenderer(GaussianRenderer const& other) = delete;
  GaussianRenderer(GaussianRenderer&& other)      = default;

  GaussianRenderer& operator=(GaussianRenderer const& other) = delete;
  GaussianRenderer& operator=(GaussianRenderer&& other) = delete;

  ~GaussianRenderer() override;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::RadianceField const& settings, int32_t cudaDevice, int32_t shDegree);
  Plugin::Settings::RadianceField const& getRadianceField();

  void update();

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:

  void loadPLY();

  std::string                            mObjectName;
  std::shared_ptr<cs::core::Settings>    mSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  Plugin::Settings::RadianceField mRadianceField;

  bool mShaderDirty              = true;
  int  mEnableLightingConnection = -1;
  int  mEnableHDRConnection      = -1;

  int32_t mCudaDevice = 0;
  int32_t mSHDegree = 0;
  int32_t mCount = 0;

  Vector3f mSceneMin{};
  Vector3f mSceneMax{};

  float* mPosCuda = nullptr;
  float* mRotCuda = nullptr;
  float* mScaleCuda = nullptr;
  float* mOpacityCuda = nullptr;
  float* mShsCuda = nullptr;
  float* mViewCuda = nullptr;
  float* mProjCuda = nullptr;
  float* mCamPosCuda = nullptr;
  float* mBackgroundCuda = nullptr;
  int*   mRectCuda = nullptr;

  GLuint                 mImageBuffer;
  cudaGraphicsResource_t mImageBufferCuda;

  GaussianData* mData;

  bool              mInteropFailed = false;
  std::vector<char> mFallbackBytes;
  float*            mFallbackBufferCuda = nullptr;

  size_t                         mAllocdGeom = 0, mAllocdBinning = 0, mAllocdImg = 0;
  void *                         mGeomPtr = nullptr, *mBinningPtr = nullptr, *mImgPtr = nullptr;
  std::function<char*(size_t N)> mGeomBufferFunc, mBinningBufferFunc, mImgBufferFunc;

  std::shared_ptr<SurfaceRenderer>   mSurfaceRenderer;
  // std::shared_ptr<SplatRenderer>   mSplatRenderer;

  struct {
    uint32_t modelMatrix       = 0;
    uint32_t viewMatrix        = 0;
    uint32_t projectionMatrix  = 0;
    uint32_t surfaceTexture    = 0;
    uint32_t radii             = 0;
    uint32_t sunIlluminance    = 0;
    uint32_t ambientBrightness = 0;
    uint32_t litSideVisible    = 0;
  } mUniforms;
};
} // namespace csp::gaussiansplatting

#endif // CSP_GAUSSIAN_SPLATTING_GAUSSIAN_RENDERER_HPP
