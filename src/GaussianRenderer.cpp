////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "GaussianRenderer.hpp"

#include "logger.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace csp::gaussiansplatting {

namespace {

float sigmoid(const float m1) {
  return 1.0f / (1.0f + exp(-m1));
}

// float inverse_sigmoid(const float m1) {
//   return log(m1 / (1.0f - m1));
// }

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

// Load the Gaussians from the given file.
template <int D>
int loadPly(const char* filename, std::vector<GaussianData::Pos>& pos, std::vector<GaussianData::SHs<3>>& shs,
    std::vector<float>& opacities, std::vector<GaussianData::Scale>& scales, std::vector<GaussianData::Rot>& rot,
    GaussianRenderer::Vector3f& minn, GaussianRenderer::Vector3f& maxx) {
  std::ifstream infile(filename, std::ios_base::binary);

  if (!infile.good())
    logger().error(
        "Unable to find model's PLY file, attempted: {}", filename);

  // "Parse" header (it has to be a specific format anyway)
  std::string buff;
  std::getline(infile, buff);
  std::getline(infile, buff);

  std::string dummy;
  std::getline(infile, buff);
  std::stringstream ss(buff);
  int               count;
  ss >> dummy >> dummy >> count;

  // Output number of Gaussians contained
  logger().info("Loading {} Gaussian splats", count);

  while (std::getline(infile, buff))
    if (buff.compare("end_header") == 0)
      break;

  // Read all Gaussians at once (AoS)
  std::vector<GaussianData::RichPoint<D>> points(count);
  infile.read((char*)points.data(), count * sizeof(GaussianData::RichPoint<D>));

  // Resize our SoA data
  pos.resize(count);
  shs.resize(count);
  scales.resize(count);
  rot.resize(count);
  opacities.resize(count);

  // Gaussians are done training, they won't move anymore. Arrange
  // them according to 3D Morton order. This means better cache
  // behavior for reading Gaussians that end up in the same tile
  // (close in 3D --> close in 2D).
  minn = GaussianRenderer::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
  maxx = -minn;
  for (int i = 0; i < count; i++) {
    maxx = maxx.cwiseMax(points[i].pos);
    minn = minn.cwiseMin(points[i].pos);
  }
  std::vector<std::pair<uint64_t, int>> mapp(count);
  for (int i = 0; i < count; i++) {
    GaussianRenderer::Vector3f rel    = (points[i].pos - minn).array() / (maxx - minn).array();
    GaussianRenderer::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
    GaussianRenderer::Vector3i xyz    = scaled.cast<int>();

    uint64_t code = 0;
    for (int i = 0; i < 21; i++) {
      code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
      code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
      code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
    }

    mapp[i].first  = code;
    mapp[i].second = i;
  }
  auto sorter = [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) {
    return a.first < b.first;
  };
  std::sort(mapp.begin(), mapp.end(), sorter);

  // Move data from AoS to SoA
  int SH_N = (D + 1) * (D + 1);
  for (int k = 0; k < count; k++) {
    int i  = mapp[k].second;
    pos[k] = points[i].pos;

    // Normalize quaternion
    float length2 = 0;
    for (int j = 0; j < 4; j++)
      length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
    float length = sqrt(length2);
    for (int j = 0; j < 4; j++)
      rot[k].rot[j] = points[i].rot.rot[j] / length;

    // Exponentiate scale
    for (int j = 0; j < 3; j++)
      scales[k].scale[j] = exp(points[i].scale.scale[j]);

    // Activate alpha
    opacities[k] = sigmoid(points[i].opacity);

    shs[k].shs[0] = points[i].shs.shs[0];
    shs[k].shs[1] = points[i].shs.shs[1];
    shs[k].shs[2] = points[i].shs.shs[2];
    for (int j = 1; j < SH_N; j++) {
      shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
      shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
      shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
    }
  }
  return count;
}

} // namespace



////////////////////////////////////////////////////////////////////////////////////////////////////

GaussianRenderer::GaussianRenderer(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string objectName)
    : mObjectName(std::move(objectName))
    , mSettings(std::move(settings))
    , mSolarSystem(std::move(solarSystem)) {

  // Recreate the shader if HDR rendering mode is toggled.
  mEnableLightingConnection = mSettings->mGraphics.pEnableLighting.connect(
      [this](bool /*enabled*/) { mShaderDirty = true; });
  mEnableHDRConnection =
      mSettings->mGraphics.pEnableHDR.connect([this](bool /*enabled*/) { mShaderDirty = true; });

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GaussianRenderer::~GaussianRenderer() {
  mSettings->mGraphics.pEnableLighting.disconnect(mEnableLightingConnection);
  mSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());

  // Cleanup
  cudaFree(mPosCuda);
  cudaFree(mRotCuda);
  cudaFree(mScaleCuda);
  cudaFree(mOpacityCuda);
  cudaFree(mShsCuda);

  cudaFree(mViewCuda);
  cudaFree(mProjCuda);
  cudaFree(mCamPosCuda);
  cudaFree(mBackgroundCuda);
  cudaFree(mRectCuda);

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

  // delete mCopyRenderer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GaussianRenderer::configure(
    Plugin::Settings::RadianceField const& settings, int32_t cudaDevice, int32_t shDegree) {

  if (mRadianceField.mPLY != settings.mPLY || mCudaDevice != cudaDevice || mSHDegree != shDegree ) {
    logger().info("Loading PLY {}", settings.mPLY);

    mRadianceField = settings;
    mCudaDevice = cudaDevice;
    mSHDegree = shDegree;

    int num_devices;
    CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
    if (mCudaDevice >= num_devices) {
      if (num_devices == 0)
        logger().error("No CUDA devices detected!");
      else
        logger().error("Provided device index exceeds number of available CUDA devices!");
    }
    CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(mCudaDevice));
    cudaDeviceProp prop;
    CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, mCudaDevice));
    if (prop.major < 7) {
      logger().error("Sorry, need at least compute capability 7.0+! (got {})", prop.major);
    }

    std::vector<GaussianData::Pos>    pos;
    std::vector<GaussianData::Rot>    rot;
    std::vector<GaussianData::Scale>  scale;
    std::vector<float>  opacity;
    std::vector<GaussianData::SHs<3>> shs;
    if (mSHDegree == 0) {
      mCount = loadPly<0>(mRadianceField.mPLY.c_str(), pos, shs, opacity, scale, rot, mSceneMin, mSceneMax);
    } else if (mSHDegree == 1) {
      mCount = loadPly<1>(mRadianceField.mPLY.c_str(), pos, shs, opacity, scale, rot, mSceneMin, mSceneMax);
    } else if (mSHDegree == 2) {
      mCount = loadPly<2>(mRadianceField.mPLY.c_str(), pos, shs, opacity, scale, rot, mSceneMin, mSceneMax);
    } else if (mSHDegree == 3) {
      mCount = loadPly<3>(mRadianceField.mPLY.c_str(), pos, shs, opacity, scale, rot, mSceneMin, mSceneMax);
    }

    int P = mCount;

    // Allocate and fill the GPU data
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mPosCuda, sizeof(GaussianData::Pos) * P));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(mPosCuda, pos.data(), sizeof(GaussianData::Pos) * P, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mRotCuda, sizeof(GaussianData::Rot) * P));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(mRotCuda, rot.data(), sizeof(GaussianData::Rot) * P, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mShsCuda, sizeof(GaussianData::SHs<3>) * P));
    CUDA_SAFE_CALL_ALWAYS(
        cudaMemcpy(mShsCuda, shs.data(), sizeof(GaussianData::SHs<3>) * P, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mOpacityCuda, sizeof(float) * P));
    CUDA_SAFE_CALL_ALWAYS(
        cudaMemcpy(mOpacityCuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mScaleCuda, sizeof(GaussianData::Scale) * P));
    CUDA_SAFE_CALL_ALWAYS(
        cudaMemcpy(mScaleCuda, scale.data(), sizeof(GaussianData::Scale) * P, cudaMemcpyHostToDevice));

    // Create space for view parameters
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mViewCuda, sizeof(Matrix4f)));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mProjCuda, sizeof(Matrix4f)));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mCamPosCuda, 3 * sizeof(float)));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mBackgroundCuda, 3 * sizeof(float)));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&mRectCuda, 2 * P * sizeof(int)));

    float bg[3] = {0.f, 0.f, 0.f};
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(mBackgroundCuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

    mData = new GaussianData(pos, rot, scale,
        opacity, shs);

    mSurfaceRenderer = std::make_shared<SurfaceRenderer>();
    //mSplatRenderer = std::make_shared<SplatRenderer>();

    // Create GL buffer ready for CUDA/GL interop
    const int render_w = 800;
    const int render_h = 600;
    bool useInterop = true;

    glCreateBuffers(1, &mImageBuffer);
    glNamedBufferStorage(
        mImageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

    if (useInterop) {
      if (cudaPeekAtLastError() != cudaSuccess) {
        logger().error("A CUDA error occurred in setup: {}", cudaGetErrorString(cudaGetLastError()));
      }
      cudaGraphicsGLRegisterBuffer(
          &mImageBufferCuda, mImageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
      useInterop &= (cudaGetLastError() == cudaSuccess);
    }
    if (!useInterop) {
      mFallbackBytes.resize(render_w * render_h * 3 * sizeof(float));
      cudaMalloc(&mFallbackBufferCuda, mFallbackBytes.size());
      mInteropFailed = true;
    }

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
  }

  // Set radius for visibility culling.
  setRadii(glm::dvec3(10000));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Plugin::Settings::RadianceField const& GaussianRenderer::getRadianceField() {
  return mRadianceField;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GaussianRenderer::update() {
  auto object = mSolarSystem->getObject(mObjectName);

  if (object && object->getIsBodyVisible()) {
    //
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GaussianRenderer::Do() {
  auto object = mSolarSystem->getObject(mObjectName);
  if (!object || !object->getIsBodyVisible()) {
    return true;
  }

  cs::utils::FrameStats::ScopedTimer timer("Gaussian Renderer");

  // (Re-)Create renderer if necessary.
  if (mShaderDirty) {
    //

    mShaderDirty = false;
  }


    // Get modelview and projection matrices.
    std::array<GLfloat, 16> glMatV{};
    std::array<GLfloat, 16> glMatP{};
    glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
    glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
    glm::mat4 matM = object->getObserverRelativeTransform(glm::dvec3(0.f, 0.f, 6400000.f),
      glm::dquat(1.0, 0.0, 0.0, 0.0), 10000.0);
    glm::mat4 matV = glm::make_mat4x4(glMatV.data());
    glm::mat4 matP = glm::make_mat4x4(glMatP.data());
/*
    float sunIlluminance(1.F);
    float ambientBrightness(mSettings->mGraphics.pAmbientBrightness.get());

    // If HDR is enabled, the illuminance has to be calculated based on the scene's scale and the
    // distance to the Sun.
    if (mSettings->mGraphics.pEnableHDR.get()) {
      sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(matM[3]));
    }

    Vector3f sunDirection =
        glm::inverse(matM) * glm::dvec4(mSolarSystem->getSunDirection(matM[3]), 0.0);

    // Some calculations to get the view and plane normal in view space.
    glm::mat4 matModelView    = matV * matM;
    glm::mat4 matNormalMatrix = glm::transpose(glm::inverse(matModelView));
    glm::vec4 viewPos         = matModelView * glm::vec4(0.0F, 0.0F, 0.0F, 1.0F);
    viewPos                   = viewPos / viewPos.w;
    Vector3f planeNormal     = glm::normalize(-viewPos.xyz());
    Vector3f viewNormal      = (matNormalMatrix * glm::vec4(0.0F, 1.0F, 0.0F, 0.0F)).xyz();
  */

glm::vec4 viewPos         = glm::inverse(matV * matM) * glm::vec4(0.0F, 0.0F, 0.0F, 1.0F);
viewPos                   = viewPos / viewPos.w;

  mSurfaceRenderer->draw(mCount, *mData, 0.2f, viewPos, matP * matV * matM);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GaussianRenderer::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GaussianRenderer::loadPLY() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::gaussiansplatting
