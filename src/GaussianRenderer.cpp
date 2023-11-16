////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#include "GaussianRenderer.hpp"

#include "logger.hpp"

#include "../externals/eigen/Eigen/Eigen"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/convert.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <utility>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::gaussiansplatting {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Vector3f;
typedef Eigen::Matrix<int, 3, 1, Eigen::DontAlign>   Vector3i;

////////////////////////////////////////////////////////////////////////////////////////////////////

float sigmoid(const float m1) {
  return 1.0f / (1.0f + exp(-m1));
}

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

// Load the Gaussians from the given file. This comes more or less unmodified from here:
// https://gitlab.inria.fr/sibr/sibr_core/-/blob/fossa_compatibility/src/projects/gaussianviewer/renderer/GaussianView.cpp
template <int D>
int loadPly(const char* filename, std::vector<GaussianData::Pos>& pos,
    std::vector<GaussianData::SHs<3>>& shs, std::vector<float>& opacities,
    std::vector<GaussianData::Scale>& scales, std::vector<GaussianData::Rot>& rot, Vector3f& minn,
    Vector3f& maxx) {
  std::ifstream infile(filename, std::ios_base::binary);

  if (!infile.good()) {
    logger().error("Unable to find model's PLY file, attempted: {}", filename);
  }

  // "Parse" header (it has to be a specific format anyway).
  std::string buff;
  std::getline(infile, buff);
  std::getline(infile, buff);

  std::string dummy;
  std::getline(infile, buff);
  std::stringstream ss(buff);
  int               count;
  ss >> dummy >> dummy >> count;

  // Output number of Gaussians contained.
  logger().info("Loading {} Gaussian splats", count);

  while (std::getline(infile, buff)) {
    if (buff.compare("end_header") == 0) {
      break;
    }
  }

  // Read all Gaussians at once (AoS).
  std::vector<GaussianData::RichPoint<D>> points(count);
  infile.read((char*)points.data(), count * sizeof(GaussianData::RichPoint<D>));

  // Resize our SoA data.
  pos.resize(count);
  shs.resize(count);
  scales.resize(count);
  rot.resize(count);
  opacities.resize(count);

  // Gaussians are done training, they won't move anymore. Arrange
  // them according to 3D Morton order. This means better cache
  // behavior for reading Gaussians that end up in the same tile
  // (close in 3D --> close in 2D).
  minn = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
  maxx = -minn;
  for (int i = 0; i < count; i++) {
    maxx = maxx.cwiseMax(points[i].pos);
    minn = minn.cwiseMin(points[i].pos);
  }
  std::vector<std::pair<uint64_t, int>> mapp(count);
  for (int i = 0; i < count; i++) {
    Vector3f rel    = (points[i].pos - minn).array() / (maxx - minn).array();
    Vector3f scaled = ((float((1 << 21) - 1)) * rel);
    Vector3i xyz    = scaled.cast<int>();

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

  // Move data from AoS to SoA.
  int SH_N = (D + 1) * (D + 1);
  for (int k = 0; k < count; k++) {
    int i  = mapp[k].second;
    pos[k] = points[i].pos;

    // Normalize quaternion.
    float length2 = 0;
    for (int j = 0; j < 4; j++)
      length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
    float length = sqrt(length2);
    for (int j = 0; j < 4; j++)
      rot[k].rot[j] = points[i].rot.rot[j] / length;

    // Exponentiate scale.
    for (int j = 0; j < 3; j++)
      scales[k].scale[j] = exp(points[i].scale.scale[j]);

    // Activate alpha.
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

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

GaussianRenderer::GaussianRenderer(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string objectName)
    : mObjectName(std::move(objectName))
    , mSettings(std::move(settings))
    , mSolarSystem(std::move(solarSystem)) {

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GaussianRenderer::~GaussianRenderer() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GaussianRenderer::configure(Plugin::Settings::RadianceField const& settings,
    std::shared_ptr<Plugin::Settings>                                   pluginSettings) {

  // If the ply file or the CUDA device changed, we reload everything.
  if (mRadianceField.mPLY != settings.mPLY || mCudaDevice != pluginSettings->mCudaDevice.get()) {
    logger().info("Loading PLY {}", settings.mPLY);

    // Store the settings.
    mRadianceField  = settings;
    mCudaDevice     = pluginSettings->mCudaDevice.get();
    mPluginSettings = std::move(pluginSettings);

    // Select the CUDA device.
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

    // Load the radiance field from the ply file.
    std::vector<GaussianData::Pos>    pos;
    std::vector<GaussianData::Rot>    rot;
    std::vector<GaussianData::Scale>  scale;
    std::vector<float>                opacity;
    std::vector<GaussianData::SHs<3>> shs;

    const int shDegree = 3;
    Vector3f  sceneMin{};
    Vector3f  sceneMax{};
    mCount = loadPly<shDegree>(
        mRadianceField.mPLY.c_str(), pos, shs, opacity, scale, rot, sceneMin, sceneMax);
    mData = std::make_unique<GaussianData>(pos, rot, scale, opacity, shs);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GaussianRenderer::Do() {
  auto object = mSolarSystem->getObject(mObjectName);
  if (!object || !object->getIsBodyVisible()) {
    return true;
  }

  cs::utils::FrameStats::ScopedTimer timer("Gaussian Renderer for " + mRadianceField.mPLY);

  // Compute ENO (east-north-up) rotation for the given location on the planet.
  glm::dvec2 lngLat(cs::utils::convert::toRadians(mRadianceField.mLngLat));
  auto       normal = cs::utils::convert::lngLatToNormal(lngLat);
  auto       north  = glm::dvec3(0.0, 1.0, 0.0);

  auto x = glm::cross(north, normal);
  auto y = normal;
  auto z = glm::cross(x, y);

  x = glm::normalize(x);
  y = glm::normalize(y);
  z = glm::normalize(z);

  // Apply object-local roation.
  auto rot = glm::toQuat(glm::dmat3(x, y, z)) * mRadianceField.mRotation.get();

  // Get the cartesian position from the given geographic coordinates.
  auto pos =
      cs::utils::convert::toCartesian(lngLat, object->getRadii(), mRadianceField.mAltitude.get());

  // Get the final observer-relative transformation of the radiance field.
  glm::mat4 matM = object->getObserverRelativeTransform(pos, rot, mRadianceField.mScale.get());

  // Get view and projection matrices.
  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  glm::mat4 matV = glm::make_mat4x4(glMatV.data());
  glm::mat4 matP = glm::make_mat4x4(glMatP.data());

  // Compute the object-relative viewer position.
  glm::vec4 viewPos = glm::inverse(matV * matM) * glm::vec4(0.0F, 0.0F, 0.0F, 1.0F);
  viewPos           = viewPos / viewPos.w;

  // Finally draw the debug ellipses and/or the splats.
  if (mPluginSettings->mDrawEllipses.get()) {
    mSurfaceRenderer.draw(mCount, *mData, 0.2f, viewPos, matP * matV * matM);
  }

  if (mPluginSettings->mDrawSplats.get()) {
    mSplatRenderer.draw(mPluginSettings->mSplatScale.get(), mCount,
        mPluginSettings->mDistanceFading.get(), *mData, viewPos, matV * matM, matP);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GaussianRenderer::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::gaussiansplatting
