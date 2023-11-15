////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#ifndef CSP_GAUSSIAN_SPLATTING_GAUSSIAN_RENDERER_HPP
#define CSP_GAUSSIAN_SPLATTING_GAUSSIAN_RENDERER_HPP

#include "Plugin.hpp"
#include "renderer/GaussianData.hpp"
#include "renderer/SplatRenderer.hpp"
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

/// This class manages drawing of a single radiance field. It provides two ways for drawing the given
/// radiance field: Either using a very simple approach using instanced rendering of tiny boxes, or
/// via the splat raycasting developed for the original gaussian splatting paper.
class GaussianRenderer : public IVistaOpenGLDraw {
 public:
  GaussianRenderer(std::shared_ptr<cs::core::Settings> settings,
      std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string objectName);

  GaussianRenderer(GaussianRenderer const& other) = delete;
  GaussianRenderer(GaussianRenderer&& other)      = default;

  GaussianRenderer& operator=(GaussianRenderer const& other) = delete;
  GaussianRenderer& operator=(GaussianRenderer&& other) = delete;

  ~GaussianRenderer() override;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::RadianceField const& settings, std::shared_ptr<Plugin::Settings> pluginSettings);

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::string                            mObjectName;
  std::shared_ptr<cs::core::Settings>    mSettings;
  std::shared_ptr<Plugin::Settings> mPluginSettings;

  std::unique_ptr<VistaOpenGLNode> mGLNode;
  Plugin::Settings::RadianceField mRadianceField;

  int32_t mCudaDevice = 0;
  int32_t mCount = 0;

  std::unique_ptr<GaussianData>    mData;
  SurfaceRenderer mSurfaceRenderer;
  SplatRenderer   mSplatRenderer;
};
} // namespace csp::gaussiansplatting

#endif // CSP_GAUSSIAN_SPLATTING_GAUSSIAN_RENDERER_HPP
