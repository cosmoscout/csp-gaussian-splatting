////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#ifndef CSP_GAUSSIAN_SPLATTING_GAUSSIAN_SURFACE_RENDERER_HPP
#define CSP_GAUSSIAN_SPLATTING_GAUSSIAN_SURFACE_RENDERER_HPP

#include <VistaOGLExt/VistaGLSLShader.h>
#include <glm/glm.hpp>

namespace csp::gaussiansplatting {

/// This is a very simple renderer which draws the given radiance field point cloud using instanced
/// rendering. It can be used for debugging as radiance field data as it is much easier than the
/// SplatRenderer. Rendering happens in two stages: first, the more opaque splats are drawn with
/// depth testing enabled, then the more transparent splats are drawn additively on top.
class SurfaceRenderer {

 public:
  SurfaceRenderer();

  int draw(int count, const GaussianData& mesh, float alphaLimit, glm::vec3 const& camPos,
      glm::mat4 const& matMVP);

 private:
  VistaGLSLShader mShader;

  struct {
    uint32_t mParamMVP    = 0;
    uint32_t mParamCamPos = 0;
    uint32_t mParamLimit  = 0;
    uint32_t mParamStage  = 0;
  } mUniforms;
};

} // namespace csp::gaussiansplatting

#endif