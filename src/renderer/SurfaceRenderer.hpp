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

#ifndef CSP_GAUSSIAN_SPLATTING_GAUSSIAN_SURFACE_RENDERER_HPP
#define CSP_GAUSSIAN_SPLATTING_GAUSSIAN_SURFACE_RENDERER_HPP

#include <VistaOGLExt/VistaGLSLShader.h>
#include <glm/glm.hpp>

namespace csp::gaussiansplatting {

class SurfaceRenderer {

 public:
  /// Constructor.
  SurfaceRenderer(void);

  int draw(int                   G,
     const GaussianData& mesh,
      float alphaLimit,glm::vec3 const& camPos,  glm::mat4 const& matMVP, int w, int h);

 private:
 void makeFBO(int w, int h);

  GLuint mIdTexture;
  GLuint mColorTexture;
  GLuint mDepthBuffer;
  GLuint mFbo;
  int    mResX, mResY;

  VistaGLSLShader    mShader;   ///< Color shader.

  struct {
    uint32_t mParamMVP       = 0;
    uint32_t mParamCamPos        = 0;
    uint32_t mParamLimit  = 0;
    uint32_t mParamStage    = 0;
  } mUniforms;
};

} // namespace csp::gaussiansplatting

#endif