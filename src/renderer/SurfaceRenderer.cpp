////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#include "SurfaceRenderer.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"

namespace csp::gaussiansplatting {

////////////////////////////////////////////////////////////////////////////////////////////////////

SurfaceRenderer::SurfaceRenderer() {
  // Compile and link the shader.
  mShader.InitVertexShaderFromString(
      cs::utils::filesystem::loadToString("../share/resources/shaders/gaussian_surface.vert"));
  mShader.InitFragmentShaderFromString(
      cs::utils::filesystem::loadToString("../share/resources/shaders/gaussian_surface.frag"));
  mShader.Link();

  // Get the uniform locations.
  mUniforms.mParamCamPos = mShader.GetUniformLocation("rayOrigin");
  mUniforms.mParamMVP    = mShader.GetUniformLocation("MVP");
  mUniforms.mParamLimit  = mShader.GetUniformLocation("alpha_limit");
  mUniforms.mParamStage  = mShader.GetUniformLocation("stage");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int SurfaceRenderer::draw(int count, const GaussianData& mesh, float limit, glm::vec3 const& camPos,
    glm::mat4 const& matMVP) {

  // Bind all input buffers.
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mesh.mPosOpenGL);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh.mRotOpenGL);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mesh.mScaleOpenGL);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mesh.mAlphaOpenGL);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mesh.mColorOpenGL);

  mShader.Bind();

  glUniformMatrix4fv(mUniforms.mParamMVP, 1, GL_FALSE, glm::value_ptr(matMVP));
  glUniform3fv(mUniforms.mParamCamPos, 1, glm::value_ptr(camPos));
  glUniform1f(mUniforms.mParamLimit, limit);

  // First stage is disabled for now as it clutters the scene quite a lot. Do we find a better debug
  // visualization? glUniform1i(mUniforms.mParamStage, 0); glDrawArraysInstanced(GL_TRIANGLES, 0,
  // 36, count);

  // Second stage uses additive blending and no depth test.
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  glBlendEquation(GL_FUNC_ADD);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);

  glUniform1i(mUniforms.mParamStage, 1);
  glDrawArraysInstanced(GL_TRIANGLES, 0, 36, count);

  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);

  mShader.Release();

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::gaussiansplatting
