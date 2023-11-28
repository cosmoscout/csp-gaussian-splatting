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
  mUniforms.mParamScale  = mShader.GetUniformLocation("scale");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int SurfaceRenderer::draw(float scale, int count, const GaussianData& mesh, float limit,
    glm::vec3 const& camPos, glm::mat4 const& matMVP) {

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
  glUniform1f(mUniforms.mParamScale, scale);
  glDrawArraysInstanced(GL_TRIANGLES, 0, 36, count);

  mShader.Release();

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::gaussiansplatting
