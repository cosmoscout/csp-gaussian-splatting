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

#include "SurfaceRenderer.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"


namespace csp::gaussiansplatting {

SurfaceRenderer::SurfaceRenderer() {
  mShader.InitVertexShaderFromString(cs::utils::filesystem::loadToString("../share/resources/shaders/gaussian_surface.vert"));
  mShader.InitFragmentShaderFromString(cs::utils::filesystem::loadToString("../share/resources/shaders/gaussian_surface.frag"));
  mShader.Link();

  mUniforms.mParamCamPos = glGetUniformLocation(mShader.GetProgram(), "rayOrigin");
  mUniforms.mParamMVP    = glGetUniformLocation(mShader.GetProgram(), "MVP");
  mUniforms.mParamLimit  = glGetUniformLocation(mShader.GetProgram(), "alpha_limit");
  mUniforms.mParamStage  = glGetUniformLocation(mShader.GetProgram(), "stage");
}


int SurfaceRenderer::draw(int count, const GaussianData& mesh, float limit,glm::vec3 const& camPos,  glm::mat4 const& matMVP) {

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mesh.mPosOpenGL);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh.mRotOpenGL);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mesh.mScaleOpenGL);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mesh.mAlphaOpenGL);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mesh.mColorOpenGL);

  mShader.Bind();

  glUniformMatrix4fv(mUniforms.mParamMVP, 1, GL_FALSE, glm::value_ptr(matMVP));
  glUniform3fv(mUniforms.mParamCamPos, 1, glm::value_ptr(camPos));
  glUniform1f(mUniforms.mParamLimit, limit);
  glUniform1i(mUniforms.mParamStage, 0);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, count);




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

} // namespace csp::gaussiansplatting
