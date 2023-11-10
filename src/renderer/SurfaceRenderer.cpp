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

SurfaceRenderer::SurfaceRenderer(void) {

  mShader.InitVertexShaderFromString(cs::utils::filesystem::loadToString("../share/resources/shaders/gaussian_surface.vert"));
    mShader.InitFragmentShaderFromString(cs::utils::filesystem::loadToString("../share/resources/shaders/gaussian_surface.frag"));
    mShader.Link();

    mUniforms.mParamCamPos          = glGetUniformLocation(mShader.GetProgram(), "rayOrigin");
    mUniforms.mParamMVP          = glGetUniformLocation(mShader.GetProgram(), "MVP");
    mUniforms.mParamLimit          = glGetUniformLocation(mShader.GetProgram(), "alpha_limit");
    mUniforms.mParamStage          = glGetUniformLocation(mShader.GetProgram(), "stage");

  glCreateTextures(GL_TEXTURE_2D, 1, &mIdTexture);
  glTextureParameteri(mIdTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTextureParameteri(mIdTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glCreateTextures(GL_TEXTURE_2D, 1, &mColorTexture);
  glTextureParameteri(mColorTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTextureParameteri(mColorTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glCreateFramebuffers(1, &mFbo);
  glCreateRenderbuffers(1, &mDepthBuffer);

  makeFBO(800, 800);

}

void SurfaceRenderer::makeFBO(int w, int h) {
  mResX = w;
  mResY = h;

  glBindTexture(GL_TEXTURE_2D, mIdTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, mResX, mResY, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, 0);

  glBindTexture(GL_TEXTURE_2D, mColorTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mResX, mResY, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  glNamedRenderbufferStorage(mDepthBuffer, GL_DEPTH_COMPONENT, mResX, mResY);

  glBindFramebuffer(GL_FRAMEBUFFER, mFbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mColorTexture, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, mIdTexture, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mDepthBuffer);
}

int SurfaceRenderer::draw(int G, const GaussianData& mesh, float limit,glm::vec3 const& camPos,  glm::mat4 const& matMVP, int w, int h) {
  GLint drawFboId = 0;
glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFboId);

  glBindFramebuffer(GL_FRAMEBUFFER, mFbo);

  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  if (w != mResX || h != mResY) {
    makeFBO(w, h);
  }

  // Solid pass
  GLuint drawBuffers[2];
  drawBuffers[0] = GL_COLOR_ATTACHMENT0;
  drawBuffers[1] = GL_COLOR_ATTACHMENT1;
  glDrawBuffers(2, drawBuffers);

  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  mShader.Bind();

  glUniformMatrix4fv(mUniforms.mParamMVP, 1, GL_FALSE, glm::value_ptr(matMVP));
  glUniform3fv(mUniforms.mParamCamPos, 1, glm::value_ptr(camPos));
  glUniform1f(mUniforms.mParamLimit, limit);
  glUniform1i(mUniforms.mParamStage, 0);

  mesh.render(G);

  // Simple additive blendnig (no order)
  glDrawBuffers(1, drawBuffers);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);
  glBlendEquation(GL_FUNC_ADD);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  glUniform1i(mUniforms.mParamStage, 1);
  mesh.render(G);

  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);

  mShader.Release();

  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glBlitNamedFramebuffer(
      mFbo, drawFboId, 0, 0, mResX, mResY, 0, 0, mResX, mResY, GL_COLOR_BUFFER_BIT, GL_NEAREST);

  return 0;
}

} // namespace csp::gaussiansplatting
