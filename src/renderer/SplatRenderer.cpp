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

#include <boost/asio.hpp>
#include <core/graphics/GUI.hpp>
#include <imgui_internal.h>
#include <projects/SplatRendererer/renderer/SplatRenderer.hpp>
#include <rasterizer.h>
#include <thread>

namespace csp::gaussiansplatting {
// A simple copy renderer class. Much like the original, but this one
// reads from a buffer instead of a texture and blits the result to
// a render target.
class BufferCopyRenderer {

 public:
  BufferCopyRenderer() {
    _shader.init("CopyShader", sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
        sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

    _flip.init(_shader, "flip");
    _width.init(_shader, "width");
    _height.init(_shader, "height");
  }

  void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true) {
    if (disableTest)
      glDisable(GL_DEPTH_TEST);
    else
      glEnable(GL_DEPTH_TEST);

    _shader.begin();
    _flip.send();
    _width.send();
    _height.send();

    dst.clear();
    dst.bind();

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

    sibr::RenderUtility::renderScreenQuad();

    dst.unbind();
    _shader.end();
  }

  /** \return option to flip the texture when copying. */
  bool& flip() {
    return _flip.get();
  }
  int& width() {
    return _width.get();
  }
  int& height() {
    return _height.get();
  }

 private:
  GLShader        _shader;
  GLuniform<bool> _flip   = false; ///< Flip the texture when copying.
  GLuniform<int>  _width  = 1000;
  GLuniform<int>  _height = 800;
};


SplatRenderer::SplatRenderer(uint render_w,
    uint render_h) {
 
}



void SplatRenderer::draw(sibr::IRenderTarget& dst, const sibr::Camera& eye) {
 
  // Convert view and projection to target coordinate system
  auto view_mat = eye.view();
  auto proj_mat = eye.viewproj();
  view_mat.row(1) *= -1;
  view_mat.row(2) *= -1;
  proj_mat.row(1) *= -1;

  // Compute additional view parameters
  float tan_fovy = tan(eye.fovy() * 0.5f);
  float tan_fovx = tan_fovy * eye.aspect();

  // Copy frame-dependent data to GPU
  CUDA_SAFE_CALL(
      cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

  float* image_cuda = nullptr;
  if (!_interop_failed) {
    // Map OpenGL buffer resource for use with CUDA
    size_t bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
    CUDA_SAFE_CALL(
        cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
  } else {
    image_cuda = fallbackBufferCuda;
  }

  // Rasterize
  int*   rects  = _fastCulling ? rect_cuda : nullptr;
  float* boxmin = _cropping ? (float*)&_boxmin : nullptr;
  float* boxmax = _cropping ? (float*)&_boxmax : nullptr;
  // CudaRasterizer::Rasterizer::forward(geomBufferFunc, binningBufferFunc, imgBufferFunc, count,
  //     _sh_degree, 16, background_cuda, _resolution.x(), _resolution.y(), pos_cuda, shs_cuda,
  //     nullptr, opacity_cuda, scale_cuda, _scalingModifier, rot_cuda, nullptr, view_cuda,
  //     proj_cuda, cam_pos_cuda, tan_fovx, tan_fovy, false, image_cuda, nullptr, rects, boxmin,
  //     boxmax);

  if (!_interop_failed) {
    // Unmap OpenGL resource for use with OpenGL
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
  } else {
    CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(),
        cudaMemcpyDeviceToHost));
    glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
  }
  // Copy image contents to framebuffer
  _copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
  

  if (cudaPeekAtLastError() != cudaSuccess) {
    SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError())
             << ". Please rerun in Debug to find the exact line!";
  }
}



}
