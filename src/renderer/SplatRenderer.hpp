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

#ifndef CSP_GAUSSIAN_SPLATTING_GAUSSIAN_VIEW_HPP
#define CSP_GAUSSIAN_SPLATTING_GAUSSIAN_VIEW_HPP

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <functional>
#include <memory>

namespace CudaRasterizer {
class Rasterizer;
}

namespace csp::gaussiansplatting {

class BufferCopyRenderer;

class SplatRenderer {


 public:
  /**
   * Constructor
   * \param ibrScene The scene to use for rendering.
   * \param render_w rendering width
   * \param render_h rendering height
   */
  SplatRenderer(uint render_w, uint render_h,
      const char* file, bool* message_read, int sh_degree, bool white_bg = false,
      bool useInterop = true, int device = 0);

  void draw(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;



  virtual ~SplatRenderer() override;



 protected:
  std::string currMode = "Splats";

  bool           _cropping = false;
  sibr::Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
  char           _buff[512] = "cropped.ply";

  bool _fastCulling = true;
  int  _device      = 0;
  int  _sh_degree   = 3;

  int    count;
  float* pos_cuda;
  float* rot_cuda;
  float* scale_cuda;
  float* opacity_cuda;
  float* shs_cuda;
  int*   rect_cuda;

  GLuint                 imageBuffer;
  cudaGraphicsResource_t imageBufferCuda;

  size_t                         allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
  void *                         geomPtr = nullptr, *binningPtr = nullptr, *imgPtr = nullptr;
  std::function<char*(size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

  float* view_cuda;
  float* proj_cuda;
  float* cam_pos_cuda;
  float* background_cuda;

  float         _scalingModifier = 1.0f;
  GaussianData* gData;

  bool              _interop_failed = false;
  std::vector<char> fallback_bytes;
  float*            fallbackBufferCuda = nullptr;
  bool              accepted           = false;

  std::shared_ptr<sibr::BasicIBRScene> _scene; ///< The current scene.
  BufferCopyRenderer*                  _copyRenderer;
};

} // namespace csp::gaussiansplatting

#endif