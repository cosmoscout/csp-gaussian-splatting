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

#ifndef CSP_GAUSSIAN_SPLATTING_GAUSSIAN_DATA_HPP
#define CSP_GAUSSIAN_SPLATTING_GAUSSIAN_DATA_HPP

#include <memory>
#include <GL/glew.h>

namespace csp::gaussiansplatting {

class GaussianData {
 public:
  typedef std::shared_ptr<GaussianData> Ptr;

 public:
  /// Constructor.
  GaussianData(int num_gaussians, float* mean_data, float* rot_data, float* scale_data,
      float* alpha_data, float* color_data);

  void render(int G) const;

 private:
  int    _num_gaussians;
  GLuint meanBuffer;
  GLuint rotBuffer;
  GLuint scaleBuffer;
  GLuint alphaBuffer;
  GLuint colorBuffer;
};

}

#endif