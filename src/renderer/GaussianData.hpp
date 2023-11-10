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

#include "../externals/Eigen/Eigen"

#include <memory>
#include <vector>
#include <GL/glew.h>

namespace csp::gaussiansplatting {

class GaussianData {
  
 public:
 typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Pos;

template <int D>
struct SHs {
  float shs[(D + 1) * (D + 1) * 3];
};
struct Scale {
  float scale[3];
};
struct Rot {
  float rot[4];
};
template <int D>
struct RichPoint {
  Pos    pos;
  float  n[3];
  SHs<D> shs;
  float  opacity;
  Scale  scale;
  Rot    rot;
};

 public:
  /// Constructor.
  GaussianData( std::vector<Pos> const& pos_data, std::vector<Rot> const& rot_data, std::vector<Scale> const& scale_data,
      std::vector<float> const& alpha_data, std::vector<SHs<3>> const& color_data);

  void render(int G) const;

 private:

  GLuint mPosOpenGL;
  GLuint mRotOpenGL;
  GLuint mScaleOpenGL;
  GLuint mAlphaOpenGL;
  GLuint mColorOpenGL;
};

}

#endif