////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#ifndef CSP_GAUSSIAN_SPLATTING_GAUSSIAN_DATA_HPP
#define CSP_GAUSSIAN_SPLATTING_GAUSSIAN_DATA_HPP

#include "../externals/Eigen/Eigen"

#include <memory>
#include <vector>
#include <GL/glew.h>

namespace csp::gaussiansplatting {

/// This struct contains the GL and Cuda resources used for drawing the splats.
struct GaussianData {
  
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

  GaussianData() = default;

  /// Allocates all the GL and Cuda resources.
  GaussianData( std::vector<Pos> const& pos, std::vector<Rot> const& rot, std::vector<Scale> const& scale,
      std::vector<float> const& alpha, std::vector<SHs<3>> const& color);

  /// Frees all the resources again.
  ~GaussianData();

  /// These are used by the SurfaceRenderer.
  GLuint mPosOpenGL = 0;
  GLuint mRotOpenGL = 0;
  GLuint mScaleOpenGL = 0;
  GLuint mAlphaOpenGL = 0;
  GLuint mColorOpenGL = 0;

  /// These are used by the SplatRenderer.
  float* mPosCuda = nullptr;
  float* mRotCuda = nullptr;
  float* mScaleCuda = nullptr;
  float* mOpacityCuda = nullptr;
  float* mShsCuda = nullptr;
};

}

#endif