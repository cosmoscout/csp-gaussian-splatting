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


#include "GaussianData.hpp"

namespace csp::gaussiansplatting {

GaussianData::GaussianData(std::vector<Pos> const& pos_data, std::vector<Rot> const& rot_data, std::vector<Scale> const& scale_data,
      std::vector<float> const& alpha_data, std::vector<SHs<3>> const& color_data)
{
    glCreateBuffers(1, &mPosOpenGL);
    glCreateBuffers(1, &mRotOpenGL);
    glCreateBuffers(1, &mScaleOpenGL);
    glCreateBuffers(1, &mAlphaOpenGL);
    glCreateBuffers(1, &mColorOpenGL);
    glNamedBufferStorage(mPosOpenGL, pos_data.size() * 3 * sizeof(float), &pos_data[0], 0);
    glNamedBufferStorage(mRotOpenGL, rot_data.size() * 4 * sizeof(float), &rot_data[0], 0);
    glNamedBufferStorage(mScaleOpenGL, scale_data.size() * 3 * sizeof(float), &scale_data[0], 0);
    glNamedBufferStorage(mAlphaOpenGL, alpha_data.size() * sizeof(float), &alpha_data[0], 0);
    glNamedBufferStorage(mColorOpenGL, color_data.size() * sizeof(float) * 48, &color_data[0], 0);
}

void GaussianData::render(int G) const
{
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mPosOpenGL);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mRotOpenGL);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mScaleOpenGL);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mAlphaOpenGL);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mColorOpenGL);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, G);
}

}
