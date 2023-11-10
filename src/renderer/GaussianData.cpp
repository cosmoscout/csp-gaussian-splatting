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

GaussianData::GaussianData(int num_gaussians, float* mean_data, float* rot_data, float* scale_data, float* alpha_data, float* color_data)
{
    _num_gaussians = num_gaussians;
    glCreateBuffers(1, &meanBuffer);
    glCreateBuffers(1, &rotBuffer);
    glCreateBuffers(1, &scaleBuffer);
    glCreateBuffers(1, &alphaBuffer);
    glCreateBuffers(1, &colorBuffer);
    glNamedBufferStorage(meanBuffer, num_gaussians * 3 * sizeof(float), mean_data, 0);
    glNamedBufferStorage(rotBuffer, num_gaussians * 4 * sizeof(float), rot_data, 0);
    glNamedBufferStorage(scaleBuffer, num_gaussians * 3 * sizeof(float), scale_data, 0);
    glNamedBufferStorage(alphaBuffer, num_gaussians * sizeof(float), alpha_data, 0);
    glNamedBufferStorage(colorBuffer, num_gaussians * sizeof(float) * 48, color_data, 0);
}

void GaussianData::render(int G) const
{
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, meanBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rotBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, scaleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, alphaBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, colorBuffer);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, G);
}

}
