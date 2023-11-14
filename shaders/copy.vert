////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#version 450

out vec2 texcoord;

void main(void) {
  vec2 position = vec2(gl_VertexID & 2, (gl_VertexID << 1) & 2) * 2.0 - 1.0;
  texcoord = position * 0.5 + 0.5;

  // No tranformation here since we draw a full screen quad.
  gl_Position = vec4(position, 0, 1);
}
