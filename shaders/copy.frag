////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#version 450

layout(location = 0) out vec4 out_color;

layout(std430, binding = 0) buffer colorLayout
{
    float data[];
} source;

uniform int width;
uniform int height;

in vec2 texcoord;

void main(void)
{
	int x = int(texcoord.x * width);
	int y = height - 1 - int(texcoord.y * height);

        int size   = width * height;
        int offset = (y * width + x);
        
	float r = source.data[0 * size + offset];
	float g = source.data[1 * size + offset];
	float b = source.data[2 * size + offset];
	float a = source.data[3 * size + offset];

    out_color = vec4(r, g, b, a);
}
