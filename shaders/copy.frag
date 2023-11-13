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

	float r = source.data[0 * width * height + (y * width + x)];
	float g = source.data[1 * width * height + (y * width + x)];
	float b = source.data[2 * width * height + (y * width + x)];
	float a = source.data[3 * width * height + (y * width + x)];

    out_color = vec4(r, g, b, a);
}
