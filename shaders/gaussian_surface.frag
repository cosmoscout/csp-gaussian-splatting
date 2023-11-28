////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: Copyright (C) 2023, Inria, GRAPHDECO research group
// SPDX-License-Identifier: LicenseRef-InriaLicense

#version 430

uniform mat4 MVP;
uniform float alpha_limit;
uniform vec3 rayOrigin;

in vec3 worldPos;
in vec3 ellipsoidCenter;
in vec3 ellipsoidScale;
in mat3 ellipsoidRotation;
in vec3 colorVert;
in float alphaVert;
in flat int boxID;

layout (location = 0) out vec4 out_color;

vec3 closestEllipsoidIntersection(vec3 rayDirection) {
  // Convert ray to ellipsoid space
  dvec3 localRayOrigin = (rayOrigin - ellipsoidCenter) * ellipsoidRotation;
  dvec3 localRayDirection = normalize(rayDirection * ellipsoidRotation);

  dvec3 oneover = double(1) / dvec3(ellipsoidScale);
  
  // Compute coefficients of quadratic equation
  double a = dot(localRayDirection * oneover, localRayDirection * oneover);
  double b = 2.0 * dot(localRayDirection * oneover, localRayOrigin * oneover);
  double c = dot(localRayOrigin * oneover, localRayOrigin * oneover) - 1.0;
  
  // Compute discriminant
  double discriminant = b * b - 4.0 * a * c;
  
  // If discriminant is negative, there is no intersection
  if (discriminant < 0.0) {
    return vec3(0.0);
  }
  
  // Compute two possible solutions for t
  float t1 = float((-b - sqrt(discriminant)) / (2.0 * a));
  float t2 = float((-b + sqrt(discriminant)) / (2.0 * a));
  
  // Take the smaller positive solution as the closest intersection
  float t = min(t1, t2);
  
  // Compute intersection point in ellipsoid space
  vec3 localIntersection = vec3(localRayOrigin + t * localRayDirection);

  // Convert intersection point back to world space
  vec3 intersection = ellipsoidRotation * localIntersection + ellipsoidCenter;
  
  return intersection;
}

void main(void) {
  vec3 dir = normalize(worldPos - rayOrigin);

  vec3 intersection = closestEllipsoidIntersection(dir);

  if (intersection == vec3(0)) {
    discard;
  }

  out_color = vec4(colorVert, 1.0);
}
