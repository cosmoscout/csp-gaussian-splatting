<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# Gaussian Splatting for CosmoScout VR

A CosmoScout VR plugin which uses the [code](https://github.com/graphdeco-inria/gaussian-splatting) provided for the paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) to visualize radiance fields.

Clone this recursively to the plugins directory:

```
cd cosmoscout-vr/plugins
git clone git@gitlab.dlr.de:scivis/cosmoscout/csp-gaussian-splatting.git --recursive
```

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-gaussian-splatting": {
      "radianceFields": [
        {
          "ply": "U:\\schn_s7\\Office.ply",
          "object": "Earth",
          "lnglat": [7.889699743962, 54.18056882],
          "rotation": [-0.959404, -0.005573, -0.084561, -0.269005],
          "scale": 0.31,
          "altitude": 7.0
        },
        {
          "ply": "U:\\schn_s7\\teufelsmauer_cropped.ply",
          "object": "Earth",
          "lnglat": [11.082730297100255, 51.75784551320601],
          "rotation": [0.047756, 0.013452, -0.998744, 0.00692],
          "scale": 9,
          "altitude": 180.0
        }
      ]
    },
  }
}
```

