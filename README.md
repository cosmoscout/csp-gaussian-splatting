<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# Gaussian Splatting for CosmoScout VR

A CosmoScout VR plugin which uses the code provided for the paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) to visualize radiance fields. The code comes from two repositories:
* https://github.com/graphdeco-inria/diff-gaussian-rasterization contains the Cuda rasterizer. A fork of this repository is included in this plugin as a submodule.
* https://gitlab.inria.fr/sibr/sibr_core contains the glue-code for loading the radiance fields from disk. Our loading code is based on this repository in large parts.

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

## License Information

The original code from MPI and Inri published alongside the paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) which is used in this plugin, is only to be used for research and other non-commercial use-cases.

Hence, this plugin for CosmoScout VR contains code which is only available under a more restrictive license than the rest of CosmoScout VR. See the LICENSES directory and the SPDX tags of the individual files for more information.