////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_GAUSSIAN_SPLATTING_PLUGIN_HPP
#define CSP_GAUSSIAN_SPLATTING_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <string>
#include <vector>

namespace csp::gaussiansplatting {

class GaussianRenderer;

/// This plugin implements 3D gaussian splatting for real-time radiance field rendering. For this,
/// the plugin uses code from MPI and Inria which is available at
/// https://github.com/graphdeco-inria/gaussian-splatting
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct RadianceField {

      /// The file path to the ply radiance field data.
      std::string mPLY;

      /// The object name to which the renderer should be attached (e.g. "Earth").
      std::string mObject;

      /// The position where to place the renderer on the above body.
      glm::dvec2 mLngLat;

      /// Optional transformation relative to the location above.
      cs::utils::DefaultProperty<glm::dquat> mRotation{glm::dquat(1.0, 0.0, 0.0, 0.0)};
      cs::utils::DefaultProperty<double>     mScale{1.0};
      cs::utils::DefaultProperty<double>     mAltitude{0.0};
    };

    std::vector<RadianceField>          mRadianceFields;
    cs::utils::DefaultProperty<bool>    mDrawSplats{false};
    cs::utils::DefaultProperty<bool>    mDrawEllipses{false};
    cs::utils::DefaultProperty<bool>    mDistanceFading{true};
    cs::utils::DefaultProperty<float>   mSplatScale{1.f};
    cs::utils::DefaultProperty<int32_t> mCudaDevice{0};
  };

  void init() override;
  void deInit() override;

 private:
  void onLoad();
  void onSave();

  std::shared_ptr<Settings>                      mPluginSettings = std::make_shared<Settings>();
  std::vector<std::shared_ptr<GaussianRenderer>> mRenderers;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::gaussiansplatting

#endif // CSP_GAUSSIAN_SPLATTING_PLUGIN_HPP
