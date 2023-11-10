////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_GAUSSIAN_SPLATTING_PLUGIN_HPP
#define CSP_GAUSSIAN_SPLATTING_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <string>
#include <vector>

namespace csp::gaussiansplatting {

class GaussianRenderer;

class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct RadianceField {
      std::string mPLY;
    };

    std::vector<RadianceField>          mRadianceFields;
    cs::utils::DefaultProperty<int32_t> mCudaDevice{0};
    cs::utils::DefaultProperty<int32_t> mSHDegree{3};
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();
  void onSave();

  Settings                                       mPluginSettings;
  std::vector<std::shared_ptr<GaussianRenderer>> mRenderers;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::gaussiansplatting

#endif // CSP_GAUSSIAN_SPLATTING_PLUGIN_HPP
