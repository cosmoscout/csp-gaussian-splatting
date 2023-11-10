////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "GaussianRenderer.hpp"
#include "logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::gaussiansplatting::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::gaussiansplatting {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::RadianceField& o) {
  cs::core::Settings::deserialize(j, "ply", o.mPLY);
}

void to_json(nlohmann::json& j, Plugin::Settings::RadianceField const& o) {
  cs::core::Settings::serialize(j, "ply", o.mPLY);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "radianceFields", o.mRadianceFields);
  cs::core::Settings::deserialize(j, "cudaDevice", o.mCudaDevice);
  cs::core::Settings::deserialize(j, "shDegree", o.mSHDegree);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "radianceFields", o.mRadianceFields);
  cs::core::Settings::serialize(j, "cudaDevice", o.mCudaDevice);
  cs::core::Settings::serialize(j, "shDegree", o.mSHDegree);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-gaussian-splatting"), mPluginSettings);

  mRenderers.clear();

  // Then add new renderers.
  for (auto const& settings : mPluginSettings.mRadianceFields) {
    auto renderer = std::make_shared<GaussianRenderer>(mAllSettings, mSolarSystem, "Earth");
    renderer->configure(settings, mPluginSettings.mCudaDevice.get(), mPluginSettings.mSHDegree.get());
    mRenderers.push_back(renderer);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-gaussian-splatting"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for (auto const& renderer : mRenderers) {
    renderer->update();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::gaussiansplatting
