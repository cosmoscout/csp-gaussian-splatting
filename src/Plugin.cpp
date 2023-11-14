////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "GaussianRenderer.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-utils/utils.hpp"

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
  cs::core::Settings::deserialize(j, "object", o.mObject);
  cs::core::Settings::deserialize(j, "lnglat", o.mLngLat);
  cs::core::Settings::deserialize(j, "rotation", o.mRotation);
  cs::core::Settings::deserialize(j, "scale", o.mScale);
  cs::core::Settings::deserialize(j, "altitude", o.mAltitude);
}

void to_json(nlohmann::json& j, Plugin::Settings::RadianceField const& o) {
  cs::core::Settings::serialize(j, "ply", o.mPLY);
  cs::core::Settings::serialize(j, "object", o.mObject);
  cs::core::Settings::serialize(j, "lnglat", o.mLngLat);
  cs::core::Settings::serialize(j, "rotation", o.mRotation);
  cs::core::Settings::serialize(j, "scale", o.mScale);
  cs::core::Settings::serialize(j, "altitude", o.mAltitude);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "radianceFields", o.mRadianceFields);
  cs::core::Settings::deserialize(j, "cudaDevice", o.mCudaDevice);
  cs::core::Settings::deserialize(j, "drawSplats", o.mDrawSplats);
  cs::core::Settings::deserialize(j, "drawEllipses", o.mDrawEllipses);
  cs::core::Settings::deserialize(j, "distanceFading", o.mDistanceFading);
  cs::core::Settings::deserialize(j, "splatScale", o.mSplatScale);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "radianceFields", o.mRadianceFields);
  cs::core::Settings::serialize(j, "cudaDevice", o.mCudaDevice);
  cs::core::Settings::serialize(j, "drawSplats", o.mDrawSplats);
  cs::core::Settings::serialize(j, "drawEllipses", o.mDrawEllipses);
  cs::core::Settings::serialize(j, "distanceFading", o.mDistanceFading);
  cs::core::Settings::serialize(j, "splatScale", o.mSplatScale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  mGuiManager->addSettingsSectionToSideBarFromHTML("Gaussian Splatting", "grain",
      "../share/resources/gui/csp-gaussian-splatting.html");

  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-gaussian-splatting.js");

  mGuiManager->getGui()->registerCallback("gaussiansplatting.setEnableSplats",
      "Enables or disables the rendering of the splats.",
      std::function([this](bool value) { mPluginSettings->mDrawSplats = value; }));
  mPluginSettings->mDrawSplats.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("gaussiansplatting.setEnableSplats", enable);
  });

  mGuiManager->getGui()->registerCallback("gaussiansplatting.setEnableEllipses",
      "Enables or disables the rendering of debug ellipses.",
      std::function([this](bool value) { mPluginSettings->mDrawEllipses = value; }));
  mPluginSettings->mDrawEllipses.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("gaussiansplatting.setEnableEllipses", enable);
  });

  mGuiManager->getGui()->registerCallback("gaussiansplatting.setEnableFading",
      "Enables or disables the distance fading of the splats.",
      std::function([this](bool value) { mPluginSettings->mDistanceFading = value; }));
  mPluginSettings->mDistanceFading.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("gaussiansplatting.setEnableFading", enable);
  });

  mGuiManager->getGui()->registerCallback("gaussiansplatting.setScale",
      "Sets the apparent size of splats on screen.",
      std::function([this](double value) { mPluginSettings->mSplatScale = static_cast<float>(value); }));
  mPluginSettings->mSplatScale.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("gaussiansplatting.setScale", value); });

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mGuiManager->removeSettingsSection("Gaussian Splatting");

  mGuiManager->getGui()->callJavascript("CosmoScout.removeApi", "GaussianSplatting");

  mGuiManager->getGui()->unregisterCallback("gaussiansplatting.setEnableSplats");
  mGuiManager->getGui()->unregisterCallback("gaussiansplatting.setEnableEllipses");
  mGuiManager->getGui()->unregisterCallback("gaussiansplatting.setEnableFading");
  mGuiManager->getGui()->unregisterCallback("gaussiansplatting.setScale");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-gaussian-splatting"), *mPluginSettings);

  // For now, we reload everything. This could be optimized to re-use as many renderers as possible.
  mRenderers.clear();

  // Then add new renderers.
  for (auto const& settings : mPluginSettings->mRadianceFields) {
    auto renderer = std::make_shared<GaussianRenderer>(mAllSettings, mSolarSystem, "Earth");
    renderer->configure(settings, mPluginSettings);
    mRenderers.push_back(renderer);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-gaussian-splatting"] = *mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::gaussiansplatting
