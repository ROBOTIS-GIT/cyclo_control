// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "cyclo_teleoperation/core/mode_manager.hpp"
#include <stdexcept>

namespace cyclo_teleoperation
{
ModeManager::ModeManager()
: loader_("cyclo_teleoperation", "cyclo_teleoperation::TeleoperationMode") {}

void ModeManager::configure(rclcpp::Node & node)
{
  teleop_modes_.configure(node, loader_, "available_control_modes", "control_modes",
    "default_control_mode", false);
  model_modes_.configure(node, loader_, "available_model_action_modes", "model_action_modes",
    "", true);
  teleop_requested_ = teleop_modes_.defaultMode();
  model_requested_ = model_modes_.defaultMode();
  requested_ = teleop_requested_;
}

void ModeManager::deactivate()
{
  if (plugin_) {plugin_->deactivate(); plugin_.reset();}
  ready_ = false;
  active_ = 0;
}

void ModeManager::selectSource(bool teleop)
{
  deactivate();
  requested_ = teleop ? teleop_requested_ : model_requested_;
  pending_ = teleop;
}

void ModeManager::selectTeleop(uint16_t id, bool activate)
{
  teleop_requested_ = id;
  if (activate) {
    requested_ = id;
    pending_ = active_ != id || !ready_;
  }
}

void ModeManager::selectModel(uint16_t id)
{
  model_requested_ = requested_ = id;
  deactivate();
}

void ModeManager::requestReconfigure(bool teleop)
{
  if (teleop) {pending_ = true;} else {ready_ = false;}
}

void ModeManager::cancelTransition(PoseSequenceManager & poses)
{
  poses.cancelInitialPose();
  poses.cancelExitPose();
  phase_ = Phase::kIdle;
  started_ = pending_ = false;
  paused_presets_ = 0;
}

void ModeManager::retarget(PoseSequenceManager & poses)
{
  if (!started_ || target_ == requested_) {return;}
  if (phase_ == Phase::kExitPose) {target_ = requested_;} else {cancelTransition(poses);}
  pending_ = active_ != requested_ || !ready_;
}

void ModeManager::startInitial(PoseSequenceManager & poses, const ModeContext & context)
{
  poses.cancelExitPose();
  if (poses.automaticInitialPoseEnabled(target_)) {
    if (!poses.startInitialPose(target_, context)) {
      throw std::runtime_error("initial pose transition was rejected");
    }
    const auto groups = poses.activeInitialPoseGroups();
    poses.cancelPresets(groups);
    paused_presets_ &= ~groups;
    phase_ = Phase::kInitialPose;
  } else {
    poses.cancelInitialPose();
    phase_ = Phase::kActivate;
  }
}

void ModeManager::beginTransition(PoseSequenceManager & poses, const ModeContext & context)
{
  const auto previous = active_;
  target_ = requested_;
  paused_presets_ = poses.activePresetGroups();
  phase_ = Phase::kIdle;
  deactivate();
  if (previous != 0 && previous != target_ && poses.hasExitPose(previous)) {
    if (!poses.startExitPose(previous, context)) {
      throw std::runtime_error("exit pose transition was rejected");
    }
    const auto groups = poses.activeExitPoseGroups();
    poses.cancelPresets(groups);
    paused_presets_ &= ~groups;
    phase_ = Phase::kExitPose;
  } else {
    poses.cancelExitPose();
  }
  started_ = true;
  if (phase_ != Phase::kExitPose) {startInitial(poses, context);}
}

void ModeManager::configureTeleop(rclcpp::Node & node, const ModeConfiguration & config)
{
  const auto & entry = teleop_modes_.at(target_);
  plugin_ = loader_.createSharedInstance(entry.plugin);
  if (!plugin_->configure(node, entry.parameter_prefix, config)) {
    throw std::runtime_error("mode configuration was rejected");
  }
}

void ModeManager::activateTeleop(
  const ModeContext & context, PoseSequenceManager & poses,
  const std::vector<uint16_t> & preset_ids)
{
  if (!plugin_->activate(context)) {throw std::runtime_error("mode activation was rejected");}
  if (paused_presets_ != 0 && !poses.startPreset(paused_presets_, preset_ids, context)) {
    throw std::runtime_error("preset overlay reactivation was rejected");
  }
  active_ = target_;
  ready_ = true;
  cancelTransition(poses);
}

void ModeManager::activateModel(
  rclcpp::Node & node, const ModeConfiguration & config, const ModeContext & context)
{
  deactivate();
  const auto & entry = model_modes_.at(model_requested_);
  if (!entry.directJoint()) {
    plugin_ = loader_.createSharedInstance(entry.plugin);
    if (!plugin_->configure(node, entry.parameter_prefix, config) || !plugin_->activate(context)) {
      throw std::runtime_error("Model action mode rejected configuration or activation");
    }
  }
  active_ = model_requested_;
  ready_ = true;
  pending_ = false;
}
}  // namespace cyclo_teleoperation
