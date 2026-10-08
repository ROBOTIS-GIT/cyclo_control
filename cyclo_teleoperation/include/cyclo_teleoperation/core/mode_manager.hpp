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
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "cyclo_teleoperation/core/mode_registry.hpp"
#include "cyclo_teleoperation/core/pose_sequence_manager.hpp"

namespace cyclo_teleoperation
{
// Owns controller instances and exit -> initial -> activation sequencing.
// No ROS command publishers, source arbitration or QP integration live here.
class ModeManager
{
public:
  enum class Phase {kIdle, kExitPose, kInitialPose, kActivate};
  ModeManager();
  void configure(rclcpp::Node & node);
  const ModeRegistry & teleopModes() const {return teleop_modes_;}
  const ModeRegistry & modelModes() const {return model_modes_;}
  bool available(uint16_t id) const {return teleop_modes_.contains(id);}
  uint16_t requested() const {return requested_;}
  uint16_t active() const {return active_;}
  uint16_t target() const {return target_;}
  uint16_t modelRequested() const {return model_requested_;}
  bool ready() const {return ready_;}
  bool pending() const {return pending_;}
  bool started() const {return started_;}
  Phase phase() const {return phase_;}
  TeleoperationMode * plugin() const {return plugin_.get();}

  void selectSource(bool teleop);
  void selectTeleop(uint16_t id, bool activate);
  void selectModel(uint16_t id);
  void requestReconfigure(bool teleop);
  void deactivate();
  void invalidate() {ready_ = false;}
  void recordRequest(uint16_t id) {requested_ = id;}
  void setPending(bool pending) {pending_ = pending;}
  void cancelTransition(PoseSequenceManager & poses);
  void retarget(PoseSequenceManager & poses);
  void beginTransition(PoseSequenceManager & poses, const ModeContext & context);
  void startInitial(PoseSequenceManager & poses, const ModeContext & context);
  void configureTeleop(rclcpp::Node & node, const ModeConfiguration & config);
  void activateTeleop(
    const ModeContext & context,
    PoseSequenceManager & poses, const std::vector<uint16_t> & preset_ids);
  void activateModel(
    rclcpp::Node & node, const ModeConfiguration & config, const ModeContext & context);

private:
  pluginlib::ClassLoader<TeleoperationMode> loader_;
  ModeRegistry teleop_modes_, model_modes_;
  std::shared_ptr<TeleoperationMode> plugin_;
  uint16_t teleop_requested_ = 0, model_requested_ = 0;
  uint16_t requested_ = 0, active_ = 0, target_ = 0;
  bool ready_ = false, pending_ = false, started_ = false;
  Phase phase_ = Phase::kIdle;
  ControlGroupMask paused_presets_ = 0;
};
}  // namespace cyclo_teleoperation
