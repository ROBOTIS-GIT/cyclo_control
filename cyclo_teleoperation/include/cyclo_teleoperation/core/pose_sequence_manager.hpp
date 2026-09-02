// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "cyclo_teleoperation/core/types.hpp"

namespace cyclo_teleoperation
{
class PoseSequenceManager
{
public:
  bool configure(
    rclcpp::Node & node,
    const ModeConfiguration & configuration,
    const std::vector<int64_t> & available_modes,
    const std::vector<int64_t> & available_presets);

  bool hasInitialPose(uint16_t mode) const;
  ControlGroupMask initialPoseGroups(uint16_t mode) const;
  bool startInitialPose(uint16_t mode, const ModeContext & context);
  bool updateInitialPose(const ModeContext & context, ModeOutput & output);
  bool initialPoseMoving() const;
  ControlGroupMask activeInitialPoseGroups() const;
  void cancelInitialPose();

  bool startFinalInitialPose(
    uint16_t mode, ControlGroupMask groups, const ModeContext & context);
  bool updateFinalInitialPoses(const ModeContext & context, ModeOutput & output);
  void cancelFinalInitialPoses(ControlGroupMask groups);
  ControlGroupMask activeFinalInitialPoseGroups() const;
  ControlGroupMask movingFinalInitialPoseGroups() const;
  uint8_t finalInitialPoseState(ControlGroupId group) const;

  bool hasExitPose(uint16_t mode) const;
  bool startExitPose(uint16_t mode, const ModeContext & context);
  bool updateExitPose(const ModeContext & context, ModeOutput & output);
  bool exitPoseMoving() const;
  ControlGroupMask activeExitPoseGroups() const;
  void cancelExitPose();

  bool hasPreset(uint16_t preset, ControlGroupId group) const;
  bool startPreset(
    ControlGroupMask groups,
    const std::vector<uint16_t> & preset_ids,
    const ModeContext & context);
  bool updatePresets(const ModeContext & context, ModeOutput & output);
  void cancelPresets(ControlGroupMask groups);
  ControlGroupMask activePresetGroups() const;
  ControlGroupMask movingPresetGroups() const;
  uint8_t presetState(ControlGroupId group) const;

  const std::string & errorMessage() const {return error_message_;}

private:
  enum class Purpose : uint8_t
  {
    kNone,
    kInitialPose,
    kFinalInitialPose,
    kExitPose,
    kPreset,
  };

  struct Step
  {
    std::string name;
    Eigen::VectorXd target;
    Eigen::VectorXd auxiliary_target;
  };

  struct Sequence
  {
    std::vector<Step> steps;
    double duration = 3.0;
    double completion_tolerance = 0.03;
    double timeout = 10.0;
  };

  using GroupSequence = std::unordered_map<ControlGroupId, Sequence>;

  struct Runner
  {
    const Sequence * sequence = nullptr;
    Eigen::VectorXd start;
    Eigen::VectorXd auxiliary_start;
    size_t step_index = 0;
    double start_time = 0.0;
    Purpose purpose = Purpose::kNone;
    bool active = false;
    bool moving = false;
    uint8_t state = 0;
  };

  GroupSequence loadSequence(
    rclcpp::Node & node,
    const std::string & prefix,
    const std::vector<std::string> & step_names) const;
  bool startRunner(
    Runner & runner,
    const Sequence & sequence,
    Purpose purpose,
    const ControlGroupConfiguration & group,
    const ModeContext & context,
    bool final_step_only = false);
  bool updateRunner(
    Runner & runner,
    Purpose purpose,
    const ControlGroupConfiguration & group,
    const ModeContext & context,
    ModeOutput & output);
  bool startGroupSequence(
    const GroupSequence & sequence,
    Purpose purpose,
    ControlGroupMask groups,
    const ModeContext & context,
    bool final_step_only = false);
  bool updateRunners(Purpose purpose, const ModeContext & context, ModeOutput & output);
  static void cancelRunner(Runner & runner, Purpose purpose);
  void cancelRunners(Purpose purpose, ControlGroupMask groups);
  ControlGroupMask activeGroups(Purpose purpose, bool moving_only) const;
  const ControlGroupConfiguration * group(ControlGroupId id) const;
  uint8_t runnerState(ControlGroupId id, Purpose purpose) const;

  ModeConfiguration configuration_;
  std::unordered_map<uint16_t, GroupSequence> initial_poses_;
  std::unordered_map<uint16_t, GroupSequence> exit_poses_;
  std::unordered_map<uint16_t, GroupSequence> presets_;
  std::unordered_map<ControlGroupId, Runner> runners_;
  double kp_ = 30.0;
  double tracking_weight_ = 10.0;
  std::string error_message_;
};
}  // namespace cyclo_teleoperation
