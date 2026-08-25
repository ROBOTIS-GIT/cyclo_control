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

#include "cyclo_teleoperation/core/pose_sequence_manager.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cyclo_teleoperation
{
PoseSequenceManager::GroupSequence PoseSequenceManager::loadSequence(
  rclcpp::Node & node,
  const std::string & prefix,
  const std::vector<std::string> & step_names) const
{
  GroupSequence result;
  for (const auto & group : configuration_.control_groups) {
    Sequence sequence;
    for (const auto & step_name : step_names) {
      const std::string parameter =
        prefix + ".steps." + step_name + "." + group.name + ".positions";
      const auto values = node.get_parameter(parameter).as_double_array();
      if (values.empty()) {
        continue;
      }
      if (values.size() != group.follower_joint_indices.size()) {
        throw std::runtime_error(
                parameter + " must contain " +
                std::to_string(group.follower_joint_indices.size()) + " joints");
      }
      Step step;
      step.name = step_name;
      step.target = Eigen::Map<const Eigen::VectorXd>(values.data(), values.size());
      sequence.steps.push_back(std::move(step));
    }
    sequence.duration = node.get_parameter(prefix + ".duration").as_double();
    sequence.completion_tolerance =
      node.get_parameter(prefix + ".completion_tolerance").as_double();
    sequence.timeout = node.get_parameter(prefix + ".timeout").as_double();
    if (sequence.duration <= 0.0) {
      throw std::runtime_error(prefix + ".duration must be positive");
    }
    if (sequence.completion_tolerance <= 0.0) {
      throw std::runtime_error(prefix + ".completion_tolerance must be positive");
    }
    if (sequence.timeout < sequence.duration) {
      throw std::runtime_error(prefix + ".timeout must be no shorter than duration");
    }
    if (!sequence.steps.empty()) {
      result.emplace(group.id, std::move(sequence));
    }
  }
  return result;
}

bool PoseSequenceManager::configure(
  rclcpp::Node & node,
  const ModeConfiguration & configuration,
  const std::vector<int64_t> & available_modes,
  const std::vector<int64_t> & available_presets)
{
  configuration_ = configuration;
  if (configuration_.control_groups.empty() || configuration_.control_groups.size() > 64) {
    return false;
  }
  runners_.clear();
  for (const auto & group : configuration_.control_groups) {
    if (group.id >= 64 || group.follower_joint_indices.empty()) {
      return false;
    }
    runners_.emplace(group.id, Runner{});
  }

  kp_ = node.get_parameter("pose_sequence.kp_joint").as_double();
  tracking_weight_ = node.get_parameter("pose_sequence.tracking_weight").as_double();
  if (kp_ <= 0.0 || tracking_weight_ <= 0.0) {
    return false;
  }

  for (const int64_t raw_mode : available_modes) {
    if (raw_mode <= 0 || raw_mode > UINT16_MAX) {
      continue;
    }
    const auto mode = static_cast<uint16_t>(raw_mode);
    auto load_mode_sequence = [&](const std::string & name, auto & sequences) {
        const std::string prefix =
          "control_modes." + std::to_string(mode) + "." + name;
        if (!node.get_parameter(prefix + ".enabled").as_bool()) {
          return;
        }
        const auto step_names =
          node.get_parameter(prefix + ".step_names").as_string_array();
        if (step_names.empty()) {
          throw std::runtime_error(prefix + ".enabled is true, but step_names is empty");
        }
        auto sequence = loadSequence(node, prefix, step_names);
        if (sequence.empty()) {
          throw std::runtime_error(
                  prefix + ".enabled is true, but no control group positions are configured");
        }
        sequences[mode] = std::move(sequence);
      };
    load_mode_sequence("initial_pose", initial_poses_);
    load_mode_sequence("exit_pose", exit_poses_);
  }

  for (const int64_t raw_preset : available_presets) {
    if (raw_preset <= 0 || raw_preset > UINT16_MAX) {
      continue;
    }
    const auto preset = static_cast<uint16_t>(raw_preset);
    const std::string prefix = "presets." + std::to_string(preset);
    const auto step_names = node.get_parameter(prefix + ".step_names").as_string_array();
    if (step_names.empty()) {
      throw std::runtime_error(prefix + ".step_names is empty");
    }
    auto sequence = loadSequence(node, prefix, step_names);
    if (sequence.empty()) {
      throw std::runtime_error(prefix + " has no configured control group positions");
    }
    presets_[preset] = std::move(sequence);
  }
  return true;
}

const ControlGroupConfiguration * PoseSequenceManager::group(const ControlGroupId id) const
{
  for (const auto & candidate : configuration_.control_groups) {
    if (candidate.id == id) {
      return &candidate;
    }
  }
  return nullptr;
}

bool PoseSequenceManager::hasInitialPose(const uint16_t mode) const
{
  return initial_poses_.count(mode) != 0;
}

ControlGroupMask PoseSequenceManager::initialPoseGroups(const uint16_t mode) const
{
  const auto sequence = initial_poses_.find(mode);
  if (sequence == initial_poses_.end()) {
    return 0;
  }
  ControlGroupMask groups = 0;
  for (const auto & item : sequence->second) {
    groups |= controlGroupBit(item.first);
  }
  return groups;
}

bool PoseSequenceManager::startRunner(
  Runner & runner,
  const Sequence & sequence,
  const Purpose purpose,
  const ControlGroupConfiguration & group_config,
  const ModeContext & context,
  const bool final_step_only)
{
  if (sequence.steps.empty()) {
    return true;
  }
  runner.sequence = &sequence;
  runner.start.resize(group_config.follower_joint_indices.size());
  for (size_t i = 0; i < group_config.follower_joint_indices.size(); ++i) {
    runner.start[i] = context.follower_position[group_config.follower_joint_indices[i]];
  }
  runner.step_index = final_step_only ? sequence.steps.size() - 1 : 0;
  runner.start_time = context.now_seconds;
  runner.purpose = purpose;
  runner.active = true;
  runner.moving = true;
  runner.state = 2;
  return true;
}

bool PoseSequenceManager::startGroupSequence(
  const GroupSequence & sequence,
  const Purpose purpose,
  const ControlGroupMask groups,
  const ModeContext & context,
  const bool final_step_only)
{
  for (const auto & group_config : configuration_.control_groups) {
    if (!containsControlGroup(groups, group_config.id)) {
      continue;
    }
    const auto configured = sequence.find(group_config.id);
    if (configured == sequence.end()) {
      error_message_ = "Pose sequence is not configured for control group '" +
        group_config.name + "'";
      return false;
    }
  }
  for (const auto & group_config : configuration_.control_groups) {
    if (!containsControlGroup(groups, group_config.id)) {
      continue;
    }
    const auto configured = sequence.find(group_config.id);
    if (!startRunner(
      runners_.at(group_config.id), configured->second, purpose,
      group_config, context, final_step_only))
    {
      return false;
    }
  }
  return true;
}

bool PoseSequenceManager::startInitialPose(
  const uint16_t mode, const ModeContext & context)
{
  cancelInitialPose();
  error_message_.clear();
  const auto sequence = initial_poses_.find(mode);
  if (sequence == initial_poses_.end()) {
    return true;
  }
  return startGroupSequence(
    sequence->second, Purpose::kInitialPose, initialPoseGroups(mode), context);
}

bool PoseSequenceManager::updateRunner(
  Runner & runner,
  const Purpose purpose,
  const ControlGroupConfiguration & group_config,
  const ModeContext & context,
  ModeOutput & output)
{
  if (!runner.active || runner.purpose != purpose || runner.sequence == nullptr) {
    return true;
  }
  const Step & step = runner.sequence->steps.at(runner.step_index);
  const double elapsed = context.now_seconds - runner.start_time;
  const bool interpolating = runner.moving && elapsed < runner.sequence->duration;
  const double alpha = runner.moving ?
    std::clamp(elapsed / runner.sequence->duration, 0.0, 1.0) : 1.0;
  for (size_t i = 0; i < group_config.follower_joint_indices.size(); ++i) {
    const int index = group_config.follower_joint_indices[i];
    const double displacement = step.target[i] - runner.start[i];
    const double reference = runner.start[i] + alpha * displacement;
    const double reference_velocity = interpolating ?
      displacement / runner.sequence->duration : 0.0;
    output.desired_joint_velocity[index] =
      reference_velocity + kp_ * (reference - context.follower_position[index]);
    output.joint_tracking_weight[index] = tracking_weight_;
    output.joint_position_limit_enabled[index] = false;
  }

  if (!runner.moving) {
    return true;
  }
  double maximum_error = 0.0;
  for (size_t i = 0; i < group_config.follower_joint_indices.size(); ++i) {
    maximum_error = std::max(
      maximum_error,
      std::abs(
        step.target[i] -
        context.measured_follower_position[group_config.follower_joint_indices[i]]));
  }
  if (
    elapsed >= runner.sequence->duration &&
    maximum_error <= runner.sequence->completion_tolerance)
  {
    if (runner.step_index + 1 < runner.sequence->steps.size()) {
      ++runner.step_index;
      for (size_t i = 0; i < group_config.follower_joint_indices.size(); ++i) {
        runner.start[i] = context.follower_position[group_config.follower_joint_indices[i]];
      }
      runner.start_time = context.now_seconds;
    } else {
      runner.moving = false;
      runner.state = 3;
    }
    return true;
  }
  if (elapsed >= runner.sequence->timeout) {
    runner.active = false;
    runner.moving = false;
    runner.state = 4;
    error_message_ =
      "Pose sequence step '" + step.name + "' for control group '" +
      group_config.name + "' timed out with maximum joint error " +
      std::to_string(maximum_error);
    return false;
  }
  return true;
}

bool PoseSequenceManager::updateRunners(
  const Purpose purpose, const ModeContext & context, ModeOutput & output)
{
  for (const auto & group_config : configuration_.control_groups) {
    if (!updateRunner(
        runners_.at(group_config.id), purpose, group_config, context, output))
    {
      return false;
    }
  }
  return true;
}

bool PoseSequenceManager::updateInitialPose(
  const ModeContext & context, ModeOutput & output)
{
  return updateRunners(Purpose::kInitialPose, context, output);
}

bool PoseSequenceManager::initialPoseMoving() const
{
  return activeGroups(Purpose::kInitialPose, true) != 0;
}

ControlGroupMask PoseSequenceManager::activeInitialPoseGroups() const
{
  return activeGroups(Purpose::kInitialPose, false);
}

void PoseSequenceManager::cancelRunner(Runner & runner, const Purpose purpose)
{
  if (runner.purpose == purpose) {
    runner = Runner{};
  }
}

void PoseSequenceManager::cancelRunners(
  const Purpose purpose, const ControlGroupMask groups)
{
  for (auto & runner : runners_) {
    if (containsControlGroup(groups, runner.first)) {
      cancelRunner(runner.second, purpose);
    }
  }
}

void PoseSequenceManager::cancelInitialPose()
{
  cancelRunners(Purpose::kInitialPose, std::numeric_limits<ControlGroupMask>::max());
}

bool PoseSequenceManager::startFinalInitialPose(
  const uint16_t mode,
  const ControlGroupMask groups,
  const ModeContext & context)
{
  error_message_.clear();
  const auto sequence = initial_poses_.find(mode);
  if (sequence == initial_poses_.end()) {
    error_message_ = "Initial pose is disabled for control mode " + std::to_string(mode);
    return false;
  }
  if (groups == 0) {
    error_message_ = "No control group was selected for the initial pose";
    return false;
  }
  return startGroupSequence(
    sequence->second, Purpose::kFinalInitialPose, groups, context, true);
}

bool PoseSequenceManager::updateFinalInitialPoses(
  const ModeContext & context, ModeOutput & output)
{
  return updateRunners(Purpose::kFinalInitialPose, context, output);
}

void PoseSequenceManager::cancelFinalInitialPoses(const ControlGroupMask groups)
{
  cancelRunners(Purpose::kFinalInitialPose, groups);
}

ControlGroupMask PoseSequenceManager::activeFinalInitialPoseGroups() const
{
  return activeGroups(Purpose::kFinalInitialPose, false);
}

ControlGroupMask PoseSequenceManager::movingFinalInitialPoseGroups() const
{
  return activeGroups(Purpose::kFinalInitialPose, true);
}

uint8_t PoseSequenceManager::finalInitialPoseState(const ControlGroupId group_id) const
{
  return runnerState(group_id, Purpose::kFinalInitialPose);
}

bool PoseSequenceManager::hasExitPose(const uint16_t mode) const
{
  return exit_poses_.count(mode) != 0;
}

bool PoseSequenceManager::startExitPose(
  const uint16_t mode, const ModeContext & context)
{
  cancelExitPose();
  error_message_.clear();
  const auto sequence = exit_poses_.find(mode);
  if (sequence == exit_poses_.end()) {
    return true;
  }
  ControlGroupMask groups = 0;
  for (const auto & item : sequence->second) {
    groups |= controlGroupBit(item.first);
  }
  return startGroupSequence(sequence->second, Purpose::kExitPose, groups, context);
}

bool PoseSequenceManager::updateExitPose(
  const ModeContext & context, ModeOutput & output)
{
  return updateRunners(Purpose::kExitPose, context, output);
}

bool PoseSequenceManager::exitPoseMoving() const
{
  return activeGroups(Purpose::kExitPose, true) != 0;
}

ControlGroupMask PoseSequenceManager::activeExitPoseGroups() const
{
  return activeGroups(Purpose::kExitPose, false);
}

void PoseSequenceManager::cancelExitPose()
{
  cancelRunners(Purpose::kExitPose, std::numeric_limits<ControlGroupMask>::max());
}

bool PoseSequenceManager::hasPreset(
  const uint16_t preset, const ControlGroupId group_id) const
{
  const auto sequence = presets_.find(preset);
  return sequence != presets_.end() && sequence->second.count(group_id) != 0;
}

bool PoseSequenceManager::startPreset(
  const ControlGroupMask groups,
  const std::vector<uint16_t> & preset_ids,
  const ModeContext & context)
{
  error_message_.clear();
  std::unordered_map<ControlGroupId, const Sequence *> selected_sequences;
  for (const auto & group_config : configuration_.control_groups) {
    if (!containsControlGroup(groups, group_config.id)) {
      continue;
    }
    if (group_config.id >= preset_ids.size()) {
      error_message_ = "No preset ID was supplied for control group '" +
        group_config.name + "'";
      return false;
    }
    const auto preset = presets_.find(preset_ids[group_config.id]);
    if (preset == presets_.end()) {
      error_message_ = "Unknown preset ID for control group '" + group_config.name + "'";
      return false;
    }
    const auto sequence = preset->second.find(group_config.id);
    if (sequence == preset->second.end()) {
      error_message_ = "Preset is not configured for control group '" +
        group_config.name + "'";
      return false;
    }
    selected_sequences[group_config.id] = &sequence->second;
  }
  for (const auto & group_config : configuration_.control_groups) {
    if (!containsControlGroup(groups, group_config.id)) {
      continue;
    }
    if (!startRunner(
        runners_.at(group_config.id), *selected_sequences.at(group_config.id), Purpose::kPreset,
        group_config, context))
    {
      return false;
    }
  }
  return true;
}

bool PoseSequenceManager::updatePresets(
  const ModeContext & context, ModeOutput & output)
{
  return updateRunners(Purpose::kPreset, context, output);
}

void PoseSequenceManager::cancelPresets(const ControlGroupMask groups)
{
  cancelRunners(Purpose::kPreset, groups);
}

ControlGroupMask PoseSequenceManager::activeGroups(
  const Purpose purpose, const bool moving_only) const
{
  ControlGroupMask groups = 0;
  for (const auto & runner : runners_) {
    if (
      runner.second.purpose == purpose && runner.second.active &&
      (!moving_only || runner.second.moving))
    {
      groups |= controlGroupBit(runner.first);
    }
  }
  return groups;
}

ControlGroupMask PoseSequenceManager::activePresetGroups() const
{
  return activeGroups(Purpose::kPreset, false);
}

ControlGroupMask PoseSequenceManager::movingPresetGroups() const
{
  return activeGroups(Purpose::kPreset, true);
}

uint8_t PoseSequenceManager::runnerState(
  const ControlGroupId group_id, const Purpose purpose) const
{
  const auto runner = runners_.find(group_id);
  if (runner == runners_.end() || runner->second.purpose != purpose) {
    return 0;
  }
  return runner->second.state;
}

uint8_t PoseSequenceManager::presetState(const ControlGroupId group_id) const
{
  return runnerState(group_id, Purpose::kPreset);
}
}  // namespace cyclo_teleoperation
