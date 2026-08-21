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

#include "cyclo_teleoperation/robots/ai_worker/pose_sequence_manager.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cyclo_teleoperation::robots::ai_worker
{
PoseSequenceManager::BimanualSequence PoseSequenceManager::loadSequence(
  rclcpp::Node & node,
  const std::string & prefix,
  const std::vector<std::string> & step_names) const
{
  BimanualSequence result;
  auto load_arm = [&](const std::string & arm, const size_t expected, Sequence & sequence) {
      for (const auto & step_name : step_names) {
        const std::string parameter =
          prefix + ".steps." + step_name + "." + arm + ".positions";
        const auto values = node.get_parameter(parameter).as_double_array();
        if (values.empty()) {
          continue;
        }
        if (values.size() != expected) {
          throw std::runtime_error(
                  parameter + " must contain " + std::to_string(expected) + " joints");
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
    };
  load_arm("left", left_indices_.size(), result.left);
  load_arm("right", right_indices_.size(), result.right);
  return result;
}

bool PoseSequenceManager::configure(
  rclcpp::Node & node,
  const ModeConfiguration & configuration,
  const std::vector<int64_t> & available_modes,
  const std::vector<int64_t> & available_presets)
{
  left_indices_ = configuration.left_arm_indices;
  right_indices_ = configuration.right_arm_indices;
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
        if (sequence.left.steps.empty() && sequence.right.steps.empty()) {
          throw std::runtime_error(
                  prefix + ".enabled is true, but no arm positions are configured");
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
    if (sequence.left.steps.empty() && sequence.right.steps.empty()) {
      throw std::runtime_error(prefix + " has no configured arm positions");
    }
    presets_[preset] = std::move(sequence);
  }
  return true;
}

bool PoseSequenceManager::hasInitialPose(const uint16_t mode) const
{
  return initial_poses_.count(mode) != 0;
}

uint8_t PoseSequenceManager::initialPoseArms(const uint16_t mode) const
{
  const auto sequence = initial_poses_.find(mode);
  if (sequence == initial_poses_.end()) {
    return 0;
  }
  uint8_t arms = 0;
  if (!sequence->second.left.steps.empty()) {
    arms |= kLeftArm;
  }
  if (!sequence->second.right.steps.empty()) {
    arms |= kRightArm;
  }
  return arms;
}

bool PoseSequenceManager::startRunner(
  Runner & runner,
  const Sequence & sequence,
  const Purpose purpose,
  const std::vector<int> & indices,
  const ModeContext & context,
  const bool final_step_only)
{
  if (sequence.steps.empty()) {
    return true;
  }
  runner.sequence = &sequence;
  runner.start.resize(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    runner.start[i] = context.follower_position[indices[i]];
  }
  runner.step_index = final_step_only ? sequence.steps.size() - 1 : 0;
  runner.start_time = context.now_seconds;
  runner.purpose = purpose;
  runner.active = true;
  runner.moving = true;
  runner.state = 2;
  return true;
}

bool PoseSequenceManager::startInitialPose(
  const uint16_t mode,
  const ModeContext & context)
{
  cancelInitialPose();
  error_message_.clear();
  const auto sequence = initial_poses_.find(mode);
  if (sequence == initial_poses_.end()) {
    return true;
  }
  return startRunner(
    left_runner_, sequence->second.left, Purpose::kInitialPose,
    left_indices_, context) &&
         startRunner(
    right_runner_, sequence->second.right, Purpose::kInitialPose,
    right_indices_, context);
}

bool PoseSequenceManager::updateRunner(
  Runner & runner,
  const Purpose purpose,
  const std::vector<int> & indices,
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
  for (size_t i = 0; i < indices.size(); ++i) {
    const int index = indices[i];
    const double displacement = step.target[i] - runner.start[i];
    const double reference =
      runner.start[i] + alpha * displacement;
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
  for (size_t i = 0; i < indices.size(); ++i) {
    maximum_error = std::max(
      maximum_error,
      std::abs(step.target[i] - context.measured_follower_position[indices[i]]));
  }
  if (
    elapsed >= runner.sequence->duration &&
    maximum_error <= runner.sequence->completion_tolerance)
  {
    if (runner.step_index + 1 < runner.sequence->steps.size()) {
      ++runner.step_index;
      for (size_t i = 0; i < indices.size(); ++i) {
        runner.start[i] = context.follower_position[indices[i]];
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
      "Pose sequence step '" + step.name +
      "' timed out with maximum joint error " + std::to_string(maximum_error);
    return false;
  }
  return true;
}

bool PoseSequenceManager::updateInitialPose(
  const ModeContext & context,
  ModeOutput & output)
{
  return updateRunner(
    left_runner_, Purpose::kInitialPose, left_indices_, context, output) &&
         updateRunner(
    right_runner_, Purpose::kInitialPose, right_indices_, context, output);
}

bool PoseSequenceManager::initialPoseMoving() const
{
  return activeArms(Purpose::kInitialPose, true) != 0;
}

uint8_t PoseSequenceManager::activeInitialPoseArms() const
{
  return activeArms(Purpose::kInitialPose, false);
}

void PoseSequenceManager::cancelRunner(Runner & runner, const Purpose purpose)
{
  if (runner.purpose != purpose) {
    return;
  }
  runner = Runner{};
}

void PoseSequenceManager::cancelInitialPose()
{
  cancelRunner(left_runner_, Purpose::kInitialPose);
  cancelRunner(right_runner_, Purpose::kInitialPose);
}

bool PoseSequenceManager::startFinalInitialPose(
  const uint16_t mode,
  const uint8_t arms,
  const ModeContext & context)
{
  error_message_.clear();
  const auto sequence = initial_poses_.find(mode);
  if (sequence == initial_poses_.end()) {
    error_message_ = "Initial pose is disabled for control mode " + std::to_string(mode);
    return false;
  }

  if (arms == 0) {
    error_message_ = "No arm was selected for the initial pose";
    return false;
  }
  if ((arms & kLeftArm) != 0 && sequence->second.left.steps.empty()) {
    error_message_ = "Left initial pose is not configured for control mode " +
      std::to_string(mode);
    return false;
  }
  if ((arms & kRightArm) != 0 && sequence->second.right.steps.empty()) {
    error_message_ = "Right initial pose is not configured for control mode " +
      std::to_string(mode);
    return false;
  }

  bool success = true;
  if ((arms & kLeftArm) != 0) {
    success = startRunner(
      left_runner_, sequence->second.left, Purpose::kFinalInitialPose,
      left_indices_, context, true) && success;
  }
  if ((arms & kRightArm) != 0) {
    success = startRunner(
      right_runner_, sequence->second.right, Purpose::kFinalInitialPose,
      right_indices_, context, true) && success;
  }
  return success;
}

bool PoseSequenceManager::updateFinalInitialPoses(
  const ModeContext & context,
  ModeOutput & output)
{
  return updateRunner(
    left_runner_, Purpose::kFinalInitialPose, left_indices_, context, output) &&
         updateRunner(
    right_runner_, Purpose::kFinalInitialPose, right_indices_, context, output);
}

void PoseSequenceManager::cancelFinalInitialPoses(const uint8_t arms)
{
  if ((arms & kLeftArm) != 0) {
    cancelRunner(left_runner_, Purpose::kFinalInitialPose);
  }
  if ((arms & kRightArm) != 0) {
    cancelRunner(right_runner_, Purpose::kFinalInitialPose);
  }
}

uint8_t PoseSequenceManager::activeFinalInitialPoseArms() const
{
  return activeArms(Purpose::kFinalInitialPose, false);
}

uint8_t PoseSequenceManager::movingFinalInitialPoseArms() const
{
  return activeArms(Purpose::kFinalInitialPose, true);
}

uint8_t PoseSequenceManager::leftFinalInitialPoseState() const
{
  return left_runner_.purpose == Purpose::kFinalInitialPose ? left_runner_.state : 0;
}

uint8_t PoseSequenceManager::rightFinalInitialPoseState() const
{
  return right_runner_.purpose == Purpose::kFinalInitialPose ? right_runner_.state : 0;
}

bool PoseSequenceManager::hasExitPose(const uint16_t mode) const
{
  return exit_poses_.count(mode) != 0;
}

bool PoseSequenceManager::startExitPose(
  const uint16_t mode,
  const ModeContext & context)
{
  cancelExitPose();
  error_message_.clear();
  const auto sequence = exit_poses_.find(mode);
  if (sequence == exit_poses_.end()) {
    return true;
  }
  return startRunner(
    left_runner_, sequence->second.left, Purpose::kExitPose,
    left_indices_, context) &&
         startRunner(
    right_runner_, sequence->second.right, Purpose::kExitPose,
    right_indices_, context);
}

bool PoseSequenceManager::updateExitPose(
  const ModeContext & context,
  ModeOutput & output)
{
  return updateRunner(
    left_runner_, Purpose::kExitPose, left_indices_, context, output) &&
         updateRunner(
    right_runner_, Purpose::kExitPose, right_indices_, context, output);
}

bool PoseSequenceManager::exitPoseMoving() const
{
  return activeArms(Purpose::kExitPose, true) != 0;
}

uint8_t PoseSequenceManager::activeExitPoseArms() const
{
  return activeArms(Purpose::kExitPose, false);
}

void PoseSequenceManager::cancelExitPose()
{
  cancelRunner(left_runner_, Purpose::kExitPose);
  cancelRunner(right_runner_, Purpose::kExitPose);
}

bool PoseSequenceManager::hasPreset(const uint16_t preset, const uint8_t arm) const
{
  const auto sequence = presets_.find(preset);
  if (sequence == presets_.end()) {
    return false;
  }
  if (arm == kLeftArm) {
    return !sequence->second.left.steps.empty();
  }
  if (arm == kRightArm) {
    return !sequence->second.right.steps.empty();
  }
  return false;
}

bool PoseSequenceManager::startPreset(
  const uint8_t arms,
  const uint16_t left_preset_id,
  const uint16_t right_preset_id,
  const ModeContext & context)
{
  error_message_.clear();
  bool success = true;
  if ((arms & kLeftArm) != 0) {
    const auto preset = presets_.find(left_preset_id);
    success = preset != presets_.end() &&
      startRunner(
      left_runner_, preset->second.left, Purpose::kPreset,
      left_indices_, context) && success;
  }
  if ((arms & kRightArm) != 0) {
    const auto preset = presets_.find(right_preset_id);
    success = preset != presets_.end() &&
      startRunner(
      right_runner_, preset->second.right, Purpose::kPreset,
      right_indices_, context) && success;
  }
  return success;
}

bool PoseSequenceManager::updatePresets(
  const ModeContext & context,
  ModeOutput & output)
{
  return updateRunner(
    left_runner_, Purpose::kPreset, left_indices_, context, output) &&
         updateRunner(
    right_runner_, Purpose::kPreset, right_indices_, context, output);
}

void PoseSequenceManager::cancelPresets(const uint8_t arms)
{
  if ((arms & kLeftArm) != 0) {
    cancelRunner(left_runner_, Purpose::kPreset);
  }
  if ((arms & kRightArm) != 0) {
    cancelRunner(right_runner_, Purpose::kPreset);
  }
}

uint8_t PoseSequenceManager::activeArms(
  const Purpose purpose,
  const bool moving_only) const
{
  uint8_t arms = 0;
  if (
    left_runner_.purpose == purpose && left_runner_.active &&
    (!moving_only || left_runner_.moving))
  {
    arms |= kLeftArm;
  }
  if (
    right_runner_.purpose == purpose && right_runner_.active &&
    (!moving_only || right_runner_.moving))
  {
    arms |= kRightArm;
  }
  return arms;
}

uint8_t PoseSequenceManager::activePresetArms() const
{
  return activeArms(Purpose::kPreset, false);
}

uint8_t PoseSequenceManager::movingPresetArms() const
{
  return activeArms(Purpose::kPreset, true);
}

uint8_t PoseSequenceManager::leftPresetState() const
{
  return left_runner_.purpose == Purpose::kPreset ? left_runner_.state : 0;
}

uint8_t PoseSequenceManager::rightPresetState() const
{
  return right_runner_.purpose == Purpose::kPreset ? right_runner_.state : 0;
}
}  // namespace cyclo_teleoperation::robots::ai_worker
