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

#include "cyclo_teleoperation/robots/ai_worker/movej_mode.hpp"

#include <algorithm>
#include <cmath>

#include <pluginlib/class_list_macros.hpp>

namespace cyclo_teleoperation::robots::ai_worker
{
bool MoveJMode::configure(
  rclcpp::Node & node,
  const std::string & prefix,
  const ModeConfiguration & configuration)
{
  left_indices_ = configuration.left_arm_indices;
  right_indices_ = configuration.right_arm_indices;
  auto parameter = [&node](const std::string & name, const double default_value) {
      if (!node.has_parameter(name)) {
        return node.declare_parameter(name, default_value);
      }
      return node.get_parameter(name).as_double();
    };
  kp_joint_ = parameter(prefix + ".kp_joint", 50.0);
  tracking_weight_ = parameter(prefix + ".tracking_weight", 10.0);
  return kp_joint_ > 0.0 && tracking_weight_ > 0.0;
}

bool MoveJMode::activate(const ModeContext & context)
{
  left_trajectory_ = ArmTrajectory{};
  right_trajectory_ = ArmTrajectory{};
  onArmsEnabled(context.enabled_arms, context);
  return true;
}

void MoveJMode::beginSlowStart(
  const std::vector<int> & indices,
  ArmTrajectory & trajectory,
  const uint64_t command_sequence,
  const ModeContext & context)
{
  trajectory = ArmTrajectory{};
  trajectory.start = context.follower_position;
  trajectory.goal = context.leader_reference;
  trajectory.last_sequence = command_sequence;
  trajectory.waiting_for_command = true;
  for (const int index : indices) {
    trajectory.start[index] = context.follower_position[index];
    trajectory.goal[index] = context.leader_reference[index];
  }
}

void MoveJMode::onArmsEnabled(const uint8_t arms, const ModeContext & context)
{
  if ((arms & kLeftArm) != 0) {
    beginSlowStart(
      left_indices_, left_trajectory_, context.left_leader_sequence, context);
  }
  if ((arms & kRightArm) != 0) {
    beginSlowStart(
      right_indices_, right_trajectory_, context.right_leader_sequence, context);
  }
}

void MoveJMode::updateArm(
  const std::vector<int> & indices,
  const bool enabled,
  const double command_duration,
  const uint64_t command_sequence,
  ArmTrajectory & trajectory,
  const ModeContext & context,
  ModeOutput & output)
{
  if (!enabled) {
    return;
  }

  constexpr double kTimedCommandEpsilon = 1e-6;
  if (trajectory.waiting_for_command && command_sequence == trajectory.last_sequence) {
    for (const int index : indices) {
      output.desired_joint_velocity[index] = 0.0;
      output.joint_tracking_weight[index] = tracking_weight_;
    }
    return;
  }
  if (command_sequence != trajectory.last_sequence) {
    trajectory.last_sequence = command_sequence;
    trajectory.waiting_for_command = false;
    if (!trajectory.slow_start_complete && command_duration > kTimedCommandEpsilon) {
      trajectory.start = context.follower_position;
      trajectory.goal = context.leader_reference;
      trajectory.start_time = context.now_seconds;
      trajectory.duration = command_duration;
      trajectory.active = true;
    } else {
      trajectory.active = false;
      trajectory.slow_start_complete = true;
    }
  }

  double interpolation_ratio = 1.0;
  Eigen::VectorXd feedforward = Eigen::VectorXd::Zero(context.follower_position.size());
  if (trajectory.active) {
    const double elapsed = context.now_seconds - trajectory.start_time;
    interpolation_ratio = std::clamp(elapsed / trajectory.duration, 0.0, 1.0);
    feedforward = (trajectory.goal - trajectory.start) / trajectory.duration;
    if (interpolation_ratio >= 1.0) {
      trajectory.active = false;
    }
  }

  for (const int index : indices) {
    const double reference = trajectory.active ?
      trajectory.start[index] +
      interpolation_ratio * (trajectory.goal[index] - trajectory.start[index]) :
      context.leader_reference[index];
    const double reference_velocity = trajectory.active ? feedforward[index] : 0.0;
    output.desired_joint_velocity[index] = reference_velocity +
      kp_joint_ * (reference - context.follower_position[index]);
    output.joint_tracking_weight[index] = tracking_weight_;
  }
}

uint8_t MoveJMode::timedCommandFeedbackSyncArms(const ModeContext & context) const
{
  constexpr double kTimedCommandEpsilon = 1e-6;
  uint8_t arms = 0;
  auto add_arm = [&](
    const uint8_t arm,
    const double duration,
    const uint64_t sequence,
    const ArmTrajectory & trajectory)
    {
      if (
        (context.enabled_arms & arm) != 0 &&
        sequence != trajectory.last_sequence &&
        !trajectory.slow_start_complete &&
        duration > kTimedCommandEpsilon)
      {
        arms |= arm;
      }
    };
  add_arm(
    kLeftArm, context.left_leader_duration,
    context.left_leader_sequence, left_trajectory_);
  add_arm(
    kRightArm, context.right_leader_duration,
    context.right_leader_sequence, right_trajectory_);
  return arms;
}

bool MoveJMode::update(const ModeContext & context, ModeOutput & output)
{
  updateArm(
    left_indices_, (context.enabled_arms & kLeftArm) != 0,
    context.left_leader_duration, context.left_leader_sequence,
    left_trajectory_, context, output);
  updateArm(
    right_indices_, (context.enabled_arms & kRightArm) != 0,
    context.right_leader_duration, context.right_leader_sequence,
    right_trajectory_, context, output);
  return true;
}
}  // namespace cyclo_teleoperation::robots::ai_worker

PLUGINLIB_EXPORT_CLASS(
  cyclo_teleoperation::robots::ai_worker::MoveJMode,
  cyclo_teleoperation::TeleoperationMode)
