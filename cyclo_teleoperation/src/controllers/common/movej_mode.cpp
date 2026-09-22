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

#include "cyclo_teleoperation/controllers/common/movej_mode.hpp"

#include <pluginlib/class_list_macros.hpp>

namespace cyclo_teleoperation::controllers::common
{
bool MoveJMode::configure(
  rclcpp::Node & node,
  const std::string & prefix,
  const ModeConfiguration & configuration)
{
  configuration_ = configuration;
  trajectories_.clear();
  for (const auto & group : configuration_.control_groups) {
    trajectories_.emplace(group.id, ArmTrajectory{});
  }
  if (trajectories_.empty()) {
    return false;
  }
  auto parameter = [&node](const std::string & name, const double default_value) {
      if (!node.has_parameter(name)) {
        return node.declare_parameter(name, default_value);
      }
      return node.get_parameter(name).as_double();
    };
  kp_joint_ = parameter(prefix + ".kp_joint", 50.0);
  tracking_weight_ = parameter(prefix + ".tracking_weight", 10.0);
  constraints_ = ControllerConstraints::declareAndLoad(node, prefix + ".constraints");
  return kp_joint_ > 0.0 && tracking_weight_ > 0.0;
}

bool MoveJMode::activate(const ModeContext & context)
{
  for (auto & trajectory : trajectories_) {
    trajectory.second = ArmTrajectory{};
  }
  onGroupsEnabled(context.enabled_groups, context);
  return true;
}

void MoveJMode::beginSlowStart(
  ArmTrajectory & trajectory,
  const uint64_t command_sequence)
{
  trajectory = ArmTrajectory{};
  trajectory.last_sequence = command_sequence;
  trajectory.waiting_for_command = true;
  trajectory.timed_transition_complete = false;
}

void MoveJMode::onGroupsEnabled(
  const ControlGroupMask groups, const ModeContext & context)
{
  for (const auto & group : configuration_.control_groups) {
    if (!containsControlGroup(groups, group.id)) {
      continue;
    }
    if (group.id >= context.group_states.size()) {
      continue;
    }
    beginSlowStart(
      trajectories_.at(group.id),
      context.group_states[group.id].leader_sequence);
  }
}

void MoveJMode::updateArm(
  const ControlGroupConfiguration & group,
  const ControlGroupState & state,
  ArmTrajectory & trajectory,
  const ModeContext & context,
  ModeOutput & output)
{
  constexpr double kTimedCommandEpsilon = 1e-6;
  const double command_duration = state.leader_duration;
  const uint64_t command_sequence = state.leader_sequence;
  if (trajectory.waiting_for_command && command_sequence == trajectory.last_sequence) {
    for (const int index : group.follower_joint_indices) {
      output.desired_joint_velocity[index] = 0.0;
      output.joint_tracking_weight[index] = tracking_weight_;
    }
    return;
  }
  if (command_sequence != trajectory.last_sequence) {
    trajectory.last_sequence = command_sequence;
    trajectory.waiting_for_command = false;
    if (!trajectory.timed_transition_complete && command_duration > kTimedCommandEpsilon) {
      Eigen::VectorXd start(group.follower_joint_indices.size());
      Eigen::VectorXd goal(group.follower_joint_indices.size());
      for (size_t i = 0; i < group.follower_joint_indices.size(); ++i) {
        const int index = group.follower_joint_indices[i];
        start[i] = context.follower_position[index];
        goal[i] = context.leader_reference[index];
      }
      if (!trajectory.interpolator.start(
          start, goal, context.now_seconds, command_duration))
      {
        trajectory.timed_transition_complete = true;
      }
    } else {
      trajectory.interpolator.reset();
      trajectory.timed_transition_complete = true;
    }
  }

  JointTrajectorySample sample = trajectory.interpolator.sample(context.now_seconds);
  if (sample.complete) {
    trajectory.interpolator.reset();
  }
  if (sample.position.size() == 0 || sample.complete) {
    sample.position.resize(group.follower_joint_indices.size());
    sample.velocity.setZero(group.follower_joint_indices.size());
    for (size_t i = 0; i < group.follower_joint_indices.size(); ++i) {
      sample.position[i] = context.leader_reference[group.follower_joint_indices[i]];
    }
  }
  applyJointTrajectoryTracking(
    group, sample, context.follower_position, kp_joint_, tracking_weight_, output);
}

ControlGroupMask MoveJMode::timedCommandFeedbackSyncGroups(
  const ModeContext & context) const
{
  constexpr double kTimedCommandEpsilon = 1e-6;
  ControlGroupMask groups = 0;
  for (const auto & group : configuration_.control_groups) {
    if (
      !containsControlGroup(context.enabled_groups, group.id) ||
      group.id >= context.group_states.size())
    {
      continue;
    }
    const auto trajectory = trajectories_.find(group.id);
    if (trajectory == trajectories_.end()) {
      continue;
    }
    const auto & state = context.group_states[group.id];
    if (
      state.leader_sequence != trajectory->second.last_sequence &&
      !trajectory->second.timed_transition_complete &&
      state.leader_duration > kTimedCommandEpsilon)
    {
      groups |= controlGroupBit(group.id);
    }
  }
  return groups;
}

bool MoveJMode::update(const ModeContext & context, ModeOutput & output)
{
  constraints_.apply(configuration_, configuredControlGroups(configuration_), output);

  for (const auto & group : configuration_.control_groups) {
    if (!containsControlGroup(context.enabled_groups, group.id)) {
      continue;
    }
    if (group.id >= context.group_states.size()) {
      return false;
    }
    updateArm(
      group, context.group_states[group.id], trajectories_.at(group.id),
      context, output);
  }
  return true;
}
}  // namespace cyclo_teleoperation::controllers::common

PLUGINLIB_EXPORT_CLASS(
  cyclo_teleoperation::controllers::common::MoveJMode,
  cyclo_teleoperation::TeleoperationMode)
