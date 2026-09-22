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

#include "cyclo_teleoperation/core/joint_trajectory_interpolator.hpp"

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
    trajectories_.at(group.id).slow_start.restart(
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
  Eigen::VectorXd start(group.follower_joint_indices.size());
  Eigen::VectorXd reference(group.follower_joint_indices.size());
  for (size_t i = 0; i < group.follower_joint_indices.size(); ++i) {
    const int index = group.follower_joint_indices[i];
    start[i] = context.follower_position[index];
    reference[i] = context.leader_reference[index];
  }
  const FixedDurationSlowStartSample slow_start_sample = trajectory.slow_start.update(
    start, reference, state.leader_sequence, state.leader_duration,
    context.now_seconds);
  JointTrajectorySample sample;
  sample.position = slow_start_sample.position;
  sample.velocity = slow_start_sample.velocity;
  sample.active = slow_start_sample.active;
  sample.complete = slow_start_sample.complete;
  applyJointTrajectoryTracking(
    group, sample, context.follower_position, kp_joint_, tracking_weight_, output);
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
