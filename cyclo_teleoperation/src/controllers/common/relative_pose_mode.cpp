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

#include "cyclo_teleoperation/controllers/common/relative_pose_mode.hpp"

#include <pluginlib/class_list_macros.hpp>

#include "common/type_define.hpp"

namespace cyclo_teleoperation::controllers::common
{
bool RelativePoseMode::configure(
  rclcpp::Node & node,
  const std::string & prefix,
  const ModeConfiguration & configuration)
{
  configuration_ = configuration;
  anchors_.clear();
  for (const auto & group : configuration_.control_groups) {
    if (group.follower_eef.empty() || group.leader_eef.empty()) {
      return false;
    }
    anchors_.emplace(group.id, Anchor{});
  }
  if (anchors_.empty()) {
    return false;
  }
  auto parameter = [&node](const std::string & name, const double default_value) {
      if (!node.has_parameter(name)) {
        return node.declare_parameter(name, default_value);
      }
      return node.get_parameter(name).as_double();
    };
  kp_position_ = parameter(prefix + ".kp_position", 50.0);
  kp_orientation_ = parameter(prefix + ".kp_orientation", 50.0);
  weight_position_ = parameter(prefix + ".weight_position", 10.0);
  weight_orientation_ = parameter(prefix + ".weight_orientation", 1.0);
  return kp_position_ > 0.0 && kp_orientation_ > 0.0 &&
         weight_position_ > 0.0 && weight_orientation_ > 0.0;
}

bool RelativePoseMode::activate(const ModeContext & context)
{
  configuration_.leader_kinematics->updateState(
    context.leader_position, Eigen::VectorXd::Zero(context.leader_position.size()));
  configuration_.follower_kinematics->updateState(
    context.follower_position, context.follower_velocity);
  for (auto & anchor : anchors_) {
    anchor.second = Anchor{};
  }
  captureAnchors(context.enabled_groups);
  return true;
}

void RelativePoseMode::onGroupsEnabled(
  const ControlGroupMask groups, const ModeContext & context)
{
  configuration_.leader_kinematics->updateState(
    context.leader_position, Eigen::VectorXd::Zero(context.leader_position.size()));
  configuration_.follower_kinematics->updateState(
    context.follower_position, context.follower_velocity);
  captureAnchors(groups);
}

void RelativePoseMode::captureAnchors(const ControlGroupMask groups)
{
  for (const auto & group : configuration_.control_groups) {
    if (!containsControlGroup(groups, group.id)) {
      continue;
    }
    auto & anchor = anchors_.at(group.id);
    anchor.leader = configuration_.leader_kinematics->getPose(group.leader_eef);
    anchor.follower = configuration_.follower_kinematics->getPose(group.follower_eef);
    anchor.valid = true;
  }
}

Eigen::Matrix<double, 6, 1> RelativePoseMode::desiredVelocity(
  const Eigen::Affine3d & current,
  const Eigen::Affine3d & goal) const
{
  Eigen::Matrix<double, 6, 1> velocity =
    Eigen::Matrix<double, 6, 1>::Zero();
  velocity.head<3>() = kp_position_ * (goal.translation() - current.translation());
  velocity.tail<3>() =
    kp_orientation_ * cyclo_motion_controller::common::shortestOrientationError(
    goal.linear(), current.linear());
  return velocity;
}

bool RelativePoseMode::update(const ModeContext & context, ModeOutput & output)
{
  auto add_task = [&](const ControlGroupConfiguration & group, const Anchor & anchor) {
      const Eigen::Affine3d current =
        configuration_.follower_kinematics->getPose(group.follower_eef);
      const Eigen::Affine3d leader_current =
        configuration_.leader_kinematics->getPose(group.leader_eef);
      // Apply relative leader motion in the shared base-link axes. Composing the complete
      // transforms on the right would express translation in the anchored end-effector frame.
      Eigen::Affine3d goal = anchor.follower;
      goal.translation() += leader_current.translation() - anchor.leader.translation();
      const Eigen::Matrix3d base_frame_rotation_delta =
        leader_current.linear() * anchor.leader.linear().transpose();
      goal.linear() = base_frame_rotation_delta * anchor.follower.linear();
      TaskObjective task;
      task.link_name = group.follower_eef;
      task.desired_velocity = desiredVelocity(current, goal);
      task.weight.head<3>().setConstant(weight_position_);
      task.weight.tail<3>().setConstant(weight_orientation_);
      output.task_objectives.push_back(task);
    };

  for (const auto & group : configuration_.control_groups) {
    if (!containsControlGroup(context.enabled_groups, group.id)) {
      continue;
    }
    const auto anchor = anchors_.find(group.id);
    if (anchor == anchors_.end() || !anchor->second.valid) {
      return false;
    }
    add_task(group, anchor->second);
  }
  return true;
}
}  // namespace cyclo_teleoperation::controllers::common

PLUGINLIB_EXPORT_CLASS(
  cyclo_teleoperation::controllers::common::RelativePoseMode,
  cyclo_teleoperation::TeleoperationMode)
