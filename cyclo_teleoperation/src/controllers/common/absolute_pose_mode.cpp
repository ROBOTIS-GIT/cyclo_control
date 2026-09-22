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

#include "cyclo_teleoperation/controllers/common/absolute_pose_mode.hpp"

#include <utility>

#include <pluginlib/class_list_macros.hpp>

#include "common/type_define.hpp"

namespace cyclo_teleoperation::controllers::common
{
bool AbsolutePoseMode::configure(
  rclcpp::Node & node,
  const std::string & prefix,
  const ModeConfiguration & configuration)
{
  configuration_ = configuration;
  if (configuration_.control_groups.empty()) {
    return false;
  }
  for (const auto & group : configuration_.control_groups) {
    if (group.follower_eef.empty()) {
      return false;
    }
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
  constraints_ = ControllerConstraints::declareAndLoad(node, prefix + ".constraints");
  return kp_position_ > 0.0 && kp_orientation_ > 0.0 &&
         weight_position_ > 0.0 && weight_orientation_ > 0.0;
}

bool AbsolutePoseMode::activate(const ModeContext & /*context*/)
{
  return true;
}

void AbsolutePoseMode::onGroupsEnabled(
  const ControlGroupMask /*groups*/, const ModeContext & /*context*/)
{
}

Eigen::Matrix<double, 6, 1> AbsolutePoseMode::desiredVelocity(
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

bool AbsolutePoseMode::update(const ModeContext & context, ModeOutput & output)
{
  constraints_.apply(configuration_, configuredControlGroups(configuration_), output);
  for (const auto & group : configuration_.control_groups) {
    if (!containsControlGroup(context.enabled_groups, group.id)) {
      continue;
    }
    if (
      group.id >= context.cartesian_references.size() ||
      !context.cartesian_references[group.id].valid)
    {
      return false;
    }
    TaskObjective task;
    task.link_name = group.follower_eef;
    task.desired_velocity = desiredVelocity(
      configuration_.follower_kinematics->getPose(group.follower_eef),
      context.cartesian_references[group.id].pose);
    task.weight.head<3>().setConstant(weight_position_);
    task.weight.tail<3>().setConstant(weight_orientation_);
    output.task_objectives.push_back(std::move(task));
  }
  return true;
}
}  // namespace cyclo_teleoperation::controllers::common

PLUGINLIB_EXPORT_CLASS(
  cyclo_teleoperation::controllers::common::AbsolutePoseMode,
  cyclo_teleoperation::TeleoperationMode)
