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

#include "cyclo_teleoperation/robots/ai_worker/relative_pose_mode.hpp"

#include <pluginlib/class_list_macros.hpp>

#include "common/type_define.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
bool RelativePoseMode::configure(
  rclcpp::Node & node,
  const std::string & prefix,
  const ModeConfiguration & configuration)
{
  configuration_ = configuration;
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
  left_anchor_valid_ = false;
  right_anchor_valid_ = false;
  captureAnchor(context.enabled_arms);
  return true;
}

void RelativePoseMode::onArmsEnabled(
  const uint8_t arms, const ModeContext & context)
{
  configuration_.leader_kinematics->updateState(
    context.leader_position, Eigen::VectorXd::Zero(context.leader_position.size()));
  configuration_.follower_kinematics->updateState(
    context.follower_position, context.follower_velocity);
  captureAnchor(arms);
}

void RelativePoseMode::captureAnchor(const uint8_t arms)
{
  if ((arms & kLeftArm) != 0) {
    left_leader_anchor_ =
      configuration_.leader_kinematics->getPose(configuration_.leader_left_eef);
    left_follower_anchor_ =
      configuration_.follower_kinematics->getPose(configuration_.follower_left_eef);
    left_anchor_valid_ = true;
  }
  if ((arms & kRightArm) != 0) {
    right_leader_anchor_ =
      configuration_.leader_kinematics->getPose(configuration_.leader_right_eef);
    right_follower_anchor_ =
      configuration_.follower_kinematics->getPose(configuration_.follower_right_eef);
    right_anchor_valid_ = true;
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
  if (
    ((context.enabled_arms & kLeftArm) != 0 && !left_anchor_valid_) ||
    ((context.enabled_arms & kRightArm) != 0 && !right_anchor_valid_))
  {
    return false;
  }

  auto add_task = [&](const std::string & follower_link,
    const std::string & leader_link,
    const Eigen::Affine3d & follower_anchor,
    const Eigen::Affine3d & leader_anchor) {
      const Eigen::Affine3d current =
        configuration_.follower_kinematics->getPose(follower_link);
      const Eigen::Affine3d leader_current =
        configuration_.leader_kinematics->getPose(leader_link);
      // Apply relative leader motion in the shared base-link axes. Composing the complete
      // transforms on the right would express translation in the anchored end-effector frame.
      Eigen::Affine3d goal = follower_anchor;
      goal.translation() += leader_current.translation() - leader_anchor.translation();
      const Eigen::Matrix3d base_frame_rotation_delta =
        leader_current.linear() * leader_anchor.linear().transpose();
      goal.linear() = base_frame_rotation_delta * follower_anchor.linear();
      TaskObjective task;
      task.link_name = follower_link;
      task.desired_velocity = desiredVelocity(current, goal);
      task.weight.head<3>().setConstant(weight_position_);
      task.weight.tail<3>().setConstant(weight_orientation_);
      output.task_objectives.push_back(task);
    };

  if ((context.enabled_arms & kLeftArm) != 0) {
    add_task(
      configuration_.follower_left_eef, configuration_.leader_left_eef,
      left_follower_anchor_, left_leader_anchor_);
  }
  if ((context.enabled_arms & kRightArm) != 0) {
    add_task(
      configuration_.follower_right_eef, configuration_.leader_right_eef,
      right_follower_anchor_, right_leader_anchor_);
  }
  return true;
}
}  // namespace cyclo_teleoperation::robots::ai_worker

PLUGINLIB_EXPORT_CLASS(
  cyclo_teleoperation::robots::ai_worker::RelativePoseMode,
  cyclo_teleoperation::TeleoperationMode)
