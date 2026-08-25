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

#include "cyclo_teleoperation/robots/ai_worker/elbow_up_leader_mode.hpp"

#include <algorithm>
#include <cmath>

#include <pluginlib/class_list_macros.hpp>

#include "cyclo_teleoperation/robots/ai_worker/ai_worker_groups.hpp"
#include "common/type_define.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
bool ElbowUpLeaderMode::configure(
  rclcpp::Node & node,
  const std::string & prefix,
  const ModeConfiguration & configuration)
{
  configuration_ = configuration;
  const auto * left_group = configuration.findGroup("left");
  const auto * right_group = configuration.findGroup("right");
  if (left_group == nullptr || right_group == nullptr) {
    RCLCPP_ERROR(node.get_logger(), "Elbow-up mode requires left and right control groups");
    return false;
  }
  left_group_ = *left_group;
  right_group_ = *right_group;
  auto parameter = [&node](const std::string & name, const double default_value) {
      if (!node.has_parameter(name)) {
        return node.declare_parameter(name, default_value);
      }
      return node.get_parameter(name).as_double();
    };
  auto string_parameter = [&node](
    const std::string & name, const std::string & default_value)
    {
      if (!node.has_parameter(name)) {
        return node.declare_parameter(name, default_value);
      }
      return node.get_parameter(name).as_string();
    };
  kp_position_ = parameter(prefix + ".kp_position", 50.0);
  kp_orientation_ = parameter(prefix + ".kp_orientation", 50.0);
  weight_position_ = parameter(prefix + ".weight_position", 10.0);
  weight_orientation_ = parameter(prefix + ".weight_orientation", 1.0);
  elbow_up_velocity_ = parameter(prefix + ".elbow_up_velocity", 0.2);
  elbow_weight_ = parameter(prefix + ".elbow_weight", 1.0);
  nullspace_damping_ = parameter(prefix + ".nullspace_damping", 0.001);
  elbow_up_joint_velocity_ = parameter(prefix + ".elbow_up_joint_velocity", 1.0);
  left_elbow_link_ = string_parameter("follower_left_elbow", "arm_l_link4");
  right_elbow_link_ = string_parameter("follower_right_elbow", "arm_r_link4");
  if (
    !configuration_.follower_kinematics->hasLinkFrame(left_elbow_link_) ||
    !configuration_.follower_kinematics->hasLinkFrame(right_elbow_link_))
  {
    RCLCPP_ERROR(node.get_logger(), "An elbow-up link does not exist in the follower model");
    return false;
  }
  return kp_position_ > 0.0 && kp_orientation_ > 0.0 &&
         weight_position_ > 0.0 && weight_orientation_ > 0.0;
}

bool ElbowUpLeaderMode::activate(const ModeContext & context)
{
  configuration_.leader_kinematics->updateState(
    context.leader_position, Eigen::VectorXd::Zero(context.leader_position.size()));
  configuration_.follower_kinematics->updateState(
    context.follower_position, context.follower_velocity);
  left_anchor_valid_ = false;
  right_anchor_valid_ = false;
  captureAnchor(context.enabled_groups);
  return true;
}

void ElbowUpLeaderMode::onGroupsEnabled(
  const ControlGroupMask groups, const ModeContext & context)
{
  configuration_.leader_kinematics->updateState(
    context.leader_position, Eigen::VectorXd::Zero(context.leader_position.size()));
  configuration_.follower_kinematics->updateState(
    context.follower_position, context.follower_velocity);
  captureAnchor(groups);
}

void ElbowUpLeaderMode::captureAnchor(const ControlGroupMask groups)
{
  if ((groups & kLeftGroup) != 0) {
    left_leader_anchor_ =
      configuration_.leader_kinematics->getPose(left_group_.leader_eef);
    left_follower_anchor_ =
      configuration_.follower_kinematics->getPose(left_group_.follower_eef);
    left_anchor_valid_ = true;
  }
  if ((groups & kRightGroup) != 0) {
    right_leader_anchor_ =
      configuration_.leader_kinematics->getPose(right_group_.leader_eef);
    right_follower_anchor_ =
      configuration_.follower_kinematics->getPose(right_group_.follower_eef);
    right_anchor_valid_ = true;
  }
}

Eigen::Matrix<double, 6, 1> ElbowUpLeaderMode::desiredVelocity(
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

Eigen::VectorXd ElbowUpLeaderMode::elbowPreference(
  const ControlGroupMask enabled_groups) const
{
  const int dof = configuration_.follower_kinematics->getDof();
  std::vector<Eigen::MatrixXd> eef_jacobians;
  if ((enabled_groups & kLeftGroup) != 0) {
    eef_jacobians.push_back(
      configuration_.follower_kinematics->getJacobian(left_group_.follower_eef));
  }
  if ((enabled_groups & kRightGroup) != 0) {
    eef_jacobians.push_back(
      configuration_.follower_kinematics->getJacobian(right_group_.follower_eef));
  }
  if (eef_jacobians.empty()) {
    return Eigen::VectorXd::Zero(dof);
  }

  Eigen::MatrixXd eef(6 * eef_jacobians.size(), dof);
  for (size_t i = 0; i < eef_jacobians.size(); ++i) {
    eef.middleRows(6 * i, 6) = eef_jacobians[i];
  }
  const double damping_squared = nullspace_damping_ * nullspace_damping_;
  const Eigen::MatrixXd regularized =
    eef * eef.transpose() +
    damping_squared * Eigen::MatrixXd::Identity(eef.rows(), eef.rows());
  const Eigen::MatrixXd nullspace =
    Eigen::MatrixXd::Identity(dof, dof) -
    eef.transpose() * regularized.ldlt().solve(eef);

  auto preference = [&](const std::string & elbow_link) -> Eigen::VectorXd {
      const Eigen::MatrixXd elbow =
        configuration_.follower_kinematics->getJacobian(elbow_link);
      const Eigen::VectorXd direction = nullspace * elbow.row(2).transpose();
      const double attainable = elbow.row(2).dot(direction);
      if (attainable <= 1e-8 || elbow_up_velocity_ <= 0.0) {
        return Eigen::VectorXd::Zero(dof);
      }
      Eigen::VectorXd result = direction * (elbow_up_velocity_ / attainable);
      if (
        elbow_up_joint_velocity_ > 0.0 &&
        result.norm() > elbow_up_joint_velocity_)
      {
        result *= elbow_up_joint_velocity_ / result.norm();
      }
      return result;
    };

  Eigen::VectorXd result = Eigen::VectorXd::Zero(dof);
  if ((enabled_groups & kLeftGroup) != 0) {
    result += preference(left_elbow_link_);
  }
  if ((enabled_groups & kRightGroup) != 0) {
    result += preference(right_elbow_link_);
  }
  return result;
}

bool ElbowUpLeaderMode::update(const ModeContext & context, ModeOutput & output)
{
  if (
    ((context.enabled_groups & kLeftGroup) != 0 && !left_anchor_valid_) ||
    ((context.enabled_groups & kRightGroup) != 0 && !right_anchor_valid_))
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

  if ((context.enabled_groups & kLeftGroup) != 0) {
    add_task(
      left_group_.follower_eef, left_group_.leader_eef,
      left_follower_anchor_, left_leader_anchor_);
  }
  if ((context.enabled_groups & kRightGroup) != 0) {
    add_task(
      right_group_.follower_eef, right_group_.leader_eef,
      right_follower_anchor_, right_leader_anchor_);
  }
  output.preferred_joint_velocity = elbowPreference(context.enabled_groups);
  output.preferred_joint_velocity_weight = elbow_weight_;
  return true;
}
}  // namespace cyclo_teleoperation::robots::ai_worker

PLUGINLIB_EXPORT_CLASS(
  cyclo_teleoperation::robots::ai_worker::ElbowUpLeaderMode,
  cyclo_teleoperation::TeleoperationMode)
