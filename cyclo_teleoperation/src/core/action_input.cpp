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
#include "cyclo_teleoperation/core/action_input.hpp"
#include <algorithm>
#include <cmath>
#include <utility>
#include <stdexcept>

namespace cyclo_teleoperation
{
ActionInput::ActionInput(
  rclcpp::Node & node, RobotTeleoperation & robot,
  std::function<bool()> enabled, std::function<std::string()> reference_type)
: node_(node), robot_(robot), enabled_(std::move(enabled)),
  reference_type_(std::move(reference_type))
{
  frame_ = node.declare_parameter("action_reference_frame", "base_link");
  timeout_ = node.declare_parameter("action_timeout", 0.5);
  if (!std::isfinite(timeout_) || timeout_ <= 0.0) {
    throw std::runtime_error("action_timeout must be finite and positive");
  }
  size_t size = 0;
  for (const auto & group : robot.modeConfiguration().control_groups) {
    size = std::max(size, static_cast<size_t>(group.id) + 1);
  }
  poses_.resize(size);
  joint_received_.resize(size); pose_received_.resize(size); gripper_received_.resize(size);
  joint_times_.assign(size, node.now()); pose_times_.assign(size, node.now());
  joint_stamps_.assign(size, rclcpp::Time(0, 0, node.get_clock()->get_clock_type()));
  pose_stamps_ = joint_stamps_;
  const auto qos = rclcpp::QoS(1).reliable();
  for (const auto & group : robot.modeConfiguration().control_groups) {
    const auto id = group.id;
    const std::string prefix = "commands." + group.name;
    const auto joint_topic = node.declare_parameter(
      prefix + ".joint_topic", "/command/" + group.name + "/joint");
    const auto pose_topic = node.declare_parameter(
      prefix + ".pose_topic", "/command/" + group.name + "/pose");
    joints_.push_back(node.create_subscription<trajectory_msgs::msg::JointTrajectory>(
      joint_topic, qos, [this, id](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
          if (!enabled_()) {return;}
          rclcpp::Time stamp(0, 0, node_.get_clock()->get_clock_type());
          if (!validStamp(msg->header.stamp, joint_stamps_[id], stamp)) {return;}
          try {
            if (reference_type_() == "absolute_eef_pose") {
              if (!robot_.updateGripperReference(*msg, id)) {return;}
            } else {
              if (!robot_.updateLeaderReference(*msg, id)) {return;}
              joint_received_[id] = true;
              joint_times_[id] = stamp.nanoseconds() == 0 ? node_.now() : stamp;
            }
          } catch (const std::exception & error) {
            RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 2000,
              "Invalid joint command ignored: %s", error.what());
            return;
          }
          if (stamp.nanoseconds() != 0) {joint_stamps_[id] = stamp;}
          gripper_received_[id] = true;
      }));
    pose_inputs_.push_back(node.create_subscription<geometry_msgs::msg::PoseStamped>(
      pose_topic, qos, [this, id](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          if (!enabled_() || reference_type_() != "absolute_eef_pose" ||
          msg->header.frame_id != frame_) {return;}
          rclcpp::Time stamp(0, 0, node_.get_clock()->get_clock_type());
          if (!validStamp(msg->header.stamp, pose_stamps_[id], stamp)) {return;}
          const auto & p = msg->pose.position;
          const auto & q = msg->pose.orientation;
          Eigen::Vector3d position(p.x, p.y, p.z);
          Eigen::Quaterniond rotation(q.w, q.x, q.y, q.z);
          if (!position.allFinite() || !rotation.coeffs().allFinite() ||
          !std::isfinite(rotation.norm()) || rotation.norm() < 1e-9)
          {
            return;
          }
          poses_[id].pose.translation() = position;
          poses_[id].pose.linear() = rotation.normalized().toRotationMatrix();
          poses_[id].valid = true;
          ++poses_[id].sequence;
          pose_received_[id] = true;
          pose_times_[id] = stamp.nanoseconds() == 0 ? node_.now() : stamp;
          if (stamp.nanoseconds() != 0) {pose_stamps_[id] = stamp;}
      }));
  }
  clear();
}

bool ActionInput::validStamp(
  const builtin_interfaces::msg::Time & message, const rclcpp::Time & previous,
  rclcpp::Time & stamp) const
{
  if (message.sec < 0 || message.nanosec >= 1000000000u) {return false;}
  stamp = rclcpp::Time(message, node_.get_clock()->get_clock_type());
  // Zero means immediate, for compatibility with existing JointTrajectory publishers.
  if (stamp.nanoseconds() == 0) {return true;}
  const double age = (node_.now() - stamp).seconds();
  return stamp >= reset_time_ && stamp >= previous && age >= 0.0 && age <= timeout_;
}

void ActionInput::clear()
{
  std::fill(joint_received_.begin(), joint_received_.end(), false);
  std::fill(pose_received_.begin(), pose_received_.end(), false);
  std::fill(gripper_received_.begin(), gripper_received_.end(), false);
  for (auto & pose : poses_) {
    pose.valid = false;
  }
  reset_time_ = node_.now();
  std::fill(joint_stamps_.begin(), joint_stamps_.end(), reset_time_);
  std::fill(pose_stamps_.begin(), pose_stamps_.end(), reset_time_);
}

ControlGroupMask ActionInput::freshGroups() const
{
  const bool pose = reference_type_() == "absolute_eef_pose";
  ControlGroupMask result = 0;
  for (const auto & group : robot_.modeConfiguration().control_groups) {
    const auto id = group.id;
    const double age = (node_.now() - (pose ? pose_times_[id] : joint_times_[id])).seconds();
    if ((pose ? pose_received_[id] : joint_received_[id]) && age >= 0.0 && age <= timeout_) {
      result |= controlGroupBit(id);
    }
  }
  return result;
}
}  // namespace cyclo_teleoperation
