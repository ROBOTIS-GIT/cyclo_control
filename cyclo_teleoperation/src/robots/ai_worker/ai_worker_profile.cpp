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

#include "cyclo_teleoperation/robots/ai_worker/ai_worker_profile.hpp"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <utility>

namespace cyclo_teleoperation::robots::ai_worker
{
AIWorkerProfile::AIWorkerProfile(rclcpp::Node & node)
: node_(node)
{
}

AIWorkerProfile::~AIWorkerProfile()
{
  if (!temporary_leader_urdf_path_.empty()) {
    std::error_code error;
    std::filesystem::remove(temporary_leader_urdf_path_, error);
  }
}

bool AIWorkerProfile::initialize()
{
  const auto follower_urdf = node_.get_parameter("follower_urdf_path").as_string();
  const auto follower_srdf = node_.get_parameter("follower_srdf_path").as_string();
  auto leader_urdf = node_.get_parameter("leader_urdf_path").as_string();
  const auto leader_urdf_xml = node_.get_parameter("leader_urdf_xml").as_string();
  const auto leader_srdf = node_.get_parameter("leader_srdf_path").as_string();
  if (!leader_urdf_xml.empty()) {
    temporary_leader_urdf_path_ =
      (std::filesystem::temp_directory_path() /
      ("cyclo_teleoperation_leader_" + std::to_string(getpid()) + ".urdf")).string();
    std::ofstream output(temporary_leader_urdf_path_, std::ios::trunc);
    if (!output) {
      RCLCPP_ERROR(node_.get_logger(), "Failed to create temporary leader URDF");
      return false;
    }
    output << leader_urdf_xml;
    output.close();
    if (!output) {
      RCLCPP_ERROR(node_.get_logger(), "Failed to write temporary leader URDF");
      return false;
    }
    leader_urdf = temporary_leader_urdf_path_;
  }
  if (follower_urdf.empty() || leader_urdf.empty()) {
    RCLCPP_ERROR(
      node_.get_logger(),
      "Follower URDF path and either leader URDF path or XML are required");
    return false;
  }

  follower_kinematics_ =
    std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(
    follower_urdf, follower_srdf);
  leader_kinematics_ =
    std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(
    leader_urdf, leader_srdf);

  follower_joint_names_ = follower_kinematics_->getJointNames();
  leader_joint_names_ = leader_kinematics_->getJointNames();
  for (size_t i = 0; i < follower_joint_names_.size(); ++i) {
    follower_index_[follower_joint_names_[i]] = static_cast<int>(i);
  }
  for (size_t i = 0; i < leader_joint_names_.size(); ++i) {
    leader_index_[leader_joint_names_[i]] = static_cast<int>(i);
  }

  for (const auto & name : follower_joint_names_) {
    const int index = follower_index_.at(name);
    if (name.find("arm_l_joint") != std::string::npos) {
      left_arm_names_.push_back(name);
      left_arm_indices_.push_back(index);
    } else if (name.find("arm_r_joint") != std::string::npos) {
      right_arm_names_.push_back(name);
      right_arm_indices_.push_back(index);
    }
  }
  if (left_arm_indices_.size() != 7 || right_arm_indices_.size() != 7) {
    RCLCPP_ERROR(
      node_.get_logger(), "AI Worker profile requires 7 left and 7 right arm joints");
    return false;
  }

  const int follower_dof = follower_kinematics_->getDof();
  follower_position_.setZero(follower_dof);
  follower_velocity_.setZero(follower_dof);
  leader_reference_.setZero(follower_dof);
  leader_position_.setZero(leader_kinematics_->getDof());

  right_gripper_joint_ = node_.get_parameter("right_gripper_joint").as_string();
  left_gripper_joint_ = node_.get_parameter("left_gripper_joint").as_string();

  mode_configuration_.follower_kinematics = follower_kinematics_;
  mode_configuration_.leader_kinematics = leader_kinematics_;
  mode_configuration_.left_arm_indices = left_arm_indices_;
  mode_configuration_.right_arm_indices = right_arm_indices_;
  mode_configuration_.follower_left_eef =
    node_.get_parameter("follower_left_eef").as_string();
  mode_configuration_.follower_right_eef =
    node_.get_parameter("follower_right_eef").as_string();
  mode_configuration_.follower_left_elbow =
    node_.get_parameter("follower_left_elbow").as_string();
  mode_configuration_.follower_right_elbow =
    node_.get_parameter("follower_right_elbow").as_string();
  mode_configuration_.leader_left_eef =
    node_.get_parameter("leader_left_eef").as_string();
  mode_configuration_.leader_right_eef =
    node_.get_parameter("leader_right_eef").as_string();

  right_publisher_ =
    node_.create_publisher<trajectory_msgs::msg::JointTrajectory>(
    node_.get_parameter("right_command_topic").as_string(), 10);
  left_publisher_ =
    node_.create_publisher<trajectory_msgs::msg::JointTrajectory>(
    node_.get_parameter("left_command_topic").as_string(), 10);
  return true;
}

bool AIWorkerProfile::updateFollowerState(const sensor_msgs::msg::JointState & message)
{
  std::unordered_map<std::string, size_t> message_index;
  for (size_t i = 0; i < message.name.size(); ++i) {
    message_index[message.name[i]] = i;
  }

  size_t joint_count = 0;
  for (size_t i = 0; i < follower_joint_names_.size(); ++i) {
    const auto iter = message_index.find(follower_joint_names_[i]);
    if (iter == message_index.end()) {
      continue;
    }
    const size_t source = iter->second;
    if (source < message.position.size()) {
      follower_position_[i] = message.position[source];
      ++joint_count;
    }
    follower_velocity_[i] =
      source < message.velocity.size() ? message.velocity[source] : 0.0;
  }
  return joint_count == follower_joint_names_.size();
}

bool AIWorkerProfile::updateLeaderReference(
  const trajectory_msgs::msg::JointTrajectory & message,
  const uint8_t target_arm)
{
  if (message.points.empty() || message.points.front().positions.empty()) {
    return false;
  }
  const auto & point = message.points.front();
  const double duration = rclcpp::Duration(point.time_from_start).seconds();
  if (duration < 0.0) {
    RCLCPP_WARN(node_.get_logger(), "Leader trajectory ignored: time_from_start must be >= 0");
    return false;
  }
  const auto & requested_indices =
    target_arm == kLeftArm ? left_arm_indices_ : right_arm_indices_;
  std::vector<bool> received(follower_joint_names_.size(), false);
  size_t updated_arm_joints = 0;
  for (size_t i = 0; i < message.joint_names.size() && i < point.positions.size(); ++i) {
    const auto follower = follower_index_.find(message.joint_names[i]);
    if (follower != follower_index_.end()) {
      const int index = follower->second;
      leader_reference_[index] = point.positions[i];
      if (
        !received[index] &&
        std::find(requested_indices.begin(), requested_indices.end(), index) !=
        requested_indices.end())
      {
        received[index] = true;
        ++updated_arm_joints;
      }
    }
    const auto leader = leader_index_.find(message.joint_names[i]);
    if (leader != leader_index_.end()) {
      leader_position_[leader->second] = point.positions[i];
    }
    if (message.joint_names[i] == right_gripper_joint_ && target_arm == kRightArm) {
      right_gripper_position_ = point.positions[i];
    }
    if (message.joint_names[i] == left_gripper_joint_ && target_arm == kLeftArm) {
      left_gripper_position_ = point.positions[i];
    }
  }
  if (updated_arm_joints != requested_indices.size()) {
    return false;
  }
  if (target_arm == kLeftArm) {
    left_leader_duration_ = duration;
    ++left_leader_sequence_;
  } else if (target_arm == kRightArm) {
    right_leader_duration_ = duration;
    ++right_leader_sequence_;
  }
  return true;
}

trajectory_msgs::msg::JointTrajectory AIWorkerProfile::makeArmTrajectory(
  const std::vector<int> & indices,
  const std::vector<std::string> & names,
  const Eigen::VectorXd & command,
  const std::string & gripper_name,
  const double gripper_position) const
{
  trajectory_msgs::msg::JointTrajectory message;
  message.header.stamp = rclcpp::Time(0, 0);
  message.joint_names = names;
  message.joint_names.push_back(gripper_name);
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions.reserve(indices.size() + 1);
  for (const int index : indices) {
    point.positions.push_back(command[index]);
  }
  point.positions.push_back(gripper_position);
  point.time_from_start =
    rclcpp::Duration::from_seconds(node_.get_parameter("trajectory_time").as_double());
  message.points.push_back(std::move(point));
  return message;
}

void AIWorkerProfile::publish(const Eigen::VectorXd & command)
{
  left_publisher_->publish(makeArmTrajectory(
    left_arm_indices_, left_arm_names_, command,
    left_gripper_joint_, left_gripper_position_));
  right_publisher_->publish(makeArmTrajectory(
    right_arm_indices_, right_arm_names_, command,
    right_gripper_joint_, right_gripper_position_));
}
}  // namespace cyclo_teleoperation::robots::ai_worker
