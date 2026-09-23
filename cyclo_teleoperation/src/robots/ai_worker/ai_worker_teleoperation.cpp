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

#include "cyclo_teleoperation/robots/ai_worker/ai_worker_teleoperation.hpp"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <utility>

#include <pluginlib/class_list_macros.hpp>

namespace cyclo_teleoperation::robots::ai_worker
{
std::string AIWorkerTeleoperation::parameterName(const std::string & name) const
{
  return parameter_prefix_.empty() ? name : parameter_prefix_ + "." + name;
}

bool AIWorkerTeleoperation::configure(
  rclcpp::Node & node,
  const std::string & parameter_prefix,
  ControlInterface::RequestCallback request_callback)
{
  node_ = &node;
  parameter_prefix_ = parameter_prefix;

  auto declare_string = [this](const std::string & name, const std::string & value) {
      const auto full_name = parameterName(name);
      if (!node_->has_parameter(full_name)) {
        node_->declare_parameter(full_name, value);
      }
    };
  declare_string("follower_urdf_path", "");
  declare_string("follower_srdf_path", "");
  declare_string("leader_urdf_path", "");
  declare_string("leader_urdf_xml", "");
  declare_string("leader_srdf_path", "");
  declare_string("follower_joint_states_topic", "/joint_states");
  declare_string(
    "right_leader_topic",
    "/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory");
  declare_string(
    "left_leader_topic",
    "/leader/joint_trajectory_command_broadcaster_left/raw_joint_trajectory");
  declare_string(
    "right_command_topic",
    "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory");
  declare_string(
    "left_command_topic",
    "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory");
  declare_string("right_gripper_joint", "gripper_r_joint1");
  declare_string("left_gripper_joint", "gripper_l_joint1");
  declare_string("follower_right_eef", "arm_r_link7");
  declare_string("follower_left_eef", "arm_l_link7");
  declare_string("follower_base_frame", "base_link");
  declare_string("follower_right_eef_pose_topic", "~/follower/right/eef_pose");
  declare_string("follower_left_eef_pose_topic", "~/follower/left/eef_pose");
  declare_string(
    "follower_right_eef_reference_topic", "~/follower/right/reference_eef_pose");
  declare_string(
    "follower_left_eef_reference_topic", "~/follower/left/reference_eef_pose");
  declare_string("leader_right_eef", "arm_r_link7");
  declare_string("leader_left_eef", "arm_l_link7");
  const auto enable_leader_interface_parameter = parameterName("enable_leader_interface");
  if (!node_->has_parameter(enable_leader_interface_parameter)) {
    node_->declare_parameter(enable_leader_interface_parameter, true);
  }
  const auto publish_follower_eef_state_parameter =
    parameterName("publish_follower_eef_state");
  if (!node_->has_parameter(publish_follower_eef_state_parameter)) {
    node_->declare_parameter(publish_follower_eef_state_parameter, true);
  }
  const auto publish_eef_pose_references_parameter =
    parameterName("publish_eef_pose_references");
  if (!node_->has_parameter(publish_eef_pose_references_parameter)) {
    node_->declare_parameter(publish_eef_pose_references_parameter, true);
  }
  enable_leader_interface_ =
    node_->get_parameter(enable_leader_interface_parameter).as_bool();
  publish_follower_eef_state_ =
    node_->get_parameter(publish_follower_eef_state_parameter).as_bool();
  publish_eef_pose_references_ =
    node_->get_parameter(publish_eef_pose_references_parameter).as_bool();

  follower_joint_states_topic_ =
    node_->get_parameter(parameterName("follower_joint_states_topic")).as_string();
  leader_input_channels_ = {
    LeaderInputChannel{
      kLeftGroupId, node_->get_parameter(parameterName("left_leader_topic")).as_string()},
    LeaderInputChannel{
      kRightGroupId, node_->get_parameter(parameterName("right_leader_topic")).as_string()}};
  if (!initialize()) {
    return false;
  }
  if (!enable_leader_interface_) {
    return true;
  }
  return control_interface_.configure(
    node, parameter_prefix, mode_configuration_.control_groups,
    std::move(request_callback));
}

AIWorkerTeleoperation::~AIWorkerTeleoperation()
{
  if (!temporary_leader_urdf_path_.empty()) {
    std::error_code error;
    std::filesystem::remove(temporary_leader_urdf_path_, error);
  }
}

bool AIWorkerTeleoperation::initialize()
{
  const auto follower_urdf =
    node_->get_parameter(parameterName("follower_urdf_path")).as_string();
  const auto follower_srdf =
    node_->get_parameter(parameterName("follower_srdf_path")).as_string();
  auto leader_urdf = node_->get_parameter(parameterName("leader_urdf_path")).as_string();
  const auto leader_urdf_xml =
    node_->get_parameter(parameterName("leader_urdf_xml")).as_string();
  const auto leader_srdf =
    node_->get_parameter(parameterName("leader_srdf_path")).as_string();
  if (!leader_urdf_xml.empty()) {
    temporary_leader_urdf_path_ =
      (std::filesystem::temp_directory_path() /
      ("cyclo_teleoperation_leader_" + std::to_string(getpid()) + ".urdf")).string();
    std::ofstream output(temporary_leader_urdf_path_, std::ios::trunc);
    if (!output) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create temporary leader URDF");
      return false;
    }
    output << leader_urdf_xml;
    output.close();
    if (!output) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to write temporary leader URDF");
      return false;
    }
    leader_urdf = temporary_leader_urdf_path_;
  }
  if (follower_urdf.empty() || (enable_leader_interface_ && leader_urdf.empty())) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "Follower URDF path and either leader URDF path or XML are required");
    return false;
  }

  follower_kinematics_ =
    std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(
    follower_urdf, follower_srdf);

  follower_joint_names_ = follower_kinematics_->getJointNames();
  for (size_t i = 0; i < follower_joint_names_.size(); ++i) {
    follower_index_[follower_joint_names_[i]] = static_cast<int>(i);
  }
  if (enable_leader_interface_) {
    leader_kinematics_ =
      std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(
      leader_urdf, leader_srdf);
    leader_joint_names_ = leader_kinematics_->getJointNames();
    for (size_t i = 0; i < leader_joint_names_.size(); ++i) {
      leader_index_[leader_joint_names_[i]] = static_cast<int>(i);
    }
  } else {
    leader_joint_names_ = follower_joint_names_;
    leader_index_ = follower_index_;
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
      node_->get_logger(), "AI Worker teleoperation requires 7 left and 7 right arm joints");
    return false;
  }

  const int follower_dof = follower_kinematics_->getDof();
  follower_position_.setZero(follower_dof);
  follower_velocity_.setZero(follower_dof);
  leader_reference_.setZero(follower_dof);
  leader_position_.setZero(
    enable_leader_interface_ ? leader_kinematics_->getDof() : follower_dof);

  right_gripper_joint_ =
    node_->get_parameter(parameterName("right_gripper_joint")).as_string();
  left_gripper_joint_ =
    node_->get_parameter(parameterName("left_gripper_joint")).as_string();
  follower_base_frame_ =
    node_->get_parameter(parameterName("follower_base_frame")).as_string();

  mode_configuration_.follower_kinematics = follower_kinematics_;
  mode_configuration_.leader_kinematics = leader_kinematics_;
  mode_configuration_.control_groups = {
    ControlGroupConfiguration{
      kLeftGroupId, "left", left_arm_indices_,
      node_->get_parameter(parameterName("follower_left_eef")).as_string(),
      node_->get_parameter(parameterName("leader_left_eef")).as_string(),
      {{left_gripper_joint_, "gripper_position"}}},
    ControlGroupConfiguration{
      kRightGroupId, "right", right_arm_indices_,
      node_->get_parameter(parameterName("follower_right_eef")).as_string(),
      node_->get_parameter(parameterName("leader_right_eef")).as_string(),
      {{right_gripper_joint_, "gripper_position"}}}};
  control_group_states_.assign(mode_configuration_.control_groups.size(), ControlGroupState{});
  follower_auxiliary_position_.resize(mode_configuration_.control_groups.size());
  leader_auxiliary_reference_.resize(mode_configuration_.control_groups.size());
  for (const auto & group : mode_configuration_.control_groups) {
    follower_auxiliary_position_[group.id] =
      Eigen::VectorXd::Zero(group.auxiliary_joints.size());
    leader_auxiliary_reference_[group.id] =
      Eigen::VectorXd::Zero(group.auxiliary_joints.size());
  }

  const auto command_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
  right_publisher_ =
    node_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
    node_->get_parameter(parameterName("right_command_topic")).as_string(), command_qos);
  left_publisher_ =
    node_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
    node_->get_parameter(parameterName("left_command_topic")).as_string(), command_qos);
  if (publish_follower_eef_state_) {
    right_eef_pose_publisher_ =
      node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      node_->get_parameter(parameterName("follower_right_eef_pose_topic")).as_string(),
      rclcpp::SensorDataQoS().keep_last(1));
    left_eef_pose_publisher_ =
      node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      node_->get_parameter(parameterName("follower_left_eef_pose_topic")).as_string(),
      rclcpp::SensorDataQoS().keep_last(1));
  }
  if (publish_eef_pose_references_) {
    right_eef_reference_publisher_ =
      node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      node_->get_parameter(parameterName("follower_right_eef_reference_topic")).as_string(),
      rclcpp::SensorDataQoS().keep_last(1));
    left_eef_reference_publisher_ =
      node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      node_->get_parameter(parameterName("follower_left_eef_reference_topic")).as_string(),
      rclcpp::SensorDataQoS().keep_last(1));
  }
  return true;
}

bool AIWorkerTeleoperation::updateFollowerState(const sensor_msgs::msg::JointState & message)
{
  if (
    message.position.size() != message.name.size() ||
    (!message.velocity.empty() && message.velocity.size() != message.name.size()))
  {
    return false;
  }

  std::unordered_map<std::string, size_t> message_index;
  for (size_t i = 0; i < message.name.size(); ++i) {
    if (!message_index.emplace(message.name[i], i).second) {
      return false;
    }
  }

  Eigen::VectorXd follower_position = follower_position_;
  Eigen::VectorXd follower_velocity = follower_velocity_;
  GroupAuxiliaryPositions follower_auxiliary_position = follower_auxiliary_position_;
  for (size_t i = 0; i < follower_joint_names_.size(); ++i) {
    const auto iter = message_index.find(follower_joint_names_[i]);
    if (iter == message_index.end()) {
      return false;
    }
    const size_t source = iter->second;
    if (!std::isfinite(message.position[source])) {
      return false;
    }
    follower_position[i] = message.position[source];
    if (!message.velocity.empty()) {
      if (!std::isfinite(message.velocity[source])) {
        return false;
      }
      follower_velocity[i] = message.velocity[source];
    } else {
      follower_velocity[i] = 0.0;
    }
  }
  const auto left_gripper = message_index.find(left_gripper_joint_);
  const auto right_gripper = message_index.find(right_gripper_joint_);
  if (
    left_gripper == message_index.end() || right_gripper == message_index.end() ||
    !std::isfinite(message.position[left_gripper->second]) ||
    !std::isfinite(message.position[right_gripper->second]))
  {
    return false;
  }
  follower_auxiliary_position[kLeftGroupId][0] = message.position[left_gripper->second];
  follower_auxiliary_position[kRightGroupId][0] = message.position[right_gripper->second];

  follower_position_ = std::move(follower_position);
  follower_velocity_ = std::move(follower_velocity);
  follower_auxiliary_position_ = std::move(follower_auxiliary_position);
  if (publish_follower_eef_state_) {
    publishFollowerEefPoses(message.header);
  }
  return true;
}

void AIWorkerTeleoperation::publishFollowerEefPoses(
  const std_msgs::msg::Header & source_header)
{
  follower_kinematics_->updateState(follower_position_, follower_velocity_);
  auto make_message = [this, &source_header](const Eigen::Affine3d & pose) {
      geometry_msgs::msg::PoseStamped message;
      message.header = source_header;
      if (rclcpp::Time(message.header.stamp).nanoseconds() == 0) {
        message.header.stamp = node_->now();
      }
      message.header.frame_id = follower_base_frame_;
      message.pose.position.x = pose.translation().x();
      message.pose.position.y = pose.translation().y();
      message.pose.position.z = pose.translation().z();
      const Eigen::Quaterniond orientation(pose.linear());
      message.pose.orientation.x = orientation.x();
      message.pose.orientation.y = orientation.y();
      message.pose.orientation.z = orientation.z();
      message.pose.orientation.w = orientation.w();
      return message;
    };
  left_eef_pose_publisher_->publish(make_message(
    follower_kinematics_->getPose(mode_configuration_.control_groups.at(
      kLeftGroupId).follower_eef)));
  right_eef_pose_publisher_->publish(make_message(
    follower_kinematics_->getPose(
      mode_configuration_.control_groups.at(kRightGroupId).follower_eef)));
}

bool AIWorkerTeleoperation::updateLeaderReference(
  const trajectory_msgs::msg::JointTrajectory & message,
  const ControlGroupId target_group)
{
  if (target_group != kLeftGroupId && target_group != kRightGroupId) {
    return false;
  }
  if (message.points.empty()) {
    return false;
  }
  const auto & point = message.points.front();
  if (
    point.positions.empty() ||
    point.positions.size() != message.joint_names.size())
  {
    return false;
  }
  const double duration = rclcpp::Duration(point.time_from_start).seconds();
  if (!std::isfinite(duration) || duration < 0.0) {
    RCLCPP_WARN(node_->get_logger(), "Leader trajectory ignored: time_from_start must be >= 0");
    return false;
  }
  const auto & requested_indices =
    target_group == kLeftGroupId ? left_arm_indices_ : right_arm_indices_;
  const auto & requested_names =
    target_group == kLeftGroupId ? left_arm_names_ : right_arm_names_;
  const auto & requested_gripper =
    target_group == kLeftGroupId ? left_gripper_joint_ : right_gripper_joint_;
  if (message.joint_names.size() != requested_names.size() + 1) {
    return false;
  }

  Eigen::VectorXd leader_reference = leader_reference_;
  Eigen::VectorXd leader_position = leader_position_;
  GroupAuxiliaryPositions leader_auxiliary_reference = leader_auxiliary_reference_;
  std::unordered_set<std::string> received;
  size_t updated_arm_joints = 0;
  bool updated_gripper = false;
  for (size_t i = 0; i < message.joint_names.size(); ++i) {
    const auto & joint_name = message.joint_names[i];
    const double position = point.positions[i];
    if (!received.insert(joint_name).second || !std::isfinite(position)) {
      return false;
    }

    if (joint_name == requested_gripper) {
      leader_auxiliary_reference[target_group][0] = position;
      updated_gripper = true;
      continue;
    }

    if (std::find(requested_names.begin(), requested_names.end(), joint_name) ==
      requested_names.end())
    {
      return false;
    }

    const auto follower = follower_index_.find(joint_name);
    const auto leader = leader_index_.find(joint_name);
    if (follower == follower_index_.end() || leader == leader_index_.end()) {
      return false;
    }
    leader_reference[follower->second] = position;
    leader_position[leader->second] = position;
    ++updated_arm_joints;
  }
  if (updated_arm_joints != requested_indices.size() || !updated_gripper) {
    return false;
  }

  leader_reference_ = std::move(leader_reference);
  leader_position_ = std::move(leader_position);
  leader_auxiliary_reference_ = std::move(leader_auxiliary_reference);
  if (target_group == kLeftGroupId) {
    control_group_states_[kLeftGroupId].leader_duration = duration;
    ++control_group_states_[kLeftGroupId].leader_sequence;
  } else if (target_group == kRightGroupId) {
    control_group_states_[kRightGroupId].leader_duration = duration;
    ++control_group_states_[kRightGroupId].leader_sequence;
  }
  return true;
}

bool AIWorkerTeleoperation::updateGripperReference(
  const trajectory_msgs::msg::JointTrajectory & message,
  const ControlGroupId target_group)
{
  if (
    (target_group != kLeftGroupId && target_group != kRightGroupId) ||
    message.joint_names.size() != 1 || message.points.empty())
  {
    return false;
  }

  const auto & point = message.points.front();
  if (point.positions.size() != 1) {
    return false;
  }
  const double duration = rclcpp::Duration(point.time_from_start).seconds();
  const std::string & expected_gripper =
    target_group == kLeftGroupId ? left_gripper_joint_ : right_gripper_joint_;
  if (
    message.joint_names.front() != expected_gripper ||
    !std::isfinite(point.positions.front()) ||
    !std::isfinite(duration) || duration < 0.0)
  {
    return false;
  }
  if (
    target_group >= leader_auxiliary_reference_.size() ||
    leader_auxiliary_reference_[target_group].size() != 1)
  {
    return false;
  }

  GroupAuxiliaryPositions gripper_reference = leader_auxiliary_reference_;
  gripper_reference[target_group][0] = point.positions.front();
  leader_auxiliary_reference_ = std::move(gripper_reference);
  return true;
}

trajectory_msgs::msg::JointTrajectory AIWorkerTeleoperation::makeArmTrajectory(
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
  point.time_from_start = rclcpp::Duration(0, 0);
  message.points.push_back(std::move(point));
  return message;
}

void AIWorkerTeleoperation::publish(
  const Eigen::VectorXd & command,
  const GroupAuxiliaryPositions & auxiliary_command)
{
  left_publisher_->publish(makeArmTrajectory(
    left_arm_indices_, left_arm_names_, command,
    left_gripper_joint_, auxiliary_command.at(kLeftGroupId)[0]));
  right_publisher_->publish(makeArmTrajectory(
    right_arm_indices_, right_arm_names_, command,
    right_gripper_joint_, auxiliary_command.at(kRightGroupId)[0]));
}

void AIWorkerTeleoperation::publishEefPoseReferences(
  const Eigen::VectorXd & command,
  const std::vector<EefPoseReference> & references)
{
  if (!publish_eef_pose_references_ || command.size() != follower_position_.size()) {
    return;
  }

  std::unordered_map<ControlGroupId, Eigen::Affine3d> explicit_references;
  for (const auto & reference : references) {
    if (reference.pose.matrix().allFinite()) {
      explicit_references[reference.group_id] = reference.pose;
    }
  }

  follower_kinematics_->updateState(
    command, Eigen::VectorXd::Zero(command.size()));
  for (const auto & group : mode_configuration_.control_groups) {
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher;
    if (group.id == kLeftGroupId) {
      publisher = left_eef_reference_publisher_;
    } else if (group.id == kRightGroupId) {
      publisher = right_eef_reference_publisher_;
    } else {
      continue;
    }
    if (!publisher) {
      continue;
    }
    const auto reference = explicit_references.find(group.id);
    const Eigen::Affine3d pose = reference == explicit_references.end() ?
      follower_kinematics_->getPose(group.follower_eef) : reference->second;
    if (!pose.matrix().allFinite()) {
      continue;
    }

    geometry_msgs::msg::PoseStamped message;
    message.header.stamp = node_->now();
    message.header.frame_id = follower_base_frame_;
    message.pose.position.x = pose.translation().x();
    message.pose.position.y = pose.translation().y();
    message.pose.position.z = pose.translation().z();
    const Eigen::Quaterniond orientation(pose.linear());
    message.pose.orientation.x = orientation.x();
    message.pose.orientation.y = orientation.y();
    message.pose.orientation.z = orientation.z();
    message.pose.orientation.w = orientation.w();
    publisher->publish(message);
  }
}

void AIWorkerTeleoperation::publishStatus(const ControlStatus & status)
{
  control_interface_.publishStatus(status);
}
}  // namespace cyclo_teleoperation::robots::ai_worker

PLUGINLIB_EXPORT_CLASS(
  cyclo_teleoperation::robots::ai_worker::AIWorkerTeleoperation,
  cyclo_teleoperation::RobotTeleoperation)
