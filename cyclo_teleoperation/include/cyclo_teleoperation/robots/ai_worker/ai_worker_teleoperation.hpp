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

#pragma once

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "cyclo_teleoperation/core/robot_teleoperation.hpp"
#include "cyclo_teleoperation/robots/ai_worker/ai_worker_groups.hpp"
#include "cyclo_teleoperation/robots/ai_worker/ai_worker_control_interface.hpp"
#include "kinematics/kinematics_solver.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
class AIWorkerTeleoperation : public RobotTeleoperation
{
public:
  AIWorkerTeleoperation() = default;
  ~AIWorkerTeleoperation() override;

  bool configure(
    rclcpp::Node & node,
    const std::string & parameter_prefix,
    ControlInterface::RequestCallback request_callback) override;

  std::string robotName() const override {return "ai_worker";}
  int dof() const override {return static_cast<int>(follower_position_.size());}

  const Eigen::VectorXd & followerPosition() const override {return follower_position_;}
  const Eigen::VectorXd & followerVelocity() const override {return follower_velocity_;}
  const Eigen::VectorXd & leaderReference() const override {return leader_reference_;}
  const Eigen::VectorXd & leaderPosition() const override {return leader_position_;}
  const GroupAuxiliaryPositions & followerAuxiliaryPosition() const override
  {
    return follower_auxiliary_position_;
  }
  const GroupAuxiliaryPositions & leaderAuxiliaryReference() const override
  {
    return leader_auxiliary_reference_;
  }
  const std::vector<ControlGroupState> & controlGroupStates() const override
  {
    return control_group_states_;
  }
  const ModeConfiguration & modeConfiguration() const override {return mode_configuration_;}

  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver>
  followerKinematics() const override {return follower_kinematics_;}
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver>
  leaderKinematics() const override {return leader_kinematics_;}

  std::string followerJointStatesTopic() const override
  {
    return follower_joint_states_topic_;
  }
  const std::vector<LeaderInputChannel> & leaderInputChannels() const override
  {
    return leader_input_channels_;
  }

  bool updateFollowerState(const sensor_msgs::msg::JointState & message) override;
  bool updateLeaderReference(
    const trajectory_msgs::msg::JointTrajectory & message,
    ControlGroupId target_group) override;
  void publish(
    const Eigen::VectorXd & command,
    const GroupAuxiliaryPositions & auxiliary_command) override;
  void publishStatus(const ControlStatus & status) override;

private:
  bool initialize();
  std::string parameterName(const std::string & name) const;
  trajectory_msgs::msg::JointTrajectory makeArmTrajectory(
    const std::vector<int> & indices,
    const std::vector<std::string> & names,
    const Eigen::VectorXd & command,
    const std::string & gripper_name,
    double gripper_position) const;

  rclcpp::Node * node_ = nullptr;
  std::string parameter_prefix_;
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> follower_kinematics_;
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> leader_kinematics_;
  ModeConfiguration mode_configuration_;

  Eigen::VectorXd follower_position_;
  Eigen::VectorXd follower_velocity_;
  Eigen::VectorXd leader_reference_;
  Eigen::VectorXd leader_position_;
  std::vector<std::string> follower_joint_names_;
  std::vector<std::string> leader_joint_names_;
  std::unordered_map<std::string, int> follower_index_;
  std::unordered_map<std::string, int> leader_index_;
  std::vector<int> left_arm_indices_;
  std::vector<int> right_arm_indices_;
  std::vector<std::string> left_arm_names_;
  std::vector<std::string> right_arm_names_;
  std::vector<ControlGroupState> control_group_states_;
  std::string follower_joint_states_topic_;
  std::vector<LeaderInputChannel> leader_input_channels_;

  std::string right_gripper_joint_;
  std::string left_gripper_joint_;
  std::string temporary_leader_urdf_path_;
  GroupAuxiliaryPositions follower_auxiliary_position_;
  GroupAuxiliaryPositions leader_auxiliary_reference_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr right_publisher_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr left_publisher_;
  AIWorkerControlInterface control_interface_;
};
}  // namespace cyclo_teleoperation::robots::ai_worker
