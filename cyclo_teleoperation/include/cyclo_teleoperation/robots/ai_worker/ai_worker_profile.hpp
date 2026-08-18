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

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "cyclo_teleoperation/core/types.hpp"
#include "kinematics/kinematics_solver.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
class AIWorkerProfile
{
public:
  explicit AIWorkerProfile(rclcpp::Node & node);
  ~AIWorkerProfile();

  bool initialize();
  bool updateFollowerState(const sensor_msgs::msg::JointState & message);
  bool updateLeaderReference(
    const trajectory_msgs::msg::JointTrajectory & message,
    uint8_t target_arm);
  void publish(const Eigen::VectorXd & command);

  const Eigen::VectorXd & followerPosition() const {return follower_position_;}
  const Eigen::VectorXd & followerVelocity() const {return follower_velocity_;}
  const Eigen::VectorXd & leaderReference() const {return leader_reference_;}
  const Eigen::VectorXd & leaderPosition() const {return leader_position_;}
  const std::vector<int> & leftArmIndices() const {return left_arm_indices_;}
  const std::vector<int> & rightArmIndices() const {return right_arm_indices_;}
  const ModeConfiguration & modeConfiguration() const {return mode_configuration_;}
  int dof() const {return static_cast<int>(follower_position_.size());}

  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver>
  followerKinematics() const {return follower_kinematics_;}
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver>
  leaderKinematics() const {return leader_kinematics_;}

private:
  trajectory_msgs::msg::JointTrajectory makeArmTrajectory(
    const std::vector<int> & indices,
    const std::vector<std::string> & names,
    const Eigen::VectorXd & command,
    const std::string & gripper_name,
    double gripper_position) const;

  rclcpp::Node & node_;
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

  std::string right_gripper_joint_;
  std::string left_gripper_joint_;
  std::string temporary_leader_urdf_path_;
  double right_gripper_position_ = 0.0;
  double left_gripper_position_ = 0.0;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr right_publisher_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr left_publisher_;
};
}  // namespace cyclo_teleoperation::robots::ai_worker
