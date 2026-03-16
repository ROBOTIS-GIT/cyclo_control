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
//
// Author: Yeonguk Kim

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "motion_controller_core/common/type_define.h"
#include "motion_controller_core/controllers/open_manipulator/open_manipulator_movej_controller.hpp"
#include "motion_controller_core/kinematics/kinematics_solver.hpp"

namespace motion_controller_ros
{
class OmyMoveJControllerNode : public rclcpp::Node
{
public:
  OmyMoveJControllerNode();
  ~OmyMoveJControllerNode();

private:
  void initializeJointConfig();
  void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr & msg);
  void publishCurrentPose(const Eigen::Affine3d & pose) const;
  void publishTrajectory(const Eigen::VectorXd & q_command) const;
  void publishControllerError(const std::string & error) const;

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void moveJCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void controlLoopCallback();

  double control_frequency_;
  double time_step_;
  double trajectory_time_;
  double kp_joint_;
  double weight_joint_tracking_;
  double weight_damping_;
  double slack_penalty_;
  double cbf_alpha_;
  double collision_buffer_;
  double collision_safe_distance_;

  std::string urdf_path_;
  std::string srdf_path_;
  std::string base_frame_;
  std::string controlled_link_;
  std::string joint_states_topic_;
  std::string joint_command_topic_;
  std::string movej_topic_;
  std::string ee_pose_topic_;
  std::string controller_error_topic_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr movej_sub_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_command_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr controller_error_pub_;

  rclcpp::TimerBase::SharedPtr control_timer_;

  std::shared_ptr<motion_controller::kinematics::KinematicsSolver> kinematics_solver_;
  std::shared_ptr<motion_controller::controllers::OpenManipulatorMoveJController> qp_controller_;

  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd q_commanded_;

  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> joint_index_map_;
  std::unordered_map<std::string, int> model_joint_index_map_;

  bool joint_state_received_;
  bool commanded_state_initialized_;
  bool movej_target_initialized_;
  bool movej_trajectory_active_;

  rclcpp::Time motion_start_time_;
  double active_motion_duration_;
  Eigen::VectorXd movej_start_;
  Eigen::VectorXd movej_goal_;
};
}  // namespace motion_controller_ros
