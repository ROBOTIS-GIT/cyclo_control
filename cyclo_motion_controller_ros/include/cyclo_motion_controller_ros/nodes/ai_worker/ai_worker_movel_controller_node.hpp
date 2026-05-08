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
#include <robotis_interfaces/msg/move_l.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "common/type_define.hpp"
#include "controllers/ai_worker/ai_worker_movel_controller.hpp"
#include "kinematics/kinematics_solver.hpp"

namespace cyclo_motion_controller_ros
{
class AIWorkerMoveLController : public rclcpp::Node
{
public:
  AIWorkerMoveLController();
  ~AIWorkerMoveLController();

private:
  static double commandDurationSeconds(const builtin_interfaces::msg::Duration & duration_msg)
  {
    return rclcpp::Duration(duration_msg).seconds();
  }

  double control_frequency_;
  double time_step_;
  double trajectory_time_;
  double kp_position_;
  double kp_orientation_;
  double weight_position_;
  double weight_orientation_;
  double weight_damping_;
  double slack_penalty_;
  double cbf_alpha_;
  double collision_buffer_;
  double collision_safe_distance_;
  double joint_state_timeout_;
  std::string joint_states_topic_;
  std::string right_movel_topic_;
  std::string left_movel_topic_;
  std::string right_traj_topic_;
  std::string left_traj_topic_;
  std::string lift_topic_;
  double lift_vel_bound_;
  std::string r_gripper_pose_topic_;
  std::string l_gripper_pose_topic_;
  std::string controller_error_topic_;
  std::string r_gripper_name_;
  std::string l_gripper_name_;
  std::string right_gripper_joint_name_;
  std::string left_gripper_joint_name_;
  std::string urdf_path_;
  std::string srdf_path_;

  rclcpp::Subscription<robotis_interfaces::msg::MoveL>::SharedPtr right_movel_sub_;
  rclcpp::Subscription<robotis_interfaces::msg::MoveL>::SharedPtr left_movel_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_l_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr lift_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_gripper_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_gripper_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr controller_error_pub_;

  rclcpp::TimerBase::SharedPtr control_timer_;

  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> kinematics_solver_;
  std::shared_ptr<cyclo_motion_controller::controllers::AIWorkerMoveLController> qp_controller_;

  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd q_desired_;

  Eigen::Affine3d right_gripper_pose_;
  Eigen::Affine3d left_gripper_pose_;
  Eigen::Affine3d right_movel_start_pose_;
  Eigen::Affine3d left_movel_start_pose_;
  Eigen::Affine3d right_movel_goal_pose_;
  Eigen::Affine3d left_movel_goal_pose_;

  bool joint_state_received_;
  bool q_desired_initialized_;
  bool right_movel_target_initialized_;
  bool left_movel_target_initialized_;
  bool right_movel_trajectory_active_;
  bool left_movel_trajectory_active_;
  bool joint_state_timeout_active_ = false;

  rclcpp::Time right_motion_start_time_;
  rclcpp::Time left_motion_start_time_;
  rclcpp::Time last_joint_state_time_;
  double right_active_motion_duration_;
  double left_active_motion_duration_;

  double right_gripper_position_;
  double left_gripper_position_;

  std::vector<std::string> left_arm_joints_;
  std::vector<std::string> right_arm_joints_;
  std::string lift_joint_;
  int lift_joint_index_ = -1;
  std::unordered_map<std::string, int> joint_index_map_;
  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> model_joint_index_map_;

  void rightMoveLCallback(const robotis_interfaces::msg::MoveL::SharedPtr msg);
  void leftMoveLCallback(const robotis_interfaces::msg::MoveL::SharedPtr msg);
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void controlLoopCallback();

  void initializeJointConfig();
  void publishTrajectory(const Eigen::VectorXd & q_desired);
  trajectory_msgs::msg::JointTrajectory createArmTrajectoryMsg(
    const std::vector<std::string> & arm_joint_names,
    const Eigen::VectorXd & positions,
    const std::vector<int> & arm_indices) const;
  trajectory_msgs::msg::JointTrajectory createLiftTrajectoryMsg(
    std::string lift_joint_name,
    const double position) const;
  void publishGripperPose(
    const Eigen::Affine3d & r_gripper_pose,
    const Eigen::Affine3d & l_gripper_pose);
  void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr & msg);
  bool jointStateTimedOut() const;
  void syncCommandStateToFeedback();
  void syncArmStateToFeedback(
    const std::vector<std::string> & arm_joint_names,
    Eigen::VectorXd & destination) const;

  Eigen::Affine3d poseMsgToEigen(const geometry_msgs::msg::PoseStamped & pose_msg) const;
  cyclo_motion_controller::common::Vector6d computeDesiredVelocity(
    const Eigen::Affine3d & current_pose,
    const Eigen::Affine3d & goal_pose,
    const Eigen::Vector3d & feedforward_linear = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d & feedforward_angular = Eigen::Vector3d::Zero()) const;
};
}  // namespace cyclo_motion_controller_ros
