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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "common/type_define.hpp"
#include "controllers/ai_worker/ai_worker_movej_controller.hpp"
#include "kinematics/kinematics_solver.hpp"

namespace cyclo_motion_controller_ros
{
class AIWorkerMoveJController : public rclcpp::Node
{
public:
  AIWorkerMoveJController();
  ~AIWorkerMoveJController();

private:
  void initializeJointConfig();
  void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr & msg);
  void publishTrajectory(const Eigen::VectorXd & q_command) const;
  bool jointStateTimedOut() const;
  void syncCommandStateToFeedback();
  void syncRightArmToFeedback();
  void syncLeftArmToFeedback();

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void rightTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void leftTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void controlLoopCallback();

  bool updateArmTargetFromTrajectory(
    const trajectory_msgs::msg::JointTrajectory & msg,
    const std::vector<std::string> & arm_joint_names,
    const std::string & arm_name,
    Eigen::VectorXd & target_q) const;
  void assignArmSegment(
    const Eigen::VectorXd & source,
    const std::vector<std::string> & arm_joint_names,
    Eigen::VectorXd & destination) const;
  void updateGripperPositionFromTrajectory(
    const trajectory_msgs::msg::JointTrajectory & msg,
    const std::string & gripper_joint_name,
    double & gripper_position) const;

  trajectory_msgs::msg::JointTrajectory createTrajectoryMsgWithGripper(
    const std::vector<std::string> & arm_joint_names,
    const Eigen::VectorXd & positions,
    const std::vector<int> & arm_indices,
    const std::string & gripper_joint_name,
    double gripper_position) const;

  double control_frequency_;
  double time_step_;
  double trajectory_time_;
  double kp_joint_;
  double weight_tracking_;
  double weight_damping_;
  double slack_penalty_;
  double cbf_alpha_;
  double collision_buffer_;
  double collision_safe_distance_;
  double joint_state_timeout_;
  std::string joint_states_topic_;
  std::string right_traj_topic_;
  std::string left_traj_topic_;
  std::string right_traj_filtered_topic_;
  std::string left_traj_filtered_topic_;
  std::string right_gripper_joint_name_;
  std::string left_gripper_joint_name_;
  std::string urdf_path_;
  std::string srdf_path_;

  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr r_traj_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr l_traj_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_l_pub_;

  rclcpp::TimerBase::SharedPtr control_timer_;

  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> kinematics_solver_;
  std::shared_ptr<cyclo_motion_controller::controllers::AIWorkerMoveJController> qp_filter_;

  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd q_commanded_;

  Eigen::VectorXd right_movej_start_;
  Eigen::VectorXd right_movej_goal_;
  Eigen::VectorXd left_movej_start_;
  Eigen::VectorXd left_movej_goal_;

  bool joint_state_received_;
  bool commanded_state_initialized_;
  bool right_movej_target_initialized_;
  bool left_movej_target_initialized_;
  bool joint_state_timeout_active_ = false;

  double right_gripper_position_;
  double left_gripper_position_;
  rclcpp::Time last_joint_state_time_;

  std::vector<std::string> left_arm_joints_;
  std::vector<std::string> right_arm_joints_;
  std::map<std::string, int> joint_index_map_;
  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> model_joint_index_map_;
};
}  // namespace cyclo_motion_controller_ros
