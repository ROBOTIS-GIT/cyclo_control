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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "controllers/ai_worker/ai_worker_bimanual_movej_controller.hpp"
#include "kinematics/kinematics_solver.hpp"

namespace cyclo_motion_controller_ros
{
class AIWorkerBimanualMoveJController : public rclcpp::Node
{
public:
  AIWorkerBimanualMoveJController();
  ~AIWorkerBimanualMoveJController();

private:
  void initializeJointConfig();
  void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr & msg);
  void publishTrajectory(const Eigen::VectorXd & q_command) const;
  bool jointStateTimedOut() const;
  void syncCommandStateToFeedback();
  void syncRightArmToFeedback();
  void syncLeftArmToFeedback();
  double maxLeaderCommandError(
    const Eigen::VectorXd & goal,
    const std::vector<std::string> & joints) const;
  void startGraspReleaseSlowStart(bool right_arm, bool left_arm);
  void startPendingGraspReleaseSlowStart(bool right_arm, bool left_arm);
  void startPendingGraspEnableBlend(bool right_arm, bool left_arm);
  void enableGraspReleaseArmFollow(bool right_arm, bool left_arm);

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void rightTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void leftTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void graspCaptureCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void controlLoopCallback();
  void updateGripperTriggeredGraspMode();
  void enableGraspConstraint();
  void disableGraspConstraint(
    bool right_arm_follow_enabled = true,
    bool left_arm_follow_enabled = true);

  bool updateArmTargetFromTrajectory(
    const trajectory_msgs::msg::JointTrajectory & msg,
    const std::vector<std::string> & arm_joint_names,
    const std::string & arm_name,
    Eigen::VectorXd & target_q) const;
  void assignArmSegment(
    const Eigen::VectorXd & source,
    const std::vector<std::string> & arm_joint_names,
    Eigen::VectorXd & destination) const;
  bool updateGripperPositionFromTrajectory(
    const trajectory_msgs::msg::JointTrajectory & msg,
    const std::string & gripper_joint_name,
    double & gripper_position) const;
  void captureCurrentGraspConstraint();

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
  double gripper_grasp_threshold_;
  double gripper_grasp_hold_time_;
  std::string joint_states_topic_;
  std::string right_traj_topic_;
  std::string left_traj_topic_;
  std::string grasp_capture_topic_;
  std::string right_traj_filtered_topic_;
  std::string left_traj_filtered_topic_;
  std::string r_gripper_name_;
  std::string l_gripper_name_;
  std::string right_gripper_joint_name_;
  std::string left_gripper_joint_name_;
  std::string urdf_path_;
  std::string srdf_path_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr r_traj_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr l_traj_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr grasp_capture_sub_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_l_pub_;

  rclcpp::TimerBase::SharedPtr control_timer_;

  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> kinematics_solver_;
  std::shared_ptr<cyclo_motion_controller::controllers::AIWorkerBimanualMoveJController>
  qp_filter_;

  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd q_commanded_;

  Eigen::VectorXd right_movej_start_;
  Eigen::VectorXd right_movej_goal_;
  Eigen::VectorXd left_movej_start_;
  Eigen::VectorXd left_movej_goal_;
  Eigen::VectorXd right_release_hold_goal_;
  Eigen::VectorXd left_release_hold_goal_;
  Eigen::VectorXd right_grasp_enable_blend_start_;
  Eigen::VectorXd left_grasp_enable_blend_start_;

  bool joint_state_received_;
  bool commanded_state_initialized_;
  bool right_movej_target_initialized_;
  bool left_movej_target_initialized_;
  bool right_movej_trajectory_active_ = false;
  bool left_movej_trajectory_active_ = false;
  bool right_timed_streaming_active_ = false;
  bool left_timed_streaming_active_ = false;
  bool right_timed_command_received_ = false;
  bool left_timed_command_received_ = false;
  bool joint_state_timeout_active_ = false;
  bool grasp_constraint_active_ = false;
  bool manual_grasp_latch_ = false;
  bool right_gripper_command_received_ = false;
  bool left_gripper_command_received_ = false;
  bool gripper_closed_timer_active_ = false;
  bool right_gripper_open_timer_active_ = false;
  bool left_gripper_open_timer_active_ = false;
  bool grasp_release_follow_limited_ = false;
  bool right_release_follow_enabled_ = true;
  bool left_release_follow_enabled_ = true;
  bool right_grasp_release_slow_start_pending_ = false;
  bool left_grasp_release_slow_start_pending_ = false;
  bool right_grasp_release_slow_start_active_ = false;
  bool left_grasp_release_slow_start_active_ = false;
  bool right_grasp_enable_blend_pending_ = false;
  bool left_grasp_enable_blend_pending_ = false;
  bool right_grasp_enable_blend_active_ = false;
  bool left_grasp_enable_blend_active_ = false;

  double right_gripper_position_;
  double left_gripper_position_;
  double right_active_motion_duration_ = 0.0;
  double left_active_motion_duration_ = 0.0;
  rclcpp::Time right_motion_start_time_;
  rclcpp::Time left_motion_start_time_;
  rclcpp::Time right_last_timed_command_time_;
  rclcpp::Time left_last_timed_command_time_;
  rclcpp::Time last_joint_state_time_;
  rclcpp::Time gripper_closed_since_;
  rclcpp::Time right_gripper_open_since_;
  rclcpp::Time left_gripper_open_since_;
  rclcpp::Time right_grasp_release_slow_start_time_;
  rclcpp::Time left_grasp_release_slow_start_time_;
  rclcpp::Time right_grasp_enable_blend_start_time_;
  rclcpp::Time left_grasp_enable_blend_start_time_;
  Eigen::Affine3d grasp_right_to_left_ = Eigen::Affine3d::Identity();

  std::vector<std::string> left_arm_joints_;
  std::vector<std::string> right_arm_joints_;
  std::map<std::string, int> joint_index_map_;
  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> model_joint_index_map_;
};
}  // namespace cyclo_motion_controller_ros
