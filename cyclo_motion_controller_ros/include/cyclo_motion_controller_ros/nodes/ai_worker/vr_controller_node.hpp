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

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "common/type_define.hpp"
#include "controllers/ai_worker/vr_controller.hpp"
#include "kinematics/kinematics_solver.hpp"


namespace cyclo_motion_controller_ros
{
    /**
     * @brief ROS 2 wrapper node for VR teleoperation controller.
     *
     * This node subscribes to target end-effector pose and current joint states to solve
     * inverse kinematics problems for VR teleoperation using Quadratic Programming.
     */
class VRController : public rclcpp::Node
{
public:
  VRController();
  ~VRController();

private:
        // Configurable parameters
  double control_frequency_;
  double time_step_;
  double trajectory_time_;
  double kp_position_;
  double kp_orientation_;
  double weight_position_;
  double weight_orientation_;
  double weight_elbow_position_;
  double weight_damping_;
  double slack_penalty_;
  double cbf_alpha_;
  double collision_buffer_;
  double collision_safe_distance_;
  double joint_state_timeout_;
  std::string reactivate_topic_;
  std::string r_goal_pose_topic_;
  std::string l_goal_pose_topic_;
  std::string r_elbow_pose_topic_;
  std::string l_elbow_pose_topic_;
  std::string joint_states_topic_;
  std::string right_traj_topic_;
  std::string left_traj_topic_;
  std::string right_raw_traj_topic_;
  std::string left_raw_traj_topic_;
  double raw_traj_timeout_;
  std::string lift_topic_;
  double lift_vel_bound_;
  std::string r_gripper_pose_topic_;
  std::string l_gripper_pose_topic_;
  std::string r_gripper_name_;
  std::string l_gripper_name_;
  std::string r_elbow_name_;
  std::string l_elbow_name_;
  std::string right_gripper_joint_name_;
  std::string left_gripper_joint_name_;
  std::string urdf_path_;
  std::string srdf_path_;

        // Subscribers
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr l_goal_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr r_elbow_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr l_elbow_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr right_raw_traj_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr left_raw_traj_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr ref_divergence_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reactivate_sub_;

        // Publishers
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_l_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr lift_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_gripper_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_gripper_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr reference_divergence_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr controller_error_pub_;

        // Timer for control loop
  rclcpp::TimerBase::SharedPtr control_timer_;

        // Motion controller components
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> kinematics_solver_;
  std::shared_ptr<cyclo_motion_controller::controllers::VRController> qp_controller_;

        // State variables (measured)
  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;

        // State variables (observer / internal model)
  Eigen::VectorXd q_model_;
  bool model_state_initialized_ = false;

        // Commanded state
  Eigen::VectorXd q_desired_;

        // Task-space state
  Eigen::Affine3d right_gripper_pose_;
  Eigen::Affine3d left_gripper_pose_;
  Eigen::Affine3d r_goal_pose_;
  Eigen::Affine3d l_goal_pose_;
  Eigen::Affine3d r_elbow_pose_;
  Eigen::Affine3d l_elbow_pose_;

  bool r_goal_pose_received_;
  bool l_goal_pose_received_;
  bool r_elbow_pose_received_;
  bool l_elbow_pose_received_;
  bool reference_diverged_;
  rclcpp::Time activate_start_;
  bool activate_pending_;
  bool control_enabled_ = false;        // start only after reactivate service
  bool start_requested_ = false;        // reactivate has been requested
  bool joint_state_received_;
  bool joint_state_timeout_active_ = false;
  bool reactivate_state_ = false;

        // Startup reference vs current pose check
  double startup_ref_pos_threshold_ = 0.15;              // meters
  double startup_ref_ori_threshold_deg_ = 45.0;          // degrees

        // Latest gripper positions from raw joint trajectory (leader side)
  bool right_raw_gripper_received_ = false;
  bool left_raw_gripper_received_ = false;
  double right_raw_gripper_position_ = 0.0;
  double left_raw_gripper_position_ = 0.0;
  rclcpp::Time last_joint_state_time_;
  rclcpp::Time last_right_raw_traj_time_;
  rclcpp::Time last_left_raw_traj_time_;

        // Control timing
  double dt_;        // nominal time step in seconds
  rclcpp::Time last_control_time_;
  bool last_control_time_initialized_ = false;

        // Joint configuration
  std::vector<std::string> left_arm_joints_;
  std::vector<std::string> right_arm_joints_;
  std::string lift_joint_;
  int lift_joint_index_ = -1;        // index in model q/qdot; -1 if not present
  std::map<std::string, int> joint_index_map_;
  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> model_joint_index_map_;

        // Callbacks
  void rightGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void leftGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void rightElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void leftElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void rightRawTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void leftRawTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void referenceDivergenceCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void reactivateCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void controlLoopCallback();

        // Helper functions
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

        // Control computation functions
  Eigen::Affine3d computePoseMat(const geometry_msgs::msg::PoseStamped & pose) const;
  cyclo_motion_controller::common::Vector6d computeDesiredVelocity(
    const Eigen::Affine3d & current_pose,
    const Eigen::Affine3d & goal_pose) const;
};
}  // namespace cyclo_motion_controller_ros
