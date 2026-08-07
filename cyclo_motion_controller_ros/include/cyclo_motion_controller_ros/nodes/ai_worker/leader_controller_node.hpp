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
#include <std_msgs/msg/u_int8.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "common/type_define.hpp"
#include "controllers/ai_worker/elbow_up_qp_controller.hpp"
#include "kinematics/kinematics_solver.hpp"


namespace cyclo_motion_controller_ros
{
    /**
     * @brief ROS 2 wrapper node for generating end effector reference poses using leader device.
     *
     * This class implements methods to generate end effector reference poses using leader's joint configuration
     * using forward kinematics.
     */
class LeaderController : public rclcpp::Node
{
public:
  LeaderController();
  ~LeaderController();

private:
  enum class TeleopState : uint8_t
  {
    UNCONFIGURED = 0,
    DISABLED = 1,
    ENABLED = 2,
  };

        // Parameters
  double control_frequency_;
  double time_step_;
  double trajectory_time_;
  double kp_position_;
  double kp_orientation_;
  double weight_position_;
  double weight_orientation_;
  double weight_damping_;
  double elbow_up_velocity_;
  double elbow_nullspace_weight_;
  double elbow_nullspace_damping_;
  double elbow_nullspace_max_joint_velocity_;
  double slack_penalty_;
  double cbf_alpha_;
  double collision_buffer_;
  double collision_safe_distance_;
  std::string urdf_path_;
  std::string srdf_path_;
  std::string follower_urdf_path_;
  std::string follower_srdf_path_;
  std::string joint_states_topic_;
  std::string right_traj_topic_;
  std::string left_traj_topic_;
  std::string right_command_topic_;
  std::string left_command_topic_;
  std::string right_teleop_mode_service_;
  std::string left_teleop_mode_service_;
  std::string right_command_mode_topic_;
  std::string left_command_mode_topic_;
  std::string r_goal_pose_topic_;
  std::string l_goal_pose_topic_;
  std::string base_frame_id_;
  std::string r_gripper_name_;
  std::string l_gripper_name_;
  std::string follower_r_gripper_name_;
  std::string follower_l_gripper_name_;
  std::string r_elbow_name_;
  std::string l_elbow_name_;
  std::string leader_r_gripper_joint_name_;
  std::string leader_l_gripper_joint_name_;
  std::string follower_r_gripper_joint_name_;
  std::string follower_l_gripper_joint_name_;
  std::string lift_joint_name_;
  std::string model_lift_joint_name_;
  double command_timeout_;

        // Subscribers
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr r_traj_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr l_traj_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr right_teleop_mode_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr left_teleop_mode_client_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr right_command_mode_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr left_command_mode_sub_;

        // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_goal_pose_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_l_pub_;
        // Timer
  rclcpp::TimerBase::SharedPtr control_timer_;

        // Kinematics
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> leader_kinematics_;
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> follower_kinematics_;
  std::shared_ptr<cyclo_motion_controller::controllers::ElbowUpQPController> qp_controller_;

        // State
  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd follower_q_;
  Eigen::VectorXd follower_qdot_;
  Eigen::VectorXd follower_q_desired_;
  bool follower_joint_state_received_ = false;
  bool follower_command_initialized_ = false;
  bool right_traj_received_;
  bool left_traj_received_;
  bool lift_joint_received_;
  bool right_gripper_received_ = false;
  bool left_gripper_received_ = false;
  double right_gripper_position_ = 0.0;
  double left_gripper_position_ = 0.0;
  rclcpp::Time last_right_traj_time_;
  rclcpp::Time last_left_traj_time_;
  bool was_publishing_reference_ = false;
  TeleopState right_teleop_state_ = TeleopState::UNCONFIGURED;
  TeleopState left_teleop_state_ = TeleopState::UNCONFIGURED;
  bool right_service_was_ready_ = false;
  bool left_service_was_ready_ = false;
  bool right_mode_request_pending_ = false;
  bool left_mode_request_pending_ = false;
  bool right_mode_transition_pending_ = true;
  bool left_mode_transition_pending_ = true;
  bool right_goal_initialized_ = false;
  bool left_goal_initialized_ = false;
  Eigen::Affine3d right_leader_anchor_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d left_leader_anchor_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d right_follower_anchor_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d left_follower_anchor_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d right_goal_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d left_goal_ = Eigen::Affine3d::Identity();
  std::unordered_map<std::string, int> model_joint_index_map_;
  std::unordered_map<std::string, int> follower_joint_index_map_;
  std::map<std::string, int> joint_state_index_map_;
  std::vector<std::string> follower_joint_names_;
  std::vector<std::string> right_arm_joints_;
  std::vector<std::string> left_arm_joints_;
  int lift_joint_index_;
  int follower_lift_joint_index_ = -1;

        // Callbacks
  void rightTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void leftTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void controlLoopCallback();

        // Helpers
  void initializeJointConfig();
  void updateJointPositionsFromTrajectory(const trajectory_msgs::msg::JointTrajectory & msg);
  bool updateGripperFromTrajectory(
    const trajectory_msgs::msg::JointTrajectory & msg,
    const std::string & leader_gripper_joint_name, double & gripper_position);
  void updateLiftJointFromJointState(const sensor_msgs::msg::JointState & msg);
  void updateFollowerJointState(const sensor_msgs::msg::JointState & msg);
  void monitorTeleopModeServices();
  void requestRightTeleopMode();
  void requestLeftTeleopMode();
  void rightCommandModeCallback(const std_msgs::msg::UInt8::SharedPtr msg);
  void leftCommandModeCallback(const std_msgs::msg::UInt8::SharedPtr msg);
  void applyModeCommand(
    uint8_t command, TeleopState & state, bool & transition_pending,
    const char * side);
  bool parseModeResponse(const std::string & message, TeleopState & state) const;
  cyclo_motion_controller::common::Vector6d computeDesiredVelocity(
    const Eigen::Affine3d & current, const Eigen::Affine3d & goal) const;
  Eigen::VectorXd computeElbowUpPreferredJointVelocity() const;
  void publishFollowerTrajectory(const Eigen::VectorXd & desired);
  trajectory_msgs::msg::JointTrajectory makeArmTrajectory(
    const std::vector<std::string> & joint_names,
    const Eigen::VectorXd & desired, const std::string & gripper_joint_name,
    double gripper_position, bool gripper_received) const;
  geometry_msgs::msg::PoseStamped makePoseStamped(const Eigen::Affine3d & pose) const;
  Eigen::Affine3d computePoseInBaseFrame(const Eigen::Affine3d & link_pose) const;
};
}  // namespace cyclo_motion_controller_ros
