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
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "motion_controller_core/kinematics/kinematics_solver.hpp"
#include "motion_controller_core/controllers/ai_worker/joint_space_controller.hpp"


namespace motion_controller_ros
{
    /**
     * @brief ROS 2 wrapper node for applying QP filters for manipulators.
     *
     * This class implements methods to set QP filter for avoiding joint limits and self collisions
     * while tracking the desired joint trajectory.
     */
class JointSpaceController : public rclcpp::Node
{
public:
  JointSpaceController();
  ~JointSpaceController();

private:
        // Configurable parameters
  double control_frequency_;
  double time_step_;
  double weight_tracking_;
  double weight_damping_;
  double slack_penalty_;
  double cbf_alpha_;
  double collision_buffer_;
  double collision_safe_distance_;
  double command_timeout_;
  std::string joint_states_topic_;
  std::string right_traj_topic_;
  std::string left_traj_topic_;
  std::string right_traj_filtered_topic_;
  std::string left_traj_filtered_topic_;
  std::string right_gripper_joint_name_;
  std::string left_gripper_joint_name_;
  std::string urdf_path_;
  std::string srdf_path_;

        // Subscribers
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr r_traj_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr l_traj_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

        // Publishers
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_l_pub_;

        // Timer for control loop
  rclcpp::TimerBase::SharedPtr control_timer_;

        // Motion controller components
  std::shared_ptr<motion_controller::kinematics::KinematicsSolver> kinematics_solver_;
  std::shared_ptr<motion_controller::controllers::QPFilter> qp_filter_;

        // State variables
  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd q_desired_;
  Eigen::VectorXd qdot_desired_;
  bool right_traj_received_;
  bool left_traj_received_;
  bool joint_state_received_;
        /** Set true after first /joint_states message; thereafter state is last commanded pose. */
  bool open_loop_initialized_;
  double right_gripper_position_;
  double left_gripper_position_;
  rclcpp::Duration right_traj_time_from_start_;
  rclcpp::Duration left_traj_time_from_start_;
  rclcpp::Time last_right_traj_time_;
  rclcpp::Time last_left_traj_time_;

        // Joint configuration
  std::vector<std::string> left_arm_joints_;
  std::vector<std::string> right_arm_joints_;
  std::map<std::string, int> joint_index_map_;
  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> model_joint_index_map_;

        // Control loop parameters
  double dt_;        // Time step in seconds

        // Callbacks
  void rightTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void leftTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void controlLoopCallback();

        // Helper functions
  void initializeJointConfig();
  void publishTrajectory(const Eigen::VectorXd & q_desired);
  void updateDesiredVelocityFromTrajectory(
    const trajectory_msgs::msg::JointTrajectory & msg,
    Eigen::VectorXd & qdot_desired);

  trajectory_msgs::msg::JointTrajectory createTrajectoryMsgWithGripper(
    const std::vector<std::string> & arm_joint_names,
    const Eigen::VectorXd & positions,
    const std::vector<int> & arm_indices,
    const std::string & gripper_joint_name,
    double gripper_position,
    const rclcpp::Duration & time_from_start) const;

  void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr & msg);
};
}  // namespace motion_controller_ros
