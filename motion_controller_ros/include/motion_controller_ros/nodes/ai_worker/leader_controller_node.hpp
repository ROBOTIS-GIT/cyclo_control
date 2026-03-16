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
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "motion_controller_core/kinematics/kinematics_solver.hpp"


namespace motion_controller_ros
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
        // Parameters
  double control_frequency_;
  std::string urdf_path_;
  std::string srdf_path_;
  std::string joint_states_topic_;
  std::string right_traj_topic_;
  std::string left_traj_topic_;
  std::string reactivate_service_;
  std::string r_goal_pose_topic_;
  std::string l_goal_pose_topic_;
  std::string r_elbow_pose_topic_;
  std::string l_elbow_pose_topic_;
  std::string base_frame_id_;
  std::string base_frame_link_name_;
  std::string r_gripper_name_;
  std::string l_gripper_name_;
  std::string r_elbow_name_;
  std::string l_elbow_name_;
  std::string lift_joint_name_;
  std::string model_lift_joint_name_;
  double lift_joint_offset_;
  double command_timeout_;

        // Subscribers
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr r_traj_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr l_traj_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

        // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_goal_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_elbow_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_elbow_pose_pub_;

        // Service clients
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reactivate_client_;
        // Timer
  rclcpp::TimerBase::SharedPtr control_timer_;

        // Kinematics
  std::shared_ptr<motion_controller::kinematics::KinematicsSolver> kinematics_solver_;

        // State
  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  bool right_traj_received_;
  bool left_traj_received_;
  bool lift_joint_received_;
  rclcpp::Time last_right_traj_time_;
  rclcpp::Time last_left_traj_time_;
  bool was_publishing_reference_ = false;
  bool reactivate_requested_ = false;
  std::unordered_map<std::string, int> model_joint_index_map_;
  int lift_joint_index_;

        // Callbacks
  void rightTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void leftTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void controlLoopCallback();

        // Helpers
  void initializeJointConfig();
  void updateJointPositionsFromTrajectory(const trajectory_msgs::msg::JointTrajectory & msg);
  void updateLiftJointFromJointState(const sensor_msgs::msg::JointState & msg);
  bool requestReactivateOnce();
  geometry_msgs::msg::PoseStamped makePoseStamped(const Eigen::Affine3d & pose) const;
  Eigen::Affine3d computePoseInBaseFrame(const Eigen::Affine3d & link_pose) const;
};
}  // namespace motion_controller_ros
