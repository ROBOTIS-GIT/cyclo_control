// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

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

#include "robotis_interfaces/msg/move_l.hpp"
#include "controllers/open_manipulator/open_manipulator_cartesian_impedance_controller.hpp"
#include "dynamics/dynamics_solver.hpp"

namespace cyclo_motion_controller_ros
{

class OmyCartesianImpedanceControllerNode : public rclcpp::Node
{
public:
  OmyCartesianImpedanceControllerNode();
  ~OmyCartesianImpedanceControllerNode() override;

private:
  static double commandDurationSeconds(const builtin_interfaces::msg::Duration & duration_msg)
  {
    return rclcpp::Duration(duration_msg).seconds();
  }

  void initializeJointConfig();
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void moveLCallback(const robotis_interfaces::msg::MoveL::SharedPtr msg);
  void controlLoopCallback();
  void publishTorqueCommand(const Eigen::VectorXd & torque_command) const;
  void publishCurrentPose(const Eigen::Affine3d & pose) const;
  void publishControllerError(const std::string & error) const;
  bool jointStateTimedOut() const;
  Eigen::VectorXd filterJointVelocity(
    const Eigen::VectorXd & raw_velocity,
    const rclcpp::Time & stamp);
  Eigen::Affine3d poseMsgToEigen(const geometry_msgs::msg::PoseStamped & pose_msg) const;
  Eigen::VectorXd vectorFromParameter(const std::string & name, double default_value) const;
  cyclo_motion_controller::common::Vector6d vector6FromParameter(
    const std::string & name,
    double default_value) const;

  double control_frequency_;
  double joint_state_timeout_;
  double velocity_filter_cutoff_frequency_;
  std::string urdf_path_;
  std::string srdf_path_;
  std::string base_frame_;
  std::string controlled_link_;
  std::string joint_states_topic_;
  std::string joint_command_topic_;
  std::string movel_topic_;
  std::string ee_pose_topic_;
  std::string controller_error_topic_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<robotis_interfaces::msg::MoveL>::SharedPtr movel_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_command_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr controller_error_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  std::shared_ptr<cyclo_motion_controller::dynamics::DynamicsSolver> dynamics_solver_;
  std::shared_ptr<cyclo_motion_controller::controllers::OpenManipulatorCartesianImpedanceController>
  qp_controller_;

  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::Affine3d desired_pose_;
  Eigen::Affine3d movel_start_pose_;
  Eigen::Affine3d movel_goal_pose_;

  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> joint_index_map_;
  std::unordered_map<std::string, int> model_joint_index_map_;

  rclcpp::Time last_joint_state_time_;
  rclcpp::Time last_velocity_filter_time_;
  rclcpp::Time motion_start_time_;
  double active_motion_duration_{0.0};
  bool joint_state_received_{false};
  bool velocity_filter_initialized_{false};
  bool movel_target_initialized_{false};
  bool movel_trajectory_active_{false};
};

}  // namespace cyclo_motion_controller_ros
