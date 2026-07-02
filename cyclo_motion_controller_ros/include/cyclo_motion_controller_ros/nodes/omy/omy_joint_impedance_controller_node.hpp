// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#pragma once

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "common/type_define.hpp"
#include "controllers/open_manipulator/open_manipulator_joint_impedance_controller.hpp"
#include "dynamics/dynamics_solver.hpp"

namespace cyclo_motion_controller_ros
{

class OmyJointImpedanceControllerNode : public rclcpp::Node
{
public:
  OmyJointImpedanceControllerNode();
  ~OmyJointImpedanceControllerNode() override;

private:
  static double commandDurationSeconds(const builtin_interfaces::msg::Duration & duration_msg)
  {
    return rclcpp::Duration(duration_msg).seconds();
  }

  void initializeJointConfig();
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void moveJCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void controlLoopCallback();
  void publishTorqueCommand(const Eigen::VectorXd & torque_command) const;
  void publishControllerError(const std::string & error) const;
  bool jointStateTimedOut() const;
  Eigen::VectorXd filterJointVelocity(
    const Eigen::VectorXd & raw_velocity,
    const rclcpp::Time & stamp);
  Eigen::VectorXd vectorFromParameter(const std::string & name, double default_value) const;

  double control_frequency_;
  double joint_state_timeout_;
  double velocity_filter_cutoff_frequency_;
  std::string urdf_path_;
  std::string srdf_path_;
  std::string joint_states_topic_;
  std::string joint_command_topic_;
  std::string movej_topic_;
  std::string controller_error_topic_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr movej_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_command_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr controller_error_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  std::shared_ptr<cyclo_motion_controller::dynamics::DynamicsSolver> dynamics_solver_;
  std::shared_ptr<cyclo_motion_controller::controllers::OpenManipulatorJointImpedanceController>
  qp_controller_;

  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd q_desired_;
  Eigen::VectorXd qdot_desired_;

  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> joint_index_map_;
  std::unordered_map<std::string, int> model_joint_index_map_;

  rclcpp::Time last_joint_state_time_;
  rclcpp::Time last_velocity_filter_time_;
  rclcpp::Time motion_start_time_;
  double active_motion_duration_{0.0};
  Eigen::VectorXd movej_start_;
  Eigen::VectorXd movej_goal_;
  bool joint_state_received_{false};
  bool velocity_filter_initialized_{false};
  bool movej_target_initialized_{false};
  bool movej_trajectory_active_{false};
};

}  // namespace cyclo_motion_controller_ros
