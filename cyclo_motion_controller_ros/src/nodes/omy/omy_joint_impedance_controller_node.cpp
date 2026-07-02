// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cyclo_motion_controller_ros/nodes/omy/omy_joint_impedance_controller_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>

namespace cyclo_motion_controller_ros
{

OmyJointImpedanceControllerNode::OmyJointImpedanceControllerNode()
: Node("omy_joint_impedance_controller"),
  control_frequency_(100.0),
  joint_state_timeout_(0.5),
  last_joint_state_time_(this->now()),
  motion_start_time_(this->now())
{
  control_frequency_ = this->declare_parameter("control_frequency", 100.0);
  joint_state_timeout_ = this->declare_parameter("joint_state_timeout", 0.5);
  velocity_filter_cutoff_frequency_ =
    this->declare_parameter("velocity_filter_cutoff_frequency", 5.0);
  urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
  srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
  joint_states_topic_ =
    this->declare_parameter("joint_states_topic", std::string("/leader/joint_states"));
  joint_command_topic_ =
    this->declare_parameter("joint_command_topic", std::string("/leader/joint_trajectory"));
  movej_topic_ =
    this->declare_parameter("movej_topic", std::string("~/movej"));
  controller_error_topic_ =
    this->declare_parameter("controller_error_topic", std::string("~/controller_error"));

  this->declare_parameter<std::vector<double>>("stiffness", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("damping", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("torque_weight", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("torque_limit", std::vector<double>{});

  if (urdf_path_.empty()) {
    RCLCPP_FATAL(this->get_logger(), "URDF path not provided.");
    rclcpp::shutdown();
    return;
  }

  try {
    dynamics_solver_ =
      std::make_shared<cyclo_motion_controller::dynamics::DynamicsSolver>(urdf_path_, srdf_path_);
    qp_controller_ =
      std::make_shared<cyclo_motion_controller::controllers::OpenManipulatorJointImpedanceController>(
      dynamics_solver_);

    const int dof = dynamics_solver_->getKinematicsSolver()->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);
    q_desired_.setZero(dof);
    qdot_desired_.setZero(dof);
    movej_start_.setZero(dof);
    movej_goal_.setZero(dof);

    initializeJointConfig();

    qp_controller_->setGains(
      vectorFromParameter("stiffness", 20.0),
      vectorFromParameter("damping", 2.0));
    qp_controller_->setTorqueWeight(vectorFromParameter("torque_weight", 1.0));

    auto effort_limit = dynamics_solver_->getJointEffortLimit();
    const Eigen::VectorXd torque_limit = vectorFromParameter("torque_limit", 0.0);
    if ((torque_limit.array() > 0.0).all()) {
      effort_limit.first = -torque_limit;
      effort_limit.second = torque_limit;
    }
    qp_controller_->setTorqueLimits(effort_limit.first, effort_limit.second);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      this->get_logger(), "Failed to initialize OMY Joint Impedance Controller: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  joint_command_pub_ =
    this->create_publisher<trajectory_msgs::msg::JointTrajectory>(joint_command_topic_, 10);
  controller_error_pub_ =
    this->create_publisher<std_msgs::msg::String>(controller_error_topic_, 10);
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    joint_states_topic_, 10,
    std::bind(&OmyJointImpedanceControllerNode::jointStateCallback, this, std::placeholders::_1));
  movej_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    movej_topic_, 10,
    std::bind(&OmyJointImpedanceControllerNode::moveJCallback, this, std::placeholders::_1));

  const int timer_period_ms =
    std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, control_frequency_))));
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(timer_period_ms),
    std::bind(&OmyJointImpedanceControllerNode::controlLoopCallback, this));

  RCLCPP_INFO(this->get_logger(), "OMY Joint Impedance Controller initialized.");
}

OmyJointImpedanceControllerNode::~OmyJointImpedanceControllerNode()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down OMY Joint Impedance Controller.");
}

void OmyJointImpedanceControllerNode::initializeJointConfig()
{
  model_joint_names_ = dynamics_solver_->getKinematicsSolver()->getJointNames();
  model_joint_index_map_.clear();
  for (size_t i = 0; i < model_joint_names_.size(); ++i) {
    model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
  }
}

void OmyJointImpedanceControllerNode::jointStateCallback(
  const sensor_msgs::msg::JointState::SharedPtr msg)
{
  joint_index_map_.clear();
  for (size_t i = 0; i < msg->name.size(); ++i) {
    joint_index_map_[msg->name[i]] = static_cast<int>(i);
  }

  const int dof = static_cast<int>(model_joint_names_.size());
  Eigen::VectorXd raw_qdot = qdot_;
  if (raw_qdot.size() != dof) {
    raw_qdot.setZero(dof);
  }

  for (int i = 0; i < dof; ++i) {
    const auto it = joint_index_map_.find(model_joint_names_[i]);
    if (it == joint_index_map_.end()) {
      continue;
    }
    const int msg_idx = it->second;
    if (msg_idx < static_cast<int>(msg->position.size())) {
      q_[i] = msg->position[msg_idx];
    }
    if (msg_idx < static_cast<int>(msg->velocity.size())) {
      raw_qdot[i] = msg->velocity[msg_idx];
    }
  }

  const rclcpp::Time stamp = this->now();
  qdot_ = filterJointVelocity(raw_qdot, stamp);

  if (!joint_state_received_) {
    q_desired_ = q_;
    qdot_desired_.setZero();
    movej_start_ = q_;
    movej_goal_ = q_;
  }

  joint_state_received_ = true;
  last_joint_state_time_ = stamp;
}

void OmyJointImpedanceControllerNode::moveJCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!msg || msg->points.empty() || !joint_state_received_ || jointStateTimedOut()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      2000,
      "Ignoring moveJ command until joint states are available.");
    return;
  }

  const auto & point = msg->points.front();
  if (point.positions.empty()) {
    publishControllerError("moveJ command ignored: point.positions is empty.");
    return;
  }

  const double duration = commandDurationSeconds(point.time_from_start);
  if (duration <= -1.0) {
    publishControllerError("moveJ command ignored: time_from_start must be > -1.");
    return;
  }

  Eigen::VectorXd target_q = q_;
  if (!msg->joint_names.empty()) {
    for (size_t i = 0; i < msg->joint_names.size(); ++i) {
      if (i >= point.positions.size()) {
        continue;
      }
      const auto model_it = model_joint_index_map_.find(msg->joint_names[i]);
      if (model_it == model_joint_index_map_.end()) {
        continue;
      }
      target_q[model_it->second] = point.positions[i];
    }
  } else if (point.positions.size() == model_joint_names_.size()) {
    for (size_t i = 0; i < model_joint_names_.size(); ++i) {
      target_q[static_cast<int>(i)] = point.positions[i];
    }
  } else {
    publishControllerError(
      "moveJ command ignored: joint_names missing and positions size does not match model joints.");
    return;
  }

  movej_start_ = q_;
  movej_goal_ = target_q;
  active_motion_duration_ = duration;
  motion_start_time_ = this->now();
  movej_target_initialized_ = true;
  movej_trajectory_active_ = duration > 0.0;
}

void OmyJointImpedanceControllerNode::controlLoopCallback()
{
  if (!joint_state_received_ || jointStateTimedOut()) {
    return;
  }

  try {
    dynamics_solver_->updateState(q_, qdot_);
    if (!movej_target_initialized_) {
      q_desired_ = q_;
      qdot_desired_.setZero();
    } else {
      const double elapsed = (this->now() - motion_start_time_).seconds();
      if (movej_trajectory_active_ && elapsed < active_motion_duration_) {
        q_desired_ = cyclo_motion_controller::common::math_utils::cubicVector(
          elapsed,
          0.0,
          active_motion_duration_,
          movej_start_,
          movej_goal_,
          Eigen::VectorXd::Zero(movej_start_.size()),
          Eigen::VectorXd::Zero(movej_goal_.size()));
        qdot_desired_ = cyclo_motion_controller::common::math_utils::cubicDotVector(
          elapsed,
          0.0,
          active_motion_duration_,
          movej_start_,
          movej_goal_,
          Eigen::VectorXd::Zero(movej_start_.size()),
          Eigen::VectorXd::Zero(movej_goal_.size()));
      } else {
        if (movej_trajectory_active_) {
          movej_trajectory_active_ = false;
        }
        q_desired_ = movej_goal_;
        qdot_desired_.setZero();
      }
    }
    qp_controller_->setDesiredJointState(q_desired_, qdot_desired_);

    Eigen::VectorXd torque_command;
    if (!qp_controller_->getCommand(torque_command)) {
      publishControllerError("Joint impedance QP failed; publishing zero torque fallback.");
    }
    publishTorqueCommand(torque_command);
  } catch (const std::exception & e) {
    publishControllerError(e.what());
  }
}

void OmyJointImpedanceControllerNode::publishTorqueCommand(
  const Eigen::VectorXd & torque_command) const
{
  trajectory_msgs::msg::JointTrajectory traj_msg;
  traj_msg.header.stamp = this->now();
  traj_msg.joint_names = model_joint_names_;
  traj_msg.points.resize(1);
  auto & point = traj_msg.points.front();
  point.effort.resize(model_joint_names_.size(), 0.0);
  point.time_from_start = rclcpp::Duration::from_seconds(0.0);

  for (size_t i = 0; i < model_joint_names_.size() && i < static_cast<size_t>(torque_command.size());
    ++i)
  {
    point.effort[i] = torque_command[static_cast<int>(i)];
  }

  joint_command_pub_->publish(traj_msg);
}

void OmyJointImpedanceControllerNode::publishControllerError(const std::string & error) const
{
  if (!controller_error_pub_) {
    return;
  }
  std_msgs::msg::String msg;
  msg.data = error;
  controller_error_pub_->publish(msg);
  RCLCPP_WARN(this->get_logger(), "%s", error.c_str());
}

bool OmyJointImpedanceControllerNode::jointStateTimedOut() const
{
  return (this->now() - last_joint_state_time_).seconds() > joint_state_timeout_;
}

Eigen::VectorXd OmyJointImpedanceControllerNode::filterJointVelocity(
  const Eigen::VectorXd & raw_velocity,
  const rclcpp::Time & stamp)
{
  if (velocity_filter_cutoff_frequency_ <= 0.0) {
    return raw_velocity;
  }

  if (!velocity_filter_initialized_ || qdot_.size() != raw_velocity.size()) {
    velocity_filter_initialized_ = true;
    last_velocity_filter_time_ = stamp;
    return raw_velocity;
  }

  double dt = (stamp - last_velocity_filter_time_).seconds();
  if (dt <= 0.0) {
    dt = 1.0 / std::max(1.0, control_frequency_);
  }
  last_velocity_filter_time_ = stamp;

  const double tau = 1.0 / (2.0 * M_PI * velocity_filter_cutoff_frequency_);
  const double alpha = std::clamp(dt / (tau + dt), 0.0, 1.0);
  return qdot_ + alpha * (raw_velocity - qdot_);
}

Eigen::VectorXd OmyJointImpedanceControllerNode::vectorFromParameter(
  const std::string & name,
  double default_value) const
{
  const int dof = dynamics_solver_->getKinematicsSolver()->getDof();
  std::vector<double> values;
  if (!this->get_parameter(name, values) || values.empty()) {
    return Eigen::VectorXd::Constant(dof, default_value);
  }
  if (values.size() == 1) {
    return Eigen::VectorXd::Constant(dof, values.front());
  }
  if (values.size() != static_cast<size_t>(dof)) {
    RCLCPP_WARN(
      this->get_logger(), "Parameter '%s' has %zu values, expected %d. Using default.",
      name.c_str(), values.size(), dof);
    return Eigen::VectorXd::Constant(dof, default_value);
  }

  return Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<int>(values.size()));
}

}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cyclo_motion_controller_ros::OmyJointImpedanceControllerNode>());
  rclcpp::shutdown();
  return 0;
}
