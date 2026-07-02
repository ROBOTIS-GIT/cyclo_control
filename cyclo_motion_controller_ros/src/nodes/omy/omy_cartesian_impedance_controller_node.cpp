// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cyclo_motion_controller_ros/nodes/omy/omy_cartesian_impedance_controller_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>

namespace cyclo_motion_controller_ros
{

OmyCartesianImpedanceControllerNode::OmyCartesianImpedanceControllerNode()
: Node("omy_cartesian_impedance_controller"),
  control_frequency_(100.0),
  joint_state_timeout_(0.5),
  desired_pose_(Eigen::Affine3d::Identity()),
  movel_start_pose_(Eigen::Affine3d::Identity()),
  movel_goal_pose_(Eigen::Affine3d::Identity()),
  last_joint_state_time_(this->now()),
  motion_start_time_(this->now())
{
  control_frequency_ = this->declare_parameter("control_frequency", 100.0);
  joint_state_timeout_ = this->declare_parameter("joint_state_timeout", 0.5);
  velocity_filter_cutoff_frequency_ =
    this->declare_parameter("velocity_filter_cutoff_frequency", 5.0);
  urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
  srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
  base_frame_ = this->declare_parameter("base_frame", std::string("link0"));
  controlled_link_ = this->declare_parameter("controlled_link", std::string("link6"));
  joint_states_topic_ =
    this->declare_parameter("joint_states_topic", std::string("/leader/joint_states"));
  joint_command_topic_ =
    this->declare_parameter("joint_command_topic", std::string("/leader/joint_trajectory"));
  movel_topic_ = this->declare_parameter("movel_topic", std::string("~/movel"));
  ee_pose_topic_ = this->declare_parameter("ee_pose_topic", std::string("~/current_pose"));
  controller_error_topic_ =
    this->declare_parameter("controller_error_topic", std::string("~/controller_error"));

  this->declare_parameter<std::vector<double>>("stiffness", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("damping", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("nullspace_stiffness", std::vector<double>{});
  this->declare_parameter<std::vector<double>>("nullspace_damping", std::vector<double>{});
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
      std::make_shared<cyclo_motion_controller::controllers::OpenManipulatorCartesianImpedanceController>(
      dynamics_solver_, controlled_link_);

    const int dof = dynamics_solver_->getKinematicsSolver()->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);

    initializeJointConfig();

    qp_controller_->setGains(
      vector6FromParameter("stiffness", 50.0),
      vector6FromParameter("damping", 5.0));
    qp_controller_->setNullspaceGains(
      vectorFromParameter("nullspace_stiffness", 0.0),
      vectorFromParameter("nullspace_damping", 0.5));
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
      this->get_logger(), "Failed to initialize OMY Cartesian Impedance Controller: %s",
      e.what());
    rclcpp::shutdown();
    return;
  }

  joint_command_pub_ =
    this->create_publisher<trajectory_msgs::msg::JointTrajectory>(joint_command_topic_, 10);
  ee_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(ee_pose_topic_, 10);
  controller_error_pub_ =
    this->create_publisher<std_msgs::msg::String>(controller_error_topic_, 10);
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    joint_states_topic_, 10,
    std::bind(
      &OmyCartesianImpedanceControllerNode::jointStateCallback, this, std::placeholders::_1));
  movel_sub_ = this->create_subscription<robotis_interfaces::msg::MoveL>(
    movel_topic_, 10,
    std::bind(
      &OmyCartesianImpedanceControllerNode::moveLCallback, this, std::placeholders::_1));

  const int timer_period_ms =
    std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, control_frequency_))));
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(timer_period_ms),
    std::bind(&OmyCartesianImpedanceControllerNode::controlLoopCallback, this));

  RCLCPP_INFO(this->get_logger(), "OMY Cartesian Impedance Controller initialized.");
}

OmyCartesianImpedanceControllerNode::~OmyCartesianImpedanceControllerNode()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down OMY Cartesian Impedance Controller.");
}

void OmyCartesianImpedanceControllerNode::initializeJointConfig()
{
  model_joint_names_ = dynamics_solver_->getKinematicsSolver()->getJointNames();
  model_joint_index_map_.clear();
  for (size_t i = 0; i < model_joint_names_.size(); ++i) {
    model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
  }
}

void OmyCartesianImpedanceControllerNode::jointStateCallback(
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

  joint_state_received_ = true;
  last_joint_state_time_ = stamp;
}

void OmyCartesianImpedanceControllerNode::moveLCallback(
  const robotis_interfaces::msg::MoveL::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || jointStateTimedOut()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      2000,
      "Ignoring moveL command until joint states are available.");
    return;
  }

  const double duration = commandDurationSeconds(msg->time_from_start);
  if (duration <= -1.0) {
    publishControllerError("moveL command ignored: time_from_start must be > -1.");
    return;
  }

  try {
    dynamics_solver_->updateState(q_, qdot_);
    const Eigen::Affine3d current_pose =
      dynamics_solver_->getKinematicsSolver()->getPose(controlled_link_);
    movel_start_pose_ =
      movel_target_initialized_ && movel_trajectory_active_ ? desired_pose_ : current_pose;
    movel_goal_pose_ = poseMsgToEigen(msg->pose);
    desired_pose_ = movel_start_pose_;
    active_motion_duration_ = duration;
    motion_start_time_ = this->now();
    movel_target_initialized_ = true;
    movel_trajectory_active_ = duration > 0.0;
  } catch (const std::exception & e) {
    publishControllerError("moveL command ignored: " + std::string(e.what()));
  }
}

void OmyCartesianImpedanceControllerNode::controlLoopCallback()
{
  if (!joint_state_received_ || jointStateTimedOut()) {
    return;
  }

  try {
    dynamics_solver_->updateState(q_, qdot_);
    const Eigen::Affine3d current_pose =
      dynamics_solver_->getKinematicsSolver()->getPose(controlled_link_);
    publishCurrentPose(current_pose);

    cyclo_motion_controller::common::Vector6d desired_twist =
      cyclo_motion_controller::common::Vector6d::Zero();

    if (!movel_target_initialized_) {
      desired_pose_ = current_pose;
      qp_controller_->setNullspaceReference(q_);
    } else {
      const double elapsed = (this->now() - motion_start_time_).seconds();
      if (movel_trajectory_active_ && active_motion_duration_ > 0.0) {
        const double sample_time = std::clamp(elapsed, 0.0, active_motion_duration_);
        desired_pose_.translation() =
          cyclo_motion_controller::common::math_utils::cubicVector<3>(
          sample_time,
          0.0,
          active_motion_duration_,
          movel_start_pose_.translation(),
          movel_goal_pose_.translation(),
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero());
        desired_pose_.linear() =
          cyclo_motion_controller::common::math_utils::rotationCubic(
          sample_time,
          0.0,
          active_motion_duration_,
          movel_start_pose_.linear(),
          movel_goal_pose_.linear());
        desired_twist.head<3>() =
          cyclo_motion_controller::common::math_utils::cubicDotVector<3>(
          sample_time,
          0.0,
          active_motion_duration_,
          movel_start_pose_.translation(),
          movel_goal_pose_.translation(),
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero());
        desired_twist.tail<3>() =
          cyclo_motion_controller::common::math_utils::rotationCubicDot(
          sample_time,
          0.0,
          active_motion_duration_,
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          movel_start_pose_.linear(),
          movel_goal_pose_.linear());

        if (elapsed >= active_motion_duration_) {
          movel_trajectory_active_ = false;
          desired_pose_ = movel_goal_pose_;
          desired_twist.setZero();
        }
      } else {
        desired_pose_ = movel_goal_pose_;
        desired_twist.setZero();
      }
    }
    qp_controller_->setDesiredPoseAndVelocity(desired_pose_, desired_twist);

    Eigen::VectorXd torque_command;
    if (!qp_controller_->getCommand(torque_command)) {
      publishControllerError("Cartesian impedance QP failed; publishing zero torque fallback.");
    }
    publishTorqueCommand(torque_command);
  } catch (const std::exception & e) {
    publishControllerError(e.what());
  }
}

void OmyCartesianImpedanceControllerNode::publishTorqueCommand(
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

void OmyCartesianImpedanceControllerNode::publishCurrentPose(const Eigen::Affine3d & pose) const
{
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header.stamp = this->now();
  pose_msg.header.frame_id = base_frame_;
  pose_msg.pose.position.x = pose.translation().x();
  pose_msg.pose.position.y = pose.translation().y();
  pose_msg.pose.position.z = pose.translation().z();

  const Eigen::Quaterniond quat(pose.linear());
  pose_msg.pose.orientation.w = quat.w();
  pose_msg.pose.orientation.x = quat.x();
  pose_msg.pose.orientation.y = quat.y();
  pose_msg.pose.orientation.z = quat.z();
  ee_pose_pub_->publish(pose_msg);
}

void OmyCartesianImpedanceControllerNode::publishControllerError(const std::string & error) const
{
  if (!controller_error_pub_) {
    return;
  }
  std_msgs::msg::String msg;
  msg.data = error;
  controller_error_pub_->publish(msg);
  RCLCPP_WARN(this->get_logger(), "%s", error.c_str());
}

bool OmyCartesianImpedanceControllerNode::jointStateTimedOut() const
{
  return (this->now() - last_joint_state_time_).seconds() > joint_state_timeout_;
}

Eigen::VectorXd OmyCartesianImpedanceControllerNode::filterJointVelocity(
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

Eigen::Affine3d OmyCartesianImpedanceControllerNode::poseMsgToEigen(
  const geometry_msgs::msg::PoseStamped & pose_msg) const
{
  Eigen::Affine3d pose = Eigen::Affine3d::Identity();
  pose.translation() << pose_msg.pose.position.x,
    pose_msg.pose.position.y,
    pose_msg.pose.position.z;

  Eigen::Quaterniond quat(
    pose_msg.pose.orientation.w,
    pose_msg.pose.orientation.x,
    pose_msg.pose.orientation.y,
    pose_msg.pose.orientation.z);
  if (quat.norm() < 1e-9) {
    quat.setIdentity();
  } else {
    quat.normalize();
  }
  pose.linear() = quat.toRotationMatrix();
  return pose;
}

Eigen::VectorXd OmyCartesianImpedanceControllerNode::vectorFromParameter(
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

cyclo_motion_controller::common::Vector6d OmyCartesianImpedanceControllerNode::vector6FromParameter(
  const std::string & name,
  double default_value) const
{
  std::vector<double> values;
  if (!this->get_parameter(name, values) || values.empty()) {
    return cyclo_motion_controller::common::Vector6d::Constant(default_value);
  }
  if (values.size() == 1) {
    return cyclo_motion_controller::common::Vector6d::Constant(values.front());
  }
  if (values.size() != 6) {
    RCLCPP_WARN(
      this->get_logger(), "Parameter '%s' has %zu values, expected 6. Using default.",
      name.c_str(), values.size());
    return cyclo_motion_controller::common::Vector6d::Constant(default_value);
  }

  cyclo_motion_controller::common::Vector6d result;
  for (int i = 0; i < 6; ++i) {
    result[i] = values[static_cast<size_t>(i)];
  }
  return result;
}

}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<cyclo_motion_controller_ros::OmyCartesianImpedanceControllerNode>());
  rclcpp::shutdown();
  return 0;
}
