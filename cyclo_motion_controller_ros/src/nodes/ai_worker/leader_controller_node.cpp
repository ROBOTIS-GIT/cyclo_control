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

#include "cyclo_motion_controller_ros/nodes/ai_worker/leader_controller_node.hpp"

#include <algorithm>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

namespace cyclo_motion_controller_ros
{
LeaderController::LeaderController()
: Node("leader_controller"),
  right_traj_received_(false),
  left_traj_received_(false),
  lift_joint_received_(false),
  last_right_traj_time_(this->now()),
  last_left_traj_time_(this->now()),
  lift_joint_index_(-1)
{
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "Leader Controller - Starting up...");
  RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
  RCLCPP_INFO(this->get_logger(), "========================================");

  control_frequency_ = this->declare_parameter("control_frequency", 100.0);
  time_step_ = this->declare_parameter("time_step", 0.01);
  trajectory_time_ = this->declare_parameter("trajectory_time", 0.0);
  kp_position_ = this->declare_parameter("kp_position", 50.0);
  kp_orientation_ = this->declare_parameter("kp_orientation", 50.0);
  weight_position_ = this->declare_parameter("weight_position", 10.0);
  weight_orientation_ = this->declare_parameter("weight_orientation", 1.0);
  weight_damping_ = this->declare_parameter("weight_damping", 0.1);
  elbow_up_velocity_ = this->declare_parameter("elbow_up_velocity", 0.2);
  elbow_nullspace_weight_ = this->declare_parameter("elbow_nullspace_weight", 1.0);
  elbow_nullspace_damping_ = this->declare_parameter("elbow_nullspace_damping", 0.001);
  elbow_nullspace_max_joint_velocity_ = this->declare_parameter(
    "elbow_nullspace_max_joint_velocity", 1.0);
  slack_penalty_ = this->declare_parameter("slack_penalty", 1000.0);
  cbf_alpha_ = this->declare_parameter("cbf_alpha", 50.0);
  collision_buffer_ = this->declare_parameter("collision_buffer", 0.05);
  collision_safe_distance_ = this->declare_parameter("collision_safe_distance", 0.02);
  urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
  srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
  follower_urdf_path_ = this->declare_parameter("follower_urdf_path", std::string(""));
  follower_srdf_path_ = this->declare_parameter("follower_srdf_path", std::string(""));
  joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
  right_traj_topic_ = this->declare_parameter(
            "right_traj_topic",
            std::string("/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory"));
  left_traj_topic_ = this->declare_parameter(
            "left_traj_topic",
            std::string("/leader/joint_trajectory_command_broadcaster_left/raw_joint_trajectory"));
  right_command_topic_ = this->declare_parameter(
    "right_command_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"));
  left_command_topic_ = this->declare_parameter(
    "left_command_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"));
  right_teleop_enable_topic_ = this->declare_parameter(
    "right_teleop_enable_topic", std::string("/right_relative_teleop_enable"));
  left_teleop_enable_topic_ = this->declare_parameter(
    "left_teleop_enable_topic", std::string("/left_relative_teleop_enable"));
  command_timeout_ = this->declare_parameter("command_timeout", 0.1);
  r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
  l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));
  base_frame_id_ = this->declare_parameter("base_frame_id", std::string("base_link"));
  r_gripper_name_ = this->declare_parameter("r_gripper_name", std::string("arm_r_link7"));
  l_gripper_name_ = this->declare_parameter("l_gripper_name", std::string("arm_l_link7"));
  follower_r_gripper_name_ = this->declare_parameter(
    "follower_r_gripper_name", std::string("end_effector_r_link"));
  follower_l_gripper_name_ = this->declare_parameter(
    "follower_l_gripper_name", std::string("end_effector_l_link"));
  r_elbow_name_ = this->declare_parameter("r_elbow_name", std::string("arm_r_link4"));
  l_elbow_name_ = this->declare_parameter("l_elbow_name", std::string("arm_l_link4"));
  lift_joint_name_ = this->declare_parameter("lift_joint_name", std::string("lift_joint"));
  model_lift_joint_name_ = this->declare_parameter("model_lift_joint_name", std::string("joint"));

  r_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            right_traj_topic_, 10,
            std::bind(&LeaderController::rightTrajectoryCallback, this, std::placeholders::_1));
  l_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            left_traj_topic_, 10,
            std::bind(&LeaderController::leftTrajectoryCallback, this, std::placeholders::_1));
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic_, 10,
            std::bind(&LeaderController::jointStateCallback, this, std::placeholders::_1));
  right_teleop_enable_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    right_teleop_enable_topic_, 10,
    std::bind(&LeaderController::rightTeleopEnableCallback, this, std::placeholders::_1));
  left_teleop_enable_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    left_teleop_enable_topic_, 10,
    std::bind(&LeaderController::leftTeleopEnableCallback, this, std::placeholders::_1));

  r_goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            r_goal_pose_topic_, 10);
  l_goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            l_goal_pose_topic_, 10);
  arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            right_command_topic_, 10);
  arm_l_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            left_command_topic_, 10);

  RCLCPP_INFO(
    this->get_logger(), "Relative teleoperation enable topics: right=%s, left=%s",
    right_teleop_enable_topic_.c_str(), left_teleop_enable_topic_.c_str());

  try {
    if (urdf_path_.empty() || follower_urdf_path_.empty()) {
      throw std::runtime_error("Leader and follower URDF paths must be provided.");
    }
    RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path_.c_str());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to resolve robot model paths: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  try {
    if (srdf_path_.empty()) {
      RCLCPP_INFO(this->get_logger(), "SRDF path not provided. Continuing without SRDF.");
    } else {
      RCLCPP_INFO(this->get_logger(), "SRDF path: %s", srdf_path_.c_str());
    }
    RCLCPP_INFO(this->get_logger(), "Loading URDF and initializing kinematics solver...");
    leader_kinematics_ =
      std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(urdf_path_,
        srdf_path_);
    follower_kinematics_ =
      std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(
      follower_urdf_path_, follower_srdf_path_);
    qp_controller_ =
      std::make_shared<cyclo_motion_controller::controllers::ElbowUpQPController>(
      follower_kinematics_);
    qp_controller_->setControllerParams(
      slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);

            // Initialize state variables
    const int dof = leader_kinematics_->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);
    follower_q_.setZero(follower_kinematics_->getDof());
    follower_qdot_.setZero(follower_kinematics_->getDof());
    follower_q_desired_.setZero(follower_kinematics_->getDof());
    RCLCPP_INFO(this->get_logger(), "Kinematics solver initialized (DOF: %d)", dof);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize kinematics solver: %s", e.what());
    rclcpp::shutdown();
    return;
  }

        // Initialize joint configuration from URDF
  initializeJointConfig();

  const int timer_period_ms = static_cast<int>(1000.0 / control_frequency_);
  control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_period_ms),
            std::bind(&LeaderController::controlLoopCallback, this));

  if (!control_timer_) {
    RCLCPP_FATAL(this->get_logger(), "Failed to create control loop timer!");
    rclcpp::shutdown();
    return;
  }

  RCLCPP_INFO(this->get_logger(),
            "Leader Controller initialized successfully!");
  RCLCPP_INFO(this->get_logger(),
            "  - Control loop: %.1f Hz (period: %d ms)", control_frequency_, timer_period_ms);
  RCLCPP_INFO(this->get_logger(),
            "  - Subscriptions: joint_states=%s",
            joint_state_sub_ ? "OK" : "FAILED");
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "Node is ready! Waiting for messages...");
  RCLCPP_INFO(
    this->get_logger(),
    "Both arms start in hold mode; use each relative teleoperation enable topic to move them.");
}

LeaderController::~LeaderController()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down Leader Controller");
}

void LeaderController::initializeJointConfig()
{
  const auto joint_names = leader_kinematics_->getJointNames();
  model_joint_index_map_.clear();
  for (size_t i = 0; i < joint_names.size(); ++i) {
    model_joint_index_map_[joint_names[i]] = static_cast<int>(i);
  }

  auto it = model_joint_index_map_.find(model_lift_joint_name_);
  if (it != model_joint_index_map_.end()) {
    lift_joint_index_ = it->second;
  } else {
    RCLCPP_ERROR(this->get_logger(),
                "Model lift joint '%s' not found in URDF.", model_lift_joint_name_.c_str());
  }

  follower_joint_names_ = follower_kinematics_->getJointNames();
  follower_joint_index_map_.clear();
  right_arm_joints_.clear();
  left_arm_joints_.clear();
  for (size_t i = 0; i < follower_joint_names_.size(); ++i) {
    const auto & name = follower_joint_names_[i];
    follower_joint_index_map_[name] = static_cast<int>(i);
    if (name.find("arm_r_joint") != std::string::npos) {
      right_arm_joints_.push_back(name);
    } else if (name.find("arm_l_joint") != std::string::npos) {
      left_arm_joints_.push_back(name);
    } else if (name.find("lift_joint") != std::string::npos) {
      follower_lift_joint_index_ = static_cast<int>(i);
    }
  }
  std::sort(right_arm_joints_.begin(), right_arm_joints_.end());
  std::sort(left_arm_joints_.begin(), left_arm_joints_.end());
  if (follower_lift_joint_index_ >= 0) {
    follower_kinematics_->setJointVelocityBoundsByIndex(follower_lift_joint_index_, 0.0, 0.0);
  }
}

void LeaderController::rightTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  updateJointPositionsFromTrajectory(*msg);
  right_traj_received_ = true;
  last_right_traj_time_ = this->now();
}

void LeaderController::leftTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  updateJointPositionsFromTrajectory(*msg);
  left_traj_received_ = true;
  last_left_traj_time_ = this->now();
}

void LeaderController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  updateLiftJointFromJointState(*msg);
  updateFollowerJointState(*msg);
}

void LeaderController::updateFollowerJointState(const sensor_msgs::msg::JointState & msg)
{
  if (joint_state_index_map_.empty()) {
    for (size_t i = 0; i < msg.name.size(); ++i) {
      joint_state_index_map_[msg.name[i]] = static_cast<int>(i);
    }
  }
  bool complete = true;
  for (size_t i = 0; i < follower_joint_names_.size(); ++i) {
    const auto iter = joint_state_index_map_.find(follower_joint_names_[i]);
    if (iter == joint_state_index_map_.end() ||
      iter->second >= static_cast<int>(msg.position.size()))
    {
      complete = false;
      continue;
    }
    follower_q_[i] = msg.position[iter->second];
    if (iter->second < static_cast<int>(msg.velocity.size())) {
      follower_qdot_[i] = msg.velocity[iter->second];
    }
  }
  follower_joint_state_received_ = complete;
  if (!follower_command_initialized_) {
    follower_q_desired_ = follower_q_;
    follower_command_initialized_ = follower_joint_state_received_;
  }
}

void LeaderController::updateJointPositionsFromTrajectory(
  const trajectory_msgs::msg::JointTrajectory & msg)
{
  if (msg.points.empty()) {
    return;
  }
  const auto & point = msg.points.front();
  if (point.positions.empty()) {
    return;
  }

  for (size_t i = 0; i < msg.joint_names.size(); ++i) {
    auto it = model_joint_index_map_.find(msg.joint_names[i]);
    if (it == model_joint_index_map_.end()) {
      continue;
    }
    const int model_index = it->second;
    if (model_index < 0 || model_index >= q_.size()) {
      continue;
    }
    if (i < point.positions.size()) {
      q_[model_index] = point.positions[i];
    }
    if (i < point.velocities.size()) {
      qdot_[model_index] = point.velocities[i];
    }
  }
}

void LeaderController::updateLiftJointFromJointState(const sensor_msgs::msg::JointState & msg)
{
  if (lift_joint_index_ < 0 || lift_joint_index_ >= q_.size()) {
    return;
  }

  for (size_t i = 0; i < msg.name.size(); ++i) {
    if (msg.name[i] != lift_joint_name_) {
      continue;
    }
    if (i < msg.position.size()) {
      q_[lift_joint_index_] = msg.position[i];
      lift_joint_received_ = true;
    }
    if (i < msg.velocity.size()) {
      qdot_[lift_joint_index_] = msg.velocity[i];
    }
    return;
  }
}

void LeaderController::rightTeleopEnableCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (!msg || msg->data == right_teleop_enabled_) {
    return;
  }
  right_teleop_enabled_ = msg->data;
  right_mode_transition_pending_ = true;
  RCLCPP_INFO(
    this->get_logger(), "Right relative teleoperation %s",
    right_teleop_enabled_ ? "requested" : "stopped; holding current pose");
}

void LeaderController::leftTeleopEnableCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (!msg || msg->data == left_teleop_enabled_) {
    return;
  }
  left_teleop_enabled_ = msg->data;
  left_mode_transition_pending_ = true;
  RCLCPP_INFO(
    this->get_logger(), "Left relative teleoperation %s",
    left_teleop_enabled_ ? "requested" : "stopped; holding current pose");
}

void LeaderController::controlLoopCallback()
{
  const rclcpp::Time now = this->now();
  const bool right_traj_has_publisher =
    (r_traj_sub_ && r_traj_sub_->get_publisher_count() > 0);
  const bool left_traj_has_publisher =
    (l_traj_sub_ && l_traj_sub_->get_publisher_count() > 0);

  const bool right_recent =
    right_traj_has_publisher && right_traj_received_ &&
    (now - last_right_traj_time_).seconds() < command_timeout_;
  const bool left_recent =
    left_traj_has_publisher && left_traj_received_ &&
    (now - last_left_traj_time_).seconds() < command_timeout_;
  if (!follower_joint_state_received_ || !follower_command_initialized_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Waiting for complete follower joint state before solving IK");
    return;
  }

  was_publishing_reference_ = true;

  try {
    leader_kinematics_->updateState(q_, qdot_);
    const Eigen::Affine3d right_leader_current =
      computePoseInBaseFrame(leader_kinematics_->getPose(r_gripper_name_));
    const Eigen::Affine3d left_leader_current =
      computePoseInBaseFrame(leader_kinematics_->getPose(l_gripper_name_));

    Eigen::VectorXd model_state = follower_q_desired_;
    if (follower_lift_joint_index_ >= 0) {
      model_state[follower_lift_joint_index_] = follower_q_[follower_lift_joint_index_];
    }
    follower_kinematics_->updateState(model_state, follower_qdot_);
    const Eigen::Affine3d right_current = follower_kinematics_->getPose(follower_r_gripper_name_);
    const Eigen::Affine3d left_current = follower_kinematics_->getPose(follower_l_gripper_name_);

    if (right_mode_transition_pending_) {
      if (!right_teleop_enabled_) {
        right_goal_ = right_current;
        right_goal_initialized_ = true;
        right_mode_transition_pending_ = false;
      } else if (right_recent) {
        right_leader_anchor_ = right_leader_current;
        right_follower_anchor_ = right_current;
        right_goal_ = right_current;
        right_goal_initialized_ = true;
        right_mode_transition_pending_ = false;
        RCLCPP_INFO(this->get_logger(), "Right relative teleoperation started");
      }
    }
    if (left_mode_transition_pending_) {
      if (!left_teleop_enabled_) {
        left_goal_ = left_current;
        left_goal_initialized_ = true;
        left_mode_transition_pending_ = false;
      } else if (left_recent) {
        left_leader_anchor_ = left_leader_current;
        left_follower_anchor_ = left_current;
        left_goal_ = left_current;
        left_goal_initialized_ = true;
        left_mode_transition_pending_ = false;
        RCLCPP_INFO(this->get_logger(), "Left relative teleoperation started");
      }
    }

    if (right_teleop_enabled_ && !right_mode_transition_pending_ && right_recent) {
      right_goal_ = right_follower_anchor_ *
        right_leader_anchor_.inverse() * right_leader_current;
    }
    if (left_teleop_enabled_ && !left_mode_transition_pending_ && left_recent) {
      left_goal_ = left_follower_anchor_ *
        left_leader_anchor_.inverse() * left_leader_current;
    }
    if (right_teleop_enabled_ && !right_recent) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Right leader command is stale; holding the last right goal");
    }
    if (left_teleop_enabled_ && !left_recent) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Left leader command is stale; holding the last left goal");
    }
    if (!right_goal_initialized_ || !left_goal_initialized_) {
      return;
    }

    r_goal_pose_pub_->publish(makePoseStamped(right_goal_));
    l_goal_pose_pub_->publish(makePoseStamped(left_goal_));

    std::map<std::string, cyclo_motion_controller::common::Vector6d> desired;
    desired[follower_r_gripper_name_] = computeDesiredVelocity(right_current, right_goal_);
    desired[follower_l_gripper_name_] = computeDesiredVelocity(left_current, left_goal_);
    std::map<std::string, cyclo_motion_controller::common::Vector6d> weights;
    cyclo_motion_controller::common::Vector6d task_weight =
      cyclo_motion_controller::common::Vector6d::Ones();
    task_weight.head<3>().setConstant(weight_position_);
    task_weight.tail<3>().setConstant(weight_orientation_);
    weights[follower_r_gripper_name_] = task_weight;
    weights[follower_l_gripper_name_] = task_weight;
    qp_controller_->setDesiredTaskVel(desired);
    qp_controller_->setWeights(
      weights, Eigen::VectorXd::Constant(follower_kinematics_->getDof(), weight_damping_));
    qp_controller_->setPreferredJointVel(
      computeElbowUpPreferredJointVelocity(), elbow_nullspace_weight_);

    Eigen::VectorXd optimal_velocity;
    if (!qp_controller_->getOptJointVel(optimal_velocity)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000, "Elbow-up leader QP failed");
      return;
    }
    follower_q_desired_ = model_state + optimal_velocity * time_step_;
    publishFollowerTrajectory(follower_q_desired_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Leader FK/follower IK failed: %s", e.what());
  }
}

cyclo_motion_controller::common::Vector6d LeaderController::computeDesiredVelocity(
  const Eigen::Affine3d & current, const Eigen::Affine3d & goal) const
{
  cyclo_motion_controller::common::Vector6d velocity =
    cyclo_motion_controller::common::Vector6d::Zero();
  velocity.head<3>() = kp_position_ * (goal.translation() - current.translation());
  const Eigen::Matrix3d rotation_error = goal.linear() * current.linear().transpose();
  const Eigen::AngleAxisd angle_axis(rotation_error);
  velocity.tail<3>() = kp_orientation_ * angle_axis.axis() * angle_axis.angle();
  return velocity;
}

Eigen::VectorXd LeaderController::computeElbowUpPreferredJointVelocity() const
{
  const int dof = follower_kinematics_->getDof();
  const Eigen::MatrixXd right_ee = follower_kinematics_->getJacobian(
    follower_r_gripper_name_);
  const Eigen::MatrixXd left_ee = follower_kinematics_->getJacobian(
    follower_l_gripper_name_);
  const Eigen::MatrixXd right_elbow = follower_kinematics_->getJacobian(r_elbow_name_);
  const Eigen::MatrixXd left_elbow = follower_kinematics_->getJacobian(l_elbow_name_);
  Eigen::MatrixXd ee(12, dof);
  ee.topRows(6) = right_ee;
  ee.bottomRows(6) = left_ee;
  if (follower_lift_joint_index_ >= 0) {
    ee.col(follower_lift_joint_index_).setZero();
  }
  const double damping_squared = elbow_nullspace_damping_ * elbow_nullspace_damping_;
  const Eigen::MatrixXd regularized = ee * ee.transpose() +
    damping_squared * Eigen::MatrixXd::Identity(12, 12);
  const Eigen::MatrixXd nullspace = Eigen::MatrixXd::Identity(dof, dof) -
    ee.transpose() * regularized.ldlt().solve(ee);

  auto arm_preference = [this, &nullspace](
    const Eigen::MatrixXd & elbow) -> Eigen::VectorXd {
      Eigen::VectorXd gradient = elbow.row(2).transpose();
      if (follower_lift_joint_index_ >= 0) {
        gradient[follower_lift_joint_index_] = 0.0;
      }
      Eigen::VectorXd direction = nullspace * gradient;
      const double attainable = elbow.row(2).dot(direction);
      if (attainable <= 1.0e-8 || elbow_up_velocity_ <= 0.0) {
        return Eigen::VectorXd::Zero(direction.size());
      }
      Eigen::VectorXd preferred = direction * (elbow_up_velocity_ / attainable);
      if (elbow_nullspace_max_joint_velocity_ > 0.0 &&
        preferred.norm() > elbow_nullspace_max_joint_velocity_)
      {
        preferred *= elbow_nullspace_max_joint_velocity_ / preferred.norm();
      }
      return preferred;
    };
  Eigen::VectorXd preferred = arm_preference(right_elbow) + arm_preference(left_elbow);
  if (follower_lift_joint_index_ >= 0) {
    preferred[follower_lift_joint_index_] = 0.0;
  }
  return preferred;
}

void LeaderController::publishFollowerTrajectory(const Eigen::VectorXd & desired)
{
  arm_r_pub_->publish(makeArmTrajectory(right_arm_joints_, desired));
  arm_l_pub_->publish(makeArmTrajectory(left_arm_joints_, desired));
}

trajectory_msgs::msg::JointTrajectory LeaderController::makeArmTrajectory(
  const std::vector<std::string> & joint_names,
  const Eigen::VectorXd & desired) const
{
  trajectory_msgs::msg::JointTrajectory trajectory;
  trajectory.joint_names = joint_names;
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);
  for (const auto & name : joint_names) {
    const auto iter = follower_joint_index_map_.find(name);
    if (iter != follower_joint_index_map_.end()) {
      point.positions.push_back(desired[iter->second]);
    }
  }
  trajectory.points.push_back(point);
  return trajectory;
}

geometry_msgs::msg::PoseStamped LeaderController::makePoseStamped(
  const Eigen::Affine3d & pose) const
{
  geometry_msgs::msg::PoseStamped msg;
        // msg.header.stamp = this->now();
  msg.header.frame_id = base_frame_id_;
  msg.pose.position.x = pose.translation().x();
  msg.pose.position.y = pose.translation().y();
  msg.pose.position.z = pose.translation().z();

  const Eigen::Quaterniond quat(pose.linear());
  msg.pose.orientation.w = quat.w();
  msg.pose.orientation.x = quat.x();
  msg.pose.orientation.y = quat.y();
  msg.pose.orientation.z = quat.z();
  return msg;
}

Eigen::Affine3d LeaderController::computePoseInBaseFrame(
  const Eigen::Affine3d & link_pose) const
{
  if (leader_kinematics_ && leader_kinematics_->hasLinkFrame("world")) {
    const Eigen::Affine3d base_pose = leader_kinematics_->getPose("world");
    return base_pose.inverse() * link_pose;
  }
  return link_pose;
}
}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyclo_motion_controller_ros::LeaderController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
