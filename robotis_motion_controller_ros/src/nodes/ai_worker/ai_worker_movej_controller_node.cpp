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

#include "robotis_motion_controller_ros/nodes/ai_worker/ai_worker_movej_controller_node.hpp"

#include <algorithm>

namespace robotis_motion_controller_ros
{
AIWorkerMoveJController::AIWorkerMoveJController()
: Node("ai_worker_movej_controller"),
  joint_state_received_(false),
  commanded_state_initialized_(false),
  right_movej_target_initialized_(false),
  left_movej_target_initialized_(false),
  right_movej_trajectory_active_(false),
  left_movej_trajectory_active_(false),
  right_gripper_position_(0.0),
  left_gripper_position_(0.0),
  right_motion_start_time_(this->now()),
  left_motion_start_time_(this->now()),
  right_active_motion_duration_(0.0),
  left_active_motion_duration_(0.0)
{
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "AI Worker MoveJ Controller - Starting up...");
  RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
  RCLCPP_INFO(this->get_logger(), "========================================");

  control_frequency_ = this->declare_parameter("control_frequency", 100.0);
  time_step_ = this->declare_parameter("time_step", 0.01);
  trajectory_time_ = this->declare_parameter("trajectory_time", 0.05);
  kp_joint_ = this->declare_parameter("kp_joint", 6.0);
  weight_tracking_ = this->declare_parameter("weight_tracking", 1.0);
  weight_damping_ = this->declare_parameter("weight_damping", 1.0);
  slack_penalty_ = this->declare_parameter("slack_penalty", 1000.0);
  cbf_alpha_ = this->declare_parameter("cbf_alpha", 5.0);
  collision_buffer_ = this->declare_parameter("collision_buffer", 0.05);
  collision_safe_distance_ = this->declare_parameter("collision_safe_distance", 0.02);
  urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
  srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
  joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
  right_traj_topic_ = this->declare_parameter(
    "right_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory"));
  left_traj_topic_ = this->declare_parameter(
    "left_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_left/raw_joint_trajectory"));
  right_traj_filtered_topic_ = this->declare_parameter(
    "right_traj_filtered_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"));
  left_traj_filtered_topic_ = this->declare_parameter(
    "left_traj_filtered_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"));
  right_gripper_joint_name_ = this->declare_parameter(
    "right_gripper_joint", std::string("gripper_r_joint1"));
  left_gripper_joint_name_ = this->declare_parameter(
    "left_gripper_joint", std::string("gripper_l_joint1"));

  if (urdf_path_.empty()) {
    RCLCPP_FATAL(this->get_logger(), "URDF path not provided.");
    rclcpp::shutdown();
    return;
  }

  arm_r_pub_ =
    this->create_publisher<trajectory_msgs::msg::JointTrajectory>(right_traj_filtered_topic_, 10);
  arm_l_pub_ =
    this->create_publisher<trajectory_msgs::msg::JointTrajectory>(left_traj_filtered_topic_, 10);

  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    joint_states_topic_, 10,
    std::bind(&AIWorkerMoveJController::jointStateCallback, this, std::placeholders::_1));
  r_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    right_traj_topic_, 10,
    std::bind(&AIWorkerMoveJController::rightTrajectoryCallback, this, std::placeholders::_1));
  l_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    left_traj_topic_, 10,
    std::bind(&AIWorkerMoveJController::leftTrajectoryCallback, this, std::placeholders::_1));

  try {
    RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path_.c_str());
    if (srdf_path_.empty()) {
      RCLCPP_INFO(this->get_logger(), "SRDF path not provided. Continuing without SRDF.");
    } else {
      RCLCPP_INFO(this->get_logger(), "SRDF path: %s", srdf_path_.c_str());
    }

    kinematics_solver_ =
      std::make_shared<robotis_motion_controller::kinematics::KinematicsSolver>(urdf_path_,
        srdf_path_);
    qp_filter_ = std::make_shared<robotis_motion_controller::controllers::AIWorkerMoveJController>(
      kinematics_solver_, time_step_);
    qp_filter_->setControllerParams(
      slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);

    const int dof = kinematics_solver_->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);
    q_commanded_.setZero(dof);
    right_movej_start_.setZero(dof);
    right_movej_goal_.setZero(dof);
    left_movej_start_.setZero(dof);
    left_movej_goal_.setZero(dof);

    initializeJointConfig();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize filter: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  const int timer_period_ms =
    std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, control_frequency_))));
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(timer_period_ms),
    std::bind(&AIWorkerMoveJController::controlLoopCallback, this));

  RCLCPP_INFO(this->get_logger(), "AI Worker MoveJ Controller initialized successfully!");
}

AIWorkerMoveJController::~AIWorkerMoveJController()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down AI Worker MoveJ Controller");
}

void AIWorkerMoveJController::initializeJointConfig()
{
  model_joint_names_ = kinematics_solver_->getJointNames();
  model_joint_index_map_.clear();
  for (size_t i = 0; i < model_joint_names_.size(); ++i) {
    model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
  }

  left_arm_joints_.clear();
  right_arm_joints_.clear();
  for (const auto & joint_name : model_joint_names_) {
    if (joint_name.find("arm_l_joint") != std::string::npos) {
      left_arm_joints_.push_back(joint_name);
    } else if (joint_name.find("arm_r_joint") != std::string::npos) {
      right_arm_joints_.push_back(joint_name);
    }
  }

  std::sort(left_arm_joints_.begin(), left_arm_joints_.end());
  std::sort(right_arm_joints_.begin(), right_arm_joints_.end());
}

void AIWorkerMoveJController::extractJointStates(
  const sensor_msgs::msg::JointState::SharedPtr & msg)
{
  const int dof = kinematics_solver_->getDof();
  q_.setZero(dof);
  qdot_.setZero(dof);

  const int max_index = std::min<int>(dof, static_cast<int>(model_joint_names_.size()));
  for (int i = 0; i < max_index; ++i) {
    const auto & joint_name = model_joint_names_[i];
    const auto it = joint_index_map_.find(joint_name);
    if (it == joint_index_map_.end()) {
      continue;
    }
    const int msg_idx = it->second;
    if (msg_idx < static_cast<int>(msg->position.size())) {
      q_[i] = msg->position[msg_idx];
    }
    if (msg_idx < static_cast<int>(msg->velocity.size())) {
      qdot_[i] = msg->velocity[msg_idx];
    }
  }
}

void AIWorkerMoveJController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (joint_index_map_.empty()) {
    for (size_t i = 0; i < msg->name.size(); ++i) {
      joint_index_map_[msg->name[i]] = static_cast<int>(i);
    }
  }

  extractJointStates(msg);
  joint_state_received_ = true;

  if (!commanded_state_initialized_) {
    q_commanded_ = q_;
    right_movej_start_ = q_;
    right_movej_goal_ = q_;
    left_movej_start_ = q_;
    left_movej_goal_ = q_;
    commanded_state_initialized_ = true;
    right_movej_target_initialized_ = true;
    left_movej_target_initialized_ = true;
    RCLCPP_INFO(
      this->get_logger(),
      "AI Worker MoveJ Controller activated. Waiting for moveJ commands...");
  }
}

void AIWorkerMoveJController::updateGripperPositionFromTrajectory(
  const trajectory_msgs::msg::JointTrajectory & msg,
  const std::string & gripper_joint_name,
  double & gripper_position) const
{
  if (msg.points.empty() || msg.points.front().positions.empty()) {
    return;
  }

  const auto & point = msg.points.front();
  for (size_t i = 0; i < msg.joint_names.size(); ++i) {
    if (msg.joint_names[i] != gripper_joint_name) {
      continue;
    }
    if (i < point.positions.size()) {
      gripper_position = point.positions[i];
    }
    return;
  }
}

bool AIWorkerMoveJController::updateArmTargetFromTrajectory(
  const trajectory_msgs::msg::JointTrajectory & msg,
  const std::vector<std::string> & arm_joint_names,
  const std::string & arm_name,
  Eigen::VectorXd & target_q) const
{
  if (msg.points.empty()) {
    return false;
  }

  const auto & point = msg.points.front();
  if (point.positions.empty()) {
    RCLCPP_WARN(this->get_logger(), "%s moveJ ignored: positions are empty.", arm_name.c_str());
    return false;
  }

  if (!msg.joint_names.empty()) {
    for (size_t i = 0; i < msg.joint_names.size(); ++i) {
      if (i >= point.positions.size()) {
        continue;
      }
      const auto it = model_joint_index_map_.find(msg.joint_names[i]);
      if (it == model_joint_index_map_.end()) {
        continue;
      }
      target_q[it->second] = point.positions[i];
    }
    return true;
  }

  if (point.positions.size() == arm_joint_names.size()) {
    for (size_t i = 0; i < arm_joint_names.size(); ++i) {
      const auto model_it = model_joint_index_map_.find(arm_joint_names[i]);
      if (model_it == model_joint_index_map_.end()) {
        continue;
      }
      target_q[model_it->second] = point.positions[i];
    }
    return true;
  }

  RCLCPP_WARN(
    this->get_logger(),
    "%s moveJ ignored: joint_names missing and positions size does not match arm joints.",
    arm_name.c_str());
  return false;
}

void AIWorkerMoveJController::rightTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || !commanded_state_initialized_ || msg->points.empty()) {
    return;
  }

  const auto duration = rclcpp::Duration(msg->points.front().time_from_start).seconds();
  if (duration <= 1e-6) {
    RCLCPP_WARN(this->get_logger(), "Right moveJ ignored: time_from_start must be > 0.");
    return;
  }

  Eigen::VectorXd target_q = q_commanded_;
  if (!updateArmTargetFromTrajectory(*msg, right_arm_joints_, "Right", target_q)) {
    return;
  }

  updateGripperPositionFromTrajectory(*msg, right_gripper_joint_name_, right_gripper_position_);
  right_movej_start_ = q_commanded_;
  right_movej_goal_ = target_q;
  right_motion_start_time_ = this->now();
  right_active_motion_duration_ = duration;
  right_movej_target_initialized_ = true;
  right_movej_trajectory_active_ = true;
}

void AIWorkerMoveJController::leftTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || !commanded_state_initialized_ || msg->points.empty()) {
    return;
  }

  const auto duration = rclcpp::Duration(msg->points.front().time_from_start).seconds();
  if (duration <= 1e-6) {
    RCLCPP_WARN(this->get_logger(), "Left moveJ ignored: time_from_start must be > 0.");
    return;
  }

  Eigen::VectorXd target_q = q_commanded_;
  if (!updateArmTargetFromTrajectory(*msg, left_arm_joints_, "Left", target_q)) {
    return;
  }

  updateGripperPositionFromTrajectory(*msg, left_gripper_joint_name_, left_gripper_position_);
  left_movej_start_ = q_commanded_;
  left_movej_goal_ = target_q;
  left_motion_start_time_ = this->now();
  left_active_motion_duration_ = duration;
  left_movej_target_initialized_ = true;
  left_movej_trajectory_active_ = true;
}

void AIWorkerMoveJController::assignArmSegment(
  const Eigen::VectorXd & source,
  const std::vector<std::string> & arm_joint_names,
  Eigen::VectorXd & destination) const
{
  for (const auto & joint_name : arm_joint_names) {
    const auto it = model_joint_index_map_.find(joint_name);
    if (it == model_joint_index_map_.end()) {
      continue;
    }
    destination[it->second] = source[it->second];
  }
}

void AIWorkerMoveJController::controlLoopCallback()
{
  if (!joint_state_received_ || !commanded_state_initialized_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000, "Control loop waiting for joint states...");
    return;
  }

  try {
    const Eigen::VectorXd q_feedback = q_commanded_;
    kinematics_solver_->updateState(q_feedback, qdot_);

    Eigen::VectorXd q_ref = q_feedback;
    Eigen::VectorXd qdot_ref = Eigen::VectorXd::Zero(q_feedback.size());

    if (right_movej_target_initialized_) {
      const double elapsed = (this->now() - right_motion_start_time_).seconds();
      Eigen::VectorXd right_ref = right_movej_goal_;
      Eigen::VectorXd right_qdot = Eigen::VectorXd::Zero(q_feedback.size());

      if (right_movej_trajectory_active_ && elapsed < right_active_motion_duration_) {
        const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(q_feedback.size());
        right_ref = robotis_motion_controller::common::math_utils::cubicVector(
          elapsed, 0.0, right_active_motion_duration_,
          right_movej_start_, right_movej_goal_, zeros, zeros);
        right_qdot = robotis_motion_controller::common::math_utils::cubicDotVector(
          elapsed, 0.0, right_active_motion_duration_,
          right_movej_start_, right_movej_goal_, zeros, zeros);
      } else if (right_movej_trajectory_active_) {
        right_movej_trajectory_active_ = false;
      }

      assignArmSegment(right_ref, right_arm_joints_, q_ref);
      assignArmSegment(right_qdot, right_arm_joints_, qdot_ref);
    }

    if (left_movej_target_initialized_) {
      const double elapsed = (this->now() - left_motion_start_time_).seconds();
      Eigen::VectorXd left_ref = left_movej_goal_;
      Eigen::VectorXd left_qdot = Eigen::VectorXd::Zero(q_feedback.size());

      if (left_movej_trajectory_active_ && elapsed < left_active_motion_duration_) {
        const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(q_feedback.size());
        left_ref = robotis_motion_controller::common::math_utils::cubicVector(
          elapsed, 0.0, left_active_motion_duration_,
          left_movej_start_, left_movej_goal_, zeros, zeros);
        left_qdot = robotis_motion_controller::common::math_utils::cubicDotVector(
          elapsed, 0.0, left_active_motion_duration_,
          left_movej_start_, left_movej_goal_, zeros, zeros);
      } else if (left_movej_trajectory_active_) {
        left_movej_trajectory_active_ = false;
      }

      assignArmSegment(left_ref, left_arm_joints_, q_ref);
      assignArmSegment(left_qdot, left_arm_joints_, qdot_ref);
    }

    const Eigen::VectorXd desired_joint_vel = qdot_ref + kp_joint_ * (q_ref - q_feedback);
    const Eigen::VectorXd joint_weight =
      Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_tracking_;
    const Eigen::VectorXd damping_weight =
      Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;

    qp_filter_->setDesiredJointVel(desired_joint_vel);
    qp_filter_->setWeight(joint_weight, damping_weight);

    Eigen::VectorXd optimal_velocities;
    if (!qp_filter_->getOptJointVel(optimal_velocities)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000, "AI Worker MoveJ QP solver failed");
      return;
    }

    q_commanded_ = q_feedback + optimal_velocities * time_step_;
    publishTrajectory(q_commanded_);
    qdot_ = optimal_velocities;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Error in control loop: %s", e.what());
  }
}

void AIWorkerMoveJController::publishTrajectory(const Eigen::VectorXd & q_command) const
{
  std::vector<int> left_arm_indices;
  std::vector<int> right_arm_indices;

  for (const auto & joint_name : left_arm_joints_) {
    const auto it = model_joint_index_map_.find(joint_name);
    if (it != model_joint_index_map_.end()) {
      left_arm_indices.push_back(it->second);
    }
  }
  for (const auto & joint_name : right_arm_joints_) {
    const auto it = model_joint_index_map_.find(joint_name);
    if (it != model_joint_index_map_.end()) {
      right_arm_indices.push_back(it->second);
    }
  }

  if (!left_arm_indices.empty()) {
    arm_l_pub_->publish(createTrajectoryMsgWithGripper(
      left_arm_joints_, q_command, left_arm_indices, left_gripper_joint_name_,
        left_gripper_position_));
  }

  if (!right_arm_indices.empty()) {
    arm_r_pub_->publish(createTrajectoryMsgWithGripper(
      right_arm_joints_, q_command, right_arm_indices, right_gripper_joint_name_,
        right_gripper_position_));
  }
}

trajectory_msgs::msg::JointTrajectory AIWorkerMoveJController::createTrajectoryMsgWithGripper(
  const std::vector<std::string> & arm_joint_names,
  const Eigen::VectorXd & positions,
  const std::vector<int> & arm_indices,
  const std::string & gripper_joint_name,
  double gripper_position) const
{
  trajectory_msgs::msg::JointTrajectory traj_msg;
  traj_msg.header.frame_id = "";
  traj_msg.joint_names = arm_joint_names;
  traj_msg.joint_names.push_back(gripper_joint_name);

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);
  for (int idx : arm_indices) {
    if (idx >= 0 && idx < static_cast<int>(positions.size())) {
      point.positions.push_back(positions[idx]);
      point.velocities.push_back(0.0);
    }
  }
  point.positions.push_back(gripper_position);
  point.velocities.push_back(0.0);
  traj_msg.points.push_back(point);
  return traj_msg;
}

}  // namespace robotis_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<robotis_motion_controller_ros::AIWorkerMoveJController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
