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

#include "robotis_motion_controller_ros/nodes/ai_worker/ai_worker_movel_controller_node.hpp"

#include <algorithm>

namespace robotis_motion_controller_ros
{
AIWorkerMoveLController::AIWorkerMoveLController()
: Node("ai_worker_movel_controller"),
  joint_state_received_(false),
  q_desired_initialized_(false),
  right_movel_target_initialized_(false),
  left_movel_target_initialized_(false),
  right_movel_trajectory_active_(false),
  left_movel_trajectory_active_(false),
  right_motion_start_time_(this->now()),
  left_motion_start_time_(this->now()),
  right_active_motion_duration_(0.0),
  left_active_motion_duration_(0.0),
  right_gripper_position_(0.0),
  left_gripper_position_(0.0)
{
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "AI Worker MoveL Controller - Starting up...");
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
  slack_penalty_ = this->declare_parameter("slack_penalty", 1000.0);
  cbf_alpha_ = this->declare_parameter("cbf_alpha", 5.0);
  collision_buffer_ = this->declare_parameter("collision_buffer", 0.05);
  collision_safe_distance_ = this->declare_parameter("collision_safe_distance", 0.02);
  urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
  srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
  joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
  right_movel_topic_ = this->declare_parameter("right_movel_topic", std::string("/r_goal_move"));
  left_movel_topic_ = this->declare_parameter("left_movel_topic", std::string("/l_goal_move"));
  right_traj_topic_ = this->declare_parameter(
    "right_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"));
  left_traj_topic_ = this->declare_parameter(
    "left_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"));
  lift_topic_ = this->declare_parameter(
    "lift_topic",
    std::string("/leader/joystick_controller_right/joint_trajectory"));
  lift_vel_bound_ = this->declare_parameter("lift_vel_bound", 0.0);
  r_gripper_pose_topic_ = this->declare_parameter("r_gripper_pose_topic",
      std::string("/r_gripper_pose"));
  l_gripper_pose_topic_ = this->declare_parameter("l_gripper_pose_topic",
      std::string("/l_gripper_pose"));
  controller_error_topic_ = this->declare_parameter("controller_error_topic",
      std::string("~/controller_error"));
  r_gripper_name_ = this->declare_parameter("r_gripper_name", std::string("arm_r_link7"));
  l_gripper_name_ = this->declare_parameter("l_gripper_name", std::string("arm_l_link7"));
  right_gripper_joint_name_ = this->declare_parameter("right_gripper_joint",
      std::string("gripper_r_joint1"));
  left_gripper_joint_name_ = this->declare_parameter("left_gripper_joint",
      std::string("gripper_l_joint1"));

  right_movel_sub_ = this->create_subscription<robotis_interfaces::msg::MoveL>(
    right_movel_topic_, 10,
    std::bind(&AIWorkerMoveLController::rightMoveLCallback, this, std::placeholders::_1));
  left_movel_sub_ = this->create_subscription<robotis_interfaces::msg::MoveL>(
    left_movel_topic_, 10,
    std::bind(&AIWorkerMoveLController::leftMoveLCallback, this, std::placeholders::_1));
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    joint_states_topic_, 10,
    std::bind(&AIWorkerMoveLController::jointStateCallback, this, std::placeholders::_1));

  arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(right_traj_topic_, 10);
  arm_l_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(left_traj_topic_, 10);
  lift_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(lift_topic_, 10);
  r_gripper_pose_pub_ =
    this->create_publisher<geometry_msgs::msg::PoseStamped>(r_gripper_pose_topic_, 10);
  l_gripper_pose_pub_ =
    this->create_publisher<geometry_msgs::msg::PoseStamped>(l_gripper_pose_topic_, 10);
  controller_error_pub_ = this->create_publisher<std_msgs::msg::String>(controller_error_topic_,
      10);

  if (urdf_path_.empty()) {
    RCLCPP_FATAL(this->get_logger(), "URDF path not provided.");
    rclcpp::shutdown();
    return;
  }

  try {
    kinematics_solver_ =
      std::make_shared<robotis_motion_controller::kinematics::KinematicsSolver>(urdf_path_,
        srdf_path_);
    qp_controller_ =
      std::make_shared<robotis_motion_controller::controllers::AIWorkerMoveLController>(
      kinematics_solver_, time_step_);
    qp_controller_->setControllerParams(
      slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);

    const int dof = kinematics_solver_->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);
    q_desired_.setZero(dof);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize motion controller: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  initializeJointConfig();

  const int timer_period_ms =
    std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, control_frequency_))));
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(timer_period_ms),
    std::bind(&AIWorkerMoveLController::controlLoopCallback, this));

  RCLCPP_INFO(this->get_logger(), "AI Worker MoveL Controller initialized successfully!");
}

AIWorkerMoveLController::~AIWorkerMoveLController()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down AI Worker MoveL Controller");
}

void AIWorkerMoveLController::initializeJointConfig()
{
  model_joint_names_ = kinematics_solver_->getJointNames();
  model_joint_index_map_.clear();
  for (size_t i = 0; i < model_joint_names_.size(); ++i) {
    model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
  }

  left_arm_joints_.clear();
  right_arm_joints_.clear();
  lift_joint_.clear();
  lift_joint_index_ = -1;

  for (const auto & joint_name : model_joint_names_) {
    if (joint_name.find("arm_l_joint") != std::string::npos) {
      left_arm_joints_.push_back(joint_name);
    } else if (joint_name.find("arm_r_joint") != std::string::npos) {
      right_arm_joints_.push_back(joint_name);
    } else if (joint_name.find("lift_joint") != std::string::npos) {
      lift_joint_ = joint_name;
    }
  }

  std::sort(left_arm_joints_.begin(), left_arm_joints_.end());
  std::sort(right_arm_joints_.begin(), right_arm_joints_.end());

  if (!lift_joint_.empty()) {
    auto lift_it = model_joint_index_map_.find(lift_joint_);
    if (lift_it != model_joint_index_map_.end()) {
      lift_joint_index_ = lift_it->second;
      const bool locked = kinematics_solver_->setJointVelocityBoundsByIndex(
        lift_joint_index_, -lift_vel_bound_, lift_vel_bound_);
      if (!locked) {
        lift_joint_index_ = -1;
      }
    }
  }
}

void AIWorkerMoveLController::extractJointStates(
  const sensor_msgs::msg::JointState::SharedPtr & msg)
{
  const int dof = kinematics_solver_->getDof();
  q_.setZero(dof);
  qdot_.setZero(dof);

  for (size_t i = 0; i < msg->name.size(); ++i) {
    if (msg->name[i] == right_gripper_joint_name_ && i < msg->position.size()) {
      right_gripper_position_ = msg->position[i];
    }
    if (msg->name[i] == left_gripper_joint_name_ && i < msg->position.size()) {
      left_gripper_position_ = msg->position[i];
    }
  }

  const int max_index = std::min<int>(dof, static_cast<int>(model_joint_names_.size()));
  for (int i = 0; i < max_index; ++i) {
    const auto & joint_name = model_joint_names_[i];
    auto it = joint_index_map_.find(joint_name);
    if (it == joint_index_map_.end()) {
      continue;
    }
    const int idx = it->second;
    if (idx < static_cast<int>(msg->position.size())) {
      q_[i] = msg->position[idx];
    }
    if (idx < static_cast<int>(msg->velocity.size())) {
      qdot_[i] = msg->velocity[idx];
    }
  }
}

void AIWorkerMoveLController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (joint_index_map_.empty()) {
    for (size_t i = 0; i < msg->name.size(); ++i) {
      joint_index_map_[msg->name[i]] = static_cast<int>(i);
    }
  }

  extractJointStates(msg);
  joint_state_received_ = true;

  if (!q_desired_initialized_) {
    q_desired_ = q_;
    q_desired_initialized_ = true;

    kinematics_solver_->updateState(q_desired_, qdot_);
    right_gripper_pose_ = kinematics_solver_->getPose(r_gripper_name_);
    left_gripper_pose_ = kinematics_solver_->getPose(l_gripper_name_);
    right_movel_start_pose_ = right_gripper_pose_;
    left_movel_start_pose_ = left_gripper_pose_;
    right_movel_goal_pose_ = right_gripper_pose_;
    left_movel_goal_pose_ = left_gripper_pose_;
    right_movel_target_initialized_ = true;
    left_movel_target_initialized_ = true;
  }
}

void AIWorkerMoveLController::rightMoveLCallback(
  const robotis_interfaces::msg::MoveL::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || !q_desired_initialized_) {
    return;
  }

  kinematics_solver_->updateState(q_desired_, qdot_);
  right_movel_start_pose_ = kinematics_solver_->getPose(r_gripper_name_);
  right_movel_goal_pose_ = poseMsgToEigen(msg->pose);
  right_active_motion_duration_ = commandDurationSeconds(msg->time_from_start);
  right_motion_start_time_ = this->now();
  right_movel_target_initialized_ = true;
  right_movel_trajectory_active_ = right_active_motion_duration_ > 1e-6;
}

void AIWorkerMoveLController::leftMoveLCallback(const robotis_interfaces::msg::MoveL::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || !q_desired_initialized_) {
    return;
  }

  kinematics_solver_->updateState(q_desired_, qdot_);
  left_movel_start_pose_ = kinematics_solver_->getPose(l_gripper_name_);
  left_movel_goal_pose_ = poseMsgToEigen(msg->pose);
  left_active_motion_duration_ = commandDurationSeconds(msg->time_from_start);
  left_motion_start_time_ = this->now();
  left_movel_target_initialized_ = true;
  left_movel_trajectory_active_ = left_active_motion_duration_ > 1e-6;
}

Eigen::Affine3d AIWorkerMoveLController::poseMsgToEigen(
  const geometry_msgs::msg::PoseStamped & pose_msg) const
{
  Eigen::Affine3d pose = Eigen::Affine3d::Identity();
  pose.translation() << pose_msg.pose.position.x, pose_msg.pose.position.y,
    pose_msg.pose.position.z;

  const Eigen::Quaterniond quat(
    pose_msg.pose.orientation.w,
    pose_msg.pose.orientation.x,
    pose_msg.pose.orientation.y,
    pose_msg.pose.orientation.z);
  pose.linear() = quat.normalized().toRotationMatrix();
  return pose;
}

robotis_motion_controller::common::Vector6d AIWorkerMoveLController::computeDesiredVelocity(
  const Eigen::Affine3d & current_pose,
  const Eigen::Affine3d & goal_pose,
  const Eigen::Vector3d & feedforward_linear,
  const Eigen::Vector3d & feedforward_angular) const
{
  robotis_motion_controller::common::Vector6d desired_vel =
    robotis_motion_controller::common::Vector6d::Zero();

  const Eigen::Vector3d position_error = goal_pose.translation() - current_pose.translation();
  const Eigen::Matrix3d rotation_error = goal_pose.linear() * current_pose.linear().transpose();
  const Eigen::AngleAxisd angle_axis_error(rotation_error);
  const Eigen::Vector3d orientation_error =
    angle_axis_error.axis() * angle_axis_error.angle();

  desired_vel.head<3>() = feedforward_linear + kp_position_ * position_error;
  desired_vel.tail<3>() = feedforward_angular + kp_orientation_ * orientation_error;
  return desired_vel;
}

void AIWorkerMoveLController::controlLoopCallback()
{
  if (!joint_state_received_ || !q_desired_initialized_) {
    return;
  }

  try {
    Eigen::VectorXd q_feedback = q_desired_;
    if (lift_joint_index_ >= 0 && lift_joint_index_ < q_feedback.size()) {
      q_feedback[lift_joint_index_] = q_[lift_joint_index_];
    }

    kinematics_solver_->updateState(q_feedback, qdot_);
    right_gripper_pose_ = kinematics_solver_->getPose(r_gripper_name_);
    left_gripper_pose_ = kinematics_solver_->getPose(l_gripper_name_);
    publishGripperPose(right_gripper_pose_, left_gripper_pose_);

    if (!right_movel_target_initialized_ || !left_movel_target_initialized_) {
      return;
    }

    const double right_elapsed = (this->now() - right_motion_start_time_).seconds();
    const double left_elapsed = (this->now() - left_motion_start_time_).seconds();

    robotis_motion_controller::common::Vector6d right_desired_vel =
      robotis_motion_controller::common::Vector6d::Zero();
    robotis_motion_controller::common::Vector6d left_desired_vel =
      robotis_motion_controller::common::Vector6d::Zero();

    if (right_movel_trajectory_active_ && right_elapsed < right_active_motion_duration_) {
      const Eigen::Vector3d linear_ref =
        robotis_motion_controller::common::math_utils::cubicDotVector<3>(
        right_elapsed, 0.0, right_active_motion_duration_,
        right_movel_start_pose_.translation(), right_movel_goal_pose_.translation(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
      const Eigen::Vector3d position_ref =
        robotis_motion_controller::common::math_utils::cubicVector<3>(
        right_elapsed, 0.0, right_active_motion_duration_,
        right_movel_start_pose_.translation(), right_movel_goal_pose_.translation(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
      const Eigen::Matrix3d rotation_ref =
        robotis_motion_controller::common::math_utils::rotationCubic(
        right_elapsed, 0.0, right_active_motion_duration_,
        right_movel_start_pose_.linear(), right_movel_goal_pose_.linear());
      const Eigen::Vector3d angular_ref =
        robotis_motion_controller::common::math_utils::rotationCubicDot(
        right_elapsed, 0.0, right_active_motion_duration_,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        right_movel_start_pose_.linear(), right_movel_goal_pose_.linear());

      Eigen::Affine3d pose_ref = Eigen::Affine3d::Identity();
      pose_ref.translation() = position_ref;
      pose_ref.linear() = rotation_ref;
      right_desired_vel =
        computeDesiredVelocity(right_gripper_pose_, pose_ref, linear_ref, angular_ref);
    } else {
      if (right_movel_trajectory_active_) {
        right_movel_trajectory_active_ = false;
      }
      right_desired_vel = computeDesiredVelocity(right_gripper_pose_, right_movel_goal_pose_);
    }

    if (left_movel_trajectory_active_ && left_elapsed < left_active_motion_duration_) {
      const Eigen::Vector3d linear_ref =
        robotis_motion_controller::common::math_utils::cubicDotVector<3>(
        left_elapsed, 0.0, left_active_motion_duration_,
        left_movel_start_pose_.translation(), left_movel_goal_pose_.translation(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
      const Eigen::Vector3d position_ref =
        robotis_motion_controller::common::math_utils::cubicVector<3>(
        left_elapsed, 0.0, left_active_motion_duration_,
        left_movel_start_pose_.translation(), left_movel_goal_pose_.translation(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
      const Eigen::Matrix3d rotation_ref =
        robotis_motion_controller::common::math_utils::rotationCubic(
        left_elapsed, 0.0, left_active_motion_duration_,
        left_movel_start_pose_.linear(), left_movel_goal_pose_.linear());
      const Eigen::Vector3d angular_ref =
        robotis_motion_controller::common::math_utils::rotationCubicDot(
        left_elapsed, 0.0, left_active_motion_duration_,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        left_movel_start_pose_.linear(), left_movel_goal_pose_.linear());

      Eigen::Affine3d pose_ref = Eigen::Affine3d::Identity();
      pose_ref.translation() = position_ref;
      pose_ref.linear() = rotation_ref;
      left_desired_vel =
        computeDesiredVelocity(left_gripper_pose_, pose_ref, linear_ref, angular_ref);
    } else {
      if (left_movel_trajectory_active_) {
        left_movel_trajectory_active_ = false;
      }
      left_desired_vel = computeDesiredVelocity(left_gripper_pose_, left_movel_goal_pose_);
    }

    std::map<std::string, robotis_motion_controller::common::Vector6d> desired_task_velocities;
    desired_task_velocities[r_gripper_name_] = right_desired_vel;
    desired_task_velocities[l_gripper_name_] = left_desired_vel;

    std::map<std::string, robotis_motion_controller::common::Vector6d> weights;
    robotis_motion_controller::common::Vector6d right_weight =
      robotis_motion_controller::common::Vector6d::Ones();
    robotis_motion_controller::common::Vector6d left_weight =
      robotis_motion_controller::common::Vector6d::Ones();
    right_weight.head<3>().setConstant(weight_position_);
    right_weight.tail<3>().setConstant(weight_orientation_);
    left_weight.head<3>().setConstant(weight_position_);
    left_weight.tail<3>().setConstant(weight_orientation_);
    weights[r_gripper_name_] = right_weight;
    weights[l_gripper_name_] = left_weight;

    const Eigen::VectorXd damping =
      Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;
    qp_controller_->setWeight(weights, damping);
    qp_controller_->setDesiredTaskVel(desired_task_velocities);

    Eigen::VectorXd optimal_velocities;
    if (!qp_controller_->getOptJointVel(optimal_velocities)) {
      std_msgs::msg::String err;
      err.data = "AI Worker MoveL Controller: QP solve failed";
      controller_error_pub_->publish(err);
      return;
    }

    q_desired_ = q_feedback + optimal_velocities * time_step_;
    publishTrajectory(q_desired_);
  } catch (const std::exception & e) {
    std_msgs::msg::String err;
    err.data = std::string("AI Worker MoveL Controller loop error: ") + e.what();
    controller_error_pub_->publish(err);
    RCLCPP_ERROR(this->get_logger(), "%s", err.data.c_str());
  }
}

void AIWorkerMoveLController::publishTrajectory(const Eigen::VectorXd & q_desired)
{
  std::vector<int> left_arm_indices;
  std::vector<int> right_arm_indices;

  for (const auto & joint_name : left_arm_joints_) {
    auto it = model_joint_index_map_.find(joint_name);
    if (it != model_joint_index_map_.end()) {
      left_arm_indices.push_back(it->second);
    }
  }

  for (const auto & joint_name : right_arm_joints_) {
    auto it = model_joint_index_map_.find(joint_name);
    if (it != model_joint_index_map_.end()) {
      right_arm_indices.push_back(it->second);
    }
  }

  if (!left_arm_indices.empty()) {
    arm_l_pub_->publish(createTrajectoryMsgWithGripper(
      left_arm_joints_, q_desired, left_arm_indices, left_gripper_joint_name_,
        left_gripper_position_));
  }

  if (!right_arm_indices.empty()) {
    arm_r_pub_->publish(createTrajectoryMsgWithGripper(
      right_arm_joints_, q_desired, right_arm_indices, right_gripper_joint_name_,
        right_gripper_position_));
  }

  if (lift_joint_index_ >= 0 && !lift_joint_.empty() && lift_vel_bound_ != 0.0 &&
    lift_joint_index_ < q_desired.size())
  {
    lift_pub_->publish(createLiftTrajectoryMsg(lift_joint_, q_desired[lift_joint_index_]));
  }
}

trajectory_msgs::msg::JointTrajectory AIWorkerMoveLController::createTrajectoryMsgWithGripper(
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
    }
  }
  point.positions.push_back(gripper_position);
  traj_msg.points.push_back(point);
  return traj_msg;
}

trajectory_msgs::msg::JointTrajectory AIWorkerMoveLController::createLiftTrajectoryMsg(
  std::string lift_joint_name,
  const double position) const
{
  trajectory_msgs::msg::JointTrajectory traj_msg;
  traj_msg.header.frame_id = "";
  traj_msg.joint_names = {lift_joint_name};

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);
  point.positions = {position};
  traj_msg.points.push_back(point);
  return traj_msg;
}

void AIWorkerMoveLController::publishGripperPose(
  const Eigen::Affine3d & r_gripper_pose,
  const Eigen::Affine3d & l_gripper_pose)
{
  if (r_gripper_pose_pub_) {
    geometry_msgs::msg::PoseStamped r_msg;
    r_msg.header.stamp = this->now();
    r_msg.header.frame_id = "base_link";
    r_msg.pose.position.x = r_gripper_pose.translation().x();
    r_msg.pose.position.y = r_gripper_pose.translation().y();
    r_msg.pose.position.z = r_gripper_pose.translation().z();
    const Eigen::Quaterniond quat(r_gripper_pose.linear());
    r_msg.pose.orientation.w = quat.w();
    r_msg.pose.orientation.x = quat.x();
    r_msg.pose.orientation.y = quat.y();
    r_msg.pose.orientation.z = quat.z();
    r_gripper_pose_pub_->publish(r_msg);
  }

  if (l_gripper_pose_pub_) {
    geometry_msgs::msg::PoseStamped l_msg;
    l_msg.header.stamp = this->now();
    l_msg.header.frame_id = "base_link";
    l_msg.pose.position.x = l_gripper_pose.translation().x();
    l_msg.pose.position.y = l_gripper_pose.translation().y();
    l_msg.pose.position.z = l_gripper_pose.translation().z();
    const Eigen::Quaterniond quat(l_gripper_pose.linear());
    l_msg.pose.orientation.w = quat.w();
    l_msg.pose.orientation.x = quat.x();
    l_msg.pose.orientation.y = quat.y();
    l_msg.pose.orientation.z = quat.z();
    l_gripper_pose_pub_->publish(l_msg);
  }
}

}  // namespace robotis_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<robotis_motion_controller_ros::AIWorkerMoveLController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
