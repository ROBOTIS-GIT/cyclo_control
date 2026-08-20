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

#include "cyclo_motion_controller_ros/nodes/ai_worker/ai_worker_bimanual_movel_controller_node.hpp"
#include "cyclo_motion_controller_ros/utils/trajectory_command_utils.hpp"

#include <algorithm>

namespace cyclo_motion_controller_ros
{
AIWorkerBimanualMoveLControllerNode::AIWorkerBimanualMoveLControllerNode()
: Node("ai_worker_bimanual_movel_controller"),
  last_joint_state_time_(this->now()),
  right_motion_start_time_(this->now()),
  left_motion_start_time_(this->now()),
  virtual_object_motion_start_time_(this->now())
{
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "AI Worker Bimanual MoveL Controller - Starting up...");
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
  joint_state_timeout_ = this->declare_parameter("joint_state_timeout", 0.5);
  urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
  srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
  joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
  right_movel_topic_ = this->declare_parameter("right_movel_topic", std::string("/r_goal_move"));
  left_movel_topic_ = this->declare_parameter("left_movel_topic", std::string("/l_goal_move"));
  virtual_object_movel_topic_ = this->declare_parameter(
    "virtual_object_movel_topic", std::string("/virtual_object_goal_move"));
  grasp_capture_topic_ = this->declare_parameter("grasp_capture_topic",
      std::string("/capture_grasp"));
  right_traj_topic_ = this->declare_parameter(
    "right_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"));
  left_traj_topic_ = this->declare_parameter(
    "left_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"));
  right_raw_traj_topic_ = this->declare_parameter(
    "right_raw_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory"));
  left_raw_traj_topic_ = this->declare_parameter(
    "left_raw_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_left/raw_joint_trajectory"));
  lift_topic_ = this->declare_parameter(
    "lift_topic", std::string("/leader/joystick_controller_right/joint_trajectory"));
  lift_vel_bound_ = this->declare_parameter("lift_vel_bound", 0.0);
  r_gripper_pose_topic_ = this->declare_parameter("r_gripper_pose_topic",
      std::string("/r_gripper_pose"));
  l_gripper_pose_topic_ = this->declare_parameter("l_gripper_pose_topic",
      std::string("/l_gripper_pose"));
  r_gripper_name_ = this->declare_parameter("r_gripper_name", std::string("arm_r_link7"));
  l_gripper_name_ = this->declare_parameter("l_gripper_name", std::string("arm_l_link7"));
  right_gripper_joint_name_ = this->declare_parameter(
    "right_gripper_joint", std::string("gripper_r_joint1"));
  left_gripper_joint_name_ = this->declare_parameter(
    "left_gripper_joint", std::string("gripper_l_joint1"));

  if (urdf_path_.empty()) {
    RCLCPP_FATAL(this->get_logger(), "URDF path not provided.");
    rclcpp::shutdown();
    return;
  }

  right_movel_sub_ = this->create_subscription<robotis_interfaces::msg::MoveL>(
    right_movel_topic_, 10,
    std::bind(&AIWorkerBimanualMoveLControllerNode::rightMoveLCallback, this,
      std::placeholders::_1));
  left_movel_sub_ = this->create_subscription<robotis_interfaces::msg::MoveL>(
    left_movel_topic_, 10,
    std::bind(&AIWorkerBimanualMoveLControllerNode::leftMoveLCallback, this,
      std::placeholders::_1));
  virtual_object_movel_sub_ = this->create_subscription<robotis_interfaces::msg::MoveL>(
    virtual_object_movel_topic_, 10,
    std::bind(&AIWorkerBimanualMoveLControllerNode::virtualObjectMoveLCallback, this,
      std::placeholders::_1));
  grasp_capture_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    grasp_capture_topic_, 10,
    std::bind(&AIWorkerBimanualMoveLControllerNode::graspCaptureCallback, this,
      std::placeholders::_1));
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    joint_states_topic_, 10,
    std::bind(&AIWorkerBimanualMoveLControllerNode::jointStateCallback, this,
      std::placeholders::_1));
  right_raw_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    right_raw_traj_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
    std::bind(&AIWorkerBimanualMoveLControllerNode::rightRawTrajectoryCallback, this,
      std::placeholders::_1));
  left_raw_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    left_raw_traj_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
    std::bind(&AIWorkerBimanualMoveLControllerNode::leftRawTrajectoryCallback, this,
      std::placeholders::_1));

  arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(right_traj_topic_, 10);
  arm_l_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(left_traj_topic_, 10);
  lift_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(lift_topic_, 10);
  r_gripper_pose_pub_ =
    this->create_publisher<geometry_msgs::msg::PoseStamped>(r_gripper_pose_topic_, 10);
  l_gripper_pose_pub_ =
    this->create_publisher<geometry_msgs::msg::PoseStamped>(l_gripper_pose_topic_, 10);

  try {
    RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path_.c_str());
    if (srdf_path_.empty()) {
      RCLCPP_INFO(this->get_logger(), "SRDF path not provided. Continuing without SRDF.");
    } else {
      RCLCPP_INFO(this->get_logger(), "SRDF path: %s", srdf_path_.c_str());
    }

    kinematics_solver_ = std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(
      urdf_path_, srdf_path_);
    qp_controller_ =
      std::make_shared<cyclo_motion_controller::controllers::AIWorkerBimanualMoveLController>(
      kinematics_solver_, time_step_);
    qp_controller_->setControllerParams(
      slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);
    qp_controller_->setConstraintLinks(r_gripper_name_, l_gripper_name_);

    const int dof = kinematics_solver_->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);
    q_desired_.setZero(dof);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize bimanual controller: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  initializeJointConfig();

  const int timer_period_ms =
    std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, control_frequency_))));
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(timer_period_ms),
    std::bind(&AIWorkerBimanualMoveLControllerNode::controlLoopCallback, this));

  RCLCPP_INFO(this->get_logger(), "AI Worker Bimanual MoveL Controller initialized successfully!");
}

AIWorkerBimanualMoveLControllerNode::~AIWorkerBimanualMoveLControllerNode()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down AI Worker Bimanual MoveL Controller");
}

void AIWorkerBimanualMoveLControllerNode::initializeJointConfig()
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

void AIWorkerBimanualMoveLControllerNode::extractJointStates(
  const sensor_msgs::msg::JointState::SharedPtr & msg)
{
  const int dof = kinematics_solver_->getDof();
  q_.setZero(dof);
  qdot_.setZero(dof);

  if (!right_gripper_command_received_) {
    trajectory_command_utils::updateJointPosition(
      *msg, right_gripper_joint_name_, right_gripper_position_);
  }
  if (!left_gripper_command_received_) {
    trajectory_command_utils::updateJointPosition(
      *msg, left_gripper_joint_name_, left_gripper_position_);
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

void AIWorkerBimanualMoveLControllerNode::rightRawTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (msg && trajectory_command_utils::hasJoint(
      joint_index_map_, right_gripper_joint_name_))
  {
    if (trajectory_command_utils::updateJointPosition(
        *msg, right_gripper_joint_name_, right_gripper_position_))
    {
      right_gripper_command_received_ = true;
    }
  }
}

void AIWorkerBimanualMoveLControllerNode::leftRawTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (msg && trajectory_command_utils::hasJoint(
      joint_index_map_, left_gripper_joint_name_))
  {
    if (trajectory_command_utils::updateJointPosition(
        *msg, left_gripper_joint_name_, left_gripper_position_))
    {
      left_gripper_command_received_ = true;
    }
  }
}

void AIWorkerBimanualMoveLControllerNode::jointStateCallback(
  const sensor_msgs::msg::JointState::SharedPtr msg)
{
  joint_index_map_.clear();
  for (size_t i = 0; i < msg->name.size(); ++i) {
    joint_index_map_[msg->name[i]] = static_cast<int>(i);
  }

  extractJointStates(msg);
  last_joint_state_time_ = this->now();
  joint_state_received_ = true;

  const bool was_uninitialized = !q_desired_initialized_;
  const bool recovering_from_timeout = joint_state_timeout_active_;
  joint_state_timeout_active_ = false;

  if (was_uninitialized || recovering_from_timeout) {
    syncCommandStateToFeedback();
    q_desired_initialized_ = true;
    right_movel_target_initialized_ = true;
    left_movel_target_initialized_ = true;
    virtual_object_target_initialized_ = true;
    if (was_uninitialized) {
      RCLCPP_INFO(
        this->get_logger(),
        "AI Worker Bimanual MoveL Controller activated. Waiting for moveL commands...");
    }
  }
}

void AIWorkerBimanualMoveLControllerNode::rightMoveLCallback(
  const robotis_interfaces::msg::MoveL::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || jointStateTimedOut() || !q_desired_initialized_) {
    return;
  }

  syncArmStateToFeedback(right_arm_joints_, q_desired_);
  if (lift_joint_index_ >= 0 && lift_joint_index_ < q_desired_.size()) {
    q_desired_[lift_joint_index_] = q_[lift_joint_index_];
  }
  kinematics_solver_->updateState(q_desired_, qdot_);
  right_movel_start_pose_ = kinematics_solver_->getPose(r_gripper_name_);
  right_movel_goal_pose_ = poseMsgToEigen(msg->pose);
  right_active_motion_duration_ = commandDurationSeconds(msg->time_from_start);
  right_motion_start_time_ = this->now();
  right_movel_target_initialized_ = true;
  right_movel_trajectory_active_ = right_active_motion_duration_ > 0.0;
}

void AIWorkerBimanualMoveLControllerNode::leftMoveLCallback(
  const robotis_interfaces::msg::MoveL::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || jointStateTimedOut() || !q_desired_initialized_) {
    return;
  }

  syncArmStateToFeedback(left_arm_joints_, q_desired_);
  if (lift_joint_index_ >= 0 && lift_joint_index_ < q_desired_.size()) {
    q_desired_[lift_joint_index_] = q_[lift_joint_index_];
  }
  kinematics_solver_->updateState(q_desired_, qdot_);
  left_movel_start_pose_ = kinematics_solver_->getPose(l_gripper_name_);
  left_movel_goal_pose_ = poseMsgToEigen(msg->pose);
  left_active_motion_duration_ = commandDurationSeconds(msg->time_from_start);
  left_motion_start_time_ = this->now();
  left_movel_target_initialized_ = true;
  left_movel_trajectory_active_ = left_active_motion_duration_ > 0.0;
}

void AIWorkerBimanualMoveLControllerNode::virtualObjectMoveLCallback(
  const robotis_interfaces::msg::MoveL::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || jointStateTimedOut() || !q_desired_initialized_) {
    return;
  }

  Eigen::VectorXd q_feedback = q_desired_;
  if (lift_joint_index_ >= 0 && lift_joint_index_ < q_feedback.size()) {
    q_feedback[lift_joint_index_] = q_[lift_joint_index_];
  }
  kinematics_solver_->updateState(q_feedback, qdot_);
  right_gripper_pose_ = kinematics_solver_->getPose(r_gripper_name_);
  left_gripper_pose_ = kinematics_solver_->getPose(l_gripper_name_);
  virtual_object_start_pose_ = currentObjectPose();
  virtual_object_goal_pose_ = poseMsgToEigen(msg->pose);
  virtual_object_active_motion_duration_ = commandDurationSeconds(msg->time_from_start);
  virtual_object_motion_start_time_ = this->now();
  virtual_object_target_initialized_ = true;
  virtual_object_trajectory_active_ = virtual_object_active_motion_duration_ > 0.0;
}

void AIWorkerBimanualMoveLControllerNode::graspCaptureCallback(
  const std_msgs::msg::Bool::SharedPtr msg)
{
  if (!msg) {
    return;
  }

  if (!msg->data) {
    RCLCPP_INFO(this->get_logger(), "Bimanual MoveL grasp mode deactivated by capture topic.");
    grasp_constraint_active_ = false;
    qp_controller_->setRigidGraspPoseConstraint(false, Eigen::Affine3d::Identity());
    if (joint_state_received_) {
      Eigen::VectorXd q_feedback = q_desired_initialized_ ? q_desired_ : q_;
      if (lift_joint_index_ >= 0 && lift_joint_index_ < q_feedback.size()) {
        q_feedback[lift_joint_index_] = q_[lift_joint_index_];
      }
      kinematics_solver_->updateState(q_feedback, qdot_);
      right_movel_start_pose_ = kinematics_solver_->getPose(r_gripper_name_);
      left_movel_start_pose_ = kinematics_solver_->getPose(l_gripper_name_);
      right_movel_goal_pose_ = right_movel_start_pose_;
      left_movel_goal_pose_ = left_movel_start_pose_;
      right_movel_target_initialized_ = true;
      left_movel_target_initialized_ = true;
      right_movel_trajectory_active_ = false;
      left_movel_trajectory_active_ = false;
    }
    return;
  }
  RCLCPP_INFO(this->get_logger(),
      "Bimanual MoveL grasp mode activation requested by capture topic.");
  captureCurrentGraspConstraint();
}

Eigen::Affine3d AIWorkerBimanualMoveLControllerNode::poseMsgToEigen(
  const geometry_msgs::msg::PoseStamped & pose_msg) const
{
  Eigen::Affine3d pose = Eigen::Affine3d::Identity();
  pose.translation() << pose_msg.pose.position.x, pose_msg.pose.position.y,
    pose_msg.pose.position.z;
  const Eigen::Quaterniond quat =
    cyclo_motion_controller::common::normalizedQuaternion(Eigen::Quaterniond(
      pose_msg.pose.orientation.w,
      pose_msg.pose.orientation.x,
      pose_msg.pose.orientation.y,
      pose_msg.pose.orientation.z));
  pose.linear() = quat.toRotationMatrix();
  return pose;
}

cyclo_motion_controller::common::Vector6d AIWorkerBimanualMoveLControllerNode::
computeDesiredVelocity(
  const Eigen::Affine3d & current_pose,
  const Eigen::Affine3d & goal_pose,
  const Eigen::Vector3d & feedforward_linear,
  const Eigen::Vector3d & feedforward_angular) const
{
  cyclo_motion_controller::common::Vector6d desired_vel =
    cyclo_motion_controller::common::Vector6d::Zero();
  const Eigen::Vector3d position_error = goal_pose.translation() - current_pose.translation();
  const Eigen::Vector3d orientation_error =
    cyclo_motion_controller::common::shortestOrientationError(
    goal_pose.linear(), current_pose.linear());
  desired_vel.head<3>() = feedforward_linear + kp_position_ * position_error;
  desired_vel.tail<3>() = feedforward_angular + kp_orientation_ * orientation_error;
  return desired_vel;
}

Eigen::Affine3d AIWorkerBimanualMoveLControllerNode::currentObjectPose() const
{
  if (grasp_constraint_active_) {
    return right_gripper_pose_ * grasp_object_to_right_.inverse();
  }

  Eigen::Affine3d object_pose = Eigen::Affine3d::Identity();
  object_pose.translation() = 0.5 *
    (right_gripper_pose_.translation() + left_gripper_pose_.translation());
  object_pose.linear() =
    cyclo_motion_controller::common::shortestSlerp(
      Eigen::Quaterniond(right_gripper_pose_.linear()),
      Eigen::Quaterniond(left_gripper_pose_.linear()), 0.5)
    .toRotationMatrix();
  return object_pose;
}

void AIWorkerBimanualMoveLControllerNode::captureCurrentGraspConstraint()
{
  if (!joint_state_received_) {
    return;
  }
  const Eigen::VectorXd q_feedback = q_desired_initialized_ ? q_desired_ : q_;
  kinematics_solver_->updateState(q_feedback, qdot_);
  const Eigen::Affine3d right_pose = kinematics_solver_->getPose(r_gripper_name_);
  const Eigen::Affine3d left_pose = kinematics_solver_->getPose(l_gripper_name_);
  right_gripper_pose_ = right_pose;
  left_gripper_pose_ = left_pose;
  const Eigen::Affine3d object_pose = currentObjectPose();
  grasp_object_to_right_ = object_pose.inverse() * right_pose;
  grasp_object_to_left_ = object_pose.inverse() * left_pose;
  grasp_constraint_active_ = true;
  virtual_object_start_pose_ = object_pose;
  virtual_object_goal_pose_ = object_pose;
  virtual_object_target_initialized_ = true;
  virtual_object_trajectory_active_ = false;
  RCLCPP_INFO(this->get_logger(), "Bimanual MoveL grasp constraint enabled.");
}

void AIWorkerBimanualMoveLControllerNode::controlLoopCallback()
{
  if (!joint_state_received_ || !q_desired_initialized_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Bimanual MoveL control loop waiting for joint states...");
    return;
  }
  if (jointStateTimedOut()) {
    if (!joint_state_timeout_active_) {
      joint_state_timeout_active_ = true;
      right_movel_trajectory_active_ = false;
      left_movel_trajectory_active_ = false;
      virtual_object_trajectory_active_ = false;
      RCLCPP_WARN(
        this->get_logger(),
        "Joint states timed out. Holding commands until fresh feedback is received.");
    }
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

    if (!right_movel_target_initialized_ || !left_movel_target_initialized_ ||
      !virtual_object_target_initialized_)
    {
      return;
    }

    const double right_elapsed = (this->now() - right_motion_start_time_).seconds();
    const double left_elapsed = (this->now() - left_motion_start_time_).seconds();
    const double virtual_object_elapsed = (this->now() -
      virtual_object_motion_start_time_).seconds();

    Eigen::Affine3d constrained_right_goal = right_movel_goal_pose_;
    Eigen::Affine3d constrained_left_goal = left_movel_goal_pose_;
    Eigen::Vector3d right_feedforward_linear = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_feedforward_angular = Eigen::Vector3d::Zero();
    Eigen::Vector3d left_feedforward_linear = Eigen::Vector3d::Zero();
    Eigen::Vector3d left_feedforward_angular = Eigen::Vector3d::Zero();

    if (grasp_constraint_active_) {
      Eigen::Affine3d object_goal = virtual_object_goal_pose_;
      Eigen::Vector3d object_feedforward_linear = Eigen::Vector3d::Zero();
      Eigen::Vector3d object_feedforward_angular = Eigen::Vector3d::Zero();

      if (virtual_object_trajectory_active_ &&
        virtual_object_elapsed < virtual_object_active_motion_duration_)
      {
        object_feedforward_linear =
          cyclo_motion_controller::common::math_utils::cubicDotVector<3>(
          virtual_object_elapsed, 0.0, virtual_object_active_motion_duration_,
          virtual_object_start_pose_.translation(), virtual_object_goal_pose_.translation(),
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        object_goal.translation() =
          cyclo_motion_controller::common::math_utils::cubicVector<3>(
          virtual_object_elapsed, 0.0, virtual_object_active_motion_duration_,
          virtual_object_start_pose_.translation(), virtual_object_goal_pose_.translation(),
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        object_goal.linear() =
          cyclo_motion_controller::common::math_utils::rotationCubic(
          virtual_object_elapsed, 0.0, virtual_object_active_motion_duration_,
          virtual_object_start_pose_.linear(), virtual_object_goal_pose_.linear());
        object_feedforward_angular =
          cyclo_motion_controller::common::math_utils::rotationCubicDot(
          virtual_object_elapsed, 0.0, virtual_object_active_motion_duration_,
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          virtual_object_start_pose_.linear(), virtual_object_goal_pose_.linear());
      } else if (virtual_object_trajectory_active_) {
        virtual_object_trajectory_active_ = false;
      }

      constrained_right_goal = object_goal * grasp_object_to_right_;
      constrained_left_goal = object_goal * grasp_object_to_left_;
      right_feedforward_linear = object_feedforward_linear +
        object_feedforward_angular.cross(constrained_right_goal.translation() -
          object_goal.translation());
      left_feedforward_linear = object_feedforward_linear +
        object_feedforward_angular.cross(constrained_left_goal.translation() -
          object_goal.translation());
      right_feedforward_angular = object_feedforward_angular;
      left_feedforward_angular = object_feedforward_angular;
    } else {
      if (right_movel_trajectory_active_ && right_elapsed < right_active_motion_duration_) {
        right_feedforward_linear =
          cyclo_motion_controller::common::math_utils::cubicDotVector<3>(
          right_elapsed, 0.0, right_active_motion_duration_,
          right_movel_start_pose_.translation(), right_movel_goal_pose_.translation(),
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        constrained_right_goal.translation() =
          cyclo_motion_controller::common::math_utils::cubicVector<3>(
          right_elapsed, 0.0, right_active_motion_duration_,
          right_movel_start_pose_.translation(), right_movel_goal_pose_.translation(),
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        constrained_right_goal.linear() =
          cyclo_motion_controller::common::math_utils::rotationCubic(
          right_elapsed, 0.0, right_active_motion_duration_,
          right_movel_start_pose_.linear(), right_movel_goal_pose_.linear());
        right_feedforward_angular =
          cyclo_motion_controller::common::math_utils::rotationCubicDot(
          right_elapsed, 0.0, right_active_motion_duration_,
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          right_movel_start_pose_.linear(), right_movel_goal_pose_.linear());
      } else if (right_movel_trajectory_active_) {
        right_movel_trajectory_active_ = false;
      }

      if (left_movel_trajectory_active_ && left_elapsed < left_active_motion_duration_) {
        left_feedforward_linear =
          cyclo_motion_controller::common::math_utils::cubicDotVector<3>(
          left_elapsed, 0.0, left_active_motion_duration_,
          left_movel_start_pose_.translation(), left_movel_goal_pose_.translation(),
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        constrained_left_goal.translation() =
          cyclo_motion_controller::common::math_utils::cubicVector<3>(
          left_elapsed, 0.0, left_active_motion_duration_,
          left_movel_start_pose_.translation(), left_movel_goal_pose_.translation(),
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        constrained_left_goal.linear() =
          cyclo_motion_controller::common::math_utils::rotationCubic(
          left_elapsed, 0.0, left_active_motion_duration_,
          left_movel_start_pose_.linear(), left_movel_goal_pose_.linear());
        left_feedforward_angular =
          cyclo_motion_controller::common::math_utils::rotationCubicDot(
          left_elapsed, 0.0, left_active_motion_duration_,
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          left_movel_start_pose_.linear(), left_movel_goal_pose_.linear());
      } else if (left_movel_trajectory_active_) {
        left_movel_trajectory_active_ = false;
      }
    }

    std::map<std::string, cyclo_motion_controller::common::Vector6d> desired_task_velocities;
    desired_task_velocities[r_gripper_name_] =
      computeDesiredVelocity(
      right_gripper_pose_, constrained_right_goal,
      right_feedforward_linear, right_feedforward_angular);
    desired_task_velocities[l_gripper_name_] =
      computeDesiredVelocity(
      left_gripper_pose_, constrained_left_goal,
      left_feedforward_linear, left_feedforward_angular);

    std::map<std::string, cyclo_motion_controller::common::Vector6d> weights;
    cyclo_motion_controller::common::Vector6d right_weight =
      cyclo_motion_controller::common::Vector6d::Ones();
    cyclo_motion_controller::common::Vector6d left_weight =
      cyclo_motion_controller::common::Vector6d::Ones();
    right_weight.head<3>().setConstant(weight_position_);
    right_weight.tail<3>().setConstant(weight_orientation_);
    left_weight.head<3>().setConstant(weight_position_);
    left_weight.tail<3>().setConstant(weight_orientation_);
    weights[r_gripper_name_] = right_weight;
    weights[l_gripper_name_] = left_weight;

    if (grasp_constraint_active_) {
      const Eigen::Affine3d right_to_left_transform =
        grasp_object_to_right_.inverse() * grasp_object_to_left_;
      qp_controller_->setRigidGraspPoseConstraint(
        true,
        right_to_left_transform);
    } else {
      qp_controller_->setRigidGraspPoseConstraint(false, Eigen::Affine3d::Identity());
    }

    const Eigen::VectorXd damping =
      Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;
    qp_controller_->setWeight(weights, damping);
    qp_controller_->setDesiredTaskVel(desired_task_velocities);

    Eigen::VectorXd optimal_velocities;
    if (!qp_controller_->getOptJointVel(optimal_velocities)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "AI Worker Bimanual MoveL QP solver failed");
      return;
    }

    q_desired_ = q_feedback + optimal_velocities * time_step_;
    publishTrajectory(q_desired_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Bimanual controller loop error: %s", e.what());
  }
}

bool AIWorkerBimanualMoveLControllerNode::jointStateTimedOut() const
{
  return joint_state_received_ &&
         (this->now() - last_joint_state_time_).seconds() > joint_state_timeout_;
}

void AIWorkerBimanualMoveLControllerNode::syncCommandStateToFeedback()
{
  q_desired_ = q_;
  kinematics_solver_->updateState(q_desired_, qdot_);
  right_gripper_pose_ = kinematics_solver_->getPose(r_gripper_name_);
  left_gripper_pose_ = kinematics_solver_->getPose(l_gripper_name_);
  right_movel_start_pose_ = right_gripper_pose_;
  left_movel_start_pose_ = left_gripper_pose_;
  right_movel_goal_pose_ = right_gripper_pose_;
  left_movel_goal_pose_ = left_gripper_pose_;
  virtual_object_start_pose_ = currentObjectPose();
  virtual_object_goal_pose_ = virtual_object_start_pose_;
  right_movel_trajectory_active_ = false;
  left_movel_trajectory_active_ = false;
  virtual_object_trajectory_active_ = false;
}

void AIWorkerBimanualMoveLControllerNode::syncArmStateToFeedback(
  const std::vector<std::string> & arm_joint_names,
  Eigen::VectorXd & destination) const
{
  for (const auto & joint_name : arm_joint_names) {
    const auto it = model_joint_index_map_.find(joint_name);
    if (it == model_joint_index_map_.end()) {
      continue;
    }
    destination[it->second] = q_[it->second];
  }
}

void AIWorkerBimanualMoveLControllerNode::publishTrajectory(const Eigen::VectorXd & q_desired) const
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
    auto trajectory = createArmTrajectoryMsg(left_arm_joints_, q_desired, left_arm_indices);
    trajectory_command_utils::appendJointIfPresent(
      trajectory, joint_index_map_, left_gripper_joint_name_, left_gripper_position_);
    arm_l_pub_->publish(trajectory);
  }
  if (!right_arm_indices.empty()) {
    auto trajectory = createArmTrajectoryMsg(right_arm_joints_, q_desired, right_arm_indices);
    trajectory_command_utils::appendJointIfPresent(
      trajectory, joint_index_map_, right_gripper_joint_name_, right_gripper_position_);
    arm_r_pub_->publish(trajectory);
  }
  if (lift_joint_index_ >= 0 && !lift_joint_.empty() && lift_vel_bound_ != 0.0 &&
    lift_joint_index_ < q_desired.size())
  {
    lift_pub_->publish(createLiftTrajectoryMsg(lift_joint_, q_desired[lift_joint_index_]));
  }
}

trajectory_msgs::msg::JointTrajectory AIWorkerBimanualMoveLControllerNode::createArmTrajectoryMsg(
  const std::vector<std::string> & arm_joint_names,
  const Eigen::VectorXd & positions,
  const std::vector<int> & arm_indices) const
{
  trajectory_msgs::msg::JointTrajectory traj_msg;
  traj_msg.header.frame_id = "";
  traj_msg.joint_names = arm_joint_names;

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);
  for (int idx : arm_indices) {
    if (idx >= 0 && idx < static_cast<int>(positions.size())) {
      point.positions.push_back(positions[idx]);
    }
  }
  traj_msg.points.push_back(point);
  return traj_msg;
}

trajectory_msgs::msg::JointTrajectory AIWorkerBimanualMoveLControllerNode::createLiftTrajectoryMsg(
  const std::string & lift_joint_name, const double position) const
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

void AIWorkerBimanualMoveLControllerNode::publishGripperPose(
  const Eigen::Affine3d & right_pose,
  const Eigen::Affine3d & left_pose)
{
  geometry_msgs::msg::PoseStamped right_msg;
  right_msg.header.stamp = this->now();
  right_msg.header.frame_id = "base_link";
  right_msg.pose.position.x = right_pose.translation().x();
  right_msg.pose.position.y = right_pose.translation().y();
  right_msg.pose.position.z = right_pose.translation().z();
  const Eigen::Quaterniond right_quat =
    cyclo_motion_controller::common::normalizedQuaternion(right_pose.linear());
  right_msg.pose.orientation.w = right_quat.w();
  right_msg.pose.orientation.x = right_quat.x();
  right_msg.pose.orientation.y = right_quat.y();
  right_msg.pose.orientation.z = right_quat.z();
  r_gripper_pose_pub_->publish(right_msg);

  geometry_msgs::msg::PoseStamped left_msg;
  left_msg.header.stamp = this->now();
  left_msg.header.frame_id = "base_link";
  left_msg.pose.position.x = left_pose.translation().x();
  left_msg.pose.position.y = left_pose.translation().y();
  left_msg.pose.position.z = left_pose.translation().z();
  const Eigen::Quaterniond left_quat =
    cyclo_motion_controller::common::normalizedQuaternion(left_pose.linear());
  left_msg.pose.orientation.w = left_quat.w();
  left_msg.pose.orientation.x = left_quat.x();
  left_msg.pose.orientation.y = left_quat.y();
  left_msg.pose.orientation.z = left_quat.z();
  l_gripper_pose_pub_->publish(left_msg);
}

}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyclo_motion_controller_ros::AIWorkerBimanualMoveLControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
