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

#include "cyclo_motion_controller_ros/nodes/ai_worker/ai_worker_bimanual_movej_controller_node.hpp"

#include <algorithm>

#include "common/type_define.hpp"
#include "cyclo_motion_controller_ros/utils/trajectory_command_utils.hpp"

namespace cyclo_motion_controller_ros
{
namespace
{
constexpr double kGraspReleaseSlowStartErrorThreshold = 0.08;
constexpr double kGraspReleaseSlowStartJointSpeed = 0.3;
constexpr double kGraspReleaseSlowStartMaxDuration = 6.0;
constexpr double kGraspEnableBlendDuration = 4.0;
}  // namespace

AIWorkerBimanualMoveJController::AIWorkerBimanualMoveJController()
: Node("ai_worker_bimanual_movej_controller"),
  joint_state_received_(false),
  commanded_state_initialized_(false),
  right_movej_target_initialized_(false),
  left_movej_target_initialized_(false),
  right_gripper_position_(0.0),
  left_gripper_position_(0.0),
  right_motion_start_time_(this->now()),
  left_motion_start_time_(this->now()),
  last_joint_state_time_(this->now())
{
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "AI Worker Bimanual MoveJ Controller - Starting up...");
  RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
  RCLCPP_INFO(this->get_logger(), "========================================");

  control_frequency_ = this->declare_parameter("control_frequency", 100.0);
  time_step_ = this->declare_parameter("time_step", 0.01);
  trajectory_time_ = this->declare_parameter("trajectory_time", 0.0);
  kp_joint_ = this->declare_parameter("kp_joint", 6.0);
  weight_tracking_ = this->declare_parameter("weight_tracking", 1.0);
  weight_damping_ = this->declare_parameter("weight_damping", 0.1);
  slack_penalty_ = this->declare_parameter("slack_penalty", 1000.0);
  cbf_alpha_ = this->declare_parameter("cbf_alpha", 5.0);
  collision_buffer_ = this->declare_parameter("collision_buffer", 0.05);
  collision_safe_distance_ = this->declare_parameter("collision_safe_distance", 0.02);
  joint_state_timeout_ = this->declare_parameter("joint_state_timeout", 0.5);
  gripper_grasp_threshold_ = this->declare_parameter("gripper_grasp_threshold", 0.85);
  gripper_grasp_hold_time_ = this->declare_parameter("gripper_grasp_hold_time", 2.0);
  urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
  srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
  joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
  right_traj_topic_ = this->declare_parameter(
    "right_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory"));
  left_traj_topic_ = this->declare_parameter(
    "left_traj_topic",
    std::string("/leader/joint_trajectory_command_broadcaster_left/raw_joint_trajectory"));
  grasp_capture_topic_ = this->declare_parameter("grasp_capture_topic",
      std::string("/capture_grasp"));
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
  r_gripper_name_ = this->declare_parameter("r_gripper_name", std::string("arm_r_link7"));
  l_gripper_name_ = this->declare_parameter("l_gripper_name", std::string("arm_l_link7"));

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
    std::bind(&AIWorkerBimanualMoveJController::jointStateCallback, this, std::placeholders::_1));
  r_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    right_traj_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
    std::bind(&AIWorkerBimanualMoveJController::rightTrajectoryCallback, this,
      std::placeholders::_1));
  l_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    left_traj_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
    std::bind(&AIWorkerBimanualMoveJController::leftTrajectoryCallback, this,
      std::placeholders::_1));
  grasp_capture_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    grasp_capture_topic_, 10,
    std::bind(&AIWorkerBimanualMoveJController::graspCaptureCallback, this, std::placeholders::_1));

  try {
    RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path_.c_str());
    if (srdf_path_.empty()) {
      RCLCPP_INFO(this->get_logger(), "SRDF path not provided. Continuing without SRDF.");
    } else {
      RCLCPP_INFO(this->get_logger(), "SRDF path: %s", srdf_path_.c_str());
    }

    kinematics_solver_ =
      std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(urdf_path_,
        srdf_path_);
    qp_filter_ =
      std::make_shared<cyclo_motion_controller::controllers::AIWorkerBimanualMoveJController>(
      kinematics_solver_, time_step_);
    qp_filter_->setControllerParams(
      slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);
    qp_filter_->setConstraintLinks(r_gripper_name_, l_gripper_name_);

    const int dof = kinematics_solver_->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);
    q_commanded_.setZero(dof);
    right_movej_start_.setZero(dof);
    right_movej_goal_.setZero(dof);
    left_movej_start_.setZero(dof);
    left_movej_goal_.setZero(dof);
    right_release_hold_goal_.setZero(dof);
    left_release_hold_goal_.setZero(dof);
    right_grasp_enable_blend_start_.setZero(dof);
    left_grasp_enable_blend_start_.setZero(dof);
    initializeJointConfig();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize bimanual moveJ filter: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  const int timer_period_ms =
    std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, control_frequency_))));
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(timer_period_ms),
    std::bind(&AIWorkerBimanualMoveJController::controlLoopCallback, this));

  RCLCPP_INFO(this->get_logger(), "AI Worker Bimanual MoveJ Controller initialized successfully!");
}

AIWorkerBimanualMoveJController::~AIWorkerBimanualMoveJController()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down AI Worker Bimanual MoveJ Controller");
}

void AIWorkerBimanualMoveJController::initializeJointConfig()
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

void AIWorkerBimanualMoveJController::extractJointStates(
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

void AIWorkerBimanualMoveJController::jointStateCallback(
  const sensor_msgs::msg::JointState::SharedPtr msg)
{
  joint_index_map_.clear();
  for (size_t i = 0; i < msg->name.size(); ++i) {
    joint_index_map_[msg->name[i]] = static_cast<int>(i);
  }

  if (!right_gripper_command_received_) {
    trajectory_command_utils::updateJointPosition(
      *msg, right_gripper_joint_name_, right_gripper_position_);
  }
  if (!left_gripper_command_received_) {
    trajectory_command_utils::updateJointPosition(
      *msg, left_gripper_joint_name_, left_gripper_position_);
  }

  extractJointStates(msg);
  last_joint_state_time_ = this->now();
  joint_state_received_ = true;

  const bool was_uninitialized = !commanded_state_initialized_;
  const bool recovering_from_timeout = joint_state_timeout_active_;
  joint_state_timeout_active_ = false;
  updateGripperTriggeredGraspMode();

  if (was_uninitialized || recovering_from_timeout) {
    syncCommandStateToFeedback();
    commanded_state_initialized_ = true;
    right_movej_target_initialized_ = true;
    left_movej_target_initialized_ = true;
    RCLCPP_INFO(
      this->get_logger(),
      "AI Worker Bimanual MoveJ Controller activated. Waiting for moveJ commands...");
    return;
  }
}

bool AIWorkerBimanualMoveJController::updateGripperPositionFromTrajectory(
  const trajectory_msgs::msg::JointTrajectory & msg,
  const std::string & gripper_joint_name,
  double & gripper_position) const
{
  if (msg.points.empty() || msg.points.front().positions.empty()) {
    return false;
  }
  const auto & point = msg.points.front();
  for (size_t i = 0; i < msg.joint_names.size(); ++i) {
    if (msg.joint_names[i] != gripper_joint_name) {
      continue;
    }
    if (i < point.positions.size()) {
      gripper_position = point.positions[i];
      return true;
    }
    return false;
  }
  return false;
}

bool AIWorkerBimanualMoveJController::updateArmTargetFromTrajectory(
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
    RCLCPP_WARN(this->get_logger(), "%s bimanual moveJ ignored: positions are empty.",
        arm_name.c_str());
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
    "%s bimanual moveJ ignored: joint_names missing and positions size does not match arm joints.",
    arm_name.c_str());
  return false;
}

void AIWorkerBimanualMoveJController::rightTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || jointStateTimedOut() || !commanded_state_initialized_ ||
    msg->points.empty())
  {
    return;
  }
  const auto duration = rclcpp::Duration(msg->points.front().time_from_start).seconds();
  if (duration < 0.0) {
    RCLCPP_WARN(
      this->get_logger(), "Right bimanual moveJ ignored: time_from_start must be >= 0.");
    return;
  }

  const bool timed_motion = duration > 1e-6;
  if (timed_motion) {
    syncRightArmToFeedback();
  }
  Eigen::VectorXd target_q = q_commanded_;
  if (!updateArmTargetFromTrajectory(*msg, right_arm_joints_, "Right", target_q)) {
    return;
  }
  right_movej_start_ = q_commanded_;
  right_motion_start_time_ = this->now();
  right_active_motion_duration_ = duration;
  right_movej_goal_ = target_q;
  right_movej_trajectory_active_ = timed_motion;
  right_movej_target_initialized_ = true;
  if (
    trajectory_command_utils::hasJoint(joint_index_map_, right_gripper_joint_name_) &&
    updateGripperPositionFromTrajectory(
      *msg, right_gripper_joint_name_, right_gripper_position_))
  {
    right_gripper_command_received_ = true;
    updateGripperTriggeredGraspMode();
  }
  if (!right_release_follow_enabled_) {
    assignArmSegment(right_release_hold_goal_, right_arm_joints_, right_movej_goal_);
  } else {
    startPendingGraspEnableBlend(true, false);
    startPendingGraspReleaseSlowStart(true, false);
  }
}

void AIWorkerBimanualMoveJController::leftTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!msg || !joint_state_received_ || jointStateTimedOut() || !commanded_state_initialized_ ||
    msg->points.empty())
  {
    return;
  }
  const auto duration = rclcpp::Duration(msg->points.front().time_from_start).seconds();
  if (duration < 0.0) {
    RCLCPP_WARN(
      this->get_logger(), "Left bimanual moveJ ignored: time_from_start must be >= 0.");
    return;
  }

  const bool timed_motion = duration > 1e-6;
  if (timed_motion) {
    syncLeftArmToFeedback();
  }
  Eigen::VectorXd target_q = q_commanded_;
  if (!updateArmTargetFromTrajectory(*msg, left_arm_joints_, "Left", target_q)) {
    return;
  }
  left_movej_start_ = q_commanded_;
  left_motion_start_time_ = this->now();
  left_active_motion_duration_ = duration;
  left_movej_goal_ = target_q;
  left_movej_trajectory_active_ = timed_motion;
  left_movej_target_initialized_ = true;
  if (
    trajectory_command_utils::hasJoint(joint_index_map_, left_gripper_joint_name_) &&
    updateGripperPositionFromTrajectory(
      *msg, left_gripper_joint_name_, left_gripper_position_))
  {
    left_gripper_command_received_ = true;
    updateGripperTriggeredGraspMode();
  }
  if (!left_release_follow_enabled_) {
    assignArmSegment(left_release_hold_goal_, left_arm_joints_, left_movej_goal_);
  } else {
    startPendingGraspEnableBlend(false, true);
    startPendingGraspReleaseSlowStart(false, true);
  }
}

void AIWorkerBimanualMoveJController::graspCaptureCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  if (!msg->data) {
    manual_grasp_latch_ = false;
    RCLCPP_INFO(this->get_logger(), "Bimanual MoveJ grasp mode deactivated by capture topic.");
    disableGraspConstraint();
    return;
  }
  manual_grasp_latch_ = true;
  RCLCPP_INFO(this->get_logger(),
      "Bimanual MoveJ grasp mode activation requested by capture topic.");
  enableGraspConstraint();
}

void AIWorkerBimanualMoveJController::updateGripperTriggeredGraspMode()
{
  if (!right_gripper_command_received_ || !left_gripper_command_received_) {
    return;
  }

  if (manual_grasp_latch_) {
    gripper_closed_timer_active_ = false;
    right_gripper_open_timer_active_ = false;
    left_gripper_open_timer_active_ = false;
    return;
  }

  const rclcpp::Time now = this->now();
  const bool right_open = right_gripper_position_ < gripper_grasp_threshold_;
  const bool left_open = left_gripper_position_ < gripper_grasp_threshold_;
  const bool both_closed =
    right_gripper_position_ > gripper_grasp_threshold_ &&
    left_gripper_position_ > gripper_grasp_threshold_;

  bool right_open_held = false;
  bool left_open_held = false;
  if (right_open) {
    if (!right_gripper_open_timer_active_) {
      right_gripper_open_since_ = now;
      right_gripper_open_timer_active_ = true;
    } else if ((now - right_gripper_open_since_).seconds() >= gripper_grasp_hold_time_) {
      right_open_held = true;
    }
  } else {
    right_gripper_open_timer_active_ = false;
  }
  if (left_open) {
    if (!left_gripper_open_timer_active_) {
      left_gripper_open_since_ = now;
      left_gripper_open_timer_active_ = true;
    } else if ((now - left_gripper_open_since_).seconds() >= gripper_grasp_hold_time_) {
      left_open_held = true;
    }
  } else {
    left_gripper_open_timer_active_ = false;
  }

  if (!grasp_constraint_active_ && both_closed) {
    if (!gripper_closed_timer_active_) {
      gripper_closed_since_ = now;
      gripper_closed_timer_active_ = true;
    } else if ((now - gripper_closed_since_).seconds() >= gripper_grasp_hold_time_) {
      RCLCPP_INFO(
        this->get_logger(),
        "Bimanual MoveJ grasp mode activated from raw gripper commands. right=%.3f left=%.3f",
        right_gripper_position_, left_gripper_position_);
      enableGraspConstraint();
      gripper_closed_timer_active_ = false;
    }
  } else {
    gripper_closed_timer_active_ = false;
  }

  if (grasp_constraint_active_ && (right_open_held || left_open_held)) {
    RCLCPP_INFO(
      this->get_logger(),
      "Bimanual MoveJ grasp mode deactivated from raw gripper commands. "
      "right_open=%s left_open=%s right=%.3f left=%.3f",
      right_open ? "true" : "false", left_open ? "true" : "false",
      right_gripper_position_, left_gripper_position_);
    disableGraspConstraint(right_open, left_open);
  } else if (!grasp_constraint_active_ && grasp_release_follow_limited_) {
    enableGraspReleaseArmFollow(
      !right_release_follow_enabled_ && right_open_held,
      !left_release_follow_enabled_ && left_open_held);
  }
}

void AIWorkerBimanualMoveJController::enableGraspConstraint()
{
  const bool was_limited_release = grasp_release_follow_limited_;
  grasp_release_follow_limited_ = false;
  right_release_follow_enabled_ = true;
  left_release_follow_enabled_ = true;
  right_grasp_release_slow_start_pending_ = false;
  left_grasp_release_slow_start_pending_ = false;
  right_grasp_release_slow_start_active_ = false;
  left_grasp_release_slow_start_active_ = false;
  right_grasp_enable_blend_pending_ = was_limited_release;
  left_grasp_enable_blend_pending_ = was_limited_release;
  right_grasp_enable_blend_active_ = false;
  left_grasp_enable_blend_active_ = false;
  right_release_hold_goal_ = q_commanded_;
  left_release_hold_goal_ = q_commanded_;
  captureCurrentGraspConstraint();
  if (grasp_constraint_active_) {
    RCLCPP_INFO(this->get_logger(), "Bimanual MoveJ grasp constraint enabled.");
    if (was_limited_release) {
      RCLCPP_INFO(
        this->get_logger(),
        "Bimanual MoveJ grasp re-enabled after partial release. "
        "Leader rejoin will blend over %.1f seconds.",
        kGraspEnableBlendDuration);
    }
  }
}

void AIWorkerBimanualMoveJController::disableGraspConstraint(
  const bool right_arm_follow_enabled,
  const bool left_arm_follow_enabled)
{
  const bool was_active = grasp_constraint_active_;
  grasp_constraint_active_ = false;
  gripper_closed_timer_active_ = false;
  grasp_release_follow_limited_ = !(right_arm_follow_enabled && left_arm_follow_enabled);
  right_release_follow_enabled_ = right_arm_follow_enabled;
  left_release_follow_enabled_ = left_arm_follow_enabled;
  if (!right_arm_follow_enabled) {
    assignArmSegment(q_commanded_, right_arm_joints_, right_release_hold_goal_);
    assignArmSegment(right_release_hold_goal_, right_arm_joints_, right_movej_goal_);
    RCLCPP_INFO(this->get_logger(), "Bimanual MoveJ right arm hold pose captured.");
  }
  if (!left_arm_follow_enabled) {
    assignArmSegment(q_commanded_, left_arm_joints_, left_release_hold_goal_);
    assignArmSegment(left_release_hold_goal_, left_arm_joints_, left_movej_goal_);
    RCLCPP_INFO(this->get_logger(), "Bimanual MoveJ left arm hold pose captured.");
  }
  if (qp_filter_) {
    qp_filter_->setRigidGraspPoseConstraint(false, Eigen::Affine3d::Identity());
  }
  if (was_active) {
    RCLCPP_INFO(
      this->get_logger(),
      "Bimanual MoveJ grasp constraint disabled. follow_right=%s follow_left=%s",
      right_arm_follow_enabled ? "true" : "false",
      left_arm_follow_enabled ? "true" : "false");
    startGraspReleaseSlowStart(right_arm_follow_enabled, left_arm_follow_enabled);
  }
}

void AIWorkerBimanualMoveJController::captureCurrentGraspConstraint()
{
  if (!joint_state_received_) {
    return;
  }
  const Eigen::VectorXd q_feedback = commanded_state_initialized_ ? q_commanded_ : q_;
  kinematics_solver_->updateState(q_feedback, qdot_);
  const Eigen::Affine3d right_pose = kinematics_solver_->getPose(r_gripper_name_);
  const Eigen::Affine3d left_pose = kinematics_solver_->getPose(l_gripper_name_);
  grasp_right_to_left_ = right_pose.inverse() * left_pose;
  grasp_constraint_active_ = true;
}

void AIWorkerBimanualMoveJController::assignArmSegment(
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

void AIWorkerBimanualMoveJController::controlLoopCallback()
{
  if (!joint_state_received_ || !commanded_state_initialized_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Bimanual MoveJ control loop waiting for joint states...");
    return;
  }
  if (jointStateTimedOut()) {
    if (!joint_state_timeout_active_) {
      joint_state_timeout_active_ = true;
      right_movej_trajectory_active_ = false;
      left_movej_trajectory_active_ = false;
      RCLCPP_WARN(
        this->get_logger(),
        "Joint states timed out. Holding commands until fresh feedback is received.");
    }
    return;
  }

  try {
    const Eigen::VectorXd q_feedback = q_commanded_;
    kinematics_solver_->updateState(q_feedback, qdot_);

    Eigen::VectorXd q_ref = q_feedback;
    Eigen::VectorXd qdot_ref = Eigen::VectorXd::Zero(q_feedback.size());
    if (right_movej_target_initialized_ && right_release_follow_enabled_) {
      Eigen::VectorXd right_ref = right_movej_goal_;
      Eigen::VectorXd right_qdot = Eigen::VectorXd::Zero(q_feedback.size());
      const double elapsed = (this->now() - right_motion_start_time_).seconds();
      if (right_movej_trajectory_active_ && elapsed < right_active_motion_duration_) {
        const double interpolation_ratio =
          std::clamp(elapsed / right_active_motion_duration_, 0.0, 1.0);
        right_ref = right_movej_start_ +
          interpolation_ratio * (right_movej_goal_ - right_movej_start_);
        right_qdot =
          (right_movej_goal_ - right_movej_start_) / right_active_motion_duration_;
      } else if (right_movej_trajectory_active_) {
        right_movej_trajectory_active_ = false;
      }
      assignArmSegment(right_ref, right_arm_joints_, q_ref);
      assignArmSegment(right_qdot, right_arm_joints_, qdot_ref);
    } else if (!right_release_follow_enabled_) {
      assignArmSegment(right_release_hold_goal_, right_arm_joints_, q_ref);
    }
    if (left_movej_target_initialized_ && left_release_follow_enabled_) {
      Eigen::VectorXd left_ref = left_movej_goal_;
      Eigen::VectorXd left_qdot = Eigen::VectorXd::Zero(q_feedback.size());
      const double elapsed = (this->now() - left_motion_start_time_).seconds();
      if (left_movej_trajectory_active_ && elapsed < left_active_motion_duration_) {
        const double interpolation_ratio =
          std::clamp(elapsed / left_active_motion_duration_, 0.0, 1.0);
        left_ref = left_movej_start_ +
          interpolation_ratio * (left_movej_goal_ - left_movej_start_);
        left_qdot =
          (left_movej_goal_ - left_movej_start_) / left_active_motion_duration_;
      } else if (left_movej_trajectory_active_) {
        left_movej_trajectory_active_ = false;
      }
      assignArmSegment(left_ref, left_arm_joints_, q_ref);
      assignArmSegment(left_qdot, left_arm_joints_, qdot_ref);
    } else if (!left_release_follow_enabled_) {
      assignArmSegment(left_release_hold_goal_, left_arm_joints_, q_ref);
    }

    const auto apply_grasp_enable_blend =
      [this, &q_ref](
      const char * arm_name,
      const Eigen::VectorXd & blend_start,
      const Eigen::VectorXd & goal,
      const std::vector<std::string> & joints,
      bool & blend_active,
      const rclcpp::Time & blend_start_time) {
        if (!blend_active) {
          return;
        }
        const double elapsed = (this->now() - blend_start_time).seconds();
        const double alpha = std::clamp(elapsed / kGraspEnableBlendDuration, 0.0, 1.0);
        for (const auto & joint_name : joints) {
          const auto it = model_joint_index_map_.find(joint_name);
          if (it == model_joint_index_map_.end()) {
            continue;
          }
          const int index = it->second;
          if (index < 0 || index >= q_ref.size() || index >= blend_start.size() ||
            index >= goal.size())
          {
            continue;
          }
          q_ref[index] = (1.0 - alpha) * blend_start[index] + alpha * goal[index];
        }
        if (alpha >= 1.0) {
          blend_active = false;
          RCLCPP_INFO(
            this->get_logger(),
            "Bimanual MoveJ %s arm grasp-enable blend completed.", arm_name);
        }
      };
    apply_grasp_enable_blend(
      "right",
      right_grasp_enable_blend_start_,
      right_movej_goal_,
      right_arm_joints_,
      right_grasp_enable_blend_active_,
      right_grasp_enable_blend_start_time_);
    apply_grasp_enable_blend(
      "left",
      left_grasp_enable_blend_start_,
      left_movej_goal_,
      left_arm_joints_,
      left_grasp_enable_blend_active_,
      left_grasp_enable_blend_start_time_);

    Eigen::VectorXd desired_joint_vel = qdot_ref + kp_joint_ * (q_ref - q_feedback);
    const auto apply_slow_start =
      [this, &desired_joint_vel](
      const char * arm_name,
      const Eigen::VectorXd & goal,
      const std::vector<std::string> & joints,
      bool & slow_start_active,
      const rclcpp::Time & slow_start_time) {
        if (!slow_start_active) {
          return;
        }
        const double max_error = maxLeaderCommandError(goal, joints);
        const double elapsed = (this->now() - slow_start_time).seconds();
        if (max_error <= kGraspReleaseSlowStartErrorThreshold ||
          elapsed >= kGraspReleaseSlowStartMaxDuration)
        {
          slow_start_active = false;
          RCLCPP_INFO(
            this->get_logger(),
            "Bimanual MoveJ %s arm grasp-release slow start completed. error=%.4f elapsed=%.2f",
            arm_name, max_error, elapsed);
          return;
        }

        double max_desired_speed = 0.0;
        for (const auto & joint_name : joints) {
          const auto it = model_joint_index_map_.find(joint_name);
          if (it == model_joint_index_map_.end()) {
            continue;
          }
          const int index = it->second;
          if (index < 0 || index >= desired_joint_vel.size()) {
            continue;
          }
          max_desired_speed = std::max(max_desired_speed, std::abs(desired_joint_vel[index]));
        }
        if (max_desired_speed <= kGraspReleaseSlowStartJointSpeed) {
          return;
        }
        const double scale = kGraspReleaseSlowStartJointSpeed / max_desired_speed;
        for (const auto & joint_name : joints) {
          const auto it = model_joint_index_map_.find(joint_name);
          if (it == model_joint_index_map_.end()) {
            continue;
          }
          const int index = it->second;
          if (index >= 0 && index < desired_joint_vel.size()) {
            desired_joint_vel[index] *= scale;
          }
        }
      };
    apply_slow_start(
      "right",
      right_movej_goal_,
      right_arm_joints_, right_grasp_release_slow_start_active_,
      right_grasp_release_slow_start_time_);
    apply_slow_start(
      "left",
      left_movej_goal_,
      left_arm_joints_, left_grasp_release_slow_start_active_,
      left_grasp_release_slow_start_time_);
    const Eigen::VectorXd joint_weight =
      Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_tracking_;
    const Eigen::VectorXd damping =
      Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;

    if (grasp_constraint_active_) {
      qp_filter_->setRigidGraspPoseConstraint(
        true,
        grasp_right_to_left_);
    } else {
      qp_filter_->setRigidGraspPoseConstraint(false, Eigen::Affine3d::Identity());
    }

    qp_filter_->setDesiredJointVel(desired_joint_vel);
    qp_filter_->setWeight(joint_weight, damping);

    Eigen::VectorXd optimal_velocities;
    if (!qp_filter_->getOptJointVel(optimal_velocities)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "AI Worker Bimanual MoveJ QP solver failed");
      return;
    }

    q_commanded_ = q_feedback + optimal_velocities * time_step_;
    publishTrajectory(q_commanded_);
    qdot_ = optimal_velocities;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Bimanual moveJ loop error: %s", e.what());
  }
}

bool AIWorkerBimanualMoveJController::jointStateTimedOut() const
{
  return joint_state_received_ &&
         (this->now() - last_joint_state_time_).seconds() > joint_state_timeout_;
}

void AIWorkerBimanualMoveJController::syncCommandStateToFeedback()
{
  q_commanded_ = q_;
  right_movej_start_ = q_;
  right_movej_goal_ = q_;
  left_movej_start_ = q_;
  left_movej_goal_ = q_;
  right_release_hold_goal_ = q_;
  left_release_hold_goal_ = q_;
  right_grasp_enable_blend_start_ = q_;
  left_grasp_enable_blend_start_ = q_;
  right_movej_trajectory_active_ = false;
  left_movej_trajectory_active_ = false;
  grasp_release_follow_limited_ = false;
  right_release_follow_enabled_ = true;
  left_release_follow_enabled_ = true;
  right_grasp_release_slow_start_pending_ = false;
  left_grasp_release_slow_start_pending_ = false;
  right_grasp_release_slow_start_active_ = false;
  left_grasp_release_slow_start_active_ = false;
  right_grasp_enable_blend_pending_ = false;
  left_grasp_enable_blend_pending_ = false;
  right_grasp_enable_blend_active_ = false;
  left_grasp_enable_blend_active_ = false;
}

void AIWorkerBimanualMoveJController::syncRightArmToFeedback()
{
  assignArmSegment(q_, right_arm_joints_, q_commanded_);
}

void AIWorkerBimanualMoveJController::syncLeftArmToFeedback()
{
  assignArmSegment(q_, left_arm_joints_, q_commanded_);
}

double AIWorkerBimanualMoveJController::maxLeaderCommandError(
  const Eigen::VectorXd & goal,
  const std::vector<std::string> & joints) const
{
  double max_error = 0.0;
  for (const auto & joint_name : joints) {
    const auto it = model_joint_index_map_.find(joint_name);
    if (it == model_joint_index_map_.end()) {
      continue;
    }
    const int index = it->second;
    if (index < 0 || index >= q_commanded_.size() || index >= goal.size()) {
      continue;
    }
    max_error = std::max(max_error, std::abs(goal[index] - q_commanded_[index]));
  }
  return max_error;
}

void AIWorkerBimanualMoveJController::startGraspReleaseSlowStart(
  const bool right_arm,
  const bool left_arm)
{
  if (!joint_state_received_ || !commanded_state_initialized_) {
    return;
  }

  const rclcpp::Time now = this->now();
  if (right_arm) {
    right_grasp_release_slow_start_active_ = true;
    right_grasp_release_slow_start_pending_ = false;
    right_grasp_release_slow_start_time_ = now;
    RCLCPP_INFO(this->get_logger(), "Bimanual MoveJ right arm grasp-release slow start enabled.");
  }
  if (left_arm) {
    left_grasp_release_slow_start_active_ = true;
    left_grasp_release_slow_start_pending_ = false;
    left_grasp_release_slow_start_time_ = now;
    RCLCPP_INFO(this->get_logger(), "Bimanual MoveJ left arm grasp-release slow start enabled.");
  }
}

void AIWorkerBimanualMoveJController::startPendingGraspReleaseSlowStart(
  const bool right_arm,
  const bool left_arm)
{
  if (right_arm && right_grasp_release_slow_start_pending_) {
    startGraspReleaseSlowStart(true, false);
  }
  if (left_arm && left_grasp_release_slow_start_pending_) {
    startGraspReleaseSlowStart(false, true);
  }
}

void AIWorkerBimanualMoveJController::startPendingGraspEnableBlend(
  const bool right_arm,
  const bool left_arm)
{
  const rclcpp::Time now = this->now();
  if (right_arm && right_grasp_enable_blend_pending_) {
    assignArmSegment(q_commanded_, right_arm_joints_, right_grasp_enable_blend_start_);
    right_grasp_enable_blend_pending_ = false;
    right_grasp_enable_blend_active_ = true;
    right_grasp_enable_blend_start_time_ = now;
    RCLCPP_INFO(
      this->get_logger(),
      "Bimanual MoveJ right arm grasp-enable blend started for %.1f seconds.",
      kGraspEnableBlendDuration);
  }
  if (left_arm && left_grasp_enable_blend_pending_) {
    assignArmSegment(q_commanded_, left_arm_joints_, left_grasp_enable_blend_start_);
    left_grasp_enable_blend_pending_ = false;
    left_grasp_enable_blend_active_ = true;
    left_grasp_enable_blend_start_time_ = now;
    RCLCPP_INFO(
      this->get_logger(),
      "Bimanual MoveJ left arm grasp-enable blend started for %.1f seconds.",
      kGraspEnableBlendDuration);
  }
}

void AIWorkerBimanualMoveJController::enableGraspReleaseArmFollow(
  const bool right_arm,
  const bool left_arm)
{
  if (!right_arm && !left_arm) {
    return;
  }

  if (right_arm) {
    right_release_follow_enabled_ = true;
    right_grasp_release_slow_start_pending_ = true;
    RCLCPP_INFO(this->get_logger(),
        "Bimanual MoveJ right arm follow enabled after gripper release.");
  }
  if (left_arm) {
    left_release_follow_enabled_ = true;
    left_grasp_release_slow_start_pending_ = true;
    RCLCPP_INFO(this->get_logger(),
        "Bimanual MoveJ left arm follow enabled after gripper release.");
  }
  grasp_release_follow_limited_ = !(right_release_follow_enabled_ && left_release_follow_enabled_);
}

void AIWorkerBimanualMoveJController::publishTrajectory(const Eigen::VectorXd & q_command) const
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

trajectory_msgs::msg::JointTrajectory AIWorkerBimanualMoveJController::
createTrajectoryMsgWithGripper(
  const std::vector<std::string> & arm_joint_names,
  const Eigen::VectorXd & positions,
  const std::vector<int> & arm_indices,
  const std::string & gripper_joint_name,
  double gripper_position) const
{
  trajectory_msgs::msg::JointTrajectory traj_msg;
  traj_msg.header.frame_id = "";
  traj_msg.joint_names = arm_joint_names;

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);
  for (int idx : arm_indices) {
    if (idx >= 0 && idx < static_cast<int>(positions.size())) {
      point.positions.push_back(positions[idx]);
      point.velocities.push_back(0.0);
    }
  }
  traj_msg.points.push_back(point);
  trajectory_command_utils::appendJointIfPresent(
    traj_msg, joint_index_map_, gripper_joint_name, gripper_position);
  return traj_msg;
}

}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyclo_motion_controller_ros::AIWorkerBimanualMoveJController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
