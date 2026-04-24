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

#include "cyclo_motion_controller_ros/nodes/ai_worker/vr_controller_node.hpp"
#include <algorithm>
#include <cmath>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

namespace cyclo_motion_controller_ros
{
VRController::VRController()
: Node("vr_controller"),
  r_goal_pose_received_(false),
  l_goal_pose_received_(false),
  r_elbow_pose_received_(false),
  l_elbow_pose_received_(false),
  reference_diverged_(true),
  activate_pending_(false),
  joint_state_received_(false),
  dt_(0.01)
{
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "VR Controller - Starting up...");
  RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
  RCLCPP_INFO(this->get_logger(), "========================================");
  activate_start_ = this->get_clock()->now();

        // Load parameters
  control_frequency_ = this->declare_parameter("control_frequency", 100.0);
  time_step_ = this->declare_parameter("time_step", 0.01);
  trajectory_time_ = this->declare_parameter("trajectory_time", 0.0);
  kp_position_ = this->declare_parameter("kp_position", 50.0);
  kp_orientation_ = this->declare_parameter("kp_orientation", 50.0);
  weight_position_ = this->declare_parameter("weight_position", 10.0);
  weight_orientation_ = this->declare_parameter("weight_orientation", 1.0);
  weight_elbow_position_ = this->declare_parameter("weight_elbow_position", 0.5);
  weight_damping_ = this->declare_parameter("weight_damping", 0.1);
  slack_penalty_ = this->declare_parameter("slack_penalty", 1000.0);
  cbf_alpha_ = this->declare_parameter("cbf_alpha", 5.0);
  collision_buffer_ = this->declare_parameter("collision_buffer", 0.05);
  collision_safe_distance_ = this->declare_parameter("collision_safe_distance", 0.02);
  joint_state_timeout_ = this->declare_parameter("joint_state_timeout", 0.5);
  urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
  srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
  reactivate_topic_ = this->declare_parameter("reactivate_topic", std::string("/reactivate"));
  r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
  l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));
  r_elbow_pose_topic_ = this->declare_parameter("r_elbow_pose_topic", std::string("/r_subgoal_pose"));
  l_elbow_pose_topic_ = this->declare_parameter("l_elbow_pose_topic", std::string("/l_subgoal_pose"));
  joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
  right_traj_topic_ = this->declare_parameter("right_traj_topic",
      std::string("/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"));
  left_traj_topic_ = this->declare_parameter("left_traj_topic",
      std::string("/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"));
  right_raw_traj_topic_ = this->declare_parameter(
            "right_raw_traj_topic",
            std::string("/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory"));
  left_raw_traj_topic_ = this->declare_parameter(
            "left_raw_traj_topic",
            std::string("/leader/joint_trajectory_command_broadcaster_left/raw_joint_trajectory"));
  raw_traj_timeout_ = this->declare_parameter("raw_traj_timeout", 0.5);
  lift_topic_ = this->declare_parameter("lift_topic",
      std::string("/leader/joystick_controller_right/joint_trajectory"));
  lift_vel_bound_ = this->declare_parameter("lift_vel_bound", 0.0);
  r_gripper_pose_topic_ = this->declare_parameter("r_gripper_pose_topic",
      std::string("/r_gripper_pose"));
  l_gripper_pose_topic_ = this->declare_parameter("l_gripper_pose_topic",
      std::string("/l_gripper_pose"));
  r_gripper_name_ = this->declare_parameter("r_gripper_name", std::string("arm_r_link7"));
  l_gripper_name_ = this->declare_parameter("l_gripper_name", std::string("arm_l_link7"));
  r_elbow_name_ = this->declare_parameter("r_elbow_name", std::string("arm_r_link4"));
  l_elbow_name_ = this->declare_parameter("l_elbow_name", std::string("arm_l_link4"));
  right_gripper_joint_name_ = this->declare_parameter("right_gripper_joint",
      std::string("gripper_r_joint1"));
  left_gripper_joint_name_ = this->declare_parameter("left_gripper_joint",
      std::string("gripper_l_joint1"));
  startup_ref_pos_threshold_ = this->declare_parameter("startup_ref_pos_threshold", 0.15);
  startup_ref_ori_threshold_deg_ = this->declare_parameter("startup_ref_ori_threshold_deg", 45.0);

  dt_ = time_step_;
  last_joint_state_time_ = this->now();
  last_right_raw_traj_time_ = this->now();
  last_left_raw_traj_time_ = this->now();

        // Initialize subscribers
  r_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            r_goal_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            std::bind(&VRController::rightGoalPoseCallback, this, std::placeholders::_1));

  l_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            l_goal_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            std::bind(&VRController::leftGoalPoseCallback, this, std::placeholders::_1));

  r_elbow_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            r_elbow_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            std::bind(&VRController::rightElbowPoseCallback, this, std::placeholders::_1));

  l_elbow_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            l_elbow_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            std::bind(&VRController::leftElbowPoseCallback, this, std::placeholders::_1));

  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic_, 10,
            std::bind(&VRController::jointStateCallback, this, std::placeholders::_1));

  right_raw_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            right_raw_traj_topic_, 10,
            std::bind(&VRController::rightRawTrajectoryCallback, this,
      std::placeholders::_1));
  left_raw_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            left_raw_traj_topic_, 10,
            std::bind(&VRController::leftRawTrajectoryCallback, this, std::placeholders::_1));

  ref_divergence_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/reference_diverged", 10,
            std::bind(&VRController::referenceDivergenceCallback, this,
      std::placeholders::_1));
  reactivate_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            reactivate_topic_, 10,
            std::bind(&VRController::reactivateCallback, this, std::placeholders::_1));


        // Initialize publishers
  reference_divergence_pub_ = this->create_publisher<std_msgs::msg::Bool>("/reference_diverged",
      10);
  controller_error_pub_ = this->create_publisher<std_msgs::msg::String>("~/controller_error", 10);
  lift_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            lift_topic_, 10);

  arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            right_traj_topic_, 10);

  arm_l_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            left_traj_topic_, 10);

  r_gripper_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            r_gripper_pose_topic_, 10);

  l_gripper_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            l_gripper_pose_topic_, 10);

        // Initialize motion controller
  try {
    if (urdf_path_.empty()) {
      throw std::runtime_error("URDF path not provided.");
    }
    RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path_.c_str());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(),
                "Failed to resolve robot model paths: %s\n",
                e.what());
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
    kinematics_solver_ =
      std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(urdf_path_,
        srdf_path_);
    RCLCPP_INFO(this->get_logger(), "Initializing QP controller...");
    qp_controller_ =
      std::make_shared<cyclo_motion_controller::controllers::VRController>(kinematics_solver_,
        dt_);
    qp_controller_->setControllerParams(slack_penalty_, cbf_alpha_, collision_buffer_,
        collision_safe_distance_);

            // Initialize state variables
    int dof = kinematics_solver_->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);
    q_desired_.setZero(dof);

    RCLCPP_INFO(this->get_logger(), "Motion controller initialized (DOF: %d)", dof);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize motion controller: %s", e.what());
    rclcpp::shutdown();
    return;
  }

        // Initialize joint configuration from URDF
  initializeJointConfig();

        // Create control loop timer
  int timer_period_ms = static_cast<int>(1000.0 / control_frequency_);
  control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_period_ms),
            std::bind(&VRController::controlLoopCallback, this));

  if (!control_timer_) {
    RCLCPP_FATAL(this->get_logger(), "Failed to create control loop timer!");
    rclcpp::shutdown();
    return;
  }

  RCLCPP_INFO(this->get_logger(),
            "VR Controller initialized successfully!");
  RCLCPP_INFO(this->get_logger(),
            "  - Control loop: %.1f Hz (period: %d ms)", control_frequency_, timer_period_ms);
  RCLCPP_INFO(this->get_logger(),
            "  - Subscriptions: joint_states=%s",
            joint_state_sub_ ? "OK" : "FAILED");
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "Node is ready! Waiting for messages...");
  RCLCPP_WARN(
            this->get_logger(),
            "Control loop is ready. Publish Bool on '%s' to toggle control.",
            reactivate_topic_.c_str());
}

VRController::~VRController()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down VR Controller");
}

void VRController::initializeJointConfig()
{
        // Get actual joint names from the model (these should match joint_states topic)
  const auto joint_names = kinematics_solver_->getJointNames();
  model_joint_names_ = joint_names;
  model_joint_index_map_.clear();
  for (size_t i = 0; i < model_joint_names_.size(); ++i) {
    model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
  }

  left_arm_joints_.clear();
  right_arm_joints_.clear();
  lift_joint_.clear();
  lift_joint_index_ = -1;

        // Parse joint names
  for (const auto & joint_name : joint_names) {
    if (joint_name.find("arm_l_joint") != std::string::npos) {
      left_arm_joints_.push_back(joint_name);
    } else if (joint_name.find("arm_r_joint") != std::string::npos) {
      right_arm_joints_.push_back(joint_name);
    } else if (joint_name.find("lift_joint") != std::string::npos) {
      lift_joint_ = joint_name;
    }
  }

        // Sort to ensure correct ordering
  std::sort(left_arm_joints_.begin(), left_arm_joints_.end());
  std::sort(right_arm_joints_.begin(), right_arm_joints_.end());

        // Treat lift as a passive joint for the controller
  if (!lift_joint_.empty()) {
    auto lift_it = model_joint_index_map_.find(lift_joint_);
    if (lift_it != model_joint_index_map_.end()) {
      lift_joint_index_ = lift_it->second;
      const bool locked = kinematics_solver_->setJointVelocityBoundsByIndex(lift_joint_index_,
          -lift_vel_bound_, lift_vel_bound_);
      if (!locked) {
        RCLCPP_WARN(this->get_logger(), "Failed to set lift joint velocity bounds");
        lift_joint_index_ = -1;
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "Lift joint '%s' not found in model index map",
          lift_joint_.c_str());
    }
  }

        // Log joint names
  std::string left_str, right_str;
  for (const auto & j : left_arm_joints_) {
    left_str += j + " ";
  }
  for (const auto & j : right_arm_joints_) {
    right_str += j + " ";
  }
  RCLCPP_DEBUG(this->get_logger(), "Left arm joints: %s", left_str.c_str());
  RCLCPP_DEBUG(this->get_logger(), "Right arm joints: %s", right_str.c_str());
  RCLCPP_DEBUG(this->get_logger(), "Lift joint: %s", lift_joint_.c_str());
}

void VRController::rightGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  r_goal_pose_ = computePoseMat(*msg);
  r_goal_pose_received_ = true;
  RCLCPP_DEBUG(this->get_logger(), "Right goal pose received: [%.3f, %.3f, %.3f]",
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
}

void VRController::leftGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  l_goal_pose_ = computePoseMat(*msg);
  l_goal_pose_received_ = true;
  RCLCPP_DEBUG(this->get_logger(), "Left goal pose received: [%.3f, %.3f, %.3f]",
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
}

void VRController::rightElbowPoseCallback(
  const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  r_elbow_pose_ = computePoseMat(*msg);
  r_elbow_pose_received_ = true;
  RCLCPP_DEBUG(this->get_logger(), "Right elbow pose received: [%.3f, %.3f, %.3f]",
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
}

void VRController::leftElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  l_elbow_pose_ = computePoseMat(*msg);
  l_elbow_pose_received_ = true;
  RCLCPP_DEBUG(this->get_logger(), "Left elbow pose received: [%.3f, %.3f, %.3f]",
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
}

void VRController::rightRawTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!msg || msg->points.empty()) {
    return;
  }
  const auto & point = msg->points.front();
  if (point.positions.empty()) {
    return;
  }
  for (size_t i = 0; i < msg->joint_names.size(); ++i) {
    if (msg->joint_names[i] != right_gripper_joint_name_) {
      continue;
    }
    if (i < point.positions.size()) {
      right_raw_gripper_position_ = point.positions[i];
      right_raw_gripper_received_ = true;
      last_right_raw_traj_time_ = this->now();
    }
    return;
  }
}

void VRController::leftRawTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!msg || msg->points.empty()) {
    return;
  }
  const auto & point = msg->points.front();
  if (point.positions.empty()) {
    return;
  }
  for (size_t i = 0; i < msg->joint_names.size(); ++i) {
    if (msg->joint_names[i] != left_gripper_joint_name_) {
      continue;
    }
    if (i < point.positions.size()) {
      left_raw_gripper_position_ = point.positions[i];
      left_raw_gripper_received_ = true;
      last_left_raw_traj_time_ = this->now();
    }
    return;
  }
}

void VRController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  try {
            // Build joint index map on first callback
    if (joint_index_map_.empty()) {
      for (size_t i = 0; i < msg->name.size(); ++i) {
        joint_index_map_[msg->name[i]] = static_cast<int>(i);
      }
    }

    extractJointStates(msg);
    last_joint_state_time_ = this->now();
    joint_state_received_ = true;
    joint_state_timeout_active_ = false;

    if (!start_requested_ || !control_enabled_ || activate_pending_ || reference_diverged_) {
      syncCommandStateToFeedback();
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Error in jointStateCallback: %s", e.what());
  }
}

void VRController::referenceDivergenceCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (!msg->data) {
    return;
  }
        // Ignore reference divergence if controller is activating
  if (activate_pending_) {
    return;
  }
  if (!reference_diverged_) {
    RCLCPP_ERROR(this->get_logger(), "Reference divergence detected");
  }
  reference_diverged_ = true;
}

void VRController::reactivateCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  if (msg->data == reactivate_state_) {
    return;
  }

  reactivate_state_ = msg->data;
  if (reactivate_state_) {
    RCLCPP_WARN(this->get_logger(),
      "Reactivate topic '%s' set to true. "
      "Waiting for reference alignment before enabling controller.",
      reactivate_topic_.c_str());
    start_requested_ = true;
    control_enabled_ = false;
    activate_pending_ = false;
    reference_diverged_ = true;
    syncCommandStateToFeedback();
  } else {
    RCLCPP_WARN(this->get_logger(),
      "Reactivate topic '%s' set to false. Disabling controller.",
      reactivate_topic_.c_str());
    start_requested_ = false;
    control_enabled_ = false;
    activate_pending_ = false;
    reference_diverged_ = true;
    syncCommandStateToFeedback();
  }
}

void VRController::extractJointStates(const sensor_msgs::msg::JointState::SharedPtr & msg)
{
  int dof = kinematics_solver_->getDof();
  q_.setZero(dof);
  qdot_.setZero(dof);

        // Fill joint positions/velocities in model joint order
  const int max_index = std::min<int>(dof, static_cast<int>(model_joint_names_.size()));
  for (int i = 0; i < max_index; ++i) {
    const auto & joint_name = model_joint_names_[i];
    auto it = joint_index_map_.find(joint_name);
    if (it != joint_index_map_.end()) {
      int idx = it->second;
      if (idx < static_cast<int>(msg->position.size())) {
        q_[i] = msg->position[idx];
      }
                // ToDo: Add low pass filter
      if (idx < static_cast<int>(msg->velocity.size())) {
        qdot_[i] = msg->velocity[idx];
      }
    }
  }
}

void VRController::controlLoopCallback()
{
  static int loop_count = 0;
  static int debug_count = 0;

  loop_count++;

  if (!joint_state_received_) {
    if (debug_count++ % 100 == 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Control loop waiting for joint states...");
    }
    return;
  }

  if (jointStateTimedOut()) {
    if (!joint_state_timeout_active_) {
      joint_state_timeout_active_ = true;
      control_enabled_ = false;
      activate_pending_ = false;
      reference_diverged_ = true;
      syncCommandStateToFeedback();
      RCLCPP_WARN(
        this->get_logger(),
        "Joint states timed out. Holding controller until fresh feedback is received.");
    }
    return;
  }

        // Always compute & publish current gripper poses from measured state.
        // This allows external nodes to validate reference-vs-current before control starts.
  Eigen::Affine3d r_gripper_pose_meas = Eigen::Affine3d::Identity();
  Eigen::Affine3d l_gripper_pose_meas = Eigen::Affine3d::Identity();
  try {
    kinematics_solver_->updateState(q_, qdot_);
    r_gripper_pose_meas = kinematics_solver_->getPose(r_gripper_name_);
    l_gripper_pose_meas = kinematics_solver_->getPose(l_gripper_name_);
    publishGripperPose(r_gripper_pose_meas, l_gripper_pose_meas);
  } catch (const std::exception & e) {
    RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "Failed to compute/publish measured gripper pose: %s", e.what());
  }

        // Startup arming: wait for reactivate AND small goal-current pose error for both arms.
  if (!start_requested_) {
    syncCommandStateToFeedback();
    if (debug_count++ % 100 == 0) {
      RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    "Control loop waiting for reactivate topic '%s' to become true...",
                    reactivate_topic_.c_str());
    }
    return;
  }

        // Only perform the goal-vs-current mismatch check during startup (before enabling control).
  if (!control_enabled_) {
    syncCommandStateToFeedback();
    if (!r_goal_pose_received_ || !l_goal_pose_received_) {
      if (debug_count++ % 100 == 0) {
        RCLCPP_WARN_THROTTLE(
                        this->get_logger(), *this->get_clock(), 2000,
                        "Waiting for goal poses: right=%s left=%s",
                        r_goal_pose_received_ ? "OK" : "MISSING",
                        l_goal_pose_received_ ? "OK" : "MISSING");
      }
      return;
    }

    auto pose_error = [](const Eigen::Affine3d & cur, const Eigen::Affine3d & goal, double & pos_m,
      double & ori_deg) {
        pos_m = (goal.translation() - cur.translation()).norm();
        const Eigen::Quaterniond q_cur(cur.linear());
        const Eigen::Quaterniond q_goal(goal.linear());
        const double dot = std::abs(q_cur.dot(q_goal));
        const double clamped = std::min(1.0, std::max(-1.0, dot));
        const double angle_rad = 2.0 * std::acos(clamped);
        ori_deg = angle_rad * 180.0 / M_PI;
      };

    double r_pos_err = 0.0, r_ori_err = 0.0;
    double l_pos_err = 0.0, l_ori_err = 0.0;
    pose_error(r_gripper_pose_meas, r_goal_pose_, r_pos_err, r_ori_err);
    pose_error(l_gripper_pose_meas, l_goal_pose_, l_pos_err, l_ori_err);

    const bool r_ok =
      (r_pos_err <= startup_ref_pos_threshold_) &&
      (r_ori_err <= startup_ref_ori_threshold_deg_);
    const bool l_ok =
      (l_pos_err <= startup_ref_pos_threshold_) &&
      (l_ori_err <= startup_ref_ori_threshold_deg_);

    if (!(r_ok && l_ok)) {
      reference_diverged_ = true;

      if (reference_divergence_pub_) {
        std_msgs::msg::Bool msg;
        msg.data = true;
        reference_divergence_pub_->publish(msg);
      }
      if (controller_error_pub_) {
        std_msgs::msg::String err;
        err.data =
          "Startup reference mismatch: "
          "R(pos=" + std::to_string(r_pos_err) + "m, ori=" + std::to_string(r_ori_err) + "deg) "
          "L(pos=" + std::to_string(l_pos_err) + "m, ori=" + std::to_string(l_ori_err) + "deg) "
          "thresholds(pos=" + std::to_string(startup_ref_pos_threshold_) + "m, ori=" +
          std::to_string(startup_ref_ori_threshold_deg_) + "deg)";
        controller_error_pub_->publish(err);
      }

      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Startup mismatch. Waiting. "
        "R: pos=%.3f m ori=%.1f deg, "
        "L: pos=%.3f m ori=%.1f deg "
        "(thr pos=%.3f, ori=%.1f)",
        r_pos_err, r_ori_err, l_pos_err, l_ori_err, startup_ref_pos_threshold_,
        startup_ref_ori_threshold_deg_);
      return;
    }

            // Arm controller once the mismatch is small enough
    control_enabled_ = true;
    activate_start_ = this->get_clock()->now();
    activate_pending_ = true;
    reference_diverged_ = true;          // will be cleared after activation delay
    syncCommandStateToFeedback();
    RCLCPP_WARN(
                this->get_logger(),
                "Startup check passed. Activating controller.");
  }

  if (activate_pending_) {
    syncCommandStateToFeedback();
    const auto elapsed = this->get_clock()->now() - activate_start_;
    if (elapsed.seconds() >= 3.0) {
      reference_diverged_ = false;
      activate_pending_ = false;
      syncCommandStateToFeedback();
      RCLCPP_WARN(this->get_logger(), "Controller activated.");
    }
  }

  if (reference_diverged_) {
    syncCommandStateToFeedback();
    return;
  }

  debug_count = 0;

  try {
            // kinematics_solver_->updateState(q_, qdot_);
            // Use previously commanded joint goals as feedback state
    Eigen::VectorXd q_feedback =
      (q_desired_.size() == q_.size()) ? q_desired_ : q_;

    // If lift is commanded by another node, use measured lift state for
    // internal model consistency.
    if (lift_joint_index_ >= 0 && lift_joint_index_ < q_feedback.size()) {
      q_feedback[lift_joint_index_] = q_[lift_joint_index_];
    }

            // Control loop is executing - update kinematics solver with feedback state
    kinematics_solver_->updateState(q_feedback, qdot_);

            // Get current and goal end-effector poses
    right_gripper_pose_ = kinematics_solver_->getPose(r_gripper_name_);
    left_gripper_pose_ = kinematics_solver_->getPose(l_gripper_name_);
    Eigen::Affine3d right_elbow_pose = kinematics_solver_->getPose(r_elbow_name_);
    Eigen::Affine3d left_elbow_pose = kinematics_solver_->getPose(l_elbow_name_);

            // Initialize goals to current EE pose on first cycle if not received
    if (!r_goal_pose_received_ && !l_goal_pose_received_) {
      r_goal_pose_ = right_gripper_pose_;
      l_goal_pose_ = left_gripper_pose_;
    }
    if (!r_elbow_pose_received_) {
      r_elbow_pose_ = right_elbow_pose;
    }
    if (!l_elbow_pose_received_) {
      l_elbow_pose_ = left_elbow_pose;
    }

            // Publish current end-effector pose
    publishGripperPose(right_gripper_pose_, left_gripper_pose_);

            // Slow-start ramp after activation delay
    const auto activate_elapsed = this->get_clock()->now() - activate_start_;
    double slow_start_scale = 1.0;
    double slow_start_duration = 8.0;
    if (slow_start_duration > 0.0) {
      const double ramp_time = activate_elapsed.seconds() - 3.0;
      if (ramp_time < slow_start_duration) {
        slow_start_scale = std::clamp(ramp_time / slow_start_duration, 0.0, 1.0);
      }
    }

            // Compute desired velocity (scaled during slow-start)
    cyclo_motion_controller::common::Vector6d right_desired_vel =
      computeDesiredVelocity(right_gripper_pose_, r_goal_pose_) * slow_start_scale;
    cyclo_motion_controller::common::Vector6d left_desired_vel =
      computeDesiredVelocity(left_gripper_pose_, l_goal_pose_) * slow_start_scale;
    cyclo_motion_controller::common::Vector6d right_elbow_desired_vel =
      cyclo_motion_controller::common::Vector6d::Zero();
    cyclo_motion_controller::common::Vector6d left_elbow_desired_vel =
      cyclo_motion_controller::common::Vector6d::Zero();
    right_elbow_desired_vel.head(3) =
      kp_position_ * (r_elbow_pose_.translation() - right_elbow_pose.translation()) *
      slow_start_scale;
    left_elbow_desired_vel.head(3) =
      kp_position_ * (l_elbow_pose_.translation() - left_elbow_pose.translation()) *
      slow_start_scale;

    std::map<std::string, cyclo_motion_controller::common::Vector6d> desired_task_velocities;
    desired_task_velocities[r_gripper_name_] = right_desired_vel;
    desired_task_velocities[l_gripper_name_] = left_desired_vel;
    desired_task_velocities[r_elbow_name_] = right_elbow_desired_vel;
    desired_task_velocities[l_elbow_name_] = left_elbow_desired_vel;

            // Set weights for QP solver
    std::map<std::string, cyclo_motion_controller::common::Vector6d> weights;
    cyclo_motion_controller::common::Vector6d weight_right =
      cyclo_motion_controller::common::Vector6d::Ones();
    cyclo_motion_controller::common::Vector6d weight_left =
      cyclo_motion_controller::common::Vector6d::Ones();
    weight_right.head(3).setConstant(weight_position_);
    weight_right.tail(3).setConstant(weight_orientation_);
    weight_left.head(3).setConstant(weight_position_);
    weight_left.tail(3).setConstant(weight_orientation_);
    weights[r_gripper_name_] = weight_right;
    weights[l_gripper_name_] = weight_left;
    cyclo_motion_controller::common::Vector6d weight_right_elbow =
      cyclo_motion_controller::common::Vector6d::Zero();
    cyclo_motion_controller::common::Vector6d weight_left_elbow =
      cyclo_motion_controller::common::Vector6d::Zero();
    weight_right_elbow.head(3).setConstant(weight_elbow_position_);
    weight_left_elbow.head(3).setConstant(weight_elbow_position_);
    weights[r_elbow_name_] = weight_right_elbow;
    weights[l_elbow_name_] = weight_left_elbow;

    Eigen::VectorXd damping = Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;

            // Set weights and desired task velocities in QP controller
    qp_controller_->setWeight(weights, damping);
    qp_controller_->setDesiredTaskVel(desired_task_velocities);

            // Solve QP to get optimal joint velocities
    Eigen::VectorXd optimal_velocities;
    if (!qp_controller_->getOptJointVel(optimal_velocities)) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "QP solver failed to converge");
      return;
    }

            // Compute command from current state
            // q_desired_ = q_ + optimal_velocities * dt_;
    q_desired_ = q_feedback + optimal_velocities * dt_;

    // Publish trajectory commands
    publishTrajectory(q_desired_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Error in control loop: %s", e.what());
  }
}

bool VRController::jointStateTimedOut() const
{
  return joint_state_received_ &&
         (this->now() - last_joint_state_time_).seconds() > joint_state_timeout_;
}

void VRController::syncCommandStateToFeedback()
{
  if (q_.size() != q_desired_.size()) {
    return;
  }
  q_desired_ = q_;
}

Eigen::Affine3d VRController::computePoseMat(
  const geometry_msgs::msg::PoseStamped & pose) const
{
  Eigen::Affine3d pose_mat = Eigen::Affine3d::Identity();
  pose_mat.translation() << pose.pose.position.x,
    pose.pose.position.y,
    pose.pose.position.z;

  Eigen::Quaterniond quat(pose.pose.orientation.w,
    pose.pose.orientation.x,
    pose.pose.orientation.y,
    pose.pose.orientation.z);
  pose_mat.linear() = quat.toRotationMatrix();

  return pose_mat;
}

cyclo_motion_controller::common::Vector6d VRController::computeDesiredVelocity(
  const Eigen::Affine3d & current_pose,
  const Eigen::Affine3d & goal_pose) const
{
        // Compute position error
  Eigen::Vector3d pos_error = goal_pose.translation() - current_pose.translation();

        // Compute orientation error
  Eigen::Matrix3d rotation_error = goal_pose.linear() * current_pose.linear().transpose();
  Eigen::AngleAxisd angle_axis_error(rotation_error);
  Eigen::Vector3d angle_axis = angle_axis_error.axis() * angle_axis_error.angle();

  cyclo_motion_controller::common::Vector6d desired_vel =
    cyclo_motion_controller::common::Vector6d::Zero();
  desired_vel.head(3) = kp_position_ * pos_error;
  desired_vel.tail(3) = kp_orientation_ * angle_axis;

  return desired_vel;
}


void VRController::publishTrajectory(const Eigen::VectorXd & q_desired)
{
  try {
            // Build indices for each arm segment
    std::vector<int> left_arm_indices, right_arm_indices;

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

            // Publish left arm trajectory without gripper joint
    if (!left_arm_indices.empty()) {
      auto traj_left = createArmTrajectoryMsg(
                    left_arm_joints_, q_desired, left_arm_indices);
      arm_l_pub_->publish(traj_left);
    }

            // Publish right arm trajectory without gripper joint
    if (!right_arm_indices.empty()) {
      auto traj_right = createArmTrajectoryMsg(
                    right_arm_joints_, q_desired, right_arm_indices);
      arm_r_pub_->publish(traj_right);
    }

    // Publish lift trajectory
    if (lift_joint_index_ >= 0 && !lift_joint_.empty()) {
      if (lift_joint_index_ < static_cast<int>(q_desired.size())) {
        auto traj_lift = createLiftTrajectoryMsg(lift_joint_, q_desired[lift_joint_index_]);
        if (lift_vel_bound_ != 0.0) {
          lift_pub_->publish(traj_lift);
        }
      } else {
        RCLCPP_WARN(this->get_logger(), "Lift index out of range, skipping lift publish");
      }
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Error publishing trajectory: %s", e.what());
  }
}

trajectory_msgs::msg::JointTrajectory VRController::createArmTrajectoryMsg(
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

trajectory_msgs::msg::JointTrajectory VRController::createLiftTrajectoryMsg(
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

void VRController::publishGripperPose(
  const Eigen::Affine3d & r_gripper_pose,
  const Eigen::Affine3d & l_gripper_pose)
{
  if (r_gripper_pose_pub_) {
    geometry_msgs::msg::PoseStamped r_gripper_pose_msg;
    r_gripper_pose_msg.header.stamp = this->now();
    r_gripper_pose_msg.header.frame_id = "base_link";
    r_gripper_pose_msg.pose.position.x = r_gripper_pose.translation().x();
    r_gripper_pose_msg.pose.position.y = r_gripper_pose.translation().y();
    r_gripper_pose_msg.pose.position.z = r_gripper_pose.translation().z();
    Eigen::Quaterniond r_gripper_pose_quat(r_gripper_pose.linear());
    r_gripper_pose_msg.pose.orientation.w = r_gripper_pose_quat.w();
    r_gripper_pose_msg.pose.orientation.x = r_gripper_pose_quat.x();
    r_gripper_pose_msg.pose.orientation.y = r_gripper_pose_quat.y();
    r_gripper_pose_msg.pose.orientation.z = r_gripper_pose_quat.z();
    r_gripper_pose_pub_->publish(r_gripper_pose_msg);
  }
  if (l_gripper_pose_pub_) {
    geometry_msgs::msg::PoseStamped l_gripper_pose_msg;
    l_gripper_pose_msg.header.stamp = this->now();
    l_gripper_pose_msg.header.frame_id = "base_link";
    l_gripper_pose_msg.pose.position.x = l_gripper_pose.translation().x();
    l_gripper_pose_msg.pose.position.y = l_gripper_pose.translation().y();
    l_gripper_pose_msg.pose.position.z = l_gripper_pose.translation().z();
    Eigen::Quaterniond l_gripper_pose_quat(l_gripper_pose.linear());
    l_gripper_pose_msg.pose.orientation.w = l_gripper_pose_quat.w();
    l_gripper_pose_msg.pose.orientation.x = l_gripper_pose_quat.x();
    l_gripper_pose_msg.pose.orientation.y = l_gripper_pose_quat.y();
    l_gripper_pose_msg.pose.orientation.z = l_gripper_pose_quat.z();
    l_gripper_pose_pub_->publish(l_gripper_pose_msg);
  }
}

}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyclo_motion_controller_ros::VRController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
