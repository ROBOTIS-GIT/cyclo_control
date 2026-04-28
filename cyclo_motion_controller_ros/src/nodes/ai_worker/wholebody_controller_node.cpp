#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "common/type_define.hpp"
#include "controllers/ai_worker/vr_controller.hpp"
#include "kinematics/kinematics_solver.hpp"

namespace cyclo_motion_controller_ros
{
class WholebodyController : public rclcpp::Node
{
public:
  WholebodyController()
  : Node("wholebody_controller"),
    r_goal_pose_received_(false),
    l_goal_pose_received_(false),
    r_elbow_pose_received_(false),
    l_elbow_pose_received_(false),
    arm_base_pose_received_(false),
    joint_state_received_(false),
    dt_(0.01)
  {
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "Wholebody Controller - Starting up...");
    RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
    RCLCPP_INFO(this->get_logger(), "========================================");

    control_frequency_ = this->declare_parameter("control_frequency", 100.0);
    time_step_ = this->declare_parameter("time_step", 0.01);
    trajectory_time_ = this->declare_parameter("trajectory_time", 0.0);
    kp_position_ = this->declare_parameter("kp_position", 100.0);
    kp_orientation_ = this->declare_parameter("kp_orientation", 50.0);
    weight_position_ = this->declare_parameter("weight_position", 10.0);
    weight_orientation_ = this->declare_parameter("weight_orientation", 10.0);
    weight_elbow_position_ = this->declare_parameter("weight_elbow_position", 500.0);
    weight_arm_base_position_ = this->declare_parameter("weight_arm_base_position", 100.0);
    weight_damping_ = this->declare_parameter("weight_damping", 0.1);
    slack_penalty_ = this->declare_parameter("slack_penalty", 100.0);
    cbf_alpha_ = this->declare_parameter("cbf_alpha", 5.0);
    collision_buffer_ = this->declare_parameter("collision_buffer", 0.05);
    collision_safe_distance_ = this->declare_parameter("collision_safe_distance", 0.02);
    urdf_path_ = this->declare_parameter(
      "urdf_path",
      std::string(
        "/root/ros2_ws/src/cyclo_control/cyclo_motion_controller_models/models/ai_worker/"
        "ffw_sg2_follower.urdf"));
    srdf_path_ = this->declare_parameter(
      "srdf_path",
      std::string(
        "/root/ros2_ws/src/cyclo_control/cyclo_motion_controller_models/models/ai_worker/"
        "ffw_sg2_follower_default.srdf"));

    r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
    l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));
    r_elbow_pose_topic_ = this->declare_parameter("r_elbow_pose_topic", std::string("/r_elbow_pose"));
    l_elbow_pose_topic_ = this->declare_parameter("l_elbow_pose_topic", std::string("/l_elbow_pose"));
    arm_base_goal_pose_topic_ = this->declare_parameter(
      "arm_base_goal_pose_topic", std::string("/arm_base_goal_pose"));
    joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
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
    raw_traj_timeout_ = this->declare_parameter("raw_traj_timeout", 0.5);
    goal_ref_timeout_ = this->declare_parameter("goal_ref_timeout", 0.5);
    elbow_ref_timeout_ = this->declare_parameter("elbow_ref_timeout", 0.3);
    arm_base_ref_timeout_ = this->declare_parameter("arm_base_ref_timeout", 0.3);
    goal_motion_detect_timeout_ = this->declare_parameter("goal_motion_detect_timeout", 0.25);
    goal_motion_position_threshold_ =
      this->declare_parameter("goal_motion_position_threshold", 1e-4);
    goal_motion_orientation_threshold_ =
      this->declare_parameter("goal_motion_orientation_threshold", 1e-3);
    lift_transition_blend_duration_ =
      this->declare_parameter("lift_transition_blend_duration", 2.0);
    arm_base_motion_detect_timeout_ =
      this->declare_parameter("arm_base_motion_detect_timeout", 0.25);
    arm_base_motion_detect_threshold_ =
      this->declare_parameter("arm_base_motion_detect_threshold", 1e-4);
    lift_topic_ = this->declare_parameter(
      "lift_topic", std::string("/leader/joystick_controller_right/joint_trajectory"));
    lift_vel_bound_ = this->declare_parameter("lift_vel_bound", 4.8);
    lift_vel_bound_fixed_base_ = this->declare_parameter("lift_vel_bound_fixed_base", 0.0);
    lift_joint_name_ = this->declare_parameter("lift_joint_name", std::string("lift_joint"));
    r_gripper_pose_topic_ = this->declare_parameter(
      "r_gripper_pose_topic", std::string("/r_gripper_pose"));
    l_gripper_pose_topic_ = this->declare_parameter(
      "l_gripper_pose_topic", std::string("/l_gripper_pose"));
    r_gripper_name_ = this->declare_parameter(
      "r_gripper_name", std::string("end_effector_r_link"));
    l_gripper_name_ = this->declare_parameter(
      "l_gripper_name", std::string("end_effector_l_link"));
    r_elbow_name_ = this->declare_parameter("r_elbow_name", std::string("arm_r_link4"));
    l_elbow_name_ = this->declare_parameter("l_elbow_name", std::string("arm_l_link4"));
    arm_base_name_ = this->declare_parameter("arm_base_name", std::string("arm_base_link"));
    right_gripper_joint_name_ = this->declare_parameter(
      "right_gripper_joint", std::string("gripper_r_joint1"));
    left_gripper_joint_name_ = this->declare_parameter(
      "left_gripper_joint", std::string("gripper_l_joint1"));

    dt_ = time_step_;
    last_right_raw_traj_time_ = this->now();
    last_left_raw_traj_time_ = this->now();
    last_r_goal_pose_time_ = this->now();
    last_l_goal_pose_time_ = this->now();
    last_r_elbow_pose_time_ = this->now();
    last_l_elbow_pose_time_ = this->now();
    last_arm_base_pose_time_ = this->now();
    last_goal_motion_time_ = this->now();
    last_arm_base_motion_time_ = this->now();

    r_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      r_goal_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      std::bind(&WholebodyController::rightGoalPoseCallback, this, std::placeholders::_1));
    l_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      l_goal_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      std::bind(&WholebodyController::leftGoalPoseCallback, this, std::placeholders::_1));
    r_elbow_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      r_elbow_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      std::bind(&WholebodyController::rightElbowPoseCallback, this, std::placeholders::_1));
    l_elbow_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      l_elbow_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      std::bind(&WholebodyController::leftElbowPoseCallback, this, std::placeholders::_1));
    arm_base_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      arm_base_goal_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      std::bind(&WholebodyController::armBasePoseCallback, this, std::placeholders::_1));
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      joint_states_topic_, 10,
      std::bind(&WholebodyController::jointStateCallback, this, std::placeholders::_1));
    right_raw_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      right_raw_traj_topic_, 10,
      std::bind(&WholebodyController::rightRawTrajectoryCallback, this, std::placeholders::_1));
    left_raw_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      left_raw_traj_topic_, 10,
      std::bind(&WholebodyController::leftRawTrajectoryCallback, this, std::placeholders::_1));

    lift_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(lift_topic_, 10);
    arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(right_traj_topic_, 10);
    arm_l_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(left_traj_topic_, 10);
    r_gripper_pose_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseStamped>(r_gripper_pose_topic_, 10);
    l_gripper_pose_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseStamped>(l_gripper_pose_topic_, 10);

    if (urdf_path_.empty()) {
      RCLCPP_FATAL(this->get_logger(), "URDF path not provided.");
      rclcpp::shutdown();
      return;
    }

    try {
      kinematics_solver_ =
        std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(urdf_path_, srdf_path_);
      qp_controller_ =
        std::make_shared<cyclo_motion_controller::controllers::VRController>(kinematics_solver_, dt_);
      qp_controller_->setControllerParams(
        slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);
      fixed_base_kinematics_solver_ =
        std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(urdf_path_, srdf_path_);
      fixed_base_qp_controller_ =
        std::make_shared<cyclo_motion_controller::controllers::VRController>(
        fixed_base_kinematics_solver_, dt_);
      fixed_base_qp_controller_->setControllerParams(
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

    const int timer_period_ms = static_cast<int>(1000.0 / control_frequency_);
    control_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(timer_period_ms),
      std::bind(&WholebodyController::controlLoopCallback, this));

    RCLCPP_INFO(this->get_logger(), "Wholebody Controller initialized successfully!");
  }

private:
  void initializeJointConfig()
  {
    model_joint_names_ = kinematics_solver_->getJointNames();
    model_joint_index_map_.clear();
    for (size_t i = 0; i < model_joint_names_.size(); ++i) {
      model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
    }

    left_arm_joints_.clear();
    right_arm_joints_.clear();
    lift_joint_index_ = -1;

    for (const auto & joint_name : model_joint_names_) {
      if (joint_name.find("arm_l_joint") != std::string::npos) {
        left_arm_joints_.push_back(joint_name);
      } else if (joint_name.find("arm_r_joint") != std::string::npos) {
        right_arm_joints_.push_back(joint_name);
      }
    }

    std::sort(left_arm_joints_.begin(), left_arm_joints_.end());
    std::sort(right_arm_joints_.begin(), right_arm_joints_.end());

    auto lift_it = model_joint_index_map_.find(lift_joint_name_);
    if (lift_it != model_joint_index_map_.end()) {
      lift_joint_index_ = lift_it->second;
      (void)kinematics_solver_->setJointVelocityBoundsByIndex(
        lift_joint_index_, -lift_vel_bound_, lift_vel_bound_);
    } else {
      RCLCPP_ERROR(
        this->get_logger(),
        "Model lift joint '%s' not found in URDF/model joint names.",
        lift_joint_name_.c_str());
    }
  }

  void rightGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    const Eigen::Affine3d next_pose = computePoseMat(*msg);
    updateGoalMotionState(next_pose, r_goal_pose_received_ ? &r_goal_pose_ : nullptr);
    r_goal_pose_ = next_pose;
    r_goal_pose_received_ = true;
    last_r_goal_pose_time_ = this->now();
  }

  void leftGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    const Eigen::Affine3d next_pose = computePoseMat(*msg);
    updateGoalMotionState(next_pose, l_goal_pose_received_ ? &l_goal_pose_ : nullptr);
    l_goal_pose_ = next_pose;
    l_goal_pose_received_ = true;
    last_l_goal_pose_time_ = this->now();
  }

  void rightElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    r_elbow_pose_ = computePoseMat(*msg);
    r_elbow_pose_received_ = true;
    last_r_elbow_pose_time_ = this->now();
  }

  void leftElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    l_elbow_pose_ = computePoseMat(*msg);
    l_elbow_pose_received_ = true;
    last_l_elbow_pose_time_ = this->now();
  }

  void armBasePoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    const Eigen::Affine3d next_pose = computePoseMat(*msg);
    if (arm_base_pose_received_) {
      const double delta_z =
        std::abs(next_pose.translation().z() - arm_base_goal_pose_.translation().z());
      if (delta_z > arm_base_motion_detect_threshold_) {
        last_arm_base_motion_time_ = this->now();
      }
    } else {
      last_arm_base_motion_time_ = this->now();
    }
    arm_base_goal_pose_ = next_pose;
    arm_base_pose_received_ = true;
    last_arm_base_pose_time_ = this->now();
  }

  void rightRawTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
  {
    if (!msg || msg->points.empty()) {
      return;
    }
    const auto & point = msg->points.front();
    for (size_t i = 0; i < msg->joint_names.size(); ++i) {
      if (msg->joint_names[i] == right_gripper_joint_name_ && i < point.positions.size()) {
        right_raw_gripper_position_ = point.positions[i];
        right_raw_gripper_received_ = true;
        last_right_raw_traj_time_ = this->now();
        return;
      }
    }
  }

  void leftRawTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
  {
    if (!msg || msg->points.empty()) {
      return;
    }
    const auto & point = msg->points.front();
    for (size_t i = 0; i < msg->joint_names.size(); ++i) {
      if (msg->joint_names[i] == left_gripper_joint_name_ && i < point.positions.size()) {
        left_raw_gripper_position_ = point.positions[i];
        left_raw_gripper_received_ = true;
        last_left_raw_traj_time_ = this->now();
        return;
      }
    }
  }

  void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr & msg)
  {
    const int dof = kinematics_solver_->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);

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

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (joint_index_map_.empty()) {
      for (size_t i = 0; i < msg->name.size(); ++i) {
        joint_index_map_[msg->name[i]] = static_cast<int>(i);
      }
    }
    extractJointStates(msg);
    joint_state_received_ = true;

    static bool positions_initialized = false;
    if (!positions_initialized) {
      q_desired_ = q_;
      positions_initialized = true;
    }
  }

  void controlLoopCallback()
  {
    if (!joint_state_received_) {
      return;
    }

    try {
      kinematics_solver_->updateState(q_, qdot_);
      publishGripperPose(
        kinematics_solver_->getPose(r_gripper_name_),
        kinematics_solver_->getPose(l_gripper_name_));
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Failed to compute/publish measured gripper pose: %s", e.what());
    }

    try {
      const rclcpp::Time now = this->now();
      const bool right_goal_active =
        r_goal_pose_received_ && (now - last_r_goal_pose_time_).seconds() < goal_ref_timeout_;
      const bool left_goal_active =
        l_goal_pose_received_ && (now - last_l_goal_pose_time_).seconds() < goal_ref_timeout_;
      const bool right_elbow_active =
        r_elbow_pose_received_ && (now - last_r_elbow_pose_time_).seconds() < elbow_ref_timeout_;
      const bool left_elbow_active =
        l_elbow_pose_received_ && (now - last_l_elbow_pose_time_).seconds() < elbow_ref_timeout_;
      const bool arm_base_active =
        arm_base_pose_received_ && (now - last_arm_base_pose_time_).seconds() < arm_base_ref_timeout_;
      const bool arm_base_ref_moving_recently =
        arm_base_pose_received_ &&
        (now - last_arm_base_motion_time_).seconds() < arm_base_motion_detect_timeout_;
      const bool goal_ref_moving_recently =
        (now - last_goal_motion_time_).seconds() < goal_motion_detect_timeout_;
      const bool fixed_base_phase_active =
        arm_base_active &&
        !arm_base_ref_moving_recently &&
        !goal_ref_moving_recently &&
        (right_elbow_active || left_elbow_active);
      const bool hard_pose_constraint_active =
        !goal_ref_moving_recently &&
        (right_elbow_active || left_elbow_active);

      if (!right_goal_active && !left_goal_active && !right_elbow_active &&
        !left_elbow_active && !arm_base_active)
      {
        lift_transition_active_ = false;
        q_desired_ = q_;
        return;
      }

      updateLiftVelocityBound(fixed_base_phase_active ? lift_vel_bound_fixed_base_ : lift_vel_bound_);

      Eigen::VectorXd q_feedback = (q_desired_.size() == q_.size()) ? q_desired_ : q_;
      blendLiftStateForGoalTransition(
        q_feedback, now, fixed_base_phase_active && goal_ref_moving_recently);
      const auto & active_solver =
        hard_pose_constraint_active ? fixed_base_kinematics_solver_ : kinematics_solver_;
      const auto & active_controller =
        hard_pose_constraint_active ? fixed_base_qp_controller_ : qp_controller_;
      const Eigen::VectorXd & solver_state = q_feedback;
      active_solver->updateState(solver_state, qdot_);

      const Eigen::Affine3d right_gripper_pose = active_solver->getPose(r_gripper_name_);
      const Eigen::Affine3d left_gripper_pose = active_solver->getPose(l_gripper_name_);
      const Eigen::Affine3d right_elbow_pose = active_solver->getPose(r_elbow_name_);
      const Eigen::Affine3d left_elbow_pose = active_solver->getPose(l_elbow_name_);
      const Eigen::Affine3d arm_base_pose = active_solver->getPose(arm_base_name_);

      if (!right_goal_active) {
        r_goal_pose_ = right_gripper_pose;
      }
      if (!left_goal_active) {
        l_goal_pose_ = left_gripper_pose;
      }

      std::map<std::string, cyclo_motion_controller::common::Vector6d> desired_task_velocities;
      std::map<std::string, cyclo_motion_controller::common::Vector6d> hard_constraint_task_velocities;
      desired_task_velocities[r_gripper_name_] =
        computeDesiredVelocity(right_gripper_pose, r_goal_pose_);
      desired_task_velocities[l_gripper_name_] =
        computeDesiredVelocity(left_gripper_pose, l_goal_pose_);
      if (hard_pose_constraint_active) {
        hard_constraint_task_velocities[r_gripper_name_] = desired_task_velocities[r_gripper_name_];
        hard_constraint_task_velocities[l_gripper_name_] = desired_task_velocities[l_gripper_name_];
      }

      if (right_elbow_active) {
        cyclo_motion_controller::common::Vector6d right_elbow_desired_vel =
          cyclo_motion_controller::common::Vector6d::Zero();
        right_elbow_desired_vel.head(3) =
          kp_position_ * (r_elbow_pose_.translation() - right_elbow_pose.translation());
        desired_task_velocities[r_elbow_name_] = right_elbow_desired_vel;
      }
      if (left_elbow_active) {
        cyclo_motion_controller::common::Vector6d left_elbow_desired_vel =
          cyclo_motion_controller::common::Vector6d::Zero();
        left_elbow_desired_vel.head(3) =
          kp_position_ * (l_elbow_pose_.translation() - left_elbow_pose.translation());
        desired_task_velocities[l_elbow_name_] = left_elbow_desired_vel;
      }
      if (arm_base_active) {
        cyclo_motion_controller::common::Vector6d arm_base_desired_vel =
          cyclo_motion_controller::common::Vector6d::Zero();
        arm_base_desired_vel(2) =
          kp_position_ * (arm_base_goal_pose_.translation().z() - arm_base_pose.translation().z());
        desired_task_velocities[arm_base_name_] = arm_base_desired_vel;
      }

      std::map<std::string, cyclo_motion_controller::common::Vector6d> weights;
      cyclo_motion_controller::common::Vector6d weight_right =
        cyclo_motion_controller::common::Vector6d::Zero();
      cyclo_motion_controller::common::Vector6d weight_left =
        cyclo_motion_controller::common::Vector6d::Zero();
      weight_right.head(3).setConstant(weight_position_);
      weight_right.tail(3).setConstant(weight_orientation_);
      weight_left.head(3).setConstant(weight_position_);
      weight_left.tail(3).setConstant(weight_orientation_);
      weights[r_gripper_name_] = weight_right;
      weights[l_gripper_name_] = weight_left;

      if (right_elbow_active) {
        cyclo_motion_controller::common::Vector6d weight_right_elbow =
          cyclo_motion_controller::common::Vector6d::Zero();
        weight_right_elbow.head(3).setConstant(weight_elbow_position_);
        weights[r_elbow_name_] = weight_right_elbow;
      }
      if (left_elbow_active) {
        cyclo_motion_controller::common::Vector6d weight_left_elbow =
          cyclo_motion_controller::common::Vector6d::Zero();
        weight_left_elbow.head(3).setConstant(weight_elbow_position_);
        weights[l_elbow_name_] = weight_left_elbow;
      }
      if (arm_base_active) {
        cyclo_motion_controller::common::Vector6d weight_arm_base =
          cyclo_motion_controller::common::Vector6d::Zero();
        weight_arm_base(2) = weight_arm_base_position_;
        weights[arm_base_name_] = weight_arm_base;
      }

      Eigen::VectorXd damping =
        Eigen::VectorXd::Ones(active_solver->getDof()) * weight_damping_;
      active_controller->setWeight(weights, damping);
      active_controller->setDesiredTaskVel(desired_task_velocities);
      active_controller->setHardConstraintTaskVel(hard_constraint_task_velocities);

      Eigen::VectorXd optimal_velocities;
      if (!active_controller->getOptJointVel(optimal_velocities)) {
        return;
      }

      q_desired_ = solver_state + optimal_velocities * dt_;
      Eigen::VectorXd q_command = q_desired_;
      applyLiftCommandBlend(
        q_command, now, fixed_base_phase_active && goal_ref_moving_recently);
      publishTrajectory(q_command);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Error in control loop: %s", e.what());
    }
  }

  void updateLiftVelocityBound(const double velocity_bound)
  {
    if (lift_joint_index_ < 0) {
      return;
    }

    const double clamped_bound = std::max(0.0, velocity_bound);
    if (std::abs(clamped_bound - current_lift_vel_bound_) < 1e-9) {
      return;
    }

    (void)kinematics_solver_->setJointVelocityBoundsByIndex(
      lift_joint_index_, -clamped_bound, clamped_bound);
    if (fixed_base_kinematics_solver_) {
      (void)fixed_base_kinematics_solver_->setJointVelocityBoundsByIndex(
        lift_joint_index_, -clamped_bound, clamped_bound);
    }
    current_lift_vel_bound_ = clamped_bound;
  }

  void updateGoalMotionState(
    const Eigen::Affine3d & next_pose,
    const Eigen::Affine3d * previous_pose)
  {
    if (previous_pose == nullptr) {
      last_goal_motion_time_ = this->now();
      return;
    }

    const double pos_delta = (next_pose.translation() - previous_pose->translation()).norm();
    const double ori_delta =
      Eigen::Quaterniond(next_pose.linear()).angularDistance(Eigen::Quaterniond(previous_pose->linear()));
    if (
      pos_delta > goal_motion_position_threshold_ ||
      ori_delta > goal_motion_orientation_threshold_)
    {
      const rclcpp::Time now = this->now();
      if ((now - last_goal_motion_time_).seconds() >= goal_motion_detect_timeout_) {
        lift_transition_active_ = false;
      }
      last_goal_motion_time_ = now;
    }
  }

  void blendLiftStateForGoalTransition(
    Eigen::VectorXd & q_feedback,
    const rclcpp::Time & now,
    const bool should_blend)
  {
    if (lift_joint_index_ < 0 || lift_joint_index_ >= q_feedback.size()) {
      return;
    }

    if (!should_blend) {
      lift_transition_active_ = false;
      return;
    }

    if (!lift_transition_active_) {
      lift_transition_active_ = true;
      lift_transition_start_time_ = now;
      lift_transition_start_position_ = q_[lift_joint_index_];
      lift_transition_target_position_ = q_feedback[lift_joint_index_];
    } else {
      lift_transition_target_position_ = q_feedback[lift_joint_index_];
    }

    const double alpha = std::clamp(
      (now - lift_transition_start_time_).seconds() /
      std::max(lift_transition_blend_duration_, 1e-6),
      0.0, 1.0);
    q_feedback[lift_joint_index_] =
      (1.0 - alpha) * lift_transition_start_position_ +
      alpha * lift_transition_target_position_;
  }

  void applyLiftCommandBlend(
    Eigen::VectorXd & q_command,
    const rclcpp::Time & now,
    const bool should_blend) const
  {
    if (
      !should_blend ||
      lift_joint_index_ < 0 ||
      lift_joint_index_ >= q_command.size() ||
      !lift_transition_active_)
    {
      return;
    }

    const double alpha = std::clamp(
      (now - lift_transition_start_time_).seconds() /
      std::max(lift_transition_blend_duration_, 1e-6),
      0.0, 1.0);
    q_command[lift_joint_index_] =
      (1.0 - alpha) * q_[lift_joint_index_] +
      alpha * q_command[lift_joint_index_];
  }

  Eigen::Affine3d computePoseMat(const geometry_msgs::msg::PoseStamped & pose) const
  {
    Eigen::Affine3d pose_mat = Eigen::Affine3d::Identity();
    pose_mat.translation() << pose.pose.position.x, pose.pose.position.y, pose.pose.position.z;
    const Eigen::Quaterniond quat(
      pose.pose.orientation.w,
      pose.pose.orientation.x,
      pose.pose.orientation.y,
      pose.pose.orientation.z);
    pose_mat.linear() = quat.toRotationMatrix();
    return pose_mat;
  }

  cyclo_motion_controller::common::Vector6d computeDesiredVelocity(
    const Eigen::Affine3d & current_pose,
    const Eigen::Affine3d & goal_pose) const
  {
    const Eigen::Vector3d pos_error = goal_pose.translation() - current_pose.translation();
    const Eigen::Matrix3d rotation_error = goal_pose.linear() * current_pose.linear().transpose();
    const Eigen::AngleAxisd angle_axis_error(rotation_error);
    const Eigen::Vector3d angle_axis = angle_axis_error.axis() * angle_axis_error.angle();

    cyclo_motion_controller::common::Vector6d desired_vel =
      cyclo_motion_controller::common::Vector6d::Zero();
    desired_vel.head(3) = kp_position_ * pos_error;
    desired_vel.tail(3) = kp_orientation_ * angle_axis;
    return desired_vel;
  }

  trajectory_msgs::msg::JointTrajectory createTrajectoryMsgWithGripper(
    const std::vector<std::string> & arm_joint_names,
    const Eigen::VectorXd & positions,
    const std::vector<int> & arm_indices,
    const std::string & gripper_joint_name,
    const double gripper_position) const
  {
    trajectory_msgs::msg::JointTrajectory traj_msg;
    traj_msg.joint_names = arm_joint_names;
    // traj_msg.joint_names.push_back(gripper_joint_name);

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);
    for (int idx : arm_indices) {
      if (idx >= 0 && idx < positions.size()) {
        point.positions.push_back(positions[idx]);
      }
    }
    // point.positions.push_back(gripper_position);
    traj_msg.points.push_back(point);
    return traj_msg;
  }

  trajectory_msgs::msg::JointTrajectory createLiftTrajectoryMsg(
    const std::string & lift_joint_name,
    const double position) const
  {
    trajectory_msgs::msg::JointTrajectory traj_msg;
    traj_msg.joint_names = {lift_joint_name};
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);
    point.positions = {position};
    traj_msg.points.push_back(point);
    return traj_msg;
  }

  void publishTrajectory(const Eigen::VectorXd & q_desired)
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
      double gripper_pos = 0.0;
      if (left_raw_gripper_received_ &&
        (this->now() - last_left_raw_traj_time_).seconds() < raw_traj_timeout_)
      {
        gripper_pos = left_raw_gripper_position_;
      }
      arm_l_pub_->publish(createTrajectoryMsgWithGripper(
        left_arm_joints_, q_desired, left_arm_indices, left_gripper_joint_name_, gripper_pos));
    }

    if (!right_arm_indices.empty()) {
      double gripper_pos = 0.0;
      if (right_raw_gripper_received_ &&
        (this->now() - last_right_raw_traj_time_).seconds() < raw_traj_timeout_)
      {
        gripper_pos = right_raw_gripper_position_;
      }
      arm_r_pub_->publish(createTrajectoryMsgWithGripper(
        right_arm_joints_, q_desired, right_arm_indices, right_gripper_joint_name_, gripper_pos));
    }

    if (lift_joint_index_ >= 0 &&
      !lift_joint_name_.empty() &&
      lift_joint_index_ < q_desired.size() &&
      lift_vel_bound_ > 0.0)
    {
      lift_pub_->publish(createLiftTrajectoryMsg(lift_joint_name_, q_desired[lift_joint_index_]));
    }
  }

  void publishGripperPose(
    const Eigen::Affine3d & r_gripper_pose,
    const Eigen::Affine3d & l_gripper_pose)
  {
    geometry_msgs::msg::PoseStamped r_msg;
    r_msg.header.stamp = this->now();
    r_msg.header.frame_id = "base_link";
    r_msg.pose.position.x = r_gripper_pose.translation().x();
    r_msg.pose.position.y = r_gripper_pose.translation().y();
    r_msg.pose.position.z = r_gripper_pose.translation().z();
    const Eigen::Quaterniond r_q(r_gripper_pose.linear());
    r_msg.pose.orientation.w = r_q.w();
    r_msg.pose.orientation.x = r_q.x();
    r_msg.pose.orientation.y = r_q.y();
    r_msg.pose.orientation.z = r_q.z();
    r_gripper_pose_pub_->publish(r_msg);

    geometry_msgs::msg::PoseStamped l_msg;
    l_msg.header.stamp = this->now();
    l_msg.header.frame_id = "base_link";
    l_msg.pose.position.x = l_gripper_pose.translation().x();
    l_msg.pose.position.y = l_gripper_pose.translation().y();
    l_msg.pose.position.z = l_gripper_pose.translation().z();
    const Eigen::Quaterniond l_q(l_gripper_pose.linear());
    l_msg.pose.orientation.w = l_q.w();
    l_msg.pose.orientation.x = l_q.x();
    l_msg.pose.orientation.y = l_q.y();
    l_msg.pose.orientation.z = l_q.z();
    l_gripper_pose_pub_->publish(l_msg);
  }

  double control_frequency_;
  double time_step_;
  double trajectory_time_;
  double kp_position_;
  double kp_orientation_;
  double weight_position_;
  double weight_orientation_;
  double weight_elbow_position_;
  double weight_arm_base_position_;
  double weight_damping_;
  double slack_penalty_;
  double cbf_alpha_;
  double collision_buffer_;
  double collision_safe_distance_;
  double raw_traj_timeout_;
  double goal_ref_timeout_;
  double elbow_ref_timeout_;
  double arm_base_ref_timeout_;
  double goal_motion_detect_timeout_;
  double goal_motion_position_threshold_;
  double goal_motion_orientation_threshold_;
  double lift_transition_blend_duration_;
  double arm_base_motion_detect_timeout_;
  double arm_base_motion_detect_threshold_;
  double lift_vel_bound_;
  double lift_vel_bound_fixed_base_;
  double current_lift_vel_bound_ = -1.0;

  std::string r_goal_pose_topic_;
  std::string l_goal_pose_topic_;
  std::string r_elbow_pose_topic_;
  std::string l_elbow_pose_topic_;
  std::string arm_base_goal_pose_topic_;
  std::string joint_states_topic_;
  std::string right_traj_topic_;
  std::string left_traj_topic_;
  std::string right_raw_traj_topic_;
  std::string left_raw_traj_topic_;
  std::string lift_topic_;
  std::string lift_joint_name_;
  std::string r_gripper_pose_topic_;
  std::string l_gripper_pose_topic_;
  std::string r_gripper_name_;
  std::string l_gripper_name_;
  std::string r_elbow_name_;
  std::string l_elbow_name_;
  std::string arm_base_name_;
  std::string right_gripper_joint_name_;
  std::string left_gripper_joint_name_;
  std::string urdf_path_;
  std::string srdf_path_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr l_goal_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr r_elbow_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr l_elbow_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr arm_base_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr right_raw_traj_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr left_raw_traj_sub_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_l_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr lift_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_gripper_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_gripper_pose_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> kinematics_solver_;
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver>
    fixed_base_kinematics_solver_;
  std::shared_ptr<cyclo_motion_controller::controllers::VRController> qp_controller_;
  std::shared_ptr<cyclo_motion_controller::controllers::VRController> fixed_base_qp_controller_;

  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd q_desired_;

  Eigen::Affine3d r_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d l_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d r_elbow_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d l_elbow_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d arm_base_goal_pose_ = Eigen::Affine3d::Identity();

  bool r_goal_pose_received_;
  bool l_goal_pose_received_;
  bool r_elbow_pose_received_;
  bool l_elbow_pose_received_;
  bool arm_base_pose_received_;
  bool joint_state_received_;
  bool right_raw_gripper_received_ = false;
  bool left_raw_gripper_received_ = false;

  double right_raw_gripper_position_ = 0.0;
  double left_raw_gripper_position_ = 0.0;
  rclcpp::Time last_right_raw_traj_time_;
  rclcpp::Time last_left_raw_traj_time_;
  rclcpp::Time last_r_goal_pose_time_;
  rclcpp::Time last_l_goal_pose_time_;
  rclcpp::Time last_r_elbow_pose_time_;
  rclcpp::Time last_l_elbow_pose_time_;
  rclcpp::Time last_arm_base_pose_time_;
  rclcpp::Time last_goal_motion_time_;
  rclcpp::Time last_arm_base_motion_time_;
  rclcpp::Time lift_transition_start_time_;

  double dt_;
  double lift_transition_start_position_ = 0.0;
  double lift_transition_target_position_ = 0.0;
  bool lift_transition_active_ = false;
  std::vector<std::string> left_arm_joints_;
  std::vector<std::string> right_arm_joints_;
  int lift_joint_index_ = -1;
  std::map<std::string, int> joint_index_map_;
  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> model_joint_index_map_;
};
}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyclo_motion_controller_ros::WholebodyController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
