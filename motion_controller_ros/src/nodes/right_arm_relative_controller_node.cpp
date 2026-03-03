#include "motion_controller_ros/nodes/right_arm_relative_controller_node.hpp"

#include <algorithm>
#include <cmath>

namespace motion_controller_ros
{
RightArmRelativeController::RightArmRelativeController()
  : Node("right_arm_relative_controller")
{
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "Right Arm Relative Controller - Starting up...");
  RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
  RCLCPP_INFO(this->get_logger(), "========================================");

  control_frequency_ = this->declare_parameter("control_frequency", control_frequency_);
  time_step_ = this->declare_parameter("time_step", time_step_);
  trajectory_time_ = this->declare_parameter("trajectory_time", trajectory_time_);
  kp_position_ = this->declare_parameter("kp_position", kp_position_);
  kp_orientation_ = this->declare_parameter("kp_orientation", kp_orientation_);
  weight_position_ = this->declare_parameter("weight_position", weight_position_);
  weight_orientation_ = this->declare_parameter("weight_orientation", weight_orientation_);
  weight_damping_ = this->declare_parameter("weight_damping", weight_damping_);
  slack_penalty_ = this->declare_parameter("slack_penalty", slack_penalty_);
  cbf_alpha_ = this->declare_parameter("cbf_alpha", cbf_alpha_);
  collision_buffer_ = this->declare_parameter("collision_buffer", collision_buffer_);
  collision_safe_distance_ = this->declare_parameter("collision_safe_distance", collision_safe_distance_);
  command_hz_ = this->declare_parameter("command_hz", command_hz_);
  delta_in_ee_frame_ = this->declare_parameter("delta_in_ee_frame", delta_in_ee_frame_);
  lock_other_joints_ = this->declare_parameter("lock_other_joints", lock_other_joints_);

  delta_pose_topic_ = this->declare_parameter("delta_pose_topic", delta_pose_topic_);
  joint_states_topic_ = this->declare_parameter("joint_states_topic", joint_states_topic_);
  right_traj_topic_ = this->declare_parameter("right_traj_topic", right_traj_topic_);
  right_raw_traj_topic_ = this->declare_parameter("right_raw_traj_topic", right_raw_traj_topic_);
  raw_traj_timeout_ = this->declare_parameter("raw_traj_timeout", raw_traj_timeout_);
  r_gripper_pose_topic_ = this->declare_parameter("r_gripper_pose_topic", r_gripper_pose_topic_);
  base_frame_id_ = this->declare_parameter("base_frame_id", base_frame_id_);
  r_gripper_name_ = this->declare_parameter("r_gripper_name", r_gripper_name_);
  right_gripper_joint_name_ = this->declare_parameter("right_gripper_joint", right_gripper_joint_name_);
  const std::string urdf_path = this->declare_parameter("urdf_path", std::string(""));
  const std::string srdf_path = this->declare_parameter("srdf_path", std::string(""));

  dt_ = time_step_;
  last_right_raw_traj_time_ = this->now();

  if (urdf_path.empty() || srdf_path.empty()) {
    RCLCPP_FATAL(this->get_logger(), "URDF/SRDF paths must be provided via parameters (urdf_path, srdf_path)");
    rclcpp::shutdown();
    return;
  }

  try {
    RCLCPP_INFO(this->get_logger(), "Loading URDF and initializing kinematics solver...");
    kinematics_solver_ = std::make_shared<motion_controller::kinematics::KinematicsSolver>(urdf_path, srdf_path);
    RCLCPP_INFO(this->get_logger(), "Initializing QP IK controller...");
    qp_controller_ = std::make_shared<motion_controller::controllers::QPIK>(kinematics_solver_, dt_);
    qp_controller_->setControllerParams(slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);

    const int dof = kinematics_solver_->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);
    q_desired_.setZero(dof);
    RCLCPP_INFO(this->get_logger(), "Controller initialized (DOF: %d)", dof);
  } catch (const std::exception &e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize controller: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  initializeJointConfig();
  lockNonRightArmJointsIfRequested();

  // Subscribers
  delta_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    delta_pose_topic_, 10,
    std::bind(&RightArmRelativeController::deltaPoseCallback, this, std::placeholders::_1));

  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    joint_states_topic_, 10,
    std::bind(&RightArmRelativeController::jointStateCallback, this, std::placeholders::_1));

  right_raw_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    right_raw_traj_topic_, 10,
    std::bind(&RightArmRelativeController::rightRawTrajectoryCallback, this, std::placeholders::_1));

  // Publishers
  arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(right_traj_topic_, 10);
  r_gripper_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(r_gripper_pose_topic_, 10);

  // Control loop
  const int timer_period_ms = std::max(1, static_cast<int>(std::round(1000.0 / control_frequency_)));
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(timer_period_ms),
    std::bind(&RightArmRelativeController::controlLoopCallback, this));

  RCLCPP_INFO(this->get_logger(), "Right arm relative controller ready.");
  RCLCPP_INFO(this->get_logger(), "  - Control loop: %.1f Hz (period: %d ms)", control_frequency_, timer_period_ms);
  RCLCPP_INFO(this->get_logger(), "  - Delta pose topic: %s (command_hz=%.1f)", delta_pose_topic_.c_str(), command_hz_);
  RCLCPP_INFO(this->get_logger(), "  - Right traj topic: %s", right_traj_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "  - Right EE link: %s", r_gripper_name_.c_str());
  RCLCPP_INFO(this->get_logger(), "========================================");
}

RightArmRelativeController::~RightArmRelativeController()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down Right Arm Relative Controller");
}

void RightArmRelativeController::initializeJointConfig()
{
  const auto joint_names = kinematics_solver_->getJointNames();
  model_joint_names_ = joint_names;
  model_joint_index_map_.clear();
  for (size_t i = 0; i < model_joint_names_.size(); ++i) {
    model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
  }

  right_arm_joints_.clear();
  for (const auto &joint_name : joint_names) {
    if (joint_name.find("arm_r_joint") != std::string::npos) {
      right_arm_joints_.push_back(joint_name);
    }
  }
  std::sort(right_arm_joints_.begin(), right_arm_joints_.end());

  right_arm_indices_.clear();
  right_arm_indices_.reserve(right_arm_joints_.size());
  for (const auto &jn : right_arm_joints_) {
    auto it = model_joint_index_map_.find(jn);
    if (it != model_joint_index_map_.end()) {
      right_arm_indices_.push_back(it->second);
    }
  }
  std::sort(right_arm_indices_.begin(), right_arm_indices_.end());
  right_arm_indices_.erase(
    std::unique(right_arm_indices_.begin(), right_arm_indices_.end()),
    right_arm_indices_.end());

  if (right_arm_joints_.empty()) {
    RCLCPP_WARN(this->get_logger(),
      "No right arm joints detected (expected names containing 'arm_r_joint').");
  }
}

void RightArmRelativeController::lockNonRightArmJointsIfRequested()
{
  if (!lock_other_joints_) {
    return;
  }

  const int dof = kinematics_solver_->getDof();
  int locked = 0;
  for (int i = 0; i < dof; ++i) {
    if (std::binary_search(right_arm_indices_.begin(), right_arm_indices_.end(), i)) {
      continue;
    }
    if (kinematics_solver_->setJointVelocityBoundsByIndex(i, 0.0, 0.0)) {
      locked++;
    }
  }

  RCLCPP_INFO(this->get_logger(), "Locked %d non-right-arm joint velocity bounds to zero.", locked);
}

void RightArmRelativeController::rightRawTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!msg || msg->points.empty()) {
    return;
  }
  const auto &point = msg->points.front();
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

void RightArmRelativeController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (!msg) {
    return;
  }

  if (joint_index_map_.empty()) {
    joint_index_map_.reserve(msg->name.size());
    for (size_t i = 0; i < msg->name.size(); ++i) {
      joint_index_map_[msg->name[i]] = static_cast<int>(i);
    }
  }

  extractJointStates(msg);
  joint_state_received_ = true;

  if (!ee_seeded_) {
    seedGoalPoseFromJointStateOnce();
  }
}

void RightArmRelativeController::extractJointStates(const sensor_msgs::msg::JointState::SharedPtr &msg)
{
  const int dof = kinematics_solver_->getDof();
  q_.setZero(dof);
  qdot_.setZero(dof);

  const int max_index = std::min<int>(dof, static_cast<int>(model_joint_names_.size()));
  for (int i = 0; i < max_index; ++i) {
    const auto &joint_name = model_joint_names_[i];
    auto it = joint_index_map_.find(joint_name);
    if (it == joint_index_map_.end()) {
      continue;
    }
    const int idx = it->second;
    if (idx >= 0 && idx < static_cast<int>(msg->position.size())) {
      q_[i] = msg->position[idx];
    }
    if (idx >= 0 && idx < static_cast<int>(msg->velocity.size())) {
      qdot_[i] = msg->velocity[idx];
    }
  }
}

void RightArmRelativeController::seedGoalPoseFromJointStateOnce()
{
  try {
    kinematics_solver_->updateState(q_, qdot_);
    current_pose_ = kinematics_solver_->getPose(r_gripper_name_);
    goal_pose_ = current_pose_;
    interp_active_ = false;
    interp_steps_total_ = 1;
    interp_step_idx_ = 0;
    q_desired_ = q_;
    ee_seeded_ = true;
    publishGripperPose(current_pose_);
    RCLCPP_INFO(this->get_logger(),
      "Seeded right EE goal pose from joint state FK (%s).", r_gripper_name_.c_str());
  } catch (const std::exception &e) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "Failed to seed EE pose from joint state: %s", e.what());
  }
}

RightArmRelativeController::Affine3d RightArmRelativeController::poseMsgToAffine(const geometry_msgs::msg::Pose &pose)
{
  Affine3d a = Affine3d::Identity();
  a.translation() << pose.position.x, pose.position.y, pose.position.z;

  Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  if (q.norm() < 1e-9) {
    q = Quaterniond::Identity();
  } else {
    q.normalize();
  }
  a.linear() = q.toRotationMatrix();
  return a;
}

RightArmRelativeController::Affine3d RightArmRelativeController::applyDeltaPose(
  const Affine3d &start, const geometry_msgs::msg::Pose &delta) const
{
  const Affine3d d = poseMsgToAffine(delta);
  Affine3d out = start;

  const Vector3d dp = d.translation();
  const Matrix3d dR = d.linear();

  if (delta_in_ee_frame_) {
    out.translation() = start.translation() + start.linear() * dp;
    out.linear() = start.linear() * dR;
  } else {
    out.translation() = start.translation() + dp;
    out.linear() = dR * start.linear();
  }

  return out;
}

RightArmRelativeController::Affine3d RightArmRelativeController::interpolatePose(
  const Affine3d &a, const Affine3d &b, double alpha)
{
  const double t = std::clamp(alpha, 0.0, 1.0);
  Affine3d out = Affine3d::Identity();

  out.translation() = (1.0 - t) * a.translation() + t * b.translation();

  Quaterniond qa(a.linear());
  Quaterniond qb(b.linear());
  if (qa.norm() < 1e-9) qa = Quaterniond::Identity();
  if (qb.norm() < 1e-9) qb = Quaterniond::Identity();
  qa.normalize();
  qb.normalize();
  const Quaterniond q = qa.slerp(t, qb);
  out.linear() = q.toRotationMatrix();

  return out;
}

void RightArmRelativeController::deltaPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  if (!ee_seeded_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "Ignoring delta command until EE is seeded from joint states.");
    return;
  }
  if (!msg->header.frame_id.empty() && msg->header.frame_id != base_frame_id_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "Delta command frame_id='%s' != base_frame_id='%s'. Applying anyway.",
      msg->header.frame_id.c_str(), base_frame_id_.c_str());
  }

  interp_start_ = goal_pose_;
  interp_end_ = applyDeltaPose(interp_start_, msg->pose);

  const double hz = std::max(0.1, command_hz_);
  const int steps = std::max(1, static_cast<int>(std::round(control_frequency_ / hz)));
  interp_steps_total_ = steps;
  interp_step_idx_ = 0;
  interp_active_ = true;
}

void RightArmRelativeController::stepInterpolatedGoalPose()
{
  if (!interp_active_) {
    return;
  }
  interp_step_idx_++;
  const double alpha = static_cast<double>(interp_step_idx_) / static_cast<double>(interp_steps_total_);
  goal_pose_ = interpolatePose(interp_start_, interp_end_, alpha);
  if (interp_step_idx_ >= interp_steps_total_) {
    interp_active_ = false;
    goal_pose_ = interp_end_;
  }
}

motion_controller::common::Vector6d RightArmRelativeController::computeDesiredVelocity(
  const Affine3d &current_pose,
  const Affine3d &goal_pose) const
{
  const Vector3d pos_error = goal_pose.translation() - current_pose.translation();

  const Matrix3d rotation_error = goal_pose.linear() * current_pose.linear().transpose();
  const Eigen::AngleAxisd angle_axis_error(rotation_error);
  const Vector3d angle_axis = angle_axis_error.axis() * angle_axis_error.angle();

  motion_controller::common::Vector6d desired_vel = motion_controller::common::Vector6d::Zero();
  desired_vel.head(3) = kp_position_ * pos_error;
  desired_vel.tail(3) = kp_orientation_ * angle_axis;
  return desired_vel;
}

void RightArmRelativeController::publishGripperPose(const Affine3d &pose) const
{
  if (!r_gripper_pose_pub_) {
    return;
  }
  geometry_msgs::msg::PoseStamped msg;
  msg.header.stamp = this->now();
  msg.header.frame_id = base_frame_id_;
  msg.pose.position.x = pose.translation().x();
  msg.pose.position.y = pose.translation().y();
  msg.pose.position.z = pose.translation().z();
  const Quaterniond q(pose.linear());
  msg.pose.orientation.w = q.w();
  msg.pose.orientation.x = q.x();
  msg.pose.orientation.y = q.y();
  msg.pose.orientation.z = q.z();
  r_gripper_pose_pub_->publish(msg);
}

void RightArmRelativeController::controlLoopCallback()
{
  static int debug_count = 0;

  if (!joint_state_received_) {
    if (debug_count++ % 100 == 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Waiting for joint states on %s...", joint_states_topic_.c_str());
    }
    return;
  }
  if (!ee_seeded_) {
    if (debug_count++ % 100 == 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Waiting to seed EE pose from joint states...");
    }
    return;
  }

  try {
    // Update interpolated goal (per control tick)
    stepInterpolatedGoalPose();

    // Use last commanded joints as feedback state for stable IK integration
    VectorXd q_feedback = (q_desired_.size() == q_.size()) ? q_desired_ : q_;
    if (q_feedback.size() == q_.size() && !right_arm_indices_.empty()) {
      for (int i = 0; i < q_feedback.size(); ++i) {
        if (!std::binary_search(right_arm_indices_.begin(), right_arm_indices_.end(), i)) {
          q_feedback[i] = q_[i];
        }
      }
    }
    kinematics_solver_->updateState(q_feedback, qdot_);

    current_pose_ = kinematics_solver_->getPose(r_gripper_name_);
    publishGripperPose(current_pose_);

    const motion_controller::common::Vector6d desired_vel =
      computeDesiredVelocity(current_pose_, goal_pose_);

    std::map<std::string, motion_controller::common::Vector6d> desired_task_velocities;
    desired_task_velocities[r_gripper_name_] = desired_vel;

    motion_controller::common::Vector6d w = motion_controller::common::Vector6d::Ones();
    w.head(3).setConstant(weight_position_);
    w.tail(3).setConstant(weight_orientation_);

    std::map<std::string, motion_controller::common::Vector6d> weights;
    weights[r_gripper_name_] = w;

    VectorXd damping = VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;
    qp_controller_->setWeight(weights, damping);
    qp_controller_->setDesiredTaskVel(desired_task_velocities);

    VectorXd optimal_velocities;
    if (!qp_controller_->getOptJointVel(optimal_velocities)) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "QP IK failed to converge");
      return;
    }

    q_desired_ = q_feedback + optimal_velocities * dt_;
    publishTrajectory(q_desired_);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "Control loop error: %s", e.what());
  }
}

void RightArmRelativeController::publishTrajectory(const VectorXd &q_desired)
{
  if (!arm_r_pub_) {
    return;
  }

  std::vector<int> right_arm_indices;
  right_arm_indices.reserve(right_arm_joints_.size());
  for (const auto &joint_name : right_arm_joints_) {
    auto it = model_joint_index_map_.find(joint_name);
    if (it != model_joint_index_map_.end()) {
      right_arm_indices.push_back(it->second);
    }
  }

  if (right_arm_indices.empty()) {
    return;
  }

  double gripper_pos = 0.0;
  if (right_raw_gripper_received_ &&
      (this->now() - last_right_raw_traj_time_).seconds() < raw_traj_timeout_) {
    gripper_pos = right_raw_gripper_position_;
  }

  const auto traj = createTrajectoryMsgWithGripper(
    right_arm_joints_, q_desired, right_arm_indices, right_gripper_joint_name_, gripper_pos);
  arm_r_pub_->publish(traj);
}

trajectory_msgs::msg::JointTrajectory RightArmRelativeController::createTrajectoryMsgWithGripper(
  const std::vector<std::string> &arm_joint_names,
  const VectorXd &positions,
  const std::vector<int> &arm_indices,
  const std::string &gripper_joint_name,
  const double gripper_position) const
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
}  // namespace motion_controller_ros

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<motion_controller_ros::RightArmRelativeController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

