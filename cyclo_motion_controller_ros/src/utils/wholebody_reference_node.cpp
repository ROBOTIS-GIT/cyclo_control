#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace cyclo_motion_controller_ros
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

double clamp01(const double value)
{
  return std::clamp(value, 0.0, 1.0);
}

Eigen::Vector3d projectToPlane(
  const Eigen::Vector3d & vector,
  const Eigen::Vector3d & plane_normal)
{
  return vector - vector.dot(plane_normal) * plane_normal;
}

Eigen::Vector3d normalizedOrFallback(
  const Eigen::Vector3d & vector,
  const Eigen::Vector3d & fallback)
{
  if (vector.norm() > 1e-8) {
    return vector.normalized();
  }
  return fallback.normalized();
}

Eigen::Affine3d makePose(
  const Eigen::Vector3d & translation,
  const Eigen::Quaterniond & orientation)
{
  Eigen::Affine3d pose = Eigen::Affine3d::Identity();
  pose.translation() = translation;
  pose.linear() = orientation.normalized().toRotationMatrix();
  return pose;
}

Eigen::Affine3d interpolatePose(
  const Eigen::Affine3d & start_pose,
  const Eigen::Affine3d & goal_pose,
  const double alpha)
{
  const double clamped_alpha = clamp01(alpha);
  const Eigen::Vector3d translation =
    (1.0 - clamped_alpha) * start_pose.translation() +
    clamped_alpha * goal_pose.translation();
  const Eigen::Quaterniond q_start(start_pose.linear());
  const Eigen::Quaterniond q_goal(goal_pose.linear());
  return makePose(translation, q_start.slerp(clamped_alpha, q_goal));
}

struct ElbowArc
{
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d u = Eigen::Vector3d::UnitY();
  Eigen::Vector3d v = Eigen::Vector3d::UnitZ();
  double radius = 0.0;
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

enum class Phase
{
  InitialMove,
  WaitCycle1Pose,
  RedundancyFixedBase,
  WaitCycle2Pose,
  ArmBaseMotion,
  WaitCycle3Pose,
  RedundancyNaturalBase,
  ReturnToInitialPose,
  WaitOrientationComplete
};
}  // namespace

class WholebodyReferenceNode : public rclcpp::Node
{
public:
  WholebodyReferenceNode()
  : Node("wholebody_reference"),
    initialized_(false),
    fixed_base_pose_captured_(false),
    redundancy_capture_ready_(false),
    arm_base_phase_captured_(false),
    phase_(Phase::InitialMove)
  {
    control_frequency_ = this->declare_parameter("control_frequency", 100.0);
    initial_move_duration_ = this->declare_parameter("initial_move_duration", 4.0);
    redundancy_cycle1_duration_ = this->declare_parameter("redundancy_cycle1_duration", 3.5);
    redundancy_cycle2_duration_ = this->declare_parameter("redundancy_cycle2_duration", 3.5);
    redundancy_phase1_cycles_ = this->declare_parameter("redundancy_phase1_cycles", 3);
    redundancy_phase2_cycles_ = this->declare_parameter("redundancy_phase2_cycles", 2);
    arm_base_cycle_duration_ = this->declare_parameter("arm_base_cycle_duration", 3.0);
    arm_base_phase_cycles_ = this->declare_parameter("arm_base_phase_cycles", 2);
    arm_base_entry_duration_ = this->declare_parameter("arm_base_entry_duration", 1.0);
    elbow_arc_angle_deg_ = this->declare_parameter("elbow_arc_angle_deg", 30.0);
    phase1_return_past_start_ratio_ =
      this->declare_parameter("phase1_return_past_start_ratio", 0.35);
    minimum_elbow_radius_ = this->declare_parameter("minimum_elbow_radius", 0.0);
    transition_move_duration_ = this->declare_parameter("transition_move_duration", 2.0);
    arm_base_transition_duration_ = this->declare_parameter("arm_base_transition_duration", 2.0);
    cycle2_transition_outward_y_offset_ =
      this->declare_parameter("cycle2_transition_outward_y_offset", 0.08);
    gripper_position_tolerance_ = this->declare_parameter("gripper_position_tolerance", 0.02);
    gripper_orientation_tolerance_deg_ =
      this->declare_parameter("gripper_orientation_tolerance_deg", 30.0);
    arm_base_upper_z_offset_ = this->declare_parameter("arm_base_upper_z_offset", 0.0);
    arm_base_lower_z_offset_ = this->declare_parameter("arm_base_lower_z_offset", -0.15);
    base_frame_id_ = this->declare_parameter("base_frame_id", std::string("base_link"));
    orientation_start_topic_ = this->declare_parameter(
      "orientation_start_topic", std::string("/motion_demo/start_orientation"));
    orientation_done_topic_ = this->declare_parameter(
      "orientation_done_topic", std::string("/motion_demo/orientation_done"));

    r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
    l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));
    r_elbow_pose_topic_ = this->declare_parameter("r_elbow_pose_topic", std::string("/r_elbow_pose"));
    l_elbow_pose_topic_ = this->declare_parameter("l_elbow_pose_topic", std::string("/l_elbow_pose"));
    arm_base_goal_pose_topic_ = this->declare_parameter(
      "arm_base_goal_pose_topic", std::string("/arm_base_goal_pose"));

    right_gripper_frame_ = this->declare_parameter(
      "right_gripper_frame", std::string("end_effector_r_link"));
    left_gripper_frame_ = this->declare_parameter(
      "left_gripper_frame", std::string("end_effector_l_link"));
    right_shoulder_frame_ = this->declare_parameter(
      "right_shoulder_frame", std::string("arm_r_link1"));
    left_shoulder_frame_ = this->declare_parameter(
      "left_shoulder_frame", std::string("arm_l_link1"));
    right_elbow_frame_ = this->declare_parameter(
      "right_elbow_frame", std::string("arm_r_link4"));
    left_elbow_frame_ = this->declare_parameter(
      "left_elbow_frame", std::string("arm_l_link4"));
    right_wrist_frame_ = this->declare_parameter(
      "right_wrist_frame", std::string("arm_r_link7"));
    left_wrist_frame_ = this->declare_parameter(
      "left_wrist_frame", std::string("arm_l_link7"));
    arm_base_frame_ = this->declare_parameter("arm_base_frame", std::string("arm_base_link"));

    cycle1_r_goal_pose_ = declarePoseParameter(
      "cycle1_right_target_pose",
      makePose(
        Eigen::Vector3d(0.39392780661582947, -0.15, 0.85),
        Eigen::Quaterniond(0.9297504425048828, 0.0, -0.3681904673576355, 0.0)));
    cycle1_l_goal_pose_ = declarePoseParameter(
      "cycle1_left_target_pose",
      makePose(
        Eigen::Vector3d(0.39392780661582947, 0.15, 0.85),
        Eigen::Quaterniond(0.9297504425048828, 0.0, -0.3681904673576355, 0.0)));
    orientation_start_r_goal_pose_ = declarePoseParameter(
      "orientation_start_right_target_pose",
      makePose(
        Eigen::Vector3d(0.5, -0.22, 0.85),
        Eigen::Quaterniond(0.707, 0.0, -0.707, 0.0)));
    orientation_start_l_goal_pose_ = declarePoseParameter(
      "orientation_start_left_target_pose",
      makePose(
        Eigen::Vector3d(0.5, 0.22, 0.85),
        Eigen::Quaterniond(0.707, 0.0, -0.707, 0.0)));
    cycle2_r_goal_pose_ = declarePoseParameter(
      "cycle2_right_target_pose",
      makePose(Eigen::Vector3d(0.3, 0.0, 0.85), Eigen::Quaterniond(0.5, 0.5, -0.5, 0.5)));
    cycle2_l_goal_pose_ = declarePoseParameter(
      "cycle2_left_target_pose",
      makePose(Eigen::Vector3d(0.3, 0.0, 0.95), Eigen::Quaterniond(0.5, -0.5, -0.5, -0.5)));
    cycle3_r_goal_pose_ = declarePoseParameter(
      "cycle3_right_target_pose",
      makePose(
        Eigen::Vector3d(0.45392780661582947, -0.2, 0.7707512974739075),
        Eigen::Quaterniond(0.9297504425048828, 0.0, -0.3681904673576355, 0.0)));
    cycle3_l_goal_pose_ = declarePoseParameter(
      "cycle3_left_target_pose",
      makePose(
        Eigen::Vector3d(0.45392780661582947, 0.2, 0.7707512974739075),
        Eigen::Quaterniond(0.9297504425048828, 0.0, -0.3681904673576355, 0.0)));

    r_goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(r_goal_pose_topic_, 10);
    l_goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(l_goal_pose_topic_, 10);
    r_elbow_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(r_elbow_pose_topic_, 10);
    l_elbow_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(l_elbow_pose_topic_, 10);
    arm_base_goal_pose_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseStamped>(arm_base_goal_pose_topic_, 10);
    orientation_start_pub_ =
      this->create_publisher<std_msgs::msg::Bool>(orientation_start_topic_, 10);
    orientation_done_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      orientation_done_topic_, 10,
      std::bind(&WholebodyReferenceNode::orientationDoneCallback, this, std::placeholders::_1));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    const int timer_period_ms =
      std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, control_frequency_))));
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(timer_period_ms),
      std::bind(&WholebodyReferenceNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Wholebody reference node started.");
    RCLCPP_INFO(this->get_logger(), "Waiting for TF to initialize choreography start poses.");
  }

private:
  Eigen::Vector3d declareVector3Parameter(
    const std::string & name,
    const Eigen::Vector3d & default_value)
  {
    const auto values = this->declare_parameter<std::vector<double>>(
      name, {default_value.x(), default_value.y(), default_value.z()});
    if (values.size() != 3) {
      throw std::runtime_error("Parameter '" + name + "' must contain exactly 3 values.");
    }
    return Eigen::Vector3d(values[0], values[1], values[2]);
  }

  Eigen::Quaterniond declareQuaternionParameter(
    const std::string & name,
    const Eigen::Quaterniond & default_value)
  {
    const auto values = this->declare_parameter<std::vector<double>>(
      name, {default_value.x(), default_value.y(), default_value.z(), default_value.w()});
    if (values.size() != 4) {
      throw std::runtime_error("Parameter '" + name + "' must contain exactly 4 values.");
    }
    return Eigen::Quaterniond(values[3], values[0], values[1], values[2]).normalized();
  }

  Eigen::Affine3d declarePoseParameter(
    const std::string & name,
    const Eigen::Affine3d & default_pose)
  {
    return makePose(
      declareVector3Parameter(name + ".position", default_pose.translation()),
      declareQuaternionParameter(name + ".orientation", Eigen::Quaterniond(default_pose.linear())));
  }

  bool lookupPose(const std::string & child_frame, Eigen::Affine3d & pose_out) const
  {
    try {
      const geometry_msgs::msg::TransformStamped tf =
        tf_buffer_->lookupTransform(base_frame_id_, child_frame, tf2::TimePointZero);
      pose_out = makePose(
        Eigen::Vector3d(
          tf.transform.translation.x,
          tf.transform.translation.y,
          tf.transform.translation.z),
        Eigen::Quaterniond(
          tf.transform.rotation.w,
          tf.transform.rotation.x,
          tf.transform.rotation.y,
          tf.transform.rotation.z));
      return true;
    } catch (const std::exception &) {
      return false;
    }
  }

  bool initializeFromTf()
  {
    const bool all_available =
      lookupPose(right_gripper_frame_, initial_r_goal_pose_) &&
      lookupPose(left_gripper_frame_, initial_l_goal_pose_) &&
      lookupPose(arm_base_frame_, initial_arm_base_pose_);

    if (!all_available) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for TF transforms from '%s' to initial gripper/arm-base frames.",
        base_frame_id_.c_str());
      return false;
    }

    phase_start_time_ = this->now();
    initialized_ = true;
    fixed_base_pose_captured_ = false;
    active_r_goal_pose_ = cycle1_r_goal_pose_;
    active_l_goal_pose_ = cycle1_l_goal_pose_;
    phase_ = Phase::InitialMove;
    RCLCPP_INFO(this->get_logger(), "Captured initial gripper and startup arm-base poses.");
    return true;
  }

  bool captureFixedBasePoseFromTf()
  {
    fixed_base_pose_captured_ = lookupPose(arm_base_frame_, fixed_arm_base_pose_);
    if (!fixed_base_pose_captured_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for TF transform from '%s' to fixed arm-base frame.",
        base_frame_id_.c_str());
    }
    return fixed_base_pose_captured_;
  }

  bool captureRedundancyStartFromTf()
  {
    Eigen::Affine3d right_shoulder_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d left_shoulder_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d right_elbow_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d left_elbow_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d right_wrist_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d left_wrist_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d right_gripper_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d left_gripper_pose = Eigen::Affine3d::Identity();

    const bool all_available =
      lookupPose(right_shoulder_frame_, right_shoulder_pose) &&
      lookupPose(left_shoulder_frame_, left_shoulder_pose) &&
      lookupPose(right_wrist_frame_, right_wrist_pose) &&
      lookupPose(left_wrist_frame_, left_wrist_pose) &&
      lookupPose(right_gripper_frame_, right_gripper_pose) &&
      lookupPose(left_gripper_frame_, left_gripper_pose) &&
      lookupPose(right_elbow_frame_, right_elbow_pose) &&
      lookupPose(left_elbow_frame_, left_elbow_pose);

    if (!all_available) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for TF transforms from '%s' to shoulder/elbow/wrist/gripper frames.",
        base_frame_id_.c_str());
      return false;
    }

    right_shoulder_position_ = right_shoulder_pose.translation();
    left_shoulder_position_ = left_shoulder_pose.translation();
    right_wrist_position_ = right_wrist_pose.translation();
    left_wrist_position_ = left_wrist_pose.translation();
    right_elbow_start_position_ = right_elbow_pose.translation();
    left_elbow_start_position_ = left_elbow_pose.translation();
    right_elbow_orientation_ = Eigen::Quaterniond(right_elbow_pose.linear());
    left_elbow_orientation_ = Eigen::Quaterniond(left_elbow_pose.linear());
    active_r_goal_pose_ = right_gripper_pose;
    active_l_goal_pose_ = left_gripper_pose;

    redundancy_capture_ready_ =
      initializeElbowArc(
      r_arc_, right_shoulder_position_, right_elbow_start_position_,
      right_elbow_orientation_, right_wrist_position_, -1.0) &&
      initializeElbowArc(
      l_arc_, left_shoulder_position_, left_elbow_start_position_,
      left_elbow_orientation_, left_wrist_position_, 1.0);

    if (!redundancy_capture_ready_) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Failed to initialize elbow arc references.");
      return false;
    }

    right_phase1_sweep_sign_ = selectBodywardSweepSign(r_arc_, right_elbow_start_position_);
    left_phase1_sweep_sign_ = selectBodywardSweepSign(l_arc_, left_elbow_start_position_);

    return true;
  }

  bool captureArmBasePhaseStartFromTf()
  {
    arm_base_phase_captured_ = lookupPose(arm_base_frame_, arm_base_phase_start_pose_);
    if (!arm_base_phase_captured_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for TF transform from '%s' to arm-base frame.",
        base_frame_id_.c_str());
    }
    return arm_base_phase_captured_;
  }

  bool captureRestartStartFromTf()
  {
    Eigen::Affine3d right_elbow_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d left_elbow_pose = Eigen::Affine3d::Identity();

    const bool all_available =
      lookupPose(right_gripper_frame_, initial_r_goal_pose_) &&
      lookupPose(left_gripper_frame_, initial_l_goal_pose_) &&
      lookupPose(arm_base_frame_, initial_arm_base_pose_) &&
      lookupPose(right_elbow_frame_, right_elbow_pose) &&
      lookupPose(left_elbow_frame_, left_elbow_pose);

    if (!all_available) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for TF transforms from '%s' to gripper/arm-base/elbow frames to restart loop.",
        base_frame_id_.c_str());
      return false;
    }

    restart_r_elbow_pose_ = right_elbow_pose;
    restart_l_elbow_pose_ = left_elbow_pose;
    restart_elbow_pose_captured_ = true;
    return true;
  }

  void orientationDoneCallback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (msg && msg->data) {
      orientation_done_received_ = true;
    }
  }

  void enterPhase(const Phase phase)
  {
    phase_ = phase;
    phase_start_time_ = this->now();
    arm_base_transition_capture_ready_ = false;
    if (phase == Phase::WaitCycle1Pose || phase == Phase::RedundancyFixedBase) {
      fixed_base_pose_captured_ = false;
    }
    if (phase == Phase::ArmBaseMotion || phase == Phase::RedundancyNaturalBase) {
      arm_base_phase_captured_ = false;
    }
  }

  Eigen::Affine3d applyYOffset(const Eigen::Affine3d & pose, const double delta_y) const
  {
    Eigen::Affine3d shifted_pose = pose;
    shifted_pose.translation().y() += delta_y;
    return shifted_pose;
  }

  void publishTransitionGrippers(
    const Eigen::Affine3d & start_right_pose,
    const Eigen::Affine3d & start_left_pose,
    const Eigen::Affine3d & goal_right_pose,
    const Eigen::Affine3d & goal_left_pose,
    const double elapsed) const
  {
    const double alpha = clamp01(elapsed / std::max(transition_move_duration_, 1e-6));
    publishPose(r_goal_pose_pub_, interpolatePose(start_right_pose, goal_right_pose, alpha));
    publishPose(l_goal_pose_pub_, interpolatePose(start_left_pose, goal_left_pose, alpha));
  }

  void publishCycle2TransitionGrippers(const double elapsed) const
  {
    const Eigen::Affine3d right_waypoint =
      applyYOffset(cycle1_r_goal_pose_, -cycle2_transition_outward_y_offset_);
    const Eigen::Affine3d left_waypoint =
      applyYOffset(cycle1_l_goal_pose_, cycle2_transition_outward_y_offset_);
    const double half_duration = std::max(0.5 * transition_move_duration_, 1e-6);

    if (elapsed < half_duration) {
      const double alpha = clamp01(elapsed / half_duration);
      publishPose(r_goal_pose_pub_, interpolatePose(cycle1_r_goal_pose_, right_waypoint, alpha));
      publishPose(l_goal_pose_pub_, interpolatePose(cycle1_l_goal_pose_, left_waypoint, alpha));
      return;
    }

    const double alpha = clamp01((elapsed - half_duration) / half_duration);
    publishPose(r_goal_pose_pub_, interpolatePose(right_waypoint, cycle2_r_goal_pose_, alpha));
    publishPose(l_goal_pose_pub_, interpolatePose(left_waypoint, cycle2_l_goal_pose_, alpha));
  }

  void publishCycle3TransitionGrippers(const double elapsed) const
  {
    const Eigen::Affine3d right_waypoint =
      applyYOffset(cycle2_r_goal_pose_, -cycle2_transition_outward_y_offset_);
    const Eigen::Affine3d left_waypoint =
      applyYOffset(cycle2_l_goal_pose_, cycle2_transition_outward_y_offset_);
    const double half_duration = std::max(0.5 * transition_move_duration_, 1e-6);

    if (elapsed < half_duration) {
      const double alpha = clamp01(elapsed / half_duration);
      publishPose(r_goal_pose_pub_, interpolatePose(cycle2_r_goal_pose_, right_waypoint, alpha));
      publishPose(l_goal_pose_pub_, interpolatePose(cycle2_l_goal_pose_, left_waypoint, alpha));
      return;
    }

    const double alpha = clamp01((elapsed - half_duration) / half_duration);
    publishPose(r_goal_pose_pub_, interpolatePose(right_waypoint, cycle3_r_goal_pose_, alpha));
    publishPose(l_goal_pose_pub_, interpolatePose(left_waypoint, cycle3_l_goal_pose_, alpha));
  }

  bool grippersReachedGoal(
    const Eigen::Affine3d & right_goal_pose,
    const Eigen::Affine3d & left_goal_pose) const
  {
    Eigen::Affine3d right_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d left_pose = Eigen::Affine3d::Identity();
    if (!lookupPose(right_gripper_frame_, right_pose) || !lookupPose(left_gripper_frame_, left_pose)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for TF transforms from '%s' to gripper frames for pose convergence check.",
        base_frame_id_.c_str());
      return false;
    }

    const double right_pos_error =
      (right_pose.translation() - right_goal_pose.translation()).norm();
    const double left_pos_error =
      (left_pose.translation() - left_goal_pose.translation()).norm();
    const double orientation_tolerance_rad = gripper_orientation_tolerance_deg_ * kPi / 180.0;
    const double right_ori_error =
      Eigen::Quaterniond(right_pose.linear()).angularDistance(Eigen::Quaterniond(right_goal_pose.linear()));
    const double left_ori_error =
      Eigen::Quaterniond(left_pose.linear()).angularDistance(Eigen::Quaterniond(left_goal_pose.linear()));

    return right_pos_error <= gripper_position_tolerance_ &&
           left_pos_error <= gripper_position_tolerance_ &&
           right_ori_error <= orientation_tolerance_rad &&
           left_ori_error <= orientation_tolerance_rad;
  }

  Eigen::Affine3d evaluateElbowArc(const ElbowArc & arc, const double angle) const
  {
    const Eigen::Vector3d offset =
      arc.radius * (std::cos(angle) * arc.u + std::sin(angle) * arc.v);
    return makePose(arc.center + offset, arc.orientation);
  }

  double selectBodywardSweepSign(
    const ElbowArc & arc,
    const Eigen::Vector3d & elbow_start_position) const
  {
    constexpr double kProbeAngleRad = 5.0 * kPi / 180.0;
    const double start_abs_y = std::abs(elbow_start_position.y());
    const double plus_abs_y =
      std::abs(evaluateElbowArc(arc, kProbeAngleRad).translation().y());
    const double minus_abs_y =
      std::abs(evaluateElbowArc(arc, -kProbeAngleRad).translation().y());

    const bool plus_moves_inward = plus_abs_y < start_abs_y - 1e-6;
    const bool minus_moves_inward = minus_abs_y < start_abs_y - 1e-6;

    if (plus_moves_inward && !minus_moves_inward) {
      return 1.0;
    }
    if (minus_moves_inward && !plus_moves_inward) {
      return -1.0;
    }
    return (plus_abs_y <= minus_abs_y) ? 1.0 : -1.0;
  }

  double computePhase1SweepAngle(const double elapsed) const
  {
    const double cycle_duration = std::max(redundancy_cycle1_duration_, 1e-6);
    const double normalized_cycle =
      std::fmod(std::max(elapsed, 0.0), cycle_duration) / cycle_duration;
    const double inward_angle = elbow_arc_angle_deg_ * kPi / 180.0;
    const double past_start_angle =
      std::max(0.0, phase1_return_past_start_ratio_) * inward_angle;

    if (normalized_cycle < (1.0 / 3.0)) {
      const double alpha = clamp01(normalized_cycle * 3.0);
      return 0.5 * inward_angle * (1.0 - std::cos(kPi * alpha));
    }

    if (normalized_cycle < (2.0 / 3.0)) {
      const double alpha = clamp01((normalized_cycle - (1.0 / 3.0)) * 3.0);
      return inward_angle + 0.5 * (-past_start_angle - inward_angle) * (1.0 - std::cos(kPi * alpha));
    }

    const double alpha = clamp01((normalized_cycle - (2.0 / 3.0)) * 3.0);
    return -past_start_angle + 0.5 * past_start_angle * (1.0 - std::cos(kPi * alpha));
  }

  bool initializeElbowArc(
    ElbowArc & arc,
    const Eigen::Vector3d & shoulder,
    const Eigen::Vector3d & elbow,
    const Eigen::Quaterniond & elbow_orientation,
    const Eigen::Vector3d & wrist,
    const double outside_y_sign) const
  {
    Eigen::Vector3d axis = wrist - shoulder;
    const double shoulder_to_wrist = axis.norm();
    if (shoulder_to_wrist < 1e-6) {
      return false;
    }
    axis /= shoulder_to_wrist;

    const double upper_arm_length = (elbow - shoulder).norm();
    const double lower_arm_length = (wrist - elbow).norm();
    if (upper_arm_length < 1e-6 || lower_arm_length < 1e-6) {
      return false;
    }

    const double axial_distance = std::clamp(
      (upper_arm_length * upper_arm_length - lower_arm_length * lower_arm_length +
      shoulder_to_wrist * shoulder_to_wrist) /
      (2.0 * shoulder_to_wrist),
      0.0, shoulder_to_wrist);
    const Eigen::Vector3d center = shoulder + axial_distance * axis;
    const double geometric_radius = std::sqrt(std::max(
      upper_arm_length * upper_arm_length - axial_distance * axial_distance, 0.0));

    Eigen::Vector3d outward_hint(0.0, outside_y_sign, 0.0);
    outward_hint = projectToPlane(outward_hint, axis);
    if (outward_hint.norm() < 1e-6) {
      outward_hint = projectToPlane(Eigen::Vector3d::UnitZ(), axis);
    }
    outward_hint = normalizedOrFallback(outward_hint, Eigen::Vector3d::UnitZ());

    Eigen::Vector3d radial = projectToPlane(elbow - center, axis);
    const double target_radius = std::max(geometric_radius, minimum_elbow_radius_);
    if (radial.norm() < 1e-6) {
      radial = target_radius * outward_hint;
    }

    const Eigen::Vector3d u = normalizedOrFallback(radial, outward_hint);
    Eigen::Vector3d v = axis.cross(u);
    if (v.norm() < 1e-6) {
      v = axis.cross(Eigen::Vector3d::UnitZ());
      if (v.norm() < 1e-6) {
        v = axis.cross(Eigen::Vector3d::UnitX());
      }
    }
    if (v.dot(outward_hint) < 0.0) {
      v = -v;
    }
    v.normalize();

    arc.center = center;
    arc.u = u;
    arc.v = v;
    arc.radius = target_radius;
    arc.orientation = elbow_orientation;
    return true;
  }

  Eigen::Affine3d computeArmBaseGoalPose(const double elapsed) const
  {
    const double midpoint = 0.5 * (arm_base_upper_z_offset_ + arm_base_lower_z_offset_);
    const double amplitude = 0.5 * (arm_base_upper_z_offset_ - arm_base_lower_z_offset_);
    const double wave =
      midpoint + amplitude *
      std::cos(2.0 * kPi * elapsed / std::max(arm_base_cycle_duration_, 1e-6));
    const double ramp =
      (elapsed < arm_base_entry_duration_) ?
      0.5 * (1.0 - std::cos(kPi * elapsed / std::max(arm_base_entry_duration_, 1e-6))) :
      1.0;

    Eigen::Affine3d goal_pose = arm_base_phase_start_pose_;
    goal_pose.translation().z() += ramp * wave;
    return goal_pose;
  }

  void publishPose(
    const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr & publisher,
    const Eigen::Affine3d & pose) const
  {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = base_frame_id_;
    msg.pose.position.x = pose.translation().x();
    msg.pose.position.y = pose.translation().y();
    msg.pose.position.z = pose.translation().z();

    const Eigen::Quaterniond q(pose.linear());
    msg.pose.orientation.w = q.w();
    msg.pose.orientation.x = q.x();
    msg.pose.orientation.y = q.y();
    msg.pose.orientation.z = q.z();
    publisher->publish(msg);
  }

  void publishFixedGrippers() const
  {
    publishPose(r_goal_pose_pub_, active_r_goal_pose_);
    publishPose(l_goal_pose_pub_, active_l_goal_pose_);
  }

  void publishHeldElbows() const
  {
    if (!redundancy_capture_ready_) {
      return;
    }
    publishPose(r_elbow_pose_pub_, evaluateElbowArc(r_arc_, 0.0));
    publishPose(l_elbow_pose_pub_, evaluateElbowArc(l_arc_, 0.0));
  }

  void publishRestartElbows(const double elapsed) const
  {
    if (!redundancy_capture_ready_) {
      return;
    }

    const Eigen::Affine3d right_goal_pose = evaluateElbowArc(r_arc_, 0.0);
    const Eigen::Affine3d left_goal_pose = evaluateElbowArc(l_arc_, 0.0);

    if (!restart_elbow_pose_captured_) {
      publishPose(r_elbow_pose_pub_, right_goal_pose);
      publishPose(l_elbow_pose_pub_, left_goal_pose);
      return;
    }

    const double alpha = clamp01(elapsed / std::max(initial_move_duration_, 1e-6));
    publishPose(r_elbow_pose_pub_, interpolatePose(restart_r_elbow_pose_, right_goal_pose, alpha));
    publishPose(l_elbow_pose_pub_, interpolatePose(restart_l_elbow_pose_, left_goal_pose, alpha));
  }

  void publishHeldArmBaseAtElapsed(const double elapsed) const
  {
    if (!arm_base_phase_captured_) {
      return;
    }
    publishPose(arm_base_goal_pose_pub_, computeArmBaseGoalPose(elapsed));
  }

  bool captureArmBaseTransitionStartFromTf()
  {
    if (arm_base_transition_capture_ready_) {
      return true;
    }

    arm_base_transition_capture_ready_ = lookupPose(arm_base_frame_, arm_base_transition_start_pose_);
    if (!arm_base_transition_capture_ready_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for TF transform from '%s' to arm-base frame for transition smoothing.",
        base_frame_id_.c_str());
    }
    return arm_base_transition_capture_ready_;
  }

  void publishSmoothArmBaseGoal(const Eigen::Affine3d & goal_pose, const double elapsed)
  {
    if (!captureArmBaseTransitionStartFromTf()) {
      return;
    }

    const double alpha = clamp01(elapsed / std::max(arm_base_transition_duration_, 1e-6));
    publishPose(
      arm_base_goal_pose_pub_,
      interpolatePose(arm_base_transition_start_pose_, goal_pose, alpha));
  }

  void publishSmoothHeldArmBaseAtElapsed(const double target_elapsed, const double phase_elapsed)
  {
    if (!arm_base_phase_captured_) {
      return;
    }
    publishSmoothArmBaseGoal(computeArmBaseGoalPose(target_elapsed), phase_elapsed);
  }

  void timerCallback()
  {
    if (!initialized_ && !initializeFromTf()) {
      return;
    }

    const double elapsed = (this->now() - phase_start_time_).seconds();

    switch (phase_) {
      case Phase::InitialMove:
      {
        const double alpha = clamp01(elapsed / std::max(initial_move_duration_, 1e-6));
        publishPose(r_goal_pose_pub_, interpolatePose(initial_r_goal_pose_, cycle1_r_goal_pose_, alpha));
        publishPose(l_goal_pose_pub_, interpolatePose(initial_l_goal_pose_, cycle1_l_goal_pose_, alpha));
        publishSmoothArmBaseGoal(initial_arm_base_pose_, elapsed);
        publishRestartElbows(elapsed);
        if (elapsed >= initial_move_duration_) {
          active_r_goal_pose_ = cycle1_r_goal_pose_;
          active_l_goal_pose_ = cycle1_l_goal_pose_;
          restart_elbow_pose_captured_ = false;
          enterPhase(Phase::RedundancyFixedBase);
        }
        break;
      }

      case Phase::WaitCycle1Pose:
      {
        publishTransitionGrippers(
          cycle3_r_goal_pose_, cycle3_l_goal_pose_,
          cycle1_r_goal_pose_, cycle1_l_goal_pose_, elapsed);
        publishHeldElbows();
        publishSmoothHeldArmBaseAtElapsed(
          redundancy_phase2_cycles_ * redundancy_cycle2_duration_, elapsed);
        if (elapsed >= transition_move_duration_ &&
          grippersReachedGoal(cycle1_r_goal_pose_, cycle1_l_goal_pose_))
        {
          enterPhase(Phase::RedundancyFixedBase);
        }
        break;
      }

      case Phase::RedundancyFixedBase:
      {
        if (!fixed_base_pose_captured_ && !captureFixedBasePoseFromTf()) {
          return;
        }
        publishFixedGrippers();
        publishSmoothArmBaseGoal(fixed_arm_base_pose_, elapsed);
        if (!redundancy_capture_ready_ && !captureRedundancyStartFromTf()) {
          return;
        }

        const double sweep = computePhase1SweepAngle(elapsed);
        publishPose(r_elbow_pose_pub_, evaluateElbowArc(r_arc_, right_phase1_sweep_sign_ * sweep));
        publishPose(l_elbow_pose_pub_, evaluateElbowArc(l_arc_, left_phase1_sweep_sign_ * sweep));

        if (elapsed >= redundancy_phase1_cycles_ * redundancy_cycle1_duration_) {
          active_r_goal_pose_ = cycle2_r_goal_pose_;
          active_l_goal_pose_ = cycle2_l_goal_pose_;
          enterPhase(Phase::WaitCycle2Pose);
        }
        break;
      }

      case Phase::WaitCycle2Pose:
      {
        publishCycle2TransitionGrippers(elapsed);
        publishSmoothArmBaseGoal(fixed_arm_base_pose_, elapsed);
        publishHeldElbows();
        if (elapsed >= transition_move_duration_ &&
          grippersReachedGoal(cycle2_r_goal_pose_, cycle2_l_goal_pose_))
        {
          enterPhase(Phase::ArmBaseMotion);
        }
        break;
      }

      case Phase::ArmBaseMotion:
      {
        publishFixedGrippers();
        if (!arm_base_phase_captured_ && !captureArmBasePhaseStartFromTf()) {
          return;
        }
        publishSmoothArmBaseGoal(computeArmBaseGoalPose(elapsed), elapsed);

        if (elapsed >= arm_base_phase_cycles_ * arm_base_cycle_duration_) {
          active_r_goal_pose_ = cycle3_r_goal_pose_;
          active_l_goal_pose_ = cycle3_l_goal_pose_;
          enterPhase(Phase::WaitCycle3Pose);
        }
        break;
      }

      case Phase::WaitCycle3Pose:
      {
        publishCycle3TransitionGrippers(elapsed);
        publishHeldElbows();
        publishSmoothHeldArmBaseAtElapsed(
          arm_base_phase_cycles_ * arm_base_cycle_duration_, elapsed);
        if (elapsed >= transition_move_duration_ &&
          grippersReachedGoal(cycle3_r_goal_pose_, cycle3_l_goal_pose_))
        {
          enterPhase(Phase::RedundancyNaturalBase);
        }
        break;
      }

      case Phase::RedundancyNaturalBase:
      {
        publishFixedGrippers();
        if (!redundancy_capture_ready_ && !captureRedundancyStartFromTf()) {
          return;
        }

        const double sweep =
          0.5 * elbow_arc_angle_deg_ * kPi / 180.0 *
          (1.0 - std::cos(2.0 * kPi * elapsed / std::max(redundancy_cycle2_duration_, 1e-6)));
        publishPose(r_elbow_pose_pub_, evaluateElbowArc(r_arc_, sweep));
        publishPose(l_elbow_pose_pub_, evaluateElbowArc(l_arc_, sweep));

        if (elapsed >= redundancy_phase2_cycles_ * redundancy_cycle2_duration_) {
          active_r_goal_pose_ = orientation_start_r_goal_pose_;
          active_l_goal_pose_ = orientation_start_l_goal_pose_;
          enterPhase(Phase::ReturnToInitialPose);
        }
        break;
      }

      case Phase::ReturnToInitialPose:
      {
        publishTransitionGrippers(
          cycle3_r_goal_pose_, cycle3_l_goal_pose_,
          orientation_start_r_goal_pose_, orientation_start_l_goal_pose_, elapsed);
        publishHeldElbows();
        publishSmoothHeldArmBaseAtElapsed(
          redundancy_phase2_cycles_ * redundancy_cycle2_duration_, elapsed);
        if (elapsed >= transition_move_duration_ &&
          grippersReachedGoal(orientation_start_r_goal_pose_, orientation_start_l_goal_pose_))
        {
          std_msgs::msg::Bool start_msg;
          start_msg.data = true;
          orientation_start_pub_->publish(start_msg);
          orientation_done_received_ = false;
          RCLCPP_INFO(
            this->get_logger(),
            "Wholebody motion returned to initial pose. Triggering orientation phase.");
          enterPhase(Phase::WaitOrientationComplete);
        }
        break;
      }

      case Phase::WaitOrientationComplete:
      {
        if (orientation_done_received_) {
          orientation_done_received_ = false;
          if (!captureRestartStartFromTf()) {
            return;
          }
          enterPhase(Phase::InitialMove);
        }
        break;
      }
    }
  }

  double control_frequency_;
  double initial_move_duration_;
  double redundancy_cycle1_duration_;
  double redundancy_cycle2_duration_;
  double arm_base_cycle_duration_;
  double arm_base_entry_duration_;
  double elbow_arc_angle_deg_;
  double phase1_return_past_start_ratio_;
  double minimum_elbow_radius_;
  double transition_move_duration_;
  double arm_base_transition_duration_;
  double cycle2_transition_outward_y_offset_;
  double gripper_position_tolerance_;
  double gripper_orientation_tolerance_deg_;
  double arm_base_upper_z_offset_;
  double arm_base_lower_z_offset_;
  int redundancy_phase1_cycles_;
  int arm_base_phase_cycles_;
  int redundancy_phase2_cycles_;

  std::string base_frame_id_;
  std::string orientation_start_topic_;
  std::string orientation_done_topic_;
  std::string r_goal_pose_topic_;
  std::string l_goal_pose_topic_;
  std::string r_elbow_pose_topic_;
  std::string l_elbow_pose_topic_;
  std::string arm_base_goal_pose_topic_;
  std::string right_gripper_frame_;
  std::string left_gripper_frame_;
  std::string right_shoulder_frame_;
  std::string left_shoulder_frame_;
  std::string right_elbow_frame_;
  std::string left_elbow_frame_;
  std::string right_wrist_frame_;
  std::string left_wrist_frame_;
  std::string arm_base_frame_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_goal_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_elbow_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_elbow_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr arm_base_goal_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr orientation_start_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr orientation_done_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  bool initialized_;
  bool fixed_base_pose_captured_;
  bool redundancy_capture_ready_;
  bool arm_base_phase_captured_;
  bool orientation_done_received_ = false;
  bool restart_elbow_pose_captured_ = false;
  bool arm_base_transition_capture_ready_ = false;
  Phase phase_;
  rclcpp::Time phase_start_time_;

  Eigen::Affine3d initial_r_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d initial_l_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d initial_arm_base_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d cycle1_r_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d cycle1_l_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d orientation_start_r_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d orientation_start_l_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d cycle2_r_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d cycle2_l_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d cycle3_r_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d cycle3_l_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d active_r_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d active_l_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d fixed_arm_base_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d arm_base_phase_start_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d arm_base_transition_start_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d restart_r_elbow_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d restart_l_elbow_pose_ = Eigen::Affine3d::Identity();
  Eigen::Vector3d right_shoulder_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_shoulder_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_wrist_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_wrist_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_elbow_start_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_elbow_start_position_ = Eigen::Vector3d::Zero();
  double right_phase1_sweep_sign_ = 1.0;
  double left_phase1_sweep_sign_ = 1.0;
  Eigen::Quaterniond right_elbow_orientation_ = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond left_elbow_orientation_ = Eigen::Quaterniond::Identity();
  ElbowArc r_arc_;
  ElbowArc l_arc_;
};
}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyclo_motion_controller_ros::WholebodyReferenceNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
