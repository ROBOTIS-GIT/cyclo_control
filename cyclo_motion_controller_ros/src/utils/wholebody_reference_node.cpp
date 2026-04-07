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
  RedundancyFixedBase,
  ArmBaseMotion,
  RedundancyNaturalBase,
  PauseBetweenLoops
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
    initial_move_duration_ = this->declare_parameter("initial_move_duration", 6.0);
    redundancy_cycle1_duration_ = this->declare_parameter("redundancy_cycle1_duration", 1.5);
    redundancy_cycle2_duration_ = this->declare_parameter("redundancy_cycle2_duration", 2.5);
    redundancy_phase1_cycles_ = this->declare_parameter("redundancy_phase1_cycles", 4);
    redundancy_phase2_cycles_ = this->declare_parameter("redundancy_phase2_cycles", 2);
    pause_between_loops_duration_ = this->declare_parameter("pause_between_loops_duration", 3.0);
    arm_base_cycle_duration_ = this->declare_parameter("arm_base_cycle_duration", 2.5);
    arm_base_phase_cycles_ = this->declare_parameter("arm_base_phase_cycles", 2);
    arm_base_entry_duration_ = this->declare_parameter("arm_base_entry_duration", 1.0);
    elbow_arc_angle_deg_ = this->declare_parameter("elbow_arc_angle_deg", 40.0);
    minimum_elbow_radius_ = this->declare_parameter("minimum_elbow_radius", 0.0);
    right_elbow_start_y_offset_ = this->declare_parameter("right_elbow_start_y_offset", 0.15);
    left_elbow_start_y_offset_ = this->declare_parameter("left_elbow_start_y_offset", -0.15);
    arm_base_upper_z_offset_ = this->declare_parameter("arm_base_upper_z_offset", 0.1);
    arm_base_lower_z_offset_ = this->declare_parameter("arm_base_lower_z_offset", -0.1);
    base_frame_id_ = this->declare_parameter("base_frame_id", std::string("base_link"));

    r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
    l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));
    r_elbow_pose_topic_ = this->declare_parameter("r_elbow_pose_topic", std::string("/r_elbow_pose"));
    l_elbow_pose_topic_ = this->declare_parameter("l_elbow_pose_topic", std::string("/l_elbow_pose"));
    arm_base_goal_pose_topic_ = this->declare_parameter(
      "arm_base_goal_pose_topic", std::string("/arm_base_goal_pose"));

    right_gripper_frame_ = this->declare_parameter(
      "right_gripper_frame", std::string("arm_r_link7"));
    left_gripper_frame_ = this->declare_parameter(
      "left_gripper_frame", std::string("arm_l_link7"));
    right_shoulder_frame_ = this->declare_parameter(
      "right_shoulder_frame", std::string("arm_r_link1"));
    left_shoulder_frame_ = this->declare_parameter(
      "left_shoulder_frame", std::string("arm_l_link1"));
    right_elbow_frame_ = this->declare_parameter(
      "right_elbow_frame", std::string("arm_r_link4"));
    left_elbow_frame_ = this->declare_parameter(
      "left_elbow_frame", std::string("arm_l_link4"));
    arm_base_frame_ = this->declare_parameter("arm_base_frame", std::string("arm_base_link"));

    target_r_goal_pose_ = declarePoseParameter(
      "right_target_pose",
      makePose(
        Eigen::Vector3d(0.25, -0.18, 1.0),
        Eigen::Quaterniond(0.707, 0.0, -0.707, 0.0)));
    target_l_goal_pose_ = declarePoseParameter(
      "left_target_pose",
      makePose(
        Eigen::Vector3d(0.25, 0.18, 1.0),
        Eigen::Quaterniond(0.707, 0.0, -0.707, 0.0)));

    r_goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(r_goal_pose_topic_, 10);
    l_goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(l_goal_pose_topic_, 10);
    r_elbow_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(r_elbow_pose_topic_, 10);
    l_elbow_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(l_elbow_pose_topic_, 10);
    arm_base_goal_pose_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseStamped>(arm_base_goal_pose_topic_, 10);

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

    const bool all_available =
      lookupPose(right_shoulder_frame_, right_shoulder_pose) &&
      lookupPose(left_shoulder_frame_, left_shoulder_pose) &&
      lookupPose(right_elbow_frame_, right_elbow_pose) &&
      lookupPose(left_elbow_frame_, left_elbow_pose);

    if (!all_available) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for TF transforms from '%s' to shoulder/elbow frames.",
        base_frame_id_.c_str());
      return false;
    }

    right_shoulder_position_ = right_shoulder_pose.translation();
    left_shoulder_position_ = left_shoulder_pose.translation();
    right_elbow_start_position_ = right_elbow_pose.translation();
    left_elbow_start_position_ = left_elbow_pose.translation();
    right_elbow_start_position_.y() += right_elbow_start_y_offset_;
    left_elbow_start_position_.y() += left_elbow_start_y_offset_;
    right_elbow_orientation_ = Eigen::Quaterniond(right_elbow_pose.linear());
    left_elbow_orientation_ = Eigen::Quaterniond(left_elbow_pose.linear());

    redundancy_capture_ready_ =
      initializeElbowArc(
      r_arc_, right_shoulder_position_, right_elbow_start_position_,
      right_elbow_orientation_, target_r_goal_pose_, -1.0) &&
      initializeElbowArc(
      l_arc_, left_shoulder_position_, left_elbow_start_position_,
      left_elbow_orientation_, target_l_goal_pose_, 1.0);

    if (!redundancy_capture_ready_) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Failed to initialize elbow arc references.");
      return false;
    }

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

  void enterPhase(const Phase phase)
  {
    phase_ = phase;
    phase_start_time_ = this->now();
  }

  Eigen::Affine3d evaluateElbowArc(const ElbowArc & arc, const double angle) const
  {
    const Eigen::Vector3d offset =
      arc.radius * (std::cos(angle) * arc.u + std::sin(angle) * arc.v);
    return makePose(arc.center + offset, arc.orientation);
  }

  bool initializeElbowArc(
    ElbowArc & arc,
    const Eigen::Vector3d & shoulder,
    const Eigen::Vector3d & elbow,
    const Eigen::Quaterniond & elbow_orientation,
    const Eigen::Affine3d & gripper_goal_pose,
    const double outside_y_sign) const
  {
    const Eigen::Vector3d gripper = gripper_goal_pose.translation();
    Eigen::Vector3d axis = gripper - shoulder;
    if (axis.norm() < 1e-6) {
      return false;
    }
    axis.normalize();

    const Eigen::Vector3d center = shoulder + axis.dot(elbow - shoulder) * axis;

    Eigen::Vector3d outward_hint(0.0, outside_y_sign, 0.0);
    outward_hint = projectToPlane(outward_hint, axis);
    if (outward_hint.norm() < 1e-6) {
      outward_hint = projectToPlane(Eigen::Vector3d::UnitZ(), axis);
    }
    outward_hint = normalizedOrFallback(outward_hint, Eigen::Vector3d::UnitZ());

    Eigen::Vector3d radial = elbow - center;
    if (radial.norm() < minimum_elbow_radius_) {
      radial = minimum_elbow_radius_ * outward_hint;
    } else if (radial.dot(outward_hint) < 0.0) {
      radial = radial.norm() * outward_hint;
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
    arc.radius = std::max(radial.norm(), minimum_elbow_radius_);
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
    publishPose(r_goal_pose_pub_, target_r_goal_pose_);
    publishPose(l_goal_pose_pub_, target_l_goal_pose_);
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
        publishPose(r_goal_pose_pub_, interpolatePose(initial_r_goal_pose_, target_r_goal_pose_, alpha));
        publishPose(l_goal_pose_pub_, interpolatePose(initial_l_goal_pose_, target_l_goal_pose_, alpha));
        publishPose(arm_base_goal_pose_pub_, initial_arm_base_pose_);
        if (elapsed >= initial_move_duration_) {
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
        publishPose(arm_base_goal_pose_pub_, fixed_arm_base_pose_);
        if (!redundancy_capture_ready_ && !captureRedundancyStartFromTf()) {
          return;
        }

        const double sweep =
          0.5 * elbow_arc_angle_deg_ * kPi / 180.0 *
          (1.0 - std::cos(2.0 * kPi * elapsed / std::max(redundancy_cycle1_duration_, 1e-6)));
        publishPose(r_elbow_pose_pub_, evaluateElbowArc(r_arc_, sweep));
        publishPose(l_elbow_pose_pub_, evaluateElbowArc(l_arc_, sweep));

        if (elapsed >= redundancy_phase1_cycles_ * redundancy_cycle1_duration_) {
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
        publishPose(arm_base_goal_pose_pub_, computeArmBaseGoalPose(elapsed));

        if (elapsed >= arm_base_phase_cycles_ * arm_base_cycle_duration_) {
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
          enterPhase(Phase::PauseBetweenLoops);
        }
        break;
      }

      case Phase::PauseBetweenLoops:
      {
        publishFixedGrippers();
        if (!redundancy_capture_ready_ && !captureRedundancyStartFromTf()) {
          return;
        }
        publishPose(r_elbow_pose_pub_, evaluateElbowArc(r_arc_, 0.0));
        publishPose(l_elbow_pose_pub_, evaluateElbowArc(l_arc_, 0.0));

        if (elapsed >= pause_between_loops_duration_) {
          enterPhase(Phase::RedundancyFixedBase);
        }
        break;
      }
    }
  }

  double control_frequency_;
  double initial_move_duration_;
  double redundancy_cycle1_duration_;
  double redundancy_cycle2_duration_;
  double pause_between_loops_duration_;
  double arm_base_cycle_duration_;
  double arm_base_entry_duration_;
  double elbow_arc_angle_deg_;
  double minimum_elbow_radius_;
  double right_elbow_start_y_offset_;
  double left_elbow_start_y_offset_;
  double arm_base_upper_z_offset_;
  double arm_base_lower_z_offset_;
  int redundancy_phase1_cycles_;
  int arm_base_phase_cycles_;
  int redundancy_phase2_cycles_;

  std::string base_frame_id_;
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
  std::string arm_base_frame_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_goal_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_elbow_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_elbow_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr arm_base_goal_pose_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  bool initialized_;
  bool fixed_base_pose_captured_;
  bool redundancy_capture_ready_;
  bool arm_base_phase_captured_;
  Phase phase_;
  rclcpp::Time phase_start_time_;

  Eigen::Affine3d initial_r_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d initial_l_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d initial_arm_base_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d target_r_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d target_l_goal_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d fixed_arm_base_pose_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d arm_base_phase_start_pose_ = Eigen::Affine3d::Identity();
  Eigen::Vector3d right_shoulder_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_shoulder_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_elbow_start_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_elbow_start_position_ = Eigen::Vector3d::Zero();
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
