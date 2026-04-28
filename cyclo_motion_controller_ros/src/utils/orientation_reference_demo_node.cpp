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
// Author: OpenAI

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <robotis_interfaces/msg/move_l.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace cyclo_motion_controller_ros
{

class OrientationReferenceDemoNode : public rclcpp::Node
{
public:
  OrientationReferenceDemoNode()
  : Node("orientation_reference_demo_node"),
    initialized_(false),
    active_(false),
    last_phase_index_(-1)
  {
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
    right_controlled_link_ =
      this->declare_parameter<std::string>("right_controlled_link", "end_effector_r_link");
    left_controlled_link_ =
      this->declare_parameter<std::string>("left_controlled_link", "end_effector_l_link");
    right_goal_topic_ = this->declare_parameter<std::string>("right_goal_topic", "/r_goal_move");
    left_goal_topic_ = this->declare_parameter<std::string>("left_goal_topic", "/l_goal_move");
    enable_right_ = this->declare_parameter<bool>("enable_right", true);
    enable_left_ = this->declare_parameter<bool>("enable_left", true);
    start_on_launch_ = this->declare_parameter<bool>("start_on_launch", true);
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 40.0);
    command_duration_sec_ = this->declare_parameter<double>("command_duration_sec", 0.10);
    startup_delay_sec_ = this->declare_parameter<double>("startup_delay_sec", 0.0);
    loop_ = this->declare_parameter<bool>("loop", true);
    x_offset_ = this->declare_parameter<double>("x_offset", 0.1);
    y_separation_offset_ = this->declare_parameter<double>("y_separation_offset", 0.05);
    z_offset_ = this->declare_parameter<double>("z_offset", 0.1);
    z_offset_move_duration_sec_ =
      this->declare_parameter<double>("z_offset_move_duration_sec", 2.0);
    start_topic_ = this->declare_parameter<std::string>(
      "start_topic", "/motion_demo/start_orientation");
    done_topic_ = this->declare_parameter<std::string>(
      "done_topic", "/motion_demo/orientation_done");
    axis_sweep_duration_sec_ = this->declare_parameter<double>("axis_sweep_duration_sec", 4.5);
    circle_entry_duration_sec_ = this->declare_parameter<double>("circle_entry_duration_sec", 1.0);
    circle_duration_sec_ = this->declare_parameter<double>("circle_duration_sec", 6.0);
    circle_exit_duration_sec_ = this->declare_parameter<double>("circle_exit_duration_sec", 1.0);
    max_roll_deg_ = this->declare_parameter<double>("max_roll_deg", 30.0);
    max_pitch_deg_ = this->declare_parameter<double>("max_pitch_deg", 30.0);
    max_yaw_deg_ = this->declare_parameter<double>("max_yaw_deg", 30.0);
    circle_radius_deg_ = this->declare_parameter<double>("circle_radius_deg", 20.0);

    right_goal_pub_ = this->create_publisher<robotis_interfaces::msg::MoveL>(right_goal_topic_, 10);
    left_goal_pub_ = this->create_publisher<robotis_interfaces::msg::MoveL>(left_goal_topic_, 10);
    done_pub_ = this->create_publisher<std_msgs::msg::Bool>(done_topic_, 10);
    start_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      start_topic_, 10,
      std::bind(&OrientationReferenceDemoNode::startCallback, this, std::placeholders::_1));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    RCLCPP_INFO(this->get_logger(), "Orientation reference demo node started");
    RCLCPP_INFO(this->get_logger(), "  - Base frame: %s", base_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  - Right goal topic: %s", right_goal_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  - Left goal topic: %s", left_goal_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  - Right controlled link: %s", right_controlled_link_.c_str());
    RCLCPP_INFO(this->get_logger(), "  - Left controlled link: %s", left_controlled_link_.c_str());
    RCLCPP_INFO(this->get_logger(), "  - Start on launch: %s", start_on_launch_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  - Start topic: %s", start_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  - Done topic: %s", done_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  - X offset: %.3f m", x_offset_);
    RCLCPP_INFO(this->get_logger(), "  - Y separation offset: %.3f m", y_separation_offset_);
    RCLCPP_INFO(this->get_logger(), "  - Z offset: %.3f m", z_offset_);
    RCLCPP_INFO(this->get_logger(), "  - Z offset move duration: %.2f sec", z_offset_move_duration_sec_);
    RCLCPP_INFO(this->get_logger(), "  - Angle limits [deg] roll/pitch/yaw: %.1f / %.1f / %.1f",
      max_roll_deg_, max_pitch_deg_, max_yaw_deg_);
    RCLCPP_INFO(this->get_logger(), "  - Circle radius [deg]: %.1f", circle_radius_deg_);

    const int timer_period_ms =
      std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, publish_rate_hz_))));
    publish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(timer_period_ms),
      std::bind(&OrientationReferenceDemoNode::publishReference, this));

    if (start_on_launch_) {
      init_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(250),
        std::bind(&OrientationReferenceDemoNode::initializeIfReady, this));
    }
  }

private:
  struct OrientationOffset
  {
    double roll;
    double pitch;
    double yaw;
    std::string phase_name;
    int phase_index;
  };

  static double degToRad(const double degrees)
  {
    return degrees * M_PI / 180.0;
  }

  double totalDuration() const
  {
    return axis_sweep_duration_sec_ * 3.0 + circle_entry_duration_sec_ +
           circle_duration_sec_ + circle_exit_duration_sec_;
  }

  geometry_msgs::msg::PoseStamped withPositionOffset(
    const geometry_msgs::msg::PoseStamped & pose,
    const bool is_right) const
  {
    auto offset_pose = pose;
    offset_pose.pose.position.x += x_offset_;
    offset_pose.pose.position.y += is_right ? -y_separation_offset_ : y_separation_offset_;
    offset_pose.pose.position.z += z_offset_;
    return offset_pose;
  }

  geometry_msgs::msg::PoseStamped interpolatePose(
    const geometry_msgs::msg::PoseStamped & start_pose,
    const geometry_msgs::msg::PoseStamped & goal_pose,
    const double alpha) const
  {
    const double clamped_alpha = std::clamp(alpha, 0.0, 1.0);
    geometry_msgs::msg::PoseStamped pose = start_pose;
    pose.header.frame_id = base_frame_;
    pose.pose.position.x =
      (1.0 - clamped_alpha) * start_pose.pose.position.x + clamped_alpha * goal_pose.pose.position.x;
    pose.pose.position.y =
      (1.0 - clamped_alpha) * start_pose.pose.position.y + clamped_alpha * goal_pose.pose.position.y;
    pose.pose.position.z =
      (1.0 - clamped_alpha) * start_pose.pose.position.z + clamped_alpha * goal_pose.pose.position.z;
    pose.pose.orientation = start_pose.pose.orientation;
    return pose;
  }

  bool captureInitialPoses()
  {
    bool ready = true;
    if (enable_right_) {
      ready = ready && lookupPose(right_controlled_link_, start_right_pose_);
    }
    if (enable_left_) {
      ready = ready && lookupPose(left_controlled_link_, start_left_pose_);
    }
    if (ready) {
      if (enable_right_) {
        initial_right_pose_ = withPositionOffset(start_right_pose_, true);
      }
      if (enable_left_) {
        initial_left_pose_ = withPositionOffset(start_left_pose_, false);
      }
    }
    return ready;
  }

  void activateSequence()
  {
    initialized_ = true;
    active_ = true;
    last_phase_index_ = -1;
    start_time_ = this->now() + rclcpp::Duration::from_seconds(startup_delay_sec_);
  }

  void initializeIfReady()
  {
    if (initialized_ || !start_on_launch_) {
      return;
    }

    if (!captureInitialPoses()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for initial end-effector poses from TF.");
      return;
    }

    activateSequence();
    if (init_timer_) {
      init_timer_->cancel();
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Captured initial TF poses. Starting orientation demo in %.2f sec.",
      startup_delay_sec_);
  }

  void startCallback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg || !msg->data || active_) {
      return;
    }

    if (!captureInitialPoses()) {
      RCLCPP_WARN(this->get_logger(), "Received start signal, but TF poses are not ready yet.");
      return;
    }

    activateSequence();
    RCLCPP_INFO(this->get_logger(), "Starting one orientation demo cycle.");
  }

  void publishDone()
  {
    std_msgs::msg::Bool msg;
    msg.data = true;
    done_pub_->publish(msg);
    active_ = false;
    initialized_ = false;
    RCLCPP_INFO(this->get_logger(), "Orientation demo finished. Waiting for next start signal.");
  }

  bool lookupPose(const std::string & child_frame, geometry_msgs::msg::PoseStamped & pose_out)
  {
    try {
      const auto tf = tf_buffer_->lookupTransform(base_frame_, child_frame, tf2::TimePointZero);
      pose_out.header = tf.header;
      pose_out.header.frame_id = base_frame_;
      pose_out.pose.position.x = tf.transform.translation.x;
      pose_out.pose.position.y = tf.transform.translation.y;
      pose_out.pose.position.z = tf.transform.translation.z;
      pose_out.pose.orientation = tf.transform.rotation;
      return true;
    } catch (const std::exception &) {
      return false;
    }
  }

  void publishReference()
  {
    if (!initialized_ || !active_) {
      return;
    }

    const rclcpp::Time now = this->now();
    if (now < start_time_) {
      return;
    }

    const double elapsed = (now - start_time_).seconds();
    if (elapsed < z_offset_move_duration_sec_) {
      const double alpha = elapsed / std::max(z_offset_move_duration_sec_, 1e-6);
      if (enable_right_) {
        right_goal_pub_->publish(createMoveLMessage(
          interpolatePose(start_right_pose_, initial_right_pose_, alpha), OrientationOffset{0.0, 0.0, 0.0, "", -1}, now));
      }
      if (enable_left_) {
        left_goal_pub_->publish(createMoveLMessage(
          interpolatePose(start_left_pose_, initial_left_pose_, alpha), OrientationOffset{0.0, 0.0, 0.0, "", -1}, now));
      }
      return;
    }

    const double orientation_elapsed = elapsed - z_offset_move_duration_sec_;
    if (!loop_ && orientation_elapsed >= totalDuration()) {
      publishDone();
      return;
    }

    const OrientationOffset offset = evaluateOrientationOffset(orientation_elapsed);
    announcePhaseIfNeeded(offset);

    if (enable_right_) {
      right_goal_pub_->publish(createMoveLMessage(initial_right_pose_, offset, now));
    }
    if (enable_left_) {
      left_goal_pub_->publish(createMoveLMessage(
        initial_left_pose_, mirrorOrientationOffset(offset), now));
    }
  }

  OrientationOffset evaluateOrientationOffset(double elapsed_sec)
  {
    const double roll_amp = degToRad(max_roll_deg_);
    const double pitch_amp = degToRad(max_pitch_deg_);
    const double yaw_amp = degToRad(max_yaw_deg_);
    const double circle_radius = degToRad(circle_radius_deg_);

    double sequence_time = elapsed_sec;
    if (loop_ && totalDuration() > 1e-6) {
      sequence_time = std::fmod(elapsed_sec, totalDuration());
      if (sequence_time < 0.0) {
        sequence_time += totalDuration();
      }
    } else {
      sequence_time = std::clamp(elapsed_sec, 0.0, totalDuration());
    }

    if (sequence_time < axis_sweep_duration_sec_) {
      const double phase = 2.0 * M_PI * sequence_time / axis_sweep_duration_sec_;
      return {roll_amp * std::sin(phase), 0.0, 0.0, "roll sweep", 0};
    }
    sequence_time -= axis_sweep_duration_sec_;

    if (sequence_time < axis_sweep_duration_sec_) {
      const double phase = 2.0 * M_PI * sequence_time / axis_sweep_duration_sec_;
      return {0.0, pitch_amp * std::sin(phase), 0.0, "pitch sweep", 1};
    }
    sequence_time -= axis_sweep_duration_sec_;

    if (sequence_time < axis_sweep_duration_sec_) {
      const double phase = 2.0 * M_PI * sequence_time / axis_sweep_duration_sec_;
      return {0.0, 0.0, yaw_amp * std::sin(phase), "yaw sweep", 2};
    }
    sequence_time -= axis_sweep_duration_sec_;

    if (sequence_time < circle_entry_duration_sec_) {
      const double alpha = 0.5 * (1.0 - std::cos(M_PI * sequence_time / circle_entry_duration_sec_));
      return {circle_radius * alpha, 0.0, 0.0, "circle entry", 3};
    }
    sequence_time -= circle_entry_duration_sec_;

    if (sequence_time < circle_duration_sec_) {
      const double theta = 2.0 * M_PI * sequence_time / circle_duration_sec_;
      return {circle_radius * std::cos(theta), circle_radius * std::sin(theta), 0.0,
        "orientation circle", 4};
    }
    sequence_time -= circle_duration_sec_;

    const double alpha = 0.5 * (1.0 + std::cos(M_PI * sequence_time / circle_exit_duration_sec_));
    return {circle_radius * alpha, 0.0, 0.0, "circle exit", 5};
  }

  void announcePhaseIfNeeded(const OrientationOffset & offset)
  {
    if (offset.phase_index == last_phase_index_) {
      return;
    }

    last_phase_index_ = offset.phase_index;
    RCLCPP_INFO(
      this->get_logger(),
      "Switching to %s phase. Offsets [deg] roll/pitch/yaw = %.1f / %.1f / %.1f",
      offset.phase_name.c_str(),
      offset.roll * 180.0 / M_PI,
      offset.pitch * 180.0 / M_PI,
      offset.yaw * 180.0 / M_PI);
  }

  OrientationOffset mirrorOrientationOffset(const OrientationOffset & offset) const
  {
    return {
      -offset.roll,
      -offset.pitch,
      -offset.yaw,
      offset.phase_name,
      offset.phase_index
    };
  }

  robotis_interfaces::msg::MoveL createMoveLMessage(
    const geometry_msgs::msg::PoseStamped & initial_pose,
    const OrientationOffset & offset,
    const rclcpp::Time & stamp) const
  {
    robotis_interfaces::msg::MoveL msg;
    msg.pose.header.stamp = stamp;
    msg.pose.header.frame_id = base_frame_;
    msg.pose.pose.position = initial_pose.pose.position;
    msg.pose.pose.orientation = composeOffsetOrientation(initial_pose.pose.orientation, offset);
    msg.time_from_start = rclcpp::Duration::from_seconds(command_duration_sec_);
    return msg;
  }

  geometry_msgs::msg::Quaternion composeOffsetOrientation(
    const geometry_msgs::msg::Quaternion & initial_orientation,
    const OrientationOffset & offset) const
  {
    const Eigen::Quaterniond q_initial(
      initial_orientation.w,
      initial_orientation.x,
      initial_orientation.y,
      initial_orientation.z);

    const Eigen::Quaterniond q_roll(Eigen::AngleAxisd(offset.roll, Eigen::Vector3d::UnitX()));
    const Eigen::Quaterniond q_pitch(Eigen::AngleAxisd(offset.pitch, Eigen::Vector3d::UnitY()));
    const Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(offset.yaw, Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond q_target = (q_initial * q_yaw * q_pitch * q_roll).normalized();

    geometry_msgs::msg::Quaternion orientation;
    orientation.w = q_target.w();
    orientation.x = q_target.x();
    orientation.y = q_target.y();
    orientation.z = q_target.z();
    return orientation;
  }

  std::string base_frame_;
  std::string right_controlled_link_;
  std::string left_controlled_link_;
  std::string right_goal_topic_;
  std::string left_goal_topic_;
  std::string start_topic_;
  std::string done_topic_;
  bool enable_right_;
  bool enable_left_;
  bool start_on_launch_;
  double publish_rate_hz_;
  double command_duration_sec_;
  double startup_delay_sec_;
  bool loop_;
  double x_offset_;
  double y_separation_offset_;
  double z_offset_;
  double z_offset_move_duration_sec_;
  double axis_sweep_duration_sec_;
  double circle_entry_duration_sec_;
  double circle_duration_sec_;
  double circle_exit_duration_sec_;
  double max_roll_deg_;
  double max_pitch_deg_;
  double max_yaw_deg_;
  double circle_radius_deg_;
  bool initialized_;
  bool active_;
  int last_phase_index_;
  rclcpp::Time start_time_;
  geometry_msgs::msg::PoseStamped start_right_pose_;
  geometry_msgs::msg::PoseStamped start_left_pose_;
  geometry_msgs::msg::PoseStamped initial_right_pose_;
  geometry_msgs::msg::PoseStamped initial_left_pose_;
  rclcpp::Publisher<robotis_interfaces::msg::MoveL>::SharedPtr right_goal_pub_;
  rclcpp::Publisher<robotis_interfaces::msg::MoveL>::SharedPtr left_goal_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;
  rclcpp::TimerBase::SharedPtr init_timer_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cyclo_motion_controller_ros::OrientationReferenceDemoNode>());
  rclcpp::shutdown();
  return 0;
}
