#include "motion_controller_ros/reference_checker_node.hpp"

#include <cmath>

namespace motion_controller_ros
{
  ReferenceDivergenceChecker::ReferenceDivergenceChecker()
    : Node("reference_checker"),
      divergence_active_(false),
      r_goal_prev_set_(false),
      l_goal_prev_set_(false)
  {
    ref_pos_jump_threshold_ = this->declare_parameter("ref_pos_jump_threshold", 0.1);
    ref_ori_jump_threshold_deg_ = this->declare_parameter("ref_ori_jump_threshold_deg", 30.0);
    reactivate_topic_ = this->declare_parameter("reactivate_topic", std::string("/reset"));
    r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
    l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));

    reference_divergence_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      "/reference_diverged", 10);

    r_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      r_goal_pose_topic_, 10,
      std::bind(&ReferenceDivergenceChecker::rightGoalPoseCallback, this, std::placeholders::_1));
    l_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      l_goal_pose_topic_, 10,
      std::bind(&ReferenceDivergenceChecker::leftGoalPoseCallback, this, std::placeholders::_1));

    ref_reactivate_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      reactivate_topic_, 10,
      std::bind(&ReferenceDivergenceChecker::referenceReactivateCallback, this, std::placeholders::_1));
  }

  void ReferenceDivergenceChecker::rightGoalPoseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    checkReferenceJump("right_goal", r_goal_prev_, msg->pose, r_goal_prev_set_);
    r_goal_prev_ = msg->pose;
    r_goal_prev_set_ = true;
  }

  void ReferenceDivergenceChecker::leftGoalPoseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    checkReferenceJump("left_goal", l_goal_prev_, msg->pose, l_goal_prev_set_);
    l_goal_prev_ = msg->pose;
    l_goal_prev_set_ = true;
  }

  void ReferenceDivergenceChecker::referenceReactivateCallback(
    const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg->data) {
      return;
    }
    divergence_active_ = false;
  }

  void ReferenceDivergenceChecker::checkReferenceJump(
    const std::string& name,
    const geometry_msgs::msg::Pose& prev_pose,
    const geometry_msgs::msg::Pose& new_pose,
    bool has_prev)
  {
    if (!has_prev) {
      return;
    }

    const Eigen::Vector3d prev_pos(prev_pose.position.x, prev_pose.position.y, prev_pose.position.z);
    const Eigen::Vector3d new_pos(new_pose.position.x, new_pose.position.y, new_pose.position.z);
    const double pos_dist = (new_pos - prev_pos).norm();

    const Eigen::Quaterniond q_prev(
      prev_pose.orientation.w,
      prev_pose.orientation.x,
      prev_pose.orientation.y,
      prev_pose.orientation.z);
    const Eigen::Quaterniond q_new(
      new_pose.orientation.w,
      new_pose.orientation.x,
      new_pose.orientation.y,
      new_pose.orientation.z);
    const double dot = std::abs(q_prev.dot(q_new));
    const double clamped_dot = std::min(1.0, std::max(-1.0, dot));
    const double angle_rad = 2.0 * std::acos(clamped_dot);
    const double angle_deg = angle_rad * 180.0 / M_PI;

    if (pos_dist > ref_pos_jump_threshold_ || angle_deg > ref_ori_jump_threshold_deg_) {
      divergence_active_ = true;
      std_msgs::msg::Bool msg;
      msg.data = true;
      reference_divergence_pub_->publish(msg);
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Reference jump detected (%s): pos=%.3f m, ori=%.1f deg (thresholds: %.3f m, %.1f deg)",
        name.c_str(),
        pos_dist,
        angle_deg,
        ref_pos_jump_threshold_,
        ref_ori_jump_threshold_deg_);
    }
  }

}  // namespace motion_controller_ros

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<motion_controller_ros::ReferenceDivergenceChecker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
