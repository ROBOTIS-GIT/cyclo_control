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

#include "cyclo_motion_controller_ros/utils/reference_checker_node.hpp"

#include <cmath>

namespace cyclo_motion_controller_ros
{
ReferenceDivergenceChecker::ReferenceDivergenceChecker()
: Node("reference_checker"),
  r_goal_prev_set_(false),
  l_goal_prev_set_(false),
  reference_active_(false)
{
  ref_pos_jump_threshold_ = this->declare_parameter("ref_pos_jump_threshold", 0.1);
  ref_ori_jump_threshold_deg_ = this->declare_parameter("ref_ori_jump_threshold_deg", 30.0);
  r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
  l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));
  reference_divergence_topic_ = this->declare_parameter(
    "reference_divergence_topic", std::string("/reference_diverged"));
  reactivate_topic_ = this->declare_parameter("reactivate_topic", std::string("/reactivate"));

  reference_divergence_pub_ = this->create_publisher<std_msgs::msg::Bool>(
          reference_divergence_topic_, 10);

  r_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
          r_goal_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
          std::bind(&ReferenceDivergenceChecker::rightGoalPoseCallback, this,
      std::placeholders::_1));
  l_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
          l_goal_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
          std::bind(&ReferenceDivergenceChecker::leftGoalPoseCallback, this,
      std::placeholders::_1));
  ref_reactivate_sub_ = this->create_subscription<std_msgs::msg::Bool>(
          reactivate_topic_, 10,
          std::bind(&ReferenceDivergenceChecker::referenceReactivateCallback, this,
      std::placeholders::_1));
}

void ReferenceDivergenceChecker::referenceReactivateCallback(
  const std_msgs::msg::Bool::SharedPtr msg)
{
  if (!msg) {
    return;
  }

  // Relative references establish a new origin every time teleoperation is enabled.
  // Never compare a sample from the previous calibration session with the new one.
  r_goal_prev_set_ = false;
  l_goal_prev_set_ = false;
  reference_active_ = msg->data;
}

void ReferenceDivergenceChecker::rightGoalPoseCallback(
  const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!reference_active_) {
    return;
  }
  checkReferenceJump("right_goal", r_goal_prev_, msg->pose, r_goal_prev_set_);
  r_goal_prev_ = msg->pose;
  r_goal_prev_set_ = true;
}

void ReferenceDivergenceChecker::leftGoalPoseCallback(
  const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!reference_active_) {
    return;
  }
  checkReferenceJump("left_goal", l_goal_prev_, msg->pose, l_goal_prev_set_);
  l_goal_prev_ = msg->pose;
  l_goal_prev_set_ = true;
}

void ReferenceDivergenceChecker::checkReferenceJump(
  const std::string & name,
  const geometry_msgs::msg::Pose & prev_pose,
  const geometry_msgs::msg::Pose & new_pose,
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
    std_msgs::msg::Bool msg;
    msg.data = true;
    reference_divergence_pub_->publish(msg);
    reference_active_ = false;
    r_goal_prev_set_ = false;
    l_goal_prev_set_ = false;
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      2000,
      "Reference jump detected (%s): pos=%.3f m, ori=%.1f deg "
      "(thresholds: %.3f m, %.1f deg)",
      name.c_str(),
      pos_dist,
      angle_deg,
      ref_pos_jump_threshold_,
      ref_ori_jump_threshold_deg_);
  }
}

}  // namespace cyclo_motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyclo_motion_controller_ros::ReferenceDivergenceChecker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
