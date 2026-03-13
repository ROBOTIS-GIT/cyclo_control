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

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <string>

namespace motion_controller_ros
{ 
  /**
   * @brief ROS 2 node for checking reference divergence.
   * 
   * This node subscribes to target end-effector pose and checks if the reference has diverged.
   */
  class ReferenceDivergenceChecker : public rclcpp::Node
  {
    public:
      ReferenceDivergenceChecker();

    private:
      void rightGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
      void leftGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
      void referenceReactivateCallback(const std_msgs::msg::Bool::SharedPtr msg);
      void checkReferenceJump(
        const std::string& name,
        const geometry_msgs::msg::Pose& prev_pose,
        const geometry_msgs::msg::Pose& new_pose,
        bool has_prev);

      double ref_pos_jump_threshold_;
      double ref_ori_jump_threshold_deg_;
      std::string reference_divergence_topic_;
      std::string r_goal_pose_topic_;
      std::string l_goal_pose_topic_;

      rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr reference_divergence_pub_;
      rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pose_sub_;
      rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr l_goal_pose_sub_;
      rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr ref_reactivate_sub_;

      geometry_msgs::msg::Pose r_goal_prev_;
      geometry_msgs::msg::Pose l_goal_prev_;
      bool r_goal_prev_set_;
      bool l_goal_prev_set_;
  };
}  // namespace motion_controller_ros

