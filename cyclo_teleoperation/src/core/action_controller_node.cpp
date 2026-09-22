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

#include "cyclo_teleoperation/core/action_controller_node.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <robotis_interfaces/srv/set_control_mode.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "cyclo_teleoperation/core/robot_teleoperation.hpp"
#include "cyclo_teleoperation/core/soft_hold.hpp"
#include "cyclo_teleoperation/core/teleoperation_mode.hpp"
#include "cyclo_teleoperation/core/teleoperation_qp.hpp"

namespace cyclo_teleoperation
{
namespace
{
constexpr char kJointReference[] = "absolute_joint_position";
constexpr char kPoseReference[] = "absolute_eef_pose";
}

class ActionControllerNode : public rclcpp::Node
{
public:
  ActionControllerNode()
  : Node("cyclo_action_controller"),
    robot_loader_("cyclo_teleoperation", "cyclo_teleoperation::RobotTeleoperation"),
    mode_loader_("cyclo_teleoperation", "cyclo_teleoperation::TeleoperationMode")
  {
    declareParameters();
    const auto robot_plugin = get_parameter("robot.plugin").as_string();
    if (robot_plugin.empty() || !robot_loader_.isClassAvailable(robot_plugin)) {
      throw std::runtime_error("robot.plugin is not registered with pluginlib: " + robot_plugin);
    }
    robot_ = robot_loader_.createSharedInstance(robot_plugin);
    if (!robot_->configure(
        *this, get_parameter("robot.parameter_prefix").as_string(),
        [](const ControlRequest &) {}))
    {
      throw std::runtime_error("Failed to initialize robot control plugin: " + robot_plugin);
    }
    validateRobot();
    const size_t group_count = groupStateCount();
    cartesian_references_.assign(group_count, CartesianReference{});
    joint_action_received_.assign(group_count, false);
    pose_action_received_.assign(group_count, false);
    gripper_received_.assign(group_count, false);
    last_joint_action_times_.assign(group_count, rclcpp::Time(0, 0, RCL_ROS_TIME));
    last_pose_action_times_.assign(group_count, rclcpp::Time(0, 0, RCL_ROS_TIME));
    gripper_command_ = robot_->followerAuxiliaryPosition();
    gripper_hold_target_ = gripper_command_;

    qp_ = std::make_unique<TeleoperationQP>(robot_->followerKinematics());
    qp_->setControllerParameters(
      get_parameter("constraints.slack_penalty").as_double(),
      get_parameter("constraints.cbf_alpha").as_double(),
      get_parameter("constraints.collision_buffer").as_double(),
      get_parameter("constraints.collision_safe_distance").as_double());

    const auto follower_qos = rclcpp::SensorDataQoS().keep_last(1);
    follower_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      robot_->followerJointStatesTopic(), follower_qos,
      std::bind(&ActionControllerNode::followerCallback, this, std::placeholders::_1));
    createActionSubscriptions();
    set_mode_service_ = create_service<robotis_interfaces::srv::SetControlMode>(
      "~/set_control_mode",
      std::bind(
        &ActionControllerNode::setModeCallback, this,
        std::placeholders::_1, std::placeholders::_2));

    requested_mode_ = static_cast<uint16_t>(get_parameter("default_control_mode").as_int());
    const double frequency = std::max(1.0, get_parameter("control_frequency").as_double());
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / frequency),
      std::bind(&ActionControllerNode::controlLoop, this));
    RCLCPP_INFO(
      get_logger(), "Cyclo action controller is waiting for complete follower feedback");
  }

private:
  void declareParameters()
  {
    declare_parameter("robot.plugin", "");
    declare_parameter("robot.parameter_prefix", "");
    declare_parameter("control_frequency", 100.0);
    declare_parameter("joint_state_timeout", 0.5);
    declare_parameter("action_timeout", 0.5);
    declare_parameter("action_reference_frame", "base_link");
    declare_parameter("left_joint_action_topic", "~/left/raw_joint_trajectory");
    declare_parameter("right_joint_action_topic", "~/right/raw_joint_trajectory");
    declare_parameter("left_eef_action_topic", "~/left/eef_pose");
    declare_parameter("right_eef_action_topic", "~/right/eef_pose");
    declare_parameter("hold.kp", 20.0);
    declare_parameter("hold.max_correction_velocity", 0.2);
    declare_parameter("hold.tracking_weight", 100.0);
    declare_parameter("constraints.slack_penalty", 1000.0);
    declare_parameter("constraints.cbf_alpha", 50.0);
    declare_parameter("constraints.collision_buffer", 0.05);
    declare_parameter("constraints.collision_safe_distance", 0.02);
    declare_parameter("constraints.damping_weight", 0.1);
    declare_parameter<std::vector<int64_t>>(
      "available_control_modes", std::vector<int64_t>{});
    declare_parameter("default_control_mode", 1);

    const auto available_modes =
      get_parameter("available_control_modes").as_integer_array();
    for (const int64_t raw_mode : available_modes) {
      if (raw_mode <= 0 || raw_mode > UINT16_MAX) {
        throw std::runtime_error("Control mode IDs must be in the uint16 range");
      }
      const auto mode = static_cast<uint16_t>(raw_mode);
      const std::string prefix = "control_modes." + std::to_string(mode);
      const std::string plugin = declare_parameter(prefix + ".plugin", "");
      const std::string reference_type =
        declare_parameter(prefix + ".reference_type", "");
      if (plugin.empty() || !mode_loader_.isClassAvailable(plugin)) {
        throw std::runtime_error(prefix + ".plugin is not registered with pluginlib: " + plugin);
      }
      if (reference_type != kJointReference && reference_type != kPoseReference) {
        throw std::runtime_error(
                prefix + ".reference_type must be absolute_joint_position or absolute_eef_pose");
      }
      mode_plugins_[mode] = plugin;
      mode_reference_types_[mode] = reference_type;
    }
    if (mode_plugins_.empty()) {
      throw std::runtime_error("available_control_modes must not be empty");
    }
    const auto default_mode = static_cast<uint16_t>(get_parameter("default_control_mode").as_int());
    if (mode_plugins_.count(default_mode) == 0) {
      throw std::runtime_error("default_control_mode is not available");
    }
  }

  void validateRobot() const
  {
    if (
      robot_->dof() <= 0 || !robot_->followerKinematics() ||
      robot_->followerPosition().size() != robot_->dof() ||
      robot_->modeConfiguration().control_groups.empty())
    {
      throw std::runtime_error("Robot control plugin has an invalid follower model");
    }
  }

  size_t groupStateCount() const
  {
    size_t result = 0;
    for (const auto & group : robot_->modeConfiguration().control_groups) {
      result = std::max(result, static_cast<size_t>(group.id) + 1);
    }
    return result;
  }

  const ControlGroupConfiguration * groupByName(const std::string & name) const
  {
    return robot_->modeConfiguration().findGroup(name);
  }

  void createActionSubscriptions()
  {
    const auto * left = groupByName("left");
    const auto * right = groupByName("right");
    if (left == nullptr || right == nullptr) {
      throw std::runtime_error("AI Worker action control requires left and right groups");
    }
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    auto joint_callback = [this](const ControlGroupId group_id) {
        return [this, group_id](const trajectory_msgs::msg::JointTrajectory::SharedPtr message) {
                 const std::string reference_type = activeReferenceType();
                 if (reference_type == kJointReference) {
                   if (!robot_->updateLeaderReference(*message, group_id)) {
                     RCLCPP_WARN_THROTTLE(
                       get_logger(), *get_clock(), 2000,
                       "Invalid raw joint action rejected for group %u",
                       static_cast<unsigned int>(group_id));
                     return;
                   }
                   gripper_received_.at(group_id) = true;
                   joint_action_received_.at(group_id) = true;
                   last_joint_action_times_.at(group_id) = now();
                 } else if (reference_type == kPoseReference) {
                   if (!robot_->updateGripperReference(*message, group_id)) {
                     RCLCPP_WARN_THROTTLE(
                       get_logger(), *get_clock(), 2000,
                       "Invalid gripper action rejected for group %u",
                       static_cast<unsigned int>(group_id));
                     return;
                   }
                   gripper_received_.at(group_id) = true;
                 }
               };
      };
    auto pose_callback = [this](const ControlGroupId group_id) {
        return [this, group_id](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
                 if (!updatePoseReference(*message, group_id)) {
                   RCLCPP_WARN_THROTTLE(
                     get_logger(), *get_clock(), 2000,
                     "Invalid absolute EEF action rejected for group %u",
                     static_cast<unsigned int>(group_id));
                   return;
                 }
                 if (activeReferenceType() == kPoseReference) {
                   pose_action_received_.at(group_id) = true;
                   last_pose_action_times_.at(group_id) = now();
                 }
               };
      };
    joint_action_subscriptions_.push_back(
        create_subscription<trajectory_msgs::msg::JointTrajectory>(
        get_parameter("left_joint_action_topic").as_string(), qos, joint_callback(left->id)));
    joint_action_subscriptions_.push_back(
        create_subscription<trajectory_msgs::msg::JointTrajectory>(
        get_parameter("right_joint_action_topic").as_string(), qos, joint_callback(right->id)));
    pose_action_subscriptions_.push_back(create_subscription<geometry_msgs::msg::PoseStamped>(
      get_parameter("left_eef_action_topic").as_string(), qos, pose_callback(left->id)));
    pose_action_subscriptions_.push_back(create_subscription<geometry_msgs::msg::PoseStamped>(
      get_parameter("right_eef_action_topic").as_string(), qos, pose_callback(right->id)));
  }

  bool updatePoseReference(
    const geometry_msgs::msg::PoseStamped & message, const ControlGroupId group_id)
  {
    if (
      group_id >= cartesian_references_.size() ||
      message.header.frame_id != get_parameter("action_reference_frame").as_string())
    {
      return false;
    }
    const auto & p = message.pose.position;
    const auto & q = message.pose.orientation;
    if (
      !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
      !std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) ||
      !std::isfinite(q.w))
    {
      return false;
    }
    Eigen::Quaterniond orientation(q.w, q.x, q.y, q.z);
    if (orientation.norm() < 1e-9) {
      return false;
    }
    orientation.normalize();
    CartesianReference reference;
    reference.pose.translation() << p.x, p.y, p.z;
    reference.pose.linear() = orientation.toRotationMatrix();
    reference.sequence = cartesian_references_[group_id].sequence + 1;
    reference.valid = true;
    cartesian_references_[group_id] = std::move(reference);
    return true;
  }

  std::string activeReferenceType() const
  {
    const auto iter = mode_reference_types_.find(active_mode_ ==
        0 ? requested_mode_ : active_mode_);
    return iter == mode_reference_types_.end() ? std::string{} : iter->second;
  }

  void followerCallback(const sensor_msgs::msg::JointState::SharedPtr message)
  {
    if (!robot_->updateFollowerState(*message)) {
      return;
    }
    last_follower_time_ = now();
    follower_received_ = true;
    if (!command_initialized_ || follower_timeout_active_) {
      syncAllToFeedback();
      clearActionFreshness();
      follower_timeout_active_ = false;
    }
  }

  bool feedbackFresh() const
  {
    return follower_received_ &&
           (now() - last_follower_time_).seconds() <=
           get_parameter("joint_state_timeout").as_double();
  }

  ControlGroupMask freshActionGroups() const
  {
    const double timeout = get_parameter("action_timeout").as_double();
    const bool pose_mode = activeReferenceType() == kPoseReference;
    ControlGroupMask result = 0;
    for (const auto & group : robot_->modeConfiguration().control_groups) {
      const bool received = pose_mode ?
        pose_action_received_.at(group.id) : joint_action_received_.at(group.id);
      const rclcpp::Time & stamp = pose_mode ?
        last_pose_action_times_.at(group.id) : last_joint_action_times_.at(group.id);
      if (received && (now() - stamp).seconds() <= timeout) {
        result |= controlGroupBit(group.id);
      }
    }
    return result;
  }

  ModeContext makeContext(const ControlGroupMask enabled_groups) const
  {
    return ModeContext{
      command_position_, command_velocity_, robot_->followerPosition(),
      robot_->leaderReference(), robot_->leaderPosition(), cartesian_references_,
      robot_->followerAuxiliaryPosition(), robot_->controlGroupStates(),
      enabled_groups, enabled_groups, 0, now().seconds(),
      1.0 / std::max(1.0, get_parameter("control_frequency").as_double())};
  }

  void syncAllToFeedback()
  {
    command_position_ = robot_->followerPosition();
    command_velocity_ = Eigen::VectorXd::Zero(robot_->dof());
    hold_target_ = command_position_;
    gripper_command_ = robot_->followerAuxiliaryPosition();
    gripper_hold_target_ = gripper_command_;
    command_initialized_ = true;
    active_groups_ = 0;
  }

  void clearActionFreshness()
  {
    std::fill(joint_action_received_.begin(), joint_action_received_.end(), false);
    std::fill(pose_action_received_.begin(), pose_action_received_.end(), false);
    std::fill(gripper_received_.begin(), gripper_received_.end(), false);
    for (auto & reference : cartesian_references_) {
      reference.valid = false;
    }
  }

  void syncGroupsToFeedback(const ControlGroupMask groups)
  {
    for (const auto & group : robot_->modeConfiguration().control_groups) {
      if (!containsControlGroup(groups, group.id)) {
        continue;
      }
      for (const int index : group.follower_joint_indices) {
        command_position_[index] = robot_->followerPosition()[index];
        command_velocity_[index] = 0.0;
        hold_target_[index] = robot_->followerPosition()[index];
      }
      gripper_command_[group.id] = robot_->followerAuxiliaryPosition()[group.id];
      gripper_hold_target_[group.id] = gripper_command_[group.id];
    }
  }

  bool loadRequestedMode()
  {
    try {
      mode_.reset();
      mode_ = mode_loader_.createSharedInstance(mode_plugins_.at(requested_mode_));
      const std::string prefix = "control_modes." + std::to_string(requested_mode_);
      if (!mode_->configure(*this, prefix, robot_->modeConfiguration())) {
        throw std::runtime_error("mode configuration was rejected");
      }
      if (!mode_->activate(makeContext(0))) {
        throw std::runtime_error("mode activation was rejected");
      }
      active_mode_ = requested_mode_;
      active_groups_ = 0;
      clearActionFreshness();
      RCLCPP_INFO(get_logger(), "Activated action control mode %u", active_mode_);
      return true;
    } catch (const std::exception & error) {
      mode_.reset();
      active_mode_ = 0;
      RCLCPP_ERROR(get_logger(), "Failed to activate action control mode: %s", error.what());
      return false;
    }
  }

  void updateActiveGroups(const ControlGroupMask desired)
  {
    if (desired == active_groups_) {
      return;
    }
    const ControlGroupMask changed = desired ^ active_groups_;
    const ControlGroupMask enabled = desired & ~active_groups_;
    syncGroupsToFeedback(changed);
    active_groups_ = desired;
    if (enabled != 0 && mode_) {
      mode_->onGroupsEnabled(enabled, makeContext(active_groups_));
    }
  }

  void updateGripperCommand()
  {
    const auto & reference = robot_->leaderAuxiliaryReference();
    for (const auto & group : robot_->modeConfiguration().control_groups) {
      if (group.id < gripper_received_.size() && gripper_received_[group.id]) {
        gripper_command_[group.id] = reference[group.id];
        gripper_hold_target_[group.id] = reference[group.id];
      } else {
        gripper_command_[group.id] = gripper_hold_target_[group.id];
      }
    }
  }

  void controlLoop()
  {
    if (!feedbackFresh()) {
      if (follower_received_ && !follower_timeout_active_) {
        follower_timeout_active_ = true;
        command_initialized_ = false;
        active_groups_ = 0;
        RCLCPP_ERROR(get_logger(), "Follower feedback timed out; action control stopped");
      }
      return;
    }
    if (!command_initialized_) {
      syncAllToFeedback();
      clearActionFreshness();
    }
    if (!mode_ && !loadRequestedMode()) {
      return;
    }

    updateActiveGroups(freshActionGroups());
    updateGripperCommand();
    if (active_groups_ == 0) {
      command_position_ = hold_target_;
      command_velocity_.setZero();
      robot_->publish(command_position_, gripper_command_);
      return;
    }

    robot_->followerKinematics()->updateState(command_position_, command_velocity_);
    const ModeContext context = makeContext(active_groups_);
    ModeOutput output;
    output.reset(robot_->dof(), get_parameter("constraints.damping_weight").as_double());
    if (!mode_->update(context, output)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Action controller mode rejected the current reference; holding last command");
      robot_->publish(command_position_, gripper_command_);
      return;
    }
    const ControlGroupMask controlled_groups = mode_->controlledGroups(context);
    applySoftHold(
      output, command_position_, hold_target_,
      robot_->modeConfiguration().control_groups, controlled_groups,
      get_parameter("hold.kp").as_double(),
      get_parameter("hold.max_correction_velocity").as_double(),
      get_parameter("hold.tracking_weight").as_double());
    qp_->setModeOutput(output);
    qp_->setControllerParameters(
      get_parameter("constraints.slack_penalty").as_double(),
      get_parameter("constraints.cbf_alpha").as_double(),
      get_parameter("constraints.collision_buffer").as_double(),
      get_parameter("constraints.collision_safe_distance").as_double());
    Eigen::VectorXd optimal_velocity;
    if (!qp_->solve(optimal_velocity)) {
      command_velocity_.setZero();
      robot_->publish(command_position_, gripper_command_);
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Action controller QP failed; holding the last command and retrying");
      return;
    }
    command_position_ += context.dt * optimal_velocity;
    command_velocity_ = optimal_velocity;
    robot_->publish(command_position_, gripper_command_);
  }

  void setModeCallback(
    const std::shared_ptr<robotis_interfaces::srv::SetControlMode::Request> request,
    std::shared_ptr<robotis_interfaces::srv::SetControlMode::Response> response)
  {
    if (mode_plugins_.count(request->control_mode) == 0) {
      response->accepted = false;
      response->message = "Unknown action control mode";
      return;
    }
    if (active_groups_ != 0) {
      response->accepted = false;
      response->message = "Control mode can only be changed while every group is holding";
      return;
    }
    requested_mode_ = request->control_mode;
    mode_.reset();
    response->accepted = true;
    response->transition_id = ++transition_id_;
    response->message = "Action control mode change accepted";
  }

  pluginlib::ClassLoader<RobotTeleoperation> robot_loader_;
  pluginlib::ClassLoader<TeleoperationMode> mode_loader_;
  std::shared_ptr<RobotTeleoperation> robot_;
  std::shared_ptr<TeleoperationMode> mode_;
  std::unique_ptr<TeleoperationQP> qp_;
  std::unordered_map<uint16_t, std::string> mode_plugins_;
  std::unordered_map<uint16_t, std::string> mode_reference_types_;
  uint16_t requested_mode_ = 1;
  uint16_t active_mode_ = 0;
  uint64_t transition_id_ = 0;
  ControlGroupMask active_groups_ = 0;
  bool follower_received_ = false;
  bool follower_timeout_active_ = false;
  bool command_initialized_ = false;
  Eigen::VectorXd command_position_;
  Eigen::VectorXd command_velocity_;
  Eigen::VectorXd hold_target_;
  GroupAuxiliaryPositions gripper_command_;
  GroupAuxiliaryPositions gripper_hold_target_;
  GroupCartesianReferences cartesian_references_;
  std::vector<bool> joint_action_received_;
  std::vector<bool> pose_action_received_;
  std::vector<bool> gripper_received_;
  std::vector<rclcpp::Time> last_joint_action_times_;
  std::vector<rclcpp::Time> last_pose_action_times_;
  rclcpp::Time last_follower_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr follower_subscription_;
  std::vector<rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr>
  joint_action_subscriptions_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr>
  pose_action_subscriptions_;
  rclcpp::Service<robotis_interfaces::srv::SetControlMode>::SharedPtr set_mode_service_;
  rclcpp::TimerBase::SharedPtr timer_;
};

std::shared_ptr<rclcpp::Node> makeActionControllerNode()
{
  return std::make_shared<ActionControllerNode>();
}
}  // namespace cyclo_teleoperation
