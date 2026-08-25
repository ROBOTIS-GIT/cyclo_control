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

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "cyclo_teleoperation/core/robot_teleoperation.hpp"
#include "cyclo_teleoperation/core/teleoperation_mode.hpp"
#include "cyclo_teleoperation/core/teleoperation_qp.hpp"
#include "cyclo_teleoperation/core/teleoperation_node.hpp"
#include "cyclo_teleoperation/core/soft_hold.hpp"
#include "cyclo_teleoperation/core/pose_sequence_manager.hpp"

namespace cyclo_teleoperation
{
class TeleoperationNode : public rclcpp::Node
{
public:
  TeleoperationNode()
  : Node("cyclo_teleoperation"),
    robot_loader_("cyclo_teleoperation", "cyclo_teleoperation::RobotTeleoperation"),
    mode_loader_("cyclo_teleoperation", "cyclo_teleoperation::TeleoperationMode")
  {
    declareParameters();
    const auto robot_plugin = get_parameter("robot.plugin").as_string();
    if (robot_plugin.empty() || !robot_loader_.isClassAvailable(robot_plugin)) {
      throw std::runtime_error(
              "robot.plugin is not registered with pluginlib: " + robot_plugin);
    }
    robot_teleoperation_ = robot_loader_.createSharedInstance(robot_plugin);
    if (!robot_teleoperation_->configure(
        *this, get_parameter("robot.parameter_prefix").as_string(),
        std::bind(&TeleoperationNode::commandCallback, this, std::placeholders::_1)))
    {
      throw std::runtime_error("Failed to initialize robot teleoperation: " + robot_plugin);
    }
    validateRobotConfiguration();
    declarePoseGroupParameters();
    pose_sequences_ = std::make_unique<PoseSequenceManager>();
    if (!pose_sequences_->configure(
        *this, robot_teleoperation_->modeConfiguration(),
        get_parameter("available_control_modes").as_integer_array(),
        get_parameter("available_presets").as_integer_array()))
    {
      throw std::runtime_error("Failed to configure pose sequences");
    }

    qp_ = std::make_unique<TeleoperationQP>(robot_teleoperation_->followerKinematics());
    qp_->setControllerParameters(
      get_parameter("constraints.slack_penalty").as_double(),
      get_parameter("constraints.cbf_alpha").as_double(),
      get_parameter("constraints.collision_buffer").as_double(),
      get_parameter("constraints.collision_safe_distance").as_double());

    follower_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      robot_teleoperation_->followerJointStatesTopic(), 10,
      std::bind(&TeleoperationNode::followerCallback, this, std::placeholders::_1));
    size_t group_state_count = 0;
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      group_state_count = std::max(group_state_count, static_cast<size_t>(group.id) + 1);
    }
    leader_received_.assign(group_state_count, false);
    last_leader_times_.assign(group_state_count, rclcpp::Time(0, 0, RCL_ROS_TIME));
    selected_preset_ids_.assign(group_state_count, 1);
    context_group_states_.assign(group_state_count, ControlGroupState{});
    last_preset_states_.assign(group_state_count, 0);
    last_initial_pose_states_.assign(group_state_count, 0);
    for (const auto & channel : robot_teleoperation_->leaderInputChannels()) {
      leader_subscriptions_.push_back(
        create_subscription<trajectory_msgs::msg::JointTrajectory>(
          channel.topic, 10,
          [this, group = channel.group_id](
            const trajectory_msgs::msg::JointTrajectory::SharedPtr message)
          {
            if (robot_teleoperation_->updateLeaderReference(*message, group)) {
              leader_received_.at(group) = true;
              last_leader_times_.at(group) = now();
            }
          }));
    }

    requested_control_mode_ =
      static_cast<uint16_t>(get_parameter("default_control_mode").as_int());
    parameter_callback_handle_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto & parameter : parameters) {
          const std::string active_prefix =
          "control_modes." + std::to_string(active_control_mode_) + ".";
          if (
            active_control_mode_ != 0 &&
            parameter.get_name().rfind(active_prefix, 0) == 0)
          {
            transition_pending_ = true;
          }
        }
        return result;
      });
    const double frequency = std::max(1.0, get_parameter("control_frequency").as_double());
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / frequency),
      std::bind(&TeleoperationNode::controlLoop, this));
    publishStatus(
      ControlStatus::kHolding,
      "Waiting for complete follower feedback");
  }

private:
  enum class ModeTransitionPhase : uint8_t
  {
    kIdle,
    kExitPose,
    kInitialPose,
    kActivate,
  };

  void validateRobotConfiguration() const
  {
    const auto & configuration = robot_teleoperation_->modeConfiguration();
    const int dof = robot_teleoperation_->dof();
    if (dof <= 0 || !configuration.follower_kinematics || !configuration.leader_kinematics) {
      throw std::runtime_error("Robot profile has an invalid model configuration");
    }
    if (
      robot_teleoperation_->followerPosition().size() != dof ||
      robot_teleoperation_->followerVelocity().size() != dof ||
      robot_teleoperation_->leaderReference().size() != dof)
    {
      throw std::runtime_error("Robot profile state vectors do not match its follower DOF");
    }
    if (configuration.control_groups.empty()) {
      throw std::runtime_error("Robot profile must define at least one control group");
    }

    std::unordered_set<ControlGroupId> group_ids;
    std::unordered_set<std::string> group_names;
    std::unordered_set<int> assigned_joint_indices;
    ControlGroupId largest_group_id = 0;
    for (const auto & group : configuration.control_groups) {
      if (group.id >= 64 || !group_ids.insert(group.id).second) {
        throw std::runtime_error("Control-group IDs must be unique and smaller than 64");
      }
      largest_group_id = std::max(largest_group_id, group.id);
      if (group.name.empty() || !group_names.insert(group.name).second) {
        throw std::runtime_error("Control-group names must be non-empty and unique");
      }
      if (group.follower_joint_indices.empty()) {
        throw std::runtime_error("Control group '" + group.name + "' has no follower joints");
      }
      for (const int index : group.follower_joint_indices) {
        if (index < 0 || index >= dof || !assigned_joint_indices.insert(index).second) {
          throw std::runtime_error(
                  "Control-group follower joint indices must be valid and non-overlapping");
        }
      }
    }
    if (robot_teleoperation_->controlGroupStates().size() <= largest_group_id) {
      throw std::runtime_error("Robot profile does not provide state for every control group");
    }

    std::unordered_set<ControlGroupId> input_groups;
    for (const auto & channel : robot_teleoperation_->leaderInputChannels()) {
      if (
        group_ids.find(channel.group_id) == group_ids.end() || channel.topic.empty() ||
        !input_groups.insert(channel.group_id).second)
      {
        throw std::runtime_error(
                "Leader input channels must have a unique configured group and a topic");
      }
    }
    if (input_groups.size() != group_ids.size()) {
      throw std::runtime_error("Every control group must provide one leader input channel");
    }
    if (robot_teleoperation_->followerJointStatesTopic().empty()) {
      throw std::runtime_error("Robot profile follower joint-state topic must not be empty");
    }
  }

  void declareParameters()
  {
    declare_parameter("robot.plugin", "");
    declare_parameter("robot.parameter_prefix", "");
    declare_parameter("control_frequency", 100.0);
    declare_parameter("trajectory_time", 0.0);
    declare_parameter("joint_state_timeout", 0.5);
    declare_parameter("leader_command_timeout", 0.5);

    declare_parameter("hold.kp", 20.0);
    declare_parameter("hold.max_correction_velocity", 0.2);
    declare_parameter("hold.tracking_weight", 100.0);
    declare_parameter("constraints.slack_penalty", 1000.0);
    declare_parameter("constraints.cbf_alpha", 50.0);
    declare_parameter("constraints.collision_buffer", 0.05);
    declare_parameter("constraints.collision_safe_distance", 0.02);
    declare_parameter("constraints.damping_weight", 0.1);
    declare_parameter("pose_sequence.kp_joint", 30.0);
    declare_parameter("pose_sequence.tracking_weight", 10.0);

    declare_parameter<std::vector<int64_t>>(
      "available_control_modes", std::vector<int64_t>{});
    declare_parameter("default_control_mode", 1);
    const auto control_modes =
      get_parameter("available_control_modes").as_integer_array();
    if (control_modes.empty()) {
      throw std::runtime_error(
              "available_control_modes must contain at least one control mode ID");
    }
    for (const int64_t raw_mode : control_modes) {
      if (raw_mode <= 0 || raw_mode > UINT16_MAX) {
        throw std::runtime_error("Control mode IDs must be in the uint16 range");
      }
      const auto mode = static_cast<uint16_t>(raw_mode);
      if (mode_plugins_.find(mode) != mode_plugins_.end()) {
        throw std::runtime_error(
                "Duplicate control mode ID in available_control_modes: " +
                std::to_string(mode));
      }
      const std::string prefix = "control_modes." + std::to_string(mode);
      const std::string mode_name =
        declare_parameter(prefix + ".name", "mode_" + std::to_string(mode));
      const std::string plugin_name = declare_parameter(prefix + ".plugin", "");
      if (mode_name.empty()) {
        throw std::runtime_error(prefix + ".name must not be empty");
      }
      if (plugin_name.empty()) {
        throw std::runtime_error(prefix + ".plugin must be configured in the parameter YAML");
      }
      if (!mode_loader_.isClassAvailable(plugin_name)) {
        throw std::runtime_error(
                prefix + ".plugin is not registered with pluginlib: " + plugin_name);
      }
      mode_names_[mode] = mode_name;
      mode_plugins_[mode] = plugin_name;
      for (const auto & sequence_name : {std::string("initial_pose"), std::string("exit_pose")}) {
        const std::string sequence_prefix = prefix + "." + sequence_name;
        declare_parameter(sequence_prefix + ".enabled", false);
        declare_parameter<std::vector<std::string>>(
          sequence_prefix + ".step_names", std::vector<std::string>{});
        const auto step_names =
          get_parameter(sequence_prefix + ".step_names").as_string_array();
        for (const auto & step_name : step_names) {
          if (step_name.empty()) {
            throw std::runtime_error(sequence_prefix + ".step_names contains an empty name");
          }
        }
        declare_parameter(sequence_prefix + ".duration", 3.0);
        declare_parameter(sequence_prefix + ".completion_tolerance", 0.03);
        declare_parameter(sequence_prefix + ".timeout", 10.0);
      }
    }
    const int64_t raw_default_mode = get_parameter("default_control_mode").as_int();
    if (
      raw_default_mode <= 0 || raw_default_mode > UINT16_MAX ||
      mode_plugins_.find(static_cast<uint16_t>(raw_default_mode)) == mode_plugins_.end())
    {
      throw std::runtime_error(
              "default_control_mode must reference an ID in available_control_modes");
    }

    declare_parameter<std::vector<int64_t>>("available_presets", {1});
    const auto preset_ids = get_parameter("available_presets").as_integer_array();
    for (const int64_t raw_id : preset_ids) {
      if (raw_id <= 0 || raw_id > UINT16_MAX) {
        throw std::runtime_error("Preset IDs must be in the uint16 range");
      }
      const auto id = static_cast<uint16_t>(raw_id);
      const std::string prefix = "presets." + std::to_string(id);
      declare_parameter(prefix + ".name", "preset_" + std::to_string(id));
      declare_parameter<std::vector<std::string>>(
        prefix + ".step_names", std::vector<std::string>{"step0"});
      const auto preset_step_names =
        get_parameter(prefix + ".step_names").as_string_array();
      for (const auto & step_name : preset_step_names) {
        if (step_name.empty()) {
          throw std::runtime_error(prefix + ".step_names contains an empty name");
        }
      }
      declare_parameter(prefix + ".duration", 3.0);
      declare_parameter(prefix + ".completion_tolerance", 0.03);
      declare_parameter(prefix + ".timeout", 10.0);
    }
  }

  void declarePoseGroupParameters()
  {
    const auto & groups = robot_teleoperation_->modeConfiguration().control_groups;
    const auto control_modes = get_parameter("available_control_modes").as_integer_array();
    for (const int64_t raw_mode : control_modes) {
      const std::string mode_prefix = "control_modes." + std::to_string(raw_mode);
      for (const auto & sequence_name : {std::string("initial_pose"), std::string("exit_pose")}) {
        const std::string sequence_prefix = mode_prefix + "." + sequence_name;
        const auto step_names =
          get_parameter(sequence_prefix + ".step_names").as_string_array();
        for (const auto & step_name : step_names) {
          for (const auto & group : groups) {
            declare_parameter<std::vector<double>>(
              sequence_prefix + ".steps." + step_name + "." + group.name + ".positions",
              std::vector<double>{});
          }
        }
      }
    }

    const auto preset_ids = get_parameter("available_presets").as_integer_array();
    for (const int64_t raw_id : preset_ids) {
      const std::string prefix = "presets." + std::to_string(raw_id);
      const auto step_names = get_parameter(prefix + ".step_names").as_string_array();
      for (const auto & step_name : step_names) {
        for (const auto & group : groups) {
          declare_parameter<std::vector<double>>(
            prefix + ".steps." + step_name + "." + group.name + ".positions",
            std::vector<double>{});
        }
      }
    }
  }

  bool isModeAvailable(const uint16_t mode) const
  {
    const auto iter = mode_plugins_.find(mode);
    return iter != mode_plugins_.end() && !iter->second.empty();
  }

  bool areSelectedPresetsAvailable(
    const ControlGroupMask target_groups,
    const std::vector<uint16_t> & preset_ids) const
  {
    if (!pose_sequences_) {
      return false;
    }
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      if (!containsControlGroup(target_groups, group.id)) {
        continue;
      }
      if (
        group.id >= preset_ids.size() ||
        preset_ids[group.id] == 0 ||
        !pose_sequences_->hasPreset(preset_ids[group.id], group.id))
      {
        return false;
      }
    }
    return true;
  }

  void followerCallback(const sensor_msgs::msg::JointState::SharedPtr message)
  {
    if (!robot_teleoperation_->updateFollowerState(*message)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Follower state does not contain every joint required by robot profile '%s'",
        robot_teleoperation_->robotName().c_str());
      return;
    }
    last_follower_time_ = now();
    follower_received_ = true;
    if (!hold_initialized_) {
      hold_target_ = robot_teleoperation_->followerPosition();
      syncCommandToFeedback();
      hold_initialized_ = true;
      transition_pending_ = true;
    }
  }

  void commandCallback(const ControlRequest & request)
  {
    if (!isModeAvailable(request.control_mode)) {
      transition_id_ = request.transition_id;
      publishStatus(
        ControlStatus::kError,
        "Unknown or unavailable control mode: " +
        std::to_string(request.control_mode));
      return;
    }
    const ControlGroupMask preset_target = request.preset_target_groups;
    const ControlGroupMask initial_pose_target = request.initial_pose_target_groups;
    if (
      ((request.enabled_groups | preset_target | initial_pose_target) & ~allGroups()) != 0)
    {
      transition_id_ = request.transition_id;
      publishStatus(ControlStatus::kError, "Control command contains an unknown control group");
      return;
    }
    if (preset_target != 0 && initial_pose_target != 0) {
      transition_id_ = request.transition_id;
      publishStatus(
        ControlStatus::kError,
        "Preset and initial pose cannot be requested in the same command");
      return;
    }
    if (!areSelectedPresetsAvailable(preset_target, request.preset_ids)) {
      transition_id_ = request.transition_id;
      publishStatus(
        ControlStatus::kError,
        "Preset is not configured for a requested control group");
      return;
    }

    const ControlGroupMask moving_initial_pose_groups =
      pose_sequences_->movingFinalInitialPoseGroups();
    const bool changing_mode =
      mode_ready_ && request.control_mode != active_control_mode_;
    if (
      changing_mode &&
      (requested_groups_ != 0 || active_groups_ != 0 ||
      pose_sequences_->movingPresetGroups() != 0))
    {
      transition_id_ = request.transition_id;
      publishStatus(
        ControlStatus::kError,
        "Control mode can only be changed while every control group is stopped");
      return;
    }
    if (changing_mode && moving_initial_pose_groups != 0) {
      transition_id_ = request.transition_id;
      publishStatus(
        ControlStatus::kError,
        "Control mode cannot be changed while an initial pose movement is in progress");
      return;
    }
    if (initial_pose_target != 0) {
      const ControlGroupMask available_groups =
        mode_ready_ ? pose_sequences_->initialPoseGroups(active_control_mode_) : 0;
      if (
        !mode_ready_ || transition_pending_ || mode_transition_started_ ||
        request.control_mode != active_control_mode_ ||
        (initial_pose_target & available_groups) != initial_pose_target)
      {
        transition_id_ = request.transition_id;
        publishStatus(
          ControlStatus::kError,
          "Initial pose trigger ignored because it is disabled for the active control mode");
        return;
      }
      if (
        (initial_pose_target &
        (pose_sequences_->movingPresetGroups() | moving_initial_pose_groups)) != 0)
      {
        transition_id_ = request.transition_id;
        publishStatus(
          ControlStatus::kError,
          "Initial pose trigger ignored because another pose movement is in progress");
        return;
      }
    }
    if ((preset_target & moving_initial_pose_groups) != 0) {
      transition_id_ = request.transition_id;
      publishStatus(
        ControlStatus::kError,
        "Preset cannot be started while an initial pose movement is in progress");
      return;
    }

    const ControlGroupMask newly_enabled_groups =
      request.enabled_groups & ~requested_groups_;
    if ((newly_enabled_groups & moving_initial_pose_groups) != 0) {
      transition_id_ = request.transition_id;
      publishStatus(
        ControlStatus::kError,
        "Teleoperation cannot be enabled while an initial pose movement is in progress");
      return;
    }

    transition_id_ = request.transition_id;
    requested_control_mode_ = request.control_mode;
    if (
      mode_transition_started_ &&
      transition_target_mode_ != requested_control_mode_)
    {
      if (mode_transition_phase_ != ModeTransitionPhase::kExitPose) {
        pose_sequences_->cancelInitialPose();
        pose_sequences_->cancelExitPose();
        mode_transition_phase_ = ModeTransitionPhase::kIdle;
        mode_transition_started_ = false;
      } else {
        transition_target_mode_ = requested_control_mode_;
      }
    }
    requested_groups_ = request.enabled_groups;
    requested_groups_ &= ~(preset_target | initial_pose_target);
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      if (
        group.id < request.preset_ids.size() &&
        request.preset_ids[group.id] != 0)
      {
        selected_preset_ids_[group.id] = request.preset_ids[group.id];
      }
    }
    transition_pending_ =
      active_control_mode_ != requested_control_mode_ || !mode_ready_;
    groups_pending_ = !transition_pending_;
    if (preset_target != 0) {
      preset_update_pending_groups_ |= preset_target;
      preset_cancel_pending_groups_ &= ~preset_target;
    }
    if (initial_pose_target != 0) {
      final_initial_pose_update_pending_groups_ |= initial_pose_target;
      final_initial_pose_cancel_pending_groups_ &= ~initial_pose_target;
    }
    const ControlGroupMask resume_groups = newly_enabled_groups &
      pose_sequences_->activeFinalInitialPoseGroups();
    if (resume_groups != 0) {
      final_initial_pose_cancel_pending_groups_ |= resume_groups;
    }
  }

  bool feedbackFresh() const
  {
    return follower_received_ &&
           (now() - last_follower_time_).seconds() <=
           get_parameter("joint_state_timeout").as_double();
  }

  ControlGroupMask allGroups() const
  {
    ControlGroupMask groups = 0;
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      groups |= controlGroupBit(group.id);
    }
    return groups;
  }

  ControlGroupMask freshLeaderGroups() const
  {
    const double timeout = get_parameter("leader_command_timeout").as_double();
    ControlGroupMask result = 0;
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      if (
        group.id < leader_received_.size() && leader_received_[group.id] &&
        (now() - last_leader_times_[group.id]).seconds() <= timeout)
      {
        result |= controlGroupBit(group.id);
      }
    }
    return result;
  }

  ModeContext makeContext(const ControlGroupMask enabled_groups) const
  {
    context_group_states_ = robot_teleoperation_->controlGroupStates();
    if (context_group_states_.size() < selected_preset_ids_.size()) {
      context_group_states_.resize(selected_preset_ids_.size());
    }
    for (size_t i = 0; i < selected_preset_ids_.size(); ++i) {
      context_group_states_[i].selected_preset_id = selected_preset_ids_[i];
    }
    return ModeContext{
      command_position_,
      command_velocity_,
      robot_teleoperation_->followerPosition(),
      robot_teleoperation_->leaderReference(),
      robot_teleoperation_->leaderPosition(),
      context_group_states_,
      requested_groups_,
      enabled_groups,
      pose_sequences_ ?
      (pose_sequences_->activePresetGroups() |
      pose_sequences_->activeFinalInitialPoseGroups()) : 0,
      now().seconds(),
      1.0 / std::max(1.0, get_parameter("control_frequency").as_double())};
  }

  const std::vector<uint16_t> & selectedPresetIds() const
  {
    return selected_preset_ids_;
  }

  void syncGroupCommandToFeedback(const ControlGroupMask groups)
  {
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      if (!containsControlGroup(groups, group.id)) {
        continue;
      }
      for (const int index : group.follower_joint_indices) {
          command_position_[index] = robot_teleoperation_->followerPosition()[index];
          command_velocity_[index] = 0.0;
      }
    }
  }

  void captureGroupHoldTarget(const ControlGroupMask groups)
  {
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      if (!containsControlGroup(groups, group.id)) {
        continue;
      }
      for (const int index : group.follower_joint_indices) {
          hold_target_[index] = robot_teleoperation_->followerPosition()[index];
      }
    }
  }

  void updateControlledGroupOwnership(const ControlGroupMask controlled_groups)
  {
    const ControlGroupMask released_groups =
      previous_controlled_groups_ & ~controlled_groups;
    if (released_groups != 0) {
      captureGroupHoldTarget(released_groups);
      syncGroupCommandToFeedback(released_groups);
    }
    previous_controlled_groups_ = controlled_groups;
  }

  void syncCommandToFeedback()
  {
    command_position_ = robot_teleoperation_->followerPosition();
    command_velocity_ = Eigen::VectorXd::Zero(robot_teleoperation_->dof());
    command_initialized_ = true;
  }

  bool startRequestedInitialPose()
  {
    try {
      const ModeContext context = makeContext(0);
      pose_sequences_->cancelExitPose();
      if (pose_sequences_->hasInitialPose(transition_target_mode_)) {
        if (!pose_sequences_->startInitialPose(transition_target_mode_, context)) {
          throw std::runtime_error("initial pose transition was rejected");
        }
        const ControlGroupMask initial_pose_groups = pose_sequences_->activeInitialPoseGroups();
        pose_sequences_->cancelPresets(initial_pose_groups);
        paused_preset_groups_ &= ~initial_pose_groups;
        mode_transition_phase_ = ModeTransitionPhase::kInitialPose;
        publishStatus(
          ControlStatus::kLoading,
          "Moving to the initial pose for control mode " +
          std::to_string(transition_target_mode_));
      } else {
        pose_sequences_->cancelInitialPose();
        mode_transition_phase_ = ModeTransitionPhase::kActivate;
      }
      return true;
    } catch (const std::exception & error) {
      pose_sequences_->cancelInitialPose();
      pose_sequences_->cancelExitPose();
      mode_transition_phase_ = ModeTransitionPhase::kIdle;
      mode_transition_started_ = false;
      transition_pending_ = false;
      paused_preset_groups_ = 0;
      publishStatus(
        ControlStatus::kError,
        std::string("Failed to begin initial pose transition: ") + error.what());
      return false;
    }
  }

  bool beginRequestedModeTransition()
  {
    mode_transition_phase_ = ModeTransitionPhase::kIdle;
    transition_target_mode_ = requested_control_mode_;
    transition_source_mode_ = active_control_mode_;
    paused_preset_groups_ = pose_sequences_->activePresetGroups();
    pose_sequences_->cancelFinalInitialPoses(allGroups());
    final_initial_pose_update_pending_groups_ = 0;
    final_initial_pose_cancel_pending_groups_ = 0;
    publishStatus(
      ControlStatus::kLoading,
      "Loading control mode " + std::to_string(requested_control_mode_));
    hold_target_ = robot_teleoperation_->followerPosition();
    syncCommandToFeedback();
    previous_controlled_groups_ = 0;
    robot_teleoperation_->publish(hold_target_);
    active_groups_ = 0;
    if (mode_) {
      mode_->deactivate();
      mode_.reset();
    }
    mode_ready_ = false;
    active_control_mode_ = 0;

    try {
      const ModeContext context = makeContext(0);
      const bool changing_mode =
        transition_source_mode_ != 0 &&
        transition_source_mode_ != transition_target_mode_;
      if (changing_mode && pose_sequences_->hasExitPose(transition_source_mode_)) {
        if (!pose_sequences_->startExitPose(transition_source_mode_, context)) {
          throw std::runtime_error("exit pose transition was rejected");
        }
        const ControlGroupMask exit_pose_groups = pose_sequences_->activeExitPoseGroups();
        pose_sequences_->cancelPresets(exit_pose_groups);
        paused_preset_groups_ &= ~exit_pose_groups;
        mode_transition_phase_ = ModeTransitionPhase::kExitPose;
        publishStatus(
          ControlStatus::kLoading,
          "Moving to the exit pose for control mode " +
          std::to_string(transition_source_mode_));
      } else {
        pose_sequences_->cancelExitPose();
      }
      mode_transition_started_ = true;
      if (mode_transition_phase_ != ModeTransitionPhase::kExitPose) {
        return startRequestedInitialPose();
      }
      return true;
    } catch (const std::exception & error) {
      pose_sequences_->cancelInitialPose();
      pose_sequences_->cancelExitPose();
      mode_transition_phase_ = ModeTransitionPhase::kIdle;
      mode_transition_started_ = false;
      transition_pending_ = false;
      paused_preset_groups_ = 0;
      publishStatus(
        ControlStatus::kError,
        std::string("Failed to begin mode transition: ") + error.what());
      return false;
    }
  }

  bool activateRequestedMode()
  {
    publishStatus(
      ControlStatus::kActivating,
      "Activating control mode " + std::to_string(transition_target_mode_));
    try {
      mode_ =
        mode_loader_.createSharedInstance(mode_plugins_.at(transition_target_mode_));
      const std::string parameter_prefix =
        "control_modes." + std::to_string(transition_target_mode_);
      if (!mode_->configure(
          *this, parameter_prefix, robot_teleoperation_->modeConfiguration()))
      {
        throw std::runtime_error("mode configuration was rejected");
      }
      robot_teleoperation_->followerKinematics()->updateState(
        robot_teleoperation_->followerPosition(), robot_teleoperation_->followerVelocity());
      robot_teleoperation_->leaderKinematics()->updateState(
        robot_teleoperation_->leaderPosition(),
        Eigen::VectorXd::Zero(robot_teleoperation_->leaderPosition().size()));
      const ControlGroupMask initial_groups =
        requested_groups_ & freshLeaderGroups() &
        ~(pose_sequences_->movingPresetGroups() |
        pose_sequences_->movingFinalInitialPoseGroups());
      const ModeContext initial_context = makeContext(initial_groups);
      if (!mode_->activate(initial_context)) {
        throw std::runtime_error("mode activation was rejected");
      }
      if (
        paused_preset_groups_ != 0 &&
        !pose_sequences_->startPreset(
          paused_preset_groups_, selectedPresetIds(), initial_context))
      {
        throw std::runtime_error("preset overlay reactivation was rejected");
      }
      active_control_mode_ = transition_target_mode_;
      active_groups_ = initial_groups;
      mode_ready_ = true;
      transition_pending_ = false;
      mode_transition_phase_ = ModeTransitionPhase::kIdle;
      mode_transition_started_ = false;
      paused_preset_groups_ = 0;
      pose_sequences_->cancelInitialPose();
      pose_sequences_->cancelExitPose();
      groups_pending_ = requested_groups_ != active_groups_;
      const bool mode_has_output =
        (mode_->controlledGroups(initial_context) |
        pose_sequences_->activePresetGroups() |
        pose_sequences_->activeFinalInitialPoseGroups()) != 0;
      publishStatus(
        !mode_has_output ?
        ControlStatus::kHolding :
        ControlStatus::kActive,
        active_groups_ == requested_groups_ ?
        "Mode activated" : "Mode activated; waiting for fresh leader reference");
      return true;
    } catch (const std::exception & error) {
      mode_.reset();
      pose_sequences_->cancelInitialPose();
      pose_sequences_->cancelExitPose();
      mode_ready_ = false;
      active_control_mode_ = 0;
      active_groups_ = 0;
      previous_controlled_groups_ = 0;
      transition_pending_ = false;
      mode_transition_phase_ = ModeTransitionPhase::kIdle;
      mode_transition_started_ = false;
      paused_preset_groups_ = 0;
      publishStatus(
        ControlStatus::kError,
        std::string("Failed to load mode: ") + error.what());
      return false;
    }
  }

  void updateActiveGroups()
  {
    const ControlGroupMask desired =
      requested_groups_ & freshLeaderGroups() &
      ~(pose_sequences_->movingPresetGroups() |
      pose_sequences_->movingFinalInitialPoseGroups());
    if (desired == active_groups_) {
      groups_pending_ = requested_groups_ != desired;
      return;
    }

    const ControlGroupMask disabled = active_groups_ & ~desired;
    const ControlGroupMask enabled = desired & ~active_groups_;
    captureGroupHoldTarget(disabled);

    syncGroupCommandToFeedback(disabled | enabled);

    active_groups_ = desired;
    if (enabled != 0 && mode_) {
      pose_sequences_->cancelPresets(enabled);
      pose_sequences_->cancelFinalInitialPoses(enabled);
      mode_->onGroupsEnabled(enabled, makeContext(active_groups_));
    }
    groups_pending_ = requested_groups_ != active_groups_;
    publishStatus(
      active_groups_ == 0 ?
      ControlStatus::kHolding :
      ControlStatus::kActive,
      groups_pending_ ?
      ((pose_sequences_->movingPresetGroups() |
      pose_sequences_->movingFinalInitialPoseGroups()) != 0 ?
      "Waiting for pose movement to finish" :
      "Holding control group with stale leader reference") : "Control-group state updated");
  }

  bool updateModePoseTransition(const bool exit_pose)
  {
    const ModeContext context = makeContext(0);
    try {
      robot_teleoperation_->followerKinematics()->updateState(
        command_position_, command_velocity_);

      ModeOutput output;
      output.reset(
        robot_teleoperation_->dof(),
        get_parameter("constraints.damping_weight").as_double());
      const bool update_success = exit_pose ?
        pose_sequences_->updateExitPose(context, output) :
        pose_sequences_->updateInitialPose(context, output);
      if (!update_success) {
        throw std::runtime_error(pose_sequences_->errorMessage());
      }

      const ControlGroupMask controlled_groups = exit_pose ?
        pose_sequences_->activeExitPoseGroups() :
        pose_sequences_->activeInitialPoseGroups();
      applySoftHold(
        output, command_position_, hold_target_,
        robot_teleoperation_->modeConfiguration().control_groups,
        controlled_groups,
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
        robot_teleoperation_->publish(command_position_);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "%s pose QP failed; holding the last command and retrying",
          exit_pose ? "Exit" : "Initial");
        return true;
      }
      command_position_ += context.dt * optimal_velocity;
      command_velocity_ = optimal_velocity;
      robot_teleoperation_->publish(command_position_);
      const bool moving = exit_pose ?
        pose_sequences_->exitPoseMoving() :
        pose_sequences_->initialPoseMoving();
      if (!moving) {
        hold_target_ = command_position_;
        command_velocity_.setZero();
        publishStatus(
          ControlStatus::kLoading,
          exit_pose ?
          "Exit pose reached; preparing the requested control mode" :
          "Initial pose reached; control mode activation is now allowed");
      }
      return true;
    } catch (const std::exception & error) {
      hold_target_ = robot_teleoperation_->followerPosition();
      syncCommandToFeedback();
      pose_sequences_->cancelInitialPose();
      pose_sequences_->cancelExitPose();
      mode_transition_phase_ = ModeTransitionPhase::kIdle;
      mode_transition_started_ = false;
      transition_pending_ = false;
      paused_preset_groups_ = 0;
      publishStatus(
        ControlStatus::kError,
        std::string(exit_pose ? "Exit" : "Initial") +
        " pose transition failed: " + error.what());
      robot_teleoperation_->publish(hold_target_);
      return false;
    }
  }

  void controlLoop()
  {
    if (!feedbackFresh()) {
      if (follower_received_ && !feedback_error_reported_) {
        feedback_error_reported_ = true;
        hold_target_ = robot_teleoperation_->followerPosition();
        command_initialized_ = false;
        active_groups_ = 0;
        previous_controlled_groups_ = 0;
        publishStatus(
          ControlStatus::kError,
          "Follower feedback timed out; holding all control groups");
      }
      return;
    }
    feedback_error_reported_ = false;
    if (!command_initialized_) {
      syncCommandToFeedback();
    }

    if (transition_pending_ && hold_initialized_) {
      if (!mode_transition_started_ && !beginRequestedModeTransition()) {
        robot_teleoperation_->publish(hold_target_);
        return;
      }
      if (mode_transition_phase_ == ModeTransitionPhase::kExitPose) {
        if (pose_sequences_->exitPoseMoving()) {
          updateModePoseTransition(true);
          return;
        }
        if (!startRequestedInitialPose()) {
          robot_teleoperation_->publish(hold_target_);
          return;
        }
      }
      if (
        mode_transition_phase_ == ModeTransitionPhase::kInitialPose &&
        pose_sequences_->initialPoseMoving())
      {
        updateModePoseTransition(false);
        return;
      }
      mode_transition_phase_ = ModeTransitionPhase::kActivate;
      if (!activateRequestedMode()) {
        robot_teleoperation_->publish(hold_target_);
        return;
      }
    }
    if (!mode_ready_ || !mode_) {
      if (hold_initialized_) {
        robot_teleoperation_->publish(hold_target_);
      }
      return;
    }

    if (preset_cancel_pending_groups_ != 0) {
      const ControlGroupMask cancel_groups = preset_cancel_pending_groups_;
      preset_cancel_pending_groups_ = 0;
      captureGroupHoldTarget(cancel_groups);
      syncGroupCommandToFeedback(cancel_groups);
      pose_sequences_->cancelPresets(cancel_groups);
    }

    if (final_initial_pose_cancel_pending_groups_ != 0) {
      const ControlGroupMask cancel_groups = final_initial_pose_cancel_pending_groups_;
      final_initial_pose_cancel_pending_groups_ = 0;
      captureGroupHoldTarget(cancel_groups);
      syncGroupCommandToFeedback(cancel_groups);
      pose_sequences_->cancelFinalInitialPoses(cancel_groups);
    }

    if (
      groups_pending_ ||
      (requested_groups_ & freshLeaderGroups()) != active_groups_)
    {
      updateActiveGroups();
    }

    if (final_initial_pose_update_pending_groups_ != 0) {
      const ControlGroupMask update_groups = final_initial_pose_update_pending_groups_;
      final_initial_pose_update_pending_groups_ = 0;
      syncGroupCommandToFeedback(update_groups);
      pose_sequences_->cancelPresets(update_groups);
      if (!pose_sequences_->startFinalInitialPose(
          active_control_mode_, update_groups, makeContext(active_groups_)))
      {
        publishStatus(
          ControlStatus::kError,
          pose_sequences_->errorMessage());
        robot_teleoperation_->publish(command_position_);
        return;
      }
      publishStatus(
        ControlStatus::kActive,
        "Moving selected control group to the final step of the active mode initial pose");
    }

    if (preset_update_pending_groups_ != 0) {
      const ControlGroupMask update_groups = preset_update_pending_groups_;
      preset_update_pending_groups_ = 0;
      syncGroupCommandToFeedback(update_groups);
      if (!pose_sequences_->startPreset(
          update_groups, selectedPresetIds(), makeContext(active_groups_)))
      {
        publishStatus(
          ControlStatus::kError,
          "Preset update was rejected by the active mode");
        robot_teleoperation_->publish(command_position_);
        return;
      }
    }

    const ControlGroupMask timed_command_sync_groups =
      mode_->timedCommandFeedbackSyncGroups(makeContext(active_groups_));
    if (timed_command_sync_groups != 0) {
      syncGroupCommandToFeedback(timed_command_sync_groups);
    }
    const ModeContext context = makeContext(active_groups_);
    const ControlGroupMask controlled_groups =
      mode_->controlledGroups(context) |
      pose_sequences_->activePresetGroups() |
      pose_sequences_->activeFinalInitialPoseGroups();
    updateControlledGroupOwnership(controlled_groups);
    if (controlled_groups == 0) {
      command_position_ = hold_target_;
      command_velocity_.setZero();
      robot_teleoperation_->publish(hold_target_);
      return;
    }

    try {
      robot_teleoperation_->followerKinematics()->updateState(
        command_position_, command_velocity_);
      robot_teleoperation_->leaderKinematics()->updateState(
        robot_teleoperation_->leaderPosition(),
        Eigen::VectorXd::Zero(robot_teleoperation_->leaderPosition().size()));

      ModeOutput output;
      output.reset(
        robot_teleoperation_->dof(),
        get_parameter("constraints.damping_weight").as_double());
      if (!mode_->update(context, output)) {
        throw std::runtime_error("active mode rejected update");
      }
      if (!pose_sequences_->updatePresets(context, output)) {
        throw std::runtime_error(pose_sequences_->errorMessage());
      }
      if (!pose_sequences_->updateFinalInitialPoses(context, output)) {
        throw std::runtime_error(pose_sequences_->errorMessage());
      }

      const double hold_kp = get_parameter("hold.kp").as_double();
      const double max_hold_velocity =
        get_parameter("hold.max_correction_velocity").as_double();
      const double hold_weight = get_parameter("hold.tracking_weight").as_double();
      applySoftHold(
        output, command_position_, hold_target_,
        robot_teleoperation_->modeConfiguration().control_groups,
        controlled_groups, hold_kp, max_hold_velocity, hold_weight);

      qp_->setModeOutput(output);
      qp_->setControllerParameters(
        get_parameter("constraints.slack_penalty").as_double(),
        get_parameter("constraints.cbf_alpha").as_double(),
        get_parameter("constraints.collision_buffer").as_double(),
        get_parameter("constraints.collision_safe_distance").as_double());
      Eigen::VectorXd optimal_velocity;
      if (!qp_->solve(optimal_velocity)) {
        command_velocity_.setZero();
        robot_teleoperation_->publish(command_position_);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Teleoperation QP failed; holding the last command and retrying");
        return;
      }

      command_position_ += context.dt * optimal_velocity;
      command_velocity_ = optimal_velocity;
      robot_teleoperation_->publish(command_position_);

      std::vector<uint8_t> preset_states(last_preset_states_.size(), 0);
      std::vector<uint8_t> initial_pose_states(last_initial_pose_states_.size(), 0);
      for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
        preset_states[group.id] = pose_sequences_->presetState(group.id);
        initial_pose_states[group.id] =
          pose_sequences_->finalInitialPoseState(group.id);
      }
      if (
        preset_states != last_preset_states_ ||
        initial_pose_states != last_initial_pose_states_)
      {
        last_preset_states_ = std::move(preset_states);
        last_initial_pose_states_ = std::move(initial_pose_states);
        publishStatus(
          ControlStatus::kActive,
          "Pose sequence state updated");
      }
    } catch (const std::exception & error) {
      hold_target_ = robot_teleoperation_->followerPosition();
      syncCommandToFeedback();
      active_groups_ = 0;
      requested_groups_ = 0;
      previous_controlled_groups_ = 0;
      pose_sequences_->cancelPresets(allGroups());
      pose_sequences_->cancelFinalInitialPoses(allGroups());
      preset_update_pending_groups_ = 0;
      preset_cancel_pending_groups_ = 0;
      final_initial_pose_update_pending_groups_ = 0;
      final_initial_pose_cancel_pending_groups_ = 0;
      publishStatus(
        ControlStatus::kError,
        std::string("Control update failed; holding all control groups: ") + error.what());
      robot_teleoperation_->publish(hold_target_);
    }
  }

  void publishStatus(const uint8_t state, const std::string & message)
  {
    if (!robot_teleoperation_) {
      return;
    }
    ControlStatus status;
    status.transition_id = transition_id_;
    status.requested_control_mode = requested_control_mode_;
    status.active_control_mode = active_control_mode_;
    status.requested_groups = requested_groups_;
    status.active_groups = active_groups_;
    status.preset_ids = selected_preset_ids_;
    status.preset_states.assign(selected_preset_ids_.size(), 0);
    status.initial_pose_states.assign(selected_preset_ids_.size(), 0);
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      if (group.id >= status.preset_states.size()) {
        continue;
      }
      status.preset_states[group.id] =
        pose_sequences_ ? pose_sequences_->presetState(group.id) : 0;
      status.initial_pose_states[group.id] =
        pose_sequences_ ? pose_sequences_->finalInitialPoseState(group.id) : 0;
    }
    status.initial_pose_available_groups =
      pose_sequences_ && mode_ready_ ?
      pose_sequences_->initialPoseGroups(active_control_mode_) : 0;
    status.state = state;
    status.message = message;
    robot_teleoperation_->publishStatus(status);
  }

  pluginlib::ClassLoader<RobotTeleoperation> robot_loader_;
  pluginlib::ClassLoader<TeleoperationMode> mode_loader_;
  std::shared_ptr<RobotTeleoperation> robot_teleoperation_;
  std::shared_ptr<TeleoperationMode> mode_;
  std::unique_ptr<PoseSequenceManager> pose_sequences_;
  std::unique_ptr<TeleoperationQP> qp_;

  std::unordered_map<uint16_t, std::string> mode_names_;
  std::unordered_map<uint16_t, std::string> mode_plugins_;
  uint16_t requested_control_mode_ = 1;
  uint16_t active_control_mode_ = 0;
  uint16_t transition_target_mode_ = 0;
  uint16_t transition_source_mode_ = 0;
  std::vector<uint16_t> selected_preset_ids_;
  mutable std::vector<ControlGroupState> context_group_states_;
  ControlGroupMask requested_groups_ = 0;
  ControlGroupMask active_groups_ = 0;
  ControlGroupMask previous_controlled_groups_ = 0;
  uint64_t transition_id_ = 0;
  bool transition_pending_ = false;
  bool mode_transition_started_ = false;
  ModeTransitionPhase mode_transition_phase_ = ModeTransitionPhase::kIdle;
  bool groups_pending_ = false;
  ControlGroupMask preset_update_pending_groups_ = 0;
  ControlGroupMask preset_cancel_pending_groups_ = 0;
  ControlGroupMask final_initial_pose_update_pending_groups_ = 0;
  ControlGroupMask final_initial_pose_cancel_pending_groups_ = 0;
  ControlGroupMask paused_preset_groups_ = 0;
  bool mode_ready_ = false;
  bool follower_received_ = false;
  bool hold_initialized_ = false;
  std::vector<bool> leader_received_;
  bool feedback_error_reported_ = false;
  bool command_initialized_ = false;
  std::vector<uint8_t> last_preset_states_;
  std::vector<uint8_t> last_initial_pose_states_;
  Eigen::VectorXd hold_target_;
  Eigen::VectorXd command_position_;
  Eigen::VectorXd command_velocity_;
  rclcpp::Time last_follower_time_{0, 0, RCL_ROS_TIME};
  std::vector<rclcpp::Time> last_leader_times_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr follower_subscription_;
  std::vector<rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr>
  leader_subscriptions_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
    parameter_callback_handle_;
};

std::shared_ptr<rclcpp::Node> makeTeleoperationNode()
{
  return std::make_shared<TeleoperationNode>();
}
}  // namespace cyclo_teleoperation
