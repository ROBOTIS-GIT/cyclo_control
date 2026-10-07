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
#include <cmath>
#include <cstdint>
#include <limits>
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
#include "cyclo_teleoperation/core/action_input.hpp"
#include "cyclo_teleoperation/core/control_runtime.hpp"
#include "cyclo_teleoperation/core/mode_registry.hpp"
#include "cyclo_teleoperation/core/command_source_manager.hpp"

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
    // Claim ownership before creating robot publishers or mode services.
    source_manager_ = std::make_unique<CommandSourceManager>(*this,
        [this](ControlSource source) {beginSourceSwitch(source);},
        [this](ControlSource) {validateSourceSwitch();});
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
    initializeAuxiliaryCommands();
    declarePoseGroupParameters();
    pose_sequences_ = std::make_unique<PoseSequenceManager>();
    if (!pose_sequences_->configure(
        *this, robot_teleoperation_->modeConfiguration(),
        get_parameter("available_control_modes").as_integer_array(),
        get_parameter("available_presets").as_integer_array()))
    {
      throw std::runtime_error("Failed to configure pose sequences");
    }

    runtime_ = std::make_unique<ControlRuntime>(*this, robot_teleoperation_->followerKinematics());

    const auto follower_qos = rclcpp::SensorDataQoS().keep_last(1);
    follower_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      robot_teleoperation_->followerJointStatesTopic(), follower_qos,
      std::bind(&TeleoperationNode::followerCallback, this, std::placeholders::_1));
    size_t group_state_count = 0;
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      group_state_count = std::max(group_state_count, static_cast<size_t>(group.id) + 1);
    }
    leader_received_.assign(group_state_count, false);
    last_leader_times_.assign(group_state_count, rclcpp::Time(0, 0, RCL_ROS_TIME));
    selected_preset_ids_.assign(group_state_count, 1);
    context_group_states_.assign(group_state_count, ControlGroupState{});
    cartesian_references_.assign(group_state_count, CartesianReference{});
    last_preset_states_.assign(group_state_count, 0);
    last_initial_pose_states_.assign(group_state_count, 0);
    const auto latest_command_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    for (const auto & channel : robot_teleoperation_->leaderInputChannels()) {
      leader_subscriptions_.push_back(
        create_subscription<trajectory_msgs::msg::JointTrajectory>(
          channel.topic, latest_command_qos,
          [this, group = channel.group_id](
            const trajectory_msgs::msg::JointTrajectory::SharedPtr message)
          {
            if (!leader_action_enabled_) {return;}
            if (robot_teleoperation_->updateLeaderReference(*message, group)) {
              leader_received_.at(group) = true;
              last_leader_times_.at(group) = now();
            }
          }));
    }

    requested_control_mode_ =
      static_cast<uint16_t>(get_parameter("default_control_mode").as_int());
    teleop_requested_mode_ = requested_control_mode_;
    {
      action_modes_.configure(*this, mode_loader_, "available_action_modes", "action_modes",
        "default_action_mode", true);
      action_requested_mode_ = action_modes_.defaultMode();
      action_input_ = std::make_unique<ActionInput>(*this, *robot_teleoperation_,
          [this]() {return source_ == ControlSource::kAction && mode_ready_;},
          [this]() {return action_modes_.at(action_requested_mode_).reference_type;});
    }
    robot_teleoperation_->setModeRequestCallback(
      [this](uint16_t mode, uint64_t & id, std::string & message) {
        return requestMode(mode, id, message);
      });
    parameter_callback_handle_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto & parameter : parameters) {
          const std::string active_prefix =
          (source_ == ControlSource::kAction ? "action_modes." : "control_modes.") +
          std::to_string(active_control_mode_) + ".";
          if (
            active_control_mode_ != 0 &&
            parameter.get_name().rfind(active_prefix, 0) == 0)
          {
            if (source_ == ControlSource::kAction) {
              mode_ready_ = false;
              action_input_->clear();
            } else {
              transition_pending_ = true;
            }
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
    const auto initial = get_parameter("initial_source").as_string();
    if (initial != "teleop" && initial != "action") {
      throw std::runtime_error("initial_source must be teleop or action");
    }
    source_manager_->select(initial == "teleop" ?
      ControlSource::kTeleoperation : ControlSource::kAction);
  }

private:
  void validateSourceSwitch() const
  {
    if (source_ == ControlSource::kTeleoperation && mode_ready_ &&
      pose_sequences_->hasExitPose(active_control_mode_) && !feedbackFresh())
    {
      throw std::runtime_error("Fresh follower feedback is required for the exit pose");
    }
    if (mode_transition_started_ || transition_pending_ ||
      preset_update_pending_groups_ != 0 || final_initial_pose_update_pending_groups_ != 0 ||
      pose_sequences_->movingPresetGroups() != 0 ||
      pose_sequences_->movingFinalInitialPoseGroups() != 0)
    {
      throw std::runtime_error("Wait for the current mode/pose transition to finish");
    }
  }

  void failSourceSwitch(const std::string & reason)
  {
    RCLCPP_ERROR(get_logger(), "Source transition failed: %s", reason.c_str());
    setLeaderActionEnabled(false);
    source_ = ControlSource::kNone;
    source_exit_pending_ = false;
    transition_pending_ = false;
    source_manager_->complete(ControlSource::kNone);
  }

  void beginSourceSwitch(const ControlSource target)
  {
    source_target_ = target;
    try {
      if (source_ == ControlSource::kTeleoperation && mode_ready_ &&
        pose_sequences_->hasExitPose(active_control_mode_))
      {
        const auto departing_mode = active_control_mode_;
        setLeaderActionEnabled(false);
        if (mode_) {mode_->deactivate(); mode_.reset();}
        mode_ready_ = false;
        if (!pose_sequences_->startExitPose(departing_mode, makeContext(0))) {
          throw std::runtime_error("Cannot start source exit pose");
        }
        source_exit_pending_ = true;
      } else {
        switchSource(target);
      }
    } catch (const std::exception & error) {
      failSourceSwitch(error.what());
      throw;
    }
  }

  void switchSource(const ControlSource source)
  {
    // This method and both input callbacks run in the same callback group. Closing the
    // output before resetting references gives a single, atomic publication boundary.
    source_ = ControlSource::kNone;
    if (mode_) {mode_->deactivate(); mode_.reset();}
    mode_ready_ = false;
    active_control_mode_ = 0;
    const bool teleop = source == ControlSource::kTeleoperation;
    leader_action_enabled_ = !teleop;
    setLeaderActionEnabled(teleop);
    std::fill(leader_received_.begin(), leader_received_.end(), false);
    if (action_input_) {action_input_->clear();}
    if (!robot_teleoperation_->selectControlSource(teleop ? "teleop" : "action")) {
      throw std::runtime_error("Robot model rejected source selection");
    }
    runtime_ = std::make_unique<ControlRuntime>(*this, robot_teleoperation_->followerKinematics());
    if (!pose_sequences_->configure(*this, robot_teleoperation_->modeConfiguration(),
        get_parameter("available_control_modes").as_integer_array(),
        get_parameter("available_presets").as_integer_array()))
    {
      throw std::runtime_error("Failed to reconfigure pose sequences");
    }
    requested_control_mode_ = teleop ? teleop_requested_mode_ : action_requested_mode_;
    transition_pending_ = teleop;
    source_ = source;
    publishStatus(ControlStatus::kHolding, "Source selected; all groups remain stopped");
  }

  bool requestMode(const uint16_t mode, uint64_t & id, std::string & message)
  {
    id = 0;
    if (source_manager_->transitioning()) {
      message = "A source transition is in progress";
      return false;
    }
    const auto & registry = source_ == ControlSource::kAction ? action_modes_ : teleop_modes_;
    if (source_ == ControlSource::kNone || !registry.contains(mode)) {
      message = "Unknown mode for the selected source";
      return false;
    }
    if (active_groups_ != 0 || requested_groups_ != 0 || mode_transition_started_ ||
      preset_update_pending_groups_ != 0 || final_initial_pose_update_pending_groups_ != 0 ||
      pose_sequences_->movingPresetGroups() != 0 ||
      pose_sequences_->movingFinalInitialPoseGroups() != 0)
    {
      message = "All groups must be stopped and no pose movement may be in progress";
      return false;
    }
    if (source_ == ControlSource::kAction) {
      action_requested_mode_ = mode;
      if (mode_) {mode_->deactivate(); mode_.reset();}
      mode_ready_ = false;
      action_input_->clear();
    } else {
      teleop_requested_mode_ = mode;
      transition_pending_ = true;
    }
    requested_control_mode_ = mode;
    id = transition_id_ = ++next_transition_id_;
    message = "Mode request accepted";
    publishStatus(ControlStatus::kHolding, message);
    return true;
  }

  void updateAction()
  {
    try {
      if (!mode_ready_) {
        const auto & entry = action_modes_.at(action_requested_mode_);
        if (mode_) {mode_->deactivate(); mode_.reset();}
        mode_ = mode_loader_.createSharedInstance(entry.plugin);
        if (!mode_->configure(*this, entry.parameter_prefix,
            robot_teleoperation_->modeConfiguration()) || !mode_->activate(makeContext(0)))
        {
          throw std::runtime_error("Action mode rejected configuration or activation");
        }
        active_control_mode_ = action_requested_mode_;
        active_groups_ = 0;
        action_input_->clear();
        mode_ready_ = true;
        transition_pending_ = false;
        publishStatus(ControlStatus::kHolding, "Action mode activated");
      }
      const auto desired = action_input_->freshGroups();
      const auto changed = desired ^ active_groups_;
      const auto enabled = desired & ~active_groups_;
      if (changed != 0) {
        captureGroupHoldTarget(changed);
        syncGroupCommandToFeedback(changed);
        active_groups_ = desired;
        if (enabled != 0) {mode_->onGroupsEnabled(enabled, makeContext(active_groups_));}
      }
      const auto context = makeContext(active_groups_);
      const auto owned = mode_->controlledGroups(context);
      updateControlledGroupOwnership(owned);
      if (owned == 0) {
        command_position_ = hold_target_;
        command_velocity_.setZero();
        publishCommand(command_position_);
        return;
      }
      robot_teleoperation_->followerKinematics()->updateState(command_position_, command_velocity_);
      ModeOutput output;
      output.reset(robot_teleoperation_->dof(),
        get_parameter("constraints.damping_weight").as_double());
      if (!mode_->update(context, output)) {
        publishCommand(command_position_);
        return;
      }
      applySoftHold(output, command_position_, hold_target_,
        robot_teleoperation_->modeConfiguration().control_groups, owned,
        get_parameter("hold.kp").as_double(),
        get_parameter("hold.max_correction_velocity").as_double(),
        get_parameter("hold.tracking_weight").as_double());
      if (!runtime_->step(output, context.dt, command_position_, command_velocity_)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
          "Action QP failed; holding the last command and retrying");
      }
      publishCommand(command_position_, &output);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "%s", error.what());
      if (mode_) {mode_->deactivate(); mode_.reset();}
      mode_ready_ = false;
      action_input_->clear();
      active_groups_ = 0;
      previous_controlled_groups_ = 0;
      hold_target_ = robot_teleoperation_->followerPosition();
      syncCommandToFeedback();
      publishCommand(command_position_);
    }
  }

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
    if (dof <= 0 || !configuration.follower_kinematics ||
      !configuration.leader_kinematics)
    {
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
    std::unordered_set<std::string> auxiliary_joint_names;
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
      for (const auto & auxiliary : group.auxiliary_joints) {
        if (
          auxiliary.joint_name.empty() || auxiliary.pose_parameter_name.empty() ||
          !auxiliary_joint_names.insert(auxiliary.joint_name).second)
        {
          throw std::runtime_error(
                  "Control-group auxiliary joints must have unique joint names and pose fields");
        }
      }
    }
    if (robot_teleoperation_->controlGroupStates().size() <= largest_group_id) {
      throw std::runtime_error("Robot profile does not provide state for every control group");
    }
    const auto & follower_auxiliary = robot_teleoperation_->followerAuxiliaryPosition();
    const auto & leader_auxiliary = robot_teleoperation_->leaderAuxiliaryReference();
    if (
      follower_auxiliary.size() <= largest_group_id ||
      leader_auxiliary.size() <= largest_group_id)
    {
      throw std::runtime_error(
              "Robot profile does not provide auxiliary state for every control group");
    }
    for (const auto & group : configuration.control_groups) {
      const auto expected = static_cast<Eigen::Index>(group.auxiliary_joints.size());
      if (
        follower_auxiliary[group.id].size() != expected ||
        leader_auxiliary[group.id].size() != expected)
      {
        throw std::runtime_error(
                "Robot profile auxiliary state does not match control group '" +
                group.name + "'");
      }
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
    declare_parameter("initial_source", "action");
    declare_parameter("control_frequency", 100.0);
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
    teleop_modes_.configure(*this, mode_loader_, "available_control_modes", "control_modes",
      "default_control_mode", false);
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
      const std::string plugin_name = teleop_modes_.at(mode).plugin;
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
            for (const auto & auxiliary : group.auxiliary_joints) {
              declare_parameter<double>(
                sequence_prefix + ".steps." + step_name + "." + group.name + "." +
                auxiliary.pose_parameter_name,
                std::numeric_limits<double>::quiet_NaN());
            }
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
          for (const auto & auxiliary : group.auxiliary_joints) {
            declare_parameter<double>(
              prefix + ".steps." + step_name + "." + group.name + "." +
              auxiliary.pose_parameter_name,
              std::numeric_limits<double>::quiet_NaN());
          }
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
    const bool recovering = !command_initialized_ || !feedbackFresh();
    if (follower_received_ && !feedbackFresh()) {
      command_initialized_ = false;
      active_groups_ = 0;
      previous_controlled_groups_ = 0;
    }
    if (recovering && source_ == ControlSource::kAction && action_input_) {
      // Do not replay model commands received while follower feedback was unavailable.
      action_input_->clear();
    }
    last_follower_time_ = now();
    follower_received_ = true;
    if (source_ == ControlSource::kNone) {
      hold_target_ = robot_teleoperation_->followerPosition();
      syncCommandToFeedback();
      hold_initialized_ = true;
      return;
    }
    if (source_manager_ && source_manager_->ready()) {
      robot_teleoperation_->publishFollowerEefState(message->header);
    }
    if (!hold_initialized_) {
      hold_target_ = robot_teleoperation_->followerPosition();
      syncCommandToFeedback();
      hold_initialized_ = true;
      transition_pending_ = true;
    }
  }

  void setLeaderActionEnabled(const bool enabled)
  {
    if (enabled == leader_action_enabled_) {
      return;
    }

    leader_action_enabled_ = enabled;
    requested_groups_ = 0;
    active_groups_ = 0;
    previous_controlled_groups_ = 0;
    groups_pending_ = false;
    preset_update_pending_groups_ = 0;
    preset_cancel_pending_groups_ = 0;
    final_initial_pose_update_pending_groups_ = 0;
    final_initial_pose_cancel_pending_groups_ = 0;
    paused_preset_groups_ = 0;
    if (pose_sequences_) {
      pose_sequences_->cancelPresets(allGroups());
      pose_sequences_->cancelFinalInitialPoses(allGroups());
      pose_sequences_->cancelInitialPose();
      pose_sequences_->cancelExitPose();
    }
    mode_transition_phase_ = ModeTransitionPhase::kIdle;
    mode_transition_started_ = false;
    if (feedbackFresh()) {
      hold_target_ = robot_teleoperation_->followerPosition();
      syncCommandToFeedback();
      hold_initialized_ = true;
    } else {
      command_initialized_ = false;
      hold_initialized_ = false;
    }
    transition_pending_ = enabled &&
      (!mode_ready_ || active_control_mode_ != requested_control_mode_);
    publishStatus(
      ControlStatus::kHolding,
      enabled ?
      "Leader action output enabled; all control groups remain stopped" :
      "Leader action output disabled; model control may take ownership");
  }

  void commandCallback(const ControlRequest & request)
  {
    if (source_manager_->transitioning()) {return;}
    next_transition_id_ = std::max(next_transition_id_, request.transition_id);
    if (source_ != ControlSource::kTeleoperation) {
      // A stopped teleop configuration can be selected while action input owns the robot.
      // It must never modify the active action mode or active groups.
      if (request.enabled_groups == 0 && request.preset_target_groups == 0 &&
        request.initial_pose_target_groups == 0 && teleop_modes_.contains(request.control_mode))
      {
        teleop_requested_mode_ = request.control_mode;
      }
      return;
    }
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
      !leader_action_enabled_ &&
      (request.enabled_groups != 0 || preset_target != 0 || initial_pose_target != 0))
    {
      transition_id_ = request.transition_id;
      publishStatus(
        ControlStatus::kHolding,
        "Leader control request ignored while model control owns the follower command");
      return;
    }
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
    teleop_requested_mode_ = request.control_mode;
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

  void initializeAuxiliaryCommands()
  {
    auxiliary_command_ = robot_teleoperation_->followerAuxiliaryPosition();
    auxiliary_hold_target_ = auxiliary_command_;
  }

  void publishCommand(const Eigen::VectorXd & command, const ModeOutput * output = nullptr)
  {
    if (source_ == ControlSource::kNone) {
      return;
    }
    const auto & leader_auxiliary = robot_teleoperation_->leaderAuxiliaryReference();
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      const bool teleop_active = source_ == ControlSource::kTeleoperation &&
        containsControlGroup(active_groups_, group.id);
      if (source_ == ControlSource::kAction && action_input_->hasGripper(group.id)) {
        auxiliary_command_[group.id] = leader_auxiliary[group.id];
        auxiliary_hold_target_[group.id] = auxiliary_command_[group.id];
      } else if (teleop_active) {
        auxiliary_command_[group.id] = leader_auxiliary[group.id];
      } else {
        auxiliary_command_[group.id] = auxiliary_hold_target_[group.id];
      }
    }
    if (output != nullptr) {
      for (const auto & target : output->auxiliary_position_targets) {
        if (target.first >= auxiliary_command_.size()) {
          continue;
        }
        auto & command_target = auxiliary_command_[target.first];
        if (command_target.size() != target.second.size()) {
          continue;
        }
        for (Eigen::Index i = 0; i < target.second.size(); ++i) {
          if (std::isfinite(target.second[i])) {
            command_target[i] = target.second[i];
            auxiliary_hold_target_[target.first][i] = target.second[i];
          }
        }
      }
    }
    robot_teleoperation_->publish(command, auxiliary_command_);
    static const std::vector<EefPoseReference> empty_references;
    robot_teleoperation_->publishEefPoseReferences(
      command, output == nullptr ? empty_references : output->eef_pose_references);
  }

  void captureAuxiliaryCommandAsHold(const ControlGroupMask groups)
  {
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      if (containsControlGroup(groups, group.id)) {
        auxiliary_hold_target_[group.id] = auxiliary_command_[group.id];
      }
    }
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
      source_ ==
      ControlSource::kAction ? action_input_->cartesianReferences() : cartesian_references_,
      robot_teleoperation_->followerAuxiliaryPosition(),
      context_group_states_,
      requested_groups_,
      enabled_groups,
      pose_sequences_ ?
      (pose_sequences_->activePresetGroups() |
      pose_sequences_->activeFinalInitialPoseGroups()) : 0,
      now().seconds(),
      1.0 / std::max(1.0, get_parameter("control_frequency").as_double()),
      source_ == ControlSource::kAction};
  }

  const std::vector<uint16_t> & selectedPresetIds() const
  {
    return selected_preset_ids_;
  }

  void syncGroupCommandToFeedback(const ControlGroupMask groups)
  {
    const auto & follower_auxiliary = robot_teleoperation_->followerAuxiliaryPosition();
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      if (!containsControlGroup(groups, group.id)) {
        continue;
      }
      for (const int index : group.follower_joint_indices) {
        command_position_[index] = robot_teleoperation_->followerPosition()[index];
        command_velocity_[index] = 0.0;
      }
      auxiliary_command_[group.id] = follower_auxiliary[group.id];
    }
  }

  void captureGroupHoldTarget(const ControlGroupMask groups)
  {
    const auto & follower_auxiliary = robot_teleoperation_->followerAuxiliaryPosition();
    for (const auto & group : robot_teleoperation_->modeConfiguration().control_groups) {
      if (!containsControlGroup(groups, group.id)) {
        continue;
      }
      for (const int index : group.follower_joint_indices) {
        hold_target_[index] = robot_teleoperation_->followerPosition()[index];
      }
      auxiliary_hold_target_[group.id] = follower_auxiliary[group.id];
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
    initializeAuxiliaryCommands();
    command_initialized_ = true;
  }

  bool startRequestedInitialPose()
  {
    try {
      const ModeContext context = makeContext(0);
      pose_sequences_->cancelExitPose();
      if (pose_sequences_->automaticInitialPoseEnabled(transition_target_mode_)) {
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
    active_groups_ = 0;
    publishCommand(hold_target_);
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

      if (!runtime_->step(output, context.dt, command_position_, command_velocity_)) {
        command_velocity_.setZero();
        publishCommand(command_position_, &output);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "%s pose QP failed; holding the last command and retrying",
          exit_pose ? "Exit" : "Initial");
        return true;
      }
      publishCommand(command_position_, &output);
      const bool moving = exit_pose ?
        pose_sequences_->exitPoseMoving() :
        pose_sequences_->initialPoseMoving();
      if (!moving) {
        hold_target_ = command_position_;
        captureAuxiliaryCommandAsHold(controlled_groups);
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
      publishCommand(hold_target_);
      return false;
    }
  }

  void controlLoop()
  {
    if (!source_manager_->ready()) {return;}
    if (!feedbackFresh()) {
      if (follower_received_ && !feedback_error_reported_) {
        feedback_error_reported_ = true;
        hold_target_ = robot_teleoperation_->followerPosition();
        command_initialized_ = false;
        active_groups_ = 0;
        previous_controlled_groups_ = 0;
        if (action_input_) {action_input_->clear();}
        publishStatus(
          ControlStatus::kError,
          "Follower feedback timed out; holding all control groups");
      }
      return;
    }
    feedback_error_reported_ = false;
    if (!command_initialized_) {
      hold_target_ = robot_teleoperation_->followerPosition();
      syncCommandToFeedback();
      pose_sequences_->rebaseActiveSequences(makeContext(0));
    }
    if (source_exit_pending_) {
      if (!updateModePoseTransition(true)) {
        failSourceSwitch("Exit pose could not be completed");
      } else if (!pose_sequences_->exitPoseMoving()) {
        source_exit_pending_ = false;
        try {
          switchSource(source_target_);
        } catch (const std::exception & error) {
          failSourceSwitch(error.what());
        }
      }
      return;
    }
    if (source_ == ControlSource::kNone) {
      return;
    }
    if (source_ == ControlSource::kAction) {
      updateAction();
      if (source_manager_->transitioning()) {
        if (mode_ready_) {source_manager_->complete(source_);} else {
          failSourceSwitch("Action controller could not be activated");
        }
      }
      return;
    }

    if (transition_pending_ && hold_initialized_) {
      if (!mode_transition_started_ && !beginRequestedModeTransition()) {
        publishCommand(hold_target_);
        if (source_manager_->transitioning()) {failSourceSwitch("Initial pose setup failed");}
        return;
      }
      if (mode_transition_phase_ == ModeTransitionPhase::kExitPose) {
        if (pose_sequences_->exitPoseMoving()) {
          updateModePoseTransition(true);
          return;
        }
        if (!startRequestedInitialPose()) {
          publishCommand(hold_target_);
          if (source_manager_->transitioning()) {failSourceSwitch("Initial pose setup failed");}
          return;
        }
      }
      if (
        mode_transition_phase_ == ModeTransitionPhase::kInitialPose &&
        pose_sequences_->initialPoseMoving())
      {
        if (!updateModePoseTransition(false) && source_manager_->transitioning()) {
          failSourceSwitch("Initial pose could not be completed");
        }
        return;
      }
      mode_transition_phase_ = ModeTransitionPhase::kActivate;
      if (!activateRequestedMode()) {
        publishCommand(hold_target_);
        if (source_manager_->transitioning()) {failSourceSwitch("Teleop controller setup failed");}
        return;
      }
      if (source_manager_->transitioning()) {source_manager_->complete(source_);}
    }
    if (!mode_ready_ || !mode_) {
      if (hold_initialized_) {
        publishCommand(hold_target_);
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
        publishCommand(command_position_);
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
        publishCommand(command_position_);
        return;
      }
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
      publishCommand(hold_target_);
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

      if (!runtime_->step(output, context.dt, command_position_, command_velocity_)) {
        command_velocity_.setZero();
        publishCommand(command_position_, &output);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Teleoperation QP failed; holding the last command and retrying");
        return;
      }

      publishCommand(command_position_, &output);

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
      publishCommand(hold_target_);
    }
  }

  void publishStatus(const uint8_t state, const std::string & message)
  {
    if (!robot_teleoperation_ || source_ == ControlSource::kNone) {
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
  std::unique_ptr<ControlRuntime> runtime_;
  std::unique_ptr<ActionInput> action_input_;
  std::unique_ptr<CommandSourceManager> source_manager_;
  ModeRegistry teleop_modes_, action_modes_;
  ControlSource source_ = ControlSource::kNone;
  uint16_t teleop_requested_mode_ = 1, action_requested_mode_ = 1;
  ControlSource source_target_ = ControlSource::kNone;
  bool source_exit_pending_ = false;

  std::unordered_map<uint16_t, std::string> mode_names_;
  std::unordered_map<uint16_t, std::string> mode_plugins_;
  uint16_t requested_control_mode_ = 1;
  uint16_t active_control_mode_ = 0;
  uint16_t transition_target_mode_ = 0;
  uint16_t transition_source_mode_ = 0;
  std::vector<uint16_t> selected_preset_ids_;
  mutable std::vector<ControlGroupState> context_group_states_;
  GroupCartesianReferences cartesian_references_;
  ControlGroupMask requested_groups_ = 0;
  ControlGroupMask active_groups_ = 0;
  ControlGroupMask previous_controlled_groups_ = 0;
  uint64_t transition_id_ = 0;
  uint64_t next_transition_id_ = 0;
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
  bool leader_action_enabled_ = false;
  std::vector<uint8_t> last_preset_states_;
  std::vector<uint8_t> last_initial_pose_states_;
  Eigen::VectorXd hold_target_;
  Eigen::VectorXd command_position_;
  Eigen::VectorXd command_velocity_;
  GroupAuxiliaryPositions auxiliary_hold_target_;
  GroupAuxiliaryPositions auxiliary_command_;
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
