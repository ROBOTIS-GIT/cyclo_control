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
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <robotis_interfaces/msg/control_mode_command.hpp>
#include <robotis_interfaces/msg/control_mode_status.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "cyclo_teleoperation/core/teleoperation_mode.hpp"
#include "cyclo_teleoperation/core/teleoperation_qp.hpp"
#include "cyclo_teleoperation/core/soft_hold.hpp"
#include "cyclo_teleoperation/robots/ai_worker/ai_worker_profile.hpp"
#include "cyclo_teleoperation/robots/ai_worker/pose_sequence_manager.hpp"

namespace cyclo_teleoperation
{
namespace
{
std::optional<uint8_t> armsFromName(const std::string & name)
{
  if (name == "none") {
    return 0;
  }
  if (name == "left") {
    return kLeftArm;
  }
  if (name == "right") {
    return kRightArm;
  }
  if (name == "both") {
    return kBothArms;
  }
  return std::nullopt;
}

std::string armsToName(const uint8_t arms)
{
  switch (arms & kBothArms) {
    case kLeftArm:
      return "left";
    case kRightArm:
      return "right";
    case kBothArms:
      return "both";
    default:
      return "none";
  }
}
}  // namespace

class CycloTeleoperationNode : public rclcpp::Node
{
public:
  CycloTeleoperationNode()
  : Node("cyclo_teleoperation"),
    mode_loader_("cyclo_teleoperation", "cyclo_teleoperation::TeleoperationMode")
  {
    declareParameters();
    if (get_parameter("robot_type").as_string() != "ai_worker") {
      throw std::runtime_error("Only the ai_worker follower profile is implemented");
    }

    profile_ = std::make_unique<robots::ai_worker::AIWorkerProfile>(*this);
    if (!profile_->initialize()) {
      throw std::runtime_error("Failed to initialize AI Worker follower profile");
    }
    pose_sequences_ = std::make_unique<robots::ai_worker::PoseSequenceManager>();
    if (!pose_sequences_->configure(
        *this, profile_->modeConfiguration(),
        get_parameter("available_control_modes").as_integer_array(),
        get_parameter("available_presets").as_integer_array()))
    {
      throw std::runtime_error("Failed to configure AI Worker pose sequences");
    }

    qp_ = std::make_unique<TeleoperationQP>(profile_->followerKinematics());
    qp_->setControllerParameters(
      get_parameter("constraints.slack_penalty").as_double(),
      get_parameter("constraints.cbf_alpha").as_double(),
      get_parameter("constraints.collision_buffer").as_double(),
      get_parameter("constraints.collision_safe_distance").as_double());

    follower_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      get_parameter("follower_joint_states_topic").as_string(), 10,
      std::bind(&CycloTeleoperationNode::followerCallback, this, std::placeholders::_1));
    right_leader_subscription_ =
      create_subscription<trajectory_msgs::msg::JointTrajectory>(
      get_parameter("right_leader_topic").as_string(), 10,
      [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr message) {
        if (profile_->updateLeaderReference(*message, kRightArm)) {
          right_leader_received_ = true;
          last_right_leader_time_ = now();
        }
      });
    left_leader_subscription_ =
      create_subscription<trajectory_msgs::msg::JointTrajectory>(
      get_parameter("left_leader_topic").as_string(), 10,
      [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr message) {
        if (profile_->updateLeaderReference(*message, kLeftArm)) {
          left_leader_received_ = true;
          last_left_leader_time_ = now();
        }
      });

    auto command_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    command_subscription_ =
      create_subscription<robotis_interfaces::msg::ControlModeCommand>(
      get_parameter("control_command_topic").as_string(), command_qos,
      std::bind(&CycloTeleoperationNode::commandCallback, this, std::placeholders::_1));
    status_publisher_ =
      create_publisher<robotis_interfaces::msg::ControlModeStatus>(
      get_parameter("control_status_topic").as_string(), command_qos);

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
      std::bind(&CycloTeleoperationNode::controlLoop, this));
    publishStatus(
      robotis_interfaces::msg::ControlModeStatus::STATE_HOLDING,
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

  void declareParameters()
  {
    declare_parameter("robot_type", "ai_worker");
    declare_parameter("control_frequency", 100.0);
    declare_parameter("trajectory_time", 0.0);
    declare_parameter("joint_state_timeout", 0.5);
    declare_parameter("leader_command_timeout", 0.5);
    declare_parameter("follower_urdf_path", "");
    declare_parameter("follower_srdf_path", "");
    declare_parameter("leader_urdf_path", "");
    declare_parameter("leader_urdf_xml", "");
    declare_parameter("leader_srdf_path", "");

    declare_parameter("follower_joint_states_topic", "/joint_states");
    declare_parameter(
      "right_leader_topic",
      "/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory");
    declare_parameter(
      "left_leader_topic",
      "/leader/joint_trajectory_command_broadcaster_left/raw_joint_trajectory");
    declare_parameter(
      "right_command_topic",
      "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory");
    declare_parameter(
      "left_command_topic",
      "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory");
    declare_parameter("control_command_topic", "/leader/teleoperation/control_command");
    declare_parameter("control_status_topic", "/leader/teleoperation/control_status");

    declare_parameter("right_gripper_joint", "gripper_r_joint1");
    declare_parameter("left_gripper_joint", "gripper_l_joint1");
    declare_parameter("follower_right_eef", "arm_r_link7");
    declare_parameter("follower_left_eef", "arm_l_link7");
    declare_parameter("follower_right_elbow", "arm_r_link4");
    declare_parameter("follower_left_elbow", "arm_l_link4");
    declare_parameter("leader_right_eef", "arm_r_link7");
    declare_parameter("leader_left_eef", "arm_l_link7");

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
      "available_control_modes", {1, 2});
    declare_parameter("default_control_mode", 1);
    const auto control_modes =
      get_parameter("available_control_modes").as_integer_array();
    for (const int64_t raw_mode : control_modes) {
      if (raw_mode <= 0 || raw_mode > UINT16_MAX) {
        throw std::runtime_error("Control mode IDs must be in the uint16 range");
      }
      const auto mode = static_cast<uint16_t>(raw_mode);
      const std::string prefix = "control_modes." + std::to_string(mode);
      const std::string default_plugin =
        mode == 1 ? "cyclo_teleoperation/AiWorkerMoveJMode" :
        mode == 2 ? "cyclo_teleoperation/AiWorkerElbowUpLeaderMode" : "";
      mode_names_[mode] =
        declare_parameter(prefix + ".name", "mode_" + std::to_string(mode));
      mode_plugins_[mode] = declare_parameter(prefix + ".plugin", default_plugin);
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
          for (const auto & arm : {std::string("left"), std::string("right")}) {
            declare_parameter<std::vector<double>>(
              sequence_prefix + ".steps." + step_name + "." + arm + ".positions",
              std::vector<double>{});
          }
        }
        declare_parameter(sequence_prefix + ".duration", 3.0);
        declare_parameter(sequence_prefix + ".completion_tolerance", 0.03);
        declare_parameter(sequence_prefix + ".timeout", 10.0);
      }
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
        for (const auto & arm : {std::string("left"), std::string("right")}) {
          declare_parameter<std::vector<double>>(
            prefix + ".steps." + step_name + "." + arm + ".positions",
            std::vector<double>{});
        }
      }
      declare_parameter(prefix + ".duration", 3.0);
      declare_parameter(prefix + ".completion_tolerance", 0.03);
      declare_parameter(prefix + ".timeout", 10.0);
    }
  }

  bool isModeAvailable(const uint16_t mode) const
  {
    const auto iter = mode_plugins_.find(mode);
    return iter != mode_plugins_.end() && !iter->second.empty();
  }

  bool isPresetAvailable(const uint16_t preset, const uint8_t arm) const
  {
    return pose_sequences_ && pose_sequences_->hasPreset(preset, arm);
  }

  void followerCallback(const sensor_msgs::msg::JointState::SharedPtr message)
  {
    if (!profile_->updateFollowerState(*message)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Follower state does not contain every joint used by the AI Worker model");
      return;
    }
    last_follower_time_ = now();
    follower_received_ = true;
    if (!hold_initialized_) {
      hold_target_ = profile_->followerPosition();
      syncCommandToFeedback();
      hold_initialized_ = true;
      transition_pending_ = true;
    }
  }

  void commandCallback(
    const robotis_interfaces::msg::ControlModeCommand::SharedPtr message)
  {
    if (!isModeAvailable(message->control_mode)) {
      transition_id_ = message->transition_id;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        "Unknown or unavailable control mode: " +
        std::to_string(message->control_mode));
      return;
    }
    const auto requested_arms = armsFromName(message->enabled_arms);
    const auto preset_target_arms = armsFromName(message->preset_target_arm);
    const auto initial_pose_target_arms = armsFromName(message->initial_pose_target_arm);
    if (!requested_arms || !preset_target_arms || !initial_pose_target_arms) {
      transition_id_ = message->transition_id;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        "An arm selector in the control command contains an invalid value");
      return;
    }
    const uint8_t preset_target = *preset_target_arms;
    const uint8_t initial_pose_target = *initial_pose_target_arms;
    if (preset_target != 0 && initial_pose_target != 0) {
      transition_id_ = message->transition_id;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        "Preset and initial pose cannot be requested in the same command");
      return;
    }
    if (
      !isPresetAvailable(message->left_preset_id, kLeftArm) ||
      !isPresetAvailable(message->right_preset_id, kRightArm))
    {
      transition_id_ = message->transition_id;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        "Preset is not configured for the requested arm");
      return;
    }

    const uint8_t moving_initial_pose_arms =
      pose_sequences_->movingFinalInitialPoseArms();
    const bool changing_mode =
      mode_ready_ && message->control_mode != active_control_mode_;
    if (changing_mode && moving_initial_pose_arms != 0) {
      transition_id_ = message->transition_id;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        "Control mode cannot be changed while an initial pose movement is in progress");
      return;
    }
    if (initial_pose_target != 0) {
      const uint8_t available_arms =
        mode_ready_ ? pose_sequences_->initialPoseArms(active_control_mode_) : 0;
      if (
        !mode_ready_ || transition_pending_ || mode_transition_started_ ||
        message->control_mode != active_control_mode_ ||
        (initial_pose_target & available_arms) != initial_pose_target)
      {
        transition_id_ = message->transition_id;
        publishStatus(
          robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
          "Initial pose trigger ignored because it is disabled for the active control mode");
        return;
      }
      if (
        (initial_pose_target &
        (pose_sequences_->movingPresetArms() | moving_initial_pose_arms)) != 0)
      {
        transition_id_ = message->transition_id;
        publishStatus(
          robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
          "Initial pose trigger ignored because another pose movement is in progress");
        return;
      }
    }
    if ((preset_target & moving_initial_pose_arms) != 0) {
      transition_id_ = message->transition_id;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        "Preset cannot be started while an initial pose movement is in progress");
      return;
    }

    const uint8_t newly_enabled_arms = *requested_arms &
      static_cast<uint8_t>(~requested_arms_);
    if ((newly_enabled_arms & moving_initial_pose_arms) != 0) {
      transition_id_ = message->transition_id;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        "Teleoperation cannot be enabled while an initial pose movement is in progress");
      return;
    }

    transition_id_ = message->transition_id;
    requested_control_mode_ = message->control_mode;
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
    requested_arms_ = *requested_arms;
    requested_arms_ &= static_cast<uint8_t>(~(preset_target | initial_pose_target));
    left_preset_id_ = message->left_preset_id;
    right_preset_id_ = message->right_preset_id;
    transition_pending_ =
      active_control_mode_ != requested_control_mode_ || !mode_ready_;
    arms_pending_ = !transition_pending_;
    if (preset_target != 0) {
      preset_update_pending_arms_ |= preset_target;
      preset_cancel_pending_arms_ &= static_cast<uint8_t>(~preset_target);
    }
    if (initial_pose_target != 0) {
      final_initial_pose_update_pending_arms_ |= initial_pose_target;
      final_initial_pose_cancel_pending_arms_ &=
        static_cast<uint8_t>(~initial_pose_target);
    }
    const uint8_t resume_arms = newly_enabled_arms &
      pose_sequences_->activeFinalInitialPoseArms();
    if (resume_arms != 0) {
      final_initial_pose_cancel_pending_arms_ |= resume_arms;
    }
  }

  bool feedbackFresh() const
  {
    return follower_received_ &&
           (now() - last_follower_time_).seconds() <=
           get_parameter("joint_state_timeout").as_double();
  }

  uint8_t freshLeaderArms() const
  {
    const double timeout = get_parameter("leader_command_timeout").as_double();
    uint8_t result = 0;
    if (
      left_leader_received_ &&
      (now() - last_left_leader_time_).seconds() <= timeout)
    {
      result |= kLeftArm;
    }
    if (
      right_leader_received_ &&
      (now() - last_right_leader_time_).seconds() <= timeout)
    {
      result |= kRightArm;
    }
    return result;
  }

  ModeContext makeContext(const uint8_t enabled_arms) const
  {
    return ModeContext{
      command_position_,
      command_velocity_,
      profile_->followerPosition(),
      profile_->leaderReference(),
      profile_->leaderPosition(),
      requested_arms_,
      enabled_arms,
      left_preset_id_,
      right_preset_id_,
      now().seconds(),
      1.0 / std::max(1.0, get_parameter("control_frequency").as_double())};
  }

  void syncArmCommandToFeedback(const uint8_t arms)
  {
    auto sync_arm = [this, arms](const std::vector<int> & indices, const uint8_t arm) {
        if ((arms & arm) == 0) {
          return;
        }
        for (const int index : indices) {
          command_position_[index] = profile_->followerPosition()[index];
          command_velocity_[index] = 0.0;
        }
      };
    sync_arm(profile_->leftArmIndices(), kLeftArm);
    sync_arm(profile_->rightArmIndices(), kRightArm);
  }

  void captureArmHoldTarget(const uint8_t arms)
  {
    auto capture_arm = [this, arms](
      const std::vector<int> & indices, const uint8_t arm)
      {
        if ((arms & arm) == 0) {
          return;
        }
        for (const int index : indices) {
          hold_target_[index] = profile_->followerPosition()[index];
        }
      };
    capture_arm(profile_->leftArmIndices(), kLeftArm);
    capture_arm(profile_->rightArmIndices(), kRightArm);
  }

  void syncCommandToFeedback()
  {
    command_position_ = profile_->followerPosition();
    command_velocity_ = Eigen::VectorXd::Zero(profile_->dof());
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
        const uint8_t initial_pose_arms = pose_sequences_->activeInitialPoseArms();
        pose_sequences_->cancelPresets(initial_pose_arms);
        paused_preset_arms_ &= static_cast<uint8_t>(~initial_pose_arms);
        mode_transition_phase_ = ModeTransitionPhase::kInitialPose;
        publishStatus(
          robotis_interfaces::msg::ControlModeStatus::STATE_LOADING,
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
      paused_preset_arms_ = 0;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        std::string("Failed to begin initial pose transition: ") + error.what());
      return false;
    }
  }

  bool beginRequestedModeTransition()
  {
    mode_transition_phase_ = ModeTransitionPhase::kIdle;
    transition_target_mode_ = requested_control_mode_;
    transition_source_mode_ = active_control_mode_;
    paused_preset_arms_ = pose_sequences_->activePresetArms();
    pose_sequences_->cancelFinalInitialPoses(kBothArms);
    final_initial_pose_update_pending_arms_ = 0;
    final_initial_pose_cancel_pending_arms_ = 0;
    publishStatus(
      robotis_interfaces::msg::ControlModeStatus::STATE_LOADING,
      "Loading control mode " + std::to_string(requested_control_mode_));
    hold_target_ = profile_->followerPosition();
    syncCommandToFeedback();
    profile_->publish(hold_target_);
    active_arms_ = 0;
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
        const uint8_t exit_pose_arms = pose_sequences_->activeExitPoseArms();
        pose_sequences_->cancelPresets(exit_pose_arms);
        paused_preset_arms_ &= static_cast<uint8_t>(~exit_pose_arms);
        mode_transition_phase_ = ModeTransitionPhase::kExitPose;
        publishStatus(
          robotis_interfaces::msg::ControlModeStatus::STATE_LOADING,
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
      paused_preset_arms_ = 0;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        std::string("Failed to begin mode transition: ") + error.what());
      return false;
    }
  }

  bool activateRequestedMode()
  {
    publishStatus(
      robotis_interfaces::msg::ControlModeStatus::STATE_ACTIVATING,
      "Activating control mode " + std::to_string(transition_target_mode_));
    try {
      mode_ =
        mode_loader_.createSharedInstance(mode_plugins_.at(transition_target_mode_));
      const std::string parameter_prefix =
        "control_modes." + std::to_string(transition_target_mode_);
      if (!mode_->configure(
          *this, parameter_prefix, profile_->modeConfiguration()))
      {
        throw std::runtime_error("mode configuration was rejected");
      }
      profile_->followerKinematics()->updateState(
        profile_->followerPosition(), profile_->followerVelocity());
      profile_->leaderKinematics()->updateState(
        profile_->leaderPosition(),
        Eigen::VectorXd::Zero(profile_->leaderPosition().size()));
      const uint8_t initial_arms =
        requested_arms_ & freshLeaderArms() &
        static_cast<uint8_t>(
        ~(pose_sequences_->movingPresetArms() |
        pose_sequences_->movingFinalInitialPoseArms()));
      const ModeContext initial_context = makeContext(initial_arms);
      if (!mode_->activate(initial_context)) {
        throw std::runtime_error("mode activation was rejected");
      }
      if (
        paused_preset_arms_ != 0 &&
        !pose_sequences_->startPreset(
          paused_preset_arms_, left_preset_id_, right_preset_id_, initial_context))
      {
        throw std::runtime_error("preset overlay reactivation was rejected");
      }
      active_control_mode_ = transition_target_mode_;
      active_arms_ = initial_arms;
      mode_ready_ = true;
      transition_pending_ = false;
      mode_transition_phase_ = ModeTransitionPhase::kIdle;
      mode_transition_started_ = false;
      paused_preset_arms_ = 0;
      pose_sequences_->cancelInitialPose();
      pose_sequences_->cancelExitPose();
      arms_pending_ = requested_arms_ != active_arms_;
      const bool mode_has_output =
        (mode_->controlledArms(initial_context) |
        pose_sequences_->activePresetArms() |
        pose_sequences_->activeFinalInitialPoseArms()) != 0;
      publishStatus(
        !mode_has_output ?
        robotis_interfaces::msg::ControlModeStatus::STATE_HOLDING :
        robotis_interfaces::msg::ControlModeStatus::STATE_ACTIVE,
        active_arms_ == requested_arms_ ?
        "Mode activated" : "Mode activated; waiting for fresh leader reference");
      return true;
    } catch (const std::exception & error) {
      mode_.reset();
      pose_sequences_->cancelInitialPose();
      pose_sequences_->cancelExitPose();
      mode_ready_ = false;
      active_control_mode_ = 0;
      active_arms_ = 0;
      transition_pending_ = false;
      mode_transition_phase_ = ModeTransitionPhase::kIdle;
      mode_transition_started_ = false;
      paused_preset_arms_ = 0;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        std::string("Failed to load mode: ") + error.what());
      return false;
    }
  }

  void updateActiveArms()
  {
    const uint8_t desired =
      requested_arms_ & freshLeaderArms() &
      static_cast<uint8_t>(
      ~(pose_sequences_->movingPresetArms() |
      pose_sequences_->movingFinalInitialPoseArms()));
    if (desired == active_arms_) {
      arms_pending_ = requested_arms_ != desired;
      return;
    }

    const uint8_t disabled = active_arms_ & ~desired;
    const uint8_t enabled = desired & ~active_arms_;
    captureArmHoldTarget(disabled);

    syncArmCommandToFeedback(disabled | enabled);

    active_arms_ = desired;
    if (enabled != 0 && mode_) {
      pose_sequences_->cancelPresets(enabled);
      pose_sequences_->cancelFinalInitialPoses(enabled);
      mode_->onArmsEnabled(enabled, makeContext(active_arms_));
    }
    arms_pending_ = requested_arms_ != active_arms_;
    publishStatus(
      active_arms_ == 0 ?
      robotis_interfaces::msg::ControlModeStatus::STATE_HOLDING :
      robotis_interfaces::msg::ControlModeStatus::STATE_ACTIVE,
      arms_pending_ ?
      ((pose_sequences_->movingPresetArms() |
      pose_sequences_->movingFinalInitialPoseArms()) != 0 ?
      "Waiting for pose movement to finish" :
      "Holding arm with stale leader reference") : "Arm state updated");
  }

  bool updateModePoseTransition(const bool exit_pose)
  {
    const ModeContext context = makeContext(0);
    try {
      profile_->followerKinematics()->updateState(
        command_position_, command_velocity_);

      ModeOutput output;
      output.reset(
        profile_->dof(),
        get_parameter("constraints.damping_weight").as_double());
      const bool update_success = exit_pose ?
        pose_sequences_->updateExitPose(context, output) :
        pose_sequences_->updateInitialPose(context, output);
      if (!update_success) {
        throw std::runtime_error(pose_sequences_->errorMessage());
      }

      const uint8_t controlled_arms = exit_pose ?
        pose_sequences_->activeExitPoseArms() :
        pose_sequences_->activeInitialPoseArms();
      applySoftHold(
        output, command_position_, hold_target_,
        profile_->leftArmIndices(), profile_->rightArmIndices(),
        controlled_arms,
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
        profile_->publish(command_position_);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "%s pose QP failed; holding the last command and retrying",
          exit_pose ? "Exit" : "Initial");
        return true;
      }
      command_position_ += context.dt * optimal_velocity;
      command_velocity_ = optimal_velocity;
      profile_->publish(command_position_);
      const bool moving = exit_pose ?
        pose_sequences_->exitPoseMoving() :
        pose_sequences_->initialPoseMoving();
      if (!moving) {
        hold_target_ = command_position_;
        command_velocity_.setZero();
        publishStatus(
          robotis_interfaces::msg::ControlModeStatus::STATE_LOADING,
          exit_pose ?
          "Exit pose reached; preparing the requested control mode" :
          "Initial pose reached; control mode activation is now allowed");
      }
      return true;
    } catch (const std::exception & error) {
      hold_target_ = profile_->followerPosition();
      syncCommandToFeedback();
      pose_sequences_->cancelInitialPose();
      pose_sequences_->cancelExitPose();
      mode_transition_phase_ = ModeTransitionPhase::kIdle;
      mode_transition_started_ = false;
      transition_pending_ = false;
      paused_preset_arms_ = 0;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        std::string(exit_pose ? "Exit" : "Initial") +
        " pose transition failed: " + error.what());
      profile_->publish(hold_target_);
      return false;
    }
  }

  void controlLoop()
  {
    if (!feedbackFresh()) {
      if (follower_received_ && !feedback_error_reported_) {
        feedback_error_reported_ = true;
        hold_target_ = profile_->followerPosition();
        command_initialized_ = false;
        active_arms_ = 0;
        publishStatus(
          robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
          "Follower feedback timed out; holding all arms");
      }
      return;
    }
    feedback_error_reported_ = false;
    if (!command_initialized_) {
      syncCommandToFeedback();
    }

    if (transition_pending_ && hold_initialized_) {
      if (!mode_transition_started_ && !beginRequestedModeTransition()) {
        profile_->publish(hold_target_);
        return;
      }
      if (mode_transition_phase_ == ModeTransitionPhase::kExitPose) {
        if (pose_sequences_->exitPoseMoving()) {
          updateModePoseTransition(true);
          return;
        }
        if (!startRequestedInitialPose()) {
          profile_->publish(hold_target_);
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
        profile_->publish(hold_target_);
        return;
      }
    }
    if (!mode_ready_ || !mode_) {
      if (hold_initialized_) {
        profile_->publish(hold_target_);
      }
      return;
    }

    if (preset_cancel_pending_arms_ != 0) {
      const uint8_t cancel_arms = preset_cancel_pending_arms_;
      preset_cancel_pending_arms_ = 0;
      captureArmHoldTarget(cancel_arms);
      syncArmCommandToFeedback(cancel_arms);
      pose_sequences_->cancelPresets(cancel_arms);
    }

    if (final_initial_pose_cancel_pending_arms_ != 0) {
      const uint8_t cancel_arms = final_initial_pose_cancel_pending_arms_;
      final_initial_pose_cancel_pending_arms_ = 0;
      captureArmHoldTarget(cancel_arms);
      syncArmCommandToFeedback(cancel_arms);
      pose_sequences_->cancelFinalInitialPoses(cancel_arms);
    }

    if (
      arms_pending_ ||
      (requested_arms_ & freshLeaderArms()) != active_arms_)
    {
      updateActiveArms();
    }

    if (final_initial_pose_update_pending_arms_ != 0) {
      const uint8_t update_arms = final_initial_pose_update_pending_arms_;
      final_initial_pose_update_pending_arms_ = 0;
      syncArmCommandToFeedback(update_arms);
      pose_sequences_->cancelPresets(update_arms);
      if (!pose_sequences_->startFinalInitialPose(
          active_control_mode_, update_arms, makeContext(active_arms_)))
      {
        publishStatus(
          robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
          pose_sequences_->errorMessage());
        profile_->publish(command_position_);
        return;
      }
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ACTIVE,
        "Moving selected arm to the final step of the active mode initial pose");
    }

    if (preset_update_pending_arms_ != 0) {
      const uint8_t update_arms = preset_update_pending_arms_;
      preset_update_pending_arms_ = 0;
      syncArmCommandToFeedback(update_arms);
      if (!pose_sequences_->startPreset(
          update_arms, left_preset_id_, right_preset_id_, makeContext(active_arms_)))
      {
        publishStatus(
          robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
          "Preset update was rejected by the active mode");
        profile_->publish(command_position_);
        return;
      }
    }

    const ModeContext context = makeContext(active_arms_);
    const uint8_t controlled_arms =
      mode_->controlledArms(context) |
      pose_sequences_->activePresetArms() |
      pose_sequences_->activeFinalInitialPoseArms();
    if (controlled_arms == 0) {
      command_position_ = hold_target_;
      command_velocity_.setZero();
      profile_->publish(hold_target_);
      return;
    }

    try {
      profile_->followerKinematics()->updateState(
        command_position_, command_velocity_);
      profile_->leaderKinematics()->updateState(
        profile_->leaderPosition(),
        Eigen::VectorXd::Zero(profile_->leaderPosition().size()));

      ModeOutput output;
      output.reset(
        profile_->dof(),
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
        profile_->leftArmIndices(), profile_->rightArmIndices(),
        controlled_arms, hold_kp, max_hold_velocity, hold_weight);

      qp_->setModeOutput(output);
      qp_->setControllerParameters(
        get_parameter("constraints.slack_penalty").as_double(),
        get_parameter("constraints.cbf_alpha").as_double(),
        get_parameter("constraints.collision_buffer").as_double(),
        get_parameter("constraints.collision_safe_distance").as_double());
      Eigen::VectorXd optimal_velocity;
      if (!qp_->solve(optimal_velocity)) {
        command_velocity_.setZero();
        profile_->publish(command_position_);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Teleoperation QP failed; holding the last command and retrying");
        return;
      }

      command_position_ += context.dt * optimal_velocity;
      command_velocity_ = optimal_velocity;
      profile_->publish(command_position_);

      const uint8_t left_state = pose_sequences_->leftPresetState();
      const uint8_t right_state = pose_sequences_->rightPresetState();
      const uint8_t left_initial_pose_state =
        pose_sequences_->leftFinalInitialPoseState();
      const uint8_t right_initial_pose_state =
        pose_sequences_->rightFinalInitialPoseState();
      if (
        left_state != last_left_preset_state_ ||
        right_state != last_right_preset_state_ ||
        left_initial_pose_state != last_left_initial_pose_state_ ||
        right_initial_pose_state != last_right_initial_pose_state_)
      {
        last_left_preset_state_ = left_state;
        last_right_preset_state_ = right_state;
        last_left_initial_pose_state_ = left_initial_pose_state;
        last_right_initial_pose_state_ = right_initial_pose_state;
        publishStatus(
          robotis_interfaces::msg::ControlModeStatus::STATE_ACTIVE,
          "Pose sequence state updated");
      }
    } catch (const std::exception & error) {
      hold_target_ = profile_->followerPosition();
      syncCommandToFeedback();
      active_arms_ = 0;
      requested_arms_ = 0;
      pose_sequences_->cancelPresets(kBothArms);
      pose_sequences_->cancelFinalInitialPoses(kBothArms);
      preset_update_pending_arms_ = 0;
      preset_cancel_pending_arms_ = 0;
      final_initial_pose_update_pending_arms_ = 0;
      final_initial_pose_cancel_pending_arms_ = 0;
      publishStatus(
        robotis_interfaces::msg::ControlModeStatus::STATE_ERROR,
        std::string("Control update failed; holding all arms: ") + error.what());
      profile_->publish(hold_target_);
    }
  }

  void publishStatus(const uint8_t state, const std::string & message)
  {
    if (!status_publisher_) {
      return;
    }
    robotis_interfaces::msg::ControlModeStatus status;
    status.transition_id = transition_id_;
    status.requested_control_mode = requested_control_mode_;
    status.active_control_mode = active_control_mode_;
    status.requested_arms = armsToName(requested_arms_);
    status.active_arms = armsToName(active_arms_);
    status.left_preset_id = left_preset_id_;
    status.right_preset_id = right_preset_id_;
    status.left_preset_state = pose_sequences_ ? pose_sequences_->leftPresetState() : 0;
    status.right_preset_state = pose_sequences_ ? pose_sequences_->rightPresetState() : 0;
    status.initial_pose_available_arms = armsToName(
      pose_sequences_ && mode_ready_ ?
      pose_sequences_->initialPoseArms(active_control_mode_) : 0);
    status.left_initial_pose_state =
      pose_sequences_ ? pose_sequences_->leftFinalInitialPoseState() : 0;
    status.right_initial_pose_state =
      pose_sequences_ ? pose_sequences_->rightFinalInitialPoseState() : 0;
    status.state = state;
    status.message = message;
    status_publisher_->publish(status);
  }

  pluginlib::ClassLoader<TeleoperationMode> mode_loader_;
  std::shared_ptr<TeleoperationMode> mode_;
  std::unique_ptr<robots::ai_worker::AIWorkerProfile> profile_;
  std::unique_ptr<robots::ai_worker::PoseSequenceManager> pose_sequences_;
  std::unique_ptr<TeleoperationQP> qp_;

  std::unordered_map<uint16_t, std::string> mode_names_;
  std::unordered_map<uint16_t, std::string> mode_plugins_;
  uint16_t requested_control_mode_ = 1;
  uint16_t active_control_mode_ = 0;
  uint16_t transition_target_mode_ = 0;
  uint16_t transition_source_mode_ = 0;
  uint16_t left_preset_id_ = 1;
  uint16_t right_preset_id_ = 1;
  uint8_t requested_arms_ = 0;
  uint8_t active_arms_ = 0;
  uint64_t transition_id_ = 0;
  bool transition_pending_ = false;
  bool mode_transition_started_ = false;
  ModeTransitionPhase mode_transition_phase_ = ModeTransitionPhase::kIdle;
  bool arms_pending_ = false;
  uint8_t preset_update_pending_arms_ = 0;
  uint8_t preset_cancel_pending_arms_ = 0;
  uint8_t final_initial_pose_update_pending_arms_ = 0;
  uint8_t final_initial_pose_cancel_pending_arms_ = 0;
  uint8_t paused_preset_arms_ = 0;
  bool mode_ready_ = false;
  bool follower_received_ = false;
  bool hold_initialized_ = false;
  bool right_leader_received_ = false;
  bool left_leader_received_ = false;
  bool feedback_error_reported_ = false;
  bool command_initialized_ = false;
  uint8_t last_left_preset_state_ = 0;
  uint8_t last_right_preset_state_ = 0;
  uint8_t last_left_initial_pose_state_ = 0;
  uint8_t last_right_initial_pose_state_ = 0;
  Eigen::VectorXd hold_target_;
  Eigen::VectorXd command_position_;
  Eigen::VectorXd command_velocity_;
  rclcpp::Time last_follower_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_right_leader_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_left_leader_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr follower_subscription_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr
    right_leader_subscription_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr
    left_leader_subscription_;
  rclcpp::Subscription<robotis_interfaces::msg::ControlModeCommand>::SharedPtr
    command_subscription_;
  rclcpp::Publisher<robotis_interfaces::msg::ControlModeStatus>::SharedPtr status_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
    parameter_callback_handle_;
};
}  // namespace cyclo_teleoperation

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<cyclo_teleoperation::CycloTeleoperationNode>());
  } catch (const std::exception & error) {
    fprintf(stderr, "cyclo_teleoperation failed: %s\n", error.what());
  }
  rclcpp::shutdown();
  return 0;
}
