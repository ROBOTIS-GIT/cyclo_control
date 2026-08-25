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

#include "cyclo_teleoperation/robots/ai_worker/ai_worker_control_interface.hpp"

#include <algorithm>
#include <utility>

namespace cyclo_teleoperation::robots::ai_worker
{
std::string AIWorkerControlInterface::parameterName(const std::string & name) const
{
  return parameter_prefix_.empty() ? name : parameter_prefix_ + "." + name;
}

ControlGroupId AIWorkerControlInterface::groupId(const std::string & name) const
{
  for (const auto & group : groups_) {
    if (group.name == name) {
      return group.id;
    }
  }
  return kInvalidControlGroup;
}

bool AIWorkerControlInterface::configure(
  rclcpp::Node & node,
  const std::string & parameter_prefix,
  const std::vector<ControlGroupConfiguration> & groups,
  RequestCallback request_callback)
{
  node_ = &node;
  parameter_prefix_ = parameter_prefix;
  groups_ = groups;
  request_callback_ = std::move(request_callback);
  left_group_ = groupId("left");
  right_group_ = groupId("right");
  if (
    left_group_ == kInvalidControlGroup || right_group_ == kInvalidControlGroup ||
    !request_callback_)
  {
    return false;
  }

  const auto command_parameter = parameterName("control_command_topic");
  const auto status_parameter = parameterName("control_status_topic");
  if (!node_->has_parameter(command_parameter)) {
    node_->declare_parameter(command_parameter, "/leader/teleoperation/control_command");
  }
  if (!node_->has_parameter(status_parameter)) {
    node_->declare_parameter(status_parameter, "/leader/teleoperation/control_status");
  }

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  command_subscription_ =
    node_->create_subscription<robotis_interfaces::msg::ControlModeCommand>(
    node_->get_parameter(command_parameter).as_string(), qos,
    std::bind(
      &AIWorkerControlInterface::commandCallback, this, std::placeholders::_1));
  status_publisher_ =
    node_->create_publisher<robotis_interfaces::msg::ControlModeStatus>(
    node_->get_parameter(status_parameter).as_string(), qos);
  return true;
}

std::optional<ControlGroupMask> AIWorkerControlInterface::groupsFromName(
  const std::string & name) const
{
  if (name == "none") {
    return 0;
  }
  if (name == "left") {
    return controlGroupBit(left_group_);
  }
  if (name == "right") {
    return controlGroupBit(right_group_);
  }
  if (name == "both" || name == "all") {
    return controlGroupBit(left_group_) | controlGroupBit(right_group_);
  }
  return std::nullopt;
}

std::string AIWorkerControlInterface::groupsToName(const ControlGroupMask groups) const
{
  const auto left = controlGroupBit(left_group_);
  const auto right = controlGroupBit(right_group_);
  const auto selected = groups & (left | right);
  if (selected == left) {
    return "left";
  }
  if (selected == right) {
    return "right";
  }
  if (selected == (left | right)) {
    return "both";
  }
  return "none";
}

void AIWorkerControlInterface::commandCallback(
  const robotis_interfaces::msg::ControlModeCommand::SharedPtr message)
{
  const auto enabled = groupsFromName(message->enabled_arms);
  const auto preset = groupsFromName(message->preset_target_arm);
  const auto initial = groupsFromName(message->initial_pose_target_arm);
  if (!enabled || !preset || !initial) {
    ControlStatus status = last_status_;
    status.transition_id = message->transition_id;
    status.state = ControlStatus::kError;
    status.message = "A control-group selector contains an invalid value";
    publishStatus(status);
    return;
  }

  size_t state_count = 0;
  for (const auto & group : groups_) {
    state_count = std::max(state_count, static_cast<size_t>(group.id) + 1);
  }
  ControlRequest request;
  request.transition_id = message->transition_id;
  request.control_mode = message->control_mode;
  request.enabled_groups = *enabled;
  request.preset_target_groups = *preset;
  request.initial_pose_target_groups = *initial;
  request.preset_ids.assign(state_count, 0);
  request.preset_ids[left_group_] = message->left_preset_id;
  request.preset_ids[right_group_] = message->right_preset_id;
  request_callback_(request);
}

void AIWorkerControlInterface::publishStatus(const ControlStatus & input)
{
  last_status_ = input;
  if (!status_publisher_) {
    return;
  }
  robotis_interfaces::msg::ControlModeStatus status;
  status.transition_id = input.transition_id;
  status.requested_control_mode = input.requested_control_mode;
  status.active_control_mode = input.active_control_mode;
  status.requested_arms = groupsToName(input.requested_groups);
  status.active_arms = groupsToName(input.active_groups);
  if (left_group_ < input.preset_ids.size()) {
    status.left_preset_id = input.preset_ids[left_group_];
  }
  if (right_group_ < input.preset_ids.size()) {
    status.right_preset_id = input.preset_ids[right_group_];
  }
  if (left_group_ < input.preset_states.size()) {
    status.left_preset_state = input.preset_states[left_group_];
  }
  if (right_group_ < input.preset_states.size()) {
    status.right_preset_state = input.preset_states[right_group_];
  }
  status.initial_pose_available_arms =
    groupsToName(input.initial_pose_available_groups);
  if (left_group_ < input.initial_pose_states.size()) {
    status.left_initial_pose_state = input.initial_pose_states[left_group_];
  }
  if (right_group_ < input.initial_pose_states.size()) {
    status.right_initial_pose_state = input.initial_pose_states[right_group_];
  }
  status.state = input.state;
  status.message = input.message;
  status_publisher_->publish(status);
}
}  // namespace cyclo_teleoperation::robots::ai_worker
