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

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <robotis_interfaces/msg/control_mode_command.hpp>
#include <robotis_interfaces/msg/control_mode_status.hpp>

#include "cyclo_teleoperation/core/control_interface.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
class AIWorkerControlInterface : public ControlInterface
{
public:
  bool configure(
    rclcpp::Node & node,
    const std::string & parameter_prefix,
    const std::vector<ControlGroupConfiguration> & groups,
    RequestCallback request_callback) override;

  void publishStatus(const ControlStatus & status) override;

private:
  std::optional<ControlGroupMask> groupsFromName(const std::string & name) const;
  std::string groupsToName(ControlGroupMask groups) const;
  ControlGroupId groupId(const std::string & name) const;
  std::string parameterName(const std::string & name) const;
  void commandCallback(
    const robotis_interfaces::msg::ControlModeCommand::SharedPtr message);

  rclcpp::Node * node_ = nullptr;
  std::string parameter_prefix_;
  std::vector<ControlGroupConfiguration> groups_;
  RequestCallback request_callback_;
  ControlGroupId left_group_ = kInvalidControlGroup;
  ControlGroupId right_group_ = kInvalidControlGroup;
  ControlStatus last_status_;

  rclcpp::Subscription<robotis_interfaces::msg::ControlModeCommand>::SharedPtr
  command_subscription_;
  rclcpp::Publisher<robotis_interfaces::msg::ControlModeStatus>::SharedPtr
  status_publisher_;
};
}  // namespace cyclo_teleoperation::robots::ai_worker
