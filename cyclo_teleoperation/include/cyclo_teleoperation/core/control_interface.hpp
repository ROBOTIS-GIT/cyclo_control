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

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "cyclo_teleoperation/core/types.hpp"

namespace cyclo_teleoperation
{
struct ControlRequest
{
  uint64_t transition_id = 0;
  uint16_t control_mode = 0;
  ControlGroupMask enabled_groups = 0;
  ControlGroupMask preset_target_groups = 0;
  ControlGroupMask initial_pose_target_groups = 0;
  // Indexed by ControlGroupId. A zero or omitted entry preserves the current selection
  // unless that group is included in preset_target_groups.
  std::vector<uint16_t> preset_ids;
};

struct ControlStatus
{
  static constexpr uint8_t kHolding = 1;
  static constexpr uint8_t kLoading = 2;
  static constexpr uint8_t kActivating = 3;
  static constexpr uint8_t kActive = 4;
  static constexpr uint8_t kError = 5;

  uint64_t transition_id = 0;
  uint16_t requested_control_mode = 0;
  uint16_t active_control_mode = 0;
  ControlGroupMask requested_groups = 0;
  ControlGroupMask active_groups = 0;
  std::vector<uint16_t> preset_ids;
  std::vector<uint8_t> preset_states;
  ControlGroupMask initial_pose_available_groups = 0;
  std::vector<uint8_t> initial_pose_states;
  uint8_t state = kHolding;
  std::string message;
};

class ControlInterface
{
public:
  using RequestCallback = std::function<void(const ControlRequest &)>;

  virtual ~ControlInterface() = default;

  virtual bool configure(
    rclcpp::Node & node,
    const std::string & parameter_prefix,
    const std::vector<ControlGroupConfiguration> & groups,
    RequestCallback request_callback) = 0;

  virtual void publishStatus(const ControlStatus & status) = 0;
};
}  // namespace cyclo_teleoperation
