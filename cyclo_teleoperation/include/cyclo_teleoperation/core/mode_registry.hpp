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
#include <string>
#include <unordered_map>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include "cyclo_teleoperation/core/teleoperation_mode.hpp"

namespace cyclo_teleoperation
{
struct ModeEntry
{
  std::string plugin;
  std::string parameter_prefix;
  std::string reference_type;
  bool directJoint() const {return reference_type == "absolute_joint_position";}
};

class ModeRegistry
{
public:
  void configure(
    rclcpp::Node & node, pluginlib::ClassLoader<TeleoperationMode> & loader,
    const std::string & list_parameter, const std::string & prefix,
    const std::string & default_parameter, bool model_action_inputs);
  const ModeEntry & at(uint16_t mode) const {return entries_.at(mode);}
  bool contains(uint16_t mode) const {return entries_.count(mode) != 0;}
  uint16_t defaultMode() const {return default_;}

private:
  std::unordered_map<uint16_t, ModeEntry> entries_;
  uint16_t default_ = 0;
};
}  // namespace cyclo_teleoperation
