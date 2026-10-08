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
#include "cyclo_teleoperation/core/mode_registry.hpp"
#include <stdexcept>

namespace cyclo_teleoperation
{
void ModeRegistry::configure(
  rclcpp::Node & node,
  pluginlib::ClassLoader<TeleoperationMode> & loader, const std::string & list_parameter,
  const std::string & prefix, const std::string & default_parameter, bool model_action_inputs)
{
  auto parameter = [&node](const std::string & key, const auto & value) {
      if (!node.has_parameter(key)) {node.declare_parameter(key, value);}
      return node.get_parameter(key);
    };
  const auto ids = parameter(list_parameter, std::vector<int64_t>{}).as_integer_array();
  for (const auto id : ids) {
    if (id <= 0 || id > UINT16_MAX || contains(static_cast<uint16_t>(id))) {
      throw std::runtime_error(list_parameter + " contains an invalid or duplicate ID");
    }
    ModeEntry entry;
    entry.parameter_prefix = prefix + "." + std::to_string(id);
    if (model_action_inputs) {
      entry.reference_type = parameter(
        entry.parameter_prefix + ".reference_type", std::string{}).as_string();
      if (entry.reference_type != "absolute_joint_position" &&
        entry.reference_type != "absolute_eef_pose")
      {
        throw std::runtime_error(entry.parameter_prefix + ": unsupported reference_type");
      }
    }
    entry.plugin = parameter(entry.parameter_prefix + ".plugin", std::string{}).as_string();
    if (entry.directJoint()) {
      if (!entry.plugin.empty()) {
        throw std::runtime_error(entry.parameter_prefix +
                ": absolute_joint_position is always direct; remove plugin and QP settings");
      }
    } else if (entry.plugin.empty() || !loader.isClassAvailable(entry.plugin)) {
      throw std::runtime_error(entry.parameter_prefix + ": unavailable plugin " + entry.plugin);
    }
    entries_.emplace(static_cast<uint16_t>(id), std::move(entry));
  }
  if (model_action_inputs) {
    default_ = 0;
    for (const auto & [id, entry] : entries_) {
      if (!entry.directJoint()) {continue;}
      if (default_ != 0) {
        throw std::runtime_error(list_parameter +
                " must contain exactly one absolute_joint_position startup mode");
      }
      default_ = id;
    }
    if (default_ == 0) {
      throw std::runtime_error(list_parameter +
              " requires an absolute_joint_position startup mode");
    }
    return;
  }
  const auto value = parameter(default_parameter, int64_t{1}).as_int();
  if (value <= 0 || value > UINT16_MAX || !contains(static_cast<uint16_t>(value))) {
    throw std::runtime_error(default_parameter + " must name a configured mode");
  }
  default_ = static_cast<uint16_t>(value);
}
}  // namespace cyclo_teleoperation
