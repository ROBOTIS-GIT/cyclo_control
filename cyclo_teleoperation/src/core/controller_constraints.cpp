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

#include "cyclo_teleoperation/core/controller_constraints.hpp"

namespace cyclo_teleoperation
{
ControllerConstraints ControllerConstraints::declareAndLoad(
  rclcpp::Node & node, const std::string & parameter_prefix)
{
  ControllerConstraints constraints;
  const std::string position_parameter = parameter_prefix + ".position";
  const std::string self_collision_parameter = parameter_prefix + ".self_collision";
  if (!node.has_parameter(position_parameter)) {
    node.declare_parameter(position_parameter, true);
  }
  if (!node.has_parameter(self_collision_parameter)) {
    node.declare_parameter(self_collision_parameter, true);
  }
  constraints.position_enabled_ = node.get_parameter(position_parameter).as_bool();
  constraints.self_collision_enabled_ =
    node.get_parameter(self_collision_parameter).as_bool();
  return constraints;
}

void ControllerConstraints::apply(
  const ModeConfiguration & configuration,
  const ControlGroupMask groups,
  ModeOutput & output) const
{
  for (const auto & group : configuration.control_groups) {
    if (!containsControlGroup(groups, group.id)) {
      continue;
    }
    for (const int index : group.follower_joint_indices) {
      if (index >= 0 && static_cast<size_t>(index) < output.joint_position_limit_enabled.size()) {
        output.joint_position_limit_enabled[index] = position_enabled_;
      }
    }
  }
  output.mergeSelfCollisionConstraint(self_collision_enabled_);
}

ControlGroupMask configuredControlGroups(const ModeConfiguration & configuration)
{
  ControlGroupMask groups = 0;
  for (const auto & group : configuration.control_groups) {
    groups |= controlGroupBit(group.id);
  }
  return groups;
}
}  // namespace cyclo_teleoperation
