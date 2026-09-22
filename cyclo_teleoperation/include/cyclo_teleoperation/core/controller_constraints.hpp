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

#include <string>

#include <rclcpp/rclcpp.hpp>

#include "cyclo_teleoperation/core/types.hpp"

namespace cyclo_teleoperation
{
class ControllerConstraints
{
public:
  static ControllerConstraints declareAndLoad(
    rclcpp::Node & node, const std::string & parameter_prefix);

  void apply(
    const ModeConfiguration & configuration,
    ControlGroupMask groups,
    ModeOutput & output) const;

  bool positionEnabled() const {return position_enabled_;}
  bool selfCollisionEnabled() const {return self_collision_enabled_;}

private:
  bool position_enabled_ = true;
  bool self_collision_enabled_ = true;
};

ControlGroupMask configuredControlGroups(const ModeConfiguration & configuration);
}  // namespace cyclo_teleoperation
