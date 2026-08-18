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

#include <rclcpp/rclcpp.hpp>

#include "cyclo_teleoperation/core/types.hpp"

namespace cyclo_teleoperation
{
class TeleoperationMode
{
public:
  virtual ~TeleoperationMode() = default;

  virtual bool configure(
    rclcpp::Node & node,
    const std::string & parameter_prefix,
    const ModeConfiguration & configuration) = 0;

  virtual bool activate(const ModeContext & context) = 0;

  virtual void onArmsEnabled(uint8_t arms, const ModeContext & context) = 0;

  virtual bool update(const ModeContext & context, ModeOutput & output) = 0;

  virtual uint8_t controlledArms(const ModeContext & context) const
  {
    return context.enabled_arms;
  }

  virtual void deactivate() {}
};
}  // namespace cyclo_teleoperation
