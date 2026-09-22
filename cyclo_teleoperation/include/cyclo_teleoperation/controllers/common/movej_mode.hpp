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

#include <Eigen/Dense>

#include <string>
#include <unordered_map>

#include "cyclo_teleoperation/core/controller_constraints.hpp"
#include "cyclo_teleoperation/core/fixed_duration_slow_start.hpp"
#include "cyclo_teleoperation/core/teleoperation_mode.hpp"

namespace cyclo_teleoperation::controllers::common
{
class MoveJMode : public TeleoperationMode
{
public:
  bool configure(
    rclcpp::Node & node,
    const std::string & parameter_prefix,
    const ModeConfiguration & configuration) override;
  bool activate(const ModeContext & context) override;
  void onGroupsEnabled(
    ControlGroupMask groups, const ModeContext & context) override;
  bool update(const ModeContext & context, ModeOutput & output) override;

private:
  struct ArmTrajectory
  {
    FixedDurationSlowStart slow_start;
  };

  void updateArm(
    const ControlGroupConfiguration & group,
    const ControlGroupState & state,
    ArmTrajectory & trajectory,
    const ModeContext & context,
    ModeOutput & output);

  ModeConfiguration configuration_;
  std::unordered_map<ControlGroupId, ArmTrajectory> trajectories_;
  ControllerConstraints constraints_;
  double kp_joint_ = 50.0;
  double tracking_weight_ = 10.0;
};
}  // namespace cyclo_teleoperation::controllers::common
