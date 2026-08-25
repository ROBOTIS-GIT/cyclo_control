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
#include <Eigen/Geometry>

#include <string>
#include <unordered_map>

#include "cyclo_teleoperation/core/teleoperation_mode.hpp"

namespace cyclo_teleoperation::controllers::common
{
class RelativePoseMode : public TeleoperationMode
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
  struct Anchor
  {
    Eigen::Affine3d leader = Eigen::Affine3d::Identity();
    Eigen::Affine3d follower = Eigen::Affine3d::Identity();
    bool valid = false;
  };

  void captureAnchors(ControlGroupMask groups);
  Eigen::Matrix<double, 6, 1> desiredVelocity(
    const Eigen::Affine3d & current,
    const Eigen::Affine3d & goal) const;

  ModeConfiguration configuration_;
  std::unordered_map<ControlGroupId, Anchor> anchors_;

  double kp_position_ = 50.0;
  double kp_orientation_ = 50.0;
  double weight_position_ = 10.0;
  double weight_orientation_ = 1.0;
};
}  // namespace cyclo_teleoperation::controllers::common
