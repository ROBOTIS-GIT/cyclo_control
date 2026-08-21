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
#include <vector>

#include "cyclo_teleoperation/core/teleoperation_mode.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
class ElbowUpLeaderMode : public TeleoperationMode
{
public:
  bool configure(
    rclcpp::Node & node,
    const std::string & parameter_prefix,
    const ModeConfiguration & configuration) override;
  bool activate(const ModeContext & context) override;
  void onArmsEnabled(uint8_t arms, const ModeContext & context) override;
  bool update(const ModeContext & context, ModeOutput & output) override;

private:
  void captureAnchor(uint8_t arms);
  Eigen::Matrix<double, 6, 1> desiredVelocity(
    const Eigen::Affine3d & current,
    const Eigen::Affine3d & goal) const;
  Eigen::VectorXd elbowPreference(uint8_t enabled_arms) const;

  ModeConfiguration configuration_;
  Eigen::Affine3d left_leader_anchor_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d right_leader_anchor_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d left_follower_anchor_ = Eigen::Affine3d::Identity();
  Eigen::Affine3d right_follower_anchor_ = Eigen::Affine3d::Identity();
  bool left_anchor_valid_ = false;
  bool right_anchor_valid_ = false;

  double kp_position_ = 50.0;
  double kp_orientation_ = 50.0;
  double weight_position_ = 10.0;
  double weight_orientation_ = 1.0;
  double elbow_up_velocity_ = 0.2;
  double elbow_weight_ = 1.0;
  double nullspace_damping_ = 0.001;
  double elbow_up_joint_velocity_ = 1.0;
};
}  // namespace cyclo_teleoperation::robots::ai_worker
