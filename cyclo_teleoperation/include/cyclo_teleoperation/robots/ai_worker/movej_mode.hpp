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
#include <vector>

#include "cyclo_teleoperation/core/teleoperation_mode.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
class MoveJMode : public TeleoperationMode
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
  void beginBlend(
    const std::vector<int> & indices,
    double now,
    const Eigen::VectorXd & follower);
  double blendAlpha(double elapsed) const;

  std::vector<int> left_indices_;
  std::vector<int> right_indices_;
  Eigen::VectorXd blend_start_;
  double left_blend_start_time_ = 0.0;
  double right_blend_start_time_ = 0.0;
  bool left_blending_ = false;
  bool right_blending_ = false;
  double kp_joint_ = 50.0;
  double tracking_weight_ = 10.0;
  double blend_duration_ = 1.5;
};
}  // namespace cyclo_teleoperation::robots::ai_worker
