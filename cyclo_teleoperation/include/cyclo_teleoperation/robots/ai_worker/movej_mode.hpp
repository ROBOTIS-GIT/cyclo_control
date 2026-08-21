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
  uint8_t timedCommandFeedbackSyncArms(const ModeContext & context) const override;

private:
  struct ArmTrajectory
  {
    Eigen::VectorXd start;
    Eigen::VectorXd goal;
    double start_time = 0.0;
    double duration = 0.0;
    uint64_t last_sequence = 0;
    bool active = false;
    bool waiting_for_command = false;
    bool slow_start_complete = false;
  };

  void beginSlowStart(
    const std::vector<int> & indices,
    ArmTrajectory & trajectory,
    uint64_t command_sequence,
    const ModeContext & context);
  void updateArm(
    const std::vector<int> & indices,
    bool enabled,
    double command_duration,
    uint64_t command_sequence,
    ArmTrajectory & trajectory,
    const ModeContext & context,
    ModeOutput & output);

  std::vector<int> left_indices_;
  std::vector<int> right_indices_;
  ArmTrajectory left_trajectory_;
  ArmTrajectory right_trajectory_;
  double kp_joint_ = 50.0;
  double tracking_weight_ = 10.0;
};
}  // namespace cyclo_teleoperation::robots::ai_worker
