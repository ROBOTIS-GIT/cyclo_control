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

#include <cstdint>
#include <vector>

#include "cyclo_teleoperation/core/types.hpp"

namespace cyclo_teleoperation
{
void applySoftHold(
  ModeOutput & output,
  const Eigen::VectorXd & command_position,
  const Eigen::VectorXd & hold_target,
  const std::vector<int> & left_arm_indices,
  const std::vector<int> & right_arm_indices,
  uint8_t controlled_arms,
  double kp,
  double max_correction_velocity,
  double tracking_weight);
}  // namespace cyclo_teleoperation
