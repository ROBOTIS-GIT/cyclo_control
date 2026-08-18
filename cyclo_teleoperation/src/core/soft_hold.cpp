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

#include "cyclo_teleoperation/core/soft_hold.hpp"

#include <algorithm>
#include <stdexcept>

namespace cyclo_teleoperation
{
void applySoftHold(
  ModeOutput & output,
  const Eigen::VectorXd & command_position,
  const Eigen::VectorXd & hold_target,
  const std::vector<int> & left_arm_indices,
  const std::vector<int> & right_arm_indices,
  const uint8_t controlled_arms,
  const double kp,
  const double max_correction_velocity,
  const double tracking_weight)
{
  const int dof = command_position.size();
  if (
    hold_target.size() != dof ||
    output.desired_joint_velocity.size() != dof ||
    output.joint_tracking_weight.size() != dof)
  {
    throw std::invalid_argument("Soft hold vectors must match the follower DOF");
  }
  if (kp < 0.0 || max_correction_velocity < 0.0 || tracking_weight <= 0.0) {
    throw std::invalid_argument("Soft hold parameters are invalid");
  }

  std::vector<bool> controlled_joint(dof, false);
  auto mark_controlled = [&](const std::vector<int> & indices, const uint8_t arm) {
      if ((controlled_arms & arm) == 0) {
        return;
      }
      for (const int index : indices) {
        if (index < 0 || index >= dof) {
          throw std::out_of_range("Soft hold joint index is outside the follower DOF");
        }
        controlled_joint[index] = true;
      }
    };
  mark_controlled(left_arm_indices, kLeftArm);
  mark_controlled(right_arm_indices, kRightArm);

  for (int index = 0; index < dof; ++index) {
    if (controlled_joint[index]) {
      continue;
    }
    output.desired_joint_velocity[index] = std::clamp(
      kp * (hold_target[index] - command_position[index]),
      -max_correction_velocity, max_correction_velocity);
    output.joint_tracking_weight[index] = tracking_weight;
  }
}
}  // namespace cyclo_teleoperation
