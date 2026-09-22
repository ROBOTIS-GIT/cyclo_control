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

#include "cyclo_teleoperation/core/joint_trajectory_interpolator.hpp"

#include <algorithm>
#include <cmath>

namespace cyclo_teleoperation
{
void JointTrajectoryInterpolator::reset()
{
  start_position_.resize(0);
  target_position_.resize(0);
  start_time_ = 0.0;
  duration_ = 0.0;
  active_ = false;
}

bool JointTrajectoryInterpolator::start(
  const Eigen::VectorXd & start_position,
  const Eigen::VectorXd & target_position,
  const double start_time,
  const double duration)
{
  if (
    start_position.size() == 0 || start_position.size() != target_position.size() ||
    !start_position.allFinite() || !target_position.allFinite() ||
    !std::isfinite(start_time) || !std::isfinite(duration) || duration <= 0.0)
  {
    reset();
    return false;
  }
  start_position_ = start_position;
  target_position_ = target_position;
  start_time_ = start_time;
  duration_ = duration;
  active_ = true;
  return true;
}

JointTrajectorySample JointTrajectoryInterpolator::sample(const double current_time) const
{
  JointTrajectorySample result;
  if (!active_) {
    return result;
  }
  const double elapsed = std::max(0.0, current_time - start_time_);
  const double alpha = std::clamp(elapsed / duration_, 0.0, 1.0);
  result.position = start_position_ + alpha * (target_position_ - start_position_);
  if (alpha < 1.0) {
    result.velocity = (target_position_ - start_position_) / duration_;
  } else {
    result.velocity.setZero(start_position_.size());
  }
  result.active = alpha < 1.0;
  result.complete = alpha >= 1.0;
  return result;
}

void applyJointTrajectoryTracking(
  const ControlGroupConfiguration & group,
  const JointTrajectorySample & sample,
  const Eigen::VectorXd & current_position,
  const double kp,
  const double tracking_weight,
  ModeOutput & output)
{
  if (
    sample.position.size() != static_cast<Eigen::Index>(group.follower_joint_indices.size()) ||
    sample.velocity.size() != sample.position.size())
  {
    return;
  }
  for (size_t i = 0; i < group.follower_joint_indices.size(); ++i) {
    const int index = group.follower_joint_indices[i];
    output.desired_joint_velocity[index] = sample.velocity[i] +
      kp * (sample.position[i] - current_position[index]);
    output.joint_tracking_weight[index] = tracking_weight;
  }
}
}  // namespace cyclo_teleoperation
