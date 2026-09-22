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

#include "cyclo_teleoperation/core/types.hpp"

namespace cyclo_teleoperation
{
struct JointTrajectorySample
{
  Eigen::VectorXd position;
  Eigen::VectorXd velocity;
  bool active = false;
  bool complete = false;
};

class JointTrajectoryInterpolator
{
public:
  void reset();
  bool start(
    const Eigen::VectorXd & start_position,
    const Eigen::VectorXd & target_position,
    double start_time,
    double duration);
  JointTrajectorySample sample(double current_time) const;

  bool active() const {return active_;}
  const Eigen::VectorXd & startPosition() const {return start_position_;}
  const Eigen::VectorXd & targetPosition() const {return target_position_;}
  double duration() const {return duration_;}

private:
  Eigen::VectorXd start_position_;
  Eigen::VectorXd target_position_;
  double start_time_ = 0.0;
  double duration_ = 0.0;
  bool active_ = false;
};

void applyJointTrajectoryTracking(
  const ControlGroupConfiguration & group,
  const JointTrajectorySample & sample,
  const Eigen::VectorXd & current_position,
  double kp,
  double tracking_weight,
  ModeOutput & output);
}  // namespace cyclo_teleoperation
