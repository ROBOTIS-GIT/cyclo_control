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

namespace cyclo_teleoperation
{
struct FixedDurationSlowStartSample
{
  Eigen::VectorXd position;
  Eigen::VectorXd velocity;
  bool active = false;
  bool complete = false;
};

class FixedDurationSlowStart
{
public:
  void reset();
  void restart(uint64_t current_sequence);

  FixedDurationSlowStartSample update(
    const Eigen::VectorXd & start_position,
    const Eigen::VectorXd & current_reference,
    uint64_t command_sequence,
    double command_duration,
    double current_time);

  bool waitingForCommand() const {return waiting_for_command_;}
  bool active() const {return active_;}
  double duration() const {return duration_;}

private:
  Eigen::VectorXd start_position_;
  uint64_t last_sequence_ = 0;
  double start_time_ = 0.0;
  double duration_ = 0.0;
  bool waiting_for_command_ = false;
  bool active_ = false;
};
}  // namespace cyclo_teleoperation
