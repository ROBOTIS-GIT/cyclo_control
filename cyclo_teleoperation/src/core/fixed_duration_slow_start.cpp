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

#include "cyclo_teleoperation/core/fixed_duration_slow_start.hpp"

#include <algorithm>
#include <cmath>

namespace cyclo_teleoperation
{
namespace
{
constexpr double kDurationEpsilon = 1e-6;
}

void FixedDurationSlowStart::reset()
{
  start_position_.resize(0);
  last_sequence_ = 0;
  start_time_ = 0.0;
  duration_ = 0.0;
  waiting_for_command_ = false;
  active_ = false;
}

void FixedDurationSlowStart::restart(const uint64_t current_sequence)
{
  reset();
  last_sequence_ = current_sequence;
  waiting_for_command_ = true;
}

FixedDurationSlowStartSample FixedDurationSlowStart::update(
  const Eigen::VectorXd & start_position,
  const Eigen::VectorXd & current_reference,
  const uint64_t command_sequence,
  const double command_duration,
  const double current_time)
{
  FixedDurationSlowStartSample sample;
  if (
    start_position.size() == 0 || start_position.size() != current_reference.size() ||
    !start_position.allFinite() || !current_reference.allFinite() ||
    !std::isfinite(command_duration) || command_duration < 0.0 ||
    !std::isfinite(current_time))
  {
    return sample;
  }

  if (waiting_for_command_ && command_sequence != last_sequence_) {
    last_sequence_ = command_sequence;
    waiting_for_command_ = false;
    if (command_duration > kDurationEpsilon) {
      start_position_ = start_position;
      start_time_ = current_time;
      duration_ = command_duration;
      active_ = true;
    }
  } else if (!waiting_for_command_ && command_sequence != last_sequence_) {
    last_sequence_ = command_sequence;
  }

  if (waiting_for_command_) {
    sample.position = start_position;
    sample.velocity.setZero(start_position.size());
    return sample;
  }

  if (!active_) {
    sample.position = current_reference;
    sample.velocity.setZero(current_reference.size());
    sample.complete = true;
    return sample;
  }

  const double elapsed = std::max(0.0, current_time - start_time_);
  const double alpha = std::clamp(elapsed / duration_, 0.0, 1.0);
  sample.position = (1.0 - alpha) * start_position_ + alpha * current_reference;
  if (alpha < 1.0) {
    sample.velocity = (current_reference - start_position_) / duration_;
    sample.active = true;
  } else {
    sample.velocity.setZero(current_reference.size());
    sample.complete = true;
    active_ = false;
  }
  return sample;
}
}  // namespace cyclo_teleoperation
