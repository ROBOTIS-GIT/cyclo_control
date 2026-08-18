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

#include "cyclo_teleoperation/robots/ai_worker/movej_mode.hpp"

#include <algorithm>
#include <cmath>

#include <pluginlib/class_list_macros.hpp>

namespace cyclo_teleoperation::robots::ai_worker
{
bool MoveJMode::configure(
  rclcpp::Node & node,
  const std::string & prefix,
  const ModeConfiguration & configuration)
{
  left_indices_ = configuration.left_arm_indices;
  right_indices_ = configuration.right_arm_indices;
  auto parameter = [&node](const std::string & name, const double default_value) {
      if (!node.has_parameter(name)) {
        return node.declare_parameter(name, default_value);
      }
      return node.get_parameter(name).as_double();
    };
  kp_joint_ = parameter(prefix + ".kp_joint", 50.0);
  tracking_weight_ = parameter(prefix + ".tracking_weight", 10.0);
  blend_duration_ = parameter(prefix + ".activation.duration", 1.5);
  return kp_joint_ > 0.0 && tracking_weight_ > 0.0 && blend_duration_ >= 0.0;
}

bool MoveJMode::activate(const ModeContext & context)
{
  blend_start_ = context.follower_position;
  left_blending_ = false;
  right_blending_ = false;
  onArmsEnabled(context.enabled_arms, context);
  return true;
}

void MoveJMode::beginBlend(
  const std::vector<int> & indices,
  const double now,
  const Eigen::VectorXd & follower)
{
  if (blend_start_.size() != follower.size()) {
    blend_start_ = follower;
  }
  for (const int index : indices) {
    blend_start_[index] = follower[index];
  }
  if (&indices == &left_indices_) {
    left_blend_start_time_ = now;
    left_blending_ = blend_duration_ > 0.0;
  } else {
    right_blend_start_time_ = now;
    right_blending_ = blend_duration_ > 0.0;
  }
}

void MoveJMode::onArmsEnabled(const uint8_t arms, const ModeContext & context)
{
  if ((arms & kLeftArm) != 0) {
    beginBlend(left_indices_, context.now_seconds, context.follower_position);
  }
  if ((arms & kRightArm) != 0) {
    beginBlend(right_indices_, context.now_seconds, context.follower_position);
  }
}

double MoveJMode::blendAlpha(const double elapsed) const
{
  if (blend_duration_ <= 0.0 || elapsed >= blend_duration_) {
    return 1.0;
  }
  const double x = std::clamp(elapsed / blend_duration_, 0.0, 1.0);
  return x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
}

bool MoveJMode::update(const ModeContext & context, ModeOutput & output)
{
  auto update_arm = [&](const std::vector<int> & indices, const bool enabled,
    bool & blending, const double start_time) {
      if (!enabled) {
        return;
      }
      const double alpha = blendAlpha(context.now_seconds - start_time);
      if (alpha >= 1.0) {
        blending = false;
      }
      for (const int index : indices) {
        const double reference =
          blending ? blend_start_[index] +
          alpha * (context.leader_reference[index] - blend_start_[index]) :
          context.leader_reference[index];
        output.desired_joint_velocity[index] =
          kp_joint_ * (reference - context.follower_position[index]);
        output.joint_tracking_weight[index] = tracking_weight_;
      }
    };

  update_arm(
    left_indices_, (context.enabled_arms & kLeftArm) != 0,
    left_blending_, left_blend_start_time_);
  update_arm(
    right_indices_, (context.enabled_arms & kRightArm) != 0,
    right_blending_, right_blend_start_time_);
  return true;
}
}  // namespace cyclo_teleoperation::robots::ai_worker

PLUGINLIB_EXPORT_CLASS(
  cyclo_teleoperation::robots::ai_worker::MoveJMode,
  cyclo_teleoperation::TeleoperationMode)
