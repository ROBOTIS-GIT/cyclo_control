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
#include <memory>
#include <utility>
#include <rclcpp/rclcpp.hpp>
#include "cyclo_teleoperation/core/teleoperation_qp.hpp"

namespace cyclo_teleoperation
{
// Shared execution of the fully assembled whole-robot objective. State is integrated
// open-loop; the caller explicitly rebases it only at enable/recovery/ownership events.
class ControlRuntime
{
public:
  ControlRuntime(
    rclcpp::Node & node,
    std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> robot)
  : node_(node), qp_(std::move(robot)) {}
  bool step(
    const ModeOutput & output, double dt, Eigen::VectorXd & position,
    Eigen::VectorXd & velocity)
  {
    qp_.setModeOutput(output);
    qp_.setControllerParameters(
      node_.get_parameter("constraints.slack_penalty").as_double(),
      node_.get_parameter("constraints.cbf_alpha").as_double(),
      node_.get_parameter("constraints.collision_buffer").as_double(),
      node_.get_parameter("constraints.collision_safe_distance").as_double());
    if (!qp_.solve(velocity)) {velocity.setZero(); return false;}
    position += dt * velocity;
    return true;
  }

private:
  rclcpp::Node & node_;
  TeleoperationQP qp_;
};
}  // namespace cyclo_teleoperation
