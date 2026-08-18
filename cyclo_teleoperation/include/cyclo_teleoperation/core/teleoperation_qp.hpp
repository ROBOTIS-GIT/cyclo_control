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

#include <memory>

#include "cyclo_teleoperation/core/types.hpp"
#include "kinematics/kinematics_solver.hpp"
#include "optimization/qp_base.hpp"

namespace cyclo_teleoperation
{
class TeleoperationQP : public cyclo_motion_controller::optimization::QPBase
{
public:
  explicit TeleoperationQP(
    std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> robot);

  void setModeOutput(const ModeOutput & output);
  void setControllerParameters(
    double slack_penalty,
    double cbf_alpha,
    double collision_buffer,
    double collision_safe_distance);
  bool solve(Eigen::VectorXd & velocity);

private:
  struct Index
  {
    int qdot_start = 0;
    int slack_q_min_start = 0;
    int slack_q_max_start = 0;
    int slack_collision_start = 0;
    int con_q_min_start = 0;
    int con_q_max_start = 0;
    int con_collision_start = 0;
    int qdot_size = 0;
    int collision_size = 0;
  } index_;

  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> robot_;
  ModeOutput output_;
  double slack_penalty_ = 1000.0;
  double cbf_alpha_ = 50.0;
  double collision_buffer_ = 0.05;
  double collision_safe_distance_ = 0.02;

  void setCost() override;
  void setBoundConstraint() override;
  void setIneqConstraint() override;
  void setEqConstraint() override;
};
}  // namespace cyclo_teleoperation
