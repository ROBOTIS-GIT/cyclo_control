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
//
// Author: Yeonguk Kim

#pragma once

#include <memory>

#include "motion_controller_core/common/type_define.hpp"
#include "motion_controller_core/kinematics/kinematics_solver.hpp"
#include "motion_controller_core/optimization/qp_base.hpp"

namespace motion_controller
{
namespace controllers
{
class AIWorkerMoveJController : public motion_controller::optimization::QPBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AIWorkerMoveJController(
    std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data,
    const double dt);

  void setWeight(const Eigen::VectorXd & w_tracking, const Eigen::VectorXd & w_damping);
  void setDesiredJointVel(const Eigen::VectorXd & qdot_desired);
  void setControllerParams(
    const double slack_penalty, const double cbf_alpha,
    const double buffer_distance, const double safe_distance);
  bool getOptJointVel(Eigen::VectorXd & opt_qdot);

private:
  struct QPIndex
  {
    int qdot_start;
    int slack_q_min_start;
    int slack_q_max_start;
    int slack_sing_start;
    int slack_sel_col_start;

    int qdot_size;
    int slack_q_min_size;
    int slack_q_max_size;
    int slack_sing_size;
    int slack_sel_col_size;

    int con_q_min_start;
    int con_q_max_start;
    int con_sing_start;
    int con_sel_col_start;

    int con_q_min_size;
    int con_q_max_size;
    int con_sing_size;
    int con_sel_col_size;
  } si_index_;

  std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data_;
  double dt_;
  int joint_dof_;

  Eigen::VectorXd qdot_desired_;
  Eigen::VectorXd w_tracking_;
  Eigen::VectorXd w_damping_;
  double slack_penalty_;
  double cbf_alpha_;
  double collision_buffer_;
  double collision_safe_distance_;

  void setCost() override;
  void setBoundConstraint() override;
  void setIneqConstraint() override;
  void setEqConstraint() override;
};
}  // namespace controllers
}  // namespace motion_controller
