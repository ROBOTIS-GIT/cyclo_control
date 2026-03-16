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
#include <string>

#include "motion_controller_core/common/type_define.hpp"
#include "motion_controller_core/kinematics/kinematics_solver.hpp"
#include "motion_controller_core/optimization/qp_base.hpp"

namespace motion_controller
{
namespace controllers
{
class OpenManipulatorMoveLController : public motion_controller::optimization::QPBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OpenManipulatorMoveLController(
    std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data,
    const std::string & controlled_link,
    double dt);

  void setControlledLink(const std::string & controlled_link);
  const std::string & getControlledLink() const {return controlled_link_;}

  void setDesiredTaskVel(const motion_controller::common::Vector6d & task_xdot_desired);
  void setWeights(
    const motion_controller::common::Vector6d & task_tracking_weight,
    const Eigen::VectorXd & damping_weight);
  void setControllerParams(
    double slack_penalty,
    double cbf_alpha,
    double buffer_distance,
    double safe_distance);

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
  std::string controlled_link_;
  double dt_;
  int joint_dof_;

  motion_controller::common::Vector6d task_xdot_desired_;
  motion_controller::common::Vector6d task_tracking_weight_;
  Eigen::VectorXd damping_weight_;

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
