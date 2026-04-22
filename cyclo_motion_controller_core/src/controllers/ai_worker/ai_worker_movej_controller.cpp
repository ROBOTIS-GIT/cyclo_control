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
// This file is derived from `dyros_robot_controller` project:
// https://github.com/JunHeonYoon/dyros_robot_controller
//
// Original work Copyright (c) 2025 JunHeonYoon, licensed under the
// Apache License 2.0. Modifications in this file are Copyright 2026
// ROBOTIS CO., LTD.
//
// Author: Yeonguk Kim

#include "controllers/ai_worker/ai_worker_movej_controller.hpp"

#include <algorithm>

namespace cyclo_motion_controller
{
namespace controllers
{

AIWorkerMoveJController::AIWorkerMoveJController(
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> robot_data,
  const double dt)
: cyclo_motion_controller::optimization::QPBase(), robot_data_(robot_data), dt_(dt)
{
  joint_dof_ = robot_data_->getDof();

  si_index_.qdot_size = joint_dof_;
  si_index_.slack_q_min_size = joint_dof_;
  si_index_.slack_q_max_size = joint_dof_;
  si_index_.slack_sing_size = 1;
  si_index_.slack_sel_col_size = robot_data_->getCollisionPairCount();
  si_index_.con_q_min_size = joint_dof_;
  si_index_.con_q_max_size = joint_dof_;
  si_index_.con_sing_size = 1;
  si_index_.con_sel_col_size = robot_data_->getCollisionPairCount();

  const int nx = si_index_.qdot_size +
    si_index_.slack_q_min_size +
    si_index_.slack_q_max_size +
    si_index_.slack_sing_size +
    si_index_.slack_sel_col_size;
  const int nbound = nx;
  const int nineq = si_index_.con_q_min_size +
    si_index_.con_q_max_size +
    si_index_.con_sing_size +
    si_index_.con_sel_col_size;
  const int neq = 0;

  QPBase::setQPsize(nx, nbound, nineq, neq);

  si_index_.qdot_start = 0;
  si_index_.slack_q_min_start = si_index_.qdot_start + si_index_.qdot_size;
  si_index_.slack_q_max_start = si_index_.slack_q_min_start + si_index_.slack_q_min_size;
  si_index_.slack_sing_start = si_index_.slack_q_max_start + si_index_.slack_q_max_size;
  si_index_.slack_sel_col_start = si_index_.slack_sing_start + si_index_.slack_sing_size;
  si_index_.con_q_min_start = 0;
  si_index_.con_q_max_start = si_index_.con_q_min_start + si_index_.con_q_min_size;
  si_index_.con_sing_start = si_index_.con_q_max_start + si_index_.con_q_max_size;
  si_index_.con_sel_col_start = si_index_.con_sing_start + si_index_.con_sing_size;

  qdot_desired_.setZero(joint_dof_);
  w_tracking_.setOnes(joint_dof_);
  w_damping_.setOnes(joint_dof_);
  slack_penalty_ = 1000.0;
  cbf_alpha_ = 5.0;
  collision_buffer_ = 0.05;
  collision_safe_distance_ = 0.02;
}

void AIWorkerMoveJController::setDesiredJointVel(const Eigen::VectorXd & qdot_desired)
{
  if (qdot_desired.size() == joint_dof_) {
    qdot_desired_ = qdot_desired;
  }
}

bool AIWorkerMoveJController::getOptJointVel(Eigen::VectorXd & opt_qdot)
{
  Eigen::MatrixXd sol;
  if (!solveQP(sol)) {
    opt_qdot.setZero();
    return false;
  } else {
    opt_qdot = sol.block(si_index_.qdot_start, 0, si_index_.qdot_size, 1);
    return true;
  }
}

void AIWorkerMoveJController::setWeight(
  const Eigen::VectorXd & w_tracking, const Eigen::VectorXd & w_damping)
{
  if (w_tracking.size() == joint_dof_) {
    w_tracking_ = w_tracking;
  }
  if (w_damping.size() == joint_dof_) {
    w_damping_ = w_damping;
  }
}

void AIWorkerMoveJController::setControllerParams(
  const double slack_penalty, const double cbf_alpha,
  const double buffer_distance, const double safe_distance)
{
  slack_penalty_ = slack_penalty;
  cbf_alpha_ = cbf_alpha;
  collision_buffer_ = buffer_distance;
  collision_safe_distance_ = safe_distance;
}

void AIWorkerMoveJController::setCost()
{
  P_ds_.setZero(nx_, nx_);
  q_ds_.setZero(nx_);

  P_ds_.block(
    si_index_.qdot_start, si_index_.qdot_start,
    si_index_.qdot_size, si_index_.qdot_size) += 2.0 * w_tracking_.asDiagonal();
  q_ds_.segment(si_index_.qdot_start, si_index_.qdot_size) +=
    -2.0 * w_tracking_.asDiagonal() * qdot_desired_;

  P_ds_.block(
    si_index_.qdot_start, si_index_.qdot_start,
    si_index_.qdot_size, si_index_.qdot_size) += 2.0 * w_damping_.asDiagonal();

  q_ds_.segment(si_index_.slack_q_min_start, si_index_.slack_q_min_size) =
    Eigen::VectorXd::Constant(si_index_.slack_q_min_size, slack_penalty_);
  q_ds_.segment(si_index_.slack_q_max_start, si_index_.slack_q_max_size) =
    Eigen::VectorXd::Constant(si_index_.slack_q_max_size, slack_penalty_);
  q_ds_(si_index_.slack_sing_start) = slack_penalty_;
  if (si_index_.slack_sel_col_size > 0) {
    q_ds_.segment(si_index_.slack_sel_col_start, si_index_.slack_sel_col_size) =
      Eigen::VectorXd::Constant(si_index_.slack_sel_col_size, slack_penalty_);
  }
}

void AIWorkerMoveJController::setBoundConstraint()
{
  l_bound_ds_.setConstant(nbc_, -OSQP_INFTY);
  u_bound_ds_.setConstant(nbc_, OSQP_INFTY);

  l_bound_ds_.segment(si_index_.qdot_start, si_index_.qdot_size) =
    robot_data_->getJointVelocityLimit().first;
  u_bound_ds_.segment(si_index_.qdot_start, si_index_.qdot_size) =
    robot_data_->getJointVelocityLimit().second;

  l_bound_ds_.segment(si_index_.slack_q_min_start, si_index_.slack_q_min_size).setZero();
  l_bound_ds_.segment(si_index_.slack_q_max_start, si_index_.slack_q_max_size).setZero();
  l_bound_ds_(si_index_.slack_sing_start) = 0.0;
  if (si_index_.slack_sel_col_size > 0) {
    l_bound_ds_.segment(si_index_.slack_sel_col_start, si_index_.slack_sel_col_size).setZero();
  }
}

void AIWorkerMoveJController::setIneqConstraint()
{
  A_ineq_ds_.setZero(nineqc_, nx_);
  l_ineq_ds_.setConstant(nineqc_, -OSQP_INFTY);
  u_ineq_ds_.setConstant(nineqc_, OSQP_INFTY);

  const Eigen::VectorXd q_min = robot_data_->getJointPositionLimit().first;
  const Eigen::VectorXd q_max = robot_data_->getJointPositionLimit().second;
  const Eigen::VectorXd q = robot_data_->getJointPosition();

  A_ineq_ds_.block(
    si_index_.con_q_min_start, si_index_.qdot_start,
    si_index_.con_q_min_size, si_index_.qdot_size) =
    Eigen::MatrixXd::Identity(si_index_.con_q_min_size, si_index_.qdot_size);
  A_ineq_ds_.block(
    si_index_.con_q_min_start, si_index_.slack_q_min_start,
    si_index_.con_q_min_size, si_index_.slack_q_min_size) =
    Eigen::MatrixXd::Identity(si_index_.con_q_min_size, si_index_.slack_q_min_size);
  l_ineq_ds_.segment(si_index_.con_q_min_start, si_index_.con_q_min_size) =
    -cbf_alpha_ * (q - q_min);

  A_ineq_ds_.block(
    si_index_.con_q_max_start, si_index_.qdot_start,
    si_index_.con_q_max_size, si_index_.qdot_size) =
    -Eigen::MatrixXd::Identity(si_index_.con_q_max_size, si_index_.qdot_size);
  A_ineq_ds_.block(
    si_index_.con_q_max_start, si_index_.slack_q_max_start,
    si_index_.con_q_max_size, si_index_.slack_q_max_size) =
    Eigen::MatrixXd::Identity(si_index_.con_q_max_size, si_index_.slack_q_max_size);
  l_ineq_ds_.segment(si_index_.con_q_max_start, si_index_.con_q_max_size) =
    -cbf_alpha_ * (q_max - q);

  if (si_index_.con_sel_col_size > 0) {
    const auto pair_results = robot_data_->getCollisionPairDistances(true, false, false);
    const int pair_count = std::min<int>(si_index_.con_sel_col_size, pair_results.size());
    for (int i = 0; i < pair_count; ++i) {
      const auto & res = pair_results[i];
      A_ineq_ds_.block(si_index_.con_sel_col_start + i, si_index_.qdot_start, 1,
            si_index_.qdot_size) =
        res.grad.transpose();
      if (si_index_.slack_sel_col_size > 0 && i < si_index_.slack_sel_col_size) {
        A_ineq_ds_(si_index_.con_sel_col_start + i, si_index_.slack_sel_col_start + i) = 1.0;
      }
      if (res.distance <= collision_buffer_) {
        l_ineq_ds_(si_index_.con_sel_col_start + i) =
          -cbf_alpha_ * (res.distance - collision_safe_distance_);
      }
    }
  }
}

void AIWorkerMoveJController::setEqConstraint()
{
  A_eq_ds_.setZero(neqc_, nx_);
  b_eq_ds_.setZero(neqc_);
}

}  // namespace controllers
}  // namespace cyclo_motion_controller
