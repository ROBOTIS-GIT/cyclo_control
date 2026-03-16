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

#include "motion_controller_core/controllers/open_manipulator/open_manipulator_controller.hpp"

#include <algorithm>
#include <stdexcept>

namespace motion_controller
{
namespace controllers
{
OpenManipulatorController::OpenManipulatorController(
  std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data,
  const std::string & controlled_link,
  double dt)
: motion_controller::optimization::QPBase(),
  robot_data_(std::move(robot_data)),
  controlled_link_(controlled_link),
  dt_(dt),
  joint_dof_(0),
  slack_penalty_(1000.0),
  cbf_alpha_(5.0),
  collision_buffer_(0.05),
  collision_safe_distance_(0.02)
{
  joint_dof_ = robot_data_->getDof();

  if (!controlled_link_.empty() && !robot_data_->hasLinkFrame(controlled_link_)) {
    throw std::runtime_error(
                "Controlled link '" + controlled_link_ + "' does not exist in the model.");
  }

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

  QPBase::setQPsize(nx, nbound, nineq, 0);

  si_index_.qdot_start = 0;
  si_index_.slack_q_min_start = si_index_.qdot_start + si_index_.qdot_size;
  si_index_.slack_q_max_start = si_index_.slack_q_min_start + si_index_.slack_q_min_size;
  si_index_.slack_sing_start = si_index_.slack_q_max_start + si_index_.slack_q_max_size;
  si_index_.slack_sel_col_start = si_index_.slack_sing_start + si_index_.slack_sing_size;
  si_index_.con_q_min_start = 0;
  si_index_.con_q_max_start = si_index_.con_q_min_start + si_index_.con_q_min_size;
  si_index_.con_sing_start = si_index_.con_q_max_start + si_index_.con_q_max_size;
  si_index_.con_sel_col_start = si_index_.con_sing_start + si_index_.con_sing_size;

  task_xdot_desired_.setZero();
  task_tracking_weight_.setOnes();
  damping_weight_.setOnes(joint_dof_);
}

void OpenManipulatorController::setControlledLink(const std::string & controlled_link)
{
  if (!controlled_link.empty() && !robot_data_->hasLinkFrame(controlled_link)) {
    throw std::runtime_error(
                "Controlled link '" + controlled_link + "' does not exist in the model.");
  }
  controlled_link_ = controlled_link;
}

void OpenManipulatorController::setDesiredTaskVel(
  const motion_controller::common::Vector6d & task_xdot_desired)
{
  task_xdot_desired_ = task_xdot_desired;
}

void OpenManipulatorController::setWeights(
  const motion_controller::common::Vector6d & task_tracking_weight,
  const Eigen::VectorXd & damping_weight)
{
  task_tracking_weight_ = task_tracking_weight;
  if (damping_weight.size() == joint_dof_) {
    damping_weight_ = damping_weight;
  }
}

void OpenManipulatorController::setControllerParams(
  double slack_penalty,
  double cbf_alpha,
  double buffer_distance,
  double safe_distance)
{
  slack_penalty_ = slack_penalty;
  cbf_alpha_ = cbf_alpha;
  collision_buffer_ = buffer_distance;
  collision_safe_distance_ = safe_distance;
}

bool OpenManipulatorController::getOptJointVel(Eigen::VectorXd & opt_qdot)
{
  Eigen::MatrixXd sol;
  if (!solveQP(sol)) {
    opt_qdot.setZero(joint_dof_);
    return false;
  }

  opt_qdot = sol.block(si_index_.qdot_start, 0, si_index_.qdot_size, 1);
  return true;
}

void OpenManipulatorController::setCost()
{
  P_ds_.setZero(nx_, nx_);
  q_ds_.setZero(nx_);

  if (!controlled_link_.empty()) {
    const Eigen::MatrixXd jacobian = robot_data_->getJacobian(controlled_link_);
    const Eigen::Matrix<double, 6, 6> w_task = task_tracking_weight_.asDiagonal();
    P_ds_.block(
                si_index_.qdot_start,
                si_index_.qdot_start,
                si_index_.qdot_size,
                si_index_.qdot_size) += 2.0 * jacobian.transpose() * w_task * jacobian;
    q_ds_.segment(si_index_.qdot_start, si_index_.qdot_size) +=
      -2.0 * jacobian.transpose() * w_task * task_xdot_desired_;
  }

  P_ds_.block(
            si_index_.qdot_start,
            si_index_.qdot_start,
            si_index_.qdot_size,
            si_index_.qdot_size) += 2.0 * damping_weight_.asDiagonal();

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

void OpenManipulatorController::setBoundConstraint()
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
    l_bound_ds_.segment(si_index_.slack_sel_col_start, si_index_.slack_sel_col_size)
    .setZero();
  }
}

void OpenManipulatorController::setIneqConstraint()
{
  A_ineq_ds_.setZero(nineqc_, nx_);
  l_ineq_ds_.setConstant(nineqc_, -OSQP_INFTY);
  u_ineq_ds_.setConstant(nineqc_, OSQP_INFTY);

  const Eigen::VectorXd q_min = robot_data_->getJointPositionLimit().first;
  const Eigen::VectorXd q_max = robot_data_->getJointPositionLimit().second;
  const Eigen::VectorXd q = robot_data_->getJointPosition();

  A_ineq_ds_.block(
            si_index_.con_q_min_start,
            si_index_.qdot_start,
            si_index_.con_q_min_size,
            si_index_.qdot_size) =
    Eigen::MatrixXd::Identity(si_index_.con_q_min_size, si_index_.qdot_size);
  A_ineq_ds_.block(
            si_index_.con_q_min_start,
            si_index_.slack_q_min_start,
            si_index_.con_q_min_size,
            si_index_.slack_q_min_size) =
    Eigen::MatrixXd::Identity(si_index_.con_q_min_size, si_index_.slack_q_min_size);
  l_ineq_ds_.segment(si_index_.con_q_min_start, si_index_.con_q_min_size) =
    -cbf_alpha_ * (q - q_min);

  A_ineq_ds_.block(
            si_index_.con_q_max_start,
            si_index_.qdot_start,
            si_index_.con_q_max_size,
            si_index_.qdot_size) =
    -Eigen::MatrixXd::Identity(si_index_.con_q_max_size, si_index_.qdot_size);
  A_ineq_ds_.block(
            si_index_.con_q_max_start,
            si_index_.slack_q_max_start,
            si_index_.con_q_max_size,
            si_index_.slack_q_max_size) =
    Eigen::MatrixXd::Identity(si_index_.con_q_max_size, si_index_.slack_q_max_size);
  l_ineq_ds_.segment(si_index_.con_q_max_start, si_index_.con_q_max_size) =
    -cbf_alpha_ * (q_max - q);

  if (si_index_.con_sel_col_size > 0) {
    const auto pair_results = robot_data_->getCollisionPairDistances(true, false, false);
    const int pair_count = std::min<int>(si_index_.con_sel_col_size, pair_results.size());
    for (int i = 0; i < pair_count; ++i) {
      const auto & res = pair_results[i];
      A_ineq_ds_.block(
                    si_index_.con_sel_col_start + i,
                    si_index_.qdot_start,
                    1,
                    si_index_.qdot_size) = res.grad.transpose();
      if (si_index_.slack_sel_col_size > 0 && i < si_index_.slack_sel_col_size) {
        A_ineq_ds_(
                        si_index_.con_sel_col_start + i,
                        si_index_.slack_sel_col_start + i) = 1.0;
      }
      if (res.distance <= collision_buffer_) {
        l_ineq_ds_(si_index_.con_sel_col_start + i) =
          -cbf_alpha_ * (res.distance - collision_safe_distance_);
      }
    }
  }
}

void OpenManipulatorController::setEqConstraint()
{
  A_eq_ds_.setZero(neqc_, nx_);
  b_eq_ds_.setZero(neqc_);
}
}  // namespace controllers
}  // namespace motion_controller
