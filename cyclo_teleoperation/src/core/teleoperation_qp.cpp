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

#include "cyclo_teleoperation/core/teleoperation_qp.hpp"

#include <algorithm>
#include <utility>

namespace cyclo_teleoperation
{
TeleoperationQP::TeleoperationQP(
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> robot)
: robot_(std::move(robot))
{
  const int dof = robot_->getDof();
  const int collision_count = robot_->getCollisionPairCount();
  index_.qdot_size = dof;
  index_.collision_size = collision_count;
  index_.slack_q_min_start = dof;
  index_.slack_q_max_start = index_.slack_q_min_start + dof;
  index_.slack_collision_start = index_.slack_q_max_start + dof;
  index_.con_q_max_start = dof;
  index_.con_collision_start = index_.con_q_max_start + dof;

  const int nx = dof * 3 + collision_count;
  const int nineq = dof * 2 + collision_count;
  QPBase::setQPsize(nx, nx, nineq, 0);

  output_.reset(dof, 0.1);
}

void TeleoperationQP::setModeOutput(const ModeOutput & output)
{
  output_ = output;
}

void TeleoperationQP::setControllerParameters(
  const double slack_penalty,
  const double cbf_alpha,
  const double collision_buffer,
  const double collision_safe_distance)
{
  slack_penalty_ = slack_penalty;
  cbf_alpha_ = cbf_alpha;
  collision_buffer_ = collision_buffer;
  collision_safe_distance_ = collision_safe_distance;
}

bool TeleoperationQP::solve(Eigen::VectorXd & velocity)
{
  Eigen::MatrixXd solution;
  if (!solveQP(solution)) {
    velocity.setZero(index_.qdot_size);
    return false;
  }
  velocity = solution.block(index_.qdot_start, 0, index_.qdot_size, 1);
  return true;
}

void TeleoperationQP::setCost()
{
  P_ds_.setZero(nx_, nx_);
  q_ds_.setZero(nx_);

  if (
    output_.desired_joint_velocity.size() == index_.qdot_size &&
    output_.joint_tracking_weight.size() == index_.qdot_size)
  {
    P_ds_.block(0, 0, index_.qdot_size, index_.qdot_size) +=
      2.0 * output_.joint_tracking_weight.asDiagonal();
    q_ds_.head(index_.qdot_size) +=
      -2.0 * output_.joint_tracking_weight.asDiagonal() *
      output_.desired_joint_velocity;
  }

  for (const auto & task : output_.task_objectives) {
    const Eigen::MatrixXd jacobian = robot_->getJacobian(task.link_name);
    P_ds_.block(0, 0, index_.qdot_size, index_.qdot_size) +=
      2.0 * jacobian.transpose() * task.weight.asDiagonal() * jacobian;
    q_ds_.head(index_.qdot_size) +=
      -2.0 * jacobian.transpose() * task.weight.asDiagonal() *
      task.desired_velocity;
  }

  for (const auto & task : output_.linear_task_objectives) {
    if (
      task.jacobian.cols() != index_.qdot_size ||
      task.jacobian.rows() != task.desired_velocity.size() ||
      task.jacobian.rows() != task.weight.size() ||
      task.weight.size() == 0 || task.weight.minCoeff() < 0.0)
    {
      continue;
    }
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic> weight =
      task.weight.asDiagonal();
    P_ds_.block(0, 0, index_.qdot_size, index_.qdot_size) +=
      2.0 * task.jacobian.transpose() * weight * task.jacobian;
    q_ds_.head(index_.qdot_size) +=
      -2.0 * task.jacobian.transpose() * weight * task.desired_velocity;
  }

  if (output_.damping_weight.size() == index_.qdot_size) {
    P_ds_.block(0, 0, index_.qdot_size, index_.qdot_size) +=
      2.0 * output_.damping_weight.asDiagonal();
  }

  if (
    output_.preferred_joint_velocity.size() == index_.qdot_size &&
    output_.preferred_joint_velocity_weight > 0.0)
  {
    P_ds_.block(0, 0, index_.qdot_size, index_.qdot_size).diagonal().array() +=
      2.0 * output_.preferred_joint_velocity_weight;
    q_ds_.head(index_.qdot_size) +=
      -2.0 * output_.preferred_joint_velocity_weight *
      output_.preferred_joint_velocity;
  }

  q_ds_.segment(index_.slack_q_min_start, index_.qdot_size).setConstant(slack_penalty_);
  q_ds_.segment(index_.slack_q_max_start, index_.qdot_size).setConstant(slack_penalty_);
  if (index_.collision_size > 0) {
    q_ds_.segment(index_.slack_collision_start, index_.collision_size).setConstant(
      slack_penalty_);
  }
}

void TeleoperationQP::setBoundConstraint()
{
  l_bound_ds_.setConstant(nbc_, -OSQP_INFTY);
  u_bound_ds_.setConstant(nbc_, OSQP_INFTY);
  l_bound_ds_.head(index_.qdot_size) = robot_->getJointVelocityLimit().first;
  u_bound_ds_.head(index_.qdot_size) = robot_->getJointVelocityLimit().second;
  l_bound_ds_.segment(index_.slack_q_min_start, index_.qdot_size).setZero();
  l_bound_ds_.segment(index_.slack_q_max_start, index_.qdot_size).setZero();
  if (index_.collision_size > 0) {
    l_bound_ds_.segment(index_.slack_collision_start, index_.collision_size).setZero();
  }
}

void TeleoperationQP::setIneqConstraint()
{
  A_ineq_ds_.setZero(nineqc_, nx_);
  l_ineq_ds_.setConstant(nineqc_, -OSQP_INFTY);
  u_ineq_ds_.setConstant(nineqc_, OSQP_INFTY);

  const Eigen::VectorXd q = robot_->getJointPosition();
  const auto limits = robot_->getJointPositionLimit();

  for (int i = 0; i < index_.qdot_size; ++i) {
    const bool position_limit_enabled =
      output_.joint_position_limit_enabled.size() !=
      static_cast<size_t>(index_.qdot_size) ||
      output_.joint_position_limit_enabled[i];
    if (!position_limit_enabled) {
      continue;
    }
    A_ineq_ds_(i, index_.qdot_start + i) = 1.0;
    A_ineq_ds_(i, index_.slack_q_min_start + i) = 1.0;
    l_ineq_ds_(i) = -cbf_alpha_ * (q[i] - limits.first[i]);

    const int maximum_row = index_.con_q_max_start + i;
    A_ineq_ds_(maximum_row, index_.qdot_start + i) = -1.0;
    A_ineq_ds_(maximum_row, index_.slack_q_max_start + i) = 1.0;
    l_ineq_ds_(maximum_row) = -cbf_alpha_ * (limits.second[i] - q[i]);
  }

  const auto distances = robot_->getCollisionPairDistances(true, false, false);
  const int count = std::min<int>(index_.collision_size, distances.size());
  for (int i = 0; i < count; ++i) {
    A_ineq_ds_.block(
      index_.con_collision_start + i, 0, 1, index_.qdot_size) =
      distances[i].grad.transpose();
    A_ineq_ds_(
      index_.con_collision_start + i,
      index_.slack_collision_start + i) = 1.0;
    if (distances[i].distance <= collision_buffer_) {
      l_ineq_ds_(index_.con_collision_start + i) =
        -cbf_alpha_ * (distances[i].distance - collision_safe_distance_);
    }
  }
}

void TeleoperationQP::setEqConstraint()
{
  A_eq_ds_.setZero(neqc_, nx_);
  b_eq_ds_.setZero(neqc_);
}
}  // namespace cyclo_teleoperation
