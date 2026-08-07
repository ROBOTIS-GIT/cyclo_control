// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0

#include "controllers/ai_worker/elbow_up_qp_controller.hpp"

#include <algorithm>

namespace cyclo_motion_controller
{
namespace controllers
{
ElbowUpQPController::ElbowUpQPController(
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> robot_data)
: robot_data_(std::move(robot_data))
{
  const int dof = robot_data_->getDof();
  const int collision_count = robot_data_->getCollisionPairCount();
  index_.qdot_size = dof;
  index_.slack_q_min_size = dof;
  index_.slack_q_max_size = dof;
  index_.slack_collision_size = collision_count;
  index_.con_q_min_size = dof;
  index_.con_q_max_size = dof;
  index_.con_collision_size = collision_count;

  index_.slack_q_min_start = dof;
  index_.slack_q_max_start = index_.slack_q_min_start + dof;
  index_.slack_collision_start = index_.slack_q_max_start + dof;
  index_.con_q_max_start = dof;
  index_.con_collision_start = index_.con_q_max_start + dof;

  const int nx = dof * 3 + collision_count;
  const int nineq = dof * 2 + collision_count;
  QPBase::setQPsize(nx, nx, nineq, 0);
  damping_weight_.setOnes(dof);
  preferred_joint_velocity_.setZero(dof);
}

void ElbowUpQPController::setDesiredTaskVel(
  const std::map<std::string, cyclo_motion_controller::common::Vector6d> & desired)
{
  desired_task_velocity_ = desired;
}

void ElbowUpQPController::setWeights(
  const std::map<std::string, cyclo_motion_controller::common::Vector6d> & tracking,
  const Eigen::VectorXd & damping)
{
  tracking_weight_ = tracking;
  damping_weight_ = damping;
}

void ElbowUpQPController::setPreferredJointVel(
  const Eigen::VectorXd & preferred, const double weight)
{
  if (preferred.size() == index_.qdot_size) {
    preferred_joint_velocity_ = preferred;
    preferred_joint_velocity_weight_ = std::max(0.0, weight);
  }
}

void ElbowUpQPController::setControllerParams(
  const double slack_penalty, const double cbf_alpha, const double collision_buffer,
  const double collision_safe_distance)
{
  slack_penalty_ = slack_penalty;
  cbf_alpha_ = cbf_alpha;
  collision_buffer_ = collision_buffer;
  collision_safe_distance_ = collision_safe_distance;
}

bool ElbowUpQPController::getOptJointVel(Eigen::VectorXd & qdot)
{
  Eigen::MatrixXd solution;
  if (!solveQP(solution)) {
    qdot.setZero(index_.qdot_size);
    return false;
  }
  qdot = solution.block(index_.qdot_start, 0, index_.qdot_size, 1);
  return true;
}

void ElbowUpQPController::setCost()
{
  P_ds_.setZero(nx_, nx_);
  q_ds_.setZero(nx_);
  for (const auto & [link, desired] : desired_task_velocity_) {
    const Eigen::MatrixXd jacobian = robot_data_->getJacobian(link);
    cyclo_motion_controller::common::Vector6d weight =
      cyclo_motion_controller::common::Vector6d::Ones();
    const auto iter = tracking_weight_.find(link);
    if (iter != tracking_weight_.end()) {
      weight = iter->second;
    }
    P_ds_.block(0, 0, index_.qdot_size, index_.qdot_size) +=
      2.0 * jacobian.transpose() * weight.asDiagonal() * jacobian;
    q_ds_.head(index_.qdot_size) +=
      -2.0 * jacobian.transpose() * weight.asDiagonal() * desired;
  }
  P_ds_.block(0, 0, index_.qdot_size, index_.qdot_size) +=
    2.0 * damping_weight_.asDiagonal();
  if (preferred_joint_velocity_weight_ > 0.0) {
    P_ds_.block(0, 0, index_.qdot_size, index_.qdot_size).diagonal().array() +=
      2.0 * preferred_joint_velocity_weight_;
    q_ds_.head(index_.qdot_size) +=
      -2.0 * preferred_joint_velocity_weight_ * preferred_joint_velocity_;
  }
  q_ds_.segment(index_.slack_q_min_start, index_.slack_q_min_size).setConstant(slack_penalty_);
  q_ds_.segment(index_.slack_q_max_start, index_.slack_q_max_size).setConstant(slack_penalty_);
  if (index_.slack_collision_size > 0) {
    q_ds_.segment(index_.slack_collision_start, index_.slack_collision_size).setConstant(
      slack_penalty_);
  }
}

void ElbowUpQPController::setBoundConstraint()
{
  l_bound_ds_.setConstant(nbc_, -OSQP_INFTY);
  u_bound_ds_.setConstant(nbc_, OSQP_INFTY);
  l_bound_ds_.head(index_.qdot_size) = robot_data_->getJointVelocityLimit().first;
  u_bound_ds_.head(index_.qdot_size) = robot_data_->getJointVelocityLimit().second;
  l_bound_ds_.segment(index_.slack_q_min_start, index_.slack_q_min_size).setZero();
  l_bound_ds_.segment(index_.slack_q_max_start, index_.slack_q_max_size).setZero();
  if (index_.slack_collision_size > 0) {
    l_bound_ds_.segment(index_.slack_collision_start, index_.slack_collision_size).setZero();
  }
}

void ElbowUpQPController::setIneqConstraint()
{
  A_ineq_ds_.setZero(nineqc_, nx_);
  l_ineq_ds_.setConstant(nineqc_, -OSQP_INFTY);
  u_ineq_ds_.setConstant(nineqc_, OSQP_INFTY);
  const Eigen::VectorXd q = robot_data_->getJointPosition();
  const auto position_limit = robot_data_->getJointPositionLimit();
  A_ineq_ds_.block(0, 0, index_.qdot_size, index_.qdot_size).setIdentity();
  A_ineq_ds_.block(
    0, index_.slack_q_min_start, index_.con_q_min_size,
    index_.slack_q_min_size).setIdentity();
  l_ineq_ds_.head(index_.con_q_min_size) = -cbf_alpha_ * (q - position_limit.first);
  A_ineq_ds_.block(
    index_.con_q_max_start, 0, index_.con_q_max_size,
    index_.qdot_size) = -Eigen::MatrixXd::Identity(index_.qdot_size, index_.qdot_size);
  A_ineq_ds_.block(
    index_.con_q_max_start, index_.slack_q_max_start,
    index_.con_q_max_size, index_.slack_q_max_size).setIdentity();
  l_ineq_ds_.segment(index_.con_q_max_start, index_.con_q_max_size) =
    -cbf_alpha_ * (position_limit.second - q);

  const auto distances = robot_data_->getCollisionPairDistances(true, false, false);
  const int count = std::min<int>(index_.con_collision_size, distances.size());
  for (int i = 0; i < count; ++i) {
    A_ineq_ds_.block(index_.con_collision_start + i, 0, 1, index_.qdot_size) =
      distances[i].grad.transpose();
    A_ineq_ds_(index_.con_collision_start + i, index_.slack_collision_start + i) = 1.0;
    if (distances[i].distance <= collision_buffer_) {
      l_ineq_ds_(index_.con_collision_start + i) =
        -cbf_alpha_ * (distances[i].distance - collision_safe_distance_);
    }
  }
}

void ElbowUpQPController::setEqConstraint()
{
  A_eq_ds_.setZero(neqc_, nx_);
  b_eq_ds_.setZero(neqc_);
}
}  // namespace controllers
}  // namespace cyclo_motion_controller
