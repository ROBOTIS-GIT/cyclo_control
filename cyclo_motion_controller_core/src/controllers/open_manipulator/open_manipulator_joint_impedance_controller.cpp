// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "controllers/open_manipulator/open_manipulator_joint_impedance_controller.hpp"

#include <algorithm>
#include <utility>

namespace cyclo_motion_controller
{
namespace controllers
{

OpenManipulatorJointImpedanceController::OpenManipulatorJointImpedanceController(
  std::shared_ptr<cyclo_motion_controller::dynamics::DynamicsSolver> robot_data)
: cyclo_motion_controller::optimization::QPBase(),
  robot_data_(std::move(robot_data)),
  joint_dof_(0)
{
  joint_dof_ = robot_data_->getKinematicsSolver()->getDof();
  QPBase::setQPsize(joint_dof_, joint_dof_, 0, 0);

  q_desired_.setZero(joint_dof_);
  qdot_desired_.setZero(joint_dof_);
  stiffness_.setConstant(joint_dof_, 20.0);
  damping_.setConstant(joint_dof_, 2.0);
  torque_weight_.setOnes(joint_dof_);
  torque_lb_.setConstant(joint_dof_, -1000.0);
  torque_ub_.setConstant(joint_dof_, 1000.0);
  torque_desired_.setZero(joint_dof_);
}

void OpenManipulatorJointImpedanceController::setDesiredJointState(
  const Eigen::Ref<const Eigen::VectorXd> & q_desired,
  const Eigen::Ref<const Eigen::VectorXd> & qdot_desired)
{
  if (q_desired.size() == joint_dof_) {
    q_desired_ = q_desired;
  }
  if (qdot_desired.size() == joint_dof_) {
    qdot_desired_ = qdot_desired;
  }
}

void OpenManipulatorJointImpedanceController::setGains(
  const Eigen::Ref<const Eigen::VectorXd> & stiffness,
  const Eigen::Ref<const Eigen::VectorXd> & damping)
{
  if (stiffness.size() == joint_dof_) {
    stiffness_ = stiffness;
  }
  if (damping.size() == joint_dof_) {
    damping_ = damping;
  }
}

void OpenManipulatorJointImpedanceController::setTorqueWeight(
  const Eigen::Ref<const Eigen::VectorXd> & torque_weight)
{
  if (torque_weight.size() == joint_dof_) {
    torque_weight_ = torque_weight;
  }
}

void OpenManipulatorJointImpedanceController::setTorqueLimits(
  const Eigen::Ref<const Eigen::VectorXd> & lower,
  const Eigen::Ref<const Eigen::VectorXd> & upper)
{
  if (lower.size() == joint_dof_) {
    torque_lb_ = lower;
  }
  if (upper.size() == joint_dof_) {
    torque_ub_ = upper;
  }
}

bool OpenManipulatorJointImpedanceController::getCommand(Eigen::VectorXd & torque_command)
{
  updateDesiredTorque();

  Eigen::MatrixXd sol;
  if (!solveQP(sol)) {
    torque_command = Eigen::VectorXd::Zero(joint_dof_);
    return false;
  }

  torque_command = sol.col(0);
  return true;
}

void OpenManipulatorJointImpedanceController::updateDesiredTorque()
{
  const Eigen::VectorXd q = robot_data_->getKinematicsSolver()->getJointPosition();
  const Eigen::VectorXd qdot = robot_data_->getKinematicsSolver()->getJointVelocity();

  torque_desired_ =
    stiffness_.asDiagonal() * (q_desired_ - q) +
    damping_.asDiagonal() * (qdot_desired_ - qdot);
}

void OpenManipulatorJointImpedanceController::setCost()
{
  P_ds_.setZero(nx_, nx_);
  q_ds_.setZero(nx_);

  P_ds_ = 2.0 * torque_weight_.asDiagonal();
  q_ds_ = -2.0 * torque_weight_.asDiagonal() * torque_desired_;
}

void OpenManipulatorJointImpedanceController::setBoundConstraint()
{
  l_bound_ds_ = torque_lb_;
  u_bound_ds_ = torque_ub_;
}

void OpenManipulatorJointImpedanceController::setIneqConstraint()
{
  A_ineq_ds_.setZero(nineqc_, nx_);
  l_ineq_ds_.setConstant(nineqc_, -OSQP_INFTY);
  u_ineq_ds_.setConstant(nineqc_, OSQP_INFTY);
}

void OpenManipulatorJointImpedanceController::setEqConstraint()
{
  A_eq_ds_.setZero(neqc_, nx_);
  b_eq_ds_.setZero(neqc_);
}

}  // namespace controllers
}  // namespace cyclo_motion_controller
