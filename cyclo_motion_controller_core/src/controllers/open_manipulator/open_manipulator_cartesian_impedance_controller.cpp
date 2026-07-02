// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "controllers/open_manipulator/open_manipulator_cartesian_impedance_controller.hpp"

#include <stdexcept>
#include <utility>

namespace cyclo_motion_controller
{
namespace controllers
{

OpenManipulatorCartesianImpedanceController::OpenManipulatorCartesianImpedanceController(
  std::shared_ptr<cyclo_motion_controller::dynamics::DynamicsSolver> robot_data,
  const std::string & controlled_link)
: cyclo_motion_controller::optimization::QPBase(),
  robot_data_(std::move(robot_data)),
  controlled_link_(controlled_link),
  joint_dof_(0)
{
  joint_dof_ = robot_data_->getKinematicsSolver()->getDof();
  if (!controlled_link_.empty() && !robot_data_->getKinematicsSolver()->hasLinkFrame(controlled_link_)) {
    throw std::runtime_error("Controlled link '" + controlled_link_ + "' does not exist in the model.");
  }

  QPBase::setQPsize(joint_dof_, joint_dof_, 0, 0);

  desired_pose_.setIdentity();
  desired_twist_.setZero();
  stiffness_.setConstant(20.0);
  damping_.setConstant(2.0);
  nullspace_stiffness_.setZero(joint_dof_);
  nullspace_damping_.setZero(joint_dof_);
  nullspace_q_desired_.setZero(joint_dof_);
  torque_weight_.setOnes(joint_dof_);
  torque_lb_.setConstant(joint_dof_, -1000.0);
  torque_ub_.setConstant(joint_dof_, 1000.0);
  torque_desired_.setZero(joint_dof_);
}

void OpenManipulatorCartesianImpedanceController::setControlledLink(const std::string & controlled_link)
{
  if (!controlled_link.empty() && !robot_data_->getKinematicsSolver()->hasLinkFrame(controlled_link)) {
    throw std::runtime_error("Controlled link '" + controlled_link + "' does not exist in the model.");
  }
  controlled_link_ = controlled_link;
}

void OpenManipulatorCartesianImpedanceController::setDesiredPose(const Eigen::Affine3d & desired_pose)
{
  desired_pose_ = desired_pose;
  desired_twist_.setZero();
}

void OpenManipulatorCartesianImpedanceController::setDesiredPoseAndVelocity(
  const Eigen::Affine3d & desired_pose,
  const cyclo_motion_controller::common::Vector6d & desired_twist)
{
  desired_pose_ = desired_pose;
  desired_twist_ = desired_twist;
}

void OpenManipulatorCartesianImpedanceController::setGains(
  const cyclo_motion_controller::common::Vector6d & stiffness,
  const cyclo_motion_controller::common::Vector6d & damping)
{
  stiffness_ = stiffness;
  damping_ = damping;
}

void OpenManipulatorCartesianImpedanceController::setNullspaceGains(
  const Eigen::Ref<const Eigen::VectorXd> & stiffness,
  const Eigen::Ref<const Eigen::VectorXd> & damping)
{
  if (stiffness.size() == joint_dof_) {
    nullspace_stiffness_ = stiffness;
  }
  if (damping.size() == joint_dof_) {
    nullspace_damping_ = damping;
  }
}

void OpenManipulatorCartesianImpedanceController::setNullspaceReference(
  const Eigen::Ref<const Eigen::VectorXd> & q_desired)
{
  if (q_desired.size() == joint_dof_) {
    nullspace_q_desired_ = q_desired;
  }
}

void OpenManipulatorCartesianImpedanceController::setTorqueWeight(
  const Eigen::Ref<const Eigen::VectorXd> & torque_weight)
{
  if (torque_weight.size() == joint_dof_) {
    torque_weight_ = torque_weight;
  }
}

void OpenManipulatorCartesianImpedanceController::setTorqueLimits(
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

bool OpenManipulatorCartesianImpedanceController::getCommand(Eigen::VectorXd & torque_command)
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

void OpenManipulatorCartesianImpedanceController::updateDesiredTorque()
{
  auto kinematics = robot_data_->getKinematicsSolver();
  const Eigen::Affine3d current_pose = kinematics->getPose(controlled_link_);
  const Eigen::MatrixXd jacobian = kinematics->getJacobian(controlled_link_);
  const Eigen::VectorXd q = kinematics->getJointPosition();
  const Eigen::VectorXd qdot = kinematics->getJointVelocity();

  cyclo_motion_controller::common::Vector6d error;
  error.head<3>() = desired_pose_.translation() - current_pose.translation();
  error.tail<3>() = cyclo_motion_controller::common::shortestOrientationError(
    desired_pose_.linear(), current_pose.linear());

  const cyclo_motion_controller::common::Vector6d xdot = jacobian * qdot;
  const cyclo_motion_controller::common::Vector6d wrench =
    stiffness_.asDiagonal() * error + damping_.asDiagonal() * (desired_twist_ - xdot);

  torque_desired_ =
    jacobian.transpose() * wrench +
    nullspace_stiffness_.asDiagonal() * (nullspace_q_desired_ - q) -
    nullspace_damping_.asDiagonal() * qdot;
}

void OpenManipulatorCartesianImpedanceController::setCost()
{
  P_ds_.setZero(nx_, nx_);
  q_ds_.setZero(nx_);

  P_ds_ = 2.0 * torque_weight_.asDiagonal();
  q_ds_ = -2.0 * torque_weight_.asDiagonal() * torque_desired_;
}

void OpenManipulatorCartesianImpedanceController::setBoundConstraint()
{
  l_bound_ds_ = torque_lb_;
  u_bound_ds_ = torque_ub_;
}

void OpenManipulatorCartesianImpedanceController::setIneqConstraint()
{
  A_ineq_ds_.setZero(nineqc_, nx_);
  l_ineq_ds_.setConstant(nineqc_, -OSQP_INFTY);
  u_ineq_ds_.setConstant(nineqc_, OSQP_INFTY);
}

void OpenManipulatorCartesianImpedanceController::setEqConstraint()
{
  A_eq_ds_.setZero(neqc_, nx_);
  b_eq_ds_.setZero(neqc_);
}

}  // namespace controllers
}  // namespace cyclo_motion_controller
