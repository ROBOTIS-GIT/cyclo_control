#include "dynamics/dynamics_solver.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace cyclo_motion_controller
{
namespace dynamics
{

namespace
{
void validatePositionSize(
  const Eigen::Ref<const VectorXd> & q,
  const pinocchio::Model & model)
{
  if (q.size() != static_cast<int>(model.nq)) {
    throw std::runtime_error(
      "Joint position size mismatch. Expected " + std::to_string(model.nq) +
      ", got " + std::to_string(q.size()));
  }
}

void validateVelocitySize(
  const Eigen::Ref<const VectorXd> & qdot,
  const pinocchio::Model & model)
{
  if (qdot.size() != static_cast<int>(model.nv)) {
    throw std::runtime_error(
      "Joint velocity size mismatch. Expected " + std::to_string(model.nv) +
      ", got " + std::to_string(qdot.size()));
  }
}
}  // namespace

DynamicsSolver::DynamicsSolver(const std::string & urdf_path, const std::string & srdf_path)
: urdf_path_(urdf_path), srdf_path_(srdf_path)
{
  if (!std::filesystem::exists(urdf_path)) {
    throw std::runtime_error("URDF file does not exist: " + urdf_path);
  }
  if (!srdf_path.empty() && !std::filesystem::exists(srdf_path)) {
    throw std::runtime_error("SRDF file does not exist: " + srdf_path);
  }

  kinematics_solver_ =
    std::make_shared<cyclo_motion_controller::kinematics::KinematicsSolver>(urdf_path, srdf_path);

  pinocchio::urdf::buildModel(urdf_path, model_, /*verbose=*/false);
  data_ = pinocchio::Data(model_);

  q_.setZero(static_cast<int>(model_.nq));
  qdot_.setZero(static_cast<int>(model_.nv));
  torque_ub_ = model_.effortLimit.head(static_cast<int>(model_.nv));
  torque_lb_ = -torque_ub_;

  M_.setZero(model_.nv, model_.nv);
  M_inv_.setZero(model_.nv, model_.nv);
  c_.setZero(model_.nv);
  g_.setZero(model_.nv);
  NLE_.setZero(model_.nv);

  updateDynamics(q_, qdot_);
}

DynamicsSolver::~DynamicsSolver()
{
}

bool DynamicsSolver::updateState(
  const Eigen::Ref<const VectorXd> & q,
  const Eigen::Ref<const VectorXd> & qdot)
{
  validatePositionSize(q, model_);
  validateVelocitySize(qdot, model_);

  q_ = q;
  qdot_ = qdot;

  if (!kinematics_solver_->updateState(q_, qdot_)) {
    return false;
  }

  if (!updateDynamics(q_, qdot_)) {
    return false;
  }

  return true;
}

bool DynamicsSolver::updateDynamics(
  const Eigen::Ref<const VectorXd> & q,
  const Eigen::Ref<const VectorXd> & qdot)
{
  validatePositionSize(q, model_);
  validateVelocitySize(qdot, model_);

  M_ = computeMassMatrix(q);
  g_ = computeGravity(q);
  NLE_ = computeNonlinearEffects(q, qdot);
  c_ = NLE_ - g_;

  M_inv_ = M_.ldlt().solve(MatrixXd::Identity(M_.rows(), M_.cols()));

  return true;
}

MatrixXd DynamicsSolver::computeMassMatrix(const Eigen::Ref<const VectorXd> & q)
{
  validatePositionSize(q, model_);

  pinocchio::Data data(model_);
  const VectorXd q_vec = q;
  pinocchio::crba(model_, data, q_vec);

  MatrixXd mass_matrix = data.M;
  mass_matrix.triangularView<Eigen::StrictlyLower>() =
    mass_matrix.transpose().triangularView<Eigen::StrictlyLower>();

  return mass_matrix;
}

VectorXd DynamicsSolver::computeGravity(const Eigen::Ref<const VectorXd> & q)
{
  validatePositionSize(q, model_);

  pinocchio::Data data(model_);
  const VectorXd q_vec = q;
  const VectorXd zero_velocity = VectorXd::Zero(model_.nv);
  const VectorXd zero_acceleration = VectorXd::Zero(model_.nv);

  return pinocchio::rnea(model_, data, q_vec, zero_velocity, zero_acceleration);
}

VectorXd DynamicsSolver::computeCoriolis(
  const Eigen::Ref<const VectorXd> & q,
  const Eigen::Ref<const VectorXd> & qdot)
{
  return computeNonlinearEffects(q, qdot) - computeGravity(q);
}

VectorXd DynamicsSolver::computeNonlinearEffects(
  const Eigen::Ref<const VectorXd> & q,
  const Eigen::Ref<const VectorXd> & qdot)
{
  validatePositionSize(q, model_);
  validateVelocitySize(qdot, model_);

  pinocchio::Data data(model_);
  const VectorXd q_vec = q;
  const VectorXd qdot_vec = qdot;

  return pinocchio::nonLinearEffects(model_, data, q_vec, qdot_vec);
}

}  // namespace dynamics
}  // namespace cyclo_motion_controller
