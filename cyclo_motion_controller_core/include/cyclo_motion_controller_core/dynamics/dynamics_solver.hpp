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

#pragma once

#include <Eigen/Dense>

#include <memory>
#include <string>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "kinematics/kinematics_solver.hpp"

namespace cyclo_motion_controller
{
namespace dynamics
{
using Eigen::MatrixXd;
using Eigen::VectorXd;

  /**
  * @brief Generic Dynamics Solver class that provides FD and ID using a selectable backend.
  */
class DynamicsSolver
{
public:
  DynamicsSolver(const std::string & urdf_path, const std::string & srdf_path);
  ~DynamicsSolver();

  /**
   * @brief Update the state of the manipulator.
   * @param q     (Eigen::VectorXd) Joint positions.
   * @param qdot  (Eigen::VectorXd) Joint velocities.
   * @return (bool) True if state update is successful.
   */
  virtual bool updateState(const Eigen::Ref<const VectorXd> & q, const Eigen::Ref<const VectorXd> & qdot);

    // ================================ Compute Functions ================================
    // Joint space
    /**
     * @brief Compute the mass matrix of the manipulator.
     * @param q (Eigen::VectorXd) Joint positions.
     * @return (Eigen::MatrixXd) Mass matrix of the manipulator.
     */
  virtual MatrixXd computeMassMatrix(const Eigen::Ref<const VectorXd> & q);

    /**
     * @brief Compute the gravity vector of the manipulator.
     * @param q (Eigen::VectorXd) Joint positions.
     * @return (Eigen::VectorXd) Gravity vector of the manipulator.
     */
  virtual VectorXd computeGravity(const Eigen::Ref<const VectorXd> & q);

    /**
     * @brief Compute the coriolis vector of the manipulator.
     * @param q     (Eigen::VectorXd) Joint positions.
     * @param qdot  (Eigen::VectorXd) Joint velocities.
     * @return (Eigen::VectorXd) Coriolis vector of the manipulator.
     */
  virtual VectorXd computeCoriolis(
    const Eigen::Ref<const VectorXd> & q,
    const Eigen::Ref<const VectorXd> & qdot);

    /**
     * @brief Compute the nonlinear effects vector of the manipulator.
     * @param q     (Eigen::VectorXd) Joint positions.
     * @param qdot  (Eigen::VectorXd) Joint velocities.
     * @return (Eigen::VectorXd) Nonlinear effects vector of the manipulator.
     */
  virtual VectorXd computeNonlinearEffects(
    const Eigen::Ref<const VectorXd> & q,
    const Eigen::Ref<const VectorXd> & qdot);

    // ================================ Get Functions ================================
  virtual const std::string getURDFPath() const {return urdf_path_;}

    /**
     * @brief Get the owned kinematics solver for FK, Jacobians, frame queries, and collision distances.
     * @return Shared pointer to the kinematics solver.
     */
  virtual std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver>
  getKinematicsSolver()
  {
    return kinematics_solver_;
  }

    /**
     * @brief Get the owned kinematics solver for FK, Jacobians, frame queries, and collision distances.
     * @return Const shared pointer to the kinematics solver.
     */
  virtual std::shared_ptr<const cyclo_motion_controller::kinematics::KinematicsSolver>
  getKinematicsSolver() const
  {
    return kinematics_solver_;
  }

    /**
     * @brief Get the mass matrix of the manipulator.
     * @return (Eigen::MatrixXd) Mass matrix of the manipulator.
     */
  virtual MatrixXd getMassMatrix() const {return M_;}

    /**
     * @brief Get the inversed mass matrix of the manipulator.
     * @return (Eigen::MatrixXd) Inversed mass matrix of the manipulator.
     */
  virtual MatrixXd getMassMatrixInv() const {return M_inv_;}

    /**
     * @brief Get the coriolis vector of the manipulator.
     * @return (Eigen::VectorXd) Coriolis vector of the manipulator.
     */
  virtual VectorXd getCoriolis() const {return c_;}

    /**
     * @brief Get the gravity vector of the manipulator.
     * @return (Eigen::VectorXd) Gravity vector of the manipulator.
     */
  virtual VectorXd getGravity() const {return g_;}

    /**
     * @brief Get the nonlinear effects vector of the manipulator.
     * @return (Eigen::VectorXd) Nonlinear effects vector of the manipulator.
     */
  virtual VectorXd getNonlinearEffects() const {return NLE_;}

    /**
     * @brief Get lower and upper joint effort limits of the manipulator.
     * @return (std::pair<Eigen::VectorXd, Eigen::VectorXd>) Joint effort limits (lower, upper).
     */
  virtual std::pair<VectorXd, VectorXd> getJointEffortLimit() const
  {
    return std::make_pair(torque_lb_, torque_ub_);
  }

protected:
    /**
    * @brief Update the dynamic parameters of the manipulator.
    * @param q     (Eigen::VectorXd) Joint positions.
    * @param qdot  (Eigen::VectorXd) Joint velocities.
    * @return (bool) True if the update was successful.
    */
  virtual bool updateDynamics(const Eigen::Ref<const VectorXd> & q, const Eigen::Ref<const VectorXd> & qdot);

  std::string urdf_path_;
  std::string srdf_path_;

  pinocchio::Model model_;
  pinocchio::Data data_;

  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> kinematics_solver_;

  VectorXd q_;          // Manipulator joint positions.
  VectorXd qdot_;       // Manipulator joint velocities.
  VectorXd torque_lb_;   // Lower joint effort limits.
  VectorXd torque_ub_;   // Upper joint effort limits.

  MatrixXd M_;          // Manipulator mass matrix.
  MatrixXd M_inv_;      // Inverse manipulator mass matrix.
  VectorXd c_;          // Coriolis/centrifugal vector.
  VectorXd g_;          // Gravity vector.
  VectorXd NLE_;        // Nonlinear effects vector.
};

}  // namespace dynamics
}  // namespace cyclo_motion_controller
