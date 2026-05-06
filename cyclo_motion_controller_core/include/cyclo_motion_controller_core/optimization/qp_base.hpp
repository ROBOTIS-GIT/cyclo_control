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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "OsqpEigen/OsqpEigen.h"

namespace cyclo_motion_controller
{
namespace optimization
{

    /**
     * @brief Base class for QP solver.
     *
     * This class provides a framework for setting up and solving QP problems.
     * It includes methods for setting the size of the QP, defining cost functions,
     * constraints, and solving the QP problem.
     */
class QPBase
{
public:
            /**
             * @brief Constructor.
             */
  QPBase() {}
            /**
             * @brief Set the size and initialize each variable of the QP problem.
             * @param nx      (int) Number of decision variables.
             * @param nbc     (int) Number of bound constraints.
             * @param nineqc  (int) Number of inequality constraints.
             * @param neqc    (int) Number of equality constraints.
             */
  void setQPsize(const int & nx, const int & nbc, const int & nineqc, const int & neqc)
  {
    assert(nbc == nx || nbc == 0);

    nx_ = nx;
    nbc_ = nbc;
    nineqc_ = nineqc;
    neqc_ = neqc;
    nc_ = nineqc_ + neqc_ + nbc_;

    P_ds_.setZero(nx_, nx_);
    q_ds_.setZero(nx_);

    A_ineq_ds_.setZero(nineqc_, nx_);
    l_ineq_ds_.setConstant(nineqc_, -OSQP_INFTY);
    u_ineq_ds_.setConstant(nineqc_, OSQP_INFTY);

    l_bound_ds_.setConstant(nbc_, -OSQP_INFTY);
    u_bound_ds_.setConstant(nbc_, OSQP_INFTY);

    A_eq_ds_.setZero(neqc_, nx_);
    b_eq_ds_.setZero(neqc_);

    A_ds_.setZero(nc_, nx_);
    l_ds_.setConstant(nc_, -OSQP_INFTY);
    u_ds_.setConstant(nc_, OSQP_INFTY);

    solver_initialized_ = false;
    has_prev_pattern_ = false;
  }
            /**
             * @brief Initialize the solver.
             * @param P (const Eigen::SparseMatrix<double>&) Hessian matrix.
             * @param A (const Eigen::SparseMatrix<double>&) Constraint matrix.
             * @param q (const Eigen::VectorXd&) Gradient vector.
             * @param l (const Eigen::VectorXd&) Lower bounds for constraints.
             * @param u (const Eigen::VectorXd&) Upper bounds for constraints.
             * @return (bool) True if the solver was initialized successfully.
             */
  bool initializeSolver(
    const Eigen::SparseMatrix<double> & P,
    const Eigen::SparseMatrix<double> & A,
    Eigen::VectorXd & q,
    Eigen::VectorXd & l,
    Eigen::VectorXd & u)
  {
    // Recreate solver from scratch to avoid stale Data() state after failed setup.
    solver_ = OsqpEigen::Solver();

    solver_.settings()->setWarmStart(true);
    solver_.settings()->getSettings()->verbose = false;

    solver_.data()->setNumberOfVariables(nx_);
    solver_.data()->setNumberOfConstraints(nc_);
    if (!solver_.data()->setHessianMatrix(P)) {return false;}
    if (!solver_.data()->setGradient(q)) {return false;}
    if (!solver_.data()->setLinearConstraintsMatrix(A)) {return false;}
    if (!solver_.data()->setLowerBound(l)) {return false;}
    if (!solver_.data()->setUpperBound(u)) {return false;}

    if (!solver_.initSolver()) {return false;}
    solver_initialized_ = true;
    return true;
  }

            /**
             * @brief Solve the QP problem.
             *
             * Solves: min (1/2) x' P x + q' x
             *         subject to l <= A x <= u
             *
             * @param sol (Eigen::MatrixXd&) Output solution matrix.
             * @return (bool) True if the QP was solved successfully.
             */
  bool solveQP(Eigen::MatrixXd & sol)
  {
    setCost();
    if (nbc_ != 0) {setBoundConstraint();}
    if (nineqc_ != 0) {setIneqConstraint();}
    if (neqc_ != 0) {setEqConstraint();}
    setConstraint();

                // Convert to sparse format for OSQP
    Eigen::SparseMatrix<double> P = P_ds_.sparseView();
    Eigen::SparseMatrix<double> A = A_ds_.sparseView();
    Eigen::VectorXd q = q_ds_;
    Eigen::VectorXd l = l_ds_;
    Eigen::VectorXd u = u_ds_;

    const bool pattern_changed =
      !has_prev_pattern_ ||
      !hasSameSparsityPattern(P, prev_hessian_pattern_) ||
      !hasSameSparsityPattern(A, prev_constraint_pattern_);

    if (!solver_initialized_ || pattern_changed) {
      if (!initializeSolver(P, A, q, l, u)) {return false;}
    } else {
      const bool updated =
        solver_.updateHessianMatrix(P) &&
        solver_.updateLinearConstraintsMatrix(A) &&
        solver_.updateGradient(q) &&
        solver_.updateBounds(l, u);

      if (!updated) {
        solver_initialized_ = false;
        if (!initializeSolver(P, A, q, l, u)) {return false;}
      }
    }

    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {return false;}

    qp_status_ = solver_.getStatus();
    if (solver_.getStatus() != OsqpEigen::Status::Solved) {return false;}

    sol = solver_.getSolution();
    prev_hessian_pattern_ = P;
    prev_constraint_pattern_ = A;
    has_prev_pattern_ = true;

    return true;
  }

private:
            /**
             * @brief Set the cost function for the QP problem.
             */
  virtual void setCost() = 0;
            /**
             * @brief Set the bound constraints such as control input limits for the QP problem.
             */
  virtual void setBoundConstraint() = 0;
            /**
             * @brief Set the inequality constraints such as joint angle limits for the QP problem.
             */
  virtual void setIneqConstraint() = 0;
            /**
             * @brief Set the equality constraints such as equations of motion for the QP problem.
             */
  virtual void setEqConstraint() = 0;
            /**
             * @brief Cumulate constraints to solve the QP problem.
             */
  void setConstraint()
  {
    A_ds_.setZero(nc_, nx_);
    l_ds_.setConstant(nc_, -OSQP_INFTY);
    u_ds_.setConstant(nc_, OSQP_INFTY);

                // Bound Constraint
    if (nbc_ != 0) {
      A_ds_.block(0, 0, nbc_, nx_).setIdentity();
      l_ds_.segment(0, nbc_) = l_bound_ds_;
      u_ds_.segment(0, nbc_) = u_bound_ds_;
    }

                // Inequality Constraint
    if (nineqc_ != 0) {
      A_ds_.block(nbc_, 0, nineqc_, nx_) = A_ineq_ds_;
      l_ds_.segment(nbc_, nineqc_) = l_ineq_ds_;
      u_ds_.segment(nbc_, nineqc_) = u_ineq_ds_;
    }

                // Equality Constraint
    if (neqc_ != 0) {
      A_ds_.block(nbc_ + nineqc_, 0, neqc_, nx_) = A_eq_ds_;
      l_ds_.segment(nbc_ + nineqc_, neqc_) = b_eq_ds_;
      u_ds_.segment(nbc_ + nineqc_, neqc_) = b_eq_ds_;
    }
  }

  static bool hasSameSparsityPattern(
    const Eigen::SparseMatrix<double> & lhs,
    const Eigen::SparseMatrix<double> & rhs)
  {
    if (
      lhs.rows() != rhs.rows() ||
      lhs.cols() != rhs.cols() ||
      lhs.outerSize() != rhs.outerSize() ||
      lhs.nonZeros() != rhs.nonZeros())
    {
      return false;
    }

    const bool same_outer = std::equal(
      lhs.outerIndexPtr(), lhs.outerIndexPtr() + lhs.outerSize() + 1, rhs.outerIndexPtr());
    const bool same_inner = std::equal(
      lhs.innerIndexPtr(), lhs.innerIndexPtr() + lhs.nonZeros(), rhs.innerIndexPtr());
    return same_outer && same_inner;
  }

protected:
  int nx_;               // number of decision variables
  int nc_;               // number of constraints (ineq + bound + eq)
  int nbc_;              // number of bound constraints
  int nineqc_;           // number of inequality constraints
  int neqc_;             // number of equality constraints

  Eigen::MatrixXd P_ds_;                           // Hessian matrix
  Eigen::VectorXd q_ds_;                           // Gradient vector

  Eigen::MatrixXd A_ds_;                           // Constraint matrix
  Eigen::VectorXd l_ds_;                           // Lower bounds for constraints
  Eigen::VectorXd u_ds_;                           // Upper bounds for constraints

  Eigen::MatrixXd A_ineq_ds_;                      // Inequality constraint matrix
  Eigen::VectorXd l_ineq_ds_;                      // Lower bounds for inequality constraints
  Eigen::VectorXd u_ineq_ds_;                      // Upper bounds for inequality constraints

  Eigen::VectorXd l_bound_ds_;                     // Lower bounds for bound constraints
  Eigen::VectorXd u_bound_ds_;                     // Upper bounds for bound constraints

  Eigen::MatrixXd A_eq_ds_;                        // Equality constraint matrix
  Eigen::VectorXd b_eq_ds_;                        // Bounds for equality constraints


  OsqpEigen::Status qp_status_;             // Status of the QP solver
  OsqpEigen::Solver solver_;                // Persistent OSQP solver
  bool solver_initialized_ {false};         // Solver init state
  Eigen::SparseMatrix<double> prev_hessian_pattern_;     // Cached Hessian pattern
  Eigen::SparseMatrix<double> prev_constraint_pattern_;  // Cached constraint pattern
  bool has_prev_pattern_ {false};                        // Pattern cache state
};
}  // namespace optimization
}  // namespace cyclo_motion_controller

namespace QP
{
using QPBase = cyclo_motion_controller::optimization::QPBase;
}  // namespace QP
