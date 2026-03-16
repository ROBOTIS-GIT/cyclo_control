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
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <fstream>

#include "OsqpEigen/OsqpEigen.h"

namespace motion_controller
{
namespace optimization
{
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;

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
  bool solveQP(MatrixXd & sol)
  {
    setCost();
    if (nbc_ != 0) {setBoundConstraint();}
    if (nineqc_ != 0) {setIneqConstraint();}
    if (neqc_ != 0) {setEqConstraint();}
    setConstraint();

                // Convert to sparse format for OSQP
    SparseMatrix<double> P = P_ds_.sparseView();
    SparseMatrix<double> A = A_ds_.sparseView();
    VectorXd q = q_ds_;
    VectorXd l = l_ds_;
    VectorXd u = u_ds_;

    OsqpEigen::Solver solver;

                // Configure solver settings
    solver.settings()->setWarmStart(true);
    solver.settings()->getSettings()->verbose = false;

                // Set QP problem data
    solver.data()->setNumberOfVariables(nx_);
    solver.data()->setNumberOfConstraints(nc_);
    if (!solver.data()->setHessianMatrix(P)) {return false;}
    if (!solver.data()->setGradient(q)) {return false;}
    if (!solver.data()->setLinearConstraintsMatrix(A)) {return false;}
    if (!solver.data()->setLowerBound(l)) {return false;}
    if (!solver.data()->setUpperBound(u)) {return false;}

                // Initialize and solve
    if (!solver.initSolver()) {return false;}
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {return false;}

    qp_status_ = solver.getStatus();
    if (solver.getStatus() != OsqpEigen::Status::Solved) {return false;}

    sol = solver.getSolution();
    solver.clearSolverVariables();
    solver.clearSolver();

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

protected:
  int nx_;               // number of decision variables
  int nc_;               // number of constraints (ineq + bound + eq)
  int nbc_;              // number of bound constraints
  int nineqc_;           // number of inequality constraints
  int neqc_;             // number of equality constraints

  MatrixXd P_ds_;                           // Hessian matrix
  VectorXd q_ds_;                           // Gradient vector

  MatrixXd A_ds_;                           // Constraint matrix
  VectorXd l_ds_;                           // Lower bounds for constraints
  VectorXd u_ds_;                           // Upper bounds for constraints

  MatrixXd A_ineq_ds_;                      // Inequality constraint matrix
  VectorXd l_ineq_ds_;                      // Lower bounds for inequality constraints
  VectorXd u_ineq_ds_;                      // Upper bounds for inequality constraints

  VectorXd l_bound_ds_;                     // Lower bounds for bound constraints
  VectorXd u_bound_ds_;                     // Upper bounds for bound constraints

  MatrixXd A_eq_ds_;                        // Equality constraint matrix
  VectorXd b_eq_ds_;                        // Bounds for equality constraints


  OsqpEigen::Status qp_status_;             // Status of the QP solver
};
}  // namespace optimization
}  // namespace motion_controller

// Legacy alias (prefer motion_controller::optimization::QPBase).
namespace QP
{
using QPBase = motion_controller::optimization::QPBase;
}  // namespace QP
