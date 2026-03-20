// Copyright (C) 2018 Giulio Romualdi
//
// Use of this source code is governed by a BSD-3-Clause
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/BSD-3-Clause

/**
 * @file Constants.hpp
 * @author Giulio Romualdi
 * @copyright  Released under the terms of the BSD 3-Clause License
 * @date 2018
 */

#ifndef OSQPEIGEN__CONSTANTS_HPP_
#define OSQPEIGEN__CONSTANTS_HPP_

#include <OsqpEigen/Compat.hpp>

/**
 * OsqpEigen namespace.
 */
namespace OsqpEigen
{
constexpr c_float INFTY = OSQP_INFTY; /**< Infinity constant. */

/**
 * Status of the solver
 */
enum class Status : int
{
  DualInfeasibleInaccurate = OSQP_DUAL_INFEASIBLE_INACCURATE,
  PrimalInfeasibleInaccurate = OSQP_PRIMAL_INFEASIBLE_INACCURATE,
  SolvedInaccurate = OSQP_SOLVED_INACCURATE,
  Solved = OSQP_SOLVED,
  MaxIterReached = OSQP_MAX_ITER_REACHED,
  PrimalInfeasible = OSQP_PRIMAL_INFEASIBLE,
  DualInfeasible = OSQP_DUAL_INFEASIBLE,
  Sigint = OSQP_SIGINT,
#if defined(PROFILING) || defined(OSQP_EIGEN_OSQP_IS_V1)
  TimeLimitReached = OSQP_TIME_LIMIT_REACHED,
#endif  // ifdef PROFILING
  NonCvx = OSQP_NON_CVX,
  Unsolved = OSQP_UNSOLVED
};

/**
 * Error status of the Solver
 */
enum class ErrorExitFlag : int
{
  NoError = 0,
  DataValidationError = OSQP_DATA_VALIDATION_ERROR,
  SettingsValidationError = OSQP_SETTINGS_VALIDATION_ERROR,
#ifdef OSQP_EIGEN_OSQP_IS_V1
  LinsysSolverLoadError = OSQP_ALGEBRA_LOAD_ERROR,
#else
  LinsysSolverLoadError = OSQP_LINSYS_SOLVER_LOAD_ERROR,
#endif
  LinsysSolverInitError = OSQP_LINSYS_SOLVER_INIT_ERROR,
  NonCvxError = OSQP_NONCVX_ERROR,
  MemAllocError = OSQP_MEM_ALLOC_ERROR,
  WorkspaceNotInitError = OSQP_WORKSPACE_NOT_INIT_ERROR
};

}  // namespace OsqpEigen

#endif  // OSQPEIGEN__CONSTANTS_HPP_
