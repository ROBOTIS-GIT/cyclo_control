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
#include <unsupported/Eigen/MatrixFunctions>

#include <cassert>

namespace cyclo_motion_controller
{
namespace common
{
// 6D twist / spatial vector
using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace collision_checker
{
/**
* @brief Result of minimum distance computation between links.
*
* Used for self-collision avoidance or proximity monitoring.
* Contains the minimum distance and its sensitivity w.r.t. joint configuration.
*/
struct MinDistResult
{
  double distance {0.0};                   // Minimum distance [m]
  Eigen::VectorXd grad;                    // Gradient of distance w.r.t joint positions
  Eigen::VectorXd grad_dot;                // Time derivative of gradient

  void setZero(const int size)
  {
    distance = 0.0;
    grad.setZero(size);
    grad_dot.setZero(size);
  }
};
}      // namespace collision_checker

namespace math_utils
{
/**
* @brief Scalar cubic interpolation with boundary position/velocity conditions.
*/
static double cubic(
  double time,                               ///< Current time
  double time_0,             ///< Start time
  double time_f,             ///< End time
  double x_0,                ///< Start state
  double x_f,                ///< End state
  double x_dot_0,            ///< Start state dot
  double x_dot_f             ///< End state dot
)
{
  double x_t;

  if (time < time_0) {
    x_t = x_0;
  } else if (time > time_f) {
    x_t = x_f;
  } else {
    double elapsed_time = time - time_0;
    double total_time = time_f - time_0;
    double total_time2 = total_time * total_time;              // pow(t,2)
    double total_time3 = total_time2 * total_time;             // pow(t,3)
    double total_x = x_f - x_0;

    x_t = x_0 + x_dot_0 * elapsed_time

      + (3 * total_x / total_time2 -
      2 * x_dot_0 / total_time -
      x_dot_f / total_time) *
      elapsed_time * elapsed_time

      + (-2 * total_x / total_time3 +
      (x_dot_0 + x_dot_f) / total_time2) *
      elapsed_time * elapsed_time * elapsed_time;
  }

  return x_t;
}

/**
* @brief Time derivative of scalar cubic interpolation.
*/
static double cubicDot(
  double time,                                  ///< Current time
  double time_0,             ///< Start time
  double time_f,             ///< End time
  double x_0,                ///< Start state
  double x_f,                ///< End state
  double x_dot_0,            ///< Start state dot
  double x_dot_f             ///< End state dot
)
{
  double x_t;

  if (time < time_0) {
    x_t = x_dot_0;
  } else if (time > time_f) {
    x_t = x_dot_f;
  } else {
    double elapsed_time = time - time_0;
    double total_time = time_f - time_0;
    double total_time2 = total_time * total_time;              // pow(t,2)
    double total_time3 = total_time2 * total_time;             // pow(t,3)
    double total_x = x_f - x_0;

    x_t = x_dot_0

      + 2 * (3 * total_x / total_time2 -
      2 * x_dot_0 / total_time -
      x_dot_f / total_time) *
      elapsed_time

      + 3 * (-2 * total_x / total_time3 +
      (x_dot_0 + x_dot_f) / total_time2) *
      elapsed_time * elapsed_time;
  }

  return x_t;
}

template<int N>
/**
* @brief Element-wise cubic interpolation for fixed-size vectors.
*/
static Eigen::Matrix<double, N, 1> cubicVector(
  double time,                                                          ///< Current time
  double time_0,             ///< Start time
  double time_f,             ///< End time
  Eigen::Matrix<double, N, 1> x_0,                   ///< Start state
  Eigen::Matrix<double, N, 1> x_f,                   ///< End state
  Eigen::Matrix<double, N, 1> x_dot_0,               ///< Start state dot
  Eigen::Matrix<double, N, 1> x_dot_f                ///< End state dot
)
{
  Eigen::Matrix<double, N, 1> res;
  for (unsigned int i = 0; i < N; i++) {
    res(i) = cubic(time, time_0, time_f, x_0(i), x_f(i), x_dot_0(i), x_dot_f(i));
  }
  return res;
}

/**
* @brief Element-wise cubic interpolation for dynamic-size vectors.
*/
static Eigen::VectorXd cubicVector(
  double time,
  double time_0,
  double time_f,
  const Eigen::VectorXd & x_0,
  const Eigen::VectorXd & x_f,
  const Eigen::VectorXd & x_dot_0,
  const Eigen::VectorXd & x_dot_f)
{
  assert(x_0.size() == x_f.size());
  Eigen::VectorXd res(x_0.size());
  for (int i = 0; i < x_0.size(); ++i) {
    res(i) = cubic(time, time_0, time_f, x_0(i), x_f(i), x_dot_0(i), x_dot_f(i));
  }
  return res;
}

template<int N>
/**
* @brief Element-wise derivative of cubic interpolation for fixed-size vectors.
*/
static Eigen::Matrix<double, N, 1> cubicDotVector(
  double time,                                                             ///< Current time
  double time_0,             ///< Start time
  double time_f,             ///< End time
  Eigen::Matrix<double, N, 1> x_0,                   ///< Start state
  Eigen::Matrix<double, N, 1> x_f,                   ///< End state
  Eigen::Matrix<double, N, 1> x_dot_0,               ///< Start state dot
  Eigen::Matrix<double, N, 1> x_dot_f                ///< End state dot
)
{
  Eigen::Matrix<double, N, 1> res;
  for (unsigned int i = 0; i < N; i++) {
    res(i) = cubicDot(time, time_0, time_f, x_0(i), x_f(i), x_dot_0(i), x_dot_f(i));
  }
  return res;
}

/**
* @brief Element-wise derivative of cubic interpolation for dynamic-size vectors.
*/
static Eigen::VectorXd cubicDotVector(
  double time,
  double time_0,
  double time_f,
  const Eigen::VectorXd & x_0,
  const Eigen::VectorXd & x_f,
  const Eigen::VectorXd & x_dot_0,
  const Eigen::VectorXd & x_dot_f)
{
  assert(x_0.size() == x_f.size());
  Eigen::VectorXd res(x_0.size());
  for (int i = 0; i < x_0.size(); ++i) {
    res(i) = cubicDot(time, time_0, time_f, x_0(i), x_f(i), x_dot_0(i), x_dot_f(i));
  }
  return res;
}

/**
* @brief Cubic interpolation of rotation matrices on SO(3).
*/
static const Eigen::Matrix3d rotationCubic(
  double time,
  double time_0,
  double time_f,
  const Eigen::Matrix3d & rotation_0,
  const Eigen::Matrix3d & rotation_f)
{
  if (time >= time_f) {
    return rotation_f;
  } else if (time < time_0) {
    return rotation_0;
  }
  double tau = cubic(time, time_0, time_f, 0, 1, 0, 0);
  Eigen::Matrix3d rot_scaler_skew;
  rot_scaler_skew = (rotation_0.transpose() * rotation_f).log();
  Eigen::Matrix3d result = rotation_0 * (rot_scaler_skew * tau).exp();

  return result;
}

/**
* @brief Angular velocity profile associated with rotationCubic().
*/
static const Eigen::Vector3d rotationCubicDot(
  double time, double time_0, double time_f,
  const Eigen::Vector3d & w_0, const Eigen::Vector3d & a_0,
  const Eigen::Matrix3d & rotation_0, const Eigen::Matrix3d & rotation_f)
{
  Eigen::Matrix3d r_skew;
  r_skew = (rotation_0.transpose() * rotation_f).log();
  Eigen::Vector3d r;
  double tau = (time - time_0) / (time_f - time_0);
  r(0) = r_skew(2, 1);
  r(1) = r_skew(0, 2);
  r(2) = r_skew(1, 0);
  Eigen::Vector3d rd;
  for (int i = 0; i < 3; i++) {
    rd(i) = cubicDot(time, time_0, time_f, 0, r(i), 0, 0);
  }
  rd = rotation_0 * rd;
  if (tau < 0) {return w_0;}
  if (tau > 1) {return Eigen::Vector3d::Zero();}
  return rd;        // 3 * a * pow(tau, 2) + 2 * b * tau + c;
}
}      // namespace math_utils

}    // namespace common
}  // namespace cyclo_motion_controller
