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

#include <memory>

#include "motion_controller_core/optimization/qp_base.hpp"
#include "motion_controller_core/kinematics/kinematics_solver.hpp"
#include "motion_controller_core/common/type_define.h"

namespace motion_controller
{
namespace controllers
{
    /**
     * @brief Class for applying QP filters for manipulators.
     *
     * This class inherits from QPBase and implements methods to set up and set
     * QP filter for avoiding joint limits, self collisions etc.
     */
class QPFilter : public motion_controller::optimization::QPBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            /**
             * @brief Constructor.
             * @param robot_data (std::shared_ptr<KinematicsSolver>) Shared pointer to the KinematicsSolver class.
             * @param dt (double) Control loop time step in seconds.
             */
  QPFilter(
    std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data,
    const double dt);
            /**
             * @brief Set the wight vector for the cost terms
             * @param link_w_tracking (std::map<std::string, Vector6d>) Weight for task space velocity tracking per links.
             * @param w_damping  (Eigen::VectorXd) Weight for joint velocity damping; its size must same as dof.
             */
  void setWeight(const Eigen::VectorXd & w_tracking, const Eigen::VectorXd & w_damping);
            /**
             * @brief Set the desired task space velocity for the link.
             * @param link_xdot_desired (std::map<std::string, Vector6d>) Desired task space velocity (6D twist) per links.
             */
  void setDesiredJointVel(const Eigen::VectorXd & qdot_desired);
            /**
             * @brief Set controller parameters.
             * @param slack_penalty    (double) Slack penalty.
             * @param cbf_alpha        (double) CBF alpha.
             * @param buffer_distance  (double) Distance buffer to activate constraints.
             * @param safe_distance    (double) Minimum safe distance.
             */
  void setControllerParams(
    const double slack_penalty, const double cbf_alpha,
    const double buffer_distance, const double safe_distance);
            /**
             * @brief Get the optimal joint velocity by solving QP.
             * @param opt_qdot    (Eigen::VectorXd) Optimal joint velocity.
             * @return (bool) True if the problem was solved successfully.
             */
  bool getOptJointVel(Eigen::VectorXd & opt_qdot);

private:
            /**
             * @brief Struct to hold the indices of the QP variables and constraints.
             */
  struct QPIndex
  {
                // decision variables
    int qdot_start;
    int slack_q_min_start;
    int slack_q_max_start;
    int slack_sing_start;
    int slack_sel_col_start;

    int qdot_size;
    int slack_q_min_size;
    int slack_q_max_size;
    int slack_sing_size;
    int slack_sel_col_size;

                // inequality
    int con_q_min_start;
    int con_q_max_start;
    int con_sing_start;                // singularity
    int con_sel_col_start;             // self collision

    int con_q_min_size;
    int con_q_max_size;
    int con_sing_size;
    int con_sel_col_size;
  } si_index_;

  std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data_;
  // Shared pointer to the robot data class.
  double dt_;  // control time step size
  int joint_dof_;  // Number of joints in the manipulator

  Eigen::VectorXd qdot_desired_;  // Desired joint velocity
  Eigen::VectorXd w_tracking_;
  // Weight for joint velocity tracking; || qdot - qdot_des ||
  Eigen::VectorXd w_damping_;  // weight for joint velocity damping; || qdot ||
  double slack_penalty_;
  double cbf_alpha_;
  double collision_buffer_;
  double collision_safe_distance_;

            /**
             * @brief Set the cost function which minimizes task space velocity error.
             */
  void setCost() override;
            /**
             * @brief Set the bound constraint which limits manipulator joint velocities and keeps all slack variables non-negative.
             */
  void setBoundConstraint() override;
            /**
             * @brief Set inequality constraints for joint limits and self collision.
             * TODO(Yeonguk Kim): Add singularity constraints.
             */
  void setIneqConstraint() override;
            /**
             * @brief Not implemented.
             */
  void setEqConstraint() override;
};
}  // namespace controllers
}  // namespace motion_controller
