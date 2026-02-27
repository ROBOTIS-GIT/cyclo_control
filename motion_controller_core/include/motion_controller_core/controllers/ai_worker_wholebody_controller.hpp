#pragma once

#include "motion_controller_core/optimization/qp_base.hpp"
#include "motion_controller_core/kinematics/kinematics_solver.hpp"
#include "motion_controller_core/common/type_define.h"
#include <vector>
#include <string>

namespace motion_controller
{
namespace controllers
{
    /**
     * @brief Class for solving inverse kinematics QP problems for manipulators.
     * 
     * This class inherits from QPBase and implements methods to set up and solve
     * inverse kinematics problems for manipulators using Quadratic Programming.
     */
    // NOTE: This is the wholebody (mobile-base + arms) QP-IK solver.
    // It intentionally has a different type name than the arm-only controller's QPIK.
    class WholebodyQPIK : public motion_controller::optimization::QPBase
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            struct BaseAccelerationConstraintConfig
            {
                bool enabled = true;
                // Limits are expressed in the robot base frame (body twist).
                double accel_vx_max = 0.0;  // m/s^2 (<=0 disables)
                double accel_vy_max = 0.0;  // m/s^2 (<=0 disables)
                double accel_wz_max = 0.0;  // rad/s^2 (<=0 disables)
            };

            
            /**
             * @brief Constructor.
             * @param robot_data (std::shared_ptr<KinematicsSolver>) Shared pointer to the KinematicsSolver class.
             * @param dt (double) Control loop time step in seconds.
             */
            WholebodyQPIK(std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data, const double dt);
            /**
             * @brief Set the wight vector for the cost terms
             * @param link_w_tracking (std::map<std::string, Vector6d>) Weight for task space velocity tracking per links.
             * @param w_damping  (Eigen::VectorXd) Weight for joint velocity damping; its size must same as dof.
             */
            void setWeight(
                const std::map<std::string, motion_controller::common::Vector6d> link_w_tracking,
                const Eigen::VectorXd w_damping);
            /**
             * @brief Set the desired task space velocity for the link.
             * @param link_xdot_desired (std::map<std::string, Vector6d>) Desired task space velocity (6D twist) per links.
             */ 
            void setDesiredTaskVel(const std::map<std::string, motion_controller::common::Vector6d> &link_xdot_desired);
            /**
             * @brief Set controller parameters.
             * @param slack_penalty    (double) Slack penalty.
             * @param cbf_alpha        (double) CBF alpha.
             * @param buffer_distance  (double) Distance buffer to activate constraints.
             * @param safe_distance    (double) Minimum safe distance.
             */
            void setControllerParams(const double slack_penalty, const double cbf_alpha, const double buffer_distance, const double safe_distance);
            /**
             * @brief Get the optimal joint velocity by solving QP.
             * @param opt_qdot    (Eigen::VectorXd) Optimal joint velocity.
             * @return (bool) True if the problem was solved successfully.
             */
            bool getOptJointVel(Eigen::VectorXd &opt_qdot);

            /**
             * @brief Enable/disable first-order actuation lag model for the mobile base joints.
             *
             * When enabled, task tracking uses an "effective" joint velocity:
             *   qdot_eff = gain .* qdot_cmd + (1-gain) .* qdot_meas
             * where gain is derived from a first-order time constant.
             */
            void enableBaseActuationLagComp(const bool enabled);

            /**
             * @brief Set the time constant (seconds) for base velocity tracking.
             *
             * gain = 1 - exp(-dt/tau). Larger tau => smaller gain (slower response).
             */
            void setBaseActuationTimeConstant(const double tau_seconds);

            void setBaseAccelerationConstraintConfig(const BaseAccelerationConstraintConfig & cfg)
            {
                base_accel_cfg_ = cfg;
            }

            void resetBaseAccelerationConstraintState()
            {
                prev_base_cmd_initialized_ = false;
                prev_base_cmd_body_.setZero();
            }

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
                int con_sing_start;    // singularity
                int con_sel_col_start; // self collision
                int con_base_accel_start; // base acceleration (body-frame)

                int con_q_min_size;
                int con_q_max_size;
                int con_sing_size;
                int con_sel_col_size;
                int con_base_accel_size;
            }si_index_;

            std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data_;  // Shared pointer to the robot data class.
            double dt_;                                           // control time step size
            int joint_dof_;                                       // Number of joints in the manipulator

            std::map<std::string, motion_controller::common::Vector6d> link_xdot_desired_; // Desired task velocity per links
            std::map<std::string, motion_controller::common::Vector6d> link_w_tracking_;   // weight for task velocity tracking per links; ||x_i_dot_des - J_i*q_dot||
            Eigen::VectorXd w_damping_;                                         // weight for joint velocity damping; || q_dot ||
            double slack_penalty_;
            double cbf_alpha_;
            double collision_buffer_;
            double collision_safe_distance_;

            // Base actuation lag compensation (wholebody base DOFs: x_joint, y_joint, pivot_joint)
            bool base_actuation_lag_comp_enabled_ = true;
            double base_actuation_tau_seconds_ = 0.15;
            int base_x_joint_index_ = -1;
            int base_y_joint_index_ = -1;
            int base_pivot_joint_index_ = -1;
            Eigen::VectorXd qdot_cmd_gain_;  // elementwise gain in [0,1]

            // Base acceleration constraint state/config (body frame)
            BaseAccelerationConstraintConfig base_accel_cfg_;
            bool prev_base_cmd_initialized_ = false;
            Eigen::Vector3d prev_base_cmd_body_ = Eigen::Vector3d::Zero(); // [vx_body, vy_body, wz]

            std::vector<uint8_t> margin_exclude_mask_;

            void updateActuationGains_();

            /**
             * @brief Set the cost function which minimizes task space velocity error.
             */
            void setCost() override;
            /**
             * @brief Set the bound constraint which limits manipulator joint velocities and keeps all slack variables non-negative.
             */
            void setBoundConstraint() override;
            /**
             * @brief Set the inequality constraints which manipulator limit joint angles and avoid self collision by 1st-order CBF. (TODO: Add singularity constraints)
             */
            void setIneqConstraint() override;
            /**
             * @brief Not implemented.
             */
            void setEqConstraint() override;
    };
} // namespace controllers
} // namespace motion_controller
