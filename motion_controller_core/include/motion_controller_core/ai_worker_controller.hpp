#pragma once
#include "motion_controller_core/QP_base.hpp"
#include "motion_controller_core/kinematics_solver.hpp"

using namespace Eigen;

namespace motion_controller_core
{
    /**
     * @brief Class for solving inverse kinematics QP problems for manipulators.
     * 
     * This class inherits from QPBase and implements methods to set up and solve
     * inverse kinematics problems for manipulators using Quadratic Programming.
     */
    class QPIK : public QP::QPBase
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            
            // Constants
            static constexpr double DEFAULT_SLACK_PENALTY = 1000.0;
            static constexpr double DEFAULT_CBF_ALPHA = 50.0;
            static constexpr double DEFAULT_MAX_SLACK = 1.0e-4;
            
            /**
             * @brief Constructor.
             * @param robot_data (std::shared_ptr<KinematicsSolver>) Shared pointer to the KinematicsSolver class.
             * @param dt (double) Control loop time step in seconds.
             */
            QPIK(std::shared_ptr<KinematicsSolver> robot_data, const double dt);
            /**
             * @brief Set the wight vector for the cost terms
             * @param link_w_tracking (std::map<std::string, Vector6d>) Weight for task space velocity tracking per links.
             * @param w_damping  (Eigen::VectorXd) Weight for joint velocity damping; its size must same as dof.
             */
            void setWeight(const std::map<std::string, Vector6d> link_w_tracking, const VectorXd w_damping);
            /**
             * @brief Set the desired task space velocity for the link.
             * @param link_xdot_desired (std::map<std::string, Vector6d>) Desired task space velocity (6D twist) per links.
             */ 
            void setDesiredTaskVel(const std::map<std::string, Vector6d> &link_xdot_desired);
            /**
             * @brief Get the optimal joint velocity by solving QP.
             * @param opt_qdot    (Eigen::VectorXd) Optimal joint velocity.
             * @return (bool) True if the problem was solved successfully.
             */
            bool getOptJointVel(VectorXd &opt_qdot);

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

                int con_q_min_size;
                int con_q_max_size;
                int con_sing_size;
                int con_sel_col_size;
            }si_index_;

            std::shared_ptr<KinematicsSolver> robot_data_;  // Shared pointer to the robot data class.
            double dt_;                                           // control time step size
            int joint_dof_;                                       // Number of joints in the manipulator

            std::map<std::string, Vector6d> link_xdot_desired_; // Desired task velocity per links
            std::map<std::string, Vector6d> link_w_tracking_;   // weight for task velocity tracking per links; ||x_i_dot_des - J_i*q_dot||
            VectorXd w_damping_;                                // weight for joint velocity damping;           || q_dot ||
                

            /**
             * @brief Set the cost function which minimizes task space velocity error.
             *        Use slack variables (s) to increase feasibility of QP.
             * 
             *       min      || x_i_dot_des - J_i*q_dot ||_Wi^2 + || q_dot ||_W2^2 + 1000*s
             *     [qdot,s]
             * 
             * =>    min     1/2 [ qdot ]^T * [ 2*J_i.T*Wi*J_i + 2*W2   0 ] * [ qdot ] + [ -2*J_i.T*Wi*x_i_dot_des ].T * [ qdot ]
             *     [qdot,s]      [   s  ]     [         0               0 ]   [   s  ]   [           1000          ]     [  s   ]
             */
            void setCost() override;
            /**
             * @brief Set the bound constraint which limits manipulator joint velocities and keeps all slack variables non-negative.
             * 
             *     subject to [ qdot_min ] <= [ qdot ] <= [ qdot_max ]
             *                [    0     ]    [   s  ]    [   inf    ]
             */
            void setBoundConstraint() override;
            /**
             * @brief Set the inequality constraints which manipulator limit joint angles and avoid singularity and self collision by 1st-order CBF.
             * 
             * 1st-order CBF condition with slack: hdot(x) >= -a*h(x) - s
             * 
             *  1.
             *     Manipulator joint angle limit: h_min(q) = q - q_min >= 0  -> hdot_min(q) = qdot
             *                                    h_max(q) = q_max - q >= 0  -> hdot_max(q) = -qdot
             *     
             *     => subject to [  I  I ] * [ qdot ] >= [ -a*(q - q_min) ]
             *                   [ -I  I ]   [  s   ]    [ -a*(q_max - q) ]
             */
            void setIneqConstraint() override;
            /**
             * @brief Not implemented.
             */
            void setEqConstraint() override;
    };
} // namespace motion_controller_core