#include "motion_controller_core/controllers/ai_worker/ai_worker_controller.hpp"
#include <algorithm>

namespace motion_controller
{
namespace controllers
{

        QPIK::QPIK(std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data, const double dt)
        : motion_controller::optimization::QPBase(), robot_data_(robot_data), dt_(dt)
        {
            joint_dof_ = robot_data_->getDof();
    
            si_index_.qdot_size            = joint_dof_;
            si_index_.slack_q_min_size     = joint_dof_;
            si_index_.slack_q_max_size     = joint_dof_;
            si_index_.slack_sing_size      = 1;
            si_index_.slack_sel_col_size   = robot_data_->getCollisionPairCount();
            si_index_.con_q_min_size       = joint_dof_;
            si_index_.con_q_max_size       = joint_dof_;
            si_index_.con_sing_size        = 1;
            si_index_.con_sel_col_size     = robot_data_->getCollisionPairCount();
            
            const int nx = si_index_.qdot_size +
                           si_index_.slack_q_min_size +
                           si_index_.slack_q_max_size +
                           si_index_.slack_sing_size +
                           si_index_.slack_sel_col_size;
            const int nbound = nx;
            const int nineq = si_index_.con_q_min_size +
                              si_index_.con_q_max_size +
                              si_index_.con_sing_size +
                              si_index_.con_sel_col_size;
            const int neq = 0;
            
            QPBase::setQPsize(nx, nbound, nineq, neq);
            
            si_index_.qdot_start          = 0;
            si_index_.slack_q_min_start   = si_index_.qdot_start + si_index_.qdot_size;
            si_index_.slack_q_max_start   = si_index_.slack_q_min_start + si_index_.slack_q_min_size;
            si_index_.slack_sing_start    = si_index_.slack_q_max_start + si_index_.slack_q_max_size;
            si_index_.slack_sel_col_start = si_index_.slack_sing_start + si_index_.slack_sing_size;
            si_index_.con_q_min_start     = 0;
            si_index_.con_q_max_start     = si_index_.con_q_min_start + si_index_.con_q_min_size;
            si_index_.con_sing_start      = si_index_.con_q_max_start + si_index_.con_q_max_size;
            si_index_.con_sel_col_start   = si_index_.con_sing_start + si_index_.con_sing_size;

            w_damping_.setOnes(joint_dof_);
        }
    
        void QPIK::setDesiredTaskVel(const std::map<std::string, motion_controller::common::Vector6d> &link_xdot_desired)
        {
            link_xdot_desired_ = link_xdot_desired;
        }
    
        bool QPIK::getOptJointVel(Eigen::VectorXd &opt_qdot)
        {
            Eigen::MatrixXd sol;
            if(!solveQP(sol))
            {
                opt_qdot.setZero();
                return false;
            }
            else
            {
                opt_qdot = sol.block(si_index_.qdot_start,0,si_index_.qdot_size,1);
                return true;
            }
        }

        void QPIK::setWeight(
            const std::map<std::string, motion_controller::common::Vector6d> link_w_tracking,
            const Eigen::VectorXd w_damping)
        {
            link_w_tracking_ = link_w_tracking;
            w_damping_ = w_damping;
        }

        void QPIK::setControllerParams(const double slack_penalty, const double cbf_alpha, const double buffer_distance, const double safe_distance)
        {
            slack_penalty_ = slack_penalty;
            cbf_alpha_ = cbf_alpha;
            collision_buffer_ = buffer_distance;
            collision_safe_distance_ = safe_distance;
        }
    
        void QPIK::setCost()
        {
            P_ds_.setZero(nx_, nx_);
            q_ds_.setZero(nx_);

            // for task space velocity tracking
            for(const auto& [link_name, xdot_desired] : link_xdot_desired_)
            {
                Eigen::MatrixXd J_i = robot_data_->getJacobian(link_name);
                motion_controller::common::Vector6d w_tracking = motion_controller::common::Vector6d::Ones();

                auto iter = link_w_tracking_.find(link_name);
                if(iter != link_w_tracking_.end()) {
                    w_tracking = iter->second;
                }

                P_ds_.block(si_index_.qdot_start,si_index_.qdot_start,si_index_.qdot_size,si_index_.qdot_size) += 2.0 * J_i.transpose() * w_tracking.asDiagonal() * J_i;
                q_ds_.segment(si_index_.qdot_start,si_index_.qdot_size) += -2.0 * J_i.transpose() * w_tracking.asDiagonal() * xdot_desired;
            }
            
            // for joint velocity damping
            P_ds_.block(si_index_.qdot_start,si_index_.qdot_start,si_index_.qdot_size,si_index_.qdot_size) += 2.0 * w_damping_.asDiagonal();
            
            // for slack variables
            q_ds_.segment(si_index_.slack_q_min_start, si_index_.slack_q_min_size) =
                Eigen::VectorXd::Constant(si_index_.slack_q_min_size, slack_penalty_);
            q_ds_.segment(si_index_.slack_q_max_start, si_index_.slack_q_max_size) =
                Eigen::VectorXd::Constant(si_index_.slack_q_max_size, slack_penalty_);
            q_ds_(si_index_.slack_sing_start) = slack_penalty_;
            if (si_index_.slack_sel_col_size > 0) {
                q_ds_.segment(si_index_.slack_sel_col_start, si_index_.slack_sel_col_size) =
                    Eigen::VectorXd::Constant(si_index_.slack_sel_col_size, slack_penalty_);
            }
        }
    
        void QPIK::setBoundConstraint()
        {
            l_bound_ds_.setConstant(nbc_, -OSQP_INFTY);
            u_bound_ds_.setConstant(nbc_, OSQP_INFTY);

            // Manipulator Joint Velocity Limit
            l_bound_ds_.segment(si_index_.qdot_start, si_index_.qdot_size) = 
                robot_data_->getJointVelocityLimit().first;
            u_bound_ds_.segment(si_index_.qdot_start, si_index_.qdot_size) = 
                robot_data_->getJointVelocityLimit().second;

            // Slack variables must be non-negative
            l_bound_ds_.segment(si_index_.slack_q_min_start, si_index_.slack_q_min_size).setZero();
            l_bound_ds_.segment(si_index_.slack_q_max_start, si_index_.slack_q_max_size).setZero();
            l_bound_ds_(si_index_.slack_sing_start) = 0.0;
            if (si_index_.slack_sel_col_size > 0) {
                l_bound_ds_.segment(si_index_.slack_sel_col_start, si_index_.slack_sel_col_size).setZero();
            }
        }
    
        void QPIK::setIneqConstraint()
        {
            A_ineq_ds_.setZero(nineqc_, nx_);
            l_ineq_ds_.setConstant(nineqc_, -OSQP_INFTY);
            u_ineq_ds_.setConstant(nineqc_, OSQP_INFTY);

            // Manipulator Joint Angle Limit (CBF)
            const Eigen::VectorXd q_min = robot_data_->getJointPositionLimit().first;
            const Eigen::VectorXd q_max = robot_data_->getJointPositionLimit().second;
            const Eigen::VectorXd q = robot_data_->getJointPosition();
                
            // Lower bound constraint: qdot + slack >= -alpha*(q - q_min)
            A_ineq_ds_.block(si_index_.con_q_min_start, si_index_.qdot_start, 
                            si_index_.con_q_min_size, si_index_.qdot_size) = 
                Eigen::MatrixXd::Identity(si_index_.con_q_min_size, si_index_.qdot_size);
            A_ineq_ds_.block(si_index_.con_q_min_start, si_index_.slack_q_min_start, 
                            si_index_.con_q_min_size, si_index_.slack_q_min_size) = 
                Eigen::MatrixXd::Identity(si_index_.con_q_min_size, si_index_.slack_q_min_size);
            l_ineq_ds_.segment(si_index_.con_q_min_start, si_index_.con_q_min_size) = 
                -cbf_alpha_ * (q - q_min);
            
            // Upper bound constraint: -qdot + slack >= -alpha*(q_max - q)
            A_ineq_ds_.block(si_index_.con_q_max_start, si_index_.qdot_start, 
                            si_index_.con_q_max_size, si_index_.qdot_size) = 
                -Eigen::MatrixXd::Identity(si_index_.con_q_max_size, si_index_.qdot_size);
            A_ineq_ds_.block(si_index_.con_q_max_start, si_index_.slack_q_max_start, 
                            si_index_.con_q_max_size, si_index_.slack_q_max_size) = 
                Eigen::MatrixXd::Identity(si_index_.con_q_max_size, si_index_.slack_q_max_size);
            l_ineq_ds_.segment(si_index_.con_q_max_start, si_index_.con_q_max_size) = 
                -cbf_alpha_ * (q_max - q);

            // self collision avoidance (CBF)
            if (si_index_.con_sel_col_size > 0)
            {
                const auto pair_results = robot_data_->getCollisionPairDistances(true, false, false);
                const int pair_count = std::min<int>(si_index_.con_sel_col_size, pair_results.size());
                for (int i = 0; i < pair_count; ++i)
                {
                    const auto &res = pair_results[i];
                    A_ineq_ds_.block(si_index_.con_sel_col_start + i, si_index_.qdot_start, 1, si_index_.qdot_size) =
                        res.grad.transpose();
                    if (si_index_.slack_sel_col_size > 0 && i < si_index_.slack_sel_col_size) {
                        A_ineq_ds_(si_index_.con_sel_col_start + i, si_index_.slack_sel_col_start + i) = 1.0;
                    }
                    if (res.distance <= collision_buffer_)
                    {
                        l_ineq_ds_(si_index_.con_sel_col_start + i) =
                            -cbf_alpha_ * (res.distance - collision_safe_distance_);
                    }
                }
            }
        }
    
        void QPIK::setEqConstraint()
        {
            A_eq_ds_.setZero(neqc_, nx_);
            b_eq_ds_.setZero(neqc_);
        }

} // namespace controllers
} // namespace motion_controller
