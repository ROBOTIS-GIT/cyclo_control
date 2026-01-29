#include "motion_controller_core/kinematics_solver.hpp"

#include <iostream>
#include <stdexcept>

namespace motion_controller_core
{
    KinematicsSolver::KinematicsSolver(const std::string& urdf_path, const std::string& srdf_path)
    : urdf_path_(urdf_path), srdf_path_(srdf_path)
    {
        if (!std::filesystem::exists(urdf_path))
        {
            throw std::runtime_error("URDF file does not exist: " + urdf_path);
        }
        if (!std::filesystem::exists(srdf_path))
        {
            throw std::runtime_error("SRDF file does not exist: " + srdf_path);
        }

        pinocchio::urdf::buildModel(urdf_path, model_, /*verbose=*/false);
        data_ = pinocchio::Data(model_);

        std::clog << "\033[33m" << "Collision model for the robot may not be perfect!" << "\033[0m" << std::endl;
        pinocchio::urdf::buildGeom(model_, urdf_path, pinocchio::COLLISION, geom_model_);
        
        geom_model_.addAllCollisionPairs();

        if (std::filesystem::exists(srdf_path))
        {
            pinocchio::srdf::removeCollisionPairs(model_, geom_model_, srdf_path);
        }
        else
        {
            std::clog << "\033[33m" << "SRDF file does not exist! : " << "\033[0m" << srdf_path << "\033[0m" << std::endl;
            std::clog << "\033[33m" << "All the collision pairs is activated!" << "\033[0m" << std::endl;
        }

        geom_data_ = pinocchio::GeometryData(geom_model_);

        // Use nq to match position vector size in Pinocchio
        dof_ = static_cast<int>(model_.nq);

        // Initialize joint space state
        q_.setZero(dof_);
        qdot_.setZero(dof_);

        // // Set joint state limits (truncate to nq)
        // if (static_cast<int>(model_.lowerPositionLimit.size()) < dof_ ||
        //     static_cast<int>(model_.upperPositionLimit.size()) < dof_) {
        //     throw std::runtime_error("Position limit size is smaller than model nq.");
        // }
        q_lb_ = model_.lowerPositionLimit.head(dof_);
        q_ub_ = model_.upperPositionLimit.head(dof_);

        // if (static_cast<int>(model_.velocityLimit.size()) < dof_) {
        //     throw std::runtime_error("Velocity limit size is smaller than model nq.");
        // }
        qdot_ub_ = model_.velocityLimit.head(dof_);
        qdot_lb_ = -qdot_ub_;

        // Clear frame collections
        link_frame_names_.clear();
        joint_frame_names_.clear();
        link_frame_set_.clear();
        joint_frame_set_.clear();
        root_link_name_.clear();

        // Collect frames
        for (pinocchio::FrameIndex i = 0; i < model_.frames.size(); ++i)
        {
            const auto& f = model_.frames[i];

            // LINK frames (URDF <link>)
            if (f.type == pinocchio::FrameType::BODY && f.name != "universe")
            {
                link_frame_names_.push_back(f.name);
                link_frame_set_.insert(f.name);

                // Root link: BODY directly attached to universe joint
                if (f.parentJoint == 0 && root_link_name_.empty())
                {
                    root_link_name_ = f.name;
                }
            }
            // JOINT frames (URDF <joint>)
            else if (f.type == pinocchio::FrameType::JOINT)
            {
                joint_frame_names_.push_back(f.name);
                joint_frame_set_.insert(f.name);
            }
        }

        if (link_frame_names_.empty())
        {
            throw std::runtime_error("No BODY frames found in model");
        }

        // Fallback root link
        if (root_link_name_.empty())
        {
            root_link_name_ = link_frame_names_.front();
        }
    }

    KinematicsSolver::~KinematicsSolver()
    {
        
    }

    bool KinematicsSolver::updateState(const VectorXd& q, const VectorXd& qdot)
    {
        // if (q.size() != q_.size() || qdot.size() != qdot_.size())
        // {
        //     throw std::runtime_error("updateState: size mismatch for q/qdot.");
        // }
        q_ = q;
        qdot_ = qdot;
        
        if(!updateKinematics(q_, qdot_)) return false;
        return true;
    }

    bool KinematicsSolver::updateKinematics(const VectorXd& q, const VectorXd& qdot)
    {
        pinocchio::computeJointJacobians(model_, data_, q);
        
        return true;
    }

    // ================================ Compute Functions ================================

    Affine3d KinematicsSolver::computePose(const VectorXd& q, const std::string& link_name)
    {
        pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
        if (link_index == static_cast<pinocchio::FrameIndex>(-1))
        {
            throw std::runtime_error("Link name not found in URDF: " + link_name);
        }
        
        pinocchio::Data data = pinocchio::Data(model_);
        pinocchio::framesForwardKinematics(model_, data, q);
        Affine3d link_pose;
        link_pose.matrix() = data.oMf[link_index].toHomogeneousMatrix();
        
        return link_pose;
    }

    MatrixXd KinematicsSolver::computeJacobian(const VectorXd& q, const std::string& link_name)
    {
        pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
        if (link_index == static_cast<pinocchio::FrameIndex>(-1))
        {
            throw std::runtime_error("Link name not found in URDF: " + link_name);
        }
        
        MatrixXd J;
        J.setZero(6, dof_);
        pinocchio::Data data = pinocchio::Data(model_);
        pinocchio::computeJointJacobians(model_, data, q);
        pinocchio::computeFrameJacobian(model_, data, q, link_index, 
                                        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
    
        return J;
    }

    MinDistResult KinematicsSolver::computeMinDistance(const VectorXd& q, const VectorXd& qdot, const bool& with_grad, const bool& with_graddot, const bool verbose)
    {
        MinDistResult result;
        result.setZero(q.size());
        
        pinocchio::Data data = pinocchio::Data(model_);
        pinocchio::GeometryData geom_data = pinocchio::GeometryData(geom_model_);
        
        pinocchio::computeDistances(model_, data, geom_model_, geom_data, q);
        
        double minDistance = std::numeric_limits<double>::max();
        int    minPairIdx  = -1;
        
        for (std::size_t idx = 0; idx < geom_data.distanceResults.size(); ++idx)
        {
            const auto &res = geom_data.distanceResults[idx];
            if (res.min_distance < minDistance)
            {
                minDistance = res.min_distance;
                minPairIdx  = static_cast<int>(idx);
            }
        }
        result.distance = minDistance;
    
        if (minPairIdx >= 0 && verbose)
        {
            const auto &pair   = geom_model_.collisionPairs[minPairIdx];
            const std::string &link1 =
                geom_model_.geometryObjects[pair.first].name;
            const std::string &link2 =
                geom_model_.geometryObjects[pair.second].name;
    
            std::cout << "[RobotDataBase] Closest links: " << link1
                    << "  <->  " << link2
                    << "   |  distance = " << minDistance << " [m]\n";
        }
    
        if(with_grad || with_graddot)
        {
            pinocchio::computeJointJacobians(model_, data, q);
            pinocchio::updateGeometryPlacements(model_, data, geom_model_, geom_data, q);
    
            const auto &pair  = geom_model_.collisionPairs[minPairIdx];
            const int geomA = pair.first,  geomB = pair.second;
            const int jointA = geom_model_.geometryObjects[geomA].parentJoint;
            const int jointB = geom_model_.geometryObjects[geomB].parentJoint;
    
            // Witness points & normal (world frame)
            const auto &res = geom_data.distanceResults[minPairIdx];
            const Vector3d pA = res.nearest_points[0];
            const Vector3d pB = res.nearest_points[1];
            const Vector3d n  = (pB - pA).normalized();
    
            // Joint-space 6 × total_dof Jacobians for the two parent joints
            MatrixXd J_jointA = MatrixXd::Zero(6, q.size());
            MatrixXd J_jointB = MatrixXd::Zero(6, q.size());
            pinocchio::getJointJacobian(model_, data, jointA, pinocchio::LOCAL_WORLD_ALIGNED, J_jointA);
            pinocchio::getJointJacobian(model_, data, jointB, pinocchio::LOCAL_WORLD_ALIGNED, J_jointB);
    
            // r = point - joint-origin (world)
            const Vector3d rA = pA - data.oMi[jointA].translation();
            const Vector3d rB = pB - data.oMi[jointB].translation();
    
            // Linear part of point Jacobian: Jp = Jv + ω×r
            auto skew = [](const Vector3d &v)->Matrix3d{
                        return (Matrix3d() <<   0, -v.z(),  v.y(),
                                            v.z(),      0, -v.x(),
                                        -v.y(),  v.x(),     0).finished(); };
    
            const MatrixXd JA = J_jointA.topRows<3>() - skew(rA) * J_jointA.bottomRows<3>();
            const MatrixXd JB = J_jointB.topRows<3>() - skew(rB) * J_jointB.bottomRows<3>();
    
            // d/dq (‖pB - pA‖) = nᵀ (J_B - J_A)
            result.grad =( n.transpose() * (JB - JA)).transpose();   // total_dof × 1
            if(minDistance < 0) result.grad *= -1.;
    
            if(with_graddot)
            {
                pinocchio::computeJointJacobiansTimeVariation(model_, data_, q, qdot);
    
                MatrixXd J_jointA_dot = MatrixXd::Zero(6, q.size());
                MatrixXd J_jointB_dot = MatrixXd::Zero(6, q.size());
                pinocchio::getJointJacobianTimeVariation(model_, data, jointA, pinocchio::LOCAL_WORLD_ALIGNED, J_jointA_dot);
                pinocchio::getJointJacobianTimeVariation(model_, data, jointB, pinocchio::LOCAL_WORLD_ALIGNED, J_jointB_dot);
    
                const Vector3d pA_dot = JA * qdot;
                const Vector3d pB_dot = JB * qdot;
    
                const Vector3d rA_dot = pA_dot - J_jointA.topRows<3>() * qdot;
                const Vector3d rB_dot = pB_dot - J_jointB.topRows<3>() * qdot;
    
                const MatrixXd JA_dot = J_jointA_dot.topRows<3>() - (skew(rA_dot) * J_jointA.bottomRows<3>() + skew(rA) * J_jointA_dot.bottomRows<3>());
                const MatrixXd JB_dot = J_jointB_dot.topRows<3>() - (skew(rB_dot) * J_jointB.bottomRows<3>() + skew(rB) * J_jointB_dot.bottomRows<3>());
    
                result.grad_dot = (n.transpose() * (JB_dot - JA_dot)).transpose(); // neglect n_dot term
            }
        }
    
        return result;  
    }

    // bool solveIK(const Isometry3d& target_pose, const VectorXd& q_init, VectorXd& q_out) override
    // {
    //     if (q_init.size() != model_.nq) return false;
        
    //     VectorXd q = q_init;
    //     const double eps = 1e-6;
    //     const int max_iter = 100;
    //     const double dt = 0.1; 
    //     const double damp = 1e-12;
        
    //     pinocchio::SE3 baseMtarget(target_pose.linear(), target_pose.translation());
    //     pinocchio::SE3 oMtarget = oMbase_inv_.inverse() * baseMtarget;
        
    //     for(int i=0; i<max_iter; ++i) {
    //         pinocchio::forwardKinematics(model_, data_, q);
    //         pinocchio::updateFramePlacements(model_, data_);
            
    //         const pinocchio::SE3& current_pose = data_.oMf[tip_frame_id_];
    //         pinocchio::Motion err = pinocchio::log6(current_pose.inverse() * oMtarget);
            
    //         if(err.toVector().norm() < eps) {
    //             q_out = q;
    //             return true;
    //         }
            
    //         pinocchio::computeFrameJacobian(model_, data_, q, tip_frame_id_, pinocchio::LOCAL, J_);
            
    //         pinocchio::Data::Matrix6x Jlog;
    //         pinocchio::Jlog6(current_pose.inverse() * oMtarget, Jlog);
    //         J_ = -Jlog * J_;
            
    //         pinocchio::Data::Matrix6 JJt;
    //         JJt.noalias() = J_ * J_.transpose();
    //         JJt.diagonal().array() += damp;
            
    //         VectorXd v = -J_.transpose() * JJt.ldlt().solve(err.toVector());
    //         q = pinocchio::integrate(model_, q, v * dt);
    //     }
        
    //     return false;
    // }

    // ================================ Get Functions ================================
    bool KinematicsSolver::hasLinkFrame(const std::string& name) const
    {
        return link_frame_set_.count(name) > 0;
    }

    bool KinematicsSolver::hasJointFrame(const std::string& name) const
    {
        return joint_frame_set_.count(name) > 0;
    }

    std::vector<std::string> KinematicsSolver::getJointNames() const
    {
        // Return joint names in the order of the generalized coordinates (q)
        std::vector<std::string> joint_names;
        joint_names.reserve(static_cast<size_t>(dof_));
        for (pinocchio::JointIndex i = 1; i < model_.joints.size(); ++i)
        {
            const auto& joint = model_.joints[i];
            if (joint.nq() > 0) {
                joint_names.push_back(model_.names[i]);
            }
        }
        return joint_names;
    }

    Affine3d KinematicsSolver::getPose(const std::string& link_name) const
    {
        pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
        if (link_index == static_cast<pinocchio::FrameIndex>(-1))
        {
            throw std::runtime_error("Link name not found in URDF: " + link_name);
        }
        
        Affine3d link_pose;
        link_pose.matrix() = data_.oMf[link_index].toHomogeneousMatrix();
    
        return link_pose;
    }
        
    MatrixXd KinematicsSolver::getJacobian(const std::string& link_name)
    {
        pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
        if (link_index == static_cast<pinocchio::FrameIndex>(-1))
        {
            throw std::runtime_error("Link name not found in URDF: " + link_name);
        }
    
        return pinocchio::getFrameJacobian(model_, data_, link_index, 
                                           pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
    }

    MinDistResult KinematicsSolver::getMinDistance(const bool& with_grad, const bool& with_graddot, const bool verbose)
    {
        MinDistResult result;
        result.setZero(q_.size());
    
        pinocchio::computeDistances(model_, data_, geom_model_, geom_data_, q_);
    
        double minDistance = std::numeric_limits<double>::max();
        int    minPairIdx  = -1;
    
        for (std::size_t idx = 0; idx < geom_data_.distanceResults.size(); ++idx)
        {
            const auto &res = geom_data_.distanceResults[idx];
            if (res.min_distance < minDistance)
            {
                minDistance = res.min_distance;
                minPairIdx  = static_cast<int>(idx);
            }
        }
        result.distance = minDistance;
    
        if (minPairIdx >= 0 && verbose)
        {
            const auto &pair   = geom_model_.collisionPairs[minPairIdx];
            const std::string &link1 =
                geom_model_.geometryObjects[pair.first].name;
            const std::string &link2 =
                geom_model_.geometryObjects[pair.second].name;
    
            std::cout << "[RobotDataBase] Closest links: " << link1
                    << "  <->  " << link2
                    << "   |  distance = " << minDistance << " [m]\n";
        }
    
        if(with_grad || with_graddot)
        {
            pinocchio::updateGeometryPlacements(model_, data_, geom_model_, geom_data_, q_);
    
            const auto &pair  = geom_model_.collisionPairs[minPairIdx];
            const int   geomA = pair.first,  geomB = pair.second;
            const int   jointA = geom_model_.geometryObjects[geomA].parentJoint;
            const int   jointB = geom_model_.geometryObjects[geomB].parentJoint;
    
            // Witness points & normal (world frame)
            const auto &res = geom_data_.distanceResults[minPairIdx];
            const Vector3d pA = res.nearest_points[0];
            const Vector3d pB = res.nearest_points[1];
            const Vector3d n  = (pB - pA).normalized();
    
            // Joint-space 6 × total_dof Jacobians for the two parent joints
            MatrixXd J_jointA = MatrixXd::Zero(6, q_.size());
            MatrixXd J_jointB = MatrixXd::Zero(6, q_.size());
            pinocchio::getJointJacobian(model_, data_, jointA, pinocchio::LOCAL_WORLD_ALIGNED, J_jointA);
            pinocchio::getJointJacobian(model_, data_, jointB, pinocchio::LOCAL_WORLD_ALIGNED, J_jointB);
    
            // r = point - joint-origin (world)
            const Vector3d rA = pA - data_.oMi[jointA].translation();
            const Vector3d rB = pB - data_.oMi[jointB].translation();
    
            // Linear part of point Jacobian: Jp = Jv + ω×r
            auto skew = [](const Vector3d &v)->Matrix3d{
                        return (Matrix3d() <<   0, -v.z(),  v.y(),
                                            v.z(),      0, -v.x(),
                                        -v.y(),  v.x(),     0).finished(); };
    
            const MatrixXd JA = J_jointA.topRows<3>() - skew(rA) * J_jointA.bottomRows<3>();
            const MatrixXd JB = J_jointB.topRows<3>() - skew(rB) * J_jointB.bottomRows<3>();
    
            // d/dq (‖pB - pA‖) = nᵀ (J_B - J_A)
            result.grad =( n.transpose() * (JB - JA)).transpose();   // total_dof × 1
            if(minDistance < 0) result.grad *= -1.;
    
            if(with_graddot)
            {
                MatrixXd J_jointA_dot = MatrixXd::Zero(6, q_.size());
                MatrixXd J_jointB_dot = MatrixXd::Zero(6, q_.size());
                pinocchio::getJointJacobianTimeVariation(model_, data_, jointA, pinocchio::LOCAL_WORLD_ALIGNED, J_jointA_dot);
                pinocchio::getJointJacobianTimeVariation(model_, data_, jointB, pinocchio::LOCAL_WORLD_ALIGNED, J_jointB_dot);
    
                const Vector3d pA_dot = JA * qdot_;
                const Vector3d pB_dot = JB * qdot_;
    
                const Vector3d rA_dot = pA_dot - J_jointA.topRows<3>() * qdot_;
                const Vector3d rB_dot = pB_dot - J_jointB.topRows<3>() * qdot_;
    
                const MatrixXd JA_dot = J_jointA_dot.topRows<3>() - (skew(rA_dot) * J_jointA.bottomRows<3>() + skew(rA) * J_jointA_dot.bottomRows<3>());
                const MatrixXd JB_dot = J_jointB_dot.topRows<3>() - (skew(rB_dot) * J_jointB.bottomRows<3>() + skew(rB) * J_jointB_dot.bottomRows<3>());
    
                result.grad_dot = (n.transpose() * (JB_dot - JA_dot)).transpose(); // neglect n_dot term
            }
        }
    
        return result;     
    }

} // namespace motion_controller_core
