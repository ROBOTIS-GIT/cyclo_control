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

#include "kinematics/kinematics_solver.hpp"

#include <iostream>
#include <stdexcept>

namespace cyclo_motion_controller
{
namespace kinematics
{

KinematicsSolver::KinematicsSolver(const std::string & urdf_path, const std::string & srdf_path)
: urdf_path_(urdf_path), srdf_path_(srdf_path)
{
  const bool has_srdf_path = !srdf_path.empty();

  if (!std::filesystem::exists(urdf_path)) {
    throw std::runtime_error("URDF file does not exist: " + urdf_path);
  }
  if (has_srdf_path && !std::filesystem::exists(srdf_path)) {
    throw std::runtime_error("SRDF file does not exist: " + srdf_path);
  }

  pinocchio::urdf::buildModel(urdf_path, model_, /*verbose=*/false);
  data_ = pinocchio::Data(model_);

  std::clog << "\033[33m" << "Collision model for the robot may not be perfect!" << "\033[0m" <<
    std::endl;
  pinocchio::urdf::buildGeom(model_, urdf_path, pinocchio::COLLISION, geom_model_);

  geom_model_.addAllCollisionPairs();

  if (has_srdf_path) {
    pinocchio::srdf::removeCollisionPairs(model_, geom_model_, srdf_path);
  } else {
    std::clog << "\033[33m" << "No SRDF path provided." << "\033[0m" << std::endl;
    std::clog << "\033[33m" << "All collision pairs remain active." << "\033[0m" << std::endl;
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
  for (pinocchio::FrameIndex i = 0; i < model_.frames.size(); ++i) {
    const auto & f = model_.frames[i];

            // LINK frames (URDF <link>)
    if (f.type == pinocchio::FrameType::BODY && f.name != "universe") {
      link_frame_names_.push_back(f.name);
      link_frame_set_.insert(f.name);

                // Root link: BODY directly attached to universe joint
      if (f.parentJoint == 0 && root_link_name_.empty()) {
        root_link_name_ = f.name;
      }
    } else if (f.type == pinocchio::FrameType::JOINT) {
      // JOINT frames (URDF <joint>)
      joint_frame_names_.push_back(f.name);
      joint_frame_set_.insert(f.name);
    }
  }

  if (link_frame_names_.empty()) {
    throw std::runtime_error("No BODY frames found in model");
  }

  // Fallback root link
  if (root_link_name_.empty()) {
    root_link_name_ = link_frame_names_.front();
  }
}

KinematicsSolver::~KinematicsSolver()
{
}

bool KinematicsSolver::updateState(const Eigen::VectorXd & q, const Eigen::VectorXd & qdot)
{
  q_ = q;
  qdot_ = qdot;

  if (!updateKinematics(q_, qdot_)) {
    return false;
  }
  return true;
}

bool KinematicsSolver::setJointVelocityBoundsByIndex(
  const int idx, const double lower,
  const double upper)
{
  if (idx < 0 || idx >= dof_) {
    return false;
  }
  qdot_lb_[idx] = lower;
  qdot_ub_[idx] = upper;
  return true;
}

bool KinematicsSolver::updateKinematics(const Eigen::VectorXd & q, const Eigen::VectorXd & qdot)
{
  pinocchio::forwardKinematics(model_, data_, q, qdot);
  pinocchio::computeJointJacobians(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);

  return true;
}

    // ================================ Compute Functions ================================

Eigen::Affine3d KinematicsSolver::computePose(
  const Eigen::VectorXd & q,
  const std::string & link_name)
{
  pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
  if (link_index >= model_.frames.size()) {
    throw std::runtime_error("Link name not found in URDF: " + link_name);
  }

  pinocchio::Data data = pinocchio::Data(model_);
  pinocchio::framesForwardKinematics(model_, data, q);
  Eigen::Affine3d link_pose;
  link_pose.matrix() = data.oMf[link_index].toHomogeneousMatrix();

  return link_pose;
}

Eigen::MatrixXd KinematicsSolver::computeJacobian(
  const Eigen::VectorXd & q,
  const std::string & link_name)
{
  pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
  if (link_index >= model_.frames.size()) {
    throw std::runtime_error("Link name not found in URDF: " + link_name);
  }

  Eigen::MatrixXd J;
  J.setZero(6, dof_);
  pinocchio::Data data = pinocchio::Data(model_);
  pinocchio::computeJointJacobians(model_, data, q);
  pinocchio::computeFrameJacobian(model_, data, q, link_index,
                                        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);

  return J;
}

    // ================================ Get Functions ================================
bool KinematicsSolver::hasLinkFrame(const std::string & name) const
{
  return link_frame_set_.count(name) > 0;
}

bool KinematicsSolver::hasJointFrame(const std::string & name) const
{
  return joint_frame_set_.count(name) > 0;
}

std::vector<std::string> KinematicsSolver::getJointNames() const
{
        // Return joint names in the order of the generalized coordinates (q)
  std::vector<std::string> joint_names;
  joint_names.reserve(static_cast<size_t>(dof_));
  for (pinocchio::JointIndex i = 1; i < model_.joints.size(); ++i) {
    const auto & joint = model_.joints[i];
    if (joint.nq() > 0) {
      joint_names.push_back(model_.names[i]);
    }
  }
  return joint_names;
}

Eigen::Affine3d KinematicsSolver::getPose(const std::string & link_name) const
{
  pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
  if (link_index >= model_.frames.size()) {
    throw std::runtime_error("Link name not found in URDF: " + link_name);
  }

  Eigen::Affine3d link_pose;
  link_pose.matrix() = data_.oMf[link_index].toHomogeneousMatrix();

  return link_pose;
}

Eigen::MatrixXd KinematicsSolver::getJacobian(const std::string & link_name)
{
  pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
  if (link_index >= model_.frames.size()) {
    throw std::runtime_error("Link name not found in URDF: " + link_name);
  }

  return pinocchio::getFrameJacobian(model_, data_, link_index,
                                           pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
}

int KinematicsSolver::getCollisionPairCount() const
{
  return static_cast<int>(geom_model_.collisionPairs.size());
}

std::vector<MinDistResult> KinematicsSolver::getCollisionPairDistances(
  const bool & with_grad,
  const bool & with_graddot,
  const bool verbose)
{
  std::vector<MinDistResult> results;
  results.resize(geom_model_.collisionPairs.size());
  for (auto & res : results) {
    res.setZero(q_.size());
  }

  if (geom_model_.collisionPairs.empty()) {
    return results;
  }

  pinocchio::computeDistances(model_, data_, geom_model_, geom_data_, q_);

  double minDistance = std::numeric_limits<double>::max();
  int minPairIdx = -1;
  for (std::size_t idx = 0; idx < geom_data_.distanceResults.size(); ++idx) {
    const auto & dist_res = geom_data_.distanceResults[idx];
    results[idx].distance = dist_res.min_distance;
    if (dist_res.min_distance < minDistance) {
      minDistance = dist_res.min_distance;
      minPairIdx = static_cast<int>(idx);
    }
  }

  if (minPairIdx >= 0 && verbose) {
    const auto & pair = geom_model_.collisionPairs[minPairIdx];
    const std::string & link1 =
      geom_model_.geometryObjects[pair.first].name;
    const std::string & link2 =
      geom_model_.geometryObjects[pair.second].name;

    std::cout       << "[KinematicsSolver] Closest links: " << link1
                    << "  <->  " << link2
                    << "   |  distance = " << minDistance << " [m]\n";
  }

  if (with_grad || with_graddot) {
    pinocchio::computeJointJacobians(model_, data_, q_);
    pinocchio::updateGeometryPlacements(model_, data_, geom_model_, geom_data_, q_);

    if (with_graddot) {
      pinocchio::computeJointJacobiansTimeVariation(model_, data_, q_, qdot_);
    }

    auto skew = [](const Eigen::Vector3d & v)->Eigen::Matrix3d{
        return (Eigen::Matrix3d() << 0, -v.z(), v.y(),
               v.z(), 0, -v.x(),
               -v.y(), v.x(), 0).finished();
      };

    for (std::size_t idx = 0; idx < geom_data_.distanceResults.size(); ++idx) {
      const auto & pair = geom_model_.collisionPairs[idx];
      const int geomA = pair.first, geomB = pair.second;
      const int jointA = geom_model_.geometryObjects[geomA].parentJoint;
      const int jointB = geom_model_.geometryObjects[geomB].parentJoint;

      const auto & dist_res = geom_data_.distanceResults[idx];
      const Eigen::Vector3d pA = dist_res.nearest_points[0];
      const Eigen::Vector3d pB = dist_res.nearest_points[1];
      Eigen::Vector3d n = pB - pA;
      const double n_norm = n.norm();
      if (n_norm > 0.0) {
        n /= n_norm;
      } else {
        n.setZero();
      }

      Eigen::MatrixXd J_jointA = Eigen::MatrixXd::Zero(6, q_.size());
      Eigen::MatrixXd J_jointB = Eigen::MatrixXd::Zero(6, q_.size());
      pinocchio::getJointJacobian(model_, data_, jointA, pinocchio::LOCAL_WORLD_ALIGNED, J_jointA);
      pinocchio::getJointJacobian(model_, data_, jointB, pinocchio::LOCAL_WORLD_ALIGNED, J_jointB);

      const Eigen::Vector3d rA = pA - data_.oMi[jointA].translation();
      const Eigen::Vector3d rB = pB - data_.oMi[jointB].translation();

      const Eigen::MatrixXd JA = J_jointA.topRows<3>() - skew(rA) * J_jointA.bottomRows<3>();
      const Eigen::MatrixXd JB = J_jointB.topRows<3>() - skew(rB) * J_jointB.bottomRows<3>();

      results[idx].grad = (n.transpose() * (JB - JA)).transpose();
      if (dist_res.min_distance < 0) {
        results[idx].grad *= -1.0;
      }

      if (with_graddot) {
        Eigen::MatrixXd J_jointA_dot = Eigen::MatrixXd::Zero(6, q_.size());
        Eigen::MatrixXd J_jointB_dot = Eigen::MatrixXd::Zero(6, q_.size());
        pinocchio::getJointJacobianTimeVariation(model_, data_, jointA,
              pinocchio::LOCAL_WORLD_ALIGNED, J_jointA_dot);
        pinocchio::getJointJacobianTimeVariation(model_, data_, jointB,
              pinocchio::LOCAL_WORLD_ALIGNED, J_jointB_dot);

        const Eigen::Vector3d pA_dot = JA * qdot_;
        const Eigen::Vector3d pB_dot = JB * qdot_;

        const Eigen::Vector3d rA_dot = pA_dot - J_jointA.topRows<3>() * qdot_;
        const Eigen::Vector3d rB_dot = pB_dot - J_jointB.topRows<3>() * qdot_;

        const Eigen::MatrixXd JA_dot = J_jointA_dot.topRows<3>() -
          (skew(rA_dot) * J_jointA.bottomRows<3>() + skew(rA) * J_jointA_dot.bottomRows<3>());
        const Eigen::MatrixXd JB_dot = J_jointB_dot.topRows<3>() -
          (skew(rB_dot) * J_jointB.bottomRows<3>() + skew(rB) * J_jointB_dot.bottomRows<3>());

        results[idx].grad_dot = (n.transpose() * (JB_dot - JA_dot)).transpose();
      }
    }
  }

  return results;
}

}  // namespace kinematics
}  // namespace cyclo_motion_controller
