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

#include <math.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>  // NOLINT(build/include_order)
#include <Eigen/Geometry>  // NOLINT(build/include_order)


// Pinocchio Includes
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/spatial/fcl-pinocchio-conversions.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>

#include "motion_controller_core/common/type_define.h"

namespace motion_controller
{
namespace kinematics
{
using motion_controller::common::collision_checker::MinDistResult;
using Eigen::Affine3d;
using Eigen::MatrixXd;
using Eigen::VectorXd;

  /**
  * @brief Generic Kinematics Solver class that provides FK and IK using a selectable backend.
  */
class KinematicsSolver
{
public:
  KinematicsSolver(const std::string & urdf_path, const std::string & srdf_path);
  ~KinematicsSolver();

    /**
    * @brief Update the state of the manipulator.
    * @param q     (Eigen::VectorXd) Joint positions.
    * @param qdot  (Eigen::VectorXd) Joint velocities.
    * @return (bool) True if state update is successful.
    */
  virtual bool updateState(const VectorXd & q, const VectorXd & qdot);

    // ================================ Compute Functions ================================
    /**
    * @brief Compute the pose of the link in the task space.
    * @param q         (Eigen::VectorXd) Joint positions.
    * @param link_name (std::string) Name of the link.
    * @return (Eigen::Affine3d) Pose of the link in the task space.
    */
  virtual Affine3d computePose(const VectorXd & q, const std::string & link_name);

    /**
    * @brief Compute the Jacobian of the link.
    * @param q         (Eigen::VectorXd) Joint positions.
    * @param link_name (std::string) Name of the link.
    * @return (Eigen::MatrixXd) Jacobian of the link.
    */
  virtual MatrixXd computeJacobian(const VectorXd & q, const std::string & link_name);

    // ================================ Get Functions ================================
  virtual const std::string getURDFPath() const {return urdf_path_;}

    // Link frames (BODY)
  virtual const std::vector<std::string> & getLinkFrameVector() const {return link_frame_names_;}
  virtual bool hasLinkFrame(const std::string & name) const;

    // Joint frames (JOINT)
  virtual const std::vector<std::string> & getJointFrameVector() const {return joint_frame_names_;}
  virtual bool hasJointFrame(const std::string & name) const;

    /**
     * @brief Get the actual joint names from the model (matching joint_states topic).
     * @return (std::vector<std::string>) Joint names from the model.
     */
  virtual std::vector<std::string> getJointNames() const;

    // Root link (base link default)
  virtual const std::string & getRootLinkName() const {return root_link_name_;}

    /**
     * @brief Get the degrees of freedom of the manipulator.
     * @return (int) Degrees of freedom of the manipulator.
     */
  virtual int getDof() const {return dof_;}

    /**
    * @brief Get the joint positions of the manipulator.
    * @return (Eigen::VectorXd) Joint positions of the manipulator.
    */
  virtual VectorXd getJointPosition() const {return q_;}

    /**
    * @brief Get the joint velocities of the manipulator.
    * @return (Eigen::VectorXd) Joint velocities of the manipulator.
    */
  virtual VectorXd getJointVelocity() const {return qdot_;}

    /**
    * @brief Get lower and upper joint position limits of the manipulator.
    * @return (std::pair<Eigen::VectorXd, Eigen::VectorXd>) Joint position limits (lower, upper) of the manipulator.
    */
  virtual std::pair<VectorXd, VectorXd> getJointPositionLimit() const
  {
    return std::make_pair(q_lb_, q_ub_);
  }

    /**
    * @brief Get lower and upper joint velocity limits of the manipulator.
    * @return (std::pair<Eigen::VectorXd, Eigen::VectorXd>) Joint velocity limits (lower, upper) of the manipulator.
    */
  virtual std::pair<VectorXd, VectorXd> getJointVelocityLimit() const
  {
    return std::make_pair(qdot_lb_, qdot_ub_);
  }

    /**
     * @brief Override joint velocity bounds for a specific generalized coordinate index.
     * @param idx   (int) Index in q/qdot (0-based).
     * @param lower (double) Lower velocity bound.
     * @param upper (double) Upper velocity bound.
     * @return (bool) True if index was valid and bounds were applied.
     */
  virtual bool setJointVelocityBoundsByIndex(const int idx, const double lower, const double upper);

    /**
    * @brief Get the pose of the link in the task space.
    * @param link_name (std::string) Name of the link.
    * @return (Eigen::Affine3d) Pose of the link in the task space.
    */
  Affine3d getPose(const std::string & link_name) const;

    /**
    * @brief Get the Jacobian of the link.
    * @param link_name (std::string) Name of the link.
    * @return (Eigen::MatrixXd) Jacobian of the link.
    */
  virtual MatrixXd getJacobian(const std::string & link_name);

    /**
     * @brief Get the number of collision pairs in the model.
     * @return (int) Number of collision pairs.
     */
  virtual int getCollisionPairCount() const;

    /**
     * @brief Get distance/gradient for all collision pairs.
     * @param with_grad     (bool) If true, get gradients for each pair.
     * @param with_graddot  (bool) If true, get gradient time variations.
     * @param verbose       (bool) If true, prints closest pair info.
     * @return (std::vector<MinDistResult>) Results per collision pair.
     */
  virtual std::vector<MinDistResult> getCollisionPairDistances(
    const bool & with_grad,
    const bool & with_graddot,
    const bool verbose);

protected:
    /**
    * @brief Update the kinematic parameters of the manipulator.
    * @param q     (Eigen::VectorXd) Joint positions.
    * @param qdot  (Eigen::VectorXd) Joint velocities.
    * @return (bool) True if the update was successful.
    */
  virtual bool updateKinematics(const VectorXd & q, const VectorXd & qdot);

  std::string urdf_path_;
  std::string srdf_path_;

  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::GeometryModel geom_model_;
  pinocchio::GeometryData geom_data_;

    // Cached frame name lists
  std::vector<std::string> link_frame_names_;     // URDF <link> names
  std::vector<std::string> joint_frame_names_;    // URDF <joint> names

  std::unordered_set<std::string> link_frame_set_;
  std::unordered_set<std::string> joint_frame_set_;

  std::string root_link_name_;

  int dof_;             // Total degrees of freedom.

  VectorXd q_;          // Manipulator joint positions.
  VectorXd qdot_;       // Manipulator joint velocities.
  VectorXd q_lb_;       // Lower joint position limits of the manipulator.
  VectorXd q_ub_;       // Upper joint position limits of the manipulator.
  VectorXd qdot_lb_;    // Lower joint velocity limits of the manipulator.
  VectorXd qdot_ub_;    // Upper joint velocity limits of the manipulator.
};

}  // namespace kinematics
}  // namespace motion_controller
