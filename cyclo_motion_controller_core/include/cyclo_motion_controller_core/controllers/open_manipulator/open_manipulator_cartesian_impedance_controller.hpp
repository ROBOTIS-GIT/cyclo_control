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

#pragma once

#include <memory>
#include <string>

#include "common/type_define.hpp"
#include "dynamics/dynamics_solver.hpp"
#include "optimization/qp_base.hpp"

namespace cyclo_motion_controller
{
namespace controllers
{

class OpenManipulatorCartesianImpedanceController : public cyclo_motion_controller::optimization::QPBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OpenManipulatorCartesianImpedanceController(
    std::shared_ptr<cyclo_motion_controller::dynamics::DynamicsSolver> robot_data,
    const std::string & controlled_link);

  void setControlledLink(const std::string & controlled_link);
  const std::string & getControlledLink() const {return controlled_link_;}

  void setDesiredPose(const Eigen::Affine3d & desired_pose);

  void setDesiredPoseAndVelocity(
    const Eigen::Affine3d & desired_pose,
    const cyclo_motion_controller::common::Vector6d & desired_twist);

  void setGains(
    const cyclo_motion_controller::common::Vector6d & stiffness,
    const cyclo_motion_controller::common::Vector6d & damping);

  void setNullspaceGains(
    const Eigen::Ref<const Eigen::VectorXd> & stiffness,
    const Eigen::Ref<const Eigen::VectorXd> & damping);

  void setNullspaceReference(const Eigen::Ref<const Eigen::VectorXd> & q_desired);

  void setTorqueWeight(const Eigen::Ref<const Eigen::VectorXd> & torque_weight);

  void setTorqueLimits(
    const Eigen::Ref<const Eigen::VectorXd> & lower,
    const Eigen::Ref<const Eigen::VectorXd> & upper);

  bool getCommand(Eigen::VectorXd & torque_command);

private:
  std::shared_ptr<cyclo_motion_controller::dynamics::DynamicsSolver> robot_data_;
  std::string controlled_link_;
  int joint_dof_;

  Eigen::Affine3d desired_pose_;
  cyclo_motion_controller::common::Vector6d desired_twist_;
  cyclo_motion_controller::common::Vector6d stiffness_;
  cyclo_motion_controller::common::Vector6d damping_;
  Eigen::VectorXd nullspace_stiffness_;
  Eigen::VectorXd nullspace_damping_;
  Eigen::VectorXd nullspace_q_desired_;
  Eigen::VectorXd torque_weight_;
  Eigen::VectorXd torque_lb_;
  Eigen::VectorXd torque_ub_;
  Eigen::VectorXd torque_desired_;

  void updateDesiredTorque();
  void setCost() override;
  void setBoundConstraint() override;
  void setIneqConstraint() override;
  void setEqConstraint() override;
};

}  // namespace controllers
}  // namespace cyclo_motion_controller
