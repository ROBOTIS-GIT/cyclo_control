// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0

#pragma once

#include <map>
#include <memory>
#include <string>

#include "common/type_define.hpp"
#include "kinematics/kinematics_solver.hpp"
#include "optimization/qp_base.hpp"

namespace cyclo_motion_controller
{
namespace controllers
{
class ElbowUpQPController : public cyclo_motion_controller::optimization::QPBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ElbowUpQPController(
    std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> robot_data);

  void setDesiredTaskVel(
    const std::map<std::string, cyclo_motion_controller::common::Vector6d> & desired);
  void setWeights(
    const std::map<std::string, cyclo_motion_controller::common::Vector6d> & tracking,
    const Eigen::VectorXd & damping);
  void setPreferredJointVel(const Eigen::VectorXd & preferred, double weight);
  void setControllerParams(
    double slack_penalty, double cbf_alpha, double collision_buffer,
    double collision_safe_distance);
  bool getOptJointVel(Eigen::VectorXd & qdot);

private:
  struct Index
  {
    int qdot_start = 0;
    int slack_q_min_start = 0;
    int slack_q_max_start = 0;
    int slack_collision_start = 0;
    int qdot_size = 0;
    int slack_q_min_size = 0;
    int slack_q_max_size = 0;
    int slack_collision_size = 0;
    int con_q_min_start = 0;
    int con_q_max_start = 0;
    int con_collision_start = 0;
    int con_q_min_size = 0;
    int con_q_max_size = 0;
    int con_collision_size = 0;
  } index_;

  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> robot_data_;
  std::map<std::string, cyclo_motion_controller::common::Vector6d> desired_task_velocity_;
  std::map<std::string, cyclo_motion_controller::common::Vector6d> tracking_weight_;
  Eigen::VectorXd damping_weight_;
  Eigen::VectorXd preferred_joint_velocity_;
  double preferred_joint_velocity_weight_ = 0.0;
  double slack_penalty_ = 1000.0;
  double cbf_alpha_ = 5.0;
  double collision_buffer_ = 0.05;
  double collision_safe_distance_ = 0.02;

  void setCost() override;
  void setBoundConstraint() override;
  void setIneqConstraint() override;
  void setEqConstraint() override;
};
}  // namespace controllers
}  // namespace cyclo_motion_controller
