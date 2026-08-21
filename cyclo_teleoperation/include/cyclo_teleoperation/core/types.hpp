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

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kinematics/kinematics_solver.hpp"

namespace cyclo_teleoperation
{
constexpr uint8_t kLeftArm = 1;
constexpr uint8_t kRightArm = 2;
constexpr uint8_t kBothArms = kLeftArm | kRightArm;

struct TaskObjective
{
  std::string link_name;
  Eigen::Matrix<double, 6, 1> desired_velocity =
    Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> weight =
    Eigen::Matrix<double, 6, 1>::Ones();
};

struct ModeOutput
{
  Eigen::VectorXd desired_joint_velocity;
  Eigen::VectorXd joint_tracking_weight;
  Eigen::VectorXd damping_weight;
  Eigen::VectorXd preferred_joint_velocity;
  std::vector<bool> joint_position_limit_enabled;
  double preferred_joint_velocity_weight = 0.0;
  std::vector<TaskObjective> task_objectives;

  void reset(const int dof, const double damping)
  {
    desired_joint_velocity.setZero(dof);
    joint_tracking_weight.setZero(dof);
    damping_weight.setConstant(dof, damping);
    preferred_joint_velocity.setZero(dof);
    joint_position_limit_enabled.assign(dof, true);
    preferred_joint_velocity_weight = 0.0;
    task_objectives.clear();
  }
};

struct ModeContext
{
  const Eigen::VectorXd & follower_position;
  const Eigen::VectorXd & follower_velocity;
  const Eigen::VectorXd & measured_follower_position;
  const Eigen::VectorXd & leader_reference;
  const Eigen::VectorXd & leader_position;
  double left_leader_duration;
  double right_leader_duration;
  uint64_t left_leader_sequence;
  uint64_t right_leader_sequence;
  uint8_t requested_arms;
  uint8_t enabled_arms;
  uint16_t left_preset_id;
  uint16_t right_preset_id;
  double now_seconds;
  double dt;
};

struct ModeConfiguration
{
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> follower_kinematics;
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> leader_kinematics;
  std::vector<int> left_arm_indices;
  std::vector<int> right_arm_indices;
  std::string follower_left_eef;
  std::string follower_right_eef;
  std::string follower_left_elbow;
  std::string follower_right_elbow;
  std::string leader_left_eef;
  std::string leader_right_eef;
};
}  // namespace cyclo_teleoperation
