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
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kinematics/kinematics_solver.hpp"

namespace cyclo_teleoperation
{
using ControlGroupId = uint8_t;
using ControlGroupMask = uint64_t;

constexpr ControlGroupId kInvalidControlGroup =
  std::numeric_limits<ControlGroupId>::max();

inline constexpr ControlGroupMask controlGroupBit(const ControlGroupId id)
{
  return id < 64 ? ControlGroupMask{1} << id : ControlGroupMask{0};
}

inline constexpr bool containsControlGroup(
  const ControlGroupMask mask, const ControlGroupId id)
{
  return (mask & controlGroupBit(id)) != 0;
}

struct ControlGroupConfiguration
{
  ControlGroupId id = kInvalidControlGroup;
  std::string name;
  std::vector<int> follower_joint_indices;
  std::string follower_eef;
  std::string leader_eef;
  struct AuxiliaryJoint
  {
    std::string joint_name;
    std::string pose_parameter_name;
  };
  std::vector<AuxiliaryJoint> auxiliary_joints;
};

using GroupAuxiliaryPositions = std::vector<Eigen::VectorXd>;

struct ControlGroupState
{
  double leader_duration = 0.0;
  uint64_t leader_sequence = 0;
  uint16_t selected_preset_id = 0;
};

struct TaskObjective
{
  std::string link_name;
  Eigen::Matrix<double, 6, 1> desired_velocity =
    Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> weight =
    Eigen::Matrix<double, 6, 1>::Ones();
};

// A controller-defined linear task in joint-velocity space. This supports relative-link
// tasks whose Jacobian cannot be represented by one end-effector link name.
struct LinearTaskObjective
{
  Eigen::MatrixXd jacobian;
  Eigen::VectorXd desired_velocity;
  Eigen::VectorXd weight;
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
  std::vector<LinearTaskObjective> linear_task_objectives;
  std::unordered_map<ControlGroupId, Eigen::VectorXd> auxiliary_position_targets;

  void reset(const int dof, const double damping)
  {
    desired_joint_velocity.setZero(dof);
    joint_tracking_weight.setZero(dof);
    damping_weight.setConstant(dof, damping);
    preferred_joint_velocity.setZero(dof);
    joint_position_limit_enabled.assign(dof, true);
    preferred_joint_velocity_weight = 0.0;
    task_objectives.clear();
    linear_task_objectives.clear();
    auxiliary_position_targets.clear();
  }
};

struct ModeContext
{
  const Eigen::VectorXd & follower_position;
  const Eigen::VectorXd & follower_velocity;
  const Eigen::VectorXd & measured_follower_position;
  const Eigen::VectorXd & leader_reference;
  const Eigen::VectorXd & leader_position;
  const GroupAuxiliaryPositions & measured_auxiliary_position;
  const std::vector<ControlGroupState> & group_states;
  ControlGroupMask requested_groups;
  ControlGroupMask enabled_groups;
  // Groups temporarily owned by a preset or final-initial-pose overlay. Modes can use
  // this to avoid producing an objective for a group while an overlay controls it.
  ControlGroupMask pose_sequence_groups;
  double now_seconds;
  double dt;
};

struct ModeConfiguration
{
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> follower_kinematics;
  std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> leader_kinematics;
  std::vector<ControlGroupConfiguration> control_groups;

  const ControlGroupConfiguration * findGroup(const std::string & name) const
  {
    for (const auto & group : control_groups) {
      if (group.name == name) {
        return &group;
      }
    }
    return nullptr;
  }
};
}  // namespace cyclo_teleoperation
