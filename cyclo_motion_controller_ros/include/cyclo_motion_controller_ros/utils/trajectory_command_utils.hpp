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

#include <string>

#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace cyclo_motion_controller_ros
{
namespace trajectory_command_utils
{
template<typename JointIndexMapT>
bool hasJoint(const JointIndexMapT & joint_index_map, const std::string & joint_name)
{
  return joint_index_map.find(joint_name) != joint_index_map.end();
}

inline bool updateJointPosition(
  const trajectory_msgs::msg::JointTrajectory & msg,
  const std::string & joint_name,
  double & position)
{
  if (msg.points.empty()) {
    return false;
  }

  const auto & point = msg.points.front();
  for (size_t i = 0; i < msg.joint_names.size(); ++i) {
    if (msg.joint_names[i] != joint_name) {
      continue;
    }
    if (i >= point.positions.size()) {
      return false;
    }
    position = point.positions[i];
    return true;
  }
  return false;
}

inline bool updateJointPosition(
  const sensor_msgs::msg::JointState & msg,
  const std::string & joint_name,
  double & position)
{
  for (size_t i = 0; i < msg.name.size(); ++i) {
    if (msg.name[i] != joint_name) {
      continue;
    }
    if (i >= msg.position.size()) {
      return false;
    }
    position = msg.position[i];
    return true;
  }
  return false;
}

template<typename JointIndexMapT>
void appendJointIfPresent(
  trajectory_msgs::msg::JointTrajectory & msg,
  const JointIndexMapT & joint_index_map,
  const std::string & joint_name,
  const double position)
{
  if (!hasJoint(joint_index_map, joint_name) || msg.points.empty()) {
    return;
  }

  msg.joint_names.push_back(joint_name);
  for (auto & point : msg.points) {
    point.positions.push_back(position);
    if (!point.velocities.empty()) {
      point.velocities.push_back(0.0);
    }
    if (!point.accelerations.empty()) {
      point.accelerations.push_back(0.0);
    }
    if (!point.effort.empty()) {
      point.effort.push_back(0.0);
    }
  }
}
}  // namespace trajectory_command_utils
}  // namespace cyclo_motion_controller_ros
