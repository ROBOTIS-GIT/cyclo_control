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

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "cyclo_teleoperation/core/control_interface.hpp"
#include "cyclo_teleoperation/core/types.hpp"

namespace cyclo_teleoperation
{
struct LeaderInputChannel
{
  ControlGroupId group_id = kInvalidControlGroup;
  std::string topic;
};

class RobotTeleoperation
{
public:
  // Control-group IDs are stable indices in [0, 63]. State vectors returned by
  // a robot plugin must be indexable by every configured group ID.
  virtual ~RobotTeleoperation() = default;

  virtual bool configure(
    rclcpp::Node & node,
    const std::string & parameter_prefix,
    ControlInterface::RequestCallback request_callback) = 0;

  virtual std::string robotName() const = 0;
  virtual int dof() const = 0;

  virtual const Eigen::VectorXd & followerPosition() const = 0;
  virtual const Eigen::VectorXd & followerVelocity() const = 0;
  virtual const Eigen::VectorXd & leaderReference() const = 0;
  virtual const Eigen::VectorXd & leaderPosition() const = 0;
  virtual const std::vector<ControlGroupState> & controlGroupStates() const = 0;

  virtual const ModeConfiguration & modeConfiguration() const = 0;
  virtual std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver>
  followerKinematics() const = 0;
  virtual std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver>
  leaderKinematics() const = 0;

  virtual std::string followerJointStatesTopic() const = 0;
  virtual const std::vector<LeaderInputChannel> & leaderInputChannels() const = 0;

  virtual bool updateFollowerState(
    const sensor_msgs::msg::JointState & message) = 0;
  virtual bool updateLeaderReference(
    const trajectory_msgs::msg::JointTrajectory & message,
    ControlGroupId target_group) = 0;
  virtual void publish(const Eigen::VectorXd & command) = 0;
  virtual void publishStatus(const ControlStatus & status) = 0;
};
}  // namespace cyclo_teleoperation
