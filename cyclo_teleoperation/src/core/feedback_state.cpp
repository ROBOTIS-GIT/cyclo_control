// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "cyclo_teleoperation/core/feedback_state.hpp"
#include <algorithm>

namespace cyclo_teleoperation
{
void FeedbackState::configure(size_t groups)
{
  leader_received_.assign(groups, false);
  leader_times_.assign(groups, rclcpp::Time(0, 0, RCL_ROS_TIME));
}

bool FeedbackState::fresh(rclcpp::Time now, double timeout) const
{
  return received_ && (now - last_time_).seconds() <= timeout;
}

FeedbackState::Reception FeedbackState::accept(rclcpp::Time now, double timeout)
{
  const bool interrupted = received_ && !fresh(now, timeout);
  const bool recovering = !command_initialized_ || !fresh(now, timeout);
  if (interrupted) {invalidateCommand(); resetOwnership();}
  last_time_ = now;
  received_ = true;
  return {recovering, interrupted};
}

bool FeedbackState::expire(const RobotTeleoperation & robot)
{
  if (!received_ || error_reported_) {return false;}
  error_reported_ = true;
  hold_ = robot.followerPosition();
  invalidateCommand();
  resetOwnership();
  return true;
}

void FeedbackState::initializeAuxiliary(const RobotTeleoperation & robot)
{
  auxiliary_ = robot.followerAuxiliaryPosition();
  auxiliary_hold_ = auxiliary_;
}

void FeedbackState::resetCommand(const RobotTeleoperation & robot)
{
  position_ = robot.followerPosition();
  velocity_ = Eigen::VectorXd::Zero(robot.dof());
  initializeAuxiliary(robot);
  command_initialized_ = true;
}

void FeedbackState::syncGroups(ControlGroupMask groups, const RobotTeleoperation & robot)
{
  for (const auto & group : robot.modeConfiguration().control_groups) {
    if (!containsControlGroup(groups, group.id)) {continue;}
    for (const int index : group.follower_joint_indices) {
      position_[index] = robot.followerPosition()[index];
      velocity_[index] = 0.0;
    }
    auxiliary_[group.id] = robot.followerAuxiliaryPosition()[group.id];
  }
}

void FeedbackState::captureHold(ControlGroupMask groups, const RobotTeleoperation & robot)
{
  for (const auto & group : robot.modeConfiguration().control_groups) {
    if (!containsControlGroup(groups, group.id)) {continue;}
    for (const int index : group.follower_joint_indices) {
      hold_[index] = robot.followerPosition()[index];
    }
    auxiliary_hold_[group.id] = robot.followerAuxiliaryPosition()[group.id];
  }
}

void FeedbackState::updateOwnership(ControlGroupMask groups, const RobotTeleoperation & robot)
{
  const auto released = previous_controlled_groups_ & ~groups;
  if (released != 0) {captureHold(released, robot); syncGroups(released, robot);}
  previous_controlled_groups_ = groups;
}

void FeedbackState::acceptLeader(ControlGroupId group, rclcpp::Time now)
{
  leader_received_.at(group) = true;
  leader_times_.at(group) = now;
}

void FeedbackState::clearLeaders()
{
  std::fill(leader_received_.begin(), leader_received_.end(), false);
}

ControlGroupMask FeedbackState::freshLeaders(
  rclcpp::Time now, double timeout, const ModeConfiguration & config) const
{
  ControlGroupMask result = 0;
  for (const auto & group : config.control_groups) {
    if (group.id < leader_received_.size() && leader_received_[group.id] &&
      (now - leader_times_[group.id]).seconds() <= timeout)
    {
      result |= controlGroupBit(group.id);
    }
  }
  return result;
}
}  // namespace cyclo_teleoperation
