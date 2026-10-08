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
#pragma once
#include <vector>
#include "cyclo_teleoperation/core/robot_teleoperation.hpp"

namespace cyclo_teleoperation
{
// Feedback freshness and open-loop command state have a single owner. Receiving
// a sample never silently overwrites a command; callers explicitly rebase at events.
class FeedbackState
{
public:
  struct Reception {bool recovering; bool interrupted;};
  void configure(size_t groups);
  bool fresh(rclcpp::Time now, double timeout) const;
  Reception accept(rclcpp::Time now, double timeout);
  bool expire(const RobotTeleoperation & robot);
  void clearError() {error_reported_ = false;}
  bool commandInitialized() const {return command_initialized_;}
  void invalidateCommand() {command_initialized_ = false;}
  bool holdInitialized() const {return hold_initialized_;}
  void setHoldInitialized(bool value) {hold_initialized_ = value;}
  void resetOwnership() {previous_controlled_groups_ = 0;}
  void resetCommand(const RobotTeleoperation & robot);
  void initializeAuxiliary(const RobotTeleoperation & robot);
  void syncGroups(ControlGroupMask groups, const RobotTeleoperation & robot);
  void captureHold(ControlGroupMask groups, const RobotTeleoperation & robot);
  void updateOwnership(ControlGroupMask groups, const RobotTeleoperation & robot);
  void acceptLeader(ControlGroupId group, rclcpp::Time now);
  void clearLeaders();
  ControlGroupMask freshLeaders(
    rclcpp::Time now, double timeout, const ModeConfiguration & config) const;

  Eigen::VectorXd & position() {return position_;}
  Eigen::VectorXd & velocity() {return velocity_;}
  Eigen::VectorXd & hold() {return hold_;}
  const Eigen::VectorXd & position() const {return position_;}
  const Eigen::VectorXd & velocity() const {return velocity_;}
  GroupAuxiliaryPositions & auxiliary() {return auxiliary_;}
  GroupAuxiliaryPositions & auxiliaryHold() {return auxiliary_hold_;}

private:
  bool received_ = false, error_reported_ = false;
  bool command_initialized_ = false, hold_initialized_ = false;
  ControlGroupMask previous_controlled_groups_ = 0;
  rclcpp::Time last_time_{0, 0, RCL_ROS_TIME};
  std::vector<bool> leader_received_;
  std::vector<rclcpp::Time> leader_times_;
  Eigen::VectorXd position_, velocity_, hold_;
  GroupAuxiliaryPositions auxiliary_, auxiliary_hold_;
};
}  // namespace cyclo_teleoperation
