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
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "cyclo_teleoperation/core/teleoperation_mode.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
class CameraAssistMoveJMode : public TeleoperationMode
{
public:
  bool configure(
    rclcpp::Node & node,
    const std::string & parameter_prefix,
    const ModeConfiguration & configuration) override;
  bool activate(const ModeContext & context) override;
  void onGroupsEnabled(
    ControlGroupMask groups, const ModeContext & context) override;
  bool update(const ModeContext & context, ModeOutput & output) override;
  ControlGroupMask timedCommandFeedbackSyncGroups(
    const ModeContext & context) const override;
  ControlGroupMask controlledGroups(const ModeContext & context) const override;

private:
  struct ArmFrames
  {
    std::string camera_link;
    Eigen::Vector3d camera_origin_offset = Eigen::Vector3d::Zero();
    Eigen::Vector3d camera_forward_axis = Eigen::Vector3d::UnitX();
    std::string gripper_link;
    Eigen::Vector3d gripper_target_offset = Eigen::Vector3d::Zero();
    double circle_angle = 0.7853981633974483;
  };

  struct LeaderTrajectory
  {
    Eigen::VectorXd start;
    Eigen::VectorXd goal;
    double start_time = 0.0;
    double duration = 0.0;
    uint64_t last_sequence = 0;
    bool waiting_for_command = false;
    bool active = false;
    bool slow_start_complete = false;
  };

  struct CameraTaskError
  {
    double distance = 0.0;
    double circle_offset = 0.0;
    double gaze_angle = 0.0;
    bool valid = false;
  };

  struct CameraTrajectory
  {
    Eigen::Vector3d start_direction = Eigen::Vector3d::UnitX();
    Eigen::Vector3d goal_direction = Eigen::Vector3d::UnitX();
    double start_distance = 0.0;
    double start_circle_offset = 0.0;
    double start_time = 0.0;
    double duration = 0.0;
    uint64_t last_sequence = 0;
    bool waiting_for_command = false;
    bool active = false;
    bool slow_start_complete = false;
  };

  ControlGroupMask cameraGroup(const ModeContext & context) const;
  void beginRoleTransition(const ModeContext & context);
  void initializeLeaderTrajectory(
    ControlGroupMask group, const ModeContext & context, LeaderTrajectory & trajectory);
  void initializeCameraTrajectory(
    ControlGroupMask camera_group, const ModeContext & context);
  void updateLeaderArm(
    ControlGroupMask group,
    const std::vector<int> & indices,
    double command_duration,
    uint64_t command_sequence,
    LeaderTrajectory & trajectory,
    const ModeContext & context,
    ModeOutput & output);
  void addCameraAssistTask(
    const ArmFrames & camera,
    const ArmFrames & target,
    const std::vector<int> & camera_indices,
    const std::vector<int> & target_indices,
    const ModeContext & context,
    ModeOutput & output);
  static uint64_t cameraCommandSequence(
    ControlGroupMask camera_group, const ModeContext & context);
  CameraTaskError cameraTaskError(
    const ArmFrames & camera, const ArmFrames & target) const;
  double cameraSlowStartDuration(
    const CameraTaskError & task_error,
    double joint_tracking_error,
    double dt) const;
  void updateSlowStartState(
    const ModeContext & context, ControlGroupMask camera_group);
  static double meanAbsoluteError(
    const std::vector<int> & indices,
    const Eigen::VectorXd & actual,
    const Eigen::VectorXd & reference);
  static Eigen::MatrixXd pointJacobian(
    const Eigen::MatrixXd & frame_jacobian,
    const Eigen::Matrix3d & frame_rotation,
    const Eigen::Vector3d & local_offset);

  ModeConfiguration configuration_;
  std::vector<int> left_indices_;
  std::vector<int> right_indices_;
  ArmFrames left_frames_;
  ArmFrames right_frames_;
  LeaderTrajectory left_trajectory_;
  LeaderTrajectory right_trajectory_;
  CameraTrajectory camera_trajectory_;
  rclcpp::Logger logger_{rclcpp::get_logger("camera_assist_movej_mode")};

  ControlGroupMask role_enabled_groups_ = 0;
  ControlGroupMask role_camera_group_ = 0;
  ControlGroupMask role_pose_sequence_groups_ = 0;
  bool role_initialized_ = false;
  bool slow_start_enabled_ = true;
  bool slow_start_active_ = false;

  double kp_joint_ = 50.0;
  double tracking_weight_ = 10.0;
  double target_distance_ = 0.2;
  double distance_kp_ = 5.0;
  double distance_weight_ = 1000.0;
  double gaze_kp_ = 5.0;
  double gaze_weight_ = 1000.0;
  double leader_arm_sync_threshold_ = 0.03;
  double camera_arm_sync_threshold_ = 0.1;
  double camera_distance_sync_threshold_ = 0.03;
  double camera_gaze_sync_threshold_ = 0.2;
};
}  // namespace cyclo_teleoperation::robots::ai_worker
