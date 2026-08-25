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

#include "cyclo_teleoperation/robots/ai_worker/camera_assist_movej_mode.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <pluginlib/class_list_macros.hpp>

#include "cyclo_teleoperation/robots/ai_worker/ai_worker_groups.hpp"
#include "common/type_define.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
bool CameraAssistMoveJMode::configure(
  rclcpp::Node & node,
  const std::string & prefix,
  const ModeConfiguration & configuration)
{
  configuration_ = configuration;
  const auto * left_group = configuration.findGroup("left");
  const auto * right_group = configuration.findGroup("right");
  if (left_group == nullptr || right_group == nullptr) {
    RCLCPP_ERROR(node.get_logger(), "Camera assist requires left and right control groups");
    return false;
  }
  left_indices_ = left_group->follower_joint_indices;
  right_indices_ = right_group->follower_joint_indices;
  logger_ = node.get_logger();

  auto double_parameter = [&node](const std::string & name, const double default_value) {
      if (!node.has_parameter(name)) {
        return node.declare_parameter(name, default_value);
      }
      return node.get_parameter(name).as_double();
    };
  auto bool_parameter = [&node](const std::string & name, const bool default_value) {
      if (!node.has_parameter(name)) {
        return node.declare_parameter(name, default_value);
      }
      return node.get_parameter(name).as_bool();
    };
  auto string_parameter = [&node](
    const std::string & name, const std::string & default_value)
    {
      if (!node.has_parameter(name)) {
        return node.declare_parameter(name, default_value);
      }
      return node.get_parameter(name).as_string();
    };
  auto vector_parameter = [&node](
    const std::string & name, const Eigen::Vector3d & default_value)
    {
      const std::vector<double> default_vector{
        default_value.x(), default_value.y(), default_value.z()};
      const std::vector<double> value = !node.has_parameter(name) ?
        node.declare_parameter<std::vector<double>>(name, default_vector) :
        node.get_parameter(name).as_double_array();
      if (value.size() != 3) {
        throw std::runtime_error(name + " must contain exactly three values");
      }
      return Eigen::Vector3d(value[0], value[1], value[2]);
    };

  slow_start_enabled_ = bool_parameter(prefix + ".slow_start.enabled", true);
  kp_joint_ = double_parameter(prefix + ".kp_joint", 50.0);
  tracking_weight_ = double_parameter(prefix + ".tracking_weight", 10.0);
  target_distance_ = double_parameter(prefix + ".target_distance", 0.2);
  distance_kp_ = double_parameter(prefix + ".distance_kp", 5.0);
  distance_weight_ = double_parameter(prefix + ".distance_weight", 1000.0);
  gaze_kp_ = double_parameter(prefix + ".gaze_kp", 5.0);
  gaze_weight_ = double_parameter(prefix + ".gaze_weight", 1000.0);
  leader_arm_sync_threshold_ =
    double_parameter(prefix + ".leader_arm_sync_threshold", 0.03);
  camera_arm_sync_threshold_ =
    double_parameter(prefix + ".camera_arm_sync_threshold", 0.1);
  camera_distance_sync_threshold_ =
    double_parameter(prefix + ".camera_distance_sync_threshold", 0.03);
  camera_gaze_sync_threshold_ =
    double_parameter(prefix + ".camera_gaze_sync_threshold", 0.2);

  auto load_frames = [&](const std::string & arm, const ArmFrames & defaults) {
      const std::string arm_prefix = prefix + ".camera_assist." + arm;
      ArmFrames frames;
      frames.camera_link = string_parameter(
        arm_prefix + ".camera_link", defaults.camera_link);
      frames.camera_origin_offset = vector_parameter(
        arm_prefix + ".camera_origin_offset", defaults.camera_origin_offset);
      frames.camera_forward_axis = vector_parameter(
        arm_prefix + ".camera_forward_axis", defaults.camera_forward_axis);
      frames.gripper_link = string_parameter(
        arm_prefix + ".gripper_link", defaults.gripper_link);
      frames.gripper_target_offset = vector_parameter(
        arm_prefix + ".gripper_target_offset", defaults.gripper_target_offset);
      const double axis_norm = frames.camera_forward_axis.norm();
      if (axis_norm <= 1e-8) {
        throw std::runtime_error(arm_prefix + ".camera_forward_axis must be non-zero");
      }
      frames.camera_forward_axis /= axis_norm;
      return frames;
    };

  ArmFrames left_defaults;
  left_defaults.camera_link = "camera_l_link";
  left_defaults.gripper_link = left_group->follower_eef;
  ArmFrames right_defaults;
  right_defaults.camera_link = "camera_r_link";
  right_defaults.gripper_link = right_group->follower_eef;
  left_frames_ = load_frames("left", left_defaults);
  right_frames_ = load_frames("right", right_defaults);

  for (const auto & frame : {
      left_frames_.camera_link, left_frames_.gripper_link,
      right_frames_.camera_link, right_frames_.gripper_link})
  {
    if (!configuration_.follower_kinematics->hasLinkFrame(frame)) {
      RCLCPP_ERROR(
        logger_, "Camera-assist frame does not exist in the follower model: %s", frame.c_str());
      return false;
    }
  }

  return kp_joint_ > 0.0 && tracking_weight_ > 0.0 &&
         target_distance_ > 0.0 && distance_kp_ > 0.0 &&
         distance_weight_ > 0.0 && gaze_kp_ > 0.0 && gaze_weight_ > 0.0 &&
         leader_arm_sync_threshold_ >= 0.0 && camera_arm_sync_threshold_ >= 0.0 &&
         camera_distance_sync_threshold_ >= 0.0 &&
         camera_gaze_sync_threshold_ >= 0.0;
}

bool CameraAssistMoveJMode::activate(const ModeContext & context)
{
  role_initialized_ = false;
  beginRoleTransition(context);
  return true;
}

void CameraAssistMoveJMode::onGroupsEnabled(
  const ControlGroupMask groups, const ModeContext & context)
{
  if ((groups & kBothGroups) != 0) {
    beginRoleTransition(context);
  }
}

ControlGroupMask CameraAssistMoveJMode::cameraGroup(const ModeContext & context) const
{
  const ControlGroupMask enabled = context.enabled_groups & kBothGroups;
  if (enabled == kLeftGroup && (context.pose_sequence_groups & kRightGroup) == 0) {
    return kRightGroup;
  }
  if (enabled == kRightGroup && (context.pose_sequence_groups & kLeftGroup) == 0) {
    return kLeftGroup;
  }
  return 0;
}

ControlGroupMask CameraAssistMoveJMode::controlledGroups(const ModeContext & context) const
{
  return (context.enabled_groups & ~context.pose_sequence_groups) | cameraGroup(context);
}

ControlGroupMask CameraAssistMoveJMode::timedCommandFeedbackSyncGroups(
  const ModeContext & context) const
{
  constexpr double kTimedCommandEpsilon = 1e-6;
  ControlGroupMask groups = 0;
  auto add_timed_arm = [&](
    const ControlGroupMask group,
    const double duration,
    const uint64_t sequence,
    const LeaderTrajectory & trajectory)
    {
      if (
        (context.enabled_groups & group) != 0 &&
        sequence != trajectory.last_sequence &&
        !trajectory.slow_start_complete &&
        duration > kTimedCommandEpsilon)
      {
        groups |= group;
      }
  };
  add_timed_arm(
    kLeftGroup, context.group_states.at(kLeftGroupId).leader_duration,
    context.group_states.at(kLeftGroupId).leader_sequence, left_trajectory_);
  add_timed_arm(
    kRightGroup, context.group_states.at(kRightGroupId).leader_duration,
    context.group_states.at(kRightGroupId).leader_sequence, right_trajectory_);

  if (!role_initialized_) {
    return groups;
  }
  const ControlGroupMask next_camera_group = cameraGroup(context);
  const bool role_changed =
    context.enabled_groups != role_enabled_groups_ ||
    context.pose_sequence_groups != role_pose_sequence_groups_ ||
    next_camera_group != role_camera_group_;
  // This is a one-shot initialization for a newly assigned camera trajectory. Once
  // update() accepts the new role, the command state advances open-loop and is never
  // reset from feedback on subsequent leader or camera samples.
  if (role_changed && next_camera_group != role_camera_group_) {
    groups |= next_camera_group;
  }
  return groups;
}

void CameraAssistMoveJMode::initializeLeaderTrajectory(
  const ControlGroupMask group,
  const ModeContext & context,
  LeaderTrajectory & trajectory)
{
  trajectory = LeaderTrajectory{};
  trajectory.start = context.follower_position;
  trajectory.goal = context.leader_reference;
  trajectory.last_sequence = group == kLeftGroup ?
    context.group_states.at(kLeftGroupId).leader_sequence :
    context.group_states.at(kRightGroupId).leader_sequence;
  trajectory.waiting_for_command = slow_start_enabled_;
  trajectory.slow_start_complete = !slow_start_enabled_;
}

uint64_t CameraAssistMoveJMode::cameraCommandSequence(
  const ControlGroupMask camera_group, const ModeContext & context)
{
  if (camera_group == kLeftGroup) {
    return context.group_states.at(kRightGroupId).leader_sequence;
  }
  if (camera_group == kRightGroup) {
    return context.group_states.at(kLeftGroupId).leader_sequence;
  }
  return 0;
}

void CameraAssistMoveJMode::initializeCameraTrajectory(
  const ControlGroupMask camera_group, const ModeContext & context)
{
  camera_trajectory_ = CameraTrajectory{};
  camera_trajectory_.last_sequence = cameraCommandSequence(camera_group, context);
  camera_trajectory_.waiting_for_command = slow_start_enabled_ && camera_group != 0;
  camera_trajectory_.slow_start_complete =
    !slow_start_enabled_ || camera_group == 0;
}

void CameraAssistMoveJMode::beginRoleTransition(const ModeContext & context)
{
  role_enabled_groups_ = context.enabled_groups & kBothGroups;
  role_pose_sequence_groups_ = context.pose_sequence_groups & kBothGroups;
  role_camera_group_ = cameraGroup(context);
  role_initialized_ = true;
  initializeLeaderTrajectory(kLeftGroup, context, left_trajectory_);
  initializeLeaderTrajectory(kRightGroup, context, right_trajectory_);
  initializeCameraTrajectory(role_camera_group_, context);
  slow_start_active_ = slow_start_enabled_ && controlledGroups(context) != 0;
  if (slow_start_active_) {
    CameraTaskError camera_error;
    if (role_camera_group_ == kLeftGroup) {
      camera_error = cameraTaskError(left_frames_, right_frames_);
    } else if (role_camera_group_ == kRightGroup) {
      camera_error = cameraTaskError(right_frames_, left_frames_);
    }
    RCLCPP_INFO(
      logger_,
      "Camera-assist independent slow start began "
      "(teleop arms=%u, camera arm=%u, distance error=%.3f m, gaze error=%.3f rad)",
      static_cast<unsigned int>(role_enabled_groups_),
      static_cast<unsigned int>(role_camera_group_),
      camera_error.valid ? camera_error.distance : 0.0,
      camera_error.valid ? camera_error.gaze_angle : 0.0);
  }
}

void CameraAssistMoveJMode::updateLeaderArm(
  const ControlGroupMask group,
  const std::vector<int> & indices,
  const double command_duration,
  const uint64_t command_sequence,
  LeaderTrajectory & trajectory,
  const ModeContext & context,
  ModeOutput & output)
{
  if ((context.enabled_groups & group) == 0) {
    return;
  }

  constexpr double kDurationEpsilon = 1e-6;
  if (trajectory.waiting_for_command && command_sequence == trajectory.last_sequence) {
    for (const int index : indices) {
      output.desired_joint_velocity[index] = 0.0;
      output.joint_tracking_weight[index] = tracking_weight_;
    }
    return;
  }
  if (command_sequence != trajectory.last_sequence) {
    trajectory.last_sequence = command_sequence;
    trajectory.waiting_for_command = false;
    if (!trajectory.slow_start_complete && command_duration > kDurationEpsilon) {
      // Keep this identical to MoveJMode: each adaptive timed command starts a
      // linear segment at follower feedback and ends at that command's leader reference.
      trajectory.start = context.follower_position;
      trajectory.goal = context.leader_reference;
      trajectory.start_time = context.now_seconds;
      trajectory.duration = command_duration;
      trajectory.active = true;
    } else {
      trajectory.active = false;
      trajectory.slow_start_complete = true;
    }
  }

  double interpolation_ratio = 1.0;
  Eigen::VectorXd feedforward = Eigen::VectorXd::Zero(context.follower_position.size());
  if (trajectory.active) {
    const double elapsed = std::max(0.0, context.now_seconds - trajectory.start_time);
    interpolation_ratio = std::clamp(elapsed / trajectory.duration, 0.0, 1.0);
    feedforward = (trajectory.goal - trajectory.start) / trajectory.duration;
    if (interpolation_ratio >= 1.0) {
      trajectory.active = false;
    }
  }

  for (const int index : indices) {
    const double reference = trajectory.active ?
      trajectory.start[index] + interpolation_ratio *
      (trajectory.goal[index] - trajectory.start[index]) :
      context.leader_reference[index];
    const double reference_velocity = trajectory.active ? feedforward[index] : 0.0;
    output.desired_joint_velocity[index] = reference_velocity +
      kp_joint_ * (reference - context.follower_position[index]);
    output.joint_tracking_weight[index] = tracking_weight_;
  }
}

Eigen::MatrixXd CameraAssistMoveJMode::pointJacobian(
  const Eigen::MatrixXd & frame_jacobian,
  const Eigen::Matrix3d & frame_rotation,
  const Eigen::Vector3d & local_offset)
{
  const Eigen::Vector3d world_offset = frame_rotation * local_offset;
  return frame_jacobian.topRows(3) -
         cyclo_motion_controller::common::skewSymmetric(world_offset) *
         frame_jacobian.bottomRows(3);
}

void CameraAssistMoveJMode::addCameraAssistTask(
  const ArmFrames & camera,
  const ArmFrames & target,
  const std::vector<int> & camera_indices,
  const std::vector<int> & target_indices,
  const ModeContext & context,
  ModeOutput & output)
{
  const Eigen::Affine3d camera_pose =
    configuration_.follower_kinematics->getPose(camera.camera_link);
  const Eigen::Affine3d target_pose =
    configuration_.follower_kinematics->getPose(target.gripper_link);
  const Eigen::MatrixXd camera_frame_jacobian =
    configuration_.follower_kinematics->getJacobian(camera.camera_link);
  const Eigen::MatrixXd target_frame_jacobian =
    configuration_.follower_kinematics->getJacobian(target.gripper_link);
  const Eigen::MatrixXd camera_point_jacobian = pointJacobian(
    camera_frame_jacobian, camera_pose.linear(), camera.camera_origin_offset);
  const Eigen::MatrixXd target_point_jacobian = pointJacobian(
    target_frame_jacobian, target_pose.linear(), target.gripper_target_offset);

  const int dof = static_cast<int>(camera_point_jacobian.cols());
  Eigen::MatrixXd controlled_camera_point_jacobian =
    Eigen::MatrixXd::Zero(3, dof);
  Eigen::MatrixXd controlled_camera_angular_jacobian =
    Eigen::MatrixXd::Zero(3, dof);
  for (const int index : camera_indices) {
    controlled_camera_point_jacobian.col(index) =
      camera_point_jacobian.col(index);
    controlled_camera_angular_jacobian.col(index) =
      camera_frame_jacobian.bottomRows(3).col(index);
  }
  Eigen::VectorXd target_joint_velocity = Eigen::VectorXd::Zero(dof);
  if (context.follower_velocity.size() == dof) {
    for (const int index : target_indices) {
      target_joint_velocity[index] = context.follower_velocity[index];
    }
  }
  const Eigen::Vector3d target_point_velocity =
    target_point_jacobian * target_joint_velocity;

  const Eigen::Vector3d camera_point =
    camera_pose.translation() + camera_pose.linear() * camera.camera_origin_offset;
  const Eigen::Vector3d target_point =
    target_pose.translation() + target_pose.linear() * target.gripper_target_offset;
  const Eigen::Vector3d camera_to_target = target_point - camera_point;
  const double distance = camera_to_target.norm();
  if (distance <= 1e-6) {
    return;
  }
  const Eigen::Vector3d target_direction = camera_to_target / distance;
  const Eigen::Vector3d camera_forward =
    camera_pose.linear() * camera.camera_forward_axis;

  const uint64_t command_sequence = cameraCommandSequence(role_camera_group_, context);
  if (
    camera_trajectory_.waiting_for_command &&
    command_sequence == camera_trajectory_.last_sequence)
  {
    // timedCommandFeedbackSyncGroups() has already synchronized this arm to feedback.
    // Keep it still until the same adaptive timed command used by MoveJ arrives.
    return;
  }
  if (command_sequence != camera_trajectory_.last_sequence) {
    const bool first_command = camera_trajectory_.waiting_for_command;
    camera_trajectory_.last_sequence = command_sequence;
    camera_trajectory_.waiting_for_command = false;
    if (!camera_trajectory_.slow_start_complete) {
      CameraTaskError task_error;
      task_error.distance = std::abs(distance - target_distance_);
      task_error.gaze_angle = std::acos(std::clamp(
        camera_forward.dot(target_direction), -1.0, 1.0));
      task_error.valid = true;
      const double joint_tracking_error = meanAbsoluteError(
        camera_indices, context.measured_follower_position,
        context.follower_position);
      const double command_duration = cameraSlowStartDuration(
        task_error, joint_tracking_error, context.dt);

      // The camera has no unique joint-space goal. Its own task-space errors therefore
      // determine an adaptive linear segment independently of the teleop arm duration.
      camera_trajectory_.start_distance = distance;
      camera_trajectory_.start_direction = camera_forward;
      camera_trajectory_.goal_direction = target_direction;
      camera_trajectory_.start_time = context.now_seconds;
      camera_trajectory_.duration = command_duration;
      camera_trajectory_.active = true;
      if (first_command) {
        RCLCPP_INFO(
          logger_,
          "Camera arm slow start began: joint error %.4f rad, "
          "distance error %.4f m, gaze error %.4f rad, duration %.3f s",
          joint_tracking_error, task_error.distance,
          task_error.gaze_angle, command_duration);
      }
    } else {
      camera_trajectory_.active = false;
      camera_trajectory_.slow_start_complete = true;
    }
  }

  bool interpolating = camera_trajectory_.active;
  double interpolation_ratio = 1.0;
  if (interpolating) {
    const double elapsed = std::max(
      0.0, context.now_seconds - camera_trajectory_.start_time);
    interpolation_ratio = std::clamp(
      elapsed / camera_trajectory_.duration, 0.0, 1.0);
    if (interpolation_ratio >= 1.0) {
      camera_trajectory_.active = false;
    }
  }

  const double reference_distance = interpolating ?
    camera_trajectory_.start_distance + interpolation_ratio *
    (target_distance_ - camera_trajectory_.start_distance) :
    target_distance_;
  Eigen::Vector3d reference_direction = target_direction;
  Eigen::Vector3d reference_direction_velocity = Eigen::Vector3d::Zero();
  double reference_distance_velocity = 0.0;
  if (interpolating) {
    const Eigen::Vector3d direction_delta =
      camera_trajectory_.goal_direction - camera_trajectory_.start_direction;
    reference_direction =
      camera_trajectory_.start_direction + interpolation_ratio * direction_delta;
    const double reference_norm = reference_direction.norm();
    reference_direction = reference_norm > 1e-8 ?
      reference_direction / reference_norm : target_direction;
    reference_direction_velocity =
      (Eigen::Matrix3d::Identity() -
      reference_direction * reference_direction.transpose()) *
      direction_delta / camera_trajectory_.duration;
    reference_distance_velocity =
      (target_distance_ - camera_trajectory_.start_distance) /
      camera_trajectory_.duration;
  }

  LinearTaskObjective distance_task;
  distance_task.jacobian.resize(1, dof);
  distance_task.jacobian.row(0) =
    -target_direction.transpose() * controlled_camera_point_jacobian;
  const double distance_correction =
    reference_distance_velocity +
    distance_kp_ * (reference_distance - distance);
  distance_task.desired_velocity = Eigen::VectorXd::Constant(
    1, distance_correction -
    (interpolating ? 0.0 : target_direction.dot(target_point_velocity)));
  distance_task.weight = Eigen::VectorXd::Constant(1, distance_weight_);
  output.linear_task_objectives.push_back(std::move(distance_task));

  const Eigen::Matrix3d direction_projection =
    Eigen::Matrix3d::Identity() -
    target_direction * target_direction.transpose();
  Eigen::Vector3d target_direction_velocity = Eigen::Vector3d::Zero();
  if (!interpolating) {
    target_direction_velocity =
      direction_projection / distance * target_point_velocity;
  }
  LinearTaskObjective gaze_task;
  gaze_task.jacobian =
    -cyclo_motion_controller::common::skewSymmetric(camera_forward) *
    controlled_camera_angular_jacobian +
    direction_projection / distance * controlled_camera_point_jacobian;
  const Eigen::Vector3d gaze_correction =
    reference_direction_velocity +
    gaze_kp_ * (reference_direction - camera_forward);
  gaze_task.desired_velocity = gaze_correction + target_direction_velocity;
  gaze_task.weight = Eigen::VectorXd::Constant(3, gaze_weight_);
  output.linear_task_objectives.push_back(std::move(gaze_task));
}

CameraAssistMoveJMode::CameraTaskError CameraAssistMoveJMode::cameraTaskError(
  const ArmFrames & camera, const ArmFrames & target) const
{
  CameraTaskError error;
  const Eigen::Affine3d camera_pose =
    configuration_.follower_kinematics->getPose(camera.camera_link);
  const Eigen::Affine3d target_pose =
    configuration_.follower_kinematics->getPose(target.gripper_link);
  const Eigen::Vector3d camera_point =
    camera_pose.translation() + camera_pose.linear() * camera.camera_origin_offset;
  const Eigen::Vector3d target_point =
    target_pose.translation() + target_pose.linear() * target.gripper_target_offset;
  const Eigen::Vector3d camera_to_target = target_point - camera_point;
  const double distance = camera_to_target.norm();
  if (distance <= 1e-6) {
    return error;
  }
  const Eigen::Vector3d target_direction = camera_to_target / distance;
  const Eigen::Vector3d camera_forward =
    camera_pose.linear() * camera.camera_forward_axis;
  error.distance = std::abs(distance - target_distance_);
  error.gaze_angle = std::acos(std::clamp(
    camera_forward.dot(target_direction), -1.0, 1.0));
  error.valid = true;
  return error;
}

double CameraAssistMoveJMode::cameraSlowStartDuration(
  const CameraTaskError & task_error,
  const double joint_tracking_error,
  const double dt) const
{
  if (!task_error.valid) {
    return std::max(dt, 1e-6);
  }

  // Keep trajectory timing independent of the high QP correction gains. The internal
  // window matches the established adaptive MoveJ window, while the existing ready
  // thresholds normalize the heterogeneous joint, distance, and gaze errors. Replanning
  // every leader sample then advances each reference by at most one threshold unit per
  // window without exposing another tuning parameter.
  constexpr double kAdaptiveWindow = 0.4;
  const double normalized_joint_error = joint_tracking_error /
    std::max(camera_arm_sync_threshold_, 1e-6);
  const double normalized_distance_error = task_error.distance /
    std::max(camera_distance_sync_threshold_, 1e-6);
  const double normalized_gaze_error = task_error.gaze_angle /
    std::max(camera_gaze_sync_threshold_, 1e-6);
  const double normalized_error = std::max(
    {normalized_joint_error, normalized_distance_error, normalized_gaze_error});
  return std::max(std::max(dt, 1e-6), kAdaptiveWindow * normalized_error);
}

double CameraAssistMoveJMode::meanAbsoluteError(
  const std::vector<int> & indices,
  const Eigen::VectorXd & actual,
  const Eigen::VectorXd & reference)
{
  if (indices.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  double error = 0.0;
  for (const int index : indices) {
    error += std::abs(actual[index] - reference[index]);
  }
  return error / static_cast<double>(indices.size());
}

void CameraAssistMoveJMode::updateSlowStartState(
  const ModeContext & context, const ControlGroupMask camera_group)
{
  auto update_leader_arm = [this, &context](
    const ControlGroupMask group, const std::vector<int> & indices,
    LeaderTrajectory & trajectory)
    {
      if (
        (context.enabled_groups & group) == 0 || trajectory.slow_start_complete ||
        trajectory.waiting_for_command)
      {
        return;
      }
      const double error = meanAbsoluteError(
        indices, context.measured_follower_position, context.leader_reference);
      if (error <= leader_arm_sync_threshold_) {
        trajectory.slow_start_complete = true;
        trajectory.active = false;
        RCLCPP_INFO(
          logger_, "Teleop arm %u slow start synchronized at mean error %.4f rad",
          static_cast<unsigned int>(group), error);
      }
    };
  update_leader_arm(kLeftGroup, left_indices_, left_trajectory_);
  update_leader_arm(kRightGroup, right_indices_, right_trajectory_);

  if (
    camera_group != 0 && !camera_trajectory_.slow_start_complete &&
    !camera_trajectory_.waiting_for_command)
  {
    const auto & camera_indices = camera_group == kLeftGroup ? left_indices_ : right_indices_;
    const CameraTaskError camera_error = camera_group == kLeftGroup ?
      cameraTaskError(left_frames_, right_frames_) :
      cameraTaskError(right_frames_, left_frames_);
    const double joint_error = meanAbsoluteError(
      camera_indices, context.measured_follower_position, context.follower_position);
    if (
      camera_error.valid && joint_error <= camera_arm_sync_threshold_ &&
      camera_error.distance <= camera_distance_sync_threshold_ &&
      camera_error.gaze_angle <= camera_gaze_sync_threshold_)
    {
      camera_trajectory_.slow_start_complete = true;
      camera_trajectory_.active = false;
      RCLCPP_INFO(
        logger_,
        "Camera arm %u slow start synchronized "
        "(joint error %.4f rad, distance error %.4f m, gaze error %.4f rad)",
        static_cast<unsigned int>(camera_group), joint_error,
        camera_error.distance, camera_error.gaze_angle);
    }
  }

  slow_start_active_ = slow_start_enabled_ &&
    (((context.enabled_groups & kLeftGroup) != 0 && !left_trajectory_.slow_start_complete) ||
    ((context.enabled_groups & kRightGroup) != 0 && !right_trajectory_.slow_start_complete) ||
    (camera_group != 0 && !camera_trajectory_.slow_start_complete));
}

bool CameraAssistMoveJMode::update(
  const ModeContext & context, ModeOutput & output)
{
  const ControlGroupMask current_camera_group = cameraGroup(context);
  if (
    !role_initialized_ || context.enabled_groups != role_enabled_groups_ ||
    context.pose_sequence_groups != role_pose_sequence_groups_ ||
    current_camera_group != role_camera_group_)
  {
    beginRoleTransition(context);
  }

  updateLeaderArm(
    kLeftGroup, left_indices_, context.group_states.at(kLeftGroupId).leader_duration,
    context.group_states.at(kLeftGroupId).leader_sequence,
    left_trajectory_, context, output);
  updateLeaderArm(
    kRightGroup, right_indices_, context.group_states.at(kRightGroupId).leader_duration,
    context.group_states.at(kRightGroupId).leader_sequence,
    right_trajectory_, context, output);

  if (current_camera_group == kLeftGroup) {
    addCameraAssistTask(
      left_frames_, right_frames_, left_indices_, right_indices_, context, output);
  } else if (current_camera_group == kRightGroup) {
    addCameraAssistTask(
      right_frames_, left_frames_, right_indices_, left_indices_, context, output);
  }

  updateSlowStartState(context, current_camera_group);
  return true;
}
}  // namespace cyclo_teleoperation::robots::ai_worker

PLUGINLIB_EXPORT_CLASS(
  cyclo_teleoperation::robots::ai_worker::CameraAssistMoveJMode,
  cyclo_teleoperation::TeleoperationMode)
