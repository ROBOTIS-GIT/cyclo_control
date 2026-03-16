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
//
// Author: Yeonguk Kim

#include "motion_controller_ros/nodes/ai_worker/leader_controller_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

namespace motion_controller_ros
{
LeaderController::LeaderController()
: Node("leader_controller"),
  right_traj_received_(false),
  left_traj_received_(false),
  lift_joint_received_(false),
  last_right_traj_time_(this->now()),
  last_left_traj_time_(this->now()),
  lift_joint_index_(-1)
{
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "Leader Controller - Starting up...");
  RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
  RCLCPP_INFO(this->get_logger(), "========================================");

  control_frequency_ = this->declare_parameter("control_frequency", 100.0);
  urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
  srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
  joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
  right_traj_topic_ = this->declare_parameter(
            "right_traj_topic",
            std::string("/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory"));
  left_traj_topic_ = this->declare_parameter(
            "left_traj_topic",
            std::string("/leader/joint_trajectory_command_broadcaster_left/raw_joint_trajectory"));
  reactivate_service_ = this->declare_parameter("reactivate_service", std::string("/reactivate"));
  command_timeout_ = this->declare_parameter("command_timeout", 0.1);
  r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
  l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));
  r_elbow_pose_topic_ = this->declare_parameter("r_elbow_pose_topic", std::string("/r_elbow_pose"));
  l_elbow_pose_topic_ = this->declare_parameter("l_elbow_pose_topic", std::string("/l_elbow_pose"));
  base_frame_id_ = this->declare_parameter("base_frame_id", std::string("base_link"));
  r_gripper_name_ = this->declare_parameter("r_gripper_name", std::string("arm_r_link7"));
  l_gripper_name_ = this->declare_parameter("l_gripper_name", std::string("arm_l_link7"));
  r_elbow_name_ = this->declare_parameter("r_elbow_name", std::string("arm_r_link4"));
  l_elbow_name_ = this->declare_parameter("l_elbow_name", std::string("arm_l_link4"));
  lift_joint_name_ = this->declare_parameter("lift_joint_name", std::string("lift_joint"));
  model_lift_joint_name_ = this->declare_parameter("model_lift_joint_name", std::string("joint"));

  r_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            right_traj_topic_, 10,
            std::bind(&LeaderController::rightTrajectoryCallback, this, std::placeholders::_1));
  l_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            left_traj_topic_, 10,
            std::bind(&LeaderController::leftTrajectoryCallback, this, std::placeholders::_1));
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic_, 10,
            std::bind(&LeaderController::jointStateCallback, this, std::placeholders::_1));

  r_goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            r_goal_pose_topic_, 10);
  l_goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            l_goal_pose_topic_, 10);
  r_elbow_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            r_elbow_pose_topic_, 10);
  l_elbow_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            l_elbow_pose_topic_, 10);

  reactivate_client_ = this->create_client<std_srvs::srv::Trigger>(reactivate_service_);
  if (!reactivate_client_) {
    RCLCPP_FATAL(this->get_logger(),
                "Failed to create reactivate service client '%s'.",
                reactivate_service_.c_str());
    rclcpp::shutdown();
    return;
  }
  if (!reactivate_client_->wait_for_service(std::chrono::seconds(2))) {
    RCLCPP_WARN(this->get_logger(),
                "Reactivate service '%s' not available yet; will retry on first publish.",
                reactivate_service_.c_str());
  } else {
    RCLCPP_INFO(this->get_logger(), "Reactivate service available: %s",
        reactivate_service_.c_str());
  }

  try {
    if (urdf_path_.empty()) {
      throw std::runtime_error("URDF path not provided.");
    }
    RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path_.c_str());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to resolve robot model paths: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  try {
    if (srdf_path_.empty()) {
      RCLCPP_INFO(this->get_logger(), "SRDF path not provided. Continuing without SRDF.");
    } else {
      RCLCPP_INFO(this->get_logger(), "SRDF path: %s", srdf_path_.c_str());
    }
    RCLCPP_INFO(this->get_logger(), "Loading URDF and initializing kinematics solver...");
    kinematics_solver_ =
      std::make_shared<motion_controller::kinematics::KinematicsSolver>(urdf_path_, srdf_path_);

            // Initialize state variables
    const int dof = kinematics_solver_->getDof();
    q_.setZero(dof);
    qdot_.setZero(dof);
    RCLCPP_INFO(this->get_logger(), "Kinematics solver initialized (DOF: %d)", dof);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize kinematics solver: %s", e.what());
    rclcpp::shutdown();
    return;
  }

        // Initialize joint configuration from URDF
  initializeJointConfig();

  const int timer_period_ms = static_cast<int>(1000.0 / control_frequency_);
  control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_period_ms),
            std::bind(&LeaderController::controlLoopCallback, this));

  if (!control_timer_) {
    RCLCPP_FATAL(this->get_logger(), "Failed to create control loop timer!");
    rclcpp::shutdown();
    return;
  }

  RCLCPP_INFO(this->get_logger(),
            "Leader Controller initialized successfully!");
  RCLCPP_INFO(this->get_logger(),
            "  - Control loop: %.1f Hz (period: %d ms)", control_frequency_, timer_period_ms);
  RCLCPP_INFO(this->get_logger(),
            "  - Subscriptions: joint_states=%s",
            joint_state_sub_ ? "OK" : "FAILED");
  RCLCPP_INFO(this->get_logger(), "========================================");
  RCLCPP_INFO(this->get_logger(), "Node is ready! Waiting for messages...");
}

LeaderController::~LeaderController()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down Leader Controller");
}

void LeaderController::initializeJointConfig()
{
  const auto joint_names = kinematics_solver_->getJointNames();
  model_joint_index_map_.clear();
  for (size_t i = 0; i < joint_names.size(); ++i) {
    model_joint_index_map_[joint_names[i]] = static_cast<int>(i);
  }

  auto it = model_joint_index_map_.find(model_lift_joint_name_);
  if (it != model_joint_index_map_.end()) {
    lift_joint_index_ = it->second;
  } else {
    RCLCPP_ERROR(this->get_logger(),
                "Model lift joint '%s' not found in URDF.", model_lift_joint_name_.c_str());
  }
}

void LeaderController::rightTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  updateJointPositionsFromTrajectory(*msg);
  right_traj_received_ = true;
  last_right_traj_time_ = this->now();
}

void LeaderController::leftTrajectoryCallback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  updateJointPositionsFromTrajectory(*msg);
  left_traj_received_ = true;
  last_left_traj_time_ = this->now();
}

void LeaderController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  updateLiftJointFromJointState(*msg);
}

void LeaderController::updateJointPositionsFromTrajectory(
  const trajectory_msgs::msg::JointTrajectory & msg)
{
  if (msg.points.empty()) {
    return;
  }
  const auto & point = msg.points.front();
  if (point.positions.empty()) {
    return;
  }

  for (size_t i = 0; i < msg.joint_names.size(); ++i) {
    auto it = model_joint_index_map_.find(msg.joint_names[i]);
    if (it == model_joint_index_map_.end()) {
      continue;
    }
    const int model_index = it->second;
    if (model_index < 0 || model_index >= q_.size()) {
      continue;
    }
    if (i < point.positions.size()) {
      q_[model_index] = point.positions[i];
    }
    if (i < point.velocities.size()) {
      qdot_[model_index] = point.velocities[i];
    }
  }
}

void LeaderController::updateLiftJointFromJointState(const sensor_msgs::msg::JointState & msg)
{
  if (lift_joint_index_ < 0 || lift_joint_index_ >= q_.size()) {
    return;
  }

  for (size_t i = 0; i < msg.name.size(); ++i) {
    if (msg.name[i] != lift_joint_name_) {
      continue;
    }
    if (i < msg.position.size()) {
      q_[lift_joint_index_] = msg.position[i];
      lift_joint_received_ = true;
    }
    if (i < msg.velocity.size()) {
      qdot_[lift_joint_index_] = msg.velocity[i];
    }
    return;
  }
}

bool LeaderController::requestReactivateOnce()
{
  if (!reactivate_client_) {
    return false;
  }
  if (!reactivate_client_->service_is_ready()) {
    return false;
  }

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  (void)reactivate_client_->async_send_request(
            request,
    [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      try {
        const auto resp = future.get();
        if (resp->success) {
          RCLCPP_INFO(this->get_logger(), "Reactivate service call succeeded: %s",
            resp->message.c_str());
        } else {
          RCLCPP_WARN(this->get_logger(), "Reactivate service call failed: %s",
            resp->message.c_str());
        }
      } catch (const std::exception & e) {
        RCLCPP_WARN(this->get_logger(), "Reactivate service call exception: %s", e.what());
      }
            });
  return true;
}

void LeaderController::controlLoopCallback()
{
  static int loop_count = 0;
  static int debug_count = 0;

  loop_count++;

  const rclcpp::Time now = this->now();
  const bool right_traj_has_publisher =
    (r_traj_sub_ && r_traj_sub_->get_publisher_count() > 0);
  const bool left_traj_has_publisher =
    (l_traj_sub_ && l_traj_sub_->get_publisher_count() > 0);

  if (!right_traj_has_publisher && !left_traj_has_publisher) {
    was_publishing_reference_ = false;
    reactivate_requested_ = false;
    if (debug_count++ % 100 == 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "No publishers on trajectory topics; skipping goal pose publish");
    }
    return;
  }

  const bool right_recent =
    right_traj_has_publisher && right_traj_received_ &&
    (now - last_right_traj_time_).seconds() < command_timeout_;
  const bool left_recent =
    left_traj_has_publisher && left_traj_received_ &&
    (now - last_left_traj_time_).seconds() < command_timeout_;
  const bool has_recent_reference = right_recent || left_recent;

        // Wait until we have at least one arm trajectory message before publishing any goal pose.
  if (!has_recent_reference) {
    was_publishing_reference_ = false;
    reactivate_requested_ = false;
    if (debug_count++ % 100 == 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Control loop waiting for joint trajectory commands...");
    }
    return;
  }

  // When a command stream starts (or resumes after timeout), trigger
  // slow-start via reactivate service. Keep publishing references even if the
  // service isn't ready yet; we'll retry calling it until it is.
  if (!reactivate_requested_) {
    if (requestReactivateOnce()) {
      reactivate_requested_ = true;
    } else {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Reactivate service '%s' not ready; will retry...",
                    reactivate_service_.c_str());
    }
  }
  was_publishing_reference_ = true;

  try {
    kinematics_solver_->updateState(q_, qdot_);

    if (right_recent) {
      const Affine3d r_pose =
        computePoseInBaseFrame(kinematics_solver_->getPose(r_gripper_name_));
      const Affine3d r_elbow_pose =
        computePoseInBaseFrame(kinematics_solver_->getPose(r_elbow_name_));
      r_goal_pose_pub_->publish(makePoseStamped(r_pose));
      r_elbow_pose_pub_->publish(makePoseStamped(r_elbow_pose));
    }

    if (left_recent) {
      const Affine3d l_pose =
        computePoseInBaseFrame(kinematics_solver_->getPose(l_gripper_name_));
      const Affine3d l_elbow_pose =
        computePoseInBaseFrame(kinematics_solver_->getPose(l_elbow_name_));
      l_goal_pose_pub_->publish(makePoseStamped(l_pose));
      l_elbow_pose_pub_->publish(makePoseStamped(l_elbow_pose));
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "FK computation failed: %s", e.what());
  }
}

geometry_msgs::msg::PoseStamped LeaderController::makePoseStamped(const Affine3d & pose) const
{
  geometry_msgs::msg::PoseStamped msg;
        // msg.header.stamp = this->now();
  msg.header.frame_id = base_frame_id_;
  msg.pose.position.x = pose.translation().x();
  msg.pose.position.y = pose.translation().y();
  msg.pose.position.z = pose.translation().z();

  const Eigen::Quaterniond quat(pose.linear());
  msg.pose.orientation.w = quat.w();
  msg.pose.orientation.x = quat.x();
  msg.pose.orientation.y = quat.y();
  msg.pose.orientation.z = quat.z();
  return msg;
}

Affine3d LeaderController::computePoseInBaseFrame(const Affine3d & link_pose) const
{
  if (kinematics_solver_ && kinematics_solver_->hasLinkFrame("world")) {
    const Affine3d base_pose = kinematics_solver_->getPose("world");
    return base_pose.inverse() * link_pose;
  }
  return link_pose;
}
}  // namespace motion_controller_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<motion_controller_ros::LeaderController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
