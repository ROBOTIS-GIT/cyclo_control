#pragma once

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "motion_controller_core/common/type_define.h"
#include "motion_controller_core/controllers/ai_worker_controller.hpp"
#include "motion_controller_core/kinematics/kinematics_solver.hpp"

namespace motion_controller_ros
{
class RightArmRelativeController : public rclcpp::Node
{
public:
  RightArmRelativeController();
  ~RightArmRelativeController();

private:
  using Affine3d = Eigen::Affine3d;
  using Matrix3d = Eigen::Matrix3d;
  using Quaterniond = Eigen::Quaterniond;
  using Vector3d = Eigen::Vector3d;
  using VectorXd = Eigen::VectorXd;

  // Parameters
  double control_frequency_{100.0};
  double time_step_{0.01};
  double trajectory_time_{0.0};
  double kp_position_{50.0};
  double kp_orientation_{50.0};
  double weight_position_{10.0};
  double weight_orientation_{1.0};
  double weight_damping_{0.1};
  double slack_penalty_{1000.0};
  double cbf_alpha_{50.0};
  double collision_buffer_{0.05};
  double collision_safe_distance_{0.02};
  double command_hz_{20.0};
  bool delta_in_ee_frame_{false};
  bool lock_other_joints_{true};

  std::string delta_pose_topic_{"/r_delta_pose"};
  std::string joint_states_topic_{"/joint_states"};
  std::string right_traj_topic_{"/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"};
  std::string right_raw_traj_topic_{"/leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory"};
  double raw_traj_timeout_{0.5};
  std::string r_gripper_pose_topic_{"/r_gripper_pose"};
  std::string base_frame_id_{"base_link"};
  std::string r_gripper_name_{"arm_r_link7"};
  std::string right_gripper_joint_name_{"gripper_r_joint1"};

  // Subscribers
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr delta_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr right_raw_traj_sub_;

  // Publishers
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_gripper_pose_pub_;

  // Timer
  rclcpp::TimerBase::SharedPtr control_timer_;

  // Core controller components
  std::shared_ptr<motion_controller::kinematics::KinematicsSolver> kinematics_solver_;
  std::shared_ptr<motion_controller::controllers::QPIK> qp_controller_;

  // Joint state
  bool joint_state_received_{false};
  bool ee_seeded_{false};
  VectorXd q_;
  VectorXd qdot_;
  VectorXd q_desired_;
  double dt_{0.01};

  // Model joint name/index bookkeeping
  std::vector<std::string> model_joint_names_;
  std::unordered_map<std::string, int> model_joint_index_map_;
  std::unordered_map<std::string, int> joint_index_map_;
  std::vector<std::string> right_arm_joints_;
  std::vector<int> right_arm_indices_;

  // Right gripper passthrough (from upstream raw trajectory)
  bool right_raw_gripper_received_{false};
  double right_raw_gripper_position_{0.0};
  rclcpp::Time last_right_raw_traj_time_;

  // Goal pose and interpolation state
  Affine3d current_pose_{Affine3d::Identity()};
  Affine3d goal_pose_{Affine3d::Identity()};
  bool interp_active_{false};
  int interp_steps_total_{1};
  int interp_step_idx_{0};
  Affine3d interp_start_{Affine3d::Identity()};
  Affine3d interp_end_{Affine3d::Identity()};

  // Callbacks
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void deltaPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void rightRawTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void controlLoopCallback();

  // Helpers
  void initializeJointConfig();
  void lockNonRightArmJointsIfRequested();
  void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr &msg);
  void seedGoalPoseFromJointStateOnce();
  void stepInterpolatedGoalPose();

  static Affine3d poseMsgToAffine(const geometry_msgs::msg::Pose &pose);
  Affine3d applyDeltaPose(const Affine3d &start, const geometry_msgs::msg::Pose &delta) const;
  static Affine3d interpolatePose(const Affine3d &a, const Affine3d &b, double alpha);

  motion_controller::common::Vector6d computeDesiredVelocity(
    const Affine3d &current_pose,
    const Affine3d &goal_pose) const;

  void publishTrajectory(const VectorXd &q_desired);
  trajectory_msgs::msg::JointTrajectory createTrajectoryMsgWithGripper(
    const std::vector<std::string> &arm_joint_names,
    const VectorXd &positions,
    const std::vector<int> &arm_indices,
    const std::string &gripper_joint_name,
    const double gripper_position) const;
  void publishGripperPose(const Affine3d &pose) const;
};
}  // namespace motion_controller_ros

