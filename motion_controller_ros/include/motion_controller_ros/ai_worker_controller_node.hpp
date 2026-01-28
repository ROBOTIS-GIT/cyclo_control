#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <memory>
#include <map>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "motion_controller_core/kinematics_solver.hpp"
#include "motion_controller_core/robot_controller.hpp"

using namespace Eigen;

namespace motion_controller_ros
{
    /**
     * @brief ROS 2 wrapper node for QP-based inverse kinematics controller.
     * 
     * This node:
     * - Runs at configurable control frequency (default 100Hz)
     * - Subscribes to /goal_pose (target end-effector pose)
     * - Subscribes to /joint_states (current robot state)
     * - Computes optimal joint velocities using QP
     * - Publishes trajectory commands to multiple joints
     */
    class AIWorkerController : public rclcpp::Node
    {
    public:
        AIWorkerController();
        ~AIWorkerController();

    private:
        // Constants
        static constexpr double DEFAULT_CONTROL_FREQUENCY = 100.0;  // Hz
        static constexpr double DEFAULT_TIME_STEP = 0.01;  // seconds (1/100Hz)
        static constexpr double DEFAULT_TRAJECTORY_TIME = 0.0;  // seconds (time_from_start for trajectory points)
        static constexpr double DEFAULT_KP = 1000.0;
        static constexpr double DEFAULT_KV = 10.0;
        static constexpr int DEBUG_LOG_INTERVAL = 100;
        static constexpr const char* R_GOAL_POSE_TOPIC = "/r_goal_pose";
        static constexpr const char* JOINT_STATES_TOPIC = "/joint_states";
        static constexpr const char* RIGHT_TRAJ_TOPIC =
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory";
        static constexpr const char* LEFT_TRAJ_TOPIC =
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory";
        // static constexpr const char* LIFT_TRAJ_TOPIC =
        //     "/leader/joystick_controller_right/joint_trajectory";
        static constexpr const char* R_GRIPPER_POSE_TOPIC = "/r_gripper_pose";
        static constexpr const char* R_GRIPPER_NAME = "arm_r_link7";
        static constexpr const char* BASE_FRAME_ID = "base_link";
        static constexpr const char* TRAJ_FRAME_ID = "";
        static constexpr const char* RIGHT_GRIPPER_JOINT = "gripper_r_joint1";
        static constexpr const char* LEFT_GRIPPER_JOINT = "gripper_l_joint1";

        // Subscribers
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pose_sub_;
        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

        // Publishers
        // rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr lift_pub_;
        rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
        rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_l_pub_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_gripper_pose_pub_;

        // Timer for control loop
        rclcpp::TimerBase::SharedPtr control_timer_;

        // Motion controller components
        std::shared_ptr<motion_controller_core::KinematicsSolver> kinematics_solver_;
        std::shared_ptr<motion_controller_core::QPIK> qp_controller_;

        // State variables
        VectorXd q_;
        VectorXd qdot_;
        VectorXd q_desired_;
        geometry_msgs::msg::PoseStamped goal_pose_;
        bool goal_pose_received_;
        bool joint_state_received_;

        // Joint configuration
        std::vector<std::string> left_arm_joints_;
        std::vector<std::string> right_arm_joints_;
        // std::string lift_joint_;
        std::string left_gripper_joint_;   // gripper_l_joint1
        std::string right_gripper_joint_;  // gripper_r_joint1
        std::map<std::string, int> joint_index_map_;
        std::vector<std::string> model_joint_names_;
        std::unordered_map<std::string, int> model_joint_index_map_;

        // Control loop parameters
        double dt_;  // Time step in seconds

        // Callbacks
        void goalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
        void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
        void controlLoopCallback();

        // Helper functions
        void initializeJointConfig();
        void publishTrajectory(const VectorXd& q_desired);
        trajectory_msgs::msg::JointTrajectory createTrajectoryMsg(
            const std::vector<std::string>& joint_names,
            const VectorXd& positions,
            const std::vector<int>& indices) const;
        trajectory_msgs::msg::JointTrajectory createTrajectoryMsgWithGripper(
            const std::vector<std::string>& arm_joint_names,
            const VectorXd& positions,
            const std::vector<int>& arm_indices,
            const std::string& gripper_joint_name) const;
        
        // Control computation functions
        Affine3d computeGoalPose() const;
        Vector6d computeDesiredVelocity(const Affine3d& current_pose, const Affine3d& goal_pose) const;
        void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr& msg);
    };
}  // namespace motion_controller_ros
