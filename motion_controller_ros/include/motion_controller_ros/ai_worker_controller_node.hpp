#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <memory>
#include <map>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "motion_controller_core/kinematics_solver.hpp"
#include "motion_controller_core/ai_worker_controller.hpp"

using namespace Eigen;

namespace motion_controller_ros
{
    /**
     * @brief ROS 2 wrapper node for AI Worker teleoperation controller.
     * 
     * This node subscribes to target end-effector pose and current joint states to solve
     * inverse kinematics problems for AI Worker using Quadratic Programming.
     */
    class AIWorkerController : public rclcpp::Node
    {
    public:
        AIWorkerController();
        ~AIWorkerController();

    private:
        // Configurable parameters
        double control_frequency_;
        double time_step_;
        double trajectory_time_;
        double kp_position_;
        double kp_orientation_;
        double weight_position_;
        double weight_orientation_;
        double weight_elbow_position_;
        double weight_damping_;
        double slack_penalty_;
        double cbf_alpha_;
        double collision_buffer_;
        double collision_safe_distance_;
        std::string reactivate_topic_;
        std::string r_goal_pose_topic_;
        std::string l_goal_pose_topic_;
        std::string r_elbow_pose_topic_;
        std::string l_elbow_pose_topic_;
        std::string joint_states_topic_;
        std::string right_traj_topic_;
        std::string left_traj_topic_;
        std::string r_gripper_pose_topic_;
        std::string l_gripper_pose_topic_;
        std::string r_gripper_name_;
        std::string l_gripper_name_;
        std::string r_elbow_name_;
        std::string l_elbow_name_;
        std::string base_frame_id_;
        std::string traj_frame_id_;
        std::string right_gripper_joint_name_;
        std::string left_gripper_joint_name_;
        std::string urdf_path_;
        std::string srdf_path_;

        // Subscribers
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pose_sub_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr l_goal_pose_sub_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr r_elbow_pose_sub_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr l_elbow_pose_sub_;
        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr ref_divergence_sub_;
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr ref_reactivate_sub_;

        // Publishers
        // rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr lift_pub_;
        rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_r_pub_;
        rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_l_pub_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_gripper_pose_pub_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_gripper_pose_pub_;

        // Timer for control loop
        rclcpp::TimerBase::SharedPtr control_timer_;

        // Motion controller components
        std::shared_ptr<motion_controller_core::KinematicsSolver> kinematics_solver_;
        std::shared_ptr<motion_controller_core::QPIK> qp_controller_;

        // State variables
        VectorXd q_;
        VectorXd qdot_;
        VectorXd q_desired_;
        Affine3d right_gripper_pose_;
        Affine3d left_gripper_pose_;
        Affine3d r_goal_pose_;
        Affine3d l_goal_pose_;
        Affine3d r_elbow_pose_;
        Affine3d l_elbow_pose_;
        bool r_goal_pose_received_;
        bool l_goal_pose_received_;
        bool r_elbow_pose_received_;
        bool l_elbow_pose_received_;
        bool reference_diverged_;
        rclcpp::Time activate_start_;
        bool activate_pending_;
        bool joint_state_received_;

        // Joint configuration
        std::vector<std::string> left_arm_joints_;
        std::vector<std::string> right_arm_joints_;
        // std::string lift_joint_;
        std::map<std::string, int> joint_index_map_;
        std::vector<std::string> model_joint_names_;
        std::unordered_map<std::string, int> model_joint_index_map_;

        // Control loop parameters
        double dt_;  // Time step in seconds

        // Callbacks
        void rightGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
        void leftGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
        void rightElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
        void leftElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
        void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
        void referenceDivergenceCallback(const std_msgs::msg::Bool::SharedPtr msg);
        void referenceReactivateCallback(const std_msgs::msg::Bool::SharedPtr msg);
        void controlLoopCallback();

        // ================================ Helper functions ================================
        /**
         * @brief Initialize the joint configuration.
         */
        void initializeJointConfig();

        /**
         * @brief Publish the desired joint trajectory.
         * @param q_desired (Eigen::VectorXd) Desired joint positions.
         */
        void publishTrajectory(const VectorXd& q_desired);

        /**
         * @brief Create a joint trajectory message.
         * @param arm_joint_names       (std::vector<std::string>) Arm joint names.
         * @param positions             (Eigen::VectorXd) Joint positions.
         * @param arm_indices           (std::vector<int>) Arm joint indices.
         * @param gripper_joint_name    (std::string) Gripper joint name.
         * @return (trajectory_msgs::msg::JointTrajectory) Joint trajectory message.
         */
        trajectory_msgs::msg::JointTrajectory createTrajectoryMsgWithGripper(
            const std::vector<std::string>& arm_joint_names,
            const VectorXd& positions,
            const std::vector<int>& arm_indices,
            const std::string& gripper_joint_name) const;

        /**
         * @brief Publish the gripper pose.
         * @param r_gripper_pose (Eigen::Affine3d) Right gripper pose.
         * @param l_gripper_pose (Eigen::Affine3d) Left gripper pose.
         */
        void publishGripperPose(const Affine3d& r_gripper_pose, const Affine3d& l_gripper_pose);

        /**
         * @brief Extract joint states from message.
         * @param msg       (sensor_msgs::msg::JointState::SharedPtr) Joint state message.
         */
         void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr& msg);
        
        // ================================ Control computation functions ================================
        /**
         * @brief Compute pose matrix out of pose message.
         * @param pose       (geometry_msgs::msg::PoseStamped) Pose message.
         * @return (Affine3d) Computed pose matrix.
         */
        Affine3d computePoseMat(const geometry_msgs::msg::PoseStamped& pose) const;

        /**
         * @brief Compute desired velocity out of current and goal poses.
         * @param current_pose       (Affine3d) Current pose.
         * @param goal_pose          (Affine3d) Goal pose.
         * @return (Vector6d) Computed desired velocity.
         */
        Vector6d computeDesiredVelocity(const Affine3d& current_pose, const Affine3d& goal_pose) const;
    };
}  // namespace motion_controller_ros
