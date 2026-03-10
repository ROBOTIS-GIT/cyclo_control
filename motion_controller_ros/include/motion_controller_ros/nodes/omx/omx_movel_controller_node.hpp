#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "motion_controller_core/common/type_define.h"
#include "motion_controller_core/controllers/omx/omx_movel_controller.hpp"
#include "motion_controller_core/kinematics/kinematics_solver.hpp"

namespace motion_controller_ros
{
    class OmxMoveLControllerNode : public rclcpp::Node
    {
    public:
        OmxMoveLControllerNode();
        ~OmxMoveLControllerNode();

    private:
        void initializeJointConfig();
        void extractJointStates(const sensor_msgs::msg::JointState::SharedPtr& msg);
        void publishCurrentPose(const Eigen::Affine3d& pose) const;
        void publishTrajectory(const Eigen::VectorXd& q_command) const;
        void publishControllerError(const std::string& error) const;

        void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
        void moveLCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
        void controlLoopCallback();

        Eigen::Affine3d poseMsgToEigen(const geometry_msgs::msg::PoseStamped& pose_msg) const;
        motion_controller::common::Vector6d computeDesiredVelocity(
            const Eigen::Affine3d& current_pose,
            const Eigen::Affine3d& goal_pose,
            const Eigen::Vector3d& feedforward_linear = Eigen::Vector3d::Zero(),
            const Eigen::Vector3d& feedforward_angular = Eigen::Vector3d::Zero()) const;

        double control_frequency_;
        double time_step_;
        double trajectory_time_;
        double default_movel_duration_;
        double kp_position_;
        double kp_orientation_;
        double weight_task_position_;
        double weight_task_orientation_;
        double weight_damping_;
        double slack_penalty_;
        double cbf_alpha_;
        double collision_buffer_;
        double collision_safe_distance_;

        std::string urdf_path_;
        std::string srdf_path_;
        std::string base_frame_;
        std::string controlled_link_;
        std::string joint_states_topic_;
        std::string joint_command_topic_;
        std::string movel_topic_;
        std::string ee_pose_topic_;
        std::string controller_error_topic_;

        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr movel_sub_;

        rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_command_pub_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_pub_;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr controller_error_pub_;

        rclcpp::TimerBase::SharedPtr control_timer_;

        std::shared_ptr<motion_controller::kinematics::KinematicsSolver> kinematics_solver_;
        std::shared_ptr<motion_controller::controllers::OMXMoveLController> qp_controller_;

        Eigen::VectorXd q_;
        Eigen::VectorXd qdot_;
        Eigen::VectorXd q_commanded_;

        std::vector<std::string> model_joint_names_;
        std::unordered_map<std::string, int> joint_index_map_;
        std::unordered_map<std::string, int> model_joint_index_map_;

        bool joint_state_received_;
        bool commanded_state_initialized_;
        bool movel_target_initialized_;
        bool movel_trajectory_active_;

        rclcpp::Time motion_start_time_;
        double active_motion_duration_;
        Eigen::Affine3d movel_start_pose_;
        Eigen::Affine3d movel_goal_pose_;
    };
}  // namespace motion_controller_ros
