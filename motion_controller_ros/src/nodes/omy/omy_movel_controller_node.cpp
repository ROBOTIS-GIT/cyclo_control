#include "motion_controller_ros/nodes/omy/omy_movel_controller_node.hpp"

#include <cmath>

#include "motion_controller_core/common/type_define.h"

namespace motion_controller_ros
{
    namespace
    {
        double clampDuration(double duration, double fallback)
        {
            return duration > 1e-6 ? duration : std::max(1e-3, fallback);
        }
    }

    OmyMoveLControllerNode::OmyMoveLControllerNode()
        : Node("omy_movel_controller"),
          joint_state_received_(false),
          commanded_state_initialized_(false),
          movel_target_initialized_(false),
          movel_trajectory_active_(false),
          motion_start_time_(this->now()),
          active_motion_duration_(0.0),
          movel_start_pose_(Eigen::Affine3d::Identity()),
          movel_goal_pose_(Eigen::Affine3d::Identity())
    {
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "OMY MoveL Controller - Starting up...");
        RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
        RCLCPP_INFO(this->get_logger(), "========================================");

        control_frequency_ = this->declare_parameter("control_frequency", 100.0);
        time_step_ = this->declare_parameter("time_step", 0.01);
        trajectory_time_ = this->declare_parameter("trajectory_time", 0.05);
        default_movel_duration_ = this->declare_parameter("default_movel_duration", 3.0);
        kp_position_ = this->declare_parameter("kp_position", 4.0);
        kp_orientation_ = this->declare_parameter("kp_orientation", 2.5);
        weight_task_position_ = this->declare_parameter("weight_task_position", 10.0);
        weight_task_orientation_ = this->declare_parameter("weight_task_orientation", 1.0);
        weight_damping_ = this->declare_parameter("weight_damping", 0.05);
        slack_penalty_ = this->declare_parameter("slack_penalty", 1000.0);
        cbf_alpha_ = this->declare_parameter("cbf_alpha", 5.0);
        collision_buffer_ = this->declare_parameter("collision_buffer", 0.05);
        collision_safe_distance_ = this->declare_parameter("collision_safe_distance", 0.02);

        urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
        srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
        base_frame_ = this->declare_parameter("base_frame", std::string("link0"));
        controlled_link_ = this->declare_parameter("controlled_link", std::string("link7"));
        joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
        joint_command_topic_ = this->declare_parameter("joint_command_topic", std::string("/omy/joint_trajectory"));
        movel_topic_ = this->declare_parameter("movel_topic", std::string("~/movel"));
        ee_pose_topic_ = this->declare_parameter("ee_pose_topic", std::string("~/current_pose"));
        controller_error_topic_ = this->declare_parameter("controller_error_topic", std::string("~/controller_error"));

        if (urdf_path_.empty() || srdf_path_.empty()) {
            RCLCPP_FATAL(this->get_logger(), "Both 'urdf_path' and 'srdf_path' parameters must be provided.");
            rclcpp::shutdown();
            return;
        }

        joint_command_pub_ =
            this->create_publisher<trajectory_msgs::msg::JointTrajectory>(joint_command_topic_, 10);
        ee_pose_pub_ =
            this->create_publisher<geometry_msgs::msg::PoseStamped>(ee_pose_topic_, 10);
        controller_error_pub_ =
            this->create_publisher<std_msgs::msg::String>(controller_error_topic_, 10);

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic_, 10,
            std::bind(&OmyMoveLControllerNode::jointStateCallback, this, std::placeholders::_1));
        movel_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            movel_topic_, 10,
            std::bind(&OmyMoveLControllerNode::moveLCallback, this, std::placeholders::_1));

        try {
            RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "SRDF path: %s", srdf_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "Loading URDF and initializing kinematics solver...");
            kinematics_solver_ =
                std::make_shared<motion_controller::kinematics::KinematicsSolver>(urdf_path_, srdf_path_);
            RCLCPP_INFO(this->get_logger(), "Initializing QP controller...");
            qp_controller_ = std::make_shared<motion_controller::controllers::OMYMoveLController>(
                kinematics_solver_, controlled_link_, time_step_);
            qp_controller_->setControllerParams(
                slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);

            q_.setZero(kinematics_solver_->getDof());
            qdot_.setZero(kinematics_solver_->getDof());
            q_commanded_.setZero(kinematics_solver_->getDof());

            initializeJointConfig();

            RCLCPP_INFO(this->get_logger(), "Motion controller initialized (DOF: %d)", kinematics_solver_->getDof());
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to initialize OMY MoveL Controller: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        const int timer_period_ms =
            std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, control_frequency_))));
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_period_ms),
            std::bind(&OmyMoveLControllerNode::controlLoopCallback, this));

        if (!control_timer_) {
            RCLCPP_FATAL(this->get_logger(), "Failed to create control loop timer!");
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(this->get_logger(), "OMY MoveL Controller initialized successfully!");
        RCLCPP_INFO(this->get_logger(), "  - Controlled link: %s", controlled_link_.c_str());
        RCLCPP_INFO(
            this->get_logger(),
            "  - Control loop: %.1f Hz (period: %d ms)",
            control_frequency_,
            timer_period_ms);
        RCLCPP_INFO(this->get_logger(), "  - MoveL command topic: %s", movel_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "Node is ready! Waiting for messages...");
    }

    OmyMoveLControllerNode::~OmyMoveLControllerNode()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down OMY MoveL Controller");
    }

    void OmyMoveLControllerNode::initializeJointConfig()
    {
        model_joint_names_ = kinematics_solver_->getJointNames();
        model_joint_index_map_.clear();
        for (size_t i = 0; i < model_joint_names_.size(); ++i) {
            model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
        }

        std::string joint_list;
        for (const auto& joint_name : model_joint_names_) {
            joint_list += joint_name + " ";
        }
        RCLCPP_INFO(this->get_logger(), "Model joints: %s", joint_list.c_str());
    }

    void OmyMoveLControllerNode::extractJointStates(const sensor_msgs::msg::JointState::SharedPtr& msg)
    {
        const int dof = kinematics_solver_->getDof();
        q_.setZero(dof);
        qdot_.setZero(dof);

        const int max_index = std::min<int>(dof, static_cast<int>(model_joint_names_.size()));
        for (int i = 0; i < max_index; ++i) {
            const auto& joint_name = model_joint_names_[i];
            const auto it = joint_index_map_.find(joint_name);
            if (it == joint_index_map_.end()) {
                continue;
            }
            const int msg_idx = it->second;
            if (msg_idx < static_cast<int>(msg->position.size())) {
                q_[i] = msg->position[msg_idx];
            }
            if (msg_idx < static_cast<int>(msg->velocity.size())) {
                qdot_[i] = msg->velocity[msg_idx];
            }
        }
    }

    void OmyMoveLControllerNode::publishCurrentPose(const Eigen::Affine3d& pose) const
    {
        if (!ee_pose_pub_) {
            return;
        }

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = this->now();
        pose_msg.header.frame_id = base_frame_;
        pose_msg.pose.position.x = pose.translation().x();
        pose_msg.pose.position.y = pose.translation().y();
        pose_msg.pose.position.z = pose.translation().z();

        const Eigen::Quaterniond quat(pose.linear());
        pose_msg.pose.orientation.w = quat.w();
        pose_msg.pose.orientation.x = quat.x();
        pose_msg.pose.orientation.y = quat.y();
        pose_msg.pose.orientation.z = quat.z();
        ee_pose_pub_->publish(pose_msg);
    }

    void OmyMoveLControllerNode::publishTrajectory(const Eigen::VectorXd& q_command) const
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.header.frame_id = "";
        traj_msg.joint_names = model_joint_names_;

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);
        for (int idx = 0; idx < q_command.size(); ++idx) {
            point.positions.push_back(q_command[idx]);
            point.velocities.push_back(0.0);
        }

        traj_msg.points.push_back(point);
        joint_command_pub_->publish(traj_msg);
    }

    void OmyMoveLControllerNode::publishControllerError(const std::string& error) const
    {
        if (!controller_error_pub_) {
            return;
        }

        std_msgs::msg::String err;
        err.data = error;
        controller_error_pub_->publish(err);
    }

    void OmyMoveLControllerNode::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (joint_index_map_.empty()) {
            for (size_t i = 0; i < msg->name.size(); ++i) {
                joint_index_map_[msg->name[i]] = static_cast<int>(i);
            }
        }

        extractJointStates(msg);
        joint_state_received_ = true;

        if (!commanded_state_initialized_) {
            q_commanded_ = q_;
            commanded_state_initialized_ = true;

            kinematics_solver_->updateState(q_commanded_, qdot_);
            movel_start_pose_ = kinematics_solver_->getPose(controlled_link_);
            movel_goal_pose_ = movel_start_pose_;
            movel_target_initialized_ = true;

            RCLCPP_INFO(this->get_logger(), "Initial end-effector pose captured for moveL control.");
        }
    }

    void OmyMoveLControllerNode::moveLCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (!msg || !joint_state_received_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Ignoring moveL command until joint states are available.");
            return;
        }

        kinematics_solver_->updateState(q_commanded_, qdot_);
        movel_start_pose_ = kinematics_solver_->getPose(controlled_link_);
        movel_goal_pose_ = poseMsgToEigen(*msg);
        active_motion_duration_ = clampDuration(default_movel_duration_, default_movel_duration_);
        motion_start_time_ = this->now();
        movel_target_initialized_ = true;
        movel_trajectory_active_ = true;

        RCLCPP_INFO(
            this->get_logger(),
            "Received moveL command (duration: %.3f s)",
            active_motion_duration_);
    }

    Eigen::Affine3d OmyMoveLControllerNode::poseMsgToEigen(const geometry_msgs::msg::PoseStamped& pose_msg) const
    {
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.translation() << pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z;

        const Eigen::Quaterniond quat(
            pose_msg.pose.orientation.w,
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z);
        pose.linear() = quat.normalized().toRotationMatrix();
        return pose;
    }

    motion_controller::common::Vector6d OmyMoveLControllerNode::computeDesiredVelocity(
        const Eigen::Affine3d& current_pose,
        const Eigen::Affine3d& goal_pose,
        const Eigen::Vector3d& feedforward_linear,
        const Eigen::Vector3d& feedforward_angular) const
    {
        motion_controller::common::Vector6d desired_vel =
            motion_controller::common::Vector6d::Zero();

        const Eigen::Vector3d position_error = goal_pose.translation() - current_pose.translation();
        const Eigen::Matrix3d rotation_error = goal_pose.linear() * current_pose.linear().transpose();
        const Eigen::AngleAxisd angle_axis_error(rotation_error);
        const Eigen::Vector3d orientation_error =
            angle_axis_error.axis() * angle_axis_error.angle();

        desired_vel.head<3>() = feedforward_linear + kp_position_ * position_error;
        desired_vel.tail<3>() = feedforward_angular + kp_orientation_ * orientation_error;
        return desired_vel;
    }

    void OmyMoveLControllerNode::controlLoopCallback()
    {
        if (!joint_state_received_ || !commanded_state_initialized_ || !movel_target_initialized_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Control loop waiting for joint states...");
            return;
        }

        try {
            const Eigen::VectorXd q_feedback = q_commanded_;
            kinematics_solver_->updateState(q_feedback, qdot_);
            const Eigen::Affine3d current_pose = kinematics_solver_->getPose(controlled_link_);
            publishCurrentPose(current_pose);

            const double elapsed = (this->now() - motion_start_time_).seconds();
            motion_controller::common::Vector6d desired_task_vel =
                motion_controller::common::Vector6d::Zero();

            if (movel_trajectory_active_ && elapsed < active_motion_duration_) {
                const Eigen::Vector3d linear_ref =
                    motion_controller::common::math_utils::cubicDotVector<3>(
                        elapsed,
                        0.0,
                        active_motion_duration_,
                        movel_start_pose_.translation(),
                        movel_goal_pose_.translation(),
                        Eigen::Vector3d::Zero(),
                        Eigen::Vector3d::Zero());
                const Eigen::Vector3d position_ref =
                    motion_controller::common::math_utils::cubicVector<3>(
                        elapsed,
                        0.0,
                        active_motion_duration_,
                        movel_start_pose_.translation(),
                        movel_goal_pose_.translation(),
                        Eigen::Vector3d::Zero(),
                        Eigen::Vector3d::Zero());
                const Eigen::Matrix3d rotation_ref =
                    motion_controller::common::math_utils::rotationCubic(
                        elapsed,
                        0.0,
                        active_motion_duration_,
                        movel_start_pose_.linear(),
                        movel_goal_pose_.linear());
                const Eigen::Vector3d angular_ref =
                    motion_controller::common::math_utils::rotationCubicDot(
                        elapsed,
                        0.0,
                        active_motion_duration_,
                        Eigen::Vector3d::Zero(),
                        Eigen::Vector3d::Zero(),
                        movel_start_pose_.linear(),
                        movel_goal_pose_.linear());

                Eigen::Affine3d pose_ref = Eigen::Affine3d::Identity();
                pose_ref.translation() = position_ref;
                pose_ref.linear() = rotation_ref;

                desired_task_vel =
                    computeDesiredVelocity(current_pose, pose_ref, linear_ref, angular_ref);
            } else {
                if (movel_trajectory_active_) {
                    movel_trajectory_active_ = false;
                    RCLCPP_INFO(this->get_logger(), "moveL command completed.");
                }
                desired_task_vel = computeDesiredVelocity(current_pose, movel_goal_pose_);
            }

            motion_controller::common::Vector6d task_weight =
                motion_controller::common::Vector6d::Zero();
            task_weight.head<3>().setConstant(weight_task_position_);
            task_weight.tail<3>().setConstant(weight_task_orientation_);
            const Eigen::VectorXd damping_weight =
                Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;

            qp_controller_->setDesiredTaskVel(desired_task_vel);
            qp_controller_->setWeights(task_weight, damping_weight);

            Eigen::VectorXd optimal_velocities;
            if (!qp_controller_->getOptJointVel(optimal_velocities)) {
                publishControllerError("OMY MoveL Controller: QP solve failed");
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    1000,
                    "OMY MoveL Controller QP solver failed");
                return;
            }

            q_commanded_ = q_feedback + optimal_velocities * time_step_;
            publishTrajectory(q_commanded_);
        } catch (const std::exception& e) {
            publishControllerError("OMY MoveL Controller loop error: " + std::string(e.what()));
            RCLCPP_ERROR(this->get_logger(), "OMY MoveL Controller loop error: %s", e.what());
        }
    }
}  // namespace motion_controller_ros

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<motion_controller_ros::OmyMoveLControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
