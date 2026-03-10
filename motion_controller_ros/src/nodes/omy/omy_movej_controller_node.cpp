#include "motion_controller_ros/nodes/omy/omy_movej_controller_node.hpp"

#include "motion_controller_core/common/type_define.h"

namespace motion_controller_ros
{
    OmyMoveJControllerNode::OmyMoveJControllerNode()
        : Node("omy_movej_controller"),
          joint_state_received_(false),
          commanded_state_initialized_(false),
          movej_target_initialized_(false),
          movej_trajectory_active_(false),
          motion_start_time_(this->now()),
          active_motion_duration_(0.0)
    {
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "OMY MoveJ Controller - Starting up...");
        RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
        RCLCPP_INFO(this->get_logger(), "========================================");

        control_frequency_ = this->declare_parameter("control_frequency", 100.0);
        time_step_ = this->declare_parameter("time_step", 0.01);
        trajectory_time_ = this->declare_parameter("trajectory_time", 0.05);
        kp_joint_ = this->declare_parameter("kp_joint", 6.0);
        weight_joint_tracking_ = this->declare_parameter("weight_joint_tracking", 2.0);
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
        movej_topic_ = this->declare_parameter("movej_topic", std::string("~/movej"));
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
            std::bind(&OmyMoveJControllerNode::jointStateCallback, this, std::placeholders::_1));
        movej_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            movej_topic_, 10,
            std::bind(&OmyMoveJControllerNode::moveJCallback, this, std::placeholders::_1));

        try {
            RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "SRDF path: %s", srdf_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "Loading URDF and initializing kinematics solver...");
            kinematics_solver_ =
                std::make_shared<motion_controller::kinematics::KinematicsSolver>(urdf_path_, srdf_path_);
            RCLCPP_INFO(this->get_logger(), "Initializing QP controller...");
            qp_controller_ = std::make_shared<motion_controller::controllers::OMYMoveJController>(
                kinematics_solver_, time_step_);
            qp_controller_->setControllerParams(
                slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);

            q_.setZero(kinematics_solver_->getDof());
            qdot_.setZero(kinematics_solver_->getDof());
            q_commanded_.setZero(kinematics_solver_->getDof());
            movej_start_.setZero(kinematics_solver_->getDof());
            movej_goal_.setZero(kinematics_solver_->getDof());

            initializeJointConfig();

            RCLCPP_INFO(this->get_logger(), "Motion controller initialized (DOF: %d)", kinematics_solver_->getDof());
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to initialize OMY MoveJ Controller: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        const int timer_period_ms =
            std::max(1, static_cast<int>(std::round(1000.0 / std::max(1.0, control_frequency_))));
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_period_ms),
            std::bind(&OmyMoveJControllerNode::controlLoopCallback, this));

        if (!control_timer_) {
            RCLCPP_FATAL(this->get_logger(), "Failed to create control loop timer!");
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(this->get_logger(), "OMY MoveJ Controller initialized successfully!");
        RCLCPP_INFO(this->get_logger(), "  - Controlled link: %s", controlled_link_.c_str());
        RCLCPP_INFO(
            this->get_logger(),
            "  - Control loop: %.1f Hz (period: %d ms)",
            control_frequency_,
            timer_period_ms);
        RCLCPP_INFO(this->get_logger(), "  - MoveJ command topic: %s", movej_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "Node is ready! Waiting for messages...");
    }

    OmyMoveJControllerNode::~OmyMoveJControllerNode()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down OMY MoveJ Controller");
    }

    void OmyMoveJControllerNode::initializeJointConfig()
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

    void OmyMoveJControllerNode::extractJointStates(const sensor_msgs::msg::JointState::SharedPtr& msg)
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

    void OmyMoveJControllerNode::publishCurrentPose(const Eigen::Affine3d& pose) const
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

    void OmyMoveJControllerNode::publishTrajectory(const Eigen::VectorXd& q_command) const
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

    void OmyMoveJControllerNode::publishControllerError(const std::string& error) const
    {
        if (!controller_error_pub_) {
            return;
        }

        std_msgs::msg::String err;
        err.data = error;
        controller_error_pub_->publish(err);
    }

    void OmyMoveJControllerNode::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
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
            movej_start_ = q_;
            movej_goal_ = q_;
            commanded_state_initialized_ = true;
            movej_target_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Initial joint state captured for moveJ control.");
        }
    }

    void OmyMoveJControllerNode::moveJCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
    {
        if (!msg || msg->points.empty() || !joint_state_received_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Ignoring moveJ command until joint states are available.");
            return;
        }

        Eigen::VectorXd target_q = q_commanded_;
        const auto& point = msg->points.front();

        if (!msg->joint_names.empty()) {
            for (size_t i = 0; i < msg->joint_names.size(); ++i) {
                if (i >= point.positions.size()) {
                    continue;
                }
                const auto it = model_joint_index_map_.find(msg->joint_names[i]);
                if (it == model_joint_index_map_.end()) {
                    continue;
                }
                target_q[it->second] = point.positions[i];
            }
        } else if (point.positions.size() == model_joint_names_.size()) {
            for (size_t i = 0; i < model_joint_names_.size(); ++i) {
                target_q[static_cast<int>(i)] = point.positions[i];
            }
        } else {
            const std::string error =
                "moveJ command ignored: joint_names missing and positions size does not match model joints.";
            publishControllerError(error);
            RCLCPP_WARN(this->get_logger(), "%s", error.c_str());
            return;
        }

        movej_start_ = q_commanded_;
        movej_goal_ = target_q;
        active_motion_duration_ = rclcpp::Duration(point.time_from_start).seconds();
        if (active_motion_duration_ <= 1e-6) {
            const std::string error =
                "moveJ command ignored: time_from_start must be greater than zero.";
            publishControllerError(error);
            RCLCPP_WARN(this->get_logger(), "%s", error.c_str());
            return;
        }
        motion_start_time_ = this->now();
        movej_target_initialized_ = true;
        movej_trajectory_active_ = true;

        RCLCPP_INFO(
            this->get_logger(),
            "Received moveJ command (duration: %.3f s)",
            active_motion_duration_);
    }

    void OmyMoveJControllerNode::controlLoopCallback()
    {
        if (!joint_state_received_ || !commanded_state_initialized_ || !movej_target_initialized_) {
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
            publishCurrentPose(kinematics_solver_->getPose(controlled_link_));

            const double elapsed = (this->now() - motion_start_time_).seconds();
            const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(movej_start_.size());

            Eigen::VectorXd q_ref = movej_goal_;
            Eigen::VectorXd qdot_ref = zeros;

            if (movej_trajectory_active_ && elapsed < active_motion_duration_) {
                q_ref = motion_controller::common::math_utils::cubicVector(
                    elapsed, 0.0, active_motion_duration_, movej_start_, movej_goal_, zeros, zeros);
                qdot_ref = motion_controller::common::math_utils::cubicDotVector(
                    elapsed, 0.0, active_motion_duration_, movej_start_, movej_goal_, zeros, zeros);
            } else if (movej_trajectory_active_) {
                movej_trajectory_active_ = false;
                RCLCPP_INFO(this->get_logger(), "moveJ command completed.");
            }

            const Eigen::VectorXd desired_joint_vel =
                qdot_ref + kp_joint_ * (q_ref - q_feedback);
            const Eigen::VectorXd joint_weight =
                Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_joint_tracking_;
            const Eigen::VectorXd damping_weight =
                Eigen::VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;

            qp_controller_->setDesiredJointVel(desired_joint_vel);
            qp_controller_->setWeights(joint_weight, damping_weight);

            Eigen::VectorXd optimal_velocities;
            if (!qp_controller_->getOptJointVel(optimal_velocities)) {
                publishControllerError("OMY MoveJ Controller: QP solve failed");
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    1000,
                    "OMY MoveJ Controller QP solver failed");
                return;
            }

            q_commanded_ = q_feedback + optimal_velocities * time_step_;
            publishTrajectory(q_commanded_);
        } catch (const std::exception& e) {
            publishControllerError("OMY MoveJ Controller loop error: " + std::string(e.what()));
            RCLCPP_ERROR(this->get_logger(), "OMY MoveJ Controller loop error: %s", e.what());
        }
    }
}  // namespace motion_controller_ros

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<motion_controller_ros::OmyMoveJControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
