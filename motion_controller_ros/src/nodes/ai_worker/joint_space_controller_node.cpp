#include "motion_controller_ros/nodes/ai_worker/joint_space_controller_node.hpp"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>

namespace motion_controller_ros
{
    JointSpaceController::JointSpaceController()
        : Node("joint_space_controller"),
          right_traj_received_(false),
          left_traj_received_(false),
          joint_state_received_(false),
          open_loop_initialized_(false),
          right_gripper_position_(0.0),
          left_gripper_position_(0.0),
          right_traj_time_from_start_(0, 0),
          left_traj_time_from_start_(0, 0),
          dt_(0.01)
    {
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "Joint Space Controller - Starting up...");
        RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
        RCLCPP_INFO(this->get_logger(), "========================================");

        control_frequency_ = this->declare_parameter("control_frequency", 100.0);
        time_step_ = this->declare_parameter("time_step", 0.01);
        weight_tracking_ = this->declare_parameter("weight_tracking", 1.0);
        weight_damping_ = this->declare_parameter("weight_damping", 1.0);
        slack_penalty_ = this->declare_parameter("slack_penalty", 1000.0);
        cbf_alpha_ = this->declare_parameter("cbf_alpha", 5.0);
        collision_buffer_ = this->declare_parameter("collision_buffer", 0.05);
        collision_safe_distance_ = this->declare_parameter("collision_safe_distance", 0.02);
        urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
        srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
        joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
        right_traj_topic_ = this->declare_parameter("right_traj_topic", std::string("/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"));
        left_traj_topic_ = this->declare_parameter("left_traj_topic", std::string("/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"));
        right_traj_filtered_topic_ = this->declare_parameter("right_traj_filtered_topic", std::string("/leader/joint_trajectory_command_broadcaster_right/joint_trajectory_filtered"));
        left_traj_filtered_topic_ = this->declare_parameter("left_traj_filtered_topic", std::string("/leader/joint_trajectory_command_broadcaster_left/joint_trajectory_filtered"));
        right_gripper_joint_name_ = this->declare_parameter("right_gripper_joint", std::string("gripper_r_joint1"));
        left_gripper_joint_name_ = this->declare_parameter("left_gripper_joint", std::string("gripper_l_joint1"));
        command_timeout_ = this->declare_parameter("command_timeout", 0.1);

        dt_ = time_step_;

        r_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            right_traj_topic_, 10,
            std::bind(&JointSpaceController::rightTrajectoryCallback, this, std::placeholders::_1));

        l_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            left_traj_topic_, 10,
            std::bind(&JointSpaceController::leftTrajectoryCallback, this, std::placeholders::_1));

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic_, 10,
            std::bind(&JointSpaceController::jointStateCallback, this, std::placeholders::_1));

        arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            right_traj_filtered_topic_, 10);
        arm_l_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            left_traj_filtered_topic_, 10);

        try {
            if (urdf_path_.empty()) {
                throw std::runtime_error("URDF path not provided.");
            }
            if (srdf_path_.empty()) {
                throw std::runtime_error("SRDF path not provided.");
            }
            RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "SRDF path: %s", srdf_path_.c_str());
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(),
                "Failed to resolve URDF/SRDF paths: %s\n",
                e.what());
            rclcpp::shutdown();
            return;
        }

        try {
            RCLCPP_INFO(this->get_logger(), "Loading URDF and initializing kinematics solver...");
            kinematics_solver_ = std::make_shared<motion_controller::kinematics::KinematicsSolver>(urdf_path_, srdf_path_);
            RCLCPP_INFO(this->get_logger(), "Initializing QP filter...");
            qp_filter_ = std::make_shared<motion_controller::controllers::QPFilter>(kinematics_solver_, dt_);
            qp_filter_->setControllerParams(slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);

            int dof = kinematics_solver_->getDof();
            q_.setZero(dof);
            qdot_.setZero(dof);
            q_desired_.setZero(dof);
            qdot_desired_.setZero(dof);

            RCLCPP_INFO(this->get_logger(), "Joint-space filter initialized (DOF: %d)", dof);
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to initialize filter: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        initializeJointConfig();

        int timer_period_ms = static_cast<int>(1000.0 / control_frequency_);
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_period_ms),
            std::bind(&JointSpaceController::controlLoopCallback, this));

        if (!control_timer_) {
            RCLCPP_FATAL(this->get_logger(), "Failed to create control loop timer!");
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(this->get_logger(),
            "Joint Space Controller initialized successfully!");
            RCLCPP_INFO(this->get_logger(), 
            "  - Control loop: %.1f Hz (period: %d ms)", control_frequency_, timer_period_ms);
        RCLCPP_INFO(this->get_logger(), 
            "  - Subscriptions: joint_states=%s",
            joint_state_sub_ ? "OK" : "FAILED");
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "Node is ready! Waiting for messages...");
    }

    JointSpaceController::~JointSpaceController()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down Joint Space Controller");
    }

    void JointSpaceController::initializeJointConfig()
    {
        const auto joint_names = kinematics_solver_->getJointNames();
        model_joint_names_ = joint_names;
        model_joint_index_map_.clear();
        for (size_t i = 0; i < model_joint_names_.size(); ++i) {
            model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
        }

        left_arm_joints_.clear();
        right_arm_joints_.clear();
        for (const auto& joint_name : joint_names) {
            if (joint_name.find("arm_l_joint") != std::string::npos) {
                left_arm_joints_.push_back(joint_name);
            } else if (joint_name.find("arm_r_joint") != std::string::npos) {
                right_arm_joints_.push_back(joint_name);
            }
        }

        std::sort(left_arm_joints_.begin(), left_arm_joints_.end());
        std::sort(right_arm_joints_.begin(), right_arm_joints_.end());
    }

    void JointSpaceController::rightTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
    {
        if (!msg->points.empty()) {
            right_traj_time_from_start_ = msg->points.front().time_from_start;
        }
        if (!msg->points.empty()) {
            const auto& point = msg->points.front();
            if (!point.positions.empty()) {
                for (size_t i = 0; i < msg->joint_names.size(); ++i) {
                    if (msg->joint_names[i] == right_gripper_joint_name_) {
                        if (i < point.positions.size()) {
                            right_gripper_position_ = point.positions[i];
                        }
                        break;
                    }
                }
            }
        }
        updateDesiredVelocityFromTrajectory(*msg, qdot_desired_);
        right_traj_received_ = true;
        last_right_traj_time_ = this->now();
    }

    void JointSpaceController::leftTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
    {
        if (!msg->points.empty()) {
            left_traj_time_from_start_ = msg->points.front().time_from_start;
        }
        if (!msg->points.empty()) {
            const auto& point = msg->points.front();
            if (!point.positions.empty()) {
                for (size_t i = 0; i < msg->joint_names.size(); ++i) {
                    if (msg->joint_names[i] == left_gripper_joint_name_) {
                        if (i < point.positions.size()) {
                            left_gripper_position_ = point.positions[i];
                        }
                        break;
                    }
                }
            }
        }
        updateDesiredVelocityFromTrajectory(*msg, qdot_desired_);
        left_traj_received_ = true;
        last_left_traj_time_ = this->now();
    }

    void JointSpaceController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        try {
            if (joint_index_map_.empty()) {
                for (size_t i = 0; i < msg->name.size(); ++i) {
                    joint_index_map_[msg->name[i]] = static_cast<int>(i);
                }
            }

            if (open_loop_initialized_) {
                return;  // State is last commanded; ignore further joint_states.
            }

            extractJointStates(msg);
            joint_state_received_ = true;
            open_loop_initialized_ = true;
            RCLCPP_INFO(this->get_logger(),
                "Initial pose captured from %s (one-time seed for open-loop)", joint_states_topic_.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in jointStateCallback: %s", e.what());
        }
    }

    void JointSpaceController::extractJointStates(const sensor_msgs::msg::JointState::SharedPtr& msg)
    {
        int dof = kinematics_solver_->getDof();
        q_.setZero(dof);
        qdot_.setZero(dof);

        const int max_index = std::min<int>(dof, static_cast<int>(model_joint_names_.size()));
        for (int i = 0; i < max_index; ++i) {
            const auto& joint_name = model_joint_names_[i];
            auto it = joint_index_map_.find(joint_name);
            if (it != joint_index_map_.end()) {
                int idx = it->second;
                if (idx < static_cast<int>(msg->position.size())) {
                    q_[i] = msg->position[idx];
                }
                if (idx < static_cast<int>(msg->velocity.size())) {
                    qdot_[i] = msg->velocity[idx];
                }
            }
        }
    }

    void JointSpaceController::updateDesiredVelocityFromTrajectory(
        const trajectory_msgs::msg::JointTrajectory& msg,
        VectorXd& qdot_desired)
    {
        if (msg.points.empty()) {
            return;
        }
        const auto& point = msg.points.front();
        const bool has_vel = !point.velocities.empty();
        const bool has_pos = !point.positions.empty();

        for (size_t i = 0; i < msg.joint_names.size(); ++i) {
            const auto& joint_name = msg.joint_names[i];
            auto model_it = model_joint_index_map_.find(joint_name);
            if (model_it == model_joint_index_map_.end()) {
                continue;
            }
            int model_idx = model_it->second;
            if (model_idx < 0 || model_idx >= qdot_desired.size()) {
                continue;
            }
            if (has_vel && i < point.velocities.size()) {
                qdot_desired[model_idx] = point.velocities[i];
            } else if (has_pos && i < point.positions.size()) {
                qdot_desired[model_idx] = (point.positions[i] - q_[model_idx]) / dt_;
            }
        }
    }

    void JointSpaceController::controlLoopCallback()
    {
        static int debug_count = 0;

        if (!open_loop_initialized_) {
            if (debug_count++ % 100 == 0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Waiting for initial pose from %s...", joint_states_topic_.c_str());
            }
            return;
        }

        try {
            const rclcpp::Time now = this->now();
            const bool right_recent = right_traj_received_ &&
                (now - last_right_traj_time_).seconds() < command_timeout_;
            const bool left_recent = left_traj_received_ &&
                (now - last_left_traj_time_).seconds() < command_timeout_;
            const bool has_recent_input = right_recent || left_recent;

            if (!has_recent_input) {
                // No trajectory received recently: hold last joint position at 100 Hz
                publishTrajectory(q_);
                return;
            }

            // qdot_desired_ is set by trajectory callbacks
            kinematics_solver_->updateState(q_, qdot_);

            VectorXd w_tracking = VectorXd::Ones(kinematics_solver_->getDof()) * weight_tracking_;
            VectorXd w_damping = VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;
            qp_filter_->setWeight(w_tracking, w_damping);
            qp_filter_->setDesiredJointVel(qdot_desired_);

            VectorXd optimal_velocities;
            if (!qp_filter_->getOptJointVel(optimal_velocities)) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "QP filter failed to converge");
                return;
            }

            q_desired_ = q_ + optimal_velocities * dt_;
            publishTrajectory(q_desired_);

            q_ = q_desired_;
            qdot_ = optimal_velocities;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in control loop: %s", e.what());
        }
    }

    void JointSpaceController::publishTrajectory(const VectorXd& q_desired)
    {
        try {
            std::vector<int> left_arm_indices, right_arm_indices;
            for (const auto& joint_name : left_arm_joints_) {
                auto it = model_joint_index_map_.find(joint_name);
                if (it != model_joint_index_map_.end()) {
                    left_arm_indices.push_back(it->second);
                }
            }
            for (const auto& joint_name : right_arm_joints_) {
                auto it = model_joint_index_map_.find(joint_name);
                if (it != model_joint_index_map_.end()) {
                    right_arm_indices.push_back(it->second);
                }
            }

            if (!left_arm_indices.empty()) {
                auto traj_left = createTrajectoryMsgWithGripper(
                    left_arm_joints_, q_desired, left_arm_indices, left_gripper_joint_name_,
                    left_gripper_position_,
                    left_traj_time_from_start_);
                arm_l_pub_->publish(traj_left);
            }

            if (!right_arm_indices.empty()) {
                auto traj_right = createTrajectoryMsgWithGripper(
                    right_arm_joints_, q_desired, right_arm_indices, right_gripper_joint_name_,
                    right_gripper_position_,
                    right_traj_time_from_start_);
                arm_r_pub_->publish(traj_right);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error publishing trajectory: %s", e.what());
        }
    }

    trajectory_msgs::msg::JointTrajectory JointSpaceController::createTrajectoryMsgWithGripper(
        const std::vector<std::string>& arm_joint_names,
        const VectorXd& positions,
        const std::vector<int>& arm_indices,
        const std::string& gripper_joint_name,
        double gripper_position,
        const rclcpp::Duration& time_from_start) const
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.header.frame_id = "";
        traj_msg.joint_names = arm_joint_names;
        traj_msg.joint_names.push_back(gripper_joint_name);

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.time_from_start = time_from_start;
        for (int idx : arm_indices) {
            if (idx >= 0 && idx < static_cast<int>(positions.size())) {
                point.positions.push_back(positions[idx]);
            }
        }
        point.positions.push_back(gripper_position);
        traj_msg.points.push_back(point);
        return traj_msg;
    }

}  // namespace motion_controller_ros

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<motion_controller_ros::JointSpaceController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
