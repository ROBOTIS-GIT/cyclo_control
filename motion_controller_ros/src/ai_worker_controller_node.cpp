#include "motion_controller_ros/ai_worker_controller_node.hpp"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>

namespace motion_controller_ros
{
    AIWorkerController::AIWorkerController()
        : Node("ai_worker_controller"),
          r_goal_pose_received_(false),
          l_goal_pose_received_(false),
          r_elbow_pose_received_(false),
          l_elbow_pose_received_(false),
          joint_state_received_(false),
          dt_(0.01)
    {
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "AI Worker Controller - Starting up...");
        RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
        RCLCPP_INFO(this->get_logger(), "========================================");

        // Load parameters
        control_frequency_ = this->declare_parameter("control_frequency", 100.0);
        time_step_ = this->declare_parameter("time_step", 0.01);
        trajectory_time_ = this->declare_parameter("trajectory_time", 0.0);
        kp_position_ = this->declare_parameter("kp_position", 50.0);
        kp_orientation_ = this->declare_parameter("kp_orientation", 50.0);
        weight_position_ = this->declare_parameter("weight_position", 10.0);
        weight_orientation_ = this->declare_parameter("weight_orientation", 1.0);
        weight_elbow_position_ = this->declare_parameter("weight_elbow_position", 0.5);
        weight_damping_ = this->declare_parameter("weight_damping", 0.1);
        debug_log_interval_ = this->declare_parameter("debug_log_interval", 100);
        r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
        l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));
        r_elbow_pose_topic_ = this->declare_parameter("r_elbow_pose_topic", std::string("/r_elbow_pose"));
        l_elbow_pose_topic_ = this->declare_parameter("l_elbow_pose_topic", std::string("/l_elbow_pose"));
        joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
        right_traj_topic_ = this->declare_parameter("right_traj_topic", std::string("/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"));
        left_traj_topic_ = this->declare_parameter("left_traj_topic", std::string("/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"));
        r_gripper_pose_topic_ = this->declare_parameter("r_gripper_pose_topic", std::string("/r_gripper_pose"));
        l_gripper_pose_topic_ = this->declare_parameter("l_gripper_pose_topic", std::string("/l_gripper_pose"));
        r_gripper_name_ = this->declare_parameter("r_gripper_name", std::string("arm_r_link7"));
        l_gripper_name_ = this->declare_parameter("l_gripper_name", std::string("arm_l_link7"));
        r_elbow_name_ = this->declare_parameter("r_elbow_name", std::string("arm_r_link4"));
        l_elbow_name_ = this->declare_parameter("l_elbow_name", std::string("arm_l_link4"));
        base_frame_id_ = this->declare_parameter("base_frame_id", std::string("base_link"));
        traj_frame_id_ = this->declare_parameter("traj_frame_id", std::string(""));
        right_gripper_joint_name_ = this->declare_parameter("right_gripper_joint", std::string("gripper_r_joint1"));
        left_gripper_joint_name_ = this->declare_parameter("left_gripper_joint", std::string("gripper_l_joint1"));

        dt_ = time_step_;

        // Initialize subscribers
        r_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            r_goal_pose_topic_, 10,
            std::bind(&AIWorkerController::rightGoalPoseCallback, this, std::placeholders::_1));

        l_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            l_goal_pose_topic_, 10,
            std::bind(&AIWorkerController::leftGoalPoseCallback, this, std::placeholders::_1));    
            
        r_elbow_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            r_elbow_pose_topic_, 10,
            std::bind(&AIWorkerController::rightElbowPoseCallback, this, std::placeholders::_1));
    
        l_elbow_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            l_elbow_pose_topic_, 10,
            std::bind(&AIWorkerController::leftElbowPoseCallback, this, std::placeholders::_1));  
        
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic_, 10,
            std::bind(&AIWorkerController::jointStateCallback, this, std::placeholders::_1));

        // Initialize publishers
        // lift_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
        //     LIFT_TRAJ_TOPIC, 10);

        arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            right_traj_topic_, 10);

        arm_l_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            left_traj_topic_, 10);

        r_gripper_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            r_gripper_pose_topic_, 10);

        l_gripper_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            l_gripper_pose_topic_, 10);    

        // Initialize motion controller
        std::string urdf_path;
        std::string srdf_path;
        try {
            std::string package_path = ament_index_cpp::get_package_share_directory("ffw_description");
            urdf_path = package_path + "/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf";
            srdf_path = package_path + "/urdf/ffw_sg2_rev1_follower/ffw.srdf";
            RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path.c_str());
            RCLCPP_INFO(this->get_logger(), "SRDF path: %s", srdf_path.c_str());
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), 
                "Failed to find ffw_description package: %s\n",
                e.what());
            rclcpp::shutdown();
            return;
        }
        
        try {
            RCLCPP_INFO(this->get_logger(), "Loading URDF and initializing kinematics solver...");
            kinematics_solver_ = std::make_shared<motion_controller_core::KinematicsSolver>(urdf_path, srdf_path);
            RCLCPP_INFO(this->get_logger(), "Initializing QP controller...");
            qp_controller_ = std::make_shared<motion_controller_core::QPIK>(kinematics_solver_, dt_);

            // Initialize state variables
            int dof = kinematics_solver_->getDof();
            q_.setZero(dof);
            qdot_.setZero(dof);
            q_desired_.setZero(dof);

            RCLCPP_INFO(this->get_logger(), "Motion controller initialized (DOF: %d)", dof);
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to initialize motion controller: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        // Initialize joint configuration from URDF
        initializeJointConfig();

        // Create control loop timer
        int timer_period_ms = static_cast<int>(1000.0 / control_frequency_);
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_period_ms),
            std::bind(&AIWorkerController::controlLoopCallback, this));
        
        if (!control_timer_) {
            RCLCPP_FATAL(this->get_logger(), "Failed to create control loop timer!");
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(this->get_logger(), 
            "AI Worker Controller initialized successfully!");
        RCLCPP_INFO(this->get_logger(), 
            "  - Control loop: %.1f Hz (period: %d ms)", control_frequency_, timer_period_ms);
        RCLCPP_INFO(this->get_logger(), 
            "  - Subscriptions: joint_states=%s",
            joint_state_sub_ ? "OK" : "FAILED");
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "Node is ready! Waiting for messages...");
    }

    AIWorkerController::~AIWorkerController()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down AI Worker Controller");
    }

    void AIWorkerController::initializeJointConfig()
    {
        // Get actual joint names from the model (these should match joint_states topic)
        const auto joint_names = kinematics_solver_->getJointNames();
        model_joint_names_ = joint_names;
        model_joint_index_map_.clear();
        for (size_t i = 0; i < model_joint_names_.size(); ++i) {
            model_joint_index_map_[model_joint_names_[i]] = static_cast<int>(i);
        }
        
        left_arm_joints_.clear();
        right_arm_joints_.clear();
        // lift_joint_.clear();
        
        // Parse joint names
        for (const auto& joint_name : joint_names) {
            if (joint_name.find("arm_l_joint") != std::string::npos) {
                left_arm_joints_.push_back(joint_name);
            } else if (joint_name.find("arm_r_joint") != std::string::npos) {
                right_arm_joints_.push_back(joint_name);
            } 
            // else if (joint_name.find("lift_joint") != std::string::npos) {
            //     lift_joint_ = joint_name;
            // }
        }
        
        // Sort to ensure correct ordering
        std::sort(left_arm_joints_.begin(), left_arm_joints_.end());
        std::sort(right_arm_joints_.begin(), right_arm_joints_.end());
        
        
        // Log joint names
        std::string left_str, right_str;
        for (const auto& j : left_arm_joints_) left_str += j + " ";
        for (const auto& j : right_arm_joints_) right_str += j + " ";
        RCLCPP_DEBUG(this->get_logger(), "Left arm joints: %s", left_str.c_str());
        RCLCPP_DEBUG(this->get_logger(), "Right arm joints: %s", right_str.c_str());
        // if (!lift_joint_.empty()) {
        //     RCLCPP_DEBUG(this->get_logger(), "Lift joint: %s", lift_joint_.c_str());
        // }
    }

    void AIWorkerController::rightGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        r_goal_pose_ = computePoseMat(*msg);
        r_goal_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Right goal pose received: [%.3f, %.3f, %.3f]", 
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void AIWorkerController::leftGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        l_goal_pose_ = computePoseMat(*msg);
        l_goal_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Left goal pose received: [%.3f, %.3f, %.3f]", 
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void AIWorkerController::rightElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        r_elbow_pose_ = computePoseMat(*msg);
        r_elbow_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Right elbow pose received: [%.3f, %.3f, %.3f]", 
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void AIWorkerController::leftElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        l_elbow_pose_ = computePoseMat(*msg);
        l_elbow_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Left elbow pose received: [%.3f, %.3f, %.3f]", 
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void AIWorkerController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        try {
            // Build joint index map on first callback
            if (joint_index_map_.empty()) {
                for (size_t i = 0; i < msg->name.size(); ++i) {
                    joint_index_map_[msg->name[i]] = static_cast<int>(i);
                }
            }

            extractJointStates(msg);
            joint_state_received_ = true;
            
            // Initialize q_desired_ from current joint positions on first callback
            static bool positions_initialized = false;
            if (!positions_initialized) {
                q_desired_ = q_;
                positions_initialized = true;
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in jointStateCallback: %s", e.what());
        }
    }

    void AIWorkerController::extractJointStates(const sensor_msgs::msg::JointState::SharedPtr& msg)
    {
        int dof = kinematics_solver_->getDof();
        q_.setZero(dof);
        qdot_.setZero(dof);

        // Fill joint positions/velocities in model joint order
        const int max_index = std::min<int>(dof, static_cast<int>(model_joint_names_.size()));
        for (int i = 0; i < max_index; ++i) {
            const auto& joint_name = model_joint_names_[i];
            auto it = joint_index_map_.find(joint_name);
            if (it != joint_index_map_.end()) {
                int idx = it->second;
                if (idx < static_cast<int>(msg->position.size())) {
                    q_[i] = msg->position[idx];
                }
                // ToDo: Add low pass filter
                if (idx < static_cast<int>(msg->velocity.size())) {
                    qdot_[i] = msg->velocity[idx];
                }
            }
        }
    }

    void AIWorkerController::controlLoopCallback()
    {
        static int loop_count = 0;
        static int debug_count = 0;
        
        loop_count++;
        
        if (!joint_state_received_) {
            if (debug_count++ % debug_log_interval_ == 0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Control loop waiting for joint states...");
            }
            return;
        }

        debug_count = 0;

        try {
            // Control loop is executing - update kinematics solver with current state
            kinematics_solver_->updateState(q_, qdot_);

            // Get current and goal end-effector poses
            right_gripper_pose_ = kinematics_solver_->getPose(r_gripper_name_);
            left_gripper_pose_ = kinematics_solver_->getPose(l_gripper_name_);
            Affine3d right_elbow_pose = kinematics_solver_->getPose(r_elbow_name_);
            Affine3d left_elbow_pose = kinematics_solver_->getPose(l_elbow_name_);

            // Initialize goals to current EE pose on first cycle if not received
            if (!r_goal_pose_received_ && !l_goal_pose_received_) {
                r_goal_pose_ = right_gripper_pose_;
                l_goal_pose_ = left_gripper_pose_;
            }
            if (!r_elbow_pose_received_) {
                r_elbow_pose_ = right_elbow_pose;
            }
            if (!l_elbow_pose_received_) {
                l_elbow_pose_ = left_elbow_pose;
            }

            // Publish current end-effector pose
            publishGripperPose(right_gripper_pose_, left_gripper_pose_);

            // Compute desired velocity
            Vector6d right_desired_vel = computeDesiredVelocity(right_gripper_pose_, r_goal_pose_);
            Vector6d left_desired_vel = computeDesiredVelocity(left_gripper_pose_, l_goal_pose_);
            Vector6d right_elbow_desired_vel = Vector6d::Zero();
            Vector6d left_elbow_desired_vel = Vector6d::Zero();
            right_elbow_desired_vel.head(3) =
                kp_position_ * (r_elbow_pose_.translation() - right_elbow_pose.translation());
            left_elbow_desired_vel.head(3) =
                kp_position_ * (l_elbow_pose_.translation() - left_elbow_pose.translation());
            
            std::map<std::string, Vector6d> desired_task_velocities;
            desired_task_velocities[r_gripper_name_] = right_desired_vel;
            desired_task_velocities[l_gripper_name_] = left_desired_vel;
            desired_task_velocities[r_elbow_name_] = right_elbow_desired_vel;
            desired_task_velocities[l_elbow_name_] = left_elbow_desired_vel;

            // Set weights for QP solver
            std::map<std::string, Vector6d> weights;
            Vector6d weight_right = Vector6d::Ones();
            Vector6d weight_left = Vector6d::Ones();
            weight_right.head(3).setConstant(weight_position_);
            weight_right.tail(3).setConstant(weight_orientation_);
            weight_left.head(3).setConstant(weight_position_);
            weight_left.tail(3).setConstant(weight_orientation_);
            weights[r_gripper_name_] = weight_right;
            weights[l_gripper_name_] = weight_left;
            Vector6d weight_right_elbow = Vector6d::Zero();
            Vector6d weight_left_elbow = Vector6d::Zero();
            weight_right_elbow.head(3).setConstant(weight_elbow_position_);
            weight_left_elbow.head(3).setConstant(weight_elbow_position_);
            weights[r_elbow_name_] = weight_right_elbow;
            weights[l_elbow_name_] = weight_left_elbow;
            
            VectorXd damping = VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;

            // Set weights and desired task velocities in QP controller
            qp_controller_->setWeight(weights, damping);
            qp_controller_->setDesiredTaskVel(desired_task_velocities);

            // Solve QP to get optimal joint velocities
            VectorXd optimal_velocities;
            if (!qp_controller_->getOptJointVel(optimal_velocities)) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                    "QP solver failed to converge");
                return;
            }

            // Compute command from current state (avoid integrating when robot does not move)
            q_desired_ = q_ + optimal_velocities * dt_;

            // Publish trajectory commands
            publishTrajectory(q_desired_);

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in control loop: %s", e.what());
        }
    }

    Affine3d AIWorkerController::computePoseMat(const geometry_msgs::msg::PoseStamped& pose) const
    {
        Affine3d pose_mat = Affine3d::Identity();
        pose_mat.translation() << pose.pose.position.x, 
                                  pose.pose.position.y, 
                                  pose.pose.position.z;
        
        Eigen::Quaterniond quat(pose.pose.orientation.w,
                                pose.pose.orientation.x,
                                pose.pose.orientation.y,
                                pose.pose.orientation.z);
        pose_mat.linear() = quat.toRotationMatrix();
        
        return pose_mat;
    }

    Vector6d AIWorkerController::computeDesiredVelocity(const Affine3d& current_pose, const Affine3d& goal_pose) const
    {
        // Compute position error
        Vector3d pos_error = goal_pose.translation() - current_pose.translation();
        
        // Compute orientation error
        Matrix3d rotation_error = goal_pose.linear() * current_pose.linear().transpose();
        Eigen::AngleAxisd angle_axis_error(rotation_error);
        Vector3d angle_axis = angle_axis_error.axis() * angle_axis_error.angle();

        Vector6d desired_vel = Vector6d::Zero();
        desired_vel.head(3) = kp_position_ * pos_error;
        desired_vel.tail(3) = kp_orientation_ * angle_axis;
        
        return desired_vel;
    }

    void AIWorkerController::publishTrajectory(const VectorXd& q_desired)
    {
        try {
            // Build indices for each arm segment
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
            
            // if (!lift_joint_.empty()) {
            //     auto it = model_joint_index_map_.find(lift_joint_);
            //     if (it != model_joint_index_map_.end()) {
            //         lift_indices.push_back(it->second);
            //     }
            // }

            // Publish left arm trajectory (include gripper joint with position 0)
            if (!left_arm_indices.empty()) {
                auto traj_left = createTrajectoryMsgWithGripper(
                    left_arm_joints_, q_desired, left_arm_indices, left_gripper_joint_name_);
                arm_l_pub_->publish(traj_left);
            }

            // Publish right arm trajectory (include gripper joint with position 0)
            if (!right_arm_indices.empty()) {
                auto traj_right = createTrajectoryMsgWithGripper(
                    right_arm_joints_, q_desired, right_arm_indices, right_gripper_joint_name_);
                arm_r_pub_->publish(traj_right);
            }

            // // Publish lift trajectory
            // if (!lift_indices.empty() && !lift_joint_.empty()) {
            //     std::vector<std::string> lift_names = {lift_joint_};
            //     auto traj_lift = createTrajectoryMsg(lift_names, q_desired, lift_indices);
            //     lift_pub_->publish(traj_lift);
            // }

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error publishing trajectory: %s", e.what());
        }
    }

    trajectory_msgs::msg::JointTrajectory AIWorkerController::createTrajectoryMsgWithGripper(
        const std::vector<std::string>& arm_joint_names,
        const VectorXd& positions,
        const std::vector<int>& arm_indices,
        const std::string& gripper_joint_name) const
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.header.frame_id = traj_frame_id_;
        
        // Add arm joint names
        traj_msg.joint_names = arm_joint_names;
        
        // Always add gripper joint name (required by controllers)
        traj_msg.joint_names.push_back(gripper_joint_name);

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);

        // Add arm joint positions
        for (int idx : arm_indices) {
            if (idx >= 0 && idx < static_cast<int>(positions.size())) {
                point.positions.push_back(positions[idx]);
            }
        }
        
        // Add gripper joint position
        point.positions.push_back(0.0);

        traj_msg.points.push_back(point);
        return traj_msg;
    }

    void AIWorkerController::publishGripperPose(const Affine3d& r_gripper_pose, const Affine3d& l_gripper_pose)
    {
        if (r_gripper_pose_pub_) {
            geometry_msgs::msg::PoseStamped r_gripper_pose_msg;
            r_gripper_pose_msg.header.stamp = this->now();
            r_gripper_pose_msg.header.frame_id = base_frame_id_;
            r_gripper_pose_msg.pose.position.x = r_gripper_pose.translation().x();
            r_gripper_pose_msg.pose.position.y = r_gripper_pose.translation().y();
            r_gripper_pose_msg.pose.position.z = r_gripper_pose.translation().z();
            Eigen::Quaterniond r_gripper_pose_quat(r_gripper_pose.linear());
            r_gripper_pose_msg.pose.orientation.w = r_gripper_pose_quat.w();
            r_gripper_pose_msg.pose.orientation.x = r_gripper_pose_quat.x();
            r_gripper_pose_msg.pose.orientation.y = r_gripper_pose_quat.y();
            r_gripper_pose_msg.pose.orientation.z = r_gripper_pose_quat.z();
            r_gripper_pose_pub_->publish(r_gripper_pose_msg);
        }
        if (l_gripper_pose_pub_) {
            geometry_msgs::msg::PoseStamped l_gripper_pose_msg;
            l_gripper_pose_msg.header.stamp = this->now();
            l_gripper_pose_msg.header.frame_id = base_frame_id_;
            l_gripper_pose_msg.pose.position.x = l_gripper_pose.translation().x();
            l_gripper_pose_msg.pose.position.y = l_gripper_pose.translation().y();
            l_gripper_pose_msg.pose.position.z = l_gripper_pose.translation().z();
            Eigen::Quaterniond l_gripper_pose_quat(l_gripper_pose.linear());
            l_gripper_pose_msg.pose.orientation.w = l_gripper_pose_quat.w();
            l_gripper_pose_msg.pose.orientation.x = l_gripper_pose_quat.x();
            l_gripper_pose_msg.pose.orientation.y = l_gripper_pose_quat.y();
            l_gripper_pose_msg.pose.orientation.z = l_gripper_pose_quat.z();
            l_gripper_pose_pub_->publish(l_gripper_pose_msg);
        }
    }

}  // namespace motion_controller_ros

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<motion_controller_ros::AIWorkerController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
