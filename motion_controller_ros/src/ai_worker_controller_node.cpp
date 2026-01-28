#include "motion_controller_ros/ai_worker_controller_node.hpp"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>

namespace motion_controller_ros
{
    AIWorkerController::AIWorkerController()
        : Node("ai_worker_controller"),
          goal_pose_received_(false),
          joint_state_received_(false),
          dt_(DEFAULT_TIME_STEP)
    {
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "AI Worker Controller - Starting up...");
        RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
        RCLCPP_INFO(this->get_logger(), "========================================");

        // Initialize subscribers
        r_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            R_GOAL_POSE_TOPIC, 10,
            std::bind(&AIWorkerController::goalPoseCallback, this, std::placeholders::_1));
        
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            JOINT_STATES_TOPIC, 10,
            std::bind(&AIWorkerController::jointStateCallback, this, std::placeholders::_1));
        
        if (!joint_state_sub_) {
            RCLCPP_FATAL(this->get_logger(), "Failed to create /joint_states subscription!");
        }

        // Initialize publishers
        // lift_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
        //     LIFT_TRAJ_TOPIC, 10);

        arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            RIGHT_TRAJ_TOPIC, 10);

        arm_l_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            LEFT_TRAJ_TOPIC, 10);

        r_gripper_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            R_GRIPPER_POSE_TOPIC, 10);

        // Initialize motion controller
        std::string urdf_path;
        try {
            std::string package_path = ament_index_cpp::get_package_share_directory("ffw_description");
            urdf_path = package_path + "/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf";
            RCLCPP_INFO(this->get_logger(), "URDF path: %s", urdf_path.c_str());
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), 
                "Failed to find ffw_description package: %s\n"
                "Make sure ffw_description package is installed and sourced.",
                e.what());
            rclcpp::shutdown();
            return;
        }
        
        try {
            RCLCPP_INFO(this->get_logger(), "Loading URDF and initializing kinematics solver...");
            kinematics_solver_ = std::make_shared<motion_controller_core::KinematicsSolver>(urdf_path);
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

        // Create control loop timer (100Hz = 10ms period)
        int timer_period_ms = static_cast<int>(1000.0 / DEFAULT_CONTROL_FREQUENCY);
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
            "  - Control loop: %.1f Hz (period: %d ms)", DEFAULT_CONTROL_FREQUENCY, timer_period_ms);
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
        
        // Parse joint names: expect arm_l_joint*, arm_r_joint*, lift_joint
        // Note: gripper joints are fixed in URDF, so Pinocchio doesn't include them in the model
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
        
        // Hardcode gripper joint names (they are fixed joints in URDF, so Pinocchio won't detect them)
        // These are required by the controllers even though they're not part of the IK calculation
        left_gripper_joint_ = LEFT_GRIPPER_JOINT;
        right_gripper_joint_ = RIGHT_GRIPPER_JOINT;
        
        
        // Log the actual joint names for debugging
        std::string left_str, right_str;
        for (const auto& j : left_arm_joints_) left_str += j + " ";
        for (const auto& j : right_arm_joints_) right_str += j + " ";
        RCLCPP_DEBUG(this->get_logger(), "Left arm joints: %s", left_str.c_str());
        RCLCPP_DEBUG(this->get_logger(), "Right arm joints: %s", right_str.c_str());
        // if (!lift_joint_.empty()) {
        //     RCLCPP_DEBUG(this->get_logger(), "Lift joint: %s", lift_joint_.c_str());
        // }
    }

    void AIWorkerController::goalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        goal_pose_ = *msg;
        goal_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Goal pose received: [%.3f, %.3f, %.3f]", 
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
        // This is critical for correct kinematics/Jacobian calculations.
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

    void AIWorkerController::controlLoopCallback()
    {
        static int loop_count = 0;
        static int debug_count = 0;
        static rclcpp::Time last_heartbeat;
        static bool first_call = true;
        
        if (first_call) {
            last_heartbeat = this->now();
            first_call = false;
        }
        
        loop_count++;
        
        // Log heartbeat every second to confirm control loop is running
        rclcpp::Time current_time = this->now();
        double elapsed = (current_time - last_heartbeat).seconds();
        if (elapsed >= 1.0) {
            double actual_freq = static_cast<double>(loop_count) / elapsed;
            RCLCPP_INFO(this->get_logger(),
                "Control loop running: %.1f Hz (target: %.1f Hz), goal_pose=%s, joint_state=%s",
                actual_freq, DEFAULT_CONTROL_FREQUENCY,
                goal_pose_received_ ? "received" : "waiting",
                joint_state_received_ ? "received" : "waiting");
            loop_count = 0;
            last_heartbeat = current_time;
        }
        
        if (!goal_pose_received_) {
            if (debug_count++ % DEBUG_LOG_INTERVAL == 0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Control loop waiting for goal pose...");
            }
            return;
        }
        
        if (!joint_state_received_) {
            if (debug_count++ % DEBUG_LOG_INTERVAL == 0) {
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
            Affine3d current_ee_pose = kinematics_solver_->computePose(q_, R_GRIPPER_NAME);
            Affine3d goal_ee_pose = computeGoalPose();

            // Publish current end-effector pose
            if (r_gripper_pose_pub_) {
                geometry_msgs::msg::PoseStamped ee_msg;
                ee_msg.header.stamp = this->now();
                ee_msg.header.frame_id = BASE_FRAME_ID;
                ee_msg.pose.position.x = current_ee_pose.translation().x();
                ee_msg.pose.position.y = current_ee_pose.translation().y();
                ee_msg.pose.position.z = current_ee_pose.translation().z();
                Eigen::Quaterniond ee_quat(current_ee_pose.linear());
                ee_msg.pose.orientation.w = ee_quat.w();
                ee_msg.pose.orientation.x = ee_quat.x();
                ee_msg.pose.orientation.y = ee_quat.y();
                ee_msg.pose.orientation.z = ee_quat.z();
                r_gripper_pose_pub_->publish(ee_msg);
            }

            // Compute desired velocity
            Vector6d desired_vel = computeDesiredVelocity(current_ee_pose, goal_ee_pose);
            
            std::map<std::string, Vector6d> desired_task_velocities;
            desired_task_velocities["arm_r_link7"] = desired_vel;

            // Set weights for QP solver
            std::map<std::string, Vector6d> weights;
            Vector6d weight_right = Vector6d::Ones();
            weights["arm_r_link7"] = weight_right;
            
            VectorXd damping = VectorXd::Ones(kinematics_solver_->getDof()) * DEFAULT_KV;

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
            
            // Log successful control loop execution (throttled)
            static int success_count = 0;
            if (++success_count % 1000 == 0) {
                RCLCPP_DEBUG(this->get_logger(), 
                    "Control loop executed successfully %d times", success_count);
            }

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in control loop: %s", e.what());
        }
    }

    Affine3d AIWorkerController::computeGoalPose() const
    {
        Affine3d goal_ee_pose = Affine3d::Identity();
        goal_ee_pose.translation() << goal_pose_.pose.position.x, 
                                     goal_pose_.pose.position.y, 
                                     goal_pose_.pose.position.z;
        
        Eigen::Quaterniond quat(goal_pose_.pose.orientation.w,
                               goal_pose_.pose.orientation.x,
                               goal_pose_.pose.orientation.y,
                               goal_pose_.pose.orientation.z);
        goal_ee_pose.linear() = quat.toRotationMatrix();
        
        return goal_ee_pose;
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
        desired_vel.head(3) = DEFAULT_KP * pos_error;
        desired_vel.tail(3) = DEFAULT_KP * angle_axis;
        
        return desired_vel;
    }

    void AIWorkerController::publishTrajectory(const VectorXd& q_desired)
    {
        try {
            // Build indices for each arm segment
            // std::vector<int> left_arm_indices, right_arm_indices, lift_indices;
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
                    left_arm_joints_, q_desired, left_arm_indices, left_gripper_joint_);
                arm_l_pub_->publish(traj_left);
            }

            // Publish right arm trajectory (include gripper joint with position 0)
            if (!right_arm_indices.empty()) {
                auto traj_right = createTrajectoryMsgWithGripper(
                    right_arm_joints_, q_desired, right_arm_indices, right_gripper_joint_);
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

    trajectory_msgs::msg::JointTrajectory AIWorkerController::createTrajectoryMsg(
        const std::vector<std::string>& joint_names,
        const VectorXd& positions,
        const std::vector<int>& indices) const
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.header.frame_id = TRAJ_FRAME_ID;
        traj_msg.joint_names = joint_names;

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.time_from_start = rclcpp::Duration::from_seconds(DEFAULT_TRAJECTORY_TIME);

        for (int idx : indices) {
            if (idx >= 0 && idx < static_cast<int>(positions.size())) {
                point.positions.push_back(positions[idx]);
            }
        }

        traj_msg.points.push_back(point);
        return traj_msg;
    }

    trajectory_msgs::msg::JointTrajectory AIWorkerController::createTrajectoryMsgWithGripper(
        const std::vector<std::string>& arm_joint_names,
        const VectorXd& positions,
        const std::vector<int>& arm_indices,
        const std::string& gripper_joint_name) const
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.header.frame_id = TRAJ_FRAME_ID;
        
        // Add arm joint names
        traj_msg.joint_names = arm_joint_names;
        
        // Always add gripper joint name (hardcoded, required by controllers)
        // Gripper joints are fixed in URDF so Pinocchio doesn't include them in the model
        traj_msg.joint_names.push_back(gripper_joint_name);

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.time_from_start = rclcpp::Duration::from_seconds(DEFAULT_TRAJECTORY_TIME);

        // Add arm joint positions
        for (int idx : arm_indices) {
            if (idx >= 0 && idx < static_cast<int>(positions.size())) {
                point.positions.push_back(positions[idx]);
            }
        }
        
        // Always add gripper joint position (always 0.0 as required by controllers)
        point.positions.push_back(0.0);

        traj_msg.points.push_back(point);
        return traj_msg;
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
