#include "motion_controller_ros/nodes/ai_worker_wholebody_controller_node.hpp"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <cmath>

namespace motion_controller_ros
{
    AIWorkerWholebodyController::AIWorkerWholebodyController()
        : Node("ai_worker_wholebody_controller"),
          r_goal_pose_received_(false),
          l_goal_pose_received_(false),
          r_elbow_pose_received_(false),
          l_elbow_pose_received_(false),
          reference_diverged_(true),
          activate_pending_(false),
          joint_state_received_(false),
          dt_(0.01)
    {
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "AI Worker Wholebody Controller - Starting up...");
        RCLCPP_INFO(this->get_logger(), "Node name: %s", this->get_name());
        RCLCPP_INFO(this->get_logger(), "========================================");
        activate_start_ = this->get_clock()->now();

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
        weight_damping_base_ = this->declare_parameter("weight_damping_base", 50.0);
        slack_penalty_ = this->declare_parameter("slack_penalty", 1000.0);
        cbf_alpha_ = this->declare_parameter("cbf_alpha", 5.0);
        collision_buffer_ = this->declare_parameter("collision_buffer", 0.05);
        collision_safe_distance_ = this->declare_parameter("collision_safe_distance", 0.02);
        urdf_path_ = this->declare_parameter("urdf_path", std::string(""));
        srdf_path_ = this->declare_parameter("srdf_path", std::string(""));
        reactivate_service_ = this->declare_parameter("reactivate_service", std::string("/reactivate"));
        r_goal_pose_topic_ = this->declare_parameter("r_goal_pose_topic", std::string("/r_goal_pose"));
        l_goal_pose_topic_ = this->declare_parameter("l_goal_pose_topic", std::string("/l_goal_pose"));
        r_elbow_pose_topic_ = this->declare_parameter("r_elbow_pose_topic", std::string("/r_elbow_pose"));
        l_elbow_pose_topic_ = this->declare_parameter("l_elbow_pose_topic", std::string("/l_elbow_pose"));
        base_goal_pose_topic_ = this->declare_parameter("base_goal_pose_topic", std::string("/vr_base_goal_pose"));
        joint_states_topic_ = this->declare_parameter("joint_states_topic", std::string("/joint_states"));
        right_traj_topic_ = this->declare_parameter("right_traj_topic", std::string("/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"));
        left_traj_topic_ = this->declare_parameter("left_traj_topic", std::string("/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"));
        lift_topic_ = this->declare_parameter("lift_topic", std::string("/leader/joystick_controller_right/joint_trajectory"));
        lift_vel_bound_ = this->declare_parameter("lift_vel_bound", 0.0);
        r_gripper_pose_topic_ = this->declare_parameter("r_gripper_pose_topic", std::string("/r_gripper_pose"));
        l_gripper_pose_topic_ = this->declare_parameter("l_gripper_pose_topic", std::string("/l_gripper_pose"));
        r_gripper_name_ = this->declare_parameter("r_gripper_name", std::string("arm_r_link7"));
        l_gripper_name_ = this->declare_parameter("l_gripper_name", std::string("arm_l_link7"));
        r_elbow_name_ = this->declare_parameter("r_elbow_name", std::string("arm_r_link4"));
        l_elbow_name_ = this->declare_parameter("l_elbow_name", std::string("arm_l_link4"));
        base_link_name_ = this->declare_parameter("base_link_name", std::string("arm_base_link"));
        right_gripper_joint_name_ = this->declare_parameter("right_gripper_joint", std::string("gripper_r_joint1"));
        left_gripper_joint_name_ = this->declare_parameter("left_gripper_joint", std::string("gripper_l_joint1"));

        weight_base_position_ = this->declare_parameter("weight_base_position", 3.0);
        weight_base_orientation_ = this->declare_parameter("weight_base_orientation", 10.0);
        
        // Base interface parameters
        odom_topic_ = this->declare_parameter("odom_topic", std::string("/odom"));
        cmd_vel_topic_ = this->declare_parameter("cmd_vel_topic", std::string("/cmd_vel"));
        cmd_vel_in_base_frame_ = this->declare_parameter("cmd_vel_in_base_frame", true);
        cmd_vel_filter_tau_ = this->declare_parameter("cmd_vel_filter_tau", 0.15);
        cmd_vel_max_accel_vx_ = this->declare_parameter("cmd_vel_max_accel_vx", 0.8);
        cmd_vel_max_accel_vy_ = this->declare_parameter("cmd_vel_max_accel_vy", 0.8);
        cmd_vel_max_accel_wz_ = this->declare_parameter("cmd_vel_max_accel_wz", 1.5);

        // Wholebody base actuation lag model (QP-side)
        base_actuation_lag_comp_enabled_ = this->declare_parameter("base_actuation_lag_comp_enabled", true);
        base_actuation_tau_seconds_ = this->declare_parameter("base_actuation_tau_seconds", 0.15);

        startup_ref_pos_threshold_ = this->declare_parameter("startup_ref_pos_threshold", 0.15);
        startup_ref_ori_threshold_deg_ = this->declare_parameter("startup_ref_ori_threshold_deg", 45.0);

        dt_ = time_step_;

        // Initialize subscribers
        r_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            r_goal_pose_topic_, 10,
            std::bind(&AIWorkerWholebodyController::rightGoalPoseCallback, this, std::placeholders::_1));

        l_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            l_goal_pose_topic_, 10,
            std::bind(&AIWorkerWholebodyController::leftGoalPoseCallback, this, std::placeholders::_1));    
            
        r_elbow_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            r_elbow_pose_topic_, 10,
            std::bind(&AIWorkerWholebodyController::rightElbowPoseCallback, this, std::placeholders::_1));
    
        l_elbow_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            l_elbow_pose_topic_, 10,
            std::bind(&AIWorkerWholebodyController::leftElbowPoseCallback, this, std::placeholders::_1));  

        base_goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            base_goal_pose_topic_, 10,
            std::bind(&AIWorkerWholebodyController::baseGoalPoseCallback, this, std::placeholders::_1));
        
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic_, 10,
            std::bind(&AIWorkerWholebodyController::jointStateCallback, this, std::placeholders::_1));

        ref_divergence_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/reference_diverged", 10,
            std::bind(&AIWorkerWholebodyController::referenceDivergenceCallback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, rclcpp::SystemDefaultsQoS(),
            std::bind(&AIWorkerWholebodyController::odomCallback, this, std::placeholders::_1));

        reactivate_srv_ = this->create_service<std_srvs::srv::Trigger>(
            reactivate_service_,
            std::bind(
                &AIWorkerWholebodyController::reactivateServiceCallback,
                this,
                std::placeholders::_1,
                std::placeholders::_2));
        if (!reactivate_srv_) {
            RCLCPP_FATAL(this->get_logger(),
                "Failed to create reactivate service server '%s'. Controller will not start.",
                reactivate_service_.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Reactivate service ready: %s", reactivate_service_.c_str());

        // Initialize publishers
        reference_divergence_pub_ = this->create_publisher<std_msgs::msg::Bool>("/reference_diverged", 10);
        controller_error_pub_ = this->create_publisher<std_msgs::msg::String>("~/controller_error", 10);

        lift_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            lift_topic_, 10);

        arm_r_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            right_traj_topic_, 10);

        arm_l_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            left_traj_topic_, 10);

        r_gripper_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            r_gripper_pose_topic_, 10);

        l_gripper_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            l_gripper_pose_topic_, 10);    

        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

        // Initialize motion controller
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
            RCLCPP_INFO(this->get_logger(), "Initializing QP controller...");
            qp_controller_ = std::make_shared<motion_controller::controllers::WholebodyQPIK>(kinematics_solver_, dt_);
            qp_controller_->setControllerParams(slack_penalty_, cbf_alpha_, collision_buffer_, collision_safe_distance_);
            qp_controller_->enableBaseActuationLagComp(base_actuation_lag_comp_enabled_);
            qp_controller_->setBaseActuationTimeConstant(base_actuation_tau_seconds_);

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
            std::bind(&AIWorkerWholebodyController::controlLoopCallback, this));
        
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
        RCLCPP_WARN(
            this->get_logger(),
            "Control loop is ready. Call '%s' to start control.",
            reactivate_service_.c_str());
    }

    AIWorkerWholebodyController::~AIWorkerWholebodyController()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down AI Worker Controller");
    }

    void AIWorkerWholebodyController::initializeJointConfig()
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
        lift_joint_.clear();
        lift_joint_index_ = -1;
        base_x_joint_index_ = -1;
        base_y_joint_index_ = -1;
        base_pivot_joint_index_ = -1;
        
        // Parse joint names
        for (const auto& joint_name : joint_names) {
            if (joint_name.find("arm_l_joint") != std::string::npos) {
                left_arm_joints_.push_back(joint_name);
            } else if (joint_name.find("arm_r_joint") != std::string::npos) {
                right_arm_joints_.push_back(joint_name);
            } 
            else if (joint_name.find("lift_joint") != std::string::npos) {
                lift_joint_ = joint_name;
            }
        }
        
        // Sort to ensure correct ordering
        std::sort(left_arm_joints_.begin(), left_arm_joints_.end());
        std::sort(right_arm_joints_.begin(), right_arm_joints_.end());

        // Treat lift as a passive joint for the controller
        if (!lift_joint_.empty()) {
            auto lift_it = model_joint_index_map_.find(lift_joint_);
            if (lift_it != model_joint_index_map_.end()) {
                lift_joint_index_ = lift_it->second;
                const bool locked = kinematics_solver_->setJointVelocityBoundsByIndex(lift_joint_index_, -lift_vel_bound_, lift_vel_bound_);
                if (!locked) {
                    RCLCPP_WARN(this->get_logger(), "Failed to set lift joint velocity bounds");
                    lift_joint_index_ = -1;
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "Lift joint '%s' not found in model index map", lift_joint_.c_str());
            }
        }

        // Base joint indices (wholebody URDF)
        auto it_x = model_joint_index_map_.find("x_joint");
        auto it_y = model_joint_index_map_.find("y_joint");
        auto it_p = model_joint_index_map_.find("pivot_joint");
        if (it_x != model_joint_index_map_.end()) base_x_joint_index_ = it_x->second;
        if (it_y != model_joint_index_map_.end()) base_y_joint_index_ = it_y->second;
        if (it_p != model_joint_index_map_.end()) base_pivot_joint_index_ = it_p->second;

        if (base_x_joint_index_ < 0 || base_y_joint_index_ < 0 || base_pivot_joint_index_ < 0) {
            RCLCPP_WARN(this->get_logger(),
                "Base joints not found in model (x_joint=%d, y_joint=%d, pivot_joint=%d). Base control disabled.",
                base_x_joint_index_, base_y_joint_index_, base_pivot_joint_index_);
        }
        
        // Log joint names
        std::string left_str, right_str;
        for (const auto& j : left_arm_joints_) left_str += j + " ";
        for (const auto& j : right_arm_joints_) right_str += j + " ";
        RCLCPP_DEBUG(this->get_logger(), "Left arm joints: %s", left_str.c_str());
        RCLCPP_DEBUG(this->get_logger(), "Right arm joints: %s", right_str.c_str());
        RCLCPP_DEBUG(this->get_logger(), "Lift joint: %s", lift_joint_.c_str());
    }

    double AIWorkerWholebodyController::yawFromQuaternion(const geometry_msgs::msg::Quaternion& q)
    {
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    void AIWorkerWholebodyController::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        odom_stamp_ = msg->header.stamp;
        odom_x_ = msg->pose.pose.position.x;
        odom_y_ = msg->pose.pose.position.y;
        odom_yaw_ = yawFromQuaternion(msg->pose.pose.orientation);

        // Assumption: twist is expressed in base_link (child frame)
        odom_vx_body_ = msg->twist.twist.linear.x;
        odom_vy_body_ = msg->twist.twist.linear.y;
        odom_wz_ = msg->twist.twist.angular.z;

        odom_received_ = true;
    }

    void AIWorkerWholebodyController::publishBaseCmdVel(const Eigen::VectorXd& optimal_velocities)
    {
        if (!cmd_vel_pub_ || !odom_received_) {
            return;
        }
        if (base_x_joint_index_ < 0 || base_y_joint_index_ < 0 || base_pivot_joint_index_ < 0) {
            return;
        }
        if (base_x_joint_index_ >= static_cast<int>(optimal_velocities.size()) ||
            base_y_joint_index_ >= static_cast<int>(optimal_velocities.size()) ||
            base_pivot_joint_index_ >= static_cast<int>(optimal_velocities.size()))
        {
            return;
        }

        const double xdot_world = optimal_velocities[base_x_joint_index_];
        const double ydot_world = optimal_velocities[base_y_joint_index_];
        const double wz = optimal_velocities[base_pivot_joint_index_];

        geometry_msgs::msg::Twist cmd_target;
        if (cmd_vel_in_base_frame_) {
            // v_body = R(-yaw) * v_world
            const double c = std::cos(odom_yaw_);
            const double s = std::sin(odom_yaw_);
            const double vx = c * xdot_world + s * ydot_world;
            const double vy = -s * xdot_world + c * ydot_world;
            cmd_target.linear.x = vx;
            cmd_target.linear.y = vy;
        } else {
            cmd_target.linear.x = xdot_world;
            cmd_target.linear.y = ydot_world;
        }
        cmd_target.angular.z = wz;



        const rclcpp::Time now = this->now();
        double dt_cmd = dt_;
        if (last_cmd_vel_initialized_) {
            const double dt_meas = (now - last_cmd_vel_stamp_).seconds();
            if (dt_meas > 1e-6 && dt_meas < 0.5) {
                dt_cmd = dt_meas;
            }
        }

        geometry_msgs::msg::Twist cmd_out = cmd_target;
        if (last_cmd_vel_initialized_) {
            auto slew = [dt_cmd](double prev, double target, double accel_max) -> double {
                if (accel_max <= 0.0) return target;
                const double max_delta = accel_max * dt_cmd;
                const double delta = std::clamp(target - prev, -max_delta, max_delta);
                return prev + delta;
            };

            cmd_out.linear.x = slew(last_cmd_vel_.linear.x, cmd_out.linear.x, cmd_vel_max_accel_vx_);
            cmd_out.linear.y = slew(last_cmd_vel_.linear.y, cmd_out.linear.y, cmd_vel_max_accel_vy_);
            cmd_out.angular.z = slew(last_cmd_vel_.angular.z, cmd_out.angular.z, cmd_vel_max_accel_wz_);

            if (cmd_vel_filter_tau_ > 0.0) {
                const double alpha = dt_cmd / (cmd_vel_filter_tau_ + dt_cmd);
                cmd_out.linear.x = last_cmd_vel_.linear.x + alpha * (cmd_out.linear.x - last_cmd_vel_.linear.x);
                cmd_out.linear.y = last_cmd_vel_.linear.y + alpha * (cmd_out.linear.y - last_cmd_vel_.linear.y);
                cmd_out.angular.z = last_cmd_vel_.angular.z + alpha * (cmd_out.angular.z - last_cmd_vel_.angular.z);
            }
        }

        cmd_vel_pub_->publish(cmd_out);
        last_cmd_vel_ = cmd_out;
        last_cmd_vel_stamp_ = now;
        last_cmd_vel_initialized_ = true;
    }

    void AIWorkerWholebodyController::publishZeroCmdVel()
    {
        if (!cmd_vel_pub_) {
            return;
        }
        geometry_msgs::msg::Twist cmd_out;
        cmd_out.linear.x = 0.0;
        cmd_out.linear.y = 0.0;
        cmd_out.linear.z = 0.0;
        cmd_out.angular.x = 0.0;
        cmd_out.angular.y = 0.0;
        cmd_out.angular.z = 0.0;

        cmd_vel_pub_->publish(cmd_out);
        last_cmd_vel_ = cmd_out;
        last_cmd_vel_stamp_ = this->now();
        last_cmd_vel_initialized_ = true;
    }

    void AIWorkerWholebodyController::rightGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        r_goal_pose_ = computePoseMat(*msg);
        r_goal_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Right goal pose received: [%.3f, %.3f, %.3f]", 
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void AIWorkerWholebodyController::leftGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        l_goal_pose_ = computePoseMat(*msg);
        l_goal_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Left goal pose received: [%.3f, %.3f, %.3f]", 
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void AIWorkerWholebodyController::rightElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        r_elbow_pose_ = computePoseMat(*msg);
        r_elbow_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Right elbow pose received: [%.3f, %.3f, %.3f]", 
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void AIWorkerWholebodyController::leftElbowPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        l_elbow_pose_ = computePoseMat(*msg);
        l_elbow_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Left elbow pose received: [%.3f, %.3f, %.3f]", 
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void AIWorkerWholebodyController::baseGoalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        base_goal_pose_ = computePoseMat(*msg);
        base_goal_pose_received_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Base goal pose received: [%.3f, %.3f, %.3f]",
            msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void AIWorkerWholebodyController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
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

    void AIWorkerWholebodyController::referenceDivergenceCallback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (!msg->data) {
            return;
        }
        // Ignore reference divergence if controller is activating
        if (activate_pending_) {
            return;
        }
        if (!reference_diverged_) {
            RCLCPP_ERROR(this->get_logger(), "Reference divergence detected");
        }
        reference_diverged_ = true;
        publishZeroCmdVel();
        if (qp_controller_) {
            qp_controller_->resetBaseAccelerationConstraintState();
        }
    }

    void AIWorkerWholebodyController::reactivateServiceCallback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request;
        RCLCPP_WARN(this->get_logger(), "Start requested (reactivate). Waiting for reference alignment...");
        start_requested_ = true;
        control_enabled_ = false;      // will become true after startup check passes
        activate_pending_ = false;     // set true when we actually enable
        reference_diverged_ = true;    // keep inhibited until activation delay finishes
        response->success = true;
        response->message =
            "Start requested. Controller will start when goal-current pose error is within threshold.";
    }

    void AIWorkerWholebodyController::extractJointStates(const sensor_msgs::msg::JointState::SharedPtr& msg)
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

        // Override base state using /odom (x_joint, y_joint, pivot_joint)
        if (odom_received_ &&
            base_x_joint_index_ >= 0 && base_y_joint_index_ >= 0 && base_pivot_joint_index_ >= 0)
        {
            q_[base_x_joint_index_] = odom_x_;
            q_[base_y_joint_index_] = odom_y_;
            q_[base_pivot_joint_index_] = odom_yaw_;

            // Convert body twist to world-frame x/y joint velocities
            const double c = std::cos(odom_yaw_);
            const double s = std::sin(odom_yaw_);
            const double vx_world = c * odom_vx_body_ - s * odom_vy_body_;
            const double vy_world = s * odom_vx_body_ + c * odom_vy_body_;
            qdot_[base_x_joint_index_] = vx_world;
            qdot_[base_y_joint_index_] = vy_world;
            qdot_[base_pivot_joint_index_] = odom_wz_;
        }
    }

    void AIWorkerWholebodyController::controlLoopCallback()
    {
        static int loop_count = 0;
        static int debug_count = 0;
        
        loop_count++;
        
        if (!joint_state_received_) {
            if (debug_count++ % 100 == 0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Control loop waiting for joint states...");
            }
            publishZeroCmdVel();
            if (qp_controller_) {
                qp_controller_->resetBaseAccelerationConstraintState();
            }
            return;
        }

        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Control loop waiting for odom (%s) to drive base joints...", odom_topic_.c_str());
            publishZeroCmdVel();
            if (qp_controller_) {
                qp_controller_->resetBaseAccelerationConstraintState();
            }
            return;
        }

        // Always compute & publish current gripper poses from measured state.
        // This allows external nodes to validate reference-vs-current before control starts.
        Affine3d r_gripper_pose_meas = Affine3d::Identity();
        Affine3d l_gripper_pose_meas = Affine3d::Identity();
        try {
            kinematics_solver_->updateState(q_, qdot_);
            r_gripper_pose_meas = kinematics_solver_->getPose(r_gripper_name_);
            l_gripper_pose_meas = kinematics_solver_->getPose(l_gripper_name_);
            publishGripperPose(r_gripper_pose_meas, l_gripper_pose_meas);
        } catch (const std::exception &e) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "Failed to compute/publish measured gripper pose: %s", e.what());
        }

        // Startup arming: wait for reactivate AND small goal-current pose error for both arms.
        if (!start_requested_) {
            if (debug_count++ % 100 == 0) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    "Control loop waiting for reactivate service '%s'...",
                    reactivate_service_.c_str());
            }
            publishZeroCmdVel();
            if (qp_controller_) {
                qp_controller_->resetBaseAccelerationConstraintState();
            }
            return;
        }

        // Only perform the goal-vs-current mismatch check during startup (before enabling control).
        if (!control_enabled_) {
            if (!r_goal_pose_received_ || !l_goal_pose_received_) {
                if (debug_count++ % 100 == 0) {
                    RCLCPP_WARN_THROTTLE(
                        this->get_logger(), *this->get_clock(), 2000,
                        "Waiting for goal poses: right=%s left=%s",
                        r_goal_pose_received_ ? "OK" : "MISSING",
                        l_goal_pose_received_ ? "OK" : "MISSING");
                }
                publishZeroCmdVel();
                if (qp_controller_) {
                    qp_controller_->resetBaseAccelerationConstraintState();
                }
                return;
            }

            auto pose_error = [](const Affine3d &cur, const Affine3d &goal, double &pos_m, double &ori_deg) {
                pos_m = (goal.translation() - cur.translation()).norm();
                const Eigen::Quaterniond q_cur(cur.linear());
                const Eigen::Quaterniond q_goal(goal.linear());
                const double dot = std::abs(q_cur.dot(q_goal));
                const double clamped = std::min(1.0, std::max(-1.0, dot));
                const double angle_rad = 2.0 * std::acos(clamped);
                ori_deg = angle_rad * 180.0 / M_PI;
            };

            double r_pos_err = 0.0, r_ori_err = 0.0;
            double l_pos_err = 0.0, l_ori_err = 0.0;
            pose_error(r_gripper_pose_meas, r_goal_pose_, r_pos_err, r_ori_err);
            pose_error(l_gripper_pose_meas, l_goal_pose_, l_pos_err, l_ori_err);

            const bool r_ok =
                (r_pos_err <= startup_ref_pos_threshold_) &&
                (r_ori_err <= startup_ref_ori_threshold_deg_);
            const bool l_ok =
                (l_pos_err <= startup_ref_pos_threshold_) &&
                (l_ori_err <= startup_ref_ori_threshold_deg_);

            if (!(r_ok && l_ok)) {
                reference_diverged_ = true;

                if (reference_divergence_pub_) {
                    std_msgs::msg::Bool msg;
                    msg.data = true;
                    reference_divergence_pub_->publish(msg);
                }
                if (controller_error_pub_) {
                    std_msgs::msg::String err;
                    err.data =
                        "Startup reference mismatch: "
                        "R(pos=" + std::to_string(r_pos_err) + "m, ori=" + std::to_string(r_ori_err) + "deg) "
                        "L(pos=" + std::to_string(l_pos_err) + "m, ori=" + std::to_string(l_ori_err) + "deg) "
                        "thresholds(pos=" + std::to_string(startup_ref_pos_threshold_) + "m, ori=" +
                        std::to_string(startup_ref_ori_threshold_deg_) + "deg)";
                    controller_error_pub_->publish(err);
                }

                RCLCPP_ERROR_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    "Startup mismatch. Waiting. R: pos=%.3f m ori=%.1f deg, L: pos=%.3f m ori=%.1f deg (thr pos=%.3f, ori=%.1f)",
                    r_pos_err, r_ori_err, l_pos_err, l_ori_err, startup_ref_pos_threshold_, startup_ref_ori_threshold_deg_);
                publishZeroCmdVel();
                if (qp_controller_) {
                    qp_controller_->resetBaseAccelerationConstraintState();
                }
                return;
            }

            // Arm controller once the mismatch is small enough
            control_enabled_ = true;
            activate_start_ = this->get_clock()->now();
            activate_pending_ = true;
            reference_diverged_ = true;  // will be cleared after activation delay
            RCLCPP_WARN(
                this->get_logger(),
                "Startup check passed. Activating controller.");
        }

        if (activate_pending_) {
            const auto elapsed = this->get_clock()->now() - activate_start_;
            if (elapsed.seconds() >= 3.0) {
                reference_diverged_ = false;
                activate_pending_ = false;
                RCLCPP_WARN(this->get_logger(), "Controller activated.");
            }
        }

        if (reference_diverged_) {
            publishZeroCmdVel();
            if (qp_controller_) {
                qp_controller_->resetBaseAccelerationConstraintState();
            }
            return;
        }

        debug_count = 0;

        try {
            // kinematics_solver_->updateState(q_, qdot_);
            // Use previously commanded joint goals as feedback state
            VectorXd q_feedback =
                (q_desired_.size() == q_.size()) ? q_desired_ : q_;

            // If lift is commanded by another node, use measured lift state for internal model consistency.
            if (lift_joint_index_ >= 0 &&
                lift_joint_index_ < static_cast<int>(q_feedback.size()) &&
                lift_joint_index_ < static_cast<int>(q_.size()))
            {
                q_feedback[lift_joint_index_] = q_[lift_joint_index_];
            }

            // Always use odom-provided base state as feedback when base is enabled.
            if (odom_received_) {
                if (base_x_joint_index_ >= 0 && base_x_joint_index_ < static_cast<int>(q_feedback.size())) {
                    q_feedback[base_x_joint_index_] = q_[base_x_joint_index_];
                }
                if (base_y_joint_index_ >= 0 && base_y_joint_index_ < static_cast<int>(q_feedback.size())) {
                    q_feedback[base_y_joint_index_] = q_[base_y_joint_index_];
                }
                if (base_pivot_joint_index_ >= 0 && base_pivot_joint_index_ < static_cast<int>(q_feedback.size())) {
                    q_feedback[base_pivot_joint_index_] = q_[base_pivot_joint_index_];
                }
            }

            // Control loop is executing - update kinematics solver with feedback state
            kinematics_solver_->updateState(q_feedback, qdot_);

            // Get current and goal end-effector poses
            right_gripper_pose_ = kinematics_solver_->getPose(r_gripper_name_);
            left_gripper_pose_ = kinematics_solver_->getPose(l_gripper_name_);
            Affine3d right_elbow_pose = kinematics_solver_->getPose(r_elbow_name_);
            Affine3d left_elbow_pose = kinematics_solver_->getPose(l_elbow_name_);
            Affine3d base_link_pose = kinematics_solver_->getPose(base_link_name_);

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
            if (!base_goal_pose_received_) {
                base_goal_pose_ = base_link_pose;
            }

            // Publish current end-effector pose
            publishGripperPose(right_gripper_pose_, left_gripper_pose_);

            // Slow-start ramp after activation delay
            const auto activate_elapsed = this->get_clock()->now() - activate_start_;
            double slow_start_scale = 1.0;
            double slow_start_duration = 8.0;
            if (slow_start_duration > 0.0) {
                const double ramp_time = activate_elapsed.seconds() - 3.0;
                if (ramp_time < slow_start_duration) {
                    slow_start_scale = std::clamp(ramp_time / slow_start_duration, 0.0, 1.0);
                }
            }

            // Compute desired velocity (scaled during slow-start)
            motion_controller::common::Vector6d right_desired_vel =
                computeDesiredVelocity(right_gripper_pose_, r_goal_pose_) * slow_start_scale;
            motion_controller::common::Vector6d left_desired_vel =
                computeDesiredVelocity(left_gripper_pose_, l_goal_pose_) * slow_start_scale;
            motion_controller::common::Vector6d right_elbow_desired_vel = motion_controller::common::Vector6d::Zero();
            motion_controller::common::Vector6d left_elbow_desired_vel = motion_controller::common::Vector6d::Zero();
            right_elbow_desired_vel.head(3) =
                kp_position_ * (r_elbow_pose_.translation() - right_elbow_pose.translation()) * slow_start_scale;
            left_elbow_desired_vel.head(3) =
                kp_position_ * (l_elbow_pose_.translation() - left_elbow_pose.translation()) * slow_start_scale;

            // Body tracking (arm_base_link): planar base
            // Track x/y position and yaw (ignore roll/pitch).
            motion_controller::common::Vector6d base_desired_vel = motion_controller::common::Vector6d::Zero();
            base_desired_vel.head(3) =
            kp_position_ * 0.1 * (base_goal_pose_.translation() - base_link_pose.translation()) * slow_start_scale;

            auto yaw_from_R = [](const Eigen::Matrix3d& R) -> double {
                return std::atan2(R(1, 0), R(0, 0));
            };
            const double yaw_cur = yaw_from_R(base_link_pose.linear());
            const double yaw_goal = yaw_from_R(base_goal_pose_.linear());
            const double yaw_err = std::atan2(std::sin(yaw_goal - yaw_cur), std::cos(yaw_goal - yaw_cur));
            base_desired_vel[5] = kp_orientation_ * 0.1 * yaw_err * slow_start_scale;
            
            std::map<std::string, motion_controller::common::Vector6d> desired_task_velocities;
            desired_task_velocities[r_gripper_name_] = right_desired_vel;
            desired_task_velocities[l_gripper_name_] = left_desired_vel;
            desired_task_velocities[r_elbow_name_] = right_elbow_desired_vel;
            desired_task_velocities[l_elbow_name_] = left_elbow_desired_vel;
            desired_task_velocities[base_link_name_] = base_desired_vel;

            // Set weights for QP solver
            std::map<std::string, motion_controller::common::Vector6d> weights;
            motion_controller::common::Vector6d weight_right = motion_controller::common::Vector6d::Ones();
            motion_controller::common::Vector6d weight_left = motion_controller::common::Vector6d::Ones();
            weight_right.head(3).setConstant(weight_position_);
            weight_right.tail(3).setConstant(weight_orientation_);
            weight_left.head(3).setConstant(weight_position_);
            weight_left.tail(3).setConstant(weight_orientation_);
            weights[r_gripper_name_] = weight_right;
            weights[l_gripper_name_] = weight_left;
            motion_controller::common::Vector6d weight_right_elbow = motion_controller::common::Vector6d::Zero();
            motion_controller::common::Vector6d weight_left_elbow = motion_controller::common::Vector6d::Zero();
            weight_right_elbow.head(3).setConstant(weight_elbow_position_);
            weight_left_elbow.head(3).setConstant(weight_elbow_position_);
            weights[r_elbow_name_] = weight_right_elbow;
            weights[l_elbow_name_] = weight_left_elbow;

            motion_controller::common::Vector6d weight_base = motion_controller::common::Vector6d::Zero();
            weight_base.head(3).setConstant(weight_base_position_);
            // planar base: only yaw weight
            weight_base[5] = weight_base_orientation_;
            weights[base_link_name_] = weight_base;
            
            VectorXd damping = VectorXd::Ones(kinematics_solver_->getDof()) * weight_damping_;
            if (base_x_joint_index_ >= 0 && base_x_joint_index_ < static_cast<int>(damping.size())) {
                damping[base_x_joint_index_] = weight_damping_base_;
            }
            if (base_y_joint_index_ >= 0 && base_y_joint_index_ < static_cast<int>(damping.size())) {
                damping[base_y_joint_index_] = weight_damping_base_;
            }
            if (base_pivot_joint_index_ >= 0 && base_pivot_joint_index_ < static_cast<int>(damping.size())) {
                damping[base_pivot_joint_index_] = weight_damping_base_;
            }

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

            // Publish base command (cmd_vel) from optimal base generalized velocities
            publishBaseCmdVel(optimal_velocities);

            // Compute command from current state
            // q_desired_ = q_ + optimal_velocities * dt_;
            q_desired_ = q_feedback + optimal_velocities * dt_;

            // Publish trajectory commands
            publishTrajectory(q_desired_);

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in control loop: %s", e.what());
        }
    }

    Affine3d AIWorkerWholebodyController::computePoseMat(const geometry_msgs::msg::PoseStamped& pose) const
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

    motion_controller::common::Vector6d AIWorkerWholebodyController::computeDesiredVelocity(
        const Affine3d& current_pose,
        const Affine3d& goal_pose) const
    {
        // Compute position error
        Vector3d pos_error = goal_pose.translation() - current_pose.translation();
        
        // Compute orientation error
        Matrix3d rotation_error = goal_pose.linear() * current_pose.linear().transpose();
        Eigen::AngleAxisd angle_axis_error(rotation_error);
        Vector3d angle_axis = angle_axis_error.axis() * angle_axis_error.angle();

        motion_controller::common::Vector6d desired_vel = motion_controller::common::Vector6d::Zero();
        desired_vel.head(3) = kp_position_ * pos_error;
        desired_vel.tail(3) = kp_orientation_ * angle_axis;
        
        return desired_vel;
    }


    void AIWorkerWholebodyController::publishTrajectory(const VectorXd& q_desired)
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

            // Publish lift trajectory
            if (lift_joint_index_ >= 0 && !lift_joint_.empty()) {
                if (lift_joint_index_ < static_cast<int>(q_desired.size())) {
                    auto traj_lift = createLiftTrajectoryMsg(lift_joint_, q_desired[lift_joint_index_]);
                    if (lift_vel_bound_ != 0.0) {
                        lift_pub_->publish(traj_lift);
                    }
                } else {
                    RCLCPP_WARN(this->get_logger(), "Lift index out of range, skipping lift publish");
                }
            }

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error publishing trajectory: %s", e.what());
        }
    }

    trajectory_msgs::msg::JointTrajectory AIWorkerWholebodyController::createTrajectoryMsgWithGripper(
        const std::vector<std::string>& arm_joint_names,
        const VectorXd& positions,
        const std::vector<int>& arm_indices,
        const std::string& gripper_joint_name) const
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.header.frame_id = "";
        
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

    trajectory_msgs::msg::JointTrajectory AIWorkerWholebodyController::createLiftTrajectoryMsg(
        std::string lift_joint_name,
        const double position) const
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.header.frame_id = "";
        
        traj_msg.joint_names = {lift_joint_name};

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_);
        point.positions = {position};
        traj_msg.points.push_back(point);
        return traj_msg;
    }

    void AIWorkerWholebodyController::publishGripperPose(const Affine3d& r_gripper_pose, const Affine3d& l_gripper_pose)
    {
        if (r_gripper_pose_pub_) {
            geometry_msgs::msg::PoseStamped r_gripper_pose_msg;
            r_gripper_pose_msg.header.stamp = this->now();
            r_gripper_pose_msg.header.frame_id = "base_link";
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
            l_gripper_pose_msg.header.frame_id = "base_link";
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
    auto node = std::make_shared<motion_controller_ros::AIWorkerWholebodyController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
