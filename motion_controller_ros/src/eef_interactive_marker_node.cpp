#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <interactive_markers/interactive_marker_server.hpp>

namespace motion_controller_ros
{
    class EefInteractiveMarkerNode : public rclcpp::Node
    {
    public:
        EefInteractiveMarkerNode()
            : Node("eef_interactive_marker_node"),
              server_(std::make_shared<interactive_markers::InteractiveMarkerServer>(
                  "eef_goal_markers",
                  this->get_node_base_interface(),
                  this->get_node_clock_interface(),
                  this->get_node_logging_interface(),
                  this->get_node_topics_interface(),
                  this->get_node_services_interface(),
                  rclcpp::QoS(100),
                  rclcpp::QoS(10))),
              right_initialized_(false),
              left_initialized_(false),
              right_locked_(false),
              left_locked_(false),
              right_pose_received_(false),
              left_pose_received_(false),
              pending_update_(false)
        {
            base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
            marker_scale_ = this->declare_parameter<double>("marker_scale", 0.2);

            r_goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/r_goal_pose", 10);
            l_goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/l_goal_pose", 10);

            r_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
                "/r_gripper_pose", 10,
                std::bind(&EefInteractiveMarkerNode::rightGripperPoseCallback, this, std::placeholders::_1));
            l_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
                "/l_gripper_pose", 10,
                std::bind(&EefInteractiveMarkerNode::leftGripperPoseCallback, this, std::placeholders::_1));

            update_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(50),
                std::bind(&EefInteractiveMarkerNode::publishMarkerUpdates, this));
        }

    private:
        void create6DofMarker(const std::string& name,
                              const std::string& description,
                              const geometry_msgs::msg::Pose& pose,
                              const std::string& frame_id,
                              double r, double g, double b)
        {
            visualization_msgs::msg::InteractiveMarker marker;
            marker.header.frame_id = frame_id;
            marker.name = name;
            marker.description = description;
            marker.scale = marker_scale_;
            marker.pose = pose;

            visualization_msgs::msg::Marker box_marker;
            box_marker.type = visualization_msgs::msg::Marker::CUBE;
            box_marker.scale.x = marker.scale * 0.2;
            box_marker.scale.y = marker.scale * 0.2;
            box_marker.scale.z = marker.scale * 0.2;
            box_marker.color.r = r;
            box_marker.color.g = g;
            box_marker.color.b = b;
            box_marker.color.a = 0.8;

            visualization_msgs::msg::InteractiveMarkerControl box_control;
            box_control.always_visible = true;
            box_control.markers.push_back(box_marker);
            marker.controls.push_back(box_control);

            addAxisControls(marker);

            server_->insert(marker, std::bind(&EefInteractiveMarkerNode::markerFeedback, this, std::placeholders::_1));
        }

        void addAxisControls(visualization_msgs::msg::InteractiveMarker& marker)
        {
            visualization_msgs::msg::InteractiveMarkerControl control;

            control.orientation.w = 1.0;
            control.orientation.x = 1.0;
            control.orientation.y = 0.0;
            control.orientation.z = 0.0;
            control.name = "rotate_x";
            control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS;
            marker.controls.push_back(control);
            control.name = "move_x";
            control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
            marker.controls.push_back(control);

            control.orientation.w = 1.0;
            control.orientation.x = 0.0;
            control.orientation.y = 1.0;
            control.orientation.z = 0.0;
            control.name = "rotate_y";
            control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS;
            marker.controls.push_back(control);
            control.name = "move_y";
            control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
            marker.controls.push_back(control);

            control.orientation.w = 1.0;
            control.orientation.x = 0.0;
            control.orientation.y = 0.0;
            control.orientation.z = 1.0;
            control.name = "rotate_z";
            control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS;
            marker.controls.push_back(control);
            control.name = "move_z";
            control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
            marker.controls.push_back(control);
        }

        void markerFeedback(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr& feedback)
        {
            if (feedback->event_type != visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE &&
                feedback->event_type != visualization_msgs::msg::InteractiveMarkerFeedback::MOUSE_UP) {
                return;
            }

            geometry_msgs::msg::PoseStamped goal_msg;
            goal_msg.header.stamp = this->get_clock()->now();
            goal_msg.header.frame_id = feedback->header.frame_id.empty() ? base_frame_ : feedback->header.frame_id;
            goal_msg.pose = feedback->pose;

            if (feedback->marker_name == "right_gripper_goal") {
                right_locked_ = true;
                r_goal_pub_->publish(goal_msg);
            } else if (feedback->marker_name == "left_gripper_goal") {
                left_locked_ = true;
                l_goal_pub_->publish(goal_msg);
            }
        }

        void rightGripperPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
        {
            if (right_locked_) {
                return;
            }

            const std::string frame_id =
                msg->header.frame_id.empty() ? base_frame_ : msg->header.frame_id;

            if (!right_initialized_) {
                create6DofMarker("right_gripper_goal", "Right gripper goal", msg->pose, frame_id, 1.0, 0.2, 0.2);
                server_->applyChanges();
                right_initialized_ = true;
            }

            last_right_pose_ = *msg;
            right_pose_received_ = true;
            pending_update_ = true;
        }

        void leftGripperPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
        {
            if (left_locked_) {
                return;
            }

            const std::string frame_id =
                msg->header.frame_id.empty() ? base_frame_ : msg->header.frame_id;

            if (!left_initialized_) {
                create6DofMarker("left_gripper_goal", "Left gripper goal", msg->pose, frame_id, 0.2, 0.2, 1.0);
                server_->applyChanges();
                left_initialized_ = true;
            }

            last_left_pose_ = *msg;
            left_pose_received_ = true;
            pending_update_ = true;
        }

        void publishMarkerUpdates()
        {
            if (!pending_update_) {
                return;
            }

            bool changed = false;
            if (!right_locked_ && right_initialized_ && right_pose_received_) {
                server_->setPose("right_gripper_goal", last_right_pose_.pose, last_right_pose_.header);
                publishGoal(last_right_pose_.pose,
                            last_right_pose_.header.frame_id.empty() ? base_frame_ : last_right_pose_.header.frame_id,
                            r_goal_pub_);
                changed = true;
            }

            if (!left_locked_ && left_initialized_ && left_pose_received_) {
                server_->setPose("left_gripper_goal", last_left_pose_.pose, last_left_pose_.header);
                publishGoal(last_left_pose_.pose,
                            last_left_pose_.header.frame_id.empty() ? base_frame_ : last_left_pose_.header.frame_id,
                            l_goal_pub_);
                changed = true;
            }

            if (changed) {
                server_->applyChanges();
            }

            pending_update_ = false;
        }

        void publishGoal(const geometry_msgs::msg::Pose& pose,
                         const std::string& frame_id,
                         const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr& pub)
        {
            geometry_msgs::msg::PoseStamped goal_msg;
            goal_msg.header.stamp = this->get_clock()->now();
            goal_msg.header.frame_id = frame_id;
            goal_msg.pose = pose;
            pub->publish(goal_msg);
        }

        std::shared_ptr<interactive_markers::InteractiveMarkerServer> server_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_goal_pub_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_goal_pub_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr r_pose_sub_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr l_pose_sub_;
        rclcpp::TimerBase::SharedPtr update_timer_;

        std::string base_frame_;
        double marker_scale_;
        bool right_initialized_;
        bool left_initialized_;
        bool right_locked_;
        bool left_locked_;
        bool right_pose_received_;
        bool left_pose_received_;
        bool pending_update_;
        geometry_msgs::msg::PoseStamped last_right_pose_;
        geometry_msgs::msg::PoseStamped last_left_pose_;
    };
}  // namespace motion_controller_ros

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<motion_controller_ros::EefInteractiveMarkerNode>());
    rclcpp::shutdown();
    return 0;
}
