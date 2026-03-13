// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Yeonguk Kim

#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace motion_controller_ros
{
    class InteractiveMarkerNode : public rclcpp::Node
    {
    public:
        InteractiveMarkerNode()
            : Node("interactive_marker_node"),
              initialized_(false),
              dragging_(false)
        {
            base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
            controlled_link_ =
                this->declare_parameter<std::string>("controlled_link", "end_effector_link");
            goal_topic_ =
                this->declare_parameter<std::string>("goal_topic", "/goal_pose");
            server_name_ =
                this->declare_parameter<std::string>("server_name", "interactive_marker");
            marker_name_ =
                this->declare_parameter<std::string>("marker_name", "goal_marker");
            marker_description_ =
                this->declare_parameter<std::string>("marker_description", "Goal marker");
            marker_scale_ = this->declare_parameter<double>("marker_scale", 0.2);
            publish_while_dragging_ =
                this->declare_parameter<bool>("publish_while_dragging", true);
            marker_color_r_ = this->declare_parameter<double>("marker_color_r", 0.2);
            marker_color_g_ = this->declare_parameter<double>("marker_color_g", 0.8);
            marker_color_b_ = this->declare_parameter<double>("marker_color_b", 0.2);

            goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(goal_topic_, 10);

            tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

            server_ = std::make_shared<interactive_markers::InteractiveMarkerServer>(
                server_name_,
                this->get_node_base_interface(),
                this->get_node_clock_interface(),
                this->get_node_logging_interface(),
                this->get_node_topics_interface(),
                this->get_node_services_interface(),
                rclcpp::QoS(100),
                rclcpp::QoS(10));

            update_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(50),
                std::bind(&InteractiveMarkerNode::initializeMarkerIfReady, this));

            RCLCPP_INFO(this->get_logger(), "Interactive marker node started");
            RCLCPP_INFO(this->get_logger(), "  - Base frame: %s", base_frame_.c_str());
            RCLCPP_INFO(this->get_logger(), "  - Controlled link: %s", controlled_link_.c_str());
            RCLCPP_INFO(this->get_logger(), "  - Goal topic: %s", goal_topic_.c_str());
            RCLCPP_INFO(this->get_logger(), "  - Marker name: %s", marker_name_.c_str());
            RCLCPP_INFO(this->get_logger(), "  - Marker server: %s", server_name_.c_str());
        }

    private:
        void create6DofMarker(const geometry_msgs::msg::Pose& pose, const std::string& frame_id)
        {
            visualization_msgs::msg::InteractiveMarker marker;
            marker.header.frame_id = frame_id;
            marker.name = marker_name_;
            marker.description = marker_description_;
            marker.scale = marker_scale_;
            marker.pose = pose;

            visualization_msgs::msg::Marker box_marker;
            box_marker.type = visualization_msgs::msg::Marker::CUBE;
            box_marker.scale.x = marker.scale * 0.2;
            box_marker.scale.y = marker.scale * 0.2;
            box_marker.scale.z = marker.scale * 0.2;
            box_marker.color.r = marker_color_r_;
            box_marker.color.g = marker_color_g_;
            box_marker.color.b = marker_color_b_;
            box_marker.color.a = 0.8;

            visualization_msgs::msg::InteractiveMarkerControl box_control;
            box_control.always_visible = true;
            box_control.markers.push_back(box_marker);
            marker.controls.push_back(box_control);

            addAxisControls(marker);

            server_->insert(
                marker,
                std::bind(&InteractiveMarkerNode::markerFeedback, this, std::placeholders::_1));
        }

        void addAxisControls(visualization_msgs::msg::InteractiveMarker& marker)
        {
            visualization_msgs::msg::InteractiveMarkerControl control;

            control.orientation.w = 1.0;
            control.orientation.x = 1.0;
            control.orientation.y = 0.0;
            control.orientation.z = 0.0;
            control.name = "rotate_x";
            control.interaction_mode =
                visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS;
            marker.controls.push_back(control);
            control.name = "move_x";
            control.interaction_mode =
                visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
            marker.controls.push_back(control);

            control.orientation.w = 1.0;
            control.orientation.x = 0.0;
            control.orientation.y = 1.0;
            control.orientation.z = 0.0;
            control.name = "rotate_y";
            control.interaction_mode =
                visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS;
            marker.controls.push_back(control);
            control.name = "move_y";
            control.interaction_mode =
                visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
            marker.controls.push_back(control);

            control.orientation.w = 1.0;
            control.orientation.x = 0.0;
            control.orientation.y = 0.0;
            control.orientation.z = 1.0;
            control.name = "rotate_z";
            control.interaction_mode =
                visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS;
            marker.controls.push_back(control);
            control.name = "move_z";
            control.interaction_mode =
                visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
            marker.controls.push_back(control);
        }

        void markerFeedback(
            const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr& feedback)
        {
            if (feedback->marker_name != marker_name_) {
                return;
            }

            if (feedback->event_type ==
                visualization_msgs::msg::InteractiveMarkerFeedback::MOUSE_DOWN) {
                dragging_ = true;
                return;
            }

            if (feedback->event_type ==
                visualization_msgs::msg::InteractiveMarkerFeedback::MOUSE_UP) {
                dragging_ = false;
                publishGoal(
                    feedback->pose,
                    feedback->header.frame_id.empty() ? base_frame_ : feedback->header.frame_id);
                return;
            }

            if (feedback->event_type ==
                    visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE &&
                dragging_ && publish_while_dragging_) {
                publishGoal(
                    feedback->pose,
                    feedback->header.frame_id.empty() ? base_frame_ : feedback->header.frame_id);
            }
        }

        void initializeMarkerIfReady()
        {
            if (initialized_) {
                return;
            }

            if (!lookupPose(controlled_link_, initial_pose_)) {
                return;
            }

            create6DofMarker(initial_pose_.pose, initial_pose_.header.frame_id);
            server_->applyChanges();
            publishGoal(initial_pose_.pose, initial_pose_.header.frame_id);
            initialized_ = true;
            update_timer_->cancel();

            RCLCPP_INFO(this->get_logger(), "Interactive marker initialized from link transform.");
        }

        bool lookupPose(const std::string& child_frame, geometry_msgs::msg::PoseStamped& pose_out)
        {
            try {
                const auto tf =
                    tf_buffer_->lookupTransform(base_frame_, child_frame, tf2::TimePointZero);
                pose_out.header = tf.header;
                pose_out.pose.position.x = tf.transform.translation.x;
                pose_out.pose.position.y = tf.transform.translation.y;
                pose_out.pose.position.z = tf.transform.translation.z;
                pose_out.pose.orientation = tf.transform.rotation;
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }

        void publishGoal(const geometry_msgs::msg::Pose& pose, const std::string& frame_id)
        {
            geometry_msgs::msg::PoseStamped goal_msg;
            goal_msg.header.stamp = this->get_clock()->now();
            goal_msg.header.frame_id = frame_id;
            goal_msg.pose = pose;
            goal_pub_->publish(goal_msg);
        }

        std::shared_ptr<interactive_markers::InteractiveMarkerServer> server_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
        rclcpp::TimerBase::SharedPtr update_timer_;
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

        std::string base_frame_;
        std::string controlled_link_;
        std::string goal_topic_;
        std::string server_name_;
        std::string marker_name_;
        std::string marker_description_;
        double marker_scale_;
        bool publish_while_dragging_;
        double marker_color_r_;
        double marker_color_g_;
        double marker_color_b_;

        bool initialized_;
        bool dragging_;
        geometry_msgs::msg::PoseStamped initial_pose_;
    };
}  // namespace motion_controller_ros

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<motion_controller_ros::InteractiveMarkerNode>());
    rclcpp::shutdown();
    return 0;
}
