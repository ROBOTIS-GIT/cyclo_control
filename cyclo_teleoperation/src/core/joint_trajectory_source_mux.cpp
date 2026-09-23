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

#include "cyclo_teleoperation/core/joint_trajectory_source_mux.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <std_msgs/msg/bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace cyclo_teleoperation
{
class JointTrajectorySourceMux : public rclcpp::Node
{
public:
  JointTrajectorySourceMux()
  : Node("joint_trajectory_source_mux")
  {
    const auto group_names = declare_parameter<std::vector<std::string>>(
      "group_names", std::vector<std::string>{});
    const auto teleoperation_topics = declare_parameter<std::vector<std::string>>(
      "teleoperation_input_topics", std::vector<std::string>{});
    const auto model_topics = declare_parameter<std::vector<std::string>>(
      "model_input_topics", std::vector<std::string>{});
    const auto output_topics = declare_parameter<std::vector<std::string>>(
      "output_topics", std::vector<std::string>{});
    const auto source_state_topic = declare_parameter(
      "command_source_state_topic", "/control/leader_action_enabled");

    const size_t group_count = group_names.size();
    if (
      group_count == 0 || teleoperation_topics.size() != group_count ||
      model_topics.size() != group_count || output_topics.size() != group_count)
    {
      throw std::runtime_error(
              "group_names and every source/output topic list must have the same non-zero size");
    }
    for (size_t i = 0; i < group_count; ++i) {
      if (
        group_names[i].empty() || teleoperation_topics[i].empty() ||
        model_topics[i].empty() || output_topics[i].empty())
      {
        throw std::runtime_error("Trajectory source mux group names and topics must not be empty");
      }
    }

    group_names_ = group_names;
    teleoperation_buffer_.resize(group_count);
    model_buffer_.resize(group_count);
    selected_received_.assign(group_count, false);

    const auto trajectory_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    for (size_t i = 0; i < group_count; ++i) {
      output_publishers_.push_back(
        create_publisher<trajectory_msgs::msg::JointTrajectory>(
          output_topics[i], trajectory_qos));
      teleoperation_subscriptions_.push_back(
        create_subscription<trajectory_msgs::msg::JointTrajectory>(
          teleoperation_topics[i], trajectory_qos,
          [this, i](const trajectory_msgs::msg::JointTrajectory::SharedPtr message) {
            receive(true, i, *message);
          }));
      model_subscriptions_.push_back(
        create_subscription<trajectory_msgs::msg::JointTrajectory>(
          model_topics[i], trajectory_qos,
          [this, i](const trajectory_msgs::msg::JointTrajectory::SharedPtr message) {
            receive(false, i, *message);
          }));
    }

    const auto source_state_qos =
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    source_state_subscription_ = create_subscription<std_msgs::msg::Bool>(
      source_state_topic, source_state_qos,
      [this](const std_msgs::msg::Bool::SharedPtr message) {
        selectTeleoperation(message->data);
      });

    RCLCPP_INFO(
      get_logger(), "Trajectory source mux started with model control selected");
  }

private:
  using Trajectory = trajectory_msgs::msg::JointTrajectory;

  void selectTeleoperation(const bool selected)
  {
    if (selected == teleoperation_selected_) {
      return;
    }
    teleoperation_selected_ = selected;
    gate_open_ = false;
    std::fill(selected_received_.begin(), selected_received_.end(), false);
    auto & buffer = selectedBuffer();
    for (auto & message : buffer) {
      message.reset();
    }
    RCLCPP_INFO(
      get_logger(), "Follower command source switched to %s; waiting for fresh group commands",
      selected ? "teleoperation" : "model");
  }

  std::vector<std::optional<Trajectory>> & selectedBuffer()
  {
    return teleoperation_selected_ ? teleoperation_buffer_ : model_buffer_;
  }

  void receive(const bool teleoperation, const size_t group, const Trajectory & message)
  {
    if (teleoperation != teleoperation_selected_ || group >= selected_received_.size()) {
      return;
    }

    auto & buffer = selectedBuffer();
    buffer[group] = message;
    selected_received_[group] = true;
    if (!gate_open_) {
      if (!std::all_of(
          selected_received_.begin(), selected_received_.end(),
          [](const bool received) {return received;}))
      {
        return;
      }
      gate_open_ = true;
      for (size_t i = 0; i < buffer.size(); ++i) {
        output_publishers_[i]->publish(buffer[i].value());
      }
      RCLCPP_INFO(
        get_logger(), "%s command source is active for all configured groups",
        teleoperation_selected_ ? "Teleoperation" : "Model");
      return;
    }
    output_publishers_[group]->publish(message);
  }

  std::vector<std::string> group_names_;
  std::vector<std::optional<Trajectory>> teleoperation_buffer_;
  std::vector<std::optional<Trajectory>> model_buffer_;
  std::vector<bool> selected_received_;
  bool teleoperation_selected_ = false;
  bool gate_open_ = false;

  std::vector<rclcpp::Publisher<Trajectory>::SharedPtr> output_publishers_;
  std::vector<rclcpp::Subscription<Trajectory>::SharedPtr>
  teleoperation_subscriptions_;
  std::vector<rclcpp::Subscription<Trajectory>::SharedPtr> model_subscriptions_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr source_state_subscription_;
};

std::shared_ptr<rclcpp::Node> makeJointTrajectorySourceMux()
{
  return std::make_shared<JointTrajectorySourceMux>();
}
}  // namespace cyclo_teleoperation
