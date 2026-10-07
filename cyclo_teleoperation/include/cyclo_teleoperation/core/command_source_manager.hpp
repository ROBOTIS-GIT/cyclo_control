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
#pragma once

#include <functional>
#include <string>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace cyclo_teleoperation
{
enum class ControlSource {kNone, kTeleoperation, kAction};
class RuntimeOwnership;

// One owner, serialized with the control timer by the node's default callback group.
// Source selection changes no ROS controller lifecycle and requires no custom interface.
class CommandSourceManager
{
public:
  using Switch = std::function<void(ControlSource)>;
  CommandSourceManager(rclcpp::Node & node, Switch change, Switch validate);
  ~CommandSourceManager();
  void select(ControlSource source);
  void complete(ControlSource source);
  bool transitioning() const {return pending_ != ControlSource::kNone;}
  bool ready() const;
  ControlSource selected() const {return source_;}

private:
  void publish(const std::string & value);
  Switch change_;
  Switch validate_;
  std::unique_ptr<RuntimeOwnership> ownership_;
  ControlSource source_ = ControlSource::kNone;
  ControlSource pending_ = ControlSource::kNone;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr toggle_service_;
  rclcpp::TimerBase::SharedPtr heartbeat_;
};
}  // namespace cyclo_teleoperation
