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
#include "cyclo_teleoperation/core/model_action_session.hpp"

namespace cyclo_teleoperation
{
enum class ControlSource {kNone, kTeleoperation, kModelAction};
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
  ControlSource activeSource() const {return active_source_;}
  void startDirect();
  void activateSource(ControlSource source, bool direct);
  void fail();
  void poll() {model_session_.poll();}
  bool modelPending() const {return model_session_.pending();}
  bool modelGranted() const {return model_granted_;}
  bool modelAvailable() const {return model_session_.available();}
  bool managedModel() const {return model_acknowledged_;}
  bool stopRequired(bool external_publishers) const;
  void observeDirect(bool observed) {direct_seen_ = direct_seen_ || observed;}
  void stopModel(bool required, ModelActionSession::Completion completion);
  void startModel(bool direct, ModelActionSession::Completion completion);
  void suspendModelOutput();
  bool jointOutputAllowed() const;
  void allowJointOutput(bool allowed) {joint_output_allowed_ = allowed;}
  bool direct() const {return active_source_ == ControlSource::kModelAction && direct_;}
  void waitForExit(ControlSource target) {exit_pending_ = true; exit_target_ = target;}
  bool exitPending() const {return exit_pending_;}
  ControlSource finishExit() {exit_pending_ = false; return exit_target_;}

private:
  void publish(const std::string & value);
  Switch change_;
  Switch validate_;
  std::unique_ptr<RuntimeOwnership> ownership_;
  ModelActionSession model_session_;
  ControlSource active_source_ = ControlSource::kNone;
  ControlSource exit_target_ = ControlSource::kNone;
  bool direct_ = true, model_granted_ = false, model_acknowledged_ = false;
  bool direct_seen_ = false, joint_output_allowed_ = false, exit_pending_ = false;
  ControlSource source_ = ControlSource::kNone;
  ControlSource pending_ = ControlSource::kNone;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr toggle_service_;
  rclcpp::TimerBase::SharedPtr heartbeat_;
};
}  // namespace cyclo_teleoperation
