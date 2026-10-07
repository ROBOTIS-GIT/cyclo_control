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
#include "cyclo_teleoperation/core/command_source_manager.hpp"
#include <stdexcept>
#include <utility>
#include <chrono>
#include "cyclo_teleoperation/core/runtime_ownership.hpp"

namespace cyclo_teleoperation
{
CommandSourceManager::CommandSourceManager(rclcpp::Node & node, Switch change, Switch validate)
: change_(std::move(change)), validate_(std::move(validate)),
  ownership_(std::make_unique<RuntimeOwnership>(node, "/source"))
{
  publisher_ = node.create_publisher<std_msgs::msg::String>(
    "/source", rclcpp::QoS(1).reliable().transient_local());
  heartbeat_ = node.create_wall_timer(std::chrono::milliseconds(100), [this]() {
        ownership_->check();
        publish(transitioning() ? "switching" : source_ ==
        ControlSource::kTeleoperation ? "teleop" :
      source_ == ControlSource::kModelAction ? "model_action" : "none");
  });
  set_service_ = node.create_service<std_srvs::srv::SetBool>(
    "/set_source", [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
      try {
        select(request->data ? ControlSource::kTeleoperation : ControlSource::kModelAction);
        response->success = true;
        response->message = "Source request accepted; /source reports transition completion";
      } catch (const std::exception & error) {
        response->message = error.what();
      }
    });
  toggle_service_ = node.create_service<std_srvs::srv::Trigger>(
    "/toggle_source", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
      try {
        select(source_ == ControlSource::kTeleoperation ?
          ControlSource::kModelAction : ControlSource::kTeleoperation);
        response->success = true;
      } catch (const std::exception & error) {
        response->message = error.what();
      }
    });
}

CommandSourceManager::~CommandSourceManager() = default;

bool CommandSourceManager::ready() const {return ownership_->ready();}

void CommandSourceManager::publish(const std::string & value)
{
  std_msgs::msg::String message;
  message.data = ready() ? value : "starting";
  publisher_->publish(message);
}

void CommandSourceManager::select(const ControlSource source)
{
  if (transitioning()) {
    throw std::runtime_error("A source transition is already in progress");
  }
  if (source == source_) {
    return;
  }
  validate_(source);
  source_ = ControlSource::kNone;
  pending_ = source;
  publish("switching");
  try {
    change_(source);
  } catch (...) {
    complete(ControlSource::kNone);
    throw;
  }
}

void CommandSourceManager::complete(const ControlSource source)
{
  source_ = source;
  pending_ = ControlSource::kNone;
  publish(source == ControlSource::kTeleoperation ? "teleop" :
    source == ControlSource::kModelAction ? "model_action" : "none");
}
}  // namespace cyclo_teleoperation
