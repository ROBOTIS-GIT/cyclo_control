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
#include <cmath>
#include "cyclo_teleoperation/core/runtime_ownership.hpp"

namespace cyclo_teleoperation
{
namespace
{
// Display names only; the /source wire values remain unchanged.
const char * controlSourceLogName(const ControlSource source)
{
  switch (source) {
    case ControlSource::kTeleoperation:
      return "teleop";
    case ControlSource::kModelAction:
      return "external";
    default:
      return "none";
  }
}
}  // namespace

CommandSourceManager::CommandSourceManager(
  rclcpp::Node & node, Switch change, Switch validate, std::function<void()> leader_lost)
: logger_(node.get_logger()), change_(std::move(change)), validate_(std::move(validate)),
  leader_lost_(std::move(leader_lost)),
  ownership_(std::make_unique<RuntimeOwnership>(node, "/source"))
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  const auto topic = node.declare_parameter<std::string>(
    "leader_heartbeat_topic", "/leader/heartbeat", descriptor);
  leader_heartbeat_timeout_ = node.declare_parameter<double>(
    "leader_heartbeat_timeout", 1.0, descriptor);
  if (!std::isfinite(leader_heartbeat_timeout_) || leader_heartbeat_timeout_ <= 0.0) {
    throw std::invalid_argument("leader_heartbeat_timeout must be finite and positive");
  }
  leader_heartbeat_ = node.create_subscription<std_msgs::msg::Empty>(
    topic, rclcpp::QoS(1).best_effort(), [this](std_msgs::msg::Empty::ConstSharedPtr) {
      leader_last_seen_ = std::chrono::steady_clock::now();
      leader_seen_ = true;
    });
  publisher_ = node.create_publisher<std_msgs::msg::String>(
    "/source", rclcpp::QoS(1).reliable().transient_local());
  heartbeat_ = node.create_wall_timer(std::chrono::milliseconds(100), [this]() {
        ownership_->check();
        checkLeader();
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
        response->message = std::string("Source change rejected: ") + error.what();
        RCLCPP_WARN(logger_, "Control source change rejected: %s", error.what());
      }
    });
  toggle_service_ = node.create_service<std_srvs::srv::Trigger>(
    "/toggle_source", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
      try {
        select(source_ == ControlSource::kTeleoperation ?
          ControlSource::kModelAction : ControlSource::kTeleoperation);
        response->success = true;
        response->message = "Source request accepted; /source reports transition completion";
      } catch (const std::exception & error) {
        response->message = std::string("Source change rejected: ") + error.what();
        RCLCPP_WARN(logger_, "Control source change rejected: %s", error.what());
      }
    });
}

CommandSourceManager::~CommandSourceManager() = default;

bool CommandSourceManager::leaderFresh() const
{
  return leader_seen_ && std::chrono::duration<double>(
    std::chrono::steady_clock::now() - leader_last_seen_).count() < leader_heartbeat_timeout_;
}

void CommandSourceManager::checkLeader()
{
  if (leaderFresh() || (active_source_ != ControlSource::kTeleoperation &&
    pending_ != ControlSource::kTeleoperation))
  {
    return;
  }
  RCLCPP_WARN(logger_,
    "Leader heartbeat lost for %.2f s; stopping teleoperation and returning source to external",
    leader_heartbeat_timeout_);
  suspendModelOutput();
  exit_pending_ = false;
  source_ = ControlSource::kNone;
  pending_ = ControlSource::kModelAction;
  publish("switching");
  try {
    leader_lost_();
    complete(ControlSource::kModelAction);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(logger_, "Leader disconnect handoff failed: %s", error.what());
    fail();
  }
}

void CommandSourceManager::startDirect()
{
  // Cold startup does not take command ownership from an existing LG2 publisher.
  pending_ = ControlSource::kModelAction;
  publish("switching");
}

void CommandSourceManager::activateSource(ControlSource source, bool direct)
{
  active_source_ = source;
  direct_ = direct;
  joint_output_allowed_ = source == ControlSource::kTeleoperation;
}

void CommandSourceManager::suspendModelOutput()
{
  model_granted_ = false;
  joint_output_allowed_ = false;
}

void CommandSourceManager::fail()
{
  suspendModelOutput();
  active_source_ = ControlSource::kNone;
  exit_pending_ = false;
  complete(ControlSource::kNone);
}

void CommandSourceManager::startModel(bool direct)
{
  direct_ = direct;
  model_granted_ = true;
  joint_output_allowed_ = !direct;
}

bool CommandSourceManager::jointOutputAllowed() const
{
  return joint_output_allowed_ && active_source_ != ControlSource::kNone && !direct();
}

bool CommandSourceManager::ready() const {return ownership_->ready();}

void CommandSourceManager::publish(const std::string & value)
{
  std_msgs::msg::String message;
  message.data = ready() ? value : "starting";
  publisher_->publish(message);
}

void CommandSourceManager::select(const ControlSource source)
{
  if (source == ControlSource::kTeleoperation && !leaderFresh()) {
    throw std::runtime_error("Leader heartbeat is unavailable; start or reconnect the leader");
  }
  if (transitioning()) {
    throw std::runtime_error("A source transition is already in progress");
  }
  if (source == source_) {
    RCLCPP_INFO(logger_, "Control source unchanged: %s", controlSourceLogName(source));
    return;
  }
  validate_(source);
  RCLCPP_INFO(logger_, "Control source change accepted: %s -> %s",
    controlSourceLogName(source_), controlSourceLogName(source));
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
  RCLCPP_INFO(logger_, "Control source: %s", controlSourceLogName(source));
}
}  // namespace cyclo_teleoperation
