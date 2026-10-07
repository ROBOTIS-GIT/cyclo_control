// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#include "cyclo_teleoperation/core/model_action_session.hpp"
#include <stdexcept>
#include <utility>

namespace cyclo_teleoperation
{
ModelActionSession::ModelActionSession(rclcpp::Node & node)
{
  const auto name = node.declare_parameter(
    "model_action_enable_service", "/model_action/set_enabled");
  client_ = node.create_client<std_srvs::srv::SetBool>(name);
}

void ModelActionSession::request(bool enabled, bool required, Completion completion)
{
  if (pending()) {throw std::runtime_error("A model publisher handoff is already pending");}
  if (!available()) {
    completion(!required, required ?
      "Model publisher must provide /model_action/set_enabled before handing off control" : "");
    return;
  }
  completion_ = std::move(completion);
  deadline_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  const auto generation = ++generation_;
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = enabled;
  const auto pending = client_->async_send_request(request,
      [this, generation](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
        if (generation != generation_ || !completion_) {return;}
        const auto response = future.get();
        finish(response->success, response->message);
    });
  request_id_ = pending.request_id;
}

void ModelActionSession::poll()
{
  if (pending() && std::chrono::steady_clock::now() >= deadline_) {
    client_->remove_pending_request(request_id_);
    ++generation_;
    // Revoke a possibly late enable. Models must also stop on /source != model_action.
    auto stop = std::make_shared<std_srvs::srv::SetBool::Request>();
    stop->data = false;
    const auto pending_stop = client_->async_send_request(stop);
    client_->remove_pending_request(pending_stop.request_id);
    finish(false,
        "Model publisher handoff timed out; external publication cannot be guaranteed stopped");
  }
}

void ModelActionSession::finish(bool success, const std::string & message)
{
  auto completion = std::move(completion_);
  completion_ = {};
  completion(success, message);
}
}  // namespace cyclo_teleoperation
