// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace cyclo_teleoperation
{
// Only negotiates publisher ownership. Joint commands never pass through this class.
// A successful stop reply means publication stopped AND queued model actions were discarded.
class ModelActionSession
{
public:
  using Completion = std::function<void(bool, const std::string &)>;
  explicit ModelActionSession(rclcpp::Node & node);
  void request(bool enabled, bool required, Completion completion);
  void poll();
  bool pending() const {return static_cast<bool>(completion_);}
  bool available() const {return client_->service_is_ready();}

private:
  void finish(bool success, const std::string & message);
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr client_;
  Completion completion_;
  int64_t request_id_ = 0;
  uint64_t generation_ = 0;
  std::chrono::steady_clock::time_point deadline_;
};
}  // namespace cyclo_teleoperation
