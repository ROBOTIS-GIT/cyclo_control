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

#include <chrono>
#include <string>
#include <rclcpp/rclcpp.hpp>

namespace cyclo_teleoperation
{
// A local process lock prevents launch races. The graph check also fails closed if
// another host/container advertises an owner for the same remapped source topic.
class RuntimeOwnership
{
public:
  RuntimeOwnership(rclcpp::Node & node, const std::string & source_topic);
  ~RuntimeOwnership();
  RuntimeOwnership(const RuntimeOwnership &) = delete;
  RuntimeOwnership & operator=(const RuntimeOwnership &) = delete;
  void check();
  bool ready() const {return ready_;}

private:
  rclcpp::Node & node_;
  std::string topic_;
  int lock_fd_ = -1;
  std::chrono::steady_clock::time_point start_;
  bool ready_ = false;
};
}  // namespace cyclo_teleoperation
