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

#include <cstdio>
#include <exception>

#include <rclcpp/rclcpp.hpp>

#include "cyclo_teleoperation/core/teleoperation_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(cyclo_teleoperation::makeTeleoperationNode());
  } catch (const std::exception & error) {
    std::fprintf(stderr, "cyclo_teleoperation failed: %s\n", error.what());
  }
  rclcpp::shutdown();
  return 0;
}
