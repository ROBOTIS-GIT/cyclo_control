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

#include <Eigen/Geometry>

#include <std_msgs/msg/float64_multi_array.hpp>

namespace cyclo_motion_controller_ros::utils
{

/**
 * @brief Convert an end-effector pose to X-VLA's position + rotation-6D layout.
 *
 * X-VLA flattens the first two columns of the rotation matrix in row-major order:
 * [x, y, z, r00, r01, r10, r11, r20, r21].
 */
inline std_msgs::msg::Float64MultiArray makeEePose6dMessage(const Eigen::Affine3d & pose)
{
  const Eigen::Vector3d & position = pose.translation();
  const Eigen::Matrix3d & rotation = pose.linear();

  std_msgs::msg::Float64MultiArray msg;
  msg.data = {
    position.x(), position.y(), position.z(),
    rotation(0, 0), rotation(0, 1),
    rotation(1, 0), rotation(1, 1),
    rotation(2, 0), rotation(2, 1)};
  return msg;
}

}  // namespace cyclo_motion_controller_ros::utils
