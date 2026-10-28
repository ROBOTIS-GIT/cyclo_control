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
#include <memory>
#include <string>
#include <vector>

namespace cyclo_teleoperation
{
// Reference generation only. No QP, ROS node, follower model or robot-specific names.
// Owned by the common leader broadcaster and called outside its real-time update.
class LeaderPoseReference
{
public:
  void configure(const std::string & urdf, const std::string & base_frame);
  Eigen::Affine3d computePose(
    const std::vector<std::string> & input_names, const std::vector<double> & values,
    const std::vector<std::string> & model_joints, const std::string & eef);
  static Eigen::Affine3d absoluteGoal(
    const Eigen::Affine3d & leader_start, const Eigen::Affine3d & leader_now,
    const Eigen::Affine3d & follower_start_in_arm, const Eigen::Affine3d & base_from_arm);
  // Uniform translation gain about the anchors; orientation and FK are unchanged.
  static Eigen::Affine3d absoluteGoal(
    const Eigen::Affine3d & leader_start, const Eigen::Affine3d & leader_now,
    const Eigen::Affine3d & follower_start_in_arm, const Eigen::Affine3d & base_from_arm,
    double translation_gain);

private:
  struct Model;
  std::shared_ptr<Model> model_;
};
}  // namespace cyclo_teleoperation
