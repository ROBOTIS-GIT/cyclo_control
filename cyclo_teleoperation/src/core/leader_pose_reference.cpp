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
#include "cyclo_teleoperation/core/leader_pose_reference.hpp"
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace cyclo_teleoperation
{
struct LeaderPoseReference::Model
{
  pinocchio::Model kinematics;
  pinocchio::Data data;
  pinocchio::FrameIndex base_frame;
  std::unordered_map<std::string, pinocchio::JointIndex> joints;
};

void LeaderPoseReference::configure(const std::string & urdf, const std::string & base_frame)
{
  auto model = std::make_shared<Model>();
  pinocchio::urdf::buildModel(urdf, model->kinematics);
  model->data = pinocchio::Data(model->kinematics);
  // Frame 1 is the URDF root body (frame 0 is Pinocchio's universe).
  const auto reference_frame = base_frame.empty() ? model->kinematics.frames.at(1).name :
    base_frame;
  if (!model->kinematics.existFrame(reference_frame)) {
    throw std::runtime_error("Leader base frame does not exist: " + reference_frame);
  }
  model->base_frame = model->kinematics.getFrameId(reference_frame);
  for (pinocchio::JointIndex id = 1; id < model->kinematics.joints.size(); ++id) {
    const auto & joint = model->kinematics.joints[id];
    if (joint.nv() != 1 || (joint.nq() != 1 && joint.nq() != 2)) {
      throw std::runtime_error("Leader reference requires scalar revolute/prismatic joints");
    }
    model->joints.emplace(model->kinematics.names[id], id);
  }
  model_ = std::move(model);
}

Eigen::Affine3d LeaderPoseReference::computePose(
  const std::vector<std::string> & input_names, const std::vector<double> & values,
  const std::vector<std::string> & model_joints, const std::string & eef)
{
  if (!model_ || input_names.size() != values.size()) {
    throw std::runtime_error("Leader reference model or joint sample is invalid");
  }
  if (!model_joints.empty() && model_joints.size() > values.size()) {
    throw std::runtime_error("Leader model joint mapping exceeds the input joint count");
  }
  auto & model = model_->kinematics;
  if (!model.existFrame(eef)) {
    throw std::runtime_error("Leader EEF frame does not exist: " + eef);
  }
  const auto end_frame = model.getFrameId(eef);
  Eigen::VectorXd position = pinocchio::neutral(model);
  std::unordered_set<pinocchio::JointIndex> mapped;
  const size_t count = model_joints.empty() ? input_names.size() : model_joints.size();
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {throw std::runtime_error("Non-finite leader joint reference");}
    const auto found = model_->joints.find(model_joints.empty() ? input_names[i] : model_joints[i]);
    if (found == model_->joints.end()) {
      if (model_joints.empty()) {continue;}  // Optional gripper is not an arm FK input.
      throw std::runtime_error("Leader model joint mapping contains an unknown joint");
    }
    if (!mapped.insert(found->second).second) {
      throw std::runtime_error("Leader model joint mapping contains duplicate joints");
    }
    const auto & joint = model.joints[found->second];
    if (joint.nq() == 1) {
      position[joint.idx_q()] = values[i];
    } else {
      // Pinocchio represents continuous revolute joints as (cos(q), sin(q)).
      position[joint.idx_q()] = std::cos(values[i]);
      position[joint.idx_q() + 1] = std::sin(values[i]);
    }
  }
  // Only joints on the relative base-to-EEF path are required. An optional
  // gripper or the opposite arm must not make a valid arm reference incomplete.
  std::unordered_set<pinocchio::JointIndex> required;
  for (const auto frame : {model_->base_frame, end_frame}) {
    for (auto id = model.frames[frame].parentJoint; id != 0; id = model.parents[id]) {
      if (required.erase(id) == 0) {required.insert(id);}
    }
  }
  for (const auto id : required) {
    if (mapped.count(id) == 0) {
      throw std::runtime_error("Leader reference is missing joint: " + model.names[id]);
    }
  }
  pinocchio::framesForwardKinematics(model, model_->data, position);
  return Eigen::Affine3d((model_->data.oMf[model_->base_frame].inverse() *
           model_->data.oMf[end_frame]).toHomogeneousMatrix());
}

Eigen::Affine3d LeaderPoseReference::absoluteGoal(
  const Eigen::Affine3d & leader_start, const Eigen::Affine3d & leader_now,
  const Eigen::Affine3d & follower_start_in_arm, const Eigen::Affine3d & base_from_arm)
{
  return absoluteGoal(leader_start, leader_now, follower_start_in_arm, base_from_arm, 1.0);
}

Eigen::Affine3d LeaderPoseReference::absoluteGoal(
  const Eigen::Affine3d & leader_start, const Eigen::Affine3d & leader_now,
  const Eigen::Affine3d & follower_start_in_arm, const Eigen::Affine3d & base_from_arm,
  const double translation_gain)
{
  if (!std::isfinite(translation_gain) || translation_gain <= 0.0) {
    throw std::invalid_argument("Cartesian translation gain must be finite and positive");
  }
  Eigen::Affine3d goal = follower_start_in_arm;
  const Eigen::Matrix3d arm_from_base = base_from_arm.linear().transpose();
  goal.translation() += arm_from_base *
    (translation_gain * (leader_now.translation() - leader_start.translation()));
  goal.linear() = arm_from_base * leader_now.linear() * leader_start.linear().transpose() *
    base_from_arm.linear() * follower_start_in_arm.linear();
  return base_from_arm * goal;
}
}  // namespace cyclo_teleoperation
