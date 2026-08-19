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

#include <Eigen/Dense>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "cyclo_teleoperation/core/types.hpp"

namespace cyclo_teleoperation::robots::ai_worker
{
class PoseSequenceManager
{
public:
  bool configure(
    rclcpp::Node & node,
    const ModeConfiguration & configuration,
    const std::vector<int64_t> & available_modes,
    const std::vector<int64_t> & available_presets);

  bool hasInitialPose(uint16_t mode) const;
  uint8_t initialPoseArms(uint16_t mode) const;
  bool startInitialPose(uint16_t mode, const ModeContext & context);
  bool updateInitialPose(const ModeContext & context, ModeOutput & output);
  bool initialPoseMoving() const;
  uint8_t activeInitialPoseArms() const;
  void cancelInitialPose();

  bool startFinalInitialPose(
    uint16_t mode, uint8_t arms, const ModeContext & context);
  bool updateFinalInitialPoses(const ModeContext & context, ModeOutput & output);
  void cancelFinalInitialPoses(uint8_t arms);
  uint8_t activeFinalInitialPoseArms() const;
  uint8_t movingFinalInitialPoseArms() const;
  uint8_t leftFinalInitialPoseState() const;
  uint8_t rightFinalInitialPoseState() const;

  bool hasExitPose(uint16_t mode) const;
  bool startExitPose(uint16_t mode, const ModeContext & context);
  bool updateExitPose(const ModeContext & context, ModeOutput & output);
  bool exitPoseMoving() const;
  uint8_t activeExitPoseArms() const;
  void cancelExitPose();

  bool hasPreset(uint16_t preset, uint8_t arm) const;
  bool startPreset(
    uint8_t arms,
    uint16_t left_preset_id,
    uint16_t right_preset_id,
    const ModeContext & context);
  bool updatePresets(const ModeContext & context, ModeOutput & output);
  void cancelPresets(uint8_t arms);
  uint8_t activePresetArms() const;
  uint8_t movingPresetArms() const;
  uint8_t leftPresetState() const;
  uint8_t rightPresetState() const;

  const std::string & errorMessage() const {return error_message_;}

private:
  enum class Purpose : uint8_t
  {
    kNone,
    kInitialPose,
    kFinalInitialPose,
    kExitPose,
    kPreset,
  };

  struct Step
  {
    std::string name;
    Eigen::VectorXd target;
  };

  struct Sequence
  {
    std::vector<Step> steps;
    double duration = 3.0;
    double completion_tolerance = 0.03;
    double timeout = 10.0;
  };

  struct BimanualSequence
  {
    Sequence left;
    Sequence right;
  };

  struct Runner
  {
    const Sequence * sequence = nullptr;
    Eigen::VectorXd start;
    size_t step_index = 0;
    double start_time = 0.0;
    Purpose purpose = Purpose::kNone;
    bool active = false;
    bool moving = false;
    uint8_t state = 0;
  };

  BimanualSequence loadSequence(
    rclcpp::Node & node,
    const std::string & prefix,
    const std::vector<std::string> & step_names) const;
  bool startRunner(
    Runner & runner,
    const Sequence & sequence,
    Purpose purpose,
    const std::vector<int> & indices,
    const ModeContext & context,
    bool final_step_only = false);
  bool updateRunner(
    Runner & runner,
    Purpose purpose,
    const std::vector<int> & indices,
    const ModeContext & context,
    ModeOutput & output);
  static double quintic(double value);
  static void cancelRunner(Runner & runner, Purpose purpose);
  uint8_t activeArms(Purpose purpose, bool moving_only) const;

  std::vector<int> left_indices_;
  std::vector<int> right_indices_;
  std::unordered_map<uint16_t, BimanualSequence> initial_poses_;
  std::unordered_map<uint16_t, BimanualSequence> exit_poses_;
  std::unordered_map<uint16_t, BimanualSequence> presets_;
  Runner left_runner_;
  Runner right_runner_;
  double kp_ = 30.0;
  double tracking_weight_ = 10.0;
  std::string error_message_;
};
}  // namespace cyclo_teleoperation::robots::ai_worker
