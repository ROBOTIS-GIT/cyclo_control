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

#include <gtest/gtest.h>

#include "cyclo_teleoperation/core/soft_hold.hpp"

namespace cyclo_teleoperation
{
TEST(ControlGroupTest, BuildsIndependentMasks)
{
  EXPECT_EQ(controlGroupBit(0), ControlGroupMask{1});
  EXPECT_EQ(controlGroupBit(3), ControlGroupMask{8});
  EXPECT_TRUE(containsControlGroup(controlGroupBit(3), 3));
  EXPECT_FALSE(containsControlGroup(controlGroupBit(3), 0));
}

TEST(SoftHoldTest, HoldsOnlyGroupsNotControlledByTheMode)
{
  std::vector<ControlGroupConfiguration> groups{
    ControlGroupConfiguration{0, "arm", {0, 2}, "", "", {}},
    ControlGroupConfiguration{1, "auxiliary", {1}, "", "", {}}};
  Eigen::VectorXd command = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd hold_target(3);
  hold_target << 1.0, 2.0, 3.0;
  ModeOutput output;
  output.desired_joint_velocity = Eigen::VectorXd::Zero(3);
  output.joint_tracking_weight = Eigen::VectorXd::Zero(3);

  applySoftHold(
    output, command, hold_target, groups, controlGroupBit(0),
    1.0, 10.0, 50.0);

  EXPECT_DOUBLE_EQ(output.desired_joint_velocity[0], 0.0);
  EXPECT_DOUBLE_EQ(output.desired_joint_velocity[1], 2.0);
  EXPECT_DOUBLE_EQ(output.desired_joint_velocity[2], 0.0);
  EXPECT_DOUBLE_EQ(output.joint_tracking_weight[0], 0.0);
  EXPECT_DOUBLE_EQ(output.joint_tracking_weight[1], 50.0);
  EXPECT_DOUBLE_EQ(output.joint_tracking_weight[2], 0.0);
}
}  // namespace cyclo_teleoperation
