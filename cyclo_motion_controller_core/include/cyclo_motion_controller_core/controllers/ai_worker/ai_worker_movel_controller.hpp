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
//
// This file is derived from `dyros_robot_controller` project:
// https://github.com/JunHeonYoon/dyros_robot_controller
//
// Original work Copyright (c) 2025 JunHeonYoon, licensed under the
// Apache License 2.0. Modifications in this file are Copyright 2026
// ROBOTIS CO., LTD.
//
// Author: Yeonguk Kim

#pragma once

#include <memory>

#include "controllers/ai_worker/vr_controller.hpp"

namespace cyclo_motion_controller
{
namespace controllers
{
class AIWorkerMoveLController : public VRController
{
public:
  AIWorkerMoveLController(
    std::shared_ptr<cyclo_motion_controller::kinematics::KinematicsSolver> robot_data,
    const double dt);
};
}  // namespace controllers
}  // namespace cyclo_motion_controller
