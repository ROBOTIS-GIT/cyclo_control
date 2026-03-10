#pragma once

#include <memory>

#include "motion_controller_core/controllers/omx/omx_movej_controller.hpp"

namespace motion_controller
{
namespace controllers
{
    class OMYMoveJController : public OMXMoveJController
    {
    public:
        OMYMoveJController(
            std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data,
            double dt);
    };
}  // namespace controllers
}  // namespace motion_controller
