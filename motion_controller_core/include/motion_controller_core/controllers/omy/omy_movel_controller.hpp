#pragma once

#include <memory>
#include <string>

#include "motion_controller_core/controllers/omx/omx_movel_controller.hpp"

namespace motion_controller
{
namespace controllers
{
    class OMYMoveLController : public OMXMoveLController
    {
    public:
        OMYMoveLController(
            std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data,
            const std::string& controlled_link,
            double dt);
    };
}  // namespace controllers
}  // namespace motion_controller
