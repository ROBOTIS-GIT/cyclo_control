#include "motion_controller_core/controllers/omy/omy_movel_controller.hpp"

namespace motion_controller
{
namespace controllers
{
    OMYMoveLController::OMYMoveLController(
        std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data,
        const std::string& controlled_link,
        double dt)
        : OMXMoveLController(std::move(robot_data), controlled_link, dt)
    {
    }
}  // namespace controllers
}  // namespace motion_controller
