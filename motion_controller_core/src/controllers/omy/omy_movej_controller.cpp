#include "motion_controller_core/controllers/omy/omy_movej_controller.hpp"

namespace motion_controller
{
namespace controllers
{
    OMYMoveJController::OMYMoveJController(
        std::shared_ptr<motion_controller::kinematics::KinematicsSolver> robot_data,
        double dt)
        : OMXMoveJController(std::move(robot_data), dt)
    {
    }
}  // namespace controllers
}  // namespace motion_controller
