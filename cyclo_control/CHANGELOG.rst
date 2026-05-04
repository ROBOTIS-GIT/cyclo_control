^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cyclo_control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.2.0 (2026-05-04)
------------------
* Added arm retargeting feature for AI Worker vr controller
* Modified AI Worker, OMY, OMX movej controller to work with leader device
* Contributors: Yeonguk Kim

0.1.4 (2026-04-20)
------------------
* Update acknowledgements to include dyros_robot_controller
* Contributors: Yeonguk Kim

0.1.3 (2026-04-09)
------------------
* Updated the `ai_worker_controller` launch file to run the hand retargeting node alongside the motion controller
* Updated the reference checker node topic QoS to best effort
* Contributors: Yeonguk Kim

0.1.2 (2026-04-06)
------------------
* Removed gripper joint from the trajectory messages
* Renamed meta package to cyclo_control
* Changed reactivate service to topic
* Contributors: Yeonguk Kim

0.1.1 (2026-03-30)
------------------
* Added acknowledgements for the dependencies used in the project
* Contributors: Hyunwoo Nam

0.1.0 (2026-03-25)
------------------
* Refactored package names and added a vendor package for dependency management
* Contributors: Yeonguk Kim

0.0.2 (2026-03-18)
------------------
* Added movel_controller and movel_controller_node for AI Worker
* Renamed the AI Worker joint_space_controller and joint_space_controller_node to movej_controller and movej_controller_node
* Contributors: Yeonguk Kim

0.0.1 (2026-03-12)
------------------
* Added motion controller library for control of articulated robots
* Added robot models for motion controllers
* Added ros wrapper for motion controller library
* Added retargeting teleoperation controller for HX5-D20
* Contributors: Yeonguk Kim, Hyunwoo Nam
