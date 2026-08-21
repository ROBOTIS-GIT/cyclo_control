^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cyclo_motion_controller_ros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.3.2 (2026-08-20)
------------------
* Unified model-aware gripper commands with AI Worker arm trajectories
* Fixed MoveJ time_from_start interpolation for AI Worker, OMX, and OMY controllers
* Preserved immediate gripper commands and zero-duration leader streaming
* Contributors: Yeonguk Kim

0.3.1 (2026-08-19)
------------------
* None

0.3.0 (2026-06-18)
------------------
* Added AI Worker bimanual MoveL and MoveJ controller nodes and launch options, including rigid grasp and virtual object commands
* Contributors: Yeonguk Kim

0.2.0 (2026-05-04)
------------------
* Modified AI Worker, OMY, OMX movej controller to work with leader device
* Contributors: Yeonguk Kim

0.1.4 (2026-04-20)
------------------
* None

0.1.3 (2026-04-09)
------------------
* Updated the `ai_worker_controller` launch file to run the hand retargeting node alongside the motion controller
* Updated the reference checker node topic QoS to best effort
* Contributors: Yeonguk Kim

0.1.2 (2026-04-06)
------------------
* Removed gripper joint from the trajectory messages
* Changed reactivate service to topic
* Contributors: Yeonguk Kim

0.1.1 (2026-03-30)
------------------
* None

0.1.0 (2026-03-25)
------------------
* Namespaced package names to avoid conflicts with other packages
* Contributors: Yeonguk Kim

0.0.2 (2026-03-18)
------------------
* Added a movel_controller_node for AI Worker
* Renamed the AI Worker joint_space_controller_node to movej_controller_node
* Contributors: Yeonguk Kim

0.0.1 (2026-03-12)
------------------
* Added ros wrapper for motion controller library
* Contributors: Yeonguk Kim
