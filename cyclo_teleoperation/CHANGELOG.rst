^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cyclo_teleoperation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.5.1 (2026-10-26)
------------------
* None

0.5.0 (2026-10-19)
------------------
* Unified teleoperation and external action control with source switching and direct absolute-joint command compatibility
* Added shared absolute EEF pose control, measured EEF states, and gripper-only action inputs
* Exported a reusable leader-reference library and moved relative pose reference generation to the leader
* Centralized fixed-duration slow start and configurable joint-position and self-collision constraints
* Separated source, mode, feedback, and input management while preserving pose sequences and feedback recovery
* Added lift-aware Cartesian frames for relative teleoperation and absolute external commands
* Contributors: Yeonguk Kim

0.4.1 (2026-09-21)
------------------
* Hardened leader and follower input validation and latest-sample handling
* Preserved slow-start and pose-sequence safety across follower feedback recovery
* Disabled joint-position constraints for MoveJ teleoperation
* Contributors: Yeonguk Kim

0.4.0 (2026-09-17)
------------------
* Initial release of the cyclo_teleoperation package
* Contributors: Yeonguk Kim
