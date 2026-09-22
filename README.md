# Cyclo Control

This repository provides control packages for the ROBOTIS Physical AI lineup.

## Repository Structure

```text
├── cyclo_motion_controller/
│   ├── CMakeLists.txt
│   └── package.xml
├── cyclo_motion_controller_core/
│   ├── include/cyclo_motion_controller_core/
│   │   ├── common/
│   │   │   └── ...
│   │   ├── controllers/
│   │   │   └── ...
│   │   ├── kinematics/
│   │   │   └── ...
│   │   └── optimization/
│   │       └── ...
│   ├── src/
│   │   ├── controllers/
│   │   │   └── ...
│   │   ├── kinematics/
│   │   │   └── ...
│   │   └── retargeting/
│   │       └── ...
│   ├── CMakeLists.txt
│   └── package.xml
├── cyclo_motion_controller_ros/
│   ├── config/
│   │   └── ...
│   ├── include/cyclo_motion_controller_ros/
│   │   ├── nodes/
│   │   │   └── ...
│   │   └── utils/
│   │       └── ...
│   ├── launch/
│   │   └── ...
│   ├── src/
│   │   ├── nodes/
│   │   │   └── ...
│   │   └── utils/
│   │       └── ...
│   ├── CMakeLists.txt
│   └── package.xml
├── cyclo_motion_controller_ros_py/
│   ├── resource/
│   │   └── ...
│   ├── scripts/
│   │   └── ...
│   ├── package.xml
│   ├── setup.cfg
│   └── setup.py
├── cyclo_motion_controller_models/
│   ├── launch/
│   │   └── ...
│   ├── models/
│   │   └── ...
│   ├── CMakeLists.txt
│   └── package.xml
├── cyclo_teleoperation/
│   ├── include/cyclo_teleoperation/
│   │   ├── controllers/
│   │   ├── core/
│   │   └── robots/
│   ├── src/
│   ├── CMakeLists.txt
│   └── package.xml
└── osqp_eigen_vendor/
    ├── cmake/
    ├── third_party/
    │   └── osqp-eigen/
    ├── CMakeLists.txt
    ├── THIRD_PARTY_NOTICES.md
    └── package.xml
```

## Directory Description

`cyclo_motion_controller/`

- Meta package containing package related to motion control.

`cyclo_motion_controller_core/`

- Core package containing kinematics solver, controllers, and retargeting utilities.
- `include/cyclo_motion_controller_core/common/`: Shared types and utility functions.
- `include/cyclo_motion_controller_core/optimization/`: QP definitions and solver interfaces.
- `src/controllers/`: Controller implementations for AI Worker and OpenManipulator.
- `src/kinematics/`: Kinematics solver implementation.
- `src/retargeting/`: Python retargeting utilities.

`cyclo_motion_controller_ros/`

- ROS 2 package containing controller nodes, launch files, and runtime configs.
- `config/`: YAML configuration files for AI Worker, OMX, and OMY controllers.
- `launch/`: Launch files for running the controller nodes.
- `src/nodes/`: ROS 2 node executables organized by robot family.
- `src/utils/`: Utility nodes such as interactive markers and reference checking.

`cyclo_motion_controller_ros_py/`

- ROS 2 Python package containing retargeting-related scripts and tests.
- `scripts/`: Python entrypoints.

`cyclo_motion_controller_models/`

- Robot model descriptions and RViz resources package.
- `launch/`: Launch files for visualizing robot models.
- `models/`: URDF/SRDF robot models used by the controller.

`cyclo_teleoperation/`

- Plugin-based runtime for leader teleoperation and model-action control.
- Loads robot and control-mode plugins while sharing follower feedback handling, QP constraints,
  soft hold, and safe mode transitions.
- See [`cyclo_teleoperation/README.md`](cyclo_teleoperation/README.md) for configuration details.

`osqp_eigen_vendor/`

- Vendor package that wraps the upstream `osqp-eigen` source tree for this repository.
- `third_party/osqp-eigen/`: Vendored upstream source.

## Install (from source)

### Prerequisites

- **ROS 2 Jazzy** installed
- **vcs** is used to import workspace dependencies, and **rosdep** is used to install system dependencies
- **numpy<2** is required

### Build in a ROS 2 workspace

Clone the repository and import workspace dependencies:

```bash
cd ~/ros2_ws/src
git clone https://github.com/ROBOTIS-GIT/cyclo_control.git
vcs import . < cyclo_control/cyclo_control_ci.repos
```

Install dependencies via `rosdep`, then build:

```bash
cd ~/ros2_ws
sudo apt update
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Run

### Teleoperation and Action Control

The `cyclo_teleoperation` package provides `cyclo_teleoperation_node` for leader-driven control and
`cyclo_action_controller_node` for model inference. Robot-specific repositories provide the robot
model, topic mappings, available control-mode plugins, and launch configuration.

### AI Worker Controllers

Launch AI Worker controllers:

```bash
ros2 launch cyclo_motion_controller_ros ai_worker_controller.launch.py
```

Launch arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `controller_type` | `movel` | Selects the controller mode. Available values are listed below. |
| `config_file` | package default | Controller YAML file. |
| `follower_urdf_path` | package default | Follower robot URDF. |
| `default_srdf_path` | package default | Default follower robot SRDF. |
| `modified_srdf_path` | package default | Follower robot SRDF with gripper-to-gripper collision disabled. |
| `disable_gripper_collisions` | `false` | Uses `modified_srdf_path` instead of `default_srdf_path`. |

`controller_type` values:

| Value | Started nodes |
| --- | --- |
| `movel` | `ai_worker_movel_controller_node` |
| `movej` | `ai_worker_movej_controller_node` |
| `bimanual_movel` | `ai_worker_bimanual_movel_controller_node` |
| `bimanual_movej` | `ai_worker_bimanual_movej_controller_node` |
| `vr` | `vr_controller_node`, `reference_checker_node` |
| `leader` | `leader_controller_node`, `vr_controller_node` |

Interactive marker arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `start_interactive_marker` | `false` | Starts MoveL interactive markers for `movel` and `bimanual_movel`. |
| `base_frame` | `base_link` | Frame used by interactive markers and MoveL goals. |
| `marker_scale` | `0.2` | Interactive marker scale. |
| `right_controlled_link` | `end_effector_r_link` | Follower link tracked by the right marker. |
| `left_controlled_link` | `end_effector_l_link` | Follower link tracked by the left marker. |
| `right_movel_topic` | `/r_goal_move` | MoveL command topic for the right marker. |
| `left_movel_topic` | `/l_goal_move` | MoveL command topic for the left marker. |
| `right_goal_pose_topic` | `/r_goal_pose` | PoseStamped mirror topic for the right marker. |
| `left_goal_pose_topic` | `/l_goal_pose` | PoseStamped mirror topic for the left marker. |

Bimanual MoveL arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `grasp_capture_topic` | `/capture_grasp` | Bool topic used to enable or disable bimanual rigid grasp mode. |
| `virtual_object_movel_topic` | `/virtual_object_goal_move` | MoveL command topic for the bimanual virtual object marker. |
| `virtual_object_pose_topic` | `/virtual_object_goal_pose` | PoseStamped mirror topic for the virtual object marker. |

VR and leader arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `reactivate_topic` | `/reactivate` | Bool topic used to toggle the VR controller. |
| `leader_urdf_path` | package default | Leader robot URDF used by `controller_type:=leader`. |
| `arm` | `true` | Enables arm retargeting in `vr` mode. |
| `hand` | `false` | Enables hand retargeting in `vr` mode. |

When `controller_type:=movel` and `start_interactive_marker:=true`, `ai_worker_controller.launch.py` starts two configurable interactive markers:

- right marker uses `right_controlled_link` and publishes MoveL commands to `right_movel_topic`
- left marker uses `left_controlled_link` and publishes MoveL commands to `left_movel_topic`

When `controller_type:=bimanual_movel` and `start_interactive_marker:=true`, the launch file starts right and left hand markers while grasp mode is inactive. These markers publish `robotis_interfaces/msg/MoveL` commands to `right_movel_topic` and `left_movel_topic`. The marker commands use `time_from_start: 0`, so the bimanual MoveL controller treats them as direct servo goals, matching the normal MoveL controller behavior.

The bimanual MoveL controller also supports rigid grasp control:

- publish `true` to `grasp_capture_topic` to capture the current relative grasp between both hands
- while grasp mode is active, the right and left hand markers are disabled
- a virtual object marker is enabled and publishes MoveL commands to `virtual_object_movel_topic`
- virtual-object MoveL commands with `time_from_start: 0` act as direct marker goals; positive durations are interpolated inside the controller
- publish `false` to `grasp_capture_topic` to release rigid grasp control and return to independent hand MoveL goals

To disable collision checking only between the two grippers, set `disable_gripper_collisions:=true`. This helps maintain smooth handover-style motions when the grippers come into contact.

Example `movel` commands for normal and bimanual MoveL:

```bash
ros2 topic pub --once /r_goal_move robotis_interfaces/msg/MoveL "{
  pose: {
    header: {frame_id: 'base_link'},
    pose: {
      position: {x: 0.35, y: -0.20, z: 0.85},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  },
  time_from_start: {sec: 2, nanosec: 0}
}"
```

```bash
ros2 topic pub --once /l_goal_move robotis_interfaces/msg/MoveL "{
  pose: {
    header: {frame_id: 'base_link'},
    pose: {
      position: {x: 0.35, y: 0.20, z: 0.85},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  },
  time_from_start: {sec: 2, nanosec: 0}
}"
```

For marker-style direct servo goals, set `time_from_start` to zero:

```bash
ros2 topic pub --once /r_goal_move robotis_interfaces/msg/MoveL "{
  pose: {
    header: {frame_id: 'base_link'},
    pose: {
      position: {x: 0.35, y: -0.20, z: 0.85},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  },
  time_from_start: {sec: 0, nanosec: 0}
}"
```

For bimanual rigid grasp mode, command the virtual object with:

```bash
ros2 topic pub --once /capture_grasp std_msgs/msg/Bool "{data: true}"
```

```bash
ros2 topic pub --once /virtual_object_goal_move robotis_interfaces/msg/MoveL "{
  pose: {
    header: {frame_id: 'base_link'},
    pose: {
      position: {x: 0.35, y: 0.00, z: 0.85},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  },
  time_from_start: {sec: 0, nanosec: 0}
}"
```

`movel` interpolation duration is supplied per command via `time_from_start`. A value of zero is intended for marker-based control and makes the controller track the latest goal directly.

Example `movej` input commands:

```bash
ros2 topic pub --once /leader/joint_trajectory_command_broadcaster_right/raw_joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3', 'arm_r_joint4', 'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7', 'gripper_r_joint1'],
  points: [
    {
      positions: [0.3, -0.2, 0.1, 0.0, 0.2, -0.1, 0.0, 0.02],
      time_from_start: {sec: 3, nanosec: 0}
    }
  ]
}"
```

```bash
ros2 topic pub --once /leader/joint_trajectory_command_broadcaster_left/raw_joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['arm_l_joint1', 'arm_l_joint2', 'arm_l_joint3', 'arm_l_joint4', 'arm_l_joint5', 'arm_l_joint6', 'arm_l_joint7', 'gripper_l_joint1'],
  points: [
    {
      positions: [-0.3, 0.2, -0.1, 0.0, -0.2, 0.1, 0.0, 0.02],
      time_from_start: {sec: 3, nanosec: 0}
    }
  ]
}"
```

`ai_worker_movej_controller` subscribes to the raw trajectory topics and republishes filtered trajectories while preserving gripper values from the input message. `ai_worker_bimanual_movej_controller` uses the same raw trajectory topics, adds bimanual rigid-grasp filtering when enabled through `grasp_capture_topic`, and republishes filtered arm trajectories.

### OMX Controllers

Launch the OMX follower controller:

```bash
ros2 launch cyclo_motion_controller_ros omx_controller.launch.py start_interactive_marker:=true
```

You can switch OMX controllers via `controller_type`:

- default launch runs `omx_movel_controller_node`
- `controller_type:=movej` runs `omx_movej_controller_node`
- `controller_type:=movel` runs `omx_movel_controller_node`

When `controller_type:=movel` and `start_interactive_marker:=true`, `omx_controller.launch.py` starts one configurable interactive marker that publishes to `marker_goal_topic`.

Example `movel` command:

```bash
ros2 topic pub --once /omx_movel_controller/movel robotis_interfaces/msg/MoveL "{
  pose: {
    pose: {
      position: {x: 0.20, y: 0.00, z: 0.18},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  },
  time_from_start: {sec: 3, nanosec: 0}
}"
```

`movel` interpolation duration is supplied per command via `time_from_start`.

Example `movej` command:

```bash
ros2 topic pub --once /omx_movej_controller/movej trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper_joint_1'],
  points: [
    {
      positions: [0.0, -0.5, 0.8, 0.0, 0.3, 0.02],
      time_from_start: {sec: 3, nanosec: 0}
    }
  ]
}"
```

`omx_movej_controller_node` republishes a patched copy of the input `movej` message, so gripper commands included in the input remain in the published trajectory.

### OMY Controllers

Launch the OMY follower controller:

```bash
ros2 launch cyclo_motion_controller_ros omy_controller.launch.py start_interactive_marker:=true
```

You can switch OMY controllers via `controller_type`:

- default launch runs `omy_movel_controller_node`
- `controller_type:=movej` runs `omy_movej_controller_node`
- `controller_type:=movel` runs `omy_movel_controller_node`

When `controller_type:=movel` and `start_interactive_marker:=true`, `omy_controller.launch.py` starts one configurable interactive marker that publishes to `marker_goal_topic`.

Example `movel` command:

```bash
ros2 topic pub --once /omy_movel_controller/movel robotis_interfaces/msg/MoveL "{
  pose: {
    pose: {
      position: {x: 0.30, y: -0.20, z: 0.5},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  },
  time_from_start: {sec: 3, nanosec: 0}
}"
```

`movel` interpolation duration is supplied per command via `time_from_start`.

Example `movej` command:

```bash
ros2 topic pub --once /omy_movej_controller/movej trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'rh_r1_joint'],
  points: [
    {
      positions: [0.0, -0.5, 0.8, 0.0, 0.3, 0.0, 1.0],
      time_from_start: {sec: 3, nanosec: 0}
    }
  ]
}"
```

`omy_movej_controller_node` also republishes a patched copy of the input `movej` message, preserving gripper values when the gripper joint is included in the input.

### Model Visualization

You can visualize the robot models used by the controllers with the launch files below.

Examples:

```bash
ros2 launch cyclo_motion_controller_models view_ffw_sg2_follower.launch.py
```

```bash
ros2 launch cyclo_motion_controller_models view_omx_f.launch.py
```

```bash
ros2 launch cyclo_motion_controller_models view_omy_f3m.launch.py
```

## Acknowledgements

This repository builds on several excellent open-source projects. The core motion controller implementations are derived from and informed by [`dyros_robot_controller`](https://github.com/JunHeonYoon/dyros_robot_controller), a project researched at Seoul National University. The retargeting modules in this repository are derived from and informed by [`dex-retargeting`](https://github.com/dexsuite/dex-retargeting). The robot kinematics used throughout the controller stack are based on [`pinocchio`](https://github.com/stack-of-tasks/pinocchio), and the optimization layer uses [`osqp-eigen`](https://github.com/robotology/osqp-eigen) for the Eigen-based OSQP solver interface.
