# ROBOTIS Motion Controller

This repository contains motion controller packages for the ROBOTIS Physical AI lineup.

## Repository Structure

```text
├── motion_controller/
│   ├── CMakeLists.txt
│   └── package.xml
├── motion_controller_core/
│   ├── include/motion_controller_core/
│   │   ├── common/
│   │   │   └── ...
│   │   ├── controllers/
│   │   │   └── ...
│   │   ├── kinematics/
│   │   │   └── ...
│   │   ├── optimization/
│   │   │   └── ...
│   │   └── retargeting/
│   │       └── ...
│   ├── src/
│   │   ├── controllers/
│   │   │   └── ...
│   │   ├── kinematics/
│   │   │   └── ...
│   │   └── retargeting/
│   │       └── ...
│   ├── CMakeLists.txt
│   └── package.xml
│
├── motion_controller_ros/
│   ├── config/
│   │   └── ...
│   ├── include/motion_controller_ros/
│   │   ├── nodes/
│   │   │   └── ...
│   │   └── utils/
│   │       └── ...
│   ├── launch/
│   │   └── ...
│   ├── src/
│   │   ├── nodes/
│   │   │   └── ...
│   │   └── utils/
│   │       └── ...
│   ├── CMakeLists.txt
│   └── package.xml
│
├── motion_controller_ros_py/
│   ├── resources/
│   │   └── ...
│   ├── scripts/
│   │   └── ...
│   ├── test/
│   │   └── ...
│   ├── package.xml
│   ├── setup.cfg
│   └── setup.py
│
└── motion_controller_models/
    ├── launch/
    │   └── ...
    ├── models/
    │   └── ...
    ├── CMakeLists.txt
    └── package.xml
```
### Directory Description
| Directory | Description |
|-----------|-------------|
| `motion_controller_core/` | Core package containing kinematics solver & motion control library |
| `motion_controller_core/include/motion_controller_core/common/` | Common headers containing shared types and utility functions |
| `motion_controller_core/include/motion_controller_core/optimization/` | QP definitions and solver interfaces |
| `motion_controller_core/src/controllers/` | Controller implementations |
| `motion_controller_core/src/kinematics/` | Kinematics solver implementation |
| `motion_controller_core/src/retargeting/` | Retargeting module |

| Directory | Description |
|-----------|-------------|
| `motion_controller_ros/` | ROS 2 package containing controller nodes, launch files, and configs |
| `motion_controller_ros/launch/` | Launch files for running the controller nodes |
| `motion_controller_ros/src/nodes/` | ROS 2 node executables for each controller type |
| `motion_controller_ros/src/utils/` | Utility nodes |

| Directory | Description |
|-----------|-------------|
| `motion_controller_ros_py/` | ROS 2 Python package containing controller nodes |
| `motion_controller_ros_py/scripts/` | Script files for running the controller nodes |

| Directory | Description |
|-----------|-------------|
| `motion_controller_models/` | Robot model descriptions package |
| `motion_controller_models/launch/` | Launch files for visualizing robot models |
| `motion_controller_models/models/` | URDF/SRDF robot models used by the controller |

## Install (from source)

### Prerequisites

- **ROS 2 Jazzy** installed
- [**robotis_interfaces**](https://github.com/ROBOTIS-GIT/robotis_interfaces) available in your workspace

### Install OSQP

```bash
cd ~/
git clone https://github.com/osqp/osqp
cd ~/osqp
mkdir build && cd build
cmake -G "Unix Makefiles" ..
cmake --build .
sudo cmake --build . --target install
```

### Install OsqpEigen (osqp-eigen)

```bash
cd ~/
mkdir osqp-eigen_install
git clone https://github.com/robotology/osqp-eigen.git
cd ~/osqp-eigen
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=~/osqp-eigen_install ../
make
sudo make install
```

### Set library path (for OsqpEigen)

```bash
echo "export OsqpEigen_DIR=$HOME/osqp-eigen_install" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$HOME/osqp-eigen_install/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### Install Pinocchio

```bash
sudo apt update
sudo apt install -y ros-jazzy-pinocchio
```

### Install NLopt (for retargeting)

```bash
pip3 install nlopt
```

### Install PyTorch (for retargeting)

```bash
pip3 install torch
```

### Build in a ROS 2 workspace

```bash
cd ~/ros2_ws/src
git clone https://github.com/ROBOTIS-GIT/motion_controller.git
cd ~/ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Run

### AI Worker Controllers

Launch AI Worker follower controller

```bash
ros2 launch motion_controller_ros ai_worker_controller.launch.py controller_type:=ai_worker start_interactive_marker:=true
```

You can switch AI Worker controllers via `controller_type`:

- `controller_type:=ai_worker` runs `ai_worker_controller_node`
- `controller_type:=joint_space` runs `joint_space_controller_node`
- `controller_type:=leader` runs `leader_controller_node` and also starts the follower controller for leader/follower use

When `start_interactive_marker:=true`, `ai_worker_controller.launch.py` starts two configurable interactive markers:

- right marker uses `right_controlled_link` and publishes to `right_goal_topic`
- left marker uses `left_controlled_link` and publishes to `left_goal_topic`

### OMX Controllers

Launch the OMX follower controller:

```bash
ros2 launch motion_controller_ros omx_controller.launch.py controller_type:=omx start_interactive_marker:=true
```

You can switch OMX controllers via `controller_type`:

- `controller_type:=omx` runs `omx_controller_node`
- `controller_type:=movej` runs `omx_movej_controller_node`
- `controller_type:=movel` runs `omx_movel_controller_node`

When `controller_type:=omx` and `start_interactive_marker:=true`, `omx_controller.launch.py` starts one configurable interactive marker that publishes to `marker_goal_topic`.

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
  joint_names: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5'],
  points: [
    {
      positions: [0.0, -0.5, 0.8, 0.0, 0.3],
      time_from_start: {sec: 3, nanosec: 0}
    }
  ]
}"
```

### OMY Controllers

Launch the OMY follower controller:

```bash
ros2 launch motion_controller_ros omy_controller.launch.py controller_type:=omy start_interactive_marker:=true
```

You can switch OMY controllers via `controller_type`:

- `controller_type:=omy` runs `omy_controller_node`
- `controller_type:=movej` runs `omy_movej_controller_node`
- `controller_type:=movel` runs `omy_movel_controller_node`

When `controller_type:=omy` and `start_interactive_marker:=true`, `omy_controller.launch.py` starts one configurable interactive marker that publishes to `marker_goal_topic`.

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
  joint_names: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
  points: [                               
    {                                            
      positions: [0.0, -0.5, 0.8, 0.0, 0.3, 0.0],
      time_from_start: {sec: 3, nanosec: 0}
    }
  ]
}"
```

### Model Visualization

You can visualize the robot models used by the controllers with the launch files below.

Examples:

```bash
ros2 launch motion_controller_models view_ffw_sg2_follower.launch.py
ros2 launch motion_controller_models view_omx_f.launch.py
ros2 launch motion_controller_models view_omy_f3m.launch.py
```