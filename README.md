# ROBOTIS Motion Controller

This repository contains motion controller packages for the ROBOTIS Physical AI lineup. It provides a QP-based inverse kinematics controller (`motion_controller_core`) and ROS 2 wrapper files (`motion_controller_ros`) that run controllers for the AI Worker model (support for additional models is on the roadmap), using Pinocchio for kinematics and OSQP for optimization.

## Install (from source)

### Prerequisites

- ROS 2 Jazzy installed

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

### Build in a ROS 2 workspace

```bash
cd ~/ros2_ws/src
git clone https://github.com/ROBOTIS-GIT/motion_controller.git
cd ~/ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Run

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# AI Worker controller is launched as default
ros2 launch motion_controller_ros controller.launch.py controller_type:=ai_worker start_interactive_marker:=true
```

You can also switch controllers via `controller_type`:

- `controller_type:=joint_space` (runs `joint_space_controller_node`)
- `controller_type:=leader` (runs `leader_controller_node` and also starts the follower controller for leader/follower use)