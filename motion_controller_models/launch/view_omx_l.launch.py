#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yeonguk Kim

"""Launch RViz and state publishers for the OMX L model."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    """Create the launch description for the OMX L viewer."""
    use_gui = LaunchConfiguration('use_gui')
    use_sim_time = LaunchConfiguration('use_sim_time')

    pkg_share = get_package_share_directory('motion_controller_models')
    urdf_path = os.path.join(pkg_share, 'models', 'omx', 'omx_l.urdf')
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'omx.rviz')

    with open(urdf_path, 'r', encoding='utf-8') as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_gui',
            default_value='true',
            description=(
                'Run joint_state_publisher_gui (recommended for viewing)'
            ),
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use /clock if available',
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
            }],
            output='screen',
        ),

        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            condition=IfCondition(use_gui),
            output='screen',
        ),

        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            condition=UnlessCondition(use_gui),
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen',
        ),
    ])
