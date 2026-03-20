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


"""Launch the OMX motion controller variants."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the launch description for OMX controllers."""
    declared_arguments = [
        DeclareLaunchArgument(
            'start_interactive_marker',
            default_value='false',
            description='Start interactive marker for marker-follow mode.',
        ),
        DeclareLaunchArgument(
            'base_frame',
            default_value='link0',
            description=(
                'Base frame for the OMX controller and interactive marker.'
            ),
        ),
        DeclareLaunchArgument(
            'urdf_path',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('robotis_motion_controller_models'),
                    'models',
                    'omx',
                    'omx_f.urdf',
                ]
            ),
            description='Path to robot URDF file.',
        ),
        DeclareLaunchArgument(
            'srdf_path',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('robotis_motion_controller_models'),
                    'models',
                    'omx',
                    'omx_f.srdf',
                ]
            ),
            description='Path to robot SRDF file.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('robotis_motion_controller_ros'),
                    'config',
                    'omx_config.yaml',
                ]
            ),
            description='Path to controller config file.',
        ),
        DeclareLaunchArgument(
            'controlled_link',
            default_value='end_effector_link',
            description='Controlled end-effector link name.',
        ),
        DeclareLaunchArgument(
            'controller_type',
            default_value='movel',
            description='Controller type (movej, movel). Default: movel.',
        ),
        DeclareLaunchArgument(
            'marker_goal_topic',
            default_value='/omx_movel_controller/movel',
            description='MoveL topic published by the interactive marker.',
        ),
        DeclareLaunchArgument(
            'marker_scale',
            default_value='0.2',
            description='Interactive marker scale.',
        ),
    ]

    start_interactive_marker = LaunchConfiguration('start_interactive_marker')
    base_frame = LaunchConfiguration('base_frame')
    urdf_path = LaunchConfiguration('urdf_path')
    srdf_path = LaunchConfiguration('srdf_path')
    config_file = LaunchConfiguration('config_file')
    controlled_link = LaunchConfiguration('controlled_link')
    controller_type = LaunchConfiguration('controller_type')
    marker_goal_topic = LaunchConfiguration('marker_goal_topic')
    marker_scale = LaunchConfiguration('marker_scale')

    omx_movej_controller_node = Node(
        package='robotis_motion_controller_ros',
        executable='omx_movej_controller_node',
        parameters=[
            config_file,
            {
                'urdf_path': urdf_path,
                'srdf_path': srdf_path,
                'base_frame': base_frame,
                'controlled_link': controlled_link,
            },
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", controller_type, "' == 'movej'"])
        ),
    )

    omx_movel_controller_node = Node(
        package='robotis_motion_controller_ros',
        executable='omx_movel_controller_node',
        parameters=[
            config_file,
            {
                'urdf_path': urdf_path,
                'srdf_path': srdf_path,
                'base_frame': base_frame,
                'controlled_link': controlled_link,
            },
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", controller_type, "' == 'movel'"])
        ),
    )

    interactive_marker_node = Node(
        package='robotis_motion_controller_ros',
        executable='interactive_marker_node',
        name='omx_interactive_marker_node',
        parameters=[
            {
                'base_frame': base_frame,
                'controlled_link': controlled_link,
                'goal_topic': marker_goal_topic,
                'server_name': 'omx_goal_marker',
                'marker_name': 'omx_goal_marker',
                'marker_description': 'OMX goal',
                'marker_scale': marker_scale,
                'marker_color_r': 0.2,
                'marker_color_g': 0.8,
                'marker_color_b': 0.2,
            }
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    controller_type,
                    "' == 'movel' and '",
                    start_interactive_marker,
                    "' == 'true'",
                ]
            )
        ),
    )

    return LaunchDescription(
        declared_arguments
        + [
            omx_movej_controller_node,
            omx_movel_controller_node,
            interactive_marker_node,
        ]
    )
