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


"""Launch the AI worker motion controller stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the launch description for AI worker controllers."""
    declared_arguments = [
        DeclareLaunchArgument(
            'start_interactive_marker',
            default_value='false',
            description='Start interactive markers for MoveL goals.',
        ),
        DeclareLaunchArgument(
            'base_frame',
            default_value='base_link',
            description='Frame for interactive markers and MoveL goals.',
        ),
        DeclareLaunchArgument(
            'reactivate_service',
            default_value='/reactivate',
            description='Service used to reactivate the VR controller.',
        ),
        DeclareLaunchArgument(
            'marker_scale',
            default_value='0.2',
            description='Interactive marker scale.',
        ),
        DeclareLaunchArgument(
            'right_controlled_link',
            default_value='arm_r_link7',
            description='Controlled link for the right interactive marker.',
        ),
        DeclareLaunchArgument(
            'left_controlled_link',
            default_value='arm_l_link7',
            description='Controlled link for the left interactive marker.',
        ),
        DeclareLaunchArgument(
            'right_movel_topic',
            default_value='/r_goal_move',
            description='MoveL topic for the right interactive marker.',
        ),
        DeclareLaunchArgument(
            'left_movel_topic',
            default_value='/l_goal_move',
            description='MoveL topic for the left interactive marker.',
        ),
        DeclareLaunchArgument(
            'follower_urdf_path',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('motion_controller_models'),
                    'models',
                    'ai_worker',
                    'ffw_sg2_follower.urdf',
                ]
            ),
            description='Path to robot URDF file.',
        ),
        DeclareLaunchArgument(
            'follower_srdf_path',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('motion_controller_models'),
                    'models',
                    'ai_worker',
                    'ffw_sg2_follower.srdf',
                ]
            ),
            description='Path to robot SRDF file.',
        ),
        DeclareLaunchArgument(
            'leader_urdf_path',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('motion_controller_models'),
                    'models',
                    'ai_worker',
                    'ffw_lg2_leader.urdf',
                ]
            ),
            description='Path to robot URDF file.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('motion_controller_ros'),
                    'config',
                    'ai_worker_config.yaml',
                ]
            ),
            description='Path to controller config file.',
        ),
        DeclareLaunchArgument(
            'controller_type',
            default_value='vr',
            description='Controller type (without _controller_node).',
        ),
    ]

    start_interactive_marker = LaunchConfiguration('start_interactive_marker')
    follower_urdf_path = LaunchConfiguration('follower_urdf_path')
    follower_srdf_path = LaunchConfiguration('follower_srdf_path')
    leader_urdf_path = LaunchConfiguration('leader_urdf_path')
    base_frame = LaunchConfiguration('base_frame')
    reactivate_service = LaunchConfiguration('reactivate_service')
    marker_scale = LaunchConfiguration('marker_scale')
    right_controlled_link = LaunchConfiguration('right_controlled_link')
    left_controlled_link = LaunchConfiguration('left_controlled_link')
    right_movel_topic = LaunchConfiguration('right_movel_topic')
    left_movel_topic = LaunchConfiguration('left_movel_topic')
    config_file = LaunchConfiguration('config_file')
    controller_type = LaunchConfiguration('controller_type')
    controller_executable = PythonExpression(
        ["'", controller_type, "_controller_node'"]
    )

    follower_controller_node = Node(
        package='motion_controller_ros',
        executable=controller_executable,
        parameters=[
            config_file,
            {
                'urdf_path': follower_urdf_path,
                'srdf_path': follower_srdf_path,
            },
        ],
        output='screen',
        condition=UnlessCondition(
            PythonExpression(["'", controller_type, "' == 'leader'"])
        ),
    )

    leader_controller_node = Node(
        package='motion_controller_ros',
        executable='leader_controller_node',
        parameters=[
            config_file,
            {
                'urdf_path': leader_urdf_path,
                'reactivate_service': reactivate_service,
            },
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", controller_type, "' == 'leader'"])
        ),
    )

    follower_with_leader_node = Node(
        package='motion_controller_ros',
        executable='vr_controller_node',
        parameters=[
            config_file,
            {
                'urdf_path': follower_urdf_path,
                'srdf_path': follower_srdf_path,
                'reactivate_service': reactivate_service,
            },
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", controller_type, "' == 'leader'"])
        ),
    )

    reference_checker_node = Node(
        package='motion_controller_ros',
        executable='reference_checker_node',
        parameters=[config_file],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", controller_type, "' == 'vr'"])
        ),
    )

    right_interactive_marker = Node(
        package='motion_controller_ros',
        executable='interactive_marker_node',
        name='right_interactive_marker_node',
        parameters=[
            {
                'base_frame': base_frame,
                'controlled_link': right_controlled_link,
                'goal_topic': right_movel_topic,
                'server_name': 'right_goal_marker_server',
                'marker_name': 'right_goal_marker',
                'marker_description': 'Right gripper goal',
                'marker_scale': marker_scale,
                'marker_color_r': 1.0,
                'marker_color_g': 0.2,
                'marker_color_b': 0.2,
            }
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression(
                ["'", controller_type, "' == 'ai_worker_movel' and '", start_interactive_marker, "' == 'true'"]
            )
        ),
    )

    left_interactive_marker = Node(
        package='motion_controller_ros',
        executable='interactive_marker_node',
        name='left_interactive_marker_node',
        parameters=[
            {
                'base_frame': base_frame,
                'controlled_link': left_controlled_link,
                'goal_topic': left_movel_topic,
                'server_name': 'left_goal_marker_server',
                'marker_name': 'left_goal_marker',
                'marker_description': 'Left gripper goal',
                'marker_scale': marker_scale,
                'marker_color_r': 0.2,
                'marker_color_g': 0.2,
                'marker_color_b': 1.0,
            }
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression(
                ["'", controller_type, "' == 'ai_worker_movel' and '", start_interactive_marker, "' == 'true'"]
            )
        ),
    )

    return LaunchDescription(
        declared_arguments
        + [
            follower_controller_node,
            leader_controller_node,
            follower_with_leader_node,
            reference_checker_node,
            right_interactive_marker,
            left_interactive_marker,
        ]
    )
