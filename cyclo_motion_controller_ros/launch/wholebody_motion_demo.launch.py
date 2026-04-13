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

"""Launch looping wholebody and orientation demo sequence."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            'follower_urdf_path',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('cyclo_motion_controller_models'),
                    'models',
                    'ai_worker',
                    'ffw_sg2_follower.urdf',
                ]
            ),
            description='Path to robot URDF file.',
        ),
        DeclareLaunchArgument(
            'default_srdf_path',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('cyclo_motion_controller_models'),
                    'models',
                    'ai_worker',
                    'ffw_sg2_follower_default.srdf',
                ]
            ),
            description='Path to default robot SRDF file.',
        ),
        DeclareLaunchArgument(
            'modified_srdf_path',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('cyclo_motion_controller_models'),
                    'models',
                    'ai_worker',
                    'ffw_sg2_follower_modified.srdf',
                ]
            ),
            description='Path to SRDF file with hand collision disabled.',
        ),
        DeclareLaunchArgument(
            'disable_gripper_collisions',
            default_value='false',
            description='Disable collision checking between arm_l_link7 and arm_r_link7.',
        ),
        DeclareLaunchArgument(
            'control_frequency',
            default_value='100.0',
            description='Control/update frequency for controller and reference node.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare('cyclo_motion_controller_ros'),
                    'config',
                    'ai_worker_config.yaml',
                ]
            ),
            description='Path to controller config file.',
        ),
    ]

    follower_urdf_path = LaunchConfiguration('follower_urdf_path')
    default_srdf_path = LaunchConfiguration('default_srdf_path')
    modified_srdf_path = LaunchConfiguration('modified_srdf_path')
    disable_gripper_collisions = LaunchConfiguration('disable_gripper_collisions')
    control_frequency = LaunchConfiguration('control_frequency')
    config_file = LaunchConfiguration('config_file')

    follower_srdf_path = PythonExpression(
        [
            "'",
            modified_srdf_path,
            "' if '",
            disable_gripper_collisions,
            "' == 'true' else '",
            default_srdf_path,
            "'",
        ]
    )

    wholebody_controller_node = Node(
        package='cyclo_motion_controller_ros',
        executable='wholebody_controller_node',
        output='screen',
        parameters=[
            {
                'urdf_path': follower_urdf_path,
                'srdf_path': follower_srdf_path,
                'control_frequency': control_frequency,
                'goal_ref_timeout': 0.15,
                'elbow_ref_timeout': 0.15,
                'arm_base_ref_timeout': 0.15,
            }
        ],
    )

    wholebody_reference_node = Node(
        package='cyclo_motion_controller_ros',
        executable='wholebody_reference_node',
        output='screen',
        parameters=[
            {
                'control_frequency': control_frequency,
            }
        ],
    )

    ai_worker_movel_controller_node = Node(
        package='cyclo_motion_controller_ros',
        executable='ai_worker_movel_controller_node',
        output='screen',
        parameters=[
            config_file,
            {
                'urdf_path': follower_urdf_path,
                'srdf_path': follower_srdf_path,
                'goal_ref_timeout': 0.15,
            },
        ],
    )

    orientation_reference_demo_node = Node(
        package='cyclo_motion_controller_ros',
        executable='orientation_reference_demo_node',
        output='screen',
        parameters=[
            {
                'right_controlled_link': 'end_effector_r_link',
                'left_controlled_link': 'end_effector_l_link',
                'start_on_launch': False,
                'startup_delay_sec': 0.0,
                'loop': False,
            }
        ],
    )

    return LaunchDescription(
        declared_arguments
        + [
            wholebody_controller_node,
            wholebody_reference_node,
            ai_worker_movel_controller_node,
            orientation_reference_demo_node,
        ]
    )
