#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('base_frame', default_value='base_link',
                              description='Frame for interactive markers and goal poses.'),
        DeclareLaunchArgument('marker_scale', default_value='0.2',
                              description='Interactive marker scale.'),
    ]

    base_frame = LaunchConfiguration('base_frame')
    marker_scale = LaunchConfiguration('marker_scale')

    controller_node = Node(
        package='motion_controller_ros',
        executable='ai_worker_controller_node',
        output='screen',
    )

    interactive_marker = Node(
        package='motion_controller_ros',
        executable='eef_interactive_marker_node',
        name='eef_interactive_marker_node',
        parameters=[{
            'base_frame': base_frame,
            'marker_scale': marker_scale,
        }],
        output='screen',
    )
    
    return LaunchDescription(
        declared_arguments + [
            controller_node,
            interactive_marker,
        ]
    )
