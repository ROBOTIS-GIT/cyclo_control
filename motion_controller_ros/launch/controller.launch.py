#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('start_interactive_marker', default_value='false',
                              description='Start interactive markers for goal poses.'),
        DeclareLaunchArgument('base_frame', default_value='base_link',
                              description='Frame for interactive markers and goal poses.'),
        DeclareLaunchArgument('marker_scale', default_value='0.2',
                              description='Interactive marker scale.'),
        DeclareLaunchArgument('urdf_path',
                              default_value=PathJoinSubstitution([
                                  FindPackageShare('ffw_description'),
                                  'urdf',
                                  'ffw_sg2_rev1_follower',
                                  'ffw_sg2_follower.urdf'
                              ]),
                              description='Path to robot URDF file.'),
        DeclareLaunchArgument('srdf_path',
                              default_value=PathJoinSubstitution([
                                  FindPackageShare('ffw_description'),
                                  'urdf',
                                  'ffw_sg2_rev1_follower',
                                  'ffw.srdf'
                              ]),
                              description='Path to robot SRDF file.'),
        DeclareLaunchArgument('config_file',
                              default_value=PathJoinSubstitution([
                                  FindPackageShare('motion_controller_ros'),
                                  'config',
                                  'ai_worker_config.yaml'
                              ]),
                              description='Path to ai_worker_controller config file.'),
    ]

    start_interactive_marker = LaunchConfiguration('start_interactive_marker')
    urdf_path = LaunchConfiguration('urdf_path')
    srdf_path = LaunchConfiguration('srdf_path')
    base_frame = LaunchConfiguration('base_frame')
    marker_scale = LaunchConfiguration('marker_scale')
    config_file = LaunchConfiguration('config_file')

    controller_node = Node(
        package='motion_controller_ros',
        executable='ai_worker_controller_node',
        parameters=[config_file, {
            'urdf_path': urdf_path,
            'srdf_path': srdf_path,
        }],
        output='screen',
    )

    reference_checker_node = Node(
        package='motion_controller_ros',
        executable='reference_checker_node',
        parameters=[config_file],
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
        condition=IfCondition(start_interactive_marker)
    )
    
    return LaunchDescription(
        declared_arguments + [
            controller_node,
            reference_checker_node,
            interactive_marker,
        ]
    )
