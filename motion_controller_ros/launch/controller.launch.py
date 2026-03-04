#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('start_interactive_marker', default_value='false',
                              description='Start interactive markers for goal poses.'),
        DeclareLaunchArgument('base_frame', default_value='base_link',
                              description='Frame for interactive markers and goal poses.'),
        DeclareLaunchArgument('reactivate_topic', default_value='/reset',
                              description='Topic to reactivate controller.'),
        DeclareLaunchArgument('marker_scale', default_value='0.2',
                              description='Interactive marker scale.'),
        DeclareLaunchArgument('follower_urdf_path',
                              default_value=PathJoinSubstitution([
                                  FindPackageShare('motion_controller_models'),
                                  'models',
                                  'ai_worker',
                                  'ffw_sg2_follower.urdf'
                              ]),
                              description='Path to robot URDF file.'),
        DeclareLaunchArgument('follower_srdf_path',
                              default_value=PathJoinSubstitution([
                                  FindPackageShare('motion_controller_models'),
                                  'models',
                                  'ai_worker',
                                  'ffw_sg2_follower.srdf'
                              ]),
                              description='Path to robot SRDF file.'),
        DeclareLaunchArgument('leader_urdf_path',
                              default_value=PathJoinSubstitution([
                                  FindPackageShare('motion_controller_models'),
                                  'models',
                                  'leader',
                                  'ffw_lg2_leader.urdf'
                              ]),
                              description='Path to robot URDF file.'),
        DeclareLaunchArgument('leader_srdf_path',
                              default_value=PathJoinSubstitution([
                                  FindPackageShare('motion_controller_models'),
                                  'models',
                                  'leader',
                                  'ffw_lg2_leader.srdf'
                              ]),
                              description='Path to robot SRDF file.'),
        DeclareLaunchArgument('config_file',
                              default_value=PathJoinSubstitution([
                                  FindPackageShare('motion_controller_ros'),
                                  'config',
                                  'controller_config.yaml'
                              ]),
                              description='Path to controller config file.'),
        DeclareLaunchArgument('controller_type',
                              default_value='ai_worker',
                              description='Controller type (without _controller_node).'),
    ]

    start_interactive_marker = LaunchConfiguration('start_interactive_marker')
    follower_urdf_path = LaunchConfiguration('follower_urdf_path')
    follower_srdf_path = LaunchConfiguration('follower_srdf_path')
    leader_urdf_path = LaunchConfiguration('leader_urdf_path')
    leader_srdf_path = LaunchConfiguration('leader_srdf_path')
    base_frame = LaunchConfiguration('base_frame')
    reactivate_topic = LaunchConfiguration('reactivate_topic')
    marker_scale = LaunchConfiguration('marker_scale')
    config_file = LaunchConfiguration('config_file')
    controller_type = LaunchConfiguration('controller_type')
    controller_executable = PythonExpression([
        "'", controller_type, "_controller_node'"
    ])

    follower_controller_node = Node(
        package='motion_controller_ros',
        executable=controller_executable,
        parameters=[config_file, {
            'urdf_path': follower_urdf_path,
            'srdf_path': follower_srdf_path,
        }],
        output='screen',
        condition=UnlessCondition(PythonExpression([
            "'", controller_type, "' == 'leader'"
        ])),
    )

    leader_controller_node = Node(
        package='motion_controller_ros',
        executable='leader_controller_node',
        parameters=[config_file, {
            'urdf_path': leader_urdf_path,
            'srdf_path': leader_srdf_path,
            'reactivate_topic': reactivate_topic,
        }],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", controller_type, "' == 'leader'"
        ])),
    )

    follower_with_leader_node = Node(
        package='motion_controller_ros',
        executable='ai_worker_controller_node',
        parameters=[config_file, {
            'urdf_path': follower_urdf_path,
            'srdf_path': follower_srdf_path,
            'reactivate_topic': reactivate_topic,
        }],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", controller_type, "' == 'leader'"
        ])),
    )

    reference_checker_node = Node(
        package='motion_controller_ros',
        executable='reference_checker_node',
        parameters=[config_file],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", controller_type, "' == 'ai_worker'"
        ])),
    )

    interactive_marker = Node(
        package='motion_controller_ros',
        executable='eef_interactive_marker_node',
        name='eef_interactive_marker_node',
        parameters=[{
            'base_frame': base_frame,
        }],
        output='screen',
        condition=IfCondition(start_interactive_marker)
    )
    
    return LaunchDescription(
        declared_arguments + [
            follower_controller_node,
            leader_controller_node,
            follower_with_leader_node,
            reference_checker_node,
            interactive_marker,
        ]
    )
