#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            'start_interactive_marker',
            default_value='false',
            description='Start interactive marker for marker-follow mode.',
        ),
        DeclareLaunchArgument(
            'base_frame',
            default_value='link0',
            description='Base frame for the OMX controller and interactive marker.',
        ),
        DeclareLaunchArgument(
            'urdf_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('motion_controller_models'),
                'models',
                'omx',
                'omx_f.urdf',
            ]),
            description='Path to robot URDF file.',
        ),
        DeclareLaunchArgument(
            'srdf_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('motion_controller_models'),
                'models',
                'omx',
                'omx_f.srdf',
            ]),
            description='Path to robot SRDF file.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('motion_controller_ros'),
                'config',
                'omx_config.yaml',
            ]),
            description='Path to controller config file.',
        ),
        DeclareLaunchArgument(
            'controlled_link',
            default_value='end_effector_link',
            description='Controlled end-effector link name.',
        ),
        DeclareLaunchArgument(
            'controller_type',
            default_value='omx',
            description='Controller type (omx, movej, movel).',
        ),
        DeclareLaunchArgument(
            'marker_goal_topic',
            default_value='/eef_goal_pose',
            description='Goal topic published by the interactive marker.',
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

    omx_controller_node = Node(
        package='motion_controller_ros',
        executable='omx_controller_node',
        parameters=[config_file, {
            'urdf_path': urdf_path,
            'srdf_path': srdf_path,
            'base_frame': base_frame,
            'controlled_link': controlled_link,
        }],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", controller_type, "' == 'omx'"
        ])),
    )

    omx_movej_controller_node = Node(
        package='motion_controller_ros',
        executable='omx_movej_controller_node',
        parameters=[config_file, {
            'urdf_path': urdf_path,
            'srdf_path': srdf_path,
            'base_frame': base_frame,
            'controlled_link': controlled_link,
        }],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", controller_type, "' == 'movej'"
        ])),
    )

    omx_movel_controller_node = Node(
        package='motion_controller_ros',
        executable='omx_movel_controller_node',
        parameters=[config_file, {
            'urdf_path': urdf_path,
            'srdf_path': srdf_path,
            'base_frame': base_frame,
            'controlled_link': controlled_link,
        }],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", controller_type, "' == 'movel'"
        ])),
    )

    interactive_marker_node = Node(
        package='motion_controller_ros',
        executable='interactive_marker_node',
        name='omx_interactive_marker_node',
        parameters=[{
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
        }],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", controller_type, "' == 'omx' and '", start_interactive_marker, "' == 'true'"
        ])),
    )

    return LaunchDescription(
        declared_arguments + [
            omx_controller_node,
            omx_movej_controller_node,
            omx_movel_controller_node,
            interactive_marker_node,
        ]
    )
