#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Launch OMY L100 joint/cartesian impedance controller nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create OMY L100 impedance launch description."""
    controller_type = LaunchConfiguration('controller_type')

    declared_arguments = [
        DeclareLaunchArgument(
            'controller_type',
            default_value='cartesian_impedance',
            description='Controller type (joint_impedance, cartesian_impedance).',
        ),
    ]

    omy_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('cyclo_motion_controller_ros'),
                    'launch',
                    'omy_controller.launch.py',
                ]
            )
        ),
        launch_arguments={
            'controller_type': controller_type,
            'urdf_path': PathJoinSubstitution(
                [
                    FindPackageShare('cyclo_motion_controller_models'),
                    'models',
                    'omy',
                    'omy_l100.urdf',
                ]
            ),
            'srdf_path': '',
            'base_frame': 'link0',
            'controlled_link': 'link6',
            'config_file': PathJoinSubstitution(
                [
                    FindPackageShare('cyclo_motion_controller_ros'),
                    'config',
                    'omy_config.yaml',
                ]
            ),
        }.items(),
    )

    return LaunchDescription(declared_arguments + [omy_controller_launch])
