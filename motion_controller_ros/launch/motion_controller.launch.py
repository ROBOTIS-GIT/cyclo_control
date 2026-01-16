from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Arguments
    base_link_arg = DeclareLaunchArgument(
        'base_link', default_value='base_link',
        description='Base link frame id'
    )

    tip_links_arg = DeclareLaunchArgument(
        'tip_links', default_value='["end_effector"]',
        description='List of tip link frame ids (e.g. ["link7", "gripper"])'
    )

    cmd_topic_arg = DeclareLaunchArgument(
        'command_topic', default_value='/joint_commands',
        description='Topic name to publish joint commands'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level', default_value='info',
        description='Logging level'
    )

    # Node
    controller_node = Node(
        package='motion_controller_ros',
        executable='motion_control_node',
        name='motion_control_node',
        output='screen',
        parameters=[{
            'base_link': LaunchConfiguration('base_link'),
            'tip_links': LaunchConfiguration('tip_links'),
            'command_topic': LaunchConfiguration('command_topic'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    return LaunchDescription([
        base_link_arg,
        tip_links_arg,
        cmd_topic_arg,
        log_level_arg,
        controller_node
    ])
