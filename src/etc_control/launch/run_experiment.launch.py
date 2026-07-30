#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sigma_arg = DeclareLaunchArgument(
        'sigma', default_value='0.1',
        description='Event-triggering threshold sigma (base value if adaptive_sigma is true)'
    )
    shape_arg = DeclareLaunchArgument(
        'shape', default_value='square',
        description="Reference trajectory shape: 'square' (default) or 'circle'"
    )
    c_arg = DeclareLaunchArgument(
        'c', default_value='0.0',
        description='Event-triggering constant c (use 0.0 for continuous baseline with sigma=0.0)'
    )
    hold_mode_arg = DeclareLaunchArgument(
        'hold_mode', default_value='zoh',
        description="Actuation hold strategy between events: 'zoh' (L1) or 'smart' (L2)"
    )
    adaptive_sigma_arg = DeclareLaunchArgument(
        'adaptive_sigma', default_value='false',
        description='L3: scale sigma down during turns (curvature-adaptive threshold)'
    )
    sigma_min_factor_arg = DeclareLaunchArgument(
        'sigma_min_factor', default_value='0.5',
        description='L3: floor multiplier applied to sigma at full turn rate (1.0 = no adaptation)'
    )
    omega_max_arg = DeclareLaunchArgument(
        'omega_max', default_value='0.5',
        description='L3: turn rate (rad/s) at which sigma reaches its floor; matches trajectory omega_turn'
    )
    bag_name_arg = DeclareLaunchArgument(
        'bag_name', default_value='experiment_run',
        description='Output folder name for the rosbag'
    )
    settle_time_arg = DeclareLaunchArgument(
        'settle_time', default_value='5.0',
        description='Seconds to wait after spawn before starting control nodes (lets pose settle)'
    )

    sigma = LaunchConfiguration('sigma')
    shape = LaunchConfiguration('shape')
    c = LaunchConfiguration('c')
    hold_mode = LaunchConfiguration('hold_mode')
    adaptive_sigma = LaunchConfiguration('adaptive_sigma')
    sigma_min_factor = LaunchConfiguration('sigma_min_factor')
    omega_max = LaunchConfiguration('omega_max')
    bag_name = LaunchConfiguration('bag_name')
    settle_time = LaunchConfiguration('settle_time')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('swarm_description'), 'launch', 'gazebo.launch.py'
            ])
        )
    )

    spawn_launch = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('swarm_description'), 'launch', 'spawn.launch.py'
                    ])
                )
            )
        ]
    )

    trajectory_generator_node = Node(
        package='etc_control',
        executable='trajectory_generator',
        name='trajectory_generator',
        output='screen',
        parameters=[{'shape': shape}]
    )

    event_trigger_node = Node(
        package='etc_control',
        executable='event_trigger_node',
        name='event_trigger_node',
        output='screen',
        parameters=[{
            'sigma': sigma,
            'c': c,
            'adaptive_sigma': adaptive_sigma,
            'sigma_min_factor': sigma_min_factor,
            'omega_max': omega_max,
        }]
    )

    esp32_emulator_node = Node(
        package='etc_control',
        executable='esp32_emulator_node',
        name='esp32_emulator_node',
        output='screen',
        parameters=[{'hold_mode': hold_mode}]
    )

    control_nodes = TimerAction(
        period=3.0,
        actions=[
            TimerAction(
                period=settle_time,
                actions=[trajectory_generator_node, event_trigger_node, esp32_emulator_node]
            )
        ]
    )

    bag_record = TimerAction(
        period=3.0,
        actions=[
            TimerAction(
                period=settle_time,
                actions=[
                    TimerAction(
                        period=2.0,
                        actions=[
                            ExecuteProcess(
                                cmd=[
                                    'ros2', 'bag', 'record',
                                    '/diff_drive_controller/odom',
                                    '/diff_drive_controller/cmd_vel',
                                    '/cmd_vel_triggered',
                                    '/cmd_vel_correction_triggered',
                                    '/trajectory_desired/twist',
                                    '/trajectory_desired/pose',
                                    '-o', bag_name
                                ],
                                output='screen'
                            )
                        ]
                    )
                ]
            )
        ]
    )

    return LaunchDescription([
        sigma_arg,
        shape_arg,
        c_arg,
        hold_mode_arg,
        adaptive_sigma_arg,
        sigma_min_factor_arg,
        omega_max_arg,
        bag_name_arg,
        settle_time_arg,
        gazebo_launch,
        spawn_launch,
        control_nodes,
        bag_record,
    ])
