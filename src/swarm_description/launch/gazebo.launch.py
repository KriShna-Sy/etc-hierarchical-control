"""
gazebo.launch.py
================
Launches Gazebo Harmonic with the confined_space world and bridges all
topics to ROS2 via ros_gz_bridge.

Motion architecture:
  - gz-sim-diff-drive-system (inside URDF) drives wheels natively in Gazebo
  - /diff_drive_controller/cmd_vel bridged ROS2→Gazebo (Twist)
  - /diff_drive_controller/odom  bridged Gazebo→ROS2 (Odometry)
  - /tf                          bridged Gazebo→ROS2 (TFMessage) ← odom→base_link

Infrastructure cameras (fixed in room):
  - OAK-D Lite  → /infra/cam_oak/{rgb,depth}/{image,camera_info,points}
  - RealSense D435 → /infra/cam_rs/{rgb,depth}/{image,camera_info,points}

Terminal 1:
    source ~/swarm2_ws/install/setup.bash
    ros2 launch swarm_description gazebo.launch.py
"""
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('swarm_description')
    world     = os.path.join(pkg_share, 'worlds', 'etc_arena.sdf')

    # Plugin search paths — sensors, diff-drive, joint-state-publisher
    gz_plugin_path = ':'.join([
        '/opt/ros/jazzy/opt/gz_sim_vendor/lib/gz-sim-8/plugins',
        '/opt/ros/jazzy/opt/gz_sim_vendor/lib',
        '/opt/ros/jazzy/lib',
    ])

    bridges = [
        # ── Simulation clock ─────────────────────────────────────────
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',

        # ── Differential drive ───────────────────────────────────────
        # ROS2 publishes Twist → Gazebo receives it (] = ROS→GZ direction)
        '/diff_drive_controller/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        # Gazebo publishes Odometry → ROS2 receives it ([ = GZ→ROS direction)
        '/diff_drive_controller/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',

        # ── TF from diff-drive plugin (odom → base_link) ─────────────
        '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',

        # ── Joint states from gz-sim-joint-state-publisher ───────────
        # gz.msgs.Model is the correct type for Gazebo 8 joint state publisher
        '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',

        # ── OAK-D Lite infrastructure camera ─────────────────────────
        # RGB
        '/infra/cam_oak/rgb/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/infra/cam_oak/rgb/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        # Depth (32FC1 float in metres from Gazebo)
        '/infra/cam_oak/depth/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/infra/cam_oak/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        # Point cloud
        '/infra/cam_oak/depth/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',

        # ── Intel RealSense D435 infrastructure camera ────────────────
        # RGB
        '/infra/cam_rs/rgb/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/infra/cam_rs/rgb/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        # Depth
        '/infra/cam_rs/depth/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/infra/cam_rs/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        # Point cloud
        '/infra/cam_rs/depth/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
    ]

    return LaunchDescription([

        # ── Environment: tell Gazebo where all plugins live ───────────
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', gz_plugin_path),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH',
                               '/opt/ros/jazzy/opt/gz_sim_vendor/share'),
        SetEnvironmentVariable(
            'LD_LIBRARY_PATH',
            gz_plugin_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        ),

        # ── Gazebo Harmonic simulation ────────────────────────────────
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world],
            output='screen'
        ),

        # ── ROS ↔ Gazebo topic bridge ─────────────────────────────────
        ExecuteProcess(
            cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge'] + bridges,
            output='screen'
        ),

    ])
