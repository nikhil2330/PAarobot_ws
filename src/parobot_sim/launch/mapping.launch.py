#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    AppendEnvironmentVariable,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_name = "parobot_sim"
    pkg_share = get_package_share_directory(pkg_name)

    # ---------- World + gz_sim ----------
    world_path = os.path.join(pkg_share, "worlds", "parobot_world.sdf")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    gz_sim_launch = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    use_sim_time = LaunchConfiguration("use_sim_time")

    # Let Gazebo find this world's resources (optional but nice)
    set_gz_resource_path = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH", pkg_share
    )

    # Gazebo
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={
            "gz_args": f"-r {world_path}",
            "on_exit_shutdown": "true",
        }.items(),
    )

    # ---------- Bridge Gazebo <-> ROS 2 ----------
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="parobot_bridge",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
        ],
        remappings=[
            ("/odometry", "/odom"),
            ("/lidar", "/scan"),
        ],
    )

    # ---------- Static TF: base_link -> lidar ----------
    # LaserScan header.frame_id is "parobot/base_link/lidar"
    static_tf_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_lidar",
        arguments=[
            "0.15", "0", "0.18",   # x y z
            "0", "0", "0",         # roll pitch yaw
            "base_link", "parobot/base_link/lidar",
        ],
        output="screen",
    )

    # ---------- Odom -> base_link TF from /odom ----------
    odom_tf_broadcaster = Node(
        package=pkg_name,
        executable="odom_tf_broadcaster",
        name="odom_tf_broadcaster",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # ---------- Random walk node ----------
    random_walk = Node(
        package=pkg_name,
        executable="random_walk_node",
        name="random_walk_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        remappings=[
            ("/cmd_vel_raw", "/cmd_vel"),
        ],
    )

    # ---------- SLAM Toolbox via online_async_launch.py ----------
    # Use the async params file that comes with slam_toolbox
    slam_params = os.path.join(
        get_package_share_directory("parobot_sim"),
        "config",
        "parobot_slam_params.yaml",
    )
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "launch",
                "online_async_launch.py",
            )
        ),
        launch_arguments={
            # NOTE: this can be a LaunchConfiguration
            "use_sim_time": use_sim_time,
            # autostart handles configure + activate internally
            "autostart": "true",
            # we are NOT using an external lifecycle manager here
            "use_lifecycle_manager": "false",
            "slam_params_file": slam_params,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time",
        ),
        set_gz_resource_path,
        start_gazebo,
        bridge,
        static_tf_lidar,
        odom_tf_broadcaster,
        random_walk,
        slam_launch,
    ])
