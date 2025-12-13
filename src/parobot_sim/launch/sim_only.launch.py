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

    world_path = os.path.join(pkg_share, "worlds", "parobot_world.sdf")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    gz_sim_launch = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    use_sim_time = LaunchConfiguration("use_sim_time")

    set_gz_resource_path = AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", pkg_share)

    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={
            "gz_args": f"-r {world_path}",
            "on_exit_shutdown": "true",
        }.items(),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="parobot_bridge",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",                 # GZ -> ROS
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",               # ROS -> GZ
            "/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",             # GZ -> ROS
            "/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",           # GZ -> ROS
        ],
        remappings=[
            ("/odometry", "/odom"),
            ("/lidar", "/scan"),
        ],
    )

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

    odom_tf_broadcaster = Node(
        package=pkg_name,
        executable="odom_tf_broadcaster",
        name="odom_tf_broadcaster",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # random_walk = Node(
    #     package=pkg_name,
    #     executable="random_walk_node",
    #     name="random_walk_node",
    #     parameters=[{"use_sim_time": use_sim_time}],
    #     output="screen",
    # )

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
        # random_walk,
    ])
