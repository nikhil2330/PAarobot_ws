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

    # Our custom world file
    world_path = os.path.join(pkg_share, "worlds", "parobot_world.sdf")

    # ros_gz_sim's launch file
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    gz_sim_launch = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    use_sim_time = LaunchConfiguration("use_sim_time")

    # Make sure Gazebo can find worlds / models from this package
    set_gz_resource_path = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        pkg_share,
    )

    # 1) Start Gazebo Sim via ros_gz_sim’s launch file
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={
            "gz_args": [" -r ", world_path],  # "-r <world_path>"
            "on_exit_shutdown": "true",
        }.items(),
    )

    # 2) Bridge cmd_vel, odom, and lidar
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="parobot_bridge",
        output="screen",
        arguments=[
            # 1) Sim time
            "/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock",

            # 2) cmd_vel: ROS <-> GZ
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",

            # 3) odom: GZ -> ROS
            "/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",

            # 4) lidar: GZ -> ROS
            "/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
        ],
        remappings=[
            # ✅ now bridge /odometry -> /odom for ROS nodes
            ("/odometry", "/odom"),
            ("/lidar", "/scan"),
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )


    # 3) Random walk node publishing /model/parobot/cmd_vel
    random_walk = Node(
        package=pkg_name,
        executable="random_walk_node",
        name="random_walk_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

        # 2.5) Static TF: base_link -> lidar
    static_tf_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_lidar",
        # xyz rpy parent_frame child_frame
        arguments=["0.15", "0", "0.18", "0", "0", "0",
                "base_link", "parobot/base_link/lidar"],
        output="screen",
    )


    odom_tf_broadcaster = Node(
        package=pkg_name,
        executable="odom_tf_broadcaster",
        name="odom_tf_broadcaster",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )


    # 4) SLAM Toolbox (still using default config)
    slam = Node(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "config",
                "mapper_params_online_sync.yaml",
            ),
            {
                "use_sim_time": use_sim_time,
                "odom_frame": "odom",
                "base_frame": "base_link",
                "map_frame": "map",
            },
        ],
    )


    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock",
        ),
        set_gz_resource_path,
        start_gazebo,
        bridge,
        static_tf_lidar,
        odom_tf_broadcaster,
        random_walk,
        slam,
    ])
