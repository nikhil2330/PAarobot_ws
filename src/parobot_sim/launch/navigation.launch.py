#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_name = "parobot_sim"
    pkg_share = get_package_share_directory(pkg_name)

    use_sim_time = LaunchConfiguration("use_sim_time")

    # 1) SIM (Gazebo + bridge + TFs)
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, "launch", "sim_only.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 2) Random walk (Mode 1). Disable via:
    # ros2 service call /random_walk_node/set_enabled std_srvs/srv/SetBool "{data: false}"
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

    # 3) SLAM toolbox (mapping continues while nav2 moves)
    slam_params = os.path.join(pkg_share, "config", "parobot_slam_params.yaml")
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "launch",
                "online_async_launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "autostart": "true",
            "use_lifecycle_manager": "false",
            "slam_params_file": slam_params,
        }.items(),
    )

    # 4) Nav2 bringup (NO 'slam' argument to avoid PythonExpression(true) crash on Jazzy)
    nav2_params = os.path.join(pkg_share, "config", "nav2_params_default.yaml")

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("nav2_bringup"),
                "launch",
                "navigation_launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "autostart": "true",
            "params_file": nav2_params,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time",
        ),
        sim,
        random_walk,
        slam,
        nav2,
    ])
