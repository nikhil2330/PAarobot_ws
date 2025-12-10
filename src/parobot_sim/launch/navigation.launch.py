from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time")
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="true"
    )

    nav2_params = os.path.join(
        get_package_share_directory("parobot_sim"),
        "config",
        "nav2_params.yaml"
    )

    nav2_bringup = Node(
        package="nav2_bringup",
        executable="bringup_launch.py",
        output="screen",
        parameters=[nav2_params, {"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        declare_use_sim_time,
        nav2_bringup
    ])
