from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_sim = get_package_share_directory('parobot_sim')

    # URDF xacro
    urdf_path = os.path.join(pkg_sim, 'urdf', 'parobot.urdf.xacro')

    # World for gz sim
    world_path = os.path.join(pkg_sim, 'worlds', 'empty.sdf')

    # ✅ Proper xacro processing for robot_description
    robot_description = {
        "robot_description": ParameterValue(
            Command(['xacro', ' ', urdf_path]),
            value_type=str,
        )
    }

    # Robot state publisher
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description],
        output='screen',
    )

    # Gazebo (gz sim) launcher
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            )
        ),
        launch_arguments={
            'gz_args': world_path,
        }.items(),
    )

    # Spawn robot from /robot_description
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'parobot',
            '-topic', 'robot_description',
        ],
        output='screen',
    )

    # ros_gz_bridge for camera, lidar, cmd_vel
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
        ],
        output='screen',
    )

    # ✅ Fake detector (OpenCV HSV-based) instead of TF/TFLite
    fake_detector = Node(
        package='parobot_sim',
        executable='fake_person_detector_node',
        name='fake_person_detector',
        output='screen',
        parameters=[{
            'min_area': 300.0,
        }]
    )

    # Your real follower + main node from PArobot_setup
    follower = Node(
        package='PArobot_setup',
        executable='follower_node',
        name='follower_node',
        output='screen',
        parameters=[{
            'image_width': 640,
        }]
    )

    main_node = Node(
        package='PArobot_setup',
        executable='main_node',
        name='main_node',
        output='screen',
    )

    return LaunchDescription([
        gz_sim,
        rsp,
        spawn,
        bridge,
        fake_detector,
        follower,
        main_node,
    ])
