from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    pkg_name = 'PArobot_setup'

    camera_node = Node(
        package=pkg_name,
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'width': 640,
            'height': 480,
            'fps': 30,
            'show_preview': False,
        }]
    )

    detector_node = Node(
        package=pkg_name,
        executable='person_detector_node',
        name='person_detector_node',
        output='screen',
        parameters=[{
            'model_dir': os.path.join(
                os.path.expanduser('~'),
                'PArobot_ws', 'install', pkg_name, 'share', pkg_name, 'models'
            ),
            'graph': 'detect.tflite',
            'labels': 'labelmap.txt',
            'score_thresh': 0.59,
        }]
    )

    follower_node = Node(
        package=pkg_name,
        executable='follower_node',
        name='follower_node',
        output='screen',
        parameters=[{
            'image_width': 640,
        }]
    )

    motor_node = Node(
        package=pkg_name,
        executable='motor_node',
        name='motor_node',
        output='screen'
    )

    main_node = Node(
        package=pkg_name,
        executable='main_node',
        name='main_node',
        output='screen'
    )

    rf_node = Node(
        package=pkg_name,
        executable='rf_remote_node',
        name='rf_remote_node',
        output='screen'
    )

    lidar_launch = Node(
        package='ydlidar_ros2_driver',
        executable='ydlidar_ros2_driver_node',
        name='ydlidar_ros2_driver_node',
        output='screen',
        parameters=[{
            'port': '/dev/ttyUSB0',
            'frame_id': 'laser_frame',
            'ignore_array': "",
            'baudrate': 115200,
            'lidar_type': 1,
            'device_type': 0,
            'sample_rate': 3,
            'abnormal_check_count': 4,
            'fixed_resolution': True,
            'reversion': False,
            'inverted': False,
            'auto_reconnect': True,
            'isSingleChannel': True,
            'intensity_bit': 0,
            'intensity': False,
            'support_motor_dtr': True,
            'angle_max': 180.0,
            'angle_min': -180.0,
            'range_max': 12.0,
            'range_min': 0.1,
            'frequency': 10.0,
            'invalid_range_is_inf': False,
            'debug': False,
        }]
    )

    return LaunchDescription([
        lidar_launch,
        camera_node,
        detector_node,
        follower_node,
        motor_node,
        main_node,
        rf_node,
    ])
