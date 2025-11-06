from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    pkg_name = 'PArobot_setup'
    home = os.path.expanduser('~')

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

    # --- Follower Node ---
    follower_node = Node(
        package=pkg_name,
        executable='follower_node',
        name='follower_node',
        output='screen',
        parameters=[{
            'image_width': 640,
            'turn_gain': 70.0,
            'forward_speed': 0.6,
            'dead_zone_ratio': 0.2,
            'turn_sensitivity': 0.9,
            'accel_rate': 0.8,
            'follow_dist_near': 0.8,
            'follow_dist_far': 1.3,
        }]
    )

    # --- Motor Node ---
    motor_node = Node(
        package=pkg_name,
        executable='motor_node',
        name='motor_node',
        output='screen'
    )

    # --- Main Supervisor Node ---
    main_node = Node(
        package=pkg_name,
        executable='main_node',
        name='main_node',
        output='screen'
    )

    # --- Optional: YDLIDAR driver (only if available) ---
    # NOTE: this assumes you have sourced the ydlidar_ros2_ws before launch
    lidar_launch = Node(
        package='ydlidar_ros2_driver',
        executable='ydlidar_ros2_driver_node',
        name='ydlidar_ros2_driver_node',
        output='screen',
        parameters=[{
            'port': '/dev/ttyUSB0',
            'baudrate': 115200,
            'frame_id': 'laser_frame',
            'frequency': 6.0,
            'range_max': 12.0,
            'range_min': 0.1,
        }]
    )

    # Return full launch description
    return LaunchDescription([
        # lidar_launch,   # uncomment later when you have LiDAR plugged in
        camera_node,
        detector_node,
        follower_node,
        motor_node,
        main_node,
    ])
