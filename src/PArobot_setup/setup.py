from setuptools import find_packages, setup

package_name = 'PArobot_setup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/PArobot_setup/launch', ['launch/robot_bringup.launch.py']),
        ('share/' + package_name + '/models', ['models/detect.tflite', 'models/labelmap.txt']),


    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='system',
    maintainer_email='system@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
            'console_scripts': [
            'motor_node = PArobot_setup.motor_node:main',
            'camera_node = PArobot_setup.camera_node:main',
            'person_detector_node = PArobot_setup.person_detector_node:main',
            'follower_node = PArobot_setup.follower_node:main',
            'main_node = PArobot_setup.main_node:main',
        ],
    }
)
