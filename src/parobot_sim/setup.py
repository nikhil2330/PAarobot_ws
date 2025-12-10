from setuptools import setup
from glob import glob
import os

package_name = 'parobot_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'),
         glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'worlds'),
         glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nikhil',
    maintainer_email='you@example.com',
    description='Simulation for PArobot: Gazebo world, random motion, SLAM.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points = {
    'console_scripts': [
        'random_walk_node = parobot_sim.random_walk_node:main',
        'odom_tf_broadcaster = parobot_sim.odom_tf_broadcaster:main',
    ],
}

)
