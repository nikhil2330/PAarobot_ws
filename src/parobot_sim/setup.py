from setuptools import find_packages, setup

package_name = 'parobot_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', ['urdf/parobot.urdf.xacro']),
        ('share/' + package_name + '/launch', ['launch/parobot_gz.launch.py']),
        # ✅ install the world from the *source* folder, NOT install path
        ('share/' + package_name + '/worlds', ['worlds/empty.sdf']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nikhil',
    maintainer_email='nikhil@todo.todo',
    description='Simulation of PArobot in Gazebo (gz sim) via ros_gz',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # ✅ fake detector for simulation (OpenCV-based)
            'fake_person_detector_node = parobot_sim.fake_person_detector_node:main',
        ],
    },
)
