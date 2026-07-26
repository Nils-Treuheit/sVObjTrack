from setuptools import find_packages, setup

package_name = 'depth_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nils',
    maintainer_email='nils.treuheit@ovgu.de',
    description='Monocular depth estimation (HYDEN-MoGe_v2) ROS2 node',
    license='FAIR-NC',
    entry_points={
        'console_scripts': [
            'depth_node = depth_ros2.depth_node:main',
        ],
    },
)
