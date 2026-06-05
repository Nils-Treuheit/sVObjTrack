from setuptools import find_packages, setup

package_name = 'yolo11_ros2'

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
    description='YOLO11 object detection ROS2 node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'yolo_node = yolo11_ros2.yolo_node:main',
        ],
    },
)