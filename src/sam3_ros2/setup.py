from setuptools import find_packages, setup

package_name = 'sam3_ros2'

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
    description='SAM 3.1 zero-shot segmentation ROS2 node',
    license='SAM',
    entry_points={
        'console_scripts': [
            'sam3_node = sam3_ros2.sam3_node:main',
        ],
    },
)
