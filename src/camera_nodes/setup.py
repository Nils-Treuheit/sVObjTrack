from setuptools import find_packages, setup

package_name = 'camera_nodes'

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
    description='USB and RealSense camera nodes for ROS2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'usb_camera = camera_nodes.usb_camera_node:main',
            'realsense_camera = camera_nodes.realsense_camera_node:main',
        ],
    },
)