from setuptools import find_packages, setup

package_name = 'locate_anything_ros2'

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
    description='LocateAnything visual grounding ROS2 node with TensorRT',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'la_node = locate_anything_ros2.la_node:main',
        ],
    },
)
