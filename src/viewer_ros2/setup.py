from setuptools import find_packages, setup

package_name = 'viewer_ros2'

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
    description='tkinter viewer for sVObjTrack detections and VQA',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'viewer_node = viewer_ros2.viewer_node:main',
        ],
    },
)
