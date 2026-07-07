from launch import LaunchDescription
from launch_ros.actions import Node
from sys import argv

def generate_launch_description():
    ''' Usage  
    ros2 launch camera_nodes cameras.launch.py 0   # - for WebCam (USB, RGB)
    ros2 launch camera_nodes cameras.launch.py 1   # - for Intel RealSense
    '''
    nodes = []
    if len(argv) > 1:
        if int(argv[1]) == 0:
            nodes.append(Node(
                package='camera_nodes',
                executable='usb_camera',
                name='usb_camera',
                output='screen',
            ))
        elif int(argv[1]) == 1:
            nodes.append(Node(
                package='camera_nodes',
                executable='realsense_camera',
                name='realsense_camera',
                output='screen',
            ))
    return LaunchDescription(nodes)

