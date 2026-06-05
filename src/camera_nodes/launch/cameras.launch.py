from launch import LaunchDescription
from launch_ros.actions import Node
from sys import argv

def generate_launch_description():
    ''' Usage  
    ros2 launch camera_nodes cameras.launch.py 0   # - for WebCam
    ros2 launch camera_nodes cameras.launch.py 1   # - for Intel RealSense
    '''
    return LaunchDescription([
        # WebCam (USB,RGB)
        Node(
            package='camera_nodes',
            executable='usb_camera',
            name='usb_camera',
            output='screen'
        ) if len(argv)>1 and int(argv[1]) == 1  else\
        # Intel RealSense
        Node(
            package='camera_nodes',
            executable='realsense_camera',
            name='realsense_camera',
            output='screen'
        ),
    ])

