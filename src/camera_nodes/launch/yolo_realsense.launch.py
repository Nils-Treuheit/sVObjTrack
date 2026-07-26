"""YOLO detection + Intel RealSense camera + viewer."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    model_id = LaunchConfiguration('model_id')
    conf = LaunchConfiguration('conf_threshold')

    return LaunchDescription([
        DeclareLaunchArgument('model_id', default_value='yolo26',
                              description='YOLO model shortcut or path'),
        DeclareLaunchArgument('conf_threshold', default_value='0.4',
                              description='Detection confidence threshold'),

        Node(package='camera_nodes', executable='realsense_camera',
             name='realsense_camera', output='screen'),

        Node(package='yolo_ros2', executable='yolo_node',
             name='yolo_node', output='screen',
             parameters=[{
                 'model_id': model_id,
                 'conf_threshold': conf,
             }]),

        Node(package='viewer_ros2', executable='viewer_node',
             name='viewer_node', output='screen'),
    ])
