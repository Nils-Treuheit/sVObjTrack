"""YOLO detection + USB webcam + viewer + depth estimation."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import conditions


def generate_launch_description():
    model_id = LaunchConfiguration('model_id')
    conf = LaunchConfiguration('conf_threshold')
    device = LaunchConfiguration('device_id')
    use_depth = LaunchConfiguration('use_depth')

    return LaunchDescription([
        DeclareLaunchArgument('model_id', default_value='yolo26',
                              description='YOLO model shortcut or path'),
        DeclareLaunchArgument('conf_threshold', default_value='0.4',
                              description='Detection confidence threshold'),
        DeclareLaunchArgument('device_id', default_value='0',
                              description='OpenCV VideoCapture device index'),
        DeclareLaunchArgument('use_depth', default_value='true',
                              description='Enable monocular depth estimation'),

        Node(package='camera_nodes', executable='usb_camera',
             name='usb_camera', output='screen',
             parameters=[{'device_id': device}]),

        Node(package='depth_ros2', executable='depth_node',
             name='depth_node', output='screen',
             condition=conditions.IfCondition(use_depth)),

        Node(package='yolo_ros2', executable='yolo_node',
             name='yolo_node', output='screen',
             parameters=[{
                 'model_id': model_id,
                 'conf_threshold': conf,
             }]),

        Node(package='viewer_ros2', executable='viewer_node',
             name='viewer_node', output='screen'),
    ])
