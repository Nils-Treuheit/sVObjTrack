"""Orchestrator + Intel RealSense camera + viewer.

The orchestrator spawns YOLO/LocateAnything as internal subprocesses.
Pass model_id='rfdetr' to use RF-DETR instead of YOLO.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    model_id = LaunchConfiguration('model_id')

    return LaunchDescription([
        DeclareLaunchArgument('model_id',
                              default_value='[yolo26, yolo26-obb, yolo26-pose]',
                              description='Detection backend: YOLO model(s) or "rfdetr"'),

        Node(package='camera_nodes', executable='realsense_camera',
             name='realsense_camera', output='screen'),

        Node(package='orchestrator', executable='orchestrator_node',
             name='orchestrator_node', output='screen',
             parameters=[{'model_id': model_id}]),

        Node(package='viewer_ros2', executable='viewer_node',
             name='viewer_node', output='screen'),
    ])
