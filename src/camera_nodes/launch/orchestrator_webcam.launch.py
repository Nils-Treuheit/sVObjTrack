"""Orchestrator + USB webcam + viewer + depth estimation.

The orchestrator spawns YOLO/LocateAnything as internal subprocesses.
Pass model_id='rfdetr' to use RF-DETR instead of YOLO.
The depth_ros2 node provides monocular depth for the YOLO node's 3D projections.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import conditions


def generate_launch_description():
    model_id = LaunchConfiguration('model_id')
    use_depth = LaunchConfiguration('use_depth')

    return LaunchDescription([
        DeclareLaunchArgument('model_id',
                              default_value='[yolo26, yolo26-obb, yolo26-pose]',
                              description='Detection backend: YOLO model(s) or "rfdetr"'),
        DeclareLaunchArgument('use_depth',
                              default_value='true',
                              description='Enable monocular depth estimation (depth_ros2)'),

        Node(package='camera_nodes', executable='usb_camera',
             name='usb_camera', output='screen'),

        Node(package='depth_ros2', executable='depth_node',
             name='depth_node', output='screen',
             condition=conditions.IfCondition(use_depth)),

        Node(package='orchestrator', executable='orchestrator_node',
             name='orchestrator_node', output='screen',
             parameters=[{'model_id': model_id}]),

        Node(package='viewer_ros2', executable='viewer_node',
             name='viewer_node', output='screen'),
    ])
