"""RF-DETR detection + Intel RealSense camera + viewer.

Runs RF-DETR in a separate venv via subprocess to avoid
conflicting package versions with the main workspace.
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

_RFDETR_VENV = '/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/rfdetr/.venv'


def generate_launch_description():
    model_size = LaunchConfiguration('model_size')
    conf = LaunchConfiguration('conf_threshold')

    env = os.environ.copy()
    env['PYTHONPATH'] = f'{_RFDETR_VENV}/lib/python3.10/site-packages:{env.get("PYTHONPATH", "")}'

    return LaunchDescription([
        DeclareLaunchArgument('model_size', default_value='nano',
                              description='RF-DETR size: nano, small, medium, large'),
        DeclareLaunchArgument('conf_threshold', default_value='0.5',
                              description='Detection confidence threshold'),

        Node(package='camera_nodes', executable='realsense_camera',
             name='realsense_camera', output='screen'),

        Node(package='rfdetr_ros2', executable='rfdetr_node',
             name='rfdetr_node', output='screen', env=env,
             parameters=[{
                 'model_size': model_size,
                 'conf_threshold': conf,
             }]),

        Node(package='viewer_ros2', executable='viewer_node',
             name='viewer_node', output='screen'),
    ])
