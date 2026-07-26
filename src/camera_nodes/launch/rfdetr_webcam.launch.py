"""RF-DETR detection + USB webcam + viewer + depth estimation."""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import conditions

_RFDETR_VENV = '/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/rfdetr/.venv'


def generate_launch_description():
    model_size = LaunchConfiguration('model_size')
    conf = LaunchConfiguration('conf_threshold')
    use_depth = LaunchConfiguration('use_depth')

    env = os.environ.copy()
    env['PYTHONPATH'] = f'{_RFDETR_VENV}/lib/python3.10/site-packages:{env.get("PYTHONPATH", "")}'

    return LaunchDescription([
        DeclareLaunchArgument('model_size', default_value='nano',
                              description='RF-DETR size: nano, small, medium, large'),
        DeclareLaunchArgument('conf_threshold', default_value='0.5',
                              description='Detection confidence threshold'),
        DeclareLaunchArgument('use_depth', default_value='true',
                              description='Enable monocular depth estimation'),

        Node(package='camera_nodes', executable='usb_camera',
             name='usb_camera', output='screen'),

        Node(package='depth_ros2', executable='depth_node',
             name='depth_node', output='screen',
             condition=conditions.IfCondition(use_depth)),

        Node(package='rfdetr_ros2', executable='rfdetr_node',
             name='rfdetr_node', output='screen', env=env,
             parameters=[{
                 'model_size': model_size,
                 'conf_threshold': conf,
             }]),

        Node(package='viewer_ros2', executable='viewer_node',
             name='viewer_node', output='screen'),
    ])
