"""LocateAnything VQA + USB webcam + viewer + depth estimation."""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch import conditions

_LA_PROJECT = '/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/locate_anything'
_TRT_LIBS = (
    f'{_LA_PROJECT}/model/tensorRT/.venv/lib/python3.10/site-packages/tensorrt_libs:'
    f'{os.path.expanduser("~")}/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:'
    '/usr/local/cuda-12.8/lib64'
)


def generate_launch_description():
    existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld = f'{_TRT_LIBS}:{existing_ld}' if existing_ld else _TRT_LIBS
    use_depth = LaunchConfiguration('use_depth')

    return LaunchDescription([
        SetEnvironmentVariable('LD_LIBRARY_PATH', new_ld),

        DeclareLaunchArgument('use_depth', default_value='true',
                              description='Enable monocular depth estimation'),

        Node(package='camera_nodes', executable='usb_camera',
             name='usb_camera', output='screen'),

        Node(package='depth_ros2', executable='depth_node',
             name='depth_node', output='screen',
             condition=conditions.IfCondition(use_depth)),

        Node(package='locate_anything_ros2', executable='la_node',
             name='la_node', output='screen',
             parameters=[{
                 'query_only': False,
                 'debug': True,
             }]),

        Node(package='viewer_ros2', executable='viewer_node',
             name='viewer_node', output='screen'),
    ])
