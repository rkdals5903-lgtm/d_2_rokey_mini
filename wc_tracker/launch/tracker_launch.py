from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='wc_tracker',
            executable='yolo_webcam_processor',
            name='yolo_webcam_processor',
            output='screen'
        ),
        Node(
            package='wc_tracker',
            executable='webcam_to_map_node',
            name='webcam_to_map_node',
            output='screen'
        ),
    ])