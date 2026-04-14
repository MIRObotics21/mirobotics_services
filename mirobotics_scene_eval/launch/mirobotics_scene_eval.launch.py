from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    image_publisher = LaunchConfiguration('image_publisher')
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_publisher',
            default_value='/camera/camera/color/image_raw',
            description='Publisher from which sub to image topic'
        ),
        Node(
            package='mirobotics_scene_eval',
            executable='scene_eval_server',
            name='scene_eval_server',
            output='screen',
            parameters=[
                {
                    'timeout_sec': 5.0,
                    'default_model_path': '/home/mirobotics/models/best.pt',
                    'min_confidence': 0.5,
                    'max_detections': 10,
                    'default_annotated_image_path': '',
                }
            ],
            remappings=[
                ('image', image_publisher),
            ],
        )
    ])