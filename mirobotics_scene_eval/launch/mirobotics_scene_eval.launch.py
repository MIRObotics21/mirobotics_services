from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    image_publisher = LaunchConfiguration('image_publisher')
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_publisher',
            default_value='image',
            description='Topic to use for scene evaluation image input'
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
                    'default_annotated_image_path': '/home/mirobotics/models/',
                }
            ],
            remappings=[
                ('image', image_publisher),
            ],
        )
    ])