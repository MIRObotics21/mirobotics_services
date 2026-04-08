from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
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
                ('image', '/camera/camera/color/image_raw'),
            ],
        )
    ])