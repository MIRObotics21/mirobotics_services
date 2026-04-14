from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pointcloud2_publisher = LaunchConfiguration('pointcloud2_publisher')
    return LaunchDescription([
        DeclareLaunchArgument(
            'pointcloud2_publisher',
            default_value='/camera/camera/depth/color/points',
            description='Publisher from which sub to pointcloud2 topic'
        ),
        Node(
            package='mirobotics_path_planner',  # change if different
            executable='path_planner_server',   # matches your filename without .py
            name='path_planner_server',
            output='screen',
            parameters=[],
            remappings=[
                ('pointcloud2', pointcloud2_publisher),
            ],
        )
    ])
