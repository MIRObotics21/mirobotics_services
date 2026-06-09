from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pointcloud2_publisher = LaunchConfiguration('pointcloud2_publisher')
    return LaunchDescription([
        DeclareLaunchArgument(
            'node_name',
            default_value='path_planner_server',
            description='ROS node name'
        ),
        DeclareLaunchArgument(
            'pointcloud2_publisher',
            default_value='/camera/camera/depth/color/points',
            description='Publisher from which sub to pointcloud2 topic'
        ),
        Node(
            package='mirobotics_path_planner',
            executable='path_planner_server',
            name=LaunchConfiguration('node_name'),
            output='screen',
            parameters=[{
                'planning_frame': 'base_link',
            }],
            remappings=[
                ('pointcloud2', pointcloud2_publisher),
            ],
        )
    ])
