from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    image_publisher = LaunchConfiguration('image_publisher')
    pointcloud2_publisher = LaunchConfiguration('pointcloud2_publisher')
    camera_info_topic = LaunchConfiguration('camera_info_topic')

    scene_eval_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('mirobotics_scene_eval'),
                'launch',
                'mirobotics_scene_eval.launch.py'
            )
        ),
        launch_arguments={
            'image_publisher': image_publisher,
        }.items()
    )

    path_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('mirobotics_path_planner'),
                'launch',
                'mirobotics_path_planner.launch.py'
            )
        ),
        launch_arguments={
            'pointcloud2_publisher': pointcloud2_publisher,
        }.items()
    )

    orchestrator_node = Node(
        package='mirobotics_orchestrator_ur',
        executable='orchestrator_ur_server',
        name='orchestrator_ur_server',
        output='screen',
        parameters=[{
            'camera_info_topic': camera_info_topic,
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'image_publisher',
            default_value='/camera/camera/color/image_raw',
            description='Image topic used by mirobotics_scene_eval'
        ),
        DeclareLaunchArgument(
            'pointcloud2_publisher',
            default_value='/camera/camera/depth/color/points',
            description='PointCloud2 topic used by mirobotics_path_planner'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera/camera/color/camera_info',
            description='CameraInfo topic used by orchestrator for RGB intrinsics'
        ),

        scene_eval_launch,
        path_planner_launch,
        orchestrator_node,
    ])