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
            package='mirobotics_path_planner',
            executable='path_planner_server',
            name='path_planner_server',
            output='screen',
            parameters=[{
                'planning_frame': 'base_link',
                # Workspace box [x, y, z] in planning_frame, in meters.
                # Example values, adjust to the real workcell. Set z of
                # workspace_min just below the table top to keep the table
                # as the bottom obstacle layer.
                'workspace_min': [-0.5, -0.5, -0.05],
                'workspace_max': [0.5, 0.5, 0.6],
                'min_points_per_voxel': 3,
                'fill_below_occupied': True,
            }],
            remappings=[
                ('pointcloud2', pointcloud2_publisher),
            ],
        )
    ])
