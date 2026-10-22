"""
Usage help of mirobotics_path_planner.

    ros2 run mirobotics_path_planner help [topic]
"""

import sys


OVERVIEW = """\
mirobotics_path_planner - voxel grid of the workspace and 3D A* path planning

Node:        path_planner_server (launch argument node_name)
Subscribes:  pointcloud2 (sensor_msgs/PointCloud2), set by launch argument pointcloud2_publisher
Services:    /capture_scene (mirobotics_msg/srv/CaptureScene)
             /plan_path     (mirobotics_msg/srv/PlanPath)
RViz:        voxel_grid_markers, planned_path_markers (MarkerArray, Durability Transient Local)

Order: call capture_scene first, plan_path plans on the grid of the last capture.

Launch:
  ros2 launch mirobotics_path_planner mirobotics_path_planner.launch.py \\
    pointcloud2_publisher:=/camera/camera/depth/color/points
"""

TOPICS = {
    'capture_scene': """\
/capture_scene (mirobotics_msg/srv/CaptureScene)
  Merges the last 3 point clouds, transforms them into planning_frame, crops them
  to the workspace box and builds the voxel grid. The grid is kept by the node.

  Request:
    voxel_size      edge length of a voxel in meters
    include_matrix  also return every voxel, default false (very long output)
  Response:
    success, error_msg
    json_matrix     {"voxel_size", "occupied_voxels", "grid": {...}}
                    with include_matrix also "matrix": [[id, x, y, z, passable], ...],
                    passable 1.0 = free, 0.0 = occupied

  Examples:
    ros2 service call /capture_scene mirobotics_msg/srv/CaptureScene "{voxel_size: 0.02}"

    ros2 service call /capture_scene mirobotics_msg/srv/CaptureScene \\
      "{voxel_size: 0.1, include_matrix: true}"

  Voxel ids are fixed by the workspace box and voxel_size:
    id = (ix * ny + iy) * nz + iz    center = origin + (index + 0.5) * voxel_size
""",
    'plan_path': """\
/plan_path (mirobotics_msg/srv/PlanPath)
  A* with 26 neighbours on the grid of the last capture_scene. Voxels near obstacles
  cost more (clearance_voxels, clearance_cost), so the path keeps its distance.

  Request (start and goal each by voxel id, or by point when the id is -1):
    start_id, goal_id         voxel ids, default -1
    start_point, goal_point   x, y, z in planning_frame in meters
  Response:
    success, error_msg
    json_path    every voxel of the path, columns [id, x, y, z, passable]
    waypoints    simplified path: start, corners, goal (yellow line in RViz)

  Examples:
    ros2 service call /plan_path mirobotics_msg/srv/PlanPath \\
      "{start_id: 500, goal_id: 2584}"

    ros2 service call /plan_path mirobotics_msg/srv/PlanPath \\
      "{start_point: {x: -0.3, y: -0.3, z: 0.4}, goal_point: {x: 0.21, y: 0.13, z: 0.13}}"

  An occupied point moves to the nearest free voxel within snap_distance.
  A point outside the workspace box is an error.
""",
    'parameters': """\
Parameters (ros2 param set /path_planner_server <name> <value>)
  planning_frame        ''                  Frame of the grid, empty = point cloud frame (launch: base_link)
  workspace_min         [-0.5, -0.5, 0.0]   Minimum corner of the workspace box in meters
  workspace_max         [0.5, 0.5, 0.6]     Maximum corner of the workspace box in meters
  min_points_per_voxel  3                   Points needed to mark a voxel occupied
  fill_below_occupied   true                Mark everything under the top surface occupied
  max_voxels            2000000             capture_scene fails for larger grids
  max_cloud_age         2.0                 Maximum age of the point clouds in seconds, 0 = no check
  publish_markers       true                Publish RViz markers
  clearance_voxels      2                   Clearance zone (2 * n + 1)^3 cube around occupied voxels
  clearance_cost        5.0                 Cost factor of the clearance zone, 1.0 = no clearance
  snap_distance         0.05                Snap distance for occupied start/goal points in meters
""",
}

FOOTER = """\
Field details: ros2 interface show mirobotics_msg/srv/CaptureScene
               ros2 interface show mirobotics_msg/srv/PlanPath
Help topics:   ros2 run mirobotics_path_planner help [{topics}]
"""


def main(args=None) -> None:
    args = sys.argv[1:] if args is None else args
    footer = FOOTER.format(topics='|'.join(TOPICS))

    if args:
        topic = args[0]

        if topic not in TOPICS:
            print(f'Unknown help topic "{topic}".\n')
            print(footer)
            sys.exit(1)

        print(TOPICS[topic])
        return

    print(OVERVIEW)
    for text in TOPICS.values():
        print(text)
    print(footer)


if __name__ == '__main__':
    main()
