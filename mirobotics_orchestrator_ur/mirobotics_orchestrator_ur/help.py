"""
Usage help of mirobotics_orchestrator_ur.

    ros2 run mirobotics_orchestrator_ur help [topic]
"""

import sys


OVERVIEW = """\
mirobotics_orchestrator_ur - scene evaluation, voxel grid and robot motion in one pipeline

Node:        orchestrator_ur_server
Subscribes:  camera_info_topic (sensor_msgs/CameraInfo) for the RGB intrinsics
Actions:     /generate_scene   (mirobotics_msg/action/GenerateScene)
             /plan_and_execute (mirobotics_msg/action/PlanAndExecute)
Uses:        /eval_scene, /capture_scene, /check_points, MoveIt 2
RViz:        scene_object_markers (MarkerArray, Durability Transient Local)

Order: generate_scene first, then plan_and_execute.

Launch (also starts mirobotics_scene_eval and mirobotics_path_planner):
  ros2 launch mirobotics_orchestrator_ur mirobotics_orchestrator_ur.launch.py \\
    image_publisher:=/camera/camera/color/image_raw \\
    pointcloud2_publisher:=/camera/camera/depth/color/points

Package help: ros2 run mirobotics_scene_eval help
              ros2 run mirobotics_path_planner help
"""

TOPICS = {
    'generate_scene': """\
/generate_scene (mirobotics_msg/action/GenerateScene)
  1. eval_scene detects objects in the image
  2. capture_scene builds the voxel grid
  3. the ray through each detection is traced through the grid: the first occupied
     voxel is the object, the first free voxel above it is the approach voxel

  Goal:
    model_path      .pt or .onnx model, empty = default model of the scene evaluation
    voxel_size      edge length of a voxel in meters
    include_matrix  also return every voxel, default false (very long output)
  Result:
    success, error_msg
    json_matrix      {"voxel_size", "occupied_voxels", "grid": {...}},
                     with include_matrix also "matrix": [[id, x, y, z, passable], ...]
    json_objects_3d  [{"id", "type", "confidence", "object_voxel_id", "object_point",
                       "hit_point", "approach_voxel_id", "approach_point"}]
                     points [x, y, z] in the grid frame, null when nothing was hit
  Feedback:
    current_step, progress 0.0 - 1.0

  Example:
    ros2 action send_goal /generate_scene mirobotics_msg/action/GenerateScene \\
      "{model_path: '/home/mirobotics/models/best.onnx', voxel_size: 0.02}" --feedback

  approach_point can be used as goal_point of /plan_path.
""",
    'plan_and_execute': """\
/plan_and_execute (mirobotics_msg/action/PlanAndExecute)
  1. TCP target: hover_height above the object top, orientation tcp_rotation
  2. joints of the target by UR3 inverse kinematics, always the same arm configuration
  3. MoveIt 2 plans the joint move, the joint travel is checked
  4. the TCP along the plan is checked against the voxel grid (check_points)
  5. the move is executed unless dry_run

  Goal:
    object_id    id from the last generate_scene, target is hover_height above it
    goal_point   x, y, z TCP target in base_link, used when object_id is -1
  Result:
    success, error_msg
  Feedback:
    current_step, progress 0.0 - 1.0

  Examples:
    ros2 action send_goal /plan_and_execute mirobotics_msg/action/PlanAndExecute \\
      "{object_id: 1}" --feedback

    ros2 action send_goal /plan_and_execute mirobotics_msg/action/PlanAndExecute \\
      "{goal_point: {x: 0.21, y: 0.13, z: 0.13}}" --feedback

  Only the TCP is checked against the voxel grid, not the arm.
  RViz: PoseArray planned_tool0_poses shows the tool0 target (Durability Transient Local).
""",
    'parameters': """\
Parameters (ros2 param set /orchestrator_ur_server <name> <value>)
  camera_info_topic  /camera/camera/color/camera_info  CameraInfo of the RGB image
  ray_max_distance   3.0    Maximum distance from the camera searched for an object in meters
  service_timeout    30.0   Maximum wait for eval_scene, capture_scene, check_points in seconds
  publish_markers    true   Publish RViz markers of the localized objects
  hover_height       0.05   Height of the TCP above the top of the object in meters
  tcp_offset         0.1866 Distance from tool0 to the TCP along the tool axis in meters
  tcp_rotation       [pi, 0.0, 0.0]  Fixed rx, ry, rz as on the pendant, default tool down
  speed_scaling      0.1    Fraction of the maximum speed and acceleration
  motion_timeout     120.0  Maximum duration of one motion in seconds
  max_joint_travel   3.14   A plan is rejected if one joint travels further, in radians
  dry_run            false  Plan and publish everything but do not move the robot
""",
}

FOOTER = """\
Field details: ros2 interface show mirobotics_msg/action/GenerateScene
               ros2 interface show mirobotics_msg/action/PlanAndExecute
Help topics:   ros2 run mirobotics_orchestrator_ur help [{topics}]
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
