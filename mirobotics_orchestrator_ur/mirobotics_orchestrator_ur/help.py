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
Uses:        /eval_scene, /capture_scene, /plan_path
RViz:        scene_object_markers (MarkerArray, Durability Transient Local)

Order: generate_scene first, plan_and_execute moves within its voxel grid.

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
  Moves the tool to a voxel of the last generate_scene with MoveIt 2.
  Being reworked for ur_ros_rtde, the interface will change.

  Goal:
    start_voxel_id, goal_voxel_id   voxel ids of the last generate_scene grid
  Result:
    success, error_msg
  Feedback:
    current_step, progress 0.0 - 1.0

  Example:
    ros2 action send_goal /plan_and_execute mirobotics_msg/action/PlanAndExecute \\
      "{start_voxel_id: 1, goal_voxel_id: 13}" --feedback
""",
    'parameters': """\
Parameters (ros2 param set /orchestrator_ur_server <name> <value>)
  camera_info_topic  /camera/camera/color/camera_info  CameraInfo of the RGB image
  ray_max_distance   3.0    Maximum distance from the camera searched for an object in meters
  service_timeout    30.0   Maximum wait for eval_scene, capture_scene, plan_path in seconds
  publish_markers    true   Publish RViz markers of the localized objects
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
