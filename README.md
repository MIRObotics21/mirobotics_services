# MIRObotics ROS 2 services

ROS 2 Humble framework for autonomous robotic manipulation using:

* YOLO-based object detection
* Voxelized environment reconstruction
* 3D A* path planning

Partial:
* MoveIt 2 motion execution

The framework consists of four ROS 2 packages:

* `mirobotics_msg`
* `mirobotics_scene_eval`
* `mirobotics_path_planner`
* `mirobotics_orchestrator_ur`

---

# Install & Build

```bash
cd ~/ros2_ws/src

git clone --branch bp_kyvala --single-branch https://github.com/MIRObotics21/mirobotics_services.git

cd ~/ros2_ws

colcon build

source install/setup.bash
```

---

# Help

Each package prints its services or actions, parameters and copy-paste examples:

```bash
ros2 run mirobotics_scene_eval help
ros2 run mirobotics_path_planner help
ros2 run mirobotics_orchestrator_ur help
```

A single topic, for example only the PlanPath service:

```bash
ros2 run mirobotics_path_planner help plan_path
```

Every request, response and feedback field is described in the interface definitions:

```bash
ros2 interface show mirobotics_msg/srv/PlanPath
```

---

# System Architecture

```text
RGB Image
    ↓
mirobotics_scene_eval
    ↓
Object detections (2D)

PointCloud2
    ↓
mirobotics_path_planner
    ↓
Voxel matrix

                ↓
mirobotics_orchestrator_ur
    ↓
Ray projection + object assignment
    ↓
3D object localization
    ↓
Path planning
    ↓
MoveIt 2 execution
    ↓
Universal Robots manipulator
```

---

# Packages

## 1. mirobotics_msg

ROS 2 interfaces package containing:

### Services

* `EvalScene.srv`
* `CaptureScene.srv`
* `PlanPath.srv`

### Actions

* `GenerateScene.action`
* `PlanAndExecute.action`

---

## 2. mirobotics_scene_eval

YOLO-based scene evaluation package. See the [preparation scripts](https://github.com/MIRObotics21/prep_scripts_for_scene_eval_service.git) repository to create a NN YOLO model.

### Features

* RGB image acquisition
* YOLO ONNX inference
* Object segmentation
* JSON object output

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `default_model_path` | `''` | Model used when the request has an empty `model_path`. Loaded at startup if the file exists. |
| `default_annotated_image_path` | `''` | Used when the request has an empty `annotated_image_path`. Empty disables saving. |
| `min_confidence` | `0.25` | Minimum detection confidence. |
| `max_detections` | `50` | Maximum number of detections per image. |
| `max_image_age` | `1.0` | Maximum age of the evaluated image in seconds. `0` disables the check. |

### Launch

```bash
ros2 launch mirobotics_scene_eval \
mirobotics_scene_eval.launch.py \
image_publisher:=/camera/camera/color/image_raw
```

### EvalScene Usage

```bash
ros2 service call \
/eval_scene \
mirobotics_msg/srv/EvalScene \
"{model_path: '/home/mirobotics/models/best.onnx',
annotated_image_path: '/home/mirobotics/output/'}"
```

`annotated_image_path` ending with an image extension (`.png`, `.jpg`, `.jpeg`, `.bmp`) is used as the file path.
Anything else is treated as a directory and the image is saved there as `annotated.png`.

### Example Output

```text
success: true
json_objects:
[
 {"id":1,"type":"modra","u":715.6,"v":246.4,"confidence":0.93},
 {"id":2,"type":"zelena","u":510.6,"v":96.2,"confidence":0.88}
]
```

---

## 3. mirobotics_path_planner

Voxelization and 3D A* planning package.

### Features

* PointCloud2 voxelization
* Workspace cropping and noise filtering
* Occupancy matrix generation
* 3D A* path planning
* JSON path output
* RViz visualization of the voxel grid

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `planning_frame` | `''` | Frame of the voxel grid. Empty keeps the point cloud frame. Launch file sets `base_link`. |
| `workspace_min` | `[-0.5, -0.5, 0.0]` | Minimum corner `[x, y, z]` of the workspace box in `planning_frame`, in meters. |
| `workspace_max` | `[0.5, 0.5, 0.6]` | Maximum corner `[x, y, z]` of the workspace box in `planning_frame`, in meters. |
| `min_points_per_voxel` | `3` | Voxel is occupied only if it contains at least this many points. |
| `fill_below_occupied` | `true` | Marks everything under the highest occupied voxel of a column as occupied. |
| `max_voxels` | `2000000` | CaptureScene fails if the grid would contain more voxels. |
| `max_cloud_age` | `2.0` | Maximum age of the buffered point clouds in seconds. `0` disables the check. |
| `publish_markers` | `true` | Publishes the voxel grid after each capture and the path after each plan for RViz. |
| `clearance_voxels` | `2` | PlanPath: free voxels inside a `(2 * clearance_voxels + 1)^3` cube around an occupied voxel form the clearance zone. `2` gives a 5x5x5 cube. |
| `clearance_cost` | `5.0` | PlanPath: entering a clearance zone voxel costs this many times more. `1.0` disables the clearance. |
| `snap_distance` | `0.05` | PlanPath: an occupied start or goal point is moved to the nearest free voxel within this distance in meters. |

The point cloud is transformed into `planning_frame` first, so the workspace box is given in robot coordinates.
Points outside the box are discarded and the voxel grid always covers the whole box.
Set `z` of `workspace_min` just below the table top to keep the table as the bottom obstacle layer.

### Visualization

After each capture the occupied voxels and the workspace box are published on
`voxel_grid_markers` (`visualization_msgs/MarkerArray`).
After each successful PlanPath call the full voxel path (thin blue line) and the
waypoints (thick yellow line with spheres) are published on `planned_path_markers`.
In RViz add a `MarkerArray` display per topic and set its Durability Policy to `Transient Local`.

### Launch

```bash
ros2 launch mirobotics_path_planner \
mirobotics_path_planner.launch.py \
pointcloud2_publisher:=/camera/camera/depth/color/points
```

---

# CaptureScene Service

### Usage

```bash
ros2 service call \
/capture_scene \
mirobotics_msg/srv/CaptureScene \
"{voxel_size: 0.1}"
```

### Example Output

```text
success: true
json_matrix:
{
 "voxel_size": 0.1,
 "occupied_voxels": 112,
 "grid": {
  "min_index": [-5, -5, 0],
  "shape": [10, 10, 6],
  "origin": [-0.5, -0.5, 0.0],
  "frame_id": "base_link"
 }
}
```

The response does not list the voxels by default, the full list floods the console.
Request it with `include_matrix: true` (GenerateScene does this for the ray projection):

```bash
ros2 service call \
/capture_scene \
mirobotics_msg/srv/CaptureScene \
"{voxel_size: 0.1, include_matrix: true}"
```

`json_matrix` then also contains `"matrix": [...]` with rows `[id, x, y, z, passable]`,
where `passable` is `1.0` for free and `0.0` for occupied voxels.
The grid itself is always kept by the path planner for PlanPath, independent of `include_matrix`.

The grid is fixed by the workspace box and `voxel_size`, so voxel ids do not change between captures.
With `shape = [nx, ny, nz]` and voxel index `[ix, iy, iz]`:

```text
id     = (ix * ny + iy) * nz + iz
center = origin + (index + 0.5) * voxel_size
```

---

# PlanPath Service

PlanPath plans on the voxel grid of the last CaptureScene call, so CaptureScene must be called first.

Start and goal are each given either by voxel id or by point in the planning frame.
A voxel id of `-1` (the default) means the corresponding point is used.
An occupied point is moved to the nearest free voxel within `snap_distance`; a point outside the workspace box is an error.

The planner is A* with 26-neighbour moves (costs `1`, `√2`, `√3`). Diagonal moves never cut a corner of an occupied voxel.
Moves into the clearance zone around obstacles cost `clearance_cost` times more, so the path keeps its distance
where possible but can still reach start and goal voxels close to obstacles.

### Usage

By voxel ids:

```bash
ros2 service call \
/plan_path \
mirobotics_msg/srv/PlanPath \
"{start_id: 500, goal_id: 2584}"
```

By points:

```bash
ros2 service call \
/plan_path \
mirobotics_msg/srv/PlanPath \
"{start_point: {x: -0.3, y: -0.3, z: 0.4},
goal_point: {x: 0.21, y: 0.13, z: 0.13}}"
```

Voxel ids depend on the workspace box and `voxel_size` used by the last CaptureScene call.

### Example Output

```text
success: true
json_path:
{
 "columns": ["id","x","y","z","passable"],
 "path": [...]
}
waypoints:
- x: -0.3
  y: -0.3
  z: 0.4
- x: 0.21
  y: 0.13
  z: 0.13
```

`json_path` contains every voxel of the path.
`waypoints` is the simplified path: the start, the points where the path changes direction and the goal.
A section of the path is replaced by a straight line only if the line stays clear of obstacles
and does not enter the clearance zone where the original path avoided it.
When a point was given, the exact point is used as the first or last waypoint instead of the voxel center
(a snapped goal uses the center of the free voxel).

---

## 4. mirobotics_orchestrator_ur

High-level orchestration package integrating:

* scene evaluation
* voxel reconstruction
* ray projection
* path planning
* MoveIt 2 execution

### Features

* object localization inside voxel matrix
* ray casting projection
* autonomous manipulation pipeline
* MoveIt 2 integration
* Universal Robots execution

---

# Launch Orchestrator

```bash
ros2 launch mirobotics_orchestrator_ur \
mirobotics_orchestrator_ur.launch.py \
image_publisher:=/camera/camera/color/image_raw \
pointcloud2_publisher:=/camera/camera/depth/color/points
```

---

# GenerateScene Action

The `GenerateScene` action performs:

1. YOLO scene evaluation (EvalScene)
2. point cloud voxelization (CaptureScene)
3. TF lookup of the camera in the frame of the voxel grid
4. ray projection and object assignment into the voxel grid

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `camera_info_topic` | `/camera/camera/color/camera_info` | CameraInfo of the RGB image used by EvalScene. |
| `ray_max_distance` | `3.0` | Maximum distance from the camera searched for an object, in meters. |
| `service_timeout` | `30.0` | Maximum time to wait for an EvalScene, CaptureScene or PlanPath response, in seconds. |
| `publish_markers` | `true` | Publishes the localized objects for RViz after each GenerateScene. |

### Visualization

After each GenerateScene the localized objects are published on `scene_object_markers`
(`visualization_msgs/MarkerArray`, Durability Policy `Transient Local` in RViz).
Markers of the previous call are deleted first.

| Namespace | Marker | Meaning |
|---|---|---|
| `object_voxels` | solid cube | Object voxel. Color per object type, assigned in order of first appearance. |
| `hit_points` | small white sphere | Exact point where the ray enters the object voxel. |
| `approach_voxels` | transparent green cube | Approach voxel, the first free voxel above the object. |
| `labels` | text | `id type confidence` above the object. |

### Usage

```bash
ros2 action send_goal \
/generate_scene \
mirobotics_msg/action/GenerateScene \
"{model_path: '/home/mirobotics/models/best.onnx',
voxel_size: 0.02}" --feedback
```

### Example Output

```text
Result:
success: true

json_matrix:
{"voxel_size": 0.02, "occupied_voxels": 10180,
 "grid": {"min_index": [-25, -25, -3], "shape": [50, 50, 33], "origin": [-0.5, -0.5, -0.06], "frame_id": "base_link"}}

json_objects_3d:
[
 {
  "id":1,
  "type":"zelena",
  "confidence":0.91,
  "object_voxel_id":58781,
  "object_point":[0.21,0.13,0.09],
  "hit_point":[0.2034,0.1281,0.1],
  "approach_voxel_id":58782,
  "approach_point":[0.21,0.13,0.11]
 }
]
```

Like CaptureScene, `json_matrix` lists every voxel only with `include_matrix: true` in the goal.
PlanAndExecute works either way, the orchestrator always keeps the full grid.

* `object_voxel_id`, `object_point`: first occupied voxel hit by the ray and its center
* `hit_point`: exact point where the ray enters that voxel
* `approach_voxel_id`, `approach_point`: first free voxel above the object voxel

All points are in the frame of the voxel grid (`planning_frame` of the path planner).
Values are `null` when the ray hits nothing or the column above the object is occupied
up to the top of the workspace box. `approach_point` can be passed directly as `goal_point` to PlanPath.

---

# PlanAndExecute Action

The `PlanAndExecute` action performs:

1. voxel path planning
2. target pose generation
3. MoveIt 2 planning
4. robot trajectory execution

### Usage

```bash
ros2 action send_goal \
/plan_and_execute \
mirobotics_msg/action/PlanAndExecute \
"{start_voxel_id: 1,
goal_voxel_id: 13}" --feedback
```

### Example Output

```text
Result:
success: true
error_msg: ''

Goal finished with status: SUCCEEDED
```

---

# Ray Projection

The framework uses ray projection to localize detected objects inside the voxelized environment.

The algorithm:

1. converts image coordinates `(u,v)` into camera rays
2. transforms rays into the frame of the voxel grid using TF
3. traverses the voxel grid voxel by voxel (Amanatides & Woo), so no voxel is skipped regardless of its size
4. assigns the first occupied voxel intersected by the ray
5. searches upwards from that voxel for the first free voxel, the approach voxel

This allows reliable 3D object localization without explicit depth segmentation.

---

# Dependencies

* ROS 2 Humble
* MoveIt 2
* Universal Robots ROS 2 Driver
* Intel RealSense ROS 2 Wrapper
* OpenCV
* NumPy
* ONNX Runtime
* TF2

---

# Tested Hardware

* Universal Robots manipulator
* OnRobot RG2 gripper
* Intel RealSense RGB-D camera

---

# Future Improvements

* dynamic obstacle updates
* grasp pose estimation
* full TCP integration into MoveIt
* multi-object task sequencing
* trajectory optimization
* grasp quality evaluation
* semantic scene understanding
* rcl_py