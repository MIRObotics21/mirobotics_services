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
/mirobotics_scene_eval/mirobotics_eval_scene \
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
| `publish_markers` | `true` | Publishes the voxel grid for RViz after each capture. |

The point cloud is transformed into `planning_frame` first, so the workspace box is given in robot coordinates.
Points outside the box are discarded and the voxel grid always covers the whole box.
Set `z` of `workspace_min` just below the table top to keep the table as the bottom obstacle layer.

### Visualization

After each capture the occupied voxels and the workspace box are published on
`voxel_grid_markers` (`visualization_msgs/MarkerArray`).
In RViz add a `MarkerArray` display and set its Durability Policy to `Transient Local`.

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
 "grid": {
  "min_index": [-5, -5, 0],
  "shape": [10, 10, 6],
  "origin": [-0.5, -0.5, 0.0],
  "frame_id": "base_link"
 },
 "matrix": [...]
}
```

Matrix rows are `[id, x, y, z, passable]`, where `passable` is `1.0` for free and `0.0` for occupied voxels.

The grid is fixed by the workspace box and `voxel_size`, so voxel ids do not change between captures.
With `shape = [nx, ny, nz]` and voxel index `[ix, iy, iz]`:

```text
id     = (ix * ny + iy) * nz + iz
center = origin + (index + 0.5) * voxel_size
```

---

# PlanPath Service

### Usage

```bash
ros2 service call \
/plan_path \
mirobotics_msg/srv/PlanPath \
"{start_id: 500, goal_id: 2584}"
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
```

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

1. YOLO scene evaluation
2. point cloud voxelization
3. ray projection
4. object assignment into voxel matrix

### Usage

```bash
ros2 action send_goal \
/generate_scene \
mirobotics_msg/action/GenerateScene \
"{model_path: '/home/mirobotics/models/best.onnx',
voxel_size: 1}" --feedback
```

### Example Output

```text
Result:
success: true

json_objects_3d:
[
 {
  "id":1,
  "type":"zelena",
  "object_voxel_id":13,
  "approach_voxel_id":null
 },
 {
  "id":2,
  "type":"modra",
  "object_voxel_id":13,
  "approach_voxel_id":null
 }
]
```

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
2. transforms rays into robot base coordinates using TF
3. traverses the voxel matrix
4. assigns the first occupied voxel intersected by the ray

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

# Build

```bash
cd ~/ros2_ws

colcon build

source install/setup.bash
```

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