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

YOLO-based scene evaluation package.

### Features

* RGB image acquisition
* YOLO ONNX inference
* Object segmentation
* JSON object output

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

### Example Output

```text
success: true
json_objects:
[
 {"id":1,"type":"modra","u":715.6,"v":246.4},
 {"id":2,"type":"zelena","u":510.6,"v":96.2}
]
```

---

## 3. mirobotics_path_planner

Voxelization and 3D A* planning package.

### Features

* PointCloud2 voxelization
* Occupancy matrix generation
* 3D A* path planning
* JSON path output

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
 "matrix": [...]
}
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
