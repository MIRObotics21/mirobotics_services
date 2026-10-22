import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


GridIndex = Tuple[int, int, int]


@dataclass
class SceneGrid:
    """Voxel grid parsed from the CaptureScene json_matrix."""

    voxel_size: float
    min_index: np.ndarray
    shape: Tuple[int, int, int]
    frame_id: str
    occupancy: np.ndarray

    def voxel_id(self, index: Sequence[int]) -> int:
        return (int(index[0]) * self.shape[1] + int(index[1])) * self.shape[2] + int(index[2])

    def center(self, index: Sequence[int]) -> List[float]:
        center = (np.asarray(index) + self.min_index + 0.5) * self.voxel_size
        return [round(float(c), 4) for c in center]


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)

    if norm == 0.0:
        raise ValueError('Cannot normalize zero-length vector')

    return vector / norm


def pixel_to_camera_ray(
    u: float,
    v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> np.ndarray:
    """
    Converts RGB pixel coordinate into a ray in camera optical frame.

    ROS optical frame convention:
        x -> right
        y -> down
        z -> forward
    """

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0

    return normalize_vector([x, y, z])


def parse_scene_grid(json_matrix: str) -> SceneGrid:
    """
    Expected format (CaptureScene of mirobotics_path_planner):
    {
        "voxel_size": 0.1,
        "grid": {"min_index": [...], "shape": [nx, ny, nz], "origin": [...], "frame_id": "base_link"},
        "matrix": [
            [id, x, y, z, passable],
            ...
        ]
    }
    Rows are ordered by id = (ix * ny + iy) * nz + iz.
    """

    data = json.loads(json_matrix)

    if not isinstance(data, dict) or 'matrix' not in data:
        raise ValueError('json_matrix must be a JSON object with key "matrix"')

    grid = data.get('grid')

    if grid is None:
        raise ValueError(
            'json_matrix has no "grid" description, '
            'CaptureScene of the current mirobotics_path_planner is required'
        )

    voxel_size = float(data['voxel_size'])
    shape = tuple(int(n) for n in grid['shape'])
    matrix = np.asarray(data['matrix'], dtype=np.float64)

    if matrix.shape != (shape[0] * shape[1] * shape[2], 5):
        raise ValueError(
            f'json_matrix has {matrix.shape[0]} rows, grid shape {list(shape)} '
            f'requires {shape[0] * shape[1] * shape[2]}'
        )

    return SceneGrid(
        voxel_size=voxel_size,
        min_index=np.asarray(grid['min_index'], dtype=np.int64),
        shape=shape,
        frame_id=str(grid.get('frame_id', '')),
        occupancy=(matrix[:, 4] < 0.5).reshape(shape),
    )


def parse_scene_objects(json_objects: str) -> List[Dict]:
    """
    Expected format:
    [
        {
            "id": 1,
            "type": "seda",
            "u": 463.36,
            "v": 263.84,
            "confidence": 0.98
        }
    ]
    """

    data = json.loads(json_objects)

    if not isinstance(data, list):
        raise ValueError('json_objects must contain a list')

    return data


def cast_ray(
    scene: SceneGrid,
    origin: Sequence[float],
    direction: Sequence[float],
    max_distance: float,
) -> Optional[Tuple[GridIndex, float]]:
    """
    Voxel traversal (Amanatides & Woo) along the ray until the first occupied voxel.

    The origin may lie outside the grid, the ray is first clipped to the grid box.
    Every voxel the ray passes through is visited exactly once, so no voxel is
    skipped regardless of its size.

    Returns:
        index of the first occupied voxel and the distance from origin to the
        point where the ray enters it, or None when nothing is hit within
        max_distance
    """

    origin = np.asarray(origin, dtype=np.float64)
    direction = normalize_vector(direction)
    voxel_size = scene.voxel_size
    shape = np.asarray(scene.shape)

    lower = scene.min_index * voxel_size
    upper = (scene.min_index + shape) * voxel_size

    # Clip the ray to the grid box (slab method)
    t_enter = 0.0
    t_exit = float(max_distance)

    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return None
            continue

        t1 = (lower[axis] - origin[axis]) / direction[axis]
        t2 = (upper[axis] - origin[axis]) / direction[axis]
        t_enter = max(t_enter, min(t1, t2))
        t_exit = min(t_exit, max(t1, t2))

    if t_enter > t_exit:
        return None

    entry = origin + direction * t_enter
    index = np.floor(entry / voxel_size).astype(np.int64) - scene.min_index
    index = np.clip(index, 0, shape - 1)

    step = np.zeros(3, dtype=np.int64)
    t_max = np.full(3, math.inf)
    t_delta = np.full(3, math.inf)

    for axis in range(3):
        if direction[axis] > 1e-12:
            step[axis] = 1
            boundary = (scene.min_index[axis] + index[axis] + 1) * voxel_size
        elif direction[axis] < -1e-12:
            step[axis] = -1
            boundary = (scene.min_index[axis] + index[axis]) * voxel_size
        else:
            continue

        t_max[axis] = (boundary - origin[axis]) / direction[axis]
        t_delta[axis] = voxel_size / abs(direction[axis])

    t = t_enter

    while True:
        if scene.occupancy[tuple(index)]:
            return (int(index[0]), int(index[1]), int(index[2])), float(t)

        axis = int(np.argmin(t_max))
        t = float(t_max[axis])

        if t > t_exit:
            return None

        index[axis] += step[axis]

        if index[axis] < 0 or index[axis] >= shape[axis]:
            return None

        t_max[axis] += t_delta[axis]


def find_approach_voxel(scene: SceneGrid, hit_index: Sequence[int]) -> Optional[GridIndex]:
    """
    First free voxel above the hit voxel in the same column.
    Returns None when the column is occupied up to the top of the grid.
    """

    ix, iy, iz = (int(i) for i in hit_index)
    free = np.flatnonzero(~scene.occupancy[ix, iy, iz + 1:])

    if free.size == 0:
        return None

    return ix, iy, iz + 1 + int(free[0])


def assign_objects_to_voxels(
    json_objects: str,
    scene: SceneGrid,
    camera_origin: Sequence[float],
    rotation_frame_camera: Sequence[Sequence[float]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    ray_max_distance: float = 3.0,
) -> str:
    """
    Assigns SceneEval 2D detections to occupied voxels.

    scene is the voxel grid from parse_scene_grid. camera_origin and
    rotation_frame_camera describe the camera optical frame in the frame of
    the voxel grid (scene.frame_id). The rotation matrix
    transforms vectors from camera optical frame into that frame.

    For every detection the ray through its pixel is traced through the grid.
    The first occupied voxel is the object voxel, the first free voxel above
    it is the approach voxel. Missing voxels are reported as None.
    """

    detections = parse_scene_objects(json_objects)
    rotation = np.asarray(rotation_frame_camera, dtype=np.float64)
    origin = np.asarray(camera_origin, dtype=np.float64)

    objects_3d = []

    for detection in detections:
        ray_camera = pixel_to_camera_ray(
            u=float(detection['u']),
            v=float(detection['v']),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        ray_frame = rotation @ ray_camera

        output_object = {
            'id': detection.get('id'),
            'type': detection.get('type'),
            'confidence': detection.get('confidence'),
            'object_voxel_id': None,
            'object_point': None,
            'hit_point': None,
            'approach_voxel_id': None,
            'approach_point': None,
        }

        hit = cast_ray(scene, origin, ray_frame, ray_max_distance)

        if hit is not None:
            hit_index, hit_distance = hit
            hit_point = origin + normalize_vector(ray_frame) * hit_distance

            output_object['object_voxel_id'] = scene.voxel_id(hit_index)
            output_object['object_point'] = scene.center(hit_index)
            output_object['hit_point'] = [round(float(c), 4) for c in hit_point]

            approach_index = find_approach_voxel(scene, hit_index)

            if approach_index is not None:
                output_object['approach_voxel_id'] = scene.voxel_id(approach_index)
                output_object['approach_point'] = scene.center(approach_index)

        objects_3d.append(output_object)

    return json.dumps(objects_3d)
