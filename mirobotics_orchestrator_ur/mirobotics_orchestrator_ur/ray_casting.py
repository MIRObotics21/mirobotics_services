import json
import math
from typing import Dict, List, Optional, Tuple


Voxel = Dict[str, float]
Detection = Dict[str, float]


def normalize_vector(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))

    if norm == 0.0:
        raise ValueError('Cannot normalize zero-length vector')

    return [v / norm for v in vector]


def pixel_to_camera_ray(
    u: float,
    v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> List[float]:
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


def parse_voxel_matrix(json_matrix: str) -> Tuple[float, List[Voxel]]:
    """
    Expected format:
    {
        "voxel_size": 0.1,
        "matrix": [
            [id, x, y, z, passable],
            ...
        ]
    }
    """

    data = json.loads(json_matrix)

    voxel_size = float(data.get('voxel_size', 0.1))
    raw_matrix = data.get('matrix', [])

    voxels = []

    for row in raw_matrix:
        if len(row) < 5:
            continue

        voxels.append({
            'id': int(row[0]),
            'x': float(row[1]),
            'y': float(row[2]),
            'z': float(row[3]),
            'passable': float(row[4]),
        })

    return voxel_size, voxels


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


def find_first_hit_voxel(
    camera_origin_base: List[float],
    ray_direction_base: List[float],
    voxels: List[Voxel],
    voxel_size: float,
    ray_step: float,
    ray_max_distance: float,
    occupied_passable_value: float = 0.0
) -> Optional[Voxel]:
    """
    Simple ray marching.

    At each ray point, find voxel whose center is closest and within half voxel size.
    The first voxel with passable == occupied_passable_value is returned.
    """

    half_voxel = voxel_size / 2.0

    occupied_voxels = [
        voxel for voxel in voxels
        if voxel['passable'] == occupied_passable_value
    ]

    if not occupied_voxels:
        return None

    distance = 0.0

    while distance <= ray_max_distance:
        px = camera_origin_base[0] + distance * ray_direction_base[0]
        py = camera_origin_base[1] + distance * ray_direction_base[1]
        pz = camera_origin_base[2] + distance * ray_direction_base[2]

        for voxel in occupied_voxels:
            if (
                abs(px - voxel['x']) <= half_voxel and
                abs(py - voxel['y']) <= half_voxel and
                abs(pz - voxel['z']) <= half_voxel
            ):
                return voxel

        distance += ray_step

    return None


def assign_objects_to_voxels(
    json_objects: str,
    json_matrix: str,
    camera_origin_base: List[float],
    rotation_base_camera: List[List[float]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    ray_step: float = 0.02,
    ray_max_distance: float = 3.0,
    occupied_passable_value: float = 0.0
) -> str:
    """
    Assigns SceneEval 2D detections to occupied voxels.

    rotation_base_camera is a 3x3 rotation matrix that transforms vectors from
    camera optical frame into base_link frame.
    """

    voxel_size, voxels = parse_voxel_matrix(json_matrix)
    detections = parse_scene_objects(json_objects)

    assigned_voxel_ids = []
    objects_3d = []

    for detection in detections:
        u = float(detection['u'])
        v = float(detection['v'])

        ray_camera = pixel_to_camera_ray(
            u=u,
            v=v,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        ray_base = [
            rotation_base_camera[0][0] * ray_camera[0] +
            rotation_base_camera[0][1] * ray_camera[1] +
            rotation_base_camera[0][2] * ray_camera[2],

            rotation_base_camera[1][0] * ray_camera[0] +
            rotation_base_camera[1][1] * ray_camera[1] +
            rotation_base_camera[1][2] * ray_camera[2],

            rotation_base_camera[2][0] * ray_camera[0] +
            rotation_base_camera[2][1] * ray_camera[1] +
            rotation_base_camera[2][2] * ray_camera[2],
        ]

        ray_base = normalize_vector(ray_base)

        hit_voxel = find_first_hit_voxel(
            camera_origin_base=camera_origin_base,
            ray_direction_base=ray_base,
            voxels=voxels,
            voxel_size=voxel_size,
            ray_step=ray_step,
            ray_max_distance=ray_max_distance,
            occupied_passable_value=occupied_passable_value
        )

        approach_voxel = None

        if hit_voxel is not None:
            approach_voxel = find_voxel_above(
                hit_voxel=hit_voxel,
                voxels=voxels,
                voxel_size=voxel_size,
                z_steps=1
            )

        output_object = {
            'id': detection.get('id'),
            'type': detection.get('type'),
            'object_voxel_id': None,
            'approach_voxel_id': None,
        }

        if hit_voxel is not None:
            output_object['object_voxel_id'] = hit_voxel['id']

        if approach_voxel is not None:
            output_object['approach_voxel_id'] = approach_voxel['id']

        objects_3d.append(output_object)

    updated_json_matrix = set_voxel_passable(
        json_matrix=json_matrix,
        voxel_ids_to_update=assigned_voxel_ids,
        new_passable_value=1.0
    )

    return json.dumps(objects_3d), updated_json_matrix

def set_voxel_passable(
    json_matrix: str,
    voxel_ids_to_update: List[int],
    new_passable_value: float = 0.0 #remove in future pls - used different method
) -> str:
    """
    Updates passable value of selected voxels inside json_matrix.
    """

    data = json.loads(json_matrix)

    matrix = data.get('matrix', [])

    voxel_ids_set = set(voxel_ids_to_update)

    for row in matrix:
        if len(row) < 5:
            continue

        voxel_id = int(row[0])

        if voxel_id in voxel_ids_set:
            row[4] = float(new_passable_value)

    return json.dumps(data)

def find_voxel_above(hit_voxel, voxels, voxel_size, z_steps=1):
    target_x = hit_voxel['x']
    target_y = hit_voxel['y']
    target_z = hit_voxel['z'] + voxel_size * z_steps

    tolerance = voxel_size * 0.25

    for voxel in voxels:
        if (
            abs(voxel['x'] - target_x) <= tolerance and
            abs(voxel['y'] - target_y) <= tolerance and
            abs(voxel['z'] - target_z) <= tolerance
        ):
            return voxel

    return None