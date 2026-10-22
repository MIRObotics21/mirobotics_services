import json
import math

import numpy as np

from mirobotics_orchestrator_ur.ray_casting import (
    assign_objects_to_voxels,
    cast_ray,
    find_approach_voxel,
    parse_scene_grid,
    pixel_to_camera_ray,
)


# Grid of the CaptureScene workspace [-0.5, -0.5, 0.0] -> [0.5, 0.5, 0.6] with 0.1 m voxels
MIN_INDEX = [-5, -5, 0]
SHAPE = (10, 10, 6)
VOXEL_SIZE = 0.1

# Camera optical frame looking straight down: x -> x, y -> -y, z (forward) -> -z
ROTATION_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
FX = FY = 500.0
CX, CY = 320.0, 240.0


def make_json_matrix(occupancy, min_index=MIN_INDEX, voxel_size=VOXEL_SIZE, frame_id='base_link'):
    """Same layout as CaptureScene: rows ordered by id = (ix * ny + iy) * nz + iz."""
    rows = []

    for voxel_id, index in enumerate(np.ndindex(occupancy.shape)):
        center = (np.asarray(index) + np.asarray(min_index) + 0.5) * voxel_size
        rows.append([voxel_id, *center.tolist(), 0.0 if occupancy[index] else 1.0])

    return json.dumps({
        'voxel_size': voxel_size,
        'grid': {
            'min_index': list(min_index),
            'shape': list(occupancy.shape),
            'origin': (np.asarray(min_index) * voxel_size).tolist(),
            'frame_id': frame_id,
        },
        'matrix': rows,
    })


def scene_with_box():
    """Box occupying index (5, 5, 0..2): x, y in [0.0, 0.1], z in [0.0, 0.3]."""
    occupancy = np.zeros(SHAPE, dtype=bool)
    occupancy[5, 5, 0:3] = True
    return parse_scene_grid(make_json_matrix(occupancy))


def test_parse_scene_grid():
    scene = scene_with_box()

    assert scene.shape == SHAPE
    assert scene.min_index.tolist() == MIN_INDEX
    assert scene.frame_id == 'base_link'
    assert scene.occupancy.sum() == 3
    assert scene.occupancy[5, 5, 2]
    assert scene.voxel_id((5, 5, 2)) == (5 * 10 + 5) * 6 + 2
    assert scene.center((5, 5, 2)) == [0.05, 0.05, 0.25]


def test_parse_scene_grid_rejects_invalid_input():
    occupancy = np.zeros(SHAPE, dtype=bool)
    data = json.loads(make_json_matrix(occupancy))

    without_grid = dict(data)
    del without_grid['grid']

    short_matrix = dict(data)
    short_matrix['matrix'] = data['matrix'][:-1]

    for bad in (without_grid, short_matrix, [1, 2, 3]):
        try:
            parse_scene_grid(json.dumps(bad))
        except ValueError:
            continue
        raise AssertionError(f'parse_scene_grid accepted {str(bad)[:80]}')


def test_pixel_to_camera_ray():
    assert np.allclose(pixel_to_camera_ray(CX, CY, FX, FY, CX, CY), [0.0, 0.0, 1.0])

    ray = pixel_to_camera_ray(CX + 500.0, CY, FX, FY, CX, CY)
    assert np.allclose(ray, [1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)])


def test_cast_ray_from_above_hits_top_of_box():
    scene = scene_with_box()

    hit = cast_ray(scene, [0.05, 0.05, 1.0], [0.0, 0.0, -1.0], 3.0)

    assert hit is not None
    index, distance = hit
    assert index == (5, 5, 2)
    assert math.isclose(distance, 0.7)


def test_cast_ray_from_side_outside_grid():
    scene = scene_with_box()

    hit = cast_ray(scene, [-2.0, 0.05, 0.15], [1.0, 0.0, 0.0], 5.0)

    assert hit is not None
    index, distance = hit
    assert index == (5, 5, 1)
    assert math.isclose(distance, 2.0)


def test_cast_ray_misses():
    scene = scene_with_box()

    # beside the box, no floor in this grid
    assert cast_ray(scene, [0.25, 0.25, 1.0], [0.0, 0.0, -1.0], 3.0) is None
    # pointing away from the grid
    assert cast_ray(scene, [0.05, 0.05, 1.0], [0.0, 0.0, 1.0], 3.0) is None
    # box is further than max_distance
    assert cast_ray(scene, [0.05, 0.05, 1.0], [0.0, 0.0, -1.0], 0.5) is None


def test_cast_ray_does_not_skip_small_voxels():
    # 1 cm voxels, single occupied voxel on a diagonal ray (the old 2 cm ray steps could miss it)
    occupancy = np.zeros((50, 50, 50), dtype=bool)
    occupancy[30, 31, 30] = True
    scene = parse_scene_grid(make_json_matrix(occupancy, [0, 0, 0], 0.01))

    origin = [0.0, 0.0105, 0.0]
    direction = [1.0, 1.0, 1.0]

    hit = cast_ray(scene, origin, direction, 3.0)

    assert hit is not None
    assert hit[0] == (30, 31, 30)


def test_find_approach_voxel_searches_upwards():
    occupancy = np.zeros(SHAPE, dtype=bool)
    occupancy[5, 5, 0:3] = True
    occupancy[2, 2, :] = True
    scene = parse_scene_grid(make_json_matrix(occupancy))

    # hit on the side of the box, approach is above its top
    assert find_approach_voxel(scene, (5, 5, 0)) == (5, 5, 3)
    assert find_approach_voxel(scene, (5, 5, 2)) == (5, 5, 3)
    # column occupied up to the top of the grid
    assert find_approach_voxel(scene, (2, 2, 0)) is None


def test_assign_objects_to_voxels():
    scene = scene_with_box()

    detections = json.dumps([
        {'id': 1, 'type': 'modra', 'u': CX, 'v': CY, 'confidence': 0.9},
        # 0.2 m beside the box at 0.7 m below the camera, nothing there
        {'id': 2, 'type': 'zelena', 'u': CX + 500.0 * 0.2 / 0.7, 'v': CY, 'confidence': 0.8},
    ])

    objects = json.loads(assign_objects_to_voxels(
        json_objects=detections,
        scene=scene,
        camera_origin=[0.05, 0.05, 1.0],
        rotation_frame_camera=ROTATION_DOWN,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    ))

    assert objects[0] == {
        'id': 1,
        'type': 'modra',
        'confidence': 0.9,
        'object_voxel_id': (5 * 10 + 5) * 6 + 2,
        'object_point': [0.05, 0.05, 0.25],
        'hit_point': [0.05, 0.05, 0.3],
        'approach_voxel_id': (5 * 10 + 5) * 6 + 3,
        'approach_point': [0.05, 0.05, 0.35],
    }

    assert objects[1]['id'] == 2
    assert objects[1]['object_voxel_id'] is None
    assert objects[1]['object_point'] is None
    assert objects[1]['approach_voxel_id'] is None
