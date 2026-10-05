import numpy as np

from mirobotics_path_planner.astar_3d import AStar3D
from mirobotics_path_planner.voxel_grid import (
    compute_grid,
    crop_to_workspace,
    fill_below_occupied,
    occupancy_to_matrix,
    quaternion_to_rotation_matrix,
    transform_points,
    voxelize,
)


WORKSPACE_MIN = [-0.5, -0.5, 0.0]
WORKSPACE_MAX = [0.5, 0.5, 0.6]


def test_compute_grid_is_fixed_and_aligned():
    min_index, shape = compute_grid(WORKSPACE_MIN, WORKSPACE_MAX, 0.1)

    assert min_index.tolist() == [-5, -5, 0]
    assert shape == (10, 10, 6)


def test_compute_grid_handles_float_noise():
    # 0.3 / 0.1 and 0.6 / 0.1 are not exact in floating point
    min_index, shape = compute_grid([0.3, 0.3, 0.3], [0.6, 0.6, 0.6], 0.1)

    assert min_index.tolist() == [3, 3, 3]
    assert shape == (3, 3, 3)


def test_compute_grid_covers_workspace_with_large_voxels():
    min_index, shape = compute_grid(WORKSPACE_MIN, WORKSPACE_MAX, 1.0)

    assert min_index.tolist() == [-1, -1, 0]
    assert shape == (2, 2, 1)


def test_compute_grid_rejects_invalid_input():
    for args in (
        (WORKSPACE_MIN, WORKSPACE_MAX, 0.0),
        (WORKSPACE_MAX, WORKSPACE_MIN, 0.1),
        ([0.0, 0.0], WORKSPACE_MAX, 0.1),
    ):
        try:
            compute_grid(*args)
        except ValueError:
            continue
        raise AssertionError(f'compute_grid accepted {args}')


def test_crop_removes_floor_and_outside_points():
    points = np.array([
        [0.0, 0.0, 0.3],     # inside
        [0.0, 0.0, -0.8],    # floor
        [1.2, 0.0, 0.3],     # outside workcell
    ])

    cropped = crop_to_workspace(points, WORKSPACE_MIN, WORKSPACE_MAX)

    assert cropped.tolist() == [[0.0, 0.0, 0.3]]


def test_voxelize_marks_expected_voxel():
    min_index, shape = compute_grid(WORKSPACE_MIN, WORKSPACE_MAX, 0.1)
    points = np.array([[0.02, 0.03, 0.25]])

    occupancy = voxelize(points, min_index, shape, 0.1)

    assert occupancy.sum() == 1
    assert occupancy[5, 5, 2]


def test_voxelize_min_points_filters_noise():
    min_index, shape = compute_grid(WORKSPACE_MIN, WORKSPACE_MAX, 0.1)
    points = np.array([
        [0.02, 0.03, 0.25],
        [0.03, 0.02, 0.26],
        [0.04, 0.04, 0.24],
        [-0.42, -0.42, 0.55],   # single stray point
    ])

    occupancy = voxelize(points, min_index, shape, 0.1, min_points_per_voxel=3)

    assert occupancy.sum() == 1
    assert occupancy[5, 5, 2]


def test_voxelize_ignores_points_outside_grid():
    min_index, shape = compute_grid(WORKSPACE_MIN, WORKSPACE_MAX, 0.1)
    points = np.array([[2.0, 0.0, 0.1], [0.0, 0.0, -3.0]])

    assert not voxelize(points, min_index, shape, 0.1).any()
    assert not voxelize(np.empty((0, 3)), min_index, shape, 0.1).any()


def test_fill_below_occupied():
    occupancy = np.zeros((2, 2, 4), dtype=bool)
    occupancy[0, 0, 2] = True

    filled = fill_below_occupied(occupancy)

    assert filled[0, 0].tolist() == [True, True, True, False]
    assert not filled[1, 1].any()


def test_occupancy_to_matrix_layout():
    min_index, shape = compute_grid(WORKSPACE_MIN, WORKSPACE_MAX, 0.1)
    occupancy = np.zeros(shape, dtype=bool)
    occupancy[5, 5, 2] = True

    matrix = occupancy_to_matrix(occupancy, min_index, 0.1)

    assert matrix.shape == (600, 5)
    assert matrix[:, 0].tolist() == list(range(600))

    occupied_id = (5 * 10 + 5) * 6 + 2
    assert np.allclose(matrix[occupied_id], [occupied_id, 0.05, 0.05, 0.25, 0.0])
    assert matrix[:, 4].sum() == 599

    # the voxel above differs only in z and has the next id
    assert np.allclose(matrix[occupied_id + 1], [occupied_id + 1, 0.05, 0.05, 0.35, 1.0])


def test_transform_points_uses_camera_mount():
    # camera_link mounted above the workcell, rotated 90 deg around y
    rotation = quaternion_to_rotation_matrix(0.0, 0.707, 0.0, 0.707)
    points = np.array([[0.5, 0.0, 0.0]])

    transformed = transform_points(points, rotation, [0.2, 0.12, 0.88])

    # camera x axis points down in base_link
    assert np.allclose(transformed, [[0.2, 0.12, 0.38]], atol=1e-6)


def test_matrix_is_plannable_around_obstacle():
    min_index, shape = compute_grid(WORKSPACE_MIN, WORKSPACE_MAX, 0.1)

    # wall across the whole y range, 3 voxels high
    wall = np.array([
        [0.05, y, z]
        for y in np.arange(-0.45, 0.5, 0.1)
        for z in (0.05, 0.15, 0.25)
    ])

    occupancy = fill_below_occupied(voxelize(wall, min_index, shape, 0.1))
    matrix = np.round(occupancy_to_matrix(occupancy, min_index, 0.1), decimals=4)

    start_id = (2 * 10 + 5) * 6 + 0
    goal_id = (8 * 10 + 5) * 6 + 0

    path = AStar3D(matrix, 0.1).plan(start_id, goal_id)

    assert path[0][0] == start_id
    assert path[-1][0] == goal_id
    assert all(row[4] == 1.0 for row in path)
    assert max(row[3] for row in path) > 0.3
