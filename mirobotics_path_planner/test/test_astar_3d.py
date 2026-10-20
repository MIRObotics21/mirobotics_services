import math

import numpy as np

from mirobotics_path_planner.astar_3d import AStar3D, dilate_cube


def empty_grid(shape=(10, 10, 10)):
    return np.zeros(shape, dtype=bool)


def path_length(path):
    return sum(
        math.dist(a, b) for a, b in zip(path[:-1], path[1:])
    )


def assert_valid_path(occupancy, path, start, goal):
    assert path[0] == tuple(start)
    assert path[-1] == tuple(goal)

    for a, b in zip(path[:-1], path[1:]):
        step = np.abs(np.subtract(a, b))
        assert step.max() == 1, (a, b)
        assert not occupancy[b], b


def raises_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return True
    return False


def test_dilate_cube_marks_cube():
    occupancy = empty_grid((7, 7, 7))
    occupancy[3, 3, 3] = True

    dilated = dilate_cube(occupancy, 2)

    assert dilated.sum() == 5 * 5 * 5
    assert dilated[1:6, 1:6, 1:6].all()
    assert not dilated[0].any()


def test_dilate_cube_clipped_at_border():
    occupancy = empty_grid((4, 4, 4))
    occupancy[0, 0, 0] = True

    assert dilate_cube(occupancy, 2).sum() == 3 * 3 * 3
    assert dilate_cube(occupancy, 0).sum() == 1


def test_free_space_diagonal_uses_26_neighbours():
    occupancy = empty_grid()
    planner = AStar3D(occupancy)

    path = planner.plan((0, 0, 0), (9, 9, 9))

    assert_valid_path(occupancy, path, (0, 0, 0), (9, 9, 9))
    # straight 3D diagonal: 9 moves of length sqrt(3)
    assert len(path) == 10
    assert math.isclose(path_length(path), 9 * math.sqrt(3))


def test_path_goes_around_wall():
    occupancy = empty_grid()
    occupancy[5, :, :7] = True     # wall with free space above z = 6
    planner = AStar3D(occupancy, clearance_voxels=0)

    path = planner.plan((2, 5, 0), (8, 5, 0))

    assert_valid_path(occupancy, path, (2, 5, 0), (8, 5, 0))
    assert max(index[2] for index in path) >= 7


def test_no_corner_cutting():
    occupancy = empty_grid((3, 3, 1))
    occupancy[1, 0, 0] = True
    occupancy[0, 1, 0] = True
    planner = AStar3D(occupancy, clearance_voxels=0)

    # the only connection is the diagonal squeezing between two occupied voxels
    assert planner.plan((0, 0, 0), (1, 1, 0)) == []


def test_unreachable_goal_returns_empty_path():
    occupancy = empty_grid()
    occupancy[5, :, :] = True
    planner = AStar3D(occupancy)

    assert planner.plan((0, 0, 0), (9, 9, 9)) == []


def test_occupied_or_outside_endpoints_raise():
    occupancy = empty_grid()
    occupancy[0, 0, 0] = True
    planner = AStar3D(occupancy)

    assert raises_value_error(planner.plan, (0, 0, 0), (5, 5, 5))
    assert raises_value_error(planner.plan, (5, 5, 5), (0, 0, 0))
    assert raises_value_error(planner.plan, (5, 5, 5), (10, 5, 5))
    assert raises_value_error(planner.plan, (-1, 5, 5), (5, 5, 5))


def test_invalid_parameters_raise():
    occupancy = empty_grid()

    assert raises_value_error(AStar3D, occupancy, -1)
    assert raises_value_error(AStar3D, occupancy, 2, 0.5)
    assert raises_value_error(AStar3D, np.zeros((0, 0, 0), dtype=bool))


def test_clearance_keeps_path_away_from_obstacle():
    occupancy = empty_grid((20, 20, 3))
    occupancy[8:12, 8:12, :] = True     # pillar in the middle
    start, goal = (1, 9, 1), (18, 10, 1)

    close = AStar3D(occupancy, clearance_voxels=2, clearance_cost=1.0).plan(start, goal)
    clear = AStar3D(occupancy, clearance_voxels=2, clearance_cost=5.0).plan(start, goal)

    def closest_distance(path):
        return min(
            max(abs(i - 9.5) - 1.5, abs(j - 9.5) - 1.5)
            for i, j, _ in path
        )

    assert_valid_path(occupancy, close, start, goal)
    assert_valid_path(occupancy, clear, start, goal)
    assert closest_distance(close) <= 1.0
    assert closest_distance(clear) >= 2.5


def test_goal_inside_clearance_zone_is_reachable():
    occupancy = empty_grid()
    occupancy[5, 5, :3] = True          # object, approach voxel directly above
    planner = AStar3D(occupancy, clearance_voxels=2, clearance_cost=5.0)

    path = planner.plan((0, 0, 9), (5, 5, 3))

    assert_valid_path(occupancy, path, (0, 0, 9), (5, 5, 3))


def test_simplify_free_space_to_two_points():
    occupancy = empty_grid()
    planner = AStar3D(occupancy)

    path = planner.plan((0, 0, 0), (9, 4, 7))

    assert planner.simplify(path) == [(0, 0, 0), (9, 4, 7)]


def test_simplify_keeps_corners_around_wall():
    occupancy = empty_grid()
    occupancy[5, :, :7] = True
    planner = AStar3D(occupancy, clearance_voxels=0)

    path = planner.plan((2, 5, 0), (8, 5, 0))
    simplified = planner.simplify(path)

    assert simplified[0] == (2, 5, 0)
    assert simplified[-1] == (8, 5, 0)
    assert 3 <= len(simplified) < len(path)

    # every simplified segment stays clear of the wall
    for a, b in zip(simplified[:-1], simplified[1:]):
        for t in np.linspace(0.0, 1.0, 50):
            point = np.rint(np.add(a, t * np.subtract(b, a))).astype(int)
            assert not occupancy[tuple(point)], (a, b, point)


def test_simplify_does_not_cut_through_clearance_zone():
    occupancy = empty_grid((20, 20, 3))
    occupancy[8:12, 8:12, :] = True
    planner = AStar3D(occupancy, clearance_voxels=2, clearance_cost=5.0)

    path = planner.plan((1, 9, 1), (18, 10, 1))
    simplified = planner.simplify(path)

    assert not any(planner.clearance_zone[index] for index in path)

    for a, b in zip(simplified[:-1], simplified[1:]):
        for t in np.linspace(0.0, 1.0, 100):
            point = tuple(np.rint(np.add(a, t * np.subtract(b, a))).astype(int))
            assert not planner.clearance_zone[point], (a, b, point)


def test_nearest_free():
    occupancy = empty_grid()
    occupancy[4:7, 4:7, 0:5] = True
    planner = AStar3D(occupancy)

    # free voxel is returned unchanged
    assert planner.nearest_free((0, 0, 0), 2) == (0, 0, 0)
    # occupied top voxel moves one voxel up rather than sideways
    assert planner.nearest_free((5, 5, 4), 2) == (5, 5, 5)
    # nothing free within one voxel of the center of the block
    assert planner.nearest_free((5, 5, 2), 1) is None
    # outside the grid
    assert planner.nearest_free((10, 0, 0), 2) is None
