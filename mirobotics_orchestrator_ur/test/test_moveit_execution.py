import math
from types import SimpleNamespace

from mirobotics_orchestrator_ur.moveit_execution import (
    find_object_target,
    first_blocked_sample,
    hover_point,
    largest_joint_travel,
    sample_tcp_path,
)
from mirobotics_orchestrator_ur.ur_kinematics import tcp_position, UR_JOINT_NAMES


OBJECTS = [
    {'id': 1, 'approach_point': [0.21, 0.13, 0.11]},
    {'id': 2, 'approach_point': None},
]

HOME = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
TARGET = [-0.8, -1.56, 1.8, -1.81, -1.57, 0.7]


def raises_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return True
    return False


def test_find_object_target_returns_approach_point():
    assert find_object_target(OBJECTS, 1) == (0.21, 0.13, 0.11)


def test_find_object_target_rejects_unknown_and_unreachable_objects():
    assert raises_value_error(find_object_target, OBJECTS, 7)
    assert raises_value_error(find_object_target, OBJECTS, 2)
    assert raises_value_error(find_object_target, [], 1)


def test_hover_point_is_measured_from_the_object_top():
    x, y, z = hover_point((0.275, -0.155, 0.035), 0.01, 0.05)

    assert (x, y) == (0.275, -0.155)
    assert abs(z - 0.08) < 1e-9


def test_largest_joint_travel_finds_full_turn():
    points = [
        SimpleNamespace(positions=[0.0, 1.0]),
        SimpleNamespace(positions=[0.5, 7.0]),
        SimpleNamespace(positions=[0.2, 1.2]),
    ]

    name, travel = largest_joint_travel(['shoulder', 'wrist_3'], points)

    assert name == 'wrist_3'
    assert travel == 6.0


def test_largest_joint_travel_of_empty_trajectory():
    assert largest_joint_travel(['shoulder'], []) == ('', 0.0)


def test_sample_tcp_path_covers_both_ends_in_small_steps():
    points = [SimpleNamespace(positions=HOME), SimpleNamespace(positions=TARGET)]

    samples = sample_tcp_path(UR_JOINT_NAMES, points, 0.1866)

    assert samples[0] == tcp_position(HOME, 0.1866)
    assert samples[-1] == tcp_position(TARGET, 0.1866)
    assert all(math.dist(a, b) < 0.02 for a, b in zip(samples, samples[1:]))


def test_sample_tcp_path_follows_joint_names_of_the_trajectory():
    names = list(reversed(UR_JOINT_NAMES))
    points = [SimpleNamespace(positions=list(reversed(TARGET)))]

    assert sample_tcp_path(names, points, 0.1866) == [tcp_position(TARGET, 0.1866)]


def test_sample_tcp_path_of_empty_trajectory():
    assert sample_tcp_path(UR_JOINT_NAMES, [], 0.1866) == []


def test_first_blocked_sample_of_free_path():
    assert first_blocked_sample([False, False, False]) is None
    assert first_blocked_sample([]) is None


def test_first_blocked_sample_skips_voxels_of_the_robot_at_the_start():
    assert first_blocked_sample([True, True, False, False]) is None


def test_first_blocked_sample_finds_obstacle_after_the_start():
    assert first_blocked_sample([True, False, False, True, False]) == 3


def test_first_blocked_sample_rejects_path_that_stays_occupied():
    assert first_blocked_sample([True, True, True]) == 2
