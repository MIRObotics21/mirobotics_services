from types import SimpleNamespace

from mirobotics_orchestrator_ur.moveit_execution import (
    find_object_target,
    largest_joint_travel,
    rescale_trajectory,
    tcp_to_tool0_position,
)


OBJECTS = [
    {'id': 1, 'approach_point': [0.21, 0.13, 0.11]},
    {'id': 2, 'approach_point': None},
]


def raises_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return True
    return False


def trajectory_point(sec, nanosec, velocities, accelerations):
    return SimpleNamespace(
        time_from_start=SimpleNamespace(sec=sec, nanosec=nanosec),
        velocities=velocities,
        accelerations=accelerations,
    )


def test_tool0_is_above_tcp_with_tool_down():
    assert tcp_to_tool0_position((0.2, 0.1, 0.05), 0.1866) == (0.2, 0.1, 0.05 + 0.1866)


def test_find_object_target_returns_approach_point():
    assert find_object_target(OBJECTS, 1) == (0.21, 0.13, 0.11)


def test_find_object_target_rejects_unknown_and_unreachable_objects():
    assert raises_value_error(find_object_target, OBJECTS, 7)
    assert raises_value_error(find_object_target, OBJECTS, 2)
    assert raises_value_error(find_object_target, [], 1)


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


def test_rescale_trajectory_slows_down_motion():
    point = trajectory_point(1, 500000000, [1.0, -2.0], [4.0])

    rescale_trajectory([point], 0.1)

    assert point.time_from_start.sec == 15
    assert point.time_from_start.nanosec == 0
    assert point.velocities == [0.1, -0.2]
    assert abs(point.accelerations[0] - 0.04) < 1e-12


def test_rescale_trajectory_keeps_nanoseconds_in_range():
    point = trajectory_point(0, 900000000, [], [])

    rescale_trajectory([point], 0.3)

    assert point.time_from_start.sec == 3
    assert point.time_from_start.nanosec == 0


def test_rescale_trajectory_rejects_invalid_scaling():
    assert raises_value_error(rescale_trajectory, [], 0.0)
    assert raises_value_error(rescale_trajectory, [], 1.5)
