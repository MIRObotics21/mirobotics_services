import math

import numpy as np

from mirobotics_orchestrator_ur.ur_kinematics import (
    forward_kinematics,
    inverse_kinematics,
    pendant_to_tool0_rotation,
    rotation_matrix_to_quaternion,
    rotation_vector_to_matrix,
    tcp_position,
    tool0_position,
)


HOME = [0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0]
TOOL_DOWN = pendant_to_tool0_rotation([math.pi, 0.0, 0.0])
TCP_OFFSET = 0.1866
OBJECTS = [(0.275, -0.155, 0.08), (0.315, -0.015, 0.07), (0.2, 0.25, 0.1), (-0.25, 0.1, 0.1)]


def test_forward_kinematics_matches_home_measured_on_the_robot():
    pose = forward_kinematics(HOME)

    assert np.allclose(pose[:3, 3], [0.0, 0.194, 0.694], atol=0.001)
    assert np.allclose(pose[:3, 2], [0.0, 1.0, 0.0], atol=1e-9)


def test_pendant_rx_pi_points_the_tool_down_in_base_link():
    assert np.allclose(TOOL_DOWN[:3, 2], [0.0, 0.0, -1.0])


def test_rotation_vector_of_zero_is_identity():
    assert np.array_equal(rotation_vector_to_matrix([0.0, 0.0, 0.0]), np.eye(3))


def test_quaternion_of_rotations_around_180_degrees():
    assert np.allclose(rotation_matrix_to_quaternion(TOOL_DOWN), [0.0, 1.0, 0.0, 0.0], atol=1e-7)
    assert np.allclose(
        rotation_matrix_to_quaternion(rotation_vector_to_matrix([math.pi, 0.0, 0.0])),
        [1.0, 0.0, 0.0, 0.0],
        atol=1e-7
    )
    assert np.allclose(
        rotation_matrix_to_quaternion(rotation_vector_to_matrix([0.0, 0.0, math.pi])),
        [0.0, 0.0, 1.0, 0.0],
        atol=1e-7
    )


def test_quaternion_of_small_rotation():
    half_angle = 0.2
    quaternion = rotation_matrix_to_quaternion(rotation_vector_to_matrix([0.0, 0.4, 0.0]))

    assert np.allclose(quaternion, [0.0, math.sin(half_angle), 0.0, math.cos(half_angle)])


def test_inverse_kinematics_reaches_the_tcp_targets():
    for target in OBJECTS:
        position = tool0_position(target, TOOL_DOWN, TCP_OFFSET)
        joints = inverse_kinematics(position, TOOL_DOWN)
        pose = forward_kinematics(joints)

        assert np.allclose(tcp_position(joints, TCP_OFFSET), target, atol=1e-6)
        assert np.allclose(pose[:3, :3], TOOL_DOWN, atol=1e-6)


def test_inverse_kinematics_keeps_the_arm_configuration():
    for target in OBJECTS:
        position = tool0_position(target, TOOL_DOWN, TCP_OFFSET)
        pan, lift, elbow, wrist_1, wrist_2, _ = inverse_kinematics(position, TOOL_DOWN)
        azimuth = math.atan2(target[1], target[0])

        assert abs(math.atan2(math.sin(pan - azimuth), math.cos(pan - azimuth))) < math.pi / 2.0
        assert -math.pi < lift < 0.0
        assert elbow > 0.0
        assert abs(wrist_2 + math.pi / 2.0) < 1e-6
        assert abs(lift + elbow + wrist_1 + math.pi / 2.0) < 1e-6


def test_inverse_kinematics_rejects_a_pose_out_of_reach():
    position = tool0_position((0.9, 0.0, 0.1), TOOL_DOWN, TCP_OFFSET)

    try:
        inverse_kinematics(position, TOOL_DOWN)
    except ValueError as exc:
        assert 'not reachable' in str(exc)
    else:
        raise AssertionError('Expected ValueError')
