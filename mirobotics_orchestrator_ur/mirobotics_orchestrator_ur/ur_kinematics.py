"""
Kinematics of the UR3 CB3 in the base_link frame of the ROS 2 driver.

Poses are 4 x 4 matrices, positions are in meters and joints in radians, in the
order of UR_JOINT_NAMES.
"""

import math
from typing import List, Sequence, Tuple

import numpy as np


UR_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Denavit-Hartenberg parameters of the UR3
D = (0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819)
A = (0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0)
ALPHA = (math.pi / 2.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0)

# base_link is the base frame of the controller rotated 180 degrees about z
BASE_LINK_FROM_CONTROLLER = np.diag([-1.0, -1.0, 1.0, 1.0])


def _link_transform(index: int, angle: float) -> np.ndarray:
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    cos_a, sin_a = math.cos(ALPHA[index]), math.sin(ALPHA[index])

    return np.array([
        [cos_t, -sin_t * cos_a, sin_t * sin_a, A[index] * cos_t],
        [sin_t, cos_t * cos_a, -cos_t * sin_a, A[index] * sin_t],
        [0.0, sin_a, cos_a, D[index]],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def forward_kinematics(joints: Sequence[float]) -> np.ndarray:
    """Return the pose of tool0 in base_link."""
    pose = np.eye(4)

    for index, angle in enumerate(joints):
        pose = pose @ _link_transform(index, float(angle))

    return BASE_LINK_FROM_CONTROLLER @ pose


def tcp_position(joints: Sequence[float], tcp_offset: float) -> Tuple[float, float, float]:
    """Return the TCP position in base_link, tcp_offset along the tool axis from tool0."""
    pose = forward_kinematics(joints)
    position = pose[:3, 3] + pose[:3, 2] * float(tcp_offset)

    return float(position[0]), float(position[1]), float(position[2])


def tool0_position(
    tcp_point: Sequence[float],
    rotation: np.ndarray,
    tcp_offset: float
) -> np.ndarray:
    """Return the tool0 position that puts the TCP on tcp_point."""
    return np.asarray(tcp_point, dtype=np.float64) - rotation[:3, 2] * float(tcp_offset)


def rotation_vector_to_matrix(rotation_vector: Sequence[float]) -> np.ndarray:
    """Convert a rotation vector rx, ry, rz, as shown on the pendant, to a rotation matrix."""
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))

    if angle == 0.0:
        return np.eye(3)

    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def pendant_to_tool0_rotation(rotation_vector: Sequence[float]) -> np.ndarray:
    """Convert pendant rx, ry, rz of the flange in the controller base to base_link."""
    return BASE_LINK_FROM_CONTROLLER[:3, :3] @ rotation_vector_to_matrix(rotation_vector)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> List[float]:
    """Return the quaternion [x, y, z, w] of a rotation matrix, also valid at 180 degrees."""
    # Largest element first, dividing by a small one is inaccurate
    w_squared = 1.0 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    x_squared = 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
    y_squared = 1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2]
    z_squared = 1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2]
    largest = max(w_squared, x_squared, y_squared, z_squared)

    if largest == w_squared:
        scale = 2.0 * math.sqrt(w_squared)
        return [
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            scale / 4.0,
        ]

    if largest == x_squared:
        scale = 2.0 * math.sqrt(x_squared)
        return [
            scale / 4.0,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
        ]

    if largest == y_squared:
        scale = 2.0 * math.sqrt(y_squared)
        return [
            (rotation[0, 1] + rotation[1, 0]) / scale,
            scale / 4.0,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
        ]

    scale = 2.0 * math.sqrt(z_squared)
    return [
        (rotation[0, 2] + rotation[2, 0]) / scale,
        (rotation[1, 2] + rotation[2, 1]) / scale,
        scale / 4.0,
        (rotation[1, 0] - rotation[0, 1]) / scale,
    ]


def _all_solutions(pose: np.ndarray) -> List[List[float]]:
    """Return the closed form solutions for a tool0 pose in the controller base."""
    wrist = pose @ np.array([0.0, 0.0, -D[5], 1.0])
    radius = math.hypot(wrist[0], wrist[1])

    if radius < D[3]:
        return []

    solutions = []
    inverse_pose = np.linalg.inv(pose)

    for shoulder_sign in (1.0, -1.0):
        pan = (
            math.atan2(wrist[1], wrist[0])
            + shoulder_sign * math.acos(D[3] / radius)
            + math.pi / 2.0
        )
        cos_pan, sin_pan = math.cos(pan), math.sin(pan)
        cos_wrist_2 = (pose[0, 3] * sin_pan - pose[1, 3] * cos_pan - D[3]) / D[5]

        if abs(cos_wrist_2) > 1.0:
            continue

        for wrist_sign in (1.0, -1.0):
            wrist_2 = wrist_sign * math.acos(cos_wrist_2)
            sin_wrist_2 = math.sin(wrist_2)

            # Wrist singularity, wrist 1 and wrist 3 are not separable
            if abs(sin_wrist_2) < 1e-6:
                continue

            wrist_3 = math.atan2(
                (-inverse_pose[1, 0] * sin_pan + inverse_pose[1, 1] * cos_pan) / sin_wrist_2,
                (inverse_pose[0, 0] * sin_pan - inverse_pose[0, 1] * cos_pan) / sin_wrist_2,
            )

            frame_1_to_4 = (
                np.linalg.inv(_link_transform(0, pan))
                @ pose
                @ np.linalg.inv(_link_transform(4, wrist_2) @ _link_transform(5, wrist_3))
            )
            # Lift and elbow are a planar two link arm in the x-y plane of frame 1
            x, y = frame_1_to_4[0, 3], frame_1_to_4[1, 3]
            cos_elbow = (x * x + y * y - A[1] * A[1] - A[2] * A[2]) / (2.0 * A[1] * A[2])

            if abs(cos_elbow) > 1.0:
                continue

            for elbow_sign in (1.0, -1.0):
                elbow = elbow_sign * math.acos(cos_elbow)
                lift = math.atan2(y, x) - math.atan2(
                    A[2] * math.sin(elbow),
                    A[1] + A[2] * math.cos(elbow)
                )
                wrist_1 = math.atan2(frame_1_to_4[1, 0], frame_1_to_4[0, 0]) - lift - elbow
                solutions.append([
                    _wrap_angle(angle)
                    for angle in (pan, lift, elbow, wrist_1, wrist_2, wrist_3)
                ])

    return solutions


def inverse_kinematics(position: Sequence[float], rotation: np.ndarray) -> List[float]:
    """
    Return the joints of a tool0 pose in base_link, always in the same arm configuration.

    The arm reaches forward over the shoulder toward the target with the elbow up and
    wrist 2 negative, every target of the workspace therefore gets a predictable pose.
    Raises ValueError when the pose has no such solution.
    """
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(position, dtype=np.float64)

    solutions = [
        solution
        for solution in _all_solutions(BASE_LINK_FROM_CONTROLLER @ pose)
        if solution[2] > 0.0 and solution[4] < 0.0
    ]

    if not solutions:
        raise ValueError(f'tool0 position {[float(c) for c in pose[:3, 3]]} is not reachable')

    azimuth = math.atan2(pose[1, 3], pose[0, 3])

    return min(solutions, key=lambda solution: abs(_wrap_angle(solution[0] - azimuth)))
