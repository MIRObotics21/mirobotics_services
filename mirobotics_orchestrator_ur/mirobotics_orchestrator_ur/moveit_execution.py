import math
from typing import Dict, List, Optional, Sequence, Tuple

from mirobotics_orchestrator_ur.ur_kinematics import tcp_position, UR_JOINT_NAMES


Position = Tuple[float, float, float]

# Largest joint change between two path samples in radians, about 1 cm at the TCP
PATH_SAMPLE_STEP = 0.02


def find_object_target(objects_3d: List[Dict], object_id: int) -> Position:
    """
    Return the approach point of an object from the last GenerateScene.

    Raises ValueError when the object is unknown or has no approach point.
    """
    obj: Optional[Dict] = next(
        (candidate for candidate in objects_3d if candidate.get('id') == object_id),
        None
    )

    if obj is None:
        known_ids = [candidate.get('id') for candidate in objects_3d]
        raise ValueError(f'Object {object_id} not found, last GenerateScene found {known_ids}')

    approach_point = obj.get('approach_point')

    if approach_point is None:
        raise ValueError(f'Object {object_id} has no approach point')

    return float(approach_point[0]), float(approach_point[1]), float(approach_point[2])


def hover_point(approach_point: Sequence[float], voxel_size: float, height: float) -> Position:
    """Return the point height above the object, whose top is the bottom of the approach voxel."""
    top = float(approach_point[2]) - float(voxel_size) / 2.0

    return float(approach_point[0]), float(approach_point[1]), top + float(height)


def largest_joint_travel(joint_names: Sequence[str], points: Sequence) -> Tuple[str, float]:
    """Return the joint with the largest range of motion in the trajectory and that range."""
    largest_name = ''
    largest_travel = 0.0

    for index, name in enumerate(joint_names):
        positions = [point.positions[index] for point in points]

        if positions and max(positions) - min(positions) > largest_travel:
            largest_name = name
            largest_travel = max(positions) - min(positions)

    return largest_name, largest_travel


def sample_tcp_path(
    joint_names: Sequence[str],
    points: Sequence,
    tcp_offset: float
) -> List[Position]:
    """Return TCP positions along a trajectory, interpolated between its points."""
    order = [list(joint_names).index(name) for name in UR_JOINT_NAMES]
    trajectory = [[point.positions[index] for index in order] for point in points]

    if not trajectory:
        return []

    samples = [tcp_position(trajectory[0], tcp_offset)]

    for start, end in zip(trajectory, trajectory[1:]):
        largest_step = max(abs(b - a) for a, b in zip(start, end))
        steps = max(1, math.ceil(largest_step / PATH_SAMPLE_STEP))

        for step in range(1, steps + 1):
            fraction = step / steps
            joints = [a + (b - a) * fraction for a, b in zip(start, end)]
            samples.append(tcp_position(joints, tcp_offset))

    return samples


def first_blocked_sample(occupied: Sequence[bool]) -> Optional[int]:
    """
    Return the index of the first occupied sample the TCP moves into, or None.

    Occupied samples at the start are skipped: the camera sees the robot, so the
    TCP can start inside its own voxels.
    """
    left_start = False

    for index, sample_occupied in enumerate(occupied):
        if not sample_occupied:
            left_start = True
        elif left_start:
            return index

    # The TCP never leaves the occupied voxels, the target is occupied too
    if occupied and not left_start:
        return len(occupied) - 1

    return None
