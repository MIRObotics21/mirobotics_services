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


def round_corners(
    points: Sequence[Sequence[float]],
    blend_radius: float,
    arc_points: int = 8
) -> List[Position]:
    """
    Replace each corner of a polyline by a smooth curve, like a blend radius on the pendant.

    The curve starts and ends blend_radius from the corner, at most half of the shorter
    adjacent segment, and stays inside the triangle of these two points and the corner.
    """
    path = [tuple(float(c) for c in point) for point in points]

    if blend_radius <= 0.0 or len(path) < 3:
        return path

    rounded = [path[0]]

    for before, corner, after in zip(path, path[1:], path[2:]):
        to_before = [b - c for b, c in zip(before, corner)]
        to_after = [a - c for a, c in zip(after, corner)]
        length_before = math.hypot(*to_before)
        length_after = math.hypot(*to_after)

        if length_before < 1e-9 or length_after < 1e-9:
            continue

        dot = sum(b * a for b, a in zip(to_before, to_after))
        cos_angle = dot / (length_before * length_after)

        # Straight on, nothing to round
        if cos_angle < -0.9999:
            rounded.append(corner)
            continue

        distance = min(blend_radius, length_before / 2.0, length_after / 2.0)
        entry = [c + b / length_before * distance for c, b in zip(corner, to_before)]
        leave = [c + a / length_after * distance for c, a in zip(corner, to_after)]

        # Quadratic Bezier curve from entry over the corner to leave
        for step in range(arc_points + 1):
            t = step / arc_points
            rounded.append(tuple(
                (1.0 - t) ** 2 * e + 2.0 * t * (1.0 - t) * c + t ** 2 * x
                for e, c, x in zip(entry, corner, leave)
            ))

    rounded.append(path[-1])

    return rounded


def rescale_trajectory(points: Sequence, speed_scaling: float) -> None:
    """
    Slow down trajectory points in place by speed_scaling (0.0 - 1.0].

    GetCartesianPath trajectories are timed at full joint speed.
    """
    if speed_scaling <= 0.0 or speed_scaling > 1.0:
        raise ValueError('speed_scaling must be in the range (0.0, 1.0].')

    for point in points:
        nanoseconds = point.time_from_start.sec * 1000000000 + point.time_from_start.nanosec
        nanoseconds = int(round(nanoseconds / speed_scaling))

        point.time_from_start.sec = nanoseconds // 1000000000
        point.time_from_start.nanosec = nanoseconds % 1000000000
        point.velocities = [v * speed_scaling for v in point.velocities]
        point.accelerations = [a * speed_scaling * speed_scaling for a in point.accelerations]


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
