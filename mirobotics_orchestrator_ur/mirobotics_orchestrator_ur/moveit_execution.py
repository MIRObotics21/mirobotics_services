from typing import Dict, List, Optional, Sequence, Tuple


Position = Tuple[float, float, float]


def tcp_to_tool0_position(point: Sequence[float], tcp_offset: float) -> Position:
    """Convert a TCP position to the tool0 position with the tool pointing straight down."""
    return float(point[0]), float(point[1]), float(point[2]) + float(tcp_offset)


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


def rescale_trajectory(points: Sequence, speed_scaling: float) -> None:
    """
    Slow down trajectory points in place by speed_scaling (0.0 - 1.0].

    GetCartesianPath of MoveIt Humble has no speed setting, its trajectories are
    timed at full joint speed.
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
