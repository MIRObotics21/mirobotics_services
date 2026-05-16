import json
from geometry_msgs.msg import Pose


def parse_json_path(json_path: str):
    data = json.loads(json_path)
    return data.get("path", [])


def voxel_path_to_poses(
    json_path: str,
    fixed_orientation=None,
    min_distance: float = 0.03,
):
    """
    Converts PlanPath json_path to MoveIt Cartesian waypoints.

    Expected path row:
        [id, x, y, z, passable]
    """

    rows = parse_json_path(json_path)

    if fixed_orientation is None:
        fixed_orientation = {
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
            "w": 0.0,
        }

    poses = []
    last_position = None

    for row in rows:
        if len(row) < 4:
            continue

        x = float(row[1])
        y = float(row[2])
        z = float(row[3])

        if last_position is not None:
            dx = x - last_position[0]
            dy = y - last_position[1]
            dz = z - last_position[2]
            distance = (dx * dx + dy * dy + dz * dz) ** 0.5

            if distance < min_distance:
                continue

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        pose.orientation.x = fixed_orientation["x"]
        pose.orientation.y = fixed_orientation["y"]
        pose.orientation.z = fixed_orientation["z"]
        pose.orientation.w = fixed_orientation["w"]

        poses.append(pose)
        last_position = (x, y, z)

    return poses