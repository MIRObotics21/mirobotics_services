import json
import threading
import time
from collections import deque
from typing import Deque

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import tf2_ros

import numpy as np

from mirobotics_msg.srv import CaptureScene, PlanPath
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


class CaptureSceneServer(Node):
    def __init__(self) -> None:
        super().__init__('path_planner_server')

        self._topic_name = 'pointcloud2'
        self._buffer_size = 3
        self._cloud_buffer: Deque[PointCloud2] = deque(maxlen=self._buffer_size)
        self._buffer_lock = threading.Lock()

        self.declare_parameter('planning_frame', '')
        # Workspace box [x, y, z] in planning_frame, in meters.
        # Points outside are discarded and the voxel grid covers exactly this box.
        self.declare_parameter('workspace_min', [-0.5, -0.5, 0.0])
        self.declare_parameter('workspace_max', [0.5, 0.5, 0.6])
        # Voxel is occupied only if it contains at least this many points.
        self.declare_parameter('min_points_per_voxel', 3)
        # Mark everything under the highest occupied voxel of a column as occupied.
        self.declare_parameter('fill_below_occupied', True)
        self.declare_parameter('max_voxels', 2000000)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._latest_scene_json = None
        self._latest_scene_matrix = None
        self._latest_voxel_size = None
        self._scene_lock = threading.Lock()

        resolved_topic = self.resolve_topic_name(self._topic_name)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._subscription = self.create_subscription(
            PointCloud2,
            self._topic_name,
            self._pointcloud_callback,
            qos,
        )

        self._capture_scene_service = self.create_service(
            CaptureScene,
            'capture_scene',
            self._handle_capture_scene,
        )

        self._plan_path_service = self.create_service(
            PlanPath,
            'plan_path',
            self._handle_plan_path,
        )

        self.get_logger().info(
            f"PathPlanner server started. Subscribing to: {resolved_topic}"
        )

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        with self._buffer_lock:
            self._cloud_buffer.append(msg)

        self.get_logger().debug(
            f"Received PointCloud2. Buffered {len(self._cloud_buffer)}/{self._buffer_size} messages."
        )

    def _pointcloud2_to_xyz_numpy(self, msg: PointCloud2) -> np.ndarray:
        """
        Convert PointCloud2 to an N x 3 NumPy array [x, y, z].
        NaN and infinite points are skipped.
        Only x, y, z fields are extracted.
        """
        points = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True,
        )

        if not isinstance(points, np.ndarray):
            # Older sensor_msgs_py returns a generator of tuples
            points = np.array(list(points), dtype=np.float32).reshape(-1, 3)
        elif points.dtype.names is not None:
            # Structured array case: fields are named x, y, z
            points = np.column_stack((
                points['x'],
                points['y'],
                points['z'],
            ))

        points = points.astype(np.float32, copy=False).reshape(-1, 3)

        return points[np.all(np.isfinite(points), axis=1)]

    def _merge_buffered_clouds(self, clouds: list[PointCloud2]) -> np.ndarray:
        matrices = []

        for cloud in clouds:
            xyz_matrix = self._pointcloud2_to_xyz_numpy(cloud)

            if xyz_matrix.size > 0:
                matrices.append(xyz_matrix)

        if not matrices:
            return np.empty((0, 3), dtype=np.float32)

        return np.vstack(matrices)

    def _transform_xyz_matrix(
            self,
            xyz_matrix: np.ndarray,
            source_frame: str,
            target_frame: str,
    ) -> np.ndarray:
        """
        Transform an N x 3 XYZ matrix from source_frame into target_frame.
        """
        if xyz_matrix.size == 0:
            return xyz_matrix

        if source_frame == target_frame:
            return xyz_matrix

        transform = self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0),
        )

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        rotation_matrix = quaternion_to_rotation_matrix(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )

        return transform_points(
            xyz_matrix,
            rotation_matrix,
            [translation.x, translation.y, translation.z],
        )

    def _handle_capture_scene(self, request: CaptureScene.Request, response: CaptureScene.Response) -> CaptureScene.Response:
        self.get_logger().info(f"Received request for voxelization with size: {float(request.voxel_size)}.")
        if request.voxel_size <= 0.0:
            response.success = False
            response.error_msg = 'voxel_size must be greater than 0.'
            response.json_matrix = ''
            return response

        # request.voxel_size is float32, rounding removes its conversion noise
        voxel_size = round(float(request.voxel_size), 6)

        with self._buffer_lock:
            buffered_clouds = list(self._cloud_buffer)

        if len(buffered_clouds) < self._buffer_size:
            response.success = False
            response.error_msg = (
                f'Not enough point clouds buffered yet. '
                f'Expected {self._buffer_size}, got {len(buffered_clouds)}.'
            )
            response.json_matrix = ''
            return response

        try:
            workspace_min = list(self.get_parameter('workspace_min').value)
            workspace_max = list(self.get_parameter('workspace_max').value)
            min_points_per_voxel = int(self.get_parameter('min_points_per_voxel').value)
            max_voxels = int(self.get_parameter('max_voxels').value)

            min_index, grid_shape = compute_grid(workspace_min, workspace_max, voxel_size)
            voxel_count = grid_shape[0] * grid_shape[1] * grid_shape[2]

            if voxel_count > max_voxels:
                response.success = False
                response.error_msg = (
                    f'voxel_size {voxel_size} produces {voxel_count} voxels, '
                    f'max_voxels is {max_voxels}.'
                )
                response.json_matrix = ''
                return response

            time_start = time.perf_counter()

            merged_matrix = self._merge_buffered_clouds(buffered_clouds)

            if merged_matrix.size == 0:
                response.success = False
                response.error_msg = 'Merged point cloud is empty.'
                response.json_matrix = ''
                return response

            time_converted = time.perf_counter()

            source_frame = buffered_clouds[-1].header.frame_id
            target_frame = self.get_parameter('planning_frame').value

            if not target_frame:
                target_frame = source_frame

            try:
                if target_frame != source_frame:
                    merged_matrix = self._transform_xyz_matrix(
                        merged_matrix,
                        source_frame,
                        target_frame,
                    )
            except Exception as exc:
                response.success = False
                response.error_msg = (
                    f'Failed to transform point cloud from {source_frame} '
                    f'to {target_frame}: {str(exc)}'
                )
                response.json_matrix = ''
                return response

            point_count = merged_matrix.shape[0]

            # Workspace bounds are given in target_frame, crop after the transform
            merged_matrix = crop_to_workspace(merged_matrix, workspace_min, workspace_max)

            if merged_matrix.size == 0:
                response.success = False
                response.error_msg = (
                    f'No points inside workspace {workspace_min} -> {workspace_max} '
                    f'in frame {target_frame}.'
                )
                response.json_matrix = ''
                return response

            time_transformed = time.perf_counter()

            occupancy = voxelize(
                merged_matrix,
                min_index,
                grid_shape,
                voxel_size,
                min_points_per_voxel,
            )

            if self.get_parameter('fill_below_occupied').value:
                occupancy = fill_below_occupied(occupancy)

            cube_matrix = occupancy_to_matrix(occupancy, min_index, voxel_size)
            cube_matrix = np.round(cube_matrix, decimals=4)

            time_voxelized = time.perf_counter()

            payload = {
                'voxel_size': round(voxel_size, 4),
                'matrix': cube_matrix.tolist(),
            }

            scene_json = json.dumps(payload)

            time_serialized = time.perf_counter()

            self.get_logger().info(
                f'Voxelization successful. {point_count} points, '
                f'{merged_matrix.shape[0]} inside workspace, '
                f'grid {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]}, '
                f'{int(occupancy.sum())} occupied voxels.'
            )
            self.get_logger().info(
                f'Timing [s]: convert {time_converted - time_start:.3f}, '
                f'transform+crop {time_transformed - time_converted:.3f}, '
                f'voxelize {time_voxelized - time_transformed:.3f}, '
                f'json {time_serialized - time_voxelized:.3f}'
            )

            with self._scene_lock:
                self._latest_scene_json = scene_json
                self._latest_scene_matrix = cube_matrix
                self._latest_voxel_size = voxel_size

            response.success = True
            response.error_msg = ''
            response.json_matrix = scene_json
            return response

        except Exception as exc:
            self.get_logger().error(f'CaptureScene processing failed: {exc}')
            response.success = False
            response.error_msg = f'CaptureScene processing failed: {str(exc)}'
            response.json_matrix = ''
            return response

    def _handle_plan_path(self, request, response):
        with self._scene_lock:
            if self._latest_scene_matrix is None:
                response.success = False
                response.error_msg = 'No captured scene available. Call CaptureScene first.'
                response.json_path = ''
                return response

            scene_matrix = self._latest_scene_matrix.copy()
            voxel_size = self._latest_voxel_size

        try:
            planner = AStar3D(scene_matrix, voxel_size)
            path = planner.plan(request.start_id, request.goal_id)

            if not path:
                response.success = False
                response.error_msg = 'No path found.'
                response.json_path = ''
                return response

            payload = {
                'columns': ['id', 'x', 'y', 'z', 'passable'],
                'path': path,
            }

            response.success = True
            response.error_msg = ''
            response.json_path = json.dumps(payload)
            return response

        except Exception as exc:
            response.success = False
            response.error_msg = str(exc)
            response.json_path = ''
            return response

def main(args=None) -> None:
    rclpy.init(args=args)
    node = CaptureSceneServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
