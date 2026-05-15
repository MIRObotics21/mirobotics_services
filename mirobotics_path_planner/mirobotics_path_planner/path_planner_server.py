import json
import threading
from collections import deque
from typing import Deque, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import numpy as np

from mirobotics_msg.srv import CaptureScene, PlanPath
from mirobotics_path_planner.astar_3d import AStar3D


class CaptureSceneServer(Node):
    """
    First-stage CaptureScene server.

    Current behavior:
    - Continuously subscribes to /camera/camera/depth/color/points
    - Keeps the latest 3 PointCloud2 messages
    - On CaptureScene request with begin=True:
        - Converts buffered clouds to NumPy XYZ matrices
        - Merges them into one N x 3 matrix
        - Serializes the matrix into JSON
        - Returns it in json_matrix

    Intended future extension:
    - Add workspace cropping
    - Add voxelization into cubes
    - Add passable/impassable labeling
    - Reuse processed scene for PlanPath service
    """

    def __init__(self) -> None:
        super().__init__('path_planner_server')

        self._topic_name = 'pointcloud2'
        self._buffer_size = 3
        self._cloud_buffer: Deque[PointCloud2] = deque(maxlen=self._buffer_size)
        self._buffer_lock = threading.Lock()

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

        self._service = self.create_service(
            CaptureScene,
            'capture_scene',
            self._handle_capture_scene,
        )

        self._service = self.create_service(
            PlanPath,
            'plan_path',
            self._handle_plan_path,
        )

        self.get_logger().info(
            f"CaptureScene server started. Subscribing to: {resolved_topic}"
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
        NaN points are skipped.
        Only x, y, z fields are extracted.
        """
        points_iter = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True,
        )

        points_list = list(points_iter)
        if not points_list:
            return np.empty((0, 3), dtype=np.float32)

        points_array = np.asarray(points_list)

        # Structured array case: fields are named x, y, z
        if points_array.dtype.names is not None:
            return np.column_stack((
                points_array['x'],
                points_array['y'],
                points_array['z'],
            )).astype(np.float32, copy=False)

        # Plain tuple/list case
        return points_array.astype(np.float32, copy=False).reshape(-1, 3)

    def _merge_buffered_clouds(self, clouds: list[PointCloud2]) -> np.ndarray:
        matrices = []

        for cloud in clouds:
            xyz_matrix = self._pointcloud2_to_xyz_numpy(cloud)

            if xyz_matrix.size > 0:
                matrices.append(xyz_matrix)

        if not matrices:
            return np.empty((0, 3), dtype=np.float32)

        return np.vstack(matrices).astype(np.float32, copy=False)

    def _voxelize_occupied_points(self, xyz_matrix: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Convert XYZ points into occupied voxel centers.

        Output matrix:
        N x 4 = [x_center, y_center, z_center, passable]

        passable:
        0.0 = impassable / occupied
        """

        if voxel_size <= 0.0:
            raise ValueError('voxel_size must be greater than 0.')

        if xyz_matrix.size == 0:
            return np.empty((0, 4), dtype=np.float32)

        voxel_indices = np.floor(xyz_matrix / voxel_size).astype(np.int32)

        unique_indices = np.unique(voxel_indices, axis=0)

        voxel_centers = (unique_indices.astype(np.float32) + 0.5) * voxel_size

        passable_column = np.zeros((voxel_centers.shape[0], 1), dtype=np.float32)

        cube_matrix = np.hstack((voxel_centers, passable_column))

        return cube_matrix.astype(np.float32, copy=False)

    def _create_full_cuboid_voxel_matrix(
            self,
            occupied_cube_matrix: np.ndarray,
            voxel_size: float,
    ) -> np.ndarray:
        if occupied_cube_matrix.size == 0:
            return np.empty((0, 4), dtype=np.float32)

        occupied_centers = occupied_cube_matrix[:, :3]

        occupied_indices = np.round(
            occupied_centers / voxel_size - 0.5
        ).astype(np.int32)

        min_idx = np.min(occupied_indices, axis=0)
        max_idx = np.max(occupied_indices, axis=0)

        x_idx = np.arange(min_idx[0], max_idx[0] + 1)
        y_idx = np.arange(min_idx[1], max_idx[1] + 1)
        z_idx = np.arange(min_idx[2], max_idx[2] + 1)

        ii, jj, kk = np.meshgrid(x_idx, y_idx, z_idx, indexing='ij')

        all_indices = np.column_stack((
            ii.ravel(),
            jj.ravel(),
            kk.ravel(),
        )).astype(np.int32)

        occupied_set = set(map(tuple, occupied_indices))

        passable = np.ones((all_indices.shape[0], 1), dtype=np.float32)

        for row_idx, voxel_idx in enumerate(map(tuple, all_indices)):
            if voxel_idx in occupied_set:
                passable[row_idx, 0] = 0.0

        centers = (all_indices.astype(np.float32) + 0.5) * voxel_size

        full_cube_matrix = np.hstack((
            centers,
            passable,
        ))

        return full_cube_matrix.astype(np.float32, copy=False)

    def _handle_capture_scene(self, request: CaptureScene.Request, response: CaptureScene.Response) -> CaptureScene.Response:
        if request.voxel_size <= 0.0:
            response.success = False
            response.error_msg = 'voxel_size must be greater than 0.'
            response.json_matrix = ''
            return response

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
            merged_matrix = self._merge_buffered_clouds(buffered_clouds)

            if merged_matrix.size == 0:
                response.success = False
                response.error_msg = 'Merged point cloud is empty.'
                response.json_matrix = ''
                return response

            self.get_logger().info(f"Merged {len(buffered_clouds)} clouds into matrix with shape {merged_matrix.shape}")

            occupied_cube_matrix = self._voxelize_occupied_points(
                merged_matrix,
                request.voxel_size
            )

            if occupied_cube_matrix.size == 0:
                response.success = False
                response.error_msg = 'Voxelization produced empty occupied cube matrix.'
                response.json_matrix = ''
                return response

            cube_matrix = self._create_full_cuboid_voxel_matrix(
                occupied_cube_matrix,
                request.voxel_size
            )

            cube_ids = np.arange(cube_matrix.shape[0], dtype=np.float64).reshape(-1, 1)
            cube_matrix = np.hstack((cube_ids, cube_matrix.astype(np.float64)))
            cube_matrix = np.round(cube_matrix.astype(np.float64), decimals=4)

            payload = {
                'voxel_size': round(float(request.voxel_size), 4),
                'matrix': cube_matrix.tolist(),
            }

            scene_json = json.dumps(payload)

            with self._scene_lock:
                self._latest_scene_json = scene_json
                self._latest_scene_matrix = cube_matrix
                self._latest_voxel_size = float(request.voxel_size)

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
