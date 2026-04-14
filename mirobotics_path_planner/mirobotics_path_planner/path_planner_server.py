import json
import threading
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# Replace this import with your actual package/service path.
# Example:
# from mirobotics_path_planner.srv import CaptureScene
from mirobotics_msg.srv import CaptureScene


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
        super().__init__('capture_scene_server')

        self._topic_name = 'pointcloud2'
        self._buffer_size = 3
        self._cloud_buffer: Deque[PointCloud2] = deque(maxlen=self._buffer_size)
        self._buffer_lock = threading.Lock()

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

        self.get_logger().info(
            f"CaptureScene server started. Subscribing to: {resolved_topic}"
        )

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        with self._buffer_lock:
            self._cloud_buffer.append(msg)

        self.get_logger().debug(
            f"Received PointCloud2. Buffered {len(self._cloud_buffer)}/{self._buffer_size} messages."
        )

    def _handle_capture_scene(self, request: CaptureScene.Request, response: CaptureScene.Response) -> CaptureScene.Response:
        if not request.begin:
            response.success = False
            response.error_msg = 'Request rejected: begin must be true.'
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
            matrices = [self._pointcloud2_to_xyz_numpy(cloud) for cloud in buffered_clouds]

            non_empty_matrices = [m for m in matrices if m.size > 0]
            if not non_empty_matrices:
                response.success = False
                response.error_msg = 'All buffered point clouds were empty after conversion.'
                response.json_matrix = ''
                return response

            merged_matrix = np.vstack(non_empty_matrices).astype(np.float32, copy=False)

            payload = {
                'frame_id': buffered_clouds[-1].header.frame_id,
                'cloud_count': len(buffered_clouds),
                'point_count': int(merged_matrix.shape[0]),
                'columns': ['x', 'y', 'z'],
                'matrix': merged_matrix.tolist(),
            }

            response.success = True
            response.error_msg = ''
            response.json_matrix = json.dumps(payload)
            return response

        except Exception as exc:
            self.get_logger().error(f'CaptureScene processing failed: {exc}')
            response.success = False
            response.error_msg = f'CaptureScene processing failed: {str(exc)}'
            response.json_matrix = ''
            return response

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
