import json
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from mirobotics_msg.srv import EvalScene


class SceneEvalServer(Node):
    def __init__(self) -> None:
        super().__init__('scene_eval_server')

        self.declare_parameter('timeout_sec', 5.0)
        self.declare_parameter('default_model_path', '')
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('max_detections', 50)

        self._bridge = CvBridge()
        self._latest_image_msg: Optional[Image] = None
        self._image_event = threading.Event()

        self._image_sub = self.create_subscription(
            Image,
            'image',
            self._image_callback,
            10,
        )

        self._service = self.create_service(
            EvalScene,
            '/mirobotics_scene_eval/mirobotics_eval_scene',
            self._handle_eval_scene,
        )

        self.get_logger().info('SceneEvalServer ready on /mirobotics_scene_eval/mirobotics_eval_scene')

    def _image_callback(self, msg: Image) -> None:
        self._latest_image_msg = msg
        self._image_event.set()

    def _wait_for_one_image(self, timeout_sec: float) -> Optional[Image]:
        self._latest_image_msg = None
        self._image_event.clear()

        got_image = self._image_event.wait(timeout=timeout_sec)
        if not got_image:
            return None

        return self._latest_image_msg

    def _handle_eval_scene(self, request: EvalScene.Request, response: EvalScene.Response):
        timeout_sec = float(self.get_parameter('timeout_sec').value)
        default_model_path = str(self.get_parameter('default_model_path').value)

        model_path = request.model_path.strip() if request.model_path.strip() else default_model_path

        if not model_path:
            response.success = False
            response.error_msg = 'No model_path provided in request and default_model_path parameter is empty.'
            response.objects_json = '[]'
            return response

        self.get_logger().info(f'Received scene evaluation request with model_path="{model_path}"')

        image_msg = self._wait_for_one_image(timeout_sec)
        if image_msg is None:
            response.success = False
            response.error_msg = f'Timeout waiting for one image on topic "image" after {timeout_sec:.2f} s.'
            response.objects_json = '[]'
            return response

        try:
            cv_image = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            response.success = False
            response.error_msg = f'Failed to convert ROS image to OpenCV image: {exc}'
            response.objects_json = '[]'
            return response

        try:
            objects = self._evaluate_scene(model_path, cv_image)
        except Exception as exc:
            response.success = False
            response.error_msg = f'Scene evaluation failed: {exc}'
            response.objects_json = '[]'
            self.get_logger().error(response.error_msg)
            return response

        response.success = True
        response.error_msg = ''
        response.objects_json = json.dumps(objects)
        self.get_logger().info(f'Scene evaluation succeeded. Returned {len(objects)} object(s).')
        return response

    def _evaluate_scene(self, model_path: str, cv_image):
        """
        Placeholder for your actual model inference.

        Return value must be a JSON-serializable Python list, for example:
        [
            {"id": 1, "type": "apple", "u": 320.0, "v": 240.0},
            {"id": 2, "type": "banana", "u": 100.0, "v": 200.0}
        ]
        """
        height, width = cv_image.shape[:2]

        # Temporary dummy output so the service can be tested end-to-end.
        objects = [
            {
                "id": 1,
                "type": "dummy_object",
                "u": float(width / 2.0),
                "v": float(height / 2.0),
            }
        ]
        return objects


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SceneEvalServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()