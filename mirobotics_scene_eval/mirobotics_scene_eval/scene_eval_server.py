import json
from pathlib import Path
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, HistoryPolicy, QoSProfile

from mirobotics_msg.srv import EvalScene
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import logging

LOGGER.setLevel(logging.ERROR)
LOGGER.disabled = True

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

class SceneEvalServer(Node):
    def __init__(self) -> None:
        super().__init__('scene_eval_server')

        self.declare_parameter('default_model_path', '')
        self.declare_parameter('min_confidence', 0.25)
        self.declare_parameter('max_detections', 50)
        self.declare_parameter('default_annotated_image_path', '')
        # Maximum accepted age of the evaluated image in seconds. 0 disables the check.
        self.declare_parameter('max_image_age', 1.0)

        self._latest_image_msg: Optional[Image] = None
        self._latest_image_time: Optional[float] = None
        self._image_lock = threading.Lock()

        self._bridge = CvBridge()
        self._model: Optional[YOLO] = None
        self._loaded_model_path: Optional[str] = None

        # Separate group so images keep arriving while a service call is running.
        self._image_callback_group = MutuallyExclusiveCallbackGroup()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._image_sub = self.create_subscription(
            Image,
            'image',
            self._image_callback,
            qos,
            callback_group=self._image_callback_group,
        )

        self._eval_scene_service = self.create_service(
            EvalScene,
            'eval_scene',
            self._handle_eval_scene,
        )

        self._preload_default_model()

        resolved_topic = self.resolve_topic_name('image')
        self.get_logger().info(
            f'SceneEval server started. Subscribing to: {resolved_topic}'
        )

    def _image_callback(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_image_msg = msg
            self._latest_image_time = time.monotonic()

    def _get_latest_image(self) -> Tuple[Optional[Image], Optional[float]]:
        """Return the latest image and its age in seconds since it was received."""
        with self._image_lock:
            if self._latest_image_msg is None:
                return None, None
            return self._latest_image_msg, time.monotonic() - self._latest_image_time

    def _get_model(self, model_path: str) -> YOLO:
        if self._model is None or self._loaded_model_path != model_path:
            self.get_logger().info(f'Loading model from: {model_path}')
            self._model = YOLO(model_path, task='detect')
            self._loaded_model_path = model_path
        return self._model

    def _preload_default_model(self) -> None:
        """
        Load default_model_path at startup so the first request is not delayed
        and a wrong path is reported immediately.
        """
        model_path = str(self.get_parameter('default_model_path').value).strip()

        if not model_path:
            return

        if not Path(model_path).is_file():
            self.get_logger().warning(
                f'default_model_path does not exist: {model_path}. '
                f'Model will be loaded on request.'
            )
            return

        try:
            model = self._get_model(model_path)
            # Dummy inference, the runtime is initialized on the first predict
            model.predict(
                source=np.zeros((480, 640, 3), dtype=np.uint8),
                verbose=False,
            )
        except Exception as exc:
            self._model = None
            self._loaded_model_path = None
            self.get_logger().error(f'Failed to preload model {model_path}: {exc}')
            return

        self.get_logger().info(f'Model preloaded: {model_path}')

    def _save_annotated_image(self, annotated, save_path: str) -> None:
        """
        save_path with an image extension is used as the file path,
        anything else is a directory that receives 'annotated.png'.
        """
        target = Path(save_path).expanduser()
        if target.suffix.lower() not in IMAGE_EXTENSIONS:
            target = target / 'annotated.png'

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(target), annotated)
        except (OSError, cv2.error) as exc:
            self.get_logger().warning(f'Failed to save annotated image {target}: {exc}')
            return

        if not ok:
            self.get_logger().warning(f'Failed to save annotated image: {target}')
        else:
            self.get_logger().info(f'Saved annotated image: {target}')

    def _evaluate_scene(self, model_path: str, cv_image, save_path: str):
        """
        Return a JSON-serializable Python list:
        [
            {
                "id": 1,
                "type": "yellow",
                "u": 320.0,
                "v": 240.0,
                "confidence": 0.91
            }
        ]
        """
        model = self._get_model(model_path)

        min_confidence = float(self.get_parameter('min_confidence').value)
        max_detections = int(self.get_parameter('max_detections').value)

        results = model.predict(
            source=cv_image,
            conf=min_confidence,
            max_det=max_detections,
            verbose=False,
        )

        objects = []
        if not results:
            return objects

        result = results[0]
        boxes = result.boxes
        names = result.names if hasattr(result, 'names') else {}

        if boxes is None or len(boxes) == 0:
            return objects

        for idx, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            label = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy

            u = float((x1 + x2) / 2.0)
            v = float((y1 + y2) / 2.0)

            objects.append(
                {
                    'id': idx,
                    'type': label,
                    'u': u,
                    'v': v,
                    'confidence': conf,
                }
            )

        if str(save_path) != '':
            self._save_annotated_image(result.plot(), str(save_path))

        return objects

    def _handle_eval_scene(self, request: EvalScene.Request, response: EvalScene.Response):
        default_model_path = str(self.get_parameter('default_model_path').value)
        default_annotated_image_path = str(self.get_parameter('default_annotated_image_path').value)

        model_path = request.model_path.strip() if request.model_path.strip() else default_model_path
        annotated_image_path = request.annotated_image_path.strip() if request.annotated_image_path.strip() else default_annotated_image_path

        if not model_path:
            response.success = False
            response.error_msg = (
                'No model_path provided in request and default_model_path parameter is empty.'
            )
            response.json_objects = '[]'
            return response

        self.get_logger().info(f'Received scene evaluation request with model_path="{model_path}"')

        image_msg, image_age = self._get_latest_image()
        if image_msg is None:
            response.success = False
            response.error_msg = 'No image received yet on topic "image".'
            response.json_objects = '[]'
            return response

        max_image_age = float(self.get_parameter('max_image_age').value)
        if max_image_age > 0.0 and image_age > max_image_age:
            response.success = False
            response.error_msg = (
                f'Latest image is stale: received {image_age:.2f} s ago, '
                f'max_image_age is {max_image_age:.2f} s.'
            )
            response.json_objects = '[]'
            self.get_logger().error(response.error_msg)
            return response

        try:
            cv_image = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            response.success = False
            response.error_msg = f'Failed to convert ROS image to OpenCV image: {exc}'
            response.json_objects = '[]'
            return response

        try:
            objects = self._evaluate_scene(model_path, cv_image, annotated_image_path)
        except Exception as exc:
            response.success = False
            response.error_msg = f'Scene evaluation failed: {exc}'
            response.json_objects = '[]'
            self.get_logger().error(response.error_msg)
            return response

        response.success = True
        response.error_msg = ''
        response.json_objects = json.dumps(objects)
        self.get_logger().info(f'Scene evaluation succeeded. Found {len(objects)} object(s).')
        return response

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SceneEvalServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()