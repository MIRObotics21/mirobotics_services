import json
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, HistoryPolicy, QoSProfile

from mirobotics_msg.srv import EvalScene
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import logging

LOGGER.setLevel(logging.ERROR)
LOGGER.disabled = True

class SceneEvalServer(Node):
    def __init__(self) -> None:
        super().__init__('scene_eval_server')

        self.declare_parameter('default_model_path', '')
        self.declare_parameter('timeout_sec', 5.0)
        self.declare_parameter('min_confidence', 0.25)
        self.declare_parameter('max_detections', 50)
        self.declare_parameter('default_annotated_image_path', '')


        self._latest_image_msg: Optional[Image] = None
        self._image_lock = threading.Lock()

        self._bridge = CvBridge()
        self._model: Optional[YOLO] = None
        self._loaded_model_path: Optional[str] = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._image_sub = self.create_subscription(
            Image,
            'image',
            self._image_callback,
            qos,
        )

        self._service = self.create_service(
            EvalScene,
            '/mirobotics_scene_eval/mirobotics_eval_scene',
            self._handle_eval_scene,
        )

        resolved_topic = self.resolve_topic_name('image')
        self.get_logger().info(
            f'SceneEvalServer ready on /mirobotics_scene_eval/mirobotics_eval_scene. '
            f'Subscribing to: {resolved_topic}'
        )

    def _image_callback(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_image_msg = msg

    def _get_latest_image(self) -> Optional[Image]:
        with self._image_lock:
            return self._latest_image_msg

    def _get_model(self, model_path: str) -> YOLO:
        if self._model is None or self._loaded_model_path != model_path:
            self.get_logger().info(f'Loading model from: {model_path}')
            self._model = YOLO(model_path)
            self._loaded_model_path = model_path
        return self._model

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
        model = YOLO(self._get_model(model_path), task='detect')

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
            annotated = result.plot()
            ok = cv2.imwrite(str(save_path)+'anotated.png', annotated)

            if not ok:
                self.get_logger().info(f"[WARNING] Failed to save annotated image: {save_path}")
            else:
                self.get_logger().info(f"Saved annotated image: {save_path}")

        return objects

    def _handle_eval_scene(self, request: EvalScene.Request, response: EvalScene.Response):
        timeout_sec = float(self.get_parameter('timeout_sec').value)
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

        image_msg = self._get_latest_image()
        if image_msg is None:
            response.success = False
            response.error_msg = 'No image received yet on topic "image".'
            response.json_objects = '[]'
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

    try:
        rclpy.spin(node)
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