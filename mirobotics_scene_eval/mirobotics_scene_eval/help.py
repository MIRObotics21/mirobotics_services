"""
Usage help of mirobotics_scene_eval.

    ros2 run mirobotics_scene_eval help [topic]
"""

import sys


OVERVIEW = """\
mirobotics_scene_eval - YOLO object detection on the latest camera image

Node:        scene_eval_server (launch argument node_name)
Subscribes:  image (sensor_msgs/Image), set by launch argument image_publisher
Service:     /eval_scene (mirobotics_msg/srv/EvalScene)

Launch:
  ros2 launch mirobotics_scene_eval mirobotics_scene_eval.launch.py \\
    image_publisher:=/camera/camera/color/image_raw
"""

TOPICS = {
    'eval_scene': """\
/eval_scene (mirobotics_msg/srv/EvalScene)
  Runs the YOLO model on the latest image and returns the detected objects.

  Request:
    model_path            .pt or .onnx model, empty = default_model_path parameter
    annotated_image_path  directory or .png/.jpg file for the annotated image,
                          empty = default_annotated_image_path parameter
  Response:
    success, error_msg
    json_objects          [{"id", "type", "u", "v", "confidence"}]
                          u, v = bounding box center in pixels

  Example:
    ros2 service call /eval_scene mirobotics_msg/srv/EvalScene \\
      "{model_path: '/home/mirobotics/models/best.onnx', annotated_image_path: '/home/mirobotics/output/'}"

  Fails when no image was received yet or the latest image is older than max_image_age.
""",
    'parameters': """\
Parameters (ros2 param set /scene_eval_server <name> <value>)
  default_model_path            ''     Model used for an empty model_path, loaded at startup if it exists
  default_annotated_image_path  ''     Used for an empty annotated_image_path, empty = no image
  min_confidence                0.25   Minimum detection confidence
  max_detections                50     Maximum number of detections per image
  max_image_age                 1.0    Maximum age of the evaluated image in seconds, 0 = no check
""",
}

FOOTER = """\
Field details: ros2 interface show mirobotics_msg/srv/EvalScene
Help topics:   ros2 run mirobotics_scene_eval help [{topics}]
"""


def main(args=None) -> None:
    args = sys.argv[1:] if args is None else args
    footer = FOOTER.format(topics='|'.join(TOPICS))

    if args:
        topic = args[0]

        if topic not in TOPICS:
            print(f'Unknown help topic "{topic}".\n')
            print(footer)
            sys.exit(1)

        print(TOPICS[topic])
        return

    print(OVERVIEW)
    for text in TOPICS.values():
        print(text)
    print(footer)


if __name__ == '__main__':
    main()
