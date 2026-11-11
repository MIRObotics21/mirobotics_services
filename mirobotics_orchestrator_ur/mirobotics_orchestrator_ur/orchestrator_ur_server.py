import math
import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Point, Pose, PoseArray
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    PlanningOptions,
)
from moveit_msgs.srv import GetCartesianPath

from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)

from mirobotics_msg.action import GenerateScene, MoveBetween, PlanAndExecute
from mirobotics_msg.srv import EvalScene, CaptureScene, CheckPoints, PlanPath

from mirobotics_orchestrator_ur.moveit_execution import (
    find_object_target,
    first_blocked_sample,
    hover_point,
    largest_joint_travel,
    rescale_trajectory,
    round_corners,
    sample_tcp_path,
)
from mirobotics_orchestrator_ur.ray_casting import assign_objects_to_voxels, parse_scene_grid
from mirobotics_orchestrator_ur.ur_kinematics import (
    inverse_kinematics,
    pendant_to_tool0_rotation,
    rotation_matrix_to_quaternion,
    tool0_position,
    UR_JOINT_NAMES,
)

BASE_FRAME = 'base_link'
SCENE_EVAL_SERVICE = '/eval_scene'
CAPTURE_SCENE_SERVICE = '/capture_scene'
CHECK_POINTS_SERVICE = '/check_points'
PLAN_PATH_SERVICE = '/plan_path'

# Marker colors per object type, assigned in order of first appearance
OBJECT_COLORS = [
    (0.9, 0.2, 0.2),
    (0.2, 0.4, 1.0),
    (0.2, 0.8, 0.2),
    (1.0, 0.8, 0.1),
    (0.8, 0.3, 0.9),
    (0.1, 0.8, 0.8),
]

MOVE_GROUP_ACTION = '/move_action'
EXECUTE_TRAJECTORY_ACTION = '/execute_trajectory'
CARTESIAN_PATH_SERVICE = '/compute_cartesian_path'
MOVE_GROUP_NAME = 'ur_manipulator'
MOVEIT_TARGET_LINK = 'tool0'
JOINT_TOLERANCE = 0.01
CARTESIAN_MAX_STEP = 0.005
# Largest joint change between two Cartesian path steps in radians, stops the path at a flip
CARTESIAN_REVOLUTE_JUMP = 0.5


class OrchestratorURServer(Node):

    def __init__(self):
        super().__init__('orchestrator_ur_server')

        self.declare_parameter(
            'camera_info_topic',
            '/camera/camera/color/camera_info'
        )

        self.camera_info_topic = self.get_parameter(
            'camera_info_topic'
        ).get_parameter_value().string_value

        # Maximum distance from the camera searched for an object, in meters.
        self.declare_parameter('ray_max_distance', 3.0)
        # Maximum time to wait for a response of the services, in seconds.
        self.declare_parameter('service_timeout', 30.0)
        # Publish object, hit point and approach voxels for RViz after each GenerateScene.
        self.declare_parameter('publish_markers', True)
        # Motions (PlanAndExecute, MoveBetween):
        # height of the TCP above the top of the object, in meters.
        self.declare_parameter('hover_height', 0.05)
        # Distance from tool0 to the TCP along the tool axis, in meters.
        self.declare_parameter('tcp_offset', 0.1866)
        # Fixed orientation rx, ry, rz of the pendant, in radians.
        self.declare_parameter('tcp_rotation', [math.pi, 0.0, 0.0])
        # MoveBetween: corners of the path are rounded from this distance before them, in
        # meters, like the blend radius on the pendant. 0 keeps sharp corners.
        self.declare_parameter('blend_radius', 0.02)
        # Fraction of the maximum speed and acceleration, (0.0 - 1.0].
        self.declare_parameter('speed_scaling', 0.1)
        # Maximum duration of one motion in seconds.
        self.declare_parameter('motion_timeout', 120.0)
        # A plan is rejected if one joint travels more than this, in radians.
        self.declare_parameter('max_joint_travel', 3.14)
        # Plan, check and publish everything but do not move the robot.
        self.declare_parameter('dry_run', False)

        self.camera_info_received = False
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.image_width = None
        self.image_height = None
        self.camera_frame = None

        self.latest_objects_3d = []
        self.latest_voxel_size = None
        self.motion_lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        # Responses are handled by another executor thread while an action waits for them.
        self.client_callback_group = ReentrantCallbackGroup()

        self.eval_scene_client = self.create_client(
            EvalScene,
            SCENE_EVAL_SERVICE,
            callback_group=self.client_callback_group
        )

        self.capture_scene_client = self.create_client(
            CaptureScene,
            CAPTURE_SCENE_SERVICE,
            callback_group=self.client_callback_group
        )

        self.check_points_client = self.create_client(
            CheckPoints,
            CHECK_POINTS_SERVICE,
            callback_group=self.client_callback_group
        )

        self.plan_path_client = self.create_client(
            PlanPath,
            PLAN_PATH_SERVICE,
            callback_group=self.client_callback_group
        )

        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            CARTESIAN_PATH_SERVICE,
            callback_group=self.client_callback_group
        )

        # Latched, RViz receives the last scene even when started later.
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.object_marker_pub = self.create_publisher(
            MarkerArray,
            'scene_object_markers',
            marker_qos
        )

        self.tool0_pose_pub = self.create_publisher(
            PoseArray,
            'planned_tool0_poses',
            marker_qos
        )

        self.action_server = ActionServer(
            self,
            GenerateScene,
            'generate_scene',
            self.execute_generate_scene_callback
        )

        self.plan_execute_action_server = ActionServer(
            self,
            PlanAndExecute,
            'plan_and_execute',
            self.execute_plan_and_execute_callback,
            callback_group=self.client_callback_group,
            cancel_callback=lambda _: CancelResponse.ACCEPT
        )

        self.move_between_action_server = ActionServer(
            self,
            MoveBetween,
            'move_between',
            self.execute_move_between_callback,
            callback_group=self.client_callback_group,
            cancel_callback=lambda _: CancelResponse.ACCEPT
        )

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            MOVE_GROUP_ACTION,
            callback_group=self.client_callback_group
        )

        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            EXECUTE_TRAJECTORY_ACTION,
            callback_group=self.client_callback_group
        )

        self.get_logger().info(
            f'Using camera info topic: {self.camera_info_topic}'
        )

        self.get_logger().info(
            'mirobotics_orchestrator_ur action server started.'
        )

    def camera_info_callback(self, msg: CameraInfo):
        # Keep the latest intrinsics, they change when the camera resolution changes
        self.camera_info_received = True

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.image_width = msg.width
        self.image_height = msg.height
        self.camera_frame = msg.header.frame_id

    def publish_feedback(self, goal_handle, current_step, progress):
        feedback = GenerateScene.Feedback()
        feedback.current_step = current_step
        feedback.progress = float(progress)
        goal_handle.publish_feedback(feedback)

        self.get_logger().info(
            f'GenerateScene feedback: {current_step} ({progress:.2f})'
        )

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """
        Convert quaternion into 3x3 rotation matrix.

        The returned matrix transforms vectors from source frame to target frame
        when using a TF transform target <- source.
        """
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)

        if norm == 0.0:
            raise ValueError('Invalid zero-length quaternion')

        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        return [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ]

    def call_service(self, client, request, timeout_sec):
        """
        Call a service and block until the response arrives or timeout_sec passes.

        The response is received by another executor thread, so this must not be
        called from a callback in client_callback_group.
        """
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: done.set())

        if not done.wait(timeout_sec):
            try:
                client.remove_pending_request(future)
            except AttributeError:
                pass
            raise TimeoutError(f'No response from {client.srv_name} within {timeout_sec:.1f} s')

        return future.result()

    def get_camera_transform(self, target_frame):
        """
        Look up the camera optical frame in target_frame.

        Returns camera_origin, the [x, y, z] of the camera optical frame in
        target_frame, and rotation_target_camera, the 3x3 rotation matrix
        camera -> target_frame.
        """
        if self.camera_frame is None:
            raise RuntimeError('Camera frame is unknown because CameraInfo was not received')

        transform = self.tf_buffer.lookup_transform(
            target_frame,
            self.camera_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=2.0)
        )

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        camera_origin = [
            translation.x,
            translation.y,
            translation.z,
        ]

        rotation_target_camera = self.quaternion_to_rotation_matrix(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w
        )

        return camera_origin, rotation_target_camera

    def publish_object_markers(self, objects_3d, voxel_size, frame_id):
        """
        Publish the localized objects for RViz.

        Per detected object:
            object voxel   - cube in the object type color
            hit point      - small white sphere where the ray enters the object voxel
            approach voxel - green transparent cube
            label          - "id type confidence" above the object
        Markers of the previous GenerateScene are deleted first.
        """
        markers = []

        clear = Marker()
        clear.header.frame_id = frame_id
        clear.action = Marker.DELETEALL
        markers.append(clear)

        type_colors = {}

        def new_marker(ns, marker_id, marker_type, position, scale, color, alpha):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.ns = ns
            marker.id = marker_id
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.pose.position = Point(
                x=float(position[0]), y=float(position[1]), z=float(position[2])
            )
            marker.pose.orientation.w = 1.0
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = alpha
            return marker

        for marker_id, obj in enumerate(objects_3d):
            object_type = str(obj.get('type'))

            if object_type not in type_colors:
                type_colors[object_type] = OBJECT_COLORS[len(type_colors) % len(OBJECT_COLORS)]

            color = type_colors[object_type]

            if obj.get('object_point') is not None:
                markers.append(new_marker(
                    'object_voxels', marker_id, Marker.CUBE,
                    obj['object_point'], voxel_size, color, 0.8
                ))

            if obj.get('hit_point') is not None:
                markers.append(new_marker(
                    'hit_points', marker_id, Marker.SPHERE,
                    obj['hit_point'], max(voxel_size * 0.3, 0.008), (1.0, 1.0, 1.0), 1.0
                ))

            if obj.get('approach_point') is not None:
                markers.append(new_marker(
                    'approach_voxels', marker_id, Marker.CUBE,
                    obj['approach_point'], voxel_size, (0.2, 1.0, 0.2), 0.35
                ))

            label_anchor = obj.get('approach_point') or obj.get('object_point')

            if label_anchor is not None:
                confidence = obj.get('confidence')
                text = f'{obj.get("id")} {object_type}'

                if confidence is not None:
                    text += f' {float(confidence):.2f}'

                label = new_marker(
                    'labels', marker_id, Marker.TEXT_VIEW_FACING,
                    [label_anchor[0], label_anchor[1], label_anchor[2] + voxel_size],
                    max(voxel_size, 0.03), (1.0, 1.0, 1.0), 1.0
                )
                label.text = text
                markers.append(label)

        self.object_marker_pub.publish(MarkerArray(markers=markers))

    def abort_generate_scene(self, goal_handle, error_msg, json_matrix=''):
        self.get_logger().error(f'GenerateScene failed: {error_msg}')

        result = GenerateScene.Result()
        result.success = False
        result.error_msg = error_msg
        result.json_matrix = json_matrix
        result.json_objects_3d = '[]'

        goal_handle.abort()
        return result

    def execute_generate_scene_callback(self, goal_handle):
        self.get_logger().info('GenerateScene action goal received')

        goal = goal_handle.request
        service_timeout = float(self.get_parameter('service_timeout').value)

        self.get_logger().info(
            f'Goal model_path="{goal.model_path}", voxel_size={goal.voxel_size}, '
            f'include_matrix={goal.include_matrix}'
        )

        if not self.camera_info_received:
            return self.abort_generate_scene(
                goal_handle,
                f'No CameraInfo received from topic: {self.camera_info_topic}'
            )

        # Intrinsics of the image that is evaluated
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        self.publish_feedback(goal_handle, 'waiting_for_eval_scene_service', 0.10)

        if not self.eval_scene_client.wait_for_service(timeout_sec=5.0):
            return self.abort_generate_scene(
                goal_handle,
                f'EvalScene service not available: {SCENE_EVAL_SERVICE}'
            )

        self.publish_feedback(goal_handle, 'calling_eval_scene', 0.25)

        eval_request = EvalScene.Request()
        eval_request.model_path = goal.model_path
        eval_request.annotated_image_path = ''

        try:
            eval_response = self.call_service(
                self.eval_scene_client, eval_request, service_timeout
            )
        except TimeoutError as exc:
            return self.abort_generate_scene(goal_handle, f'EvalScene failed: {exc}')

        if not eval_response.success:
            return self.abort_generate_scene(
                goal_handle,
                f'EvalScene failed: {eval_response.error_msg}'
            )

        self.get_logger().info(
            f'EvalScene objects: {eval_response.json_objects}'
        )

        self.publish_feedback(goal_handle, 'waiting_for_capture_scene_service', 0.45)

        if not self.capture_scene_client.wait_for_service(timeout_sec=5.0):
            return self.abort_generate_scene(
                goal_handle,
                f'CaptureScene service not available: {CAPTURE_SCENE_SERVICE}'
            )

        self.publish_feedback(goal_handle, 'calling_capture_scene', 0.60)

        capture_request = CaptureScene.Request()
        capture_request.voxel_size = goal.voxel_size
        # Ray projection needs every voxel
        capture_request.include_matrix = True

        try:
            capture_response = self.call_service(
                self.capture_scene_client, capture_request, service_timeout
            )
        except TimeoutError as exc:
            return self.abort_generate_scene(goal_handle, f'CaptureScene failed: {exc}')

        if not capture_response.success:
            return self.abort_generate_scene(
                goal_handle,
                f'CaptureScene failed: {capture_response.error_msg}'
            )

        self.get_logger().info('CaptureScene returned voxel matrix')

        try:
            scene = parse_scene_grid(capture_response.json_matrix)
        except (ValueError, KeyError) as exc:
            return self.abort_generate_scene(
                goal_handle,
                f'Invalid CaptureScene json_matrix: {exc}',
                capture_response.json_matrix if goal.include_matrix else ''
            )

        # Every voxel only on request, the full matrix floods the console of ros2 action send_goal
        result_json_matrix = (
            capture_response.json_matrix if goal.include_matrix else scene.summary_json()
        )

        grid_frame = scene.frame_id or BASE_FRAME

        self.publish_feedback(goal_handle, 'waiting_for_tf', 0.70)

        try:
            camera_origin, rotation_frame_camera = self.get_camera_transform(grid_frame)
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
            RuntimeError,
            ValueError
        ) as exc:
            return self.abort_generate_scene(
                goal_handle,
                f'Could not get TF {grid_frame} <- {self.camera_frame}: {str(exc)}',
                result_json_matrix
            )

        self.get_logger().info(
            f'Camera origin in {grid_frame}: {camera_origin}'
        )

        self.publish_feedback(goal_handle, 'ray_casting_objects_to_voxels', 0.80)

        try:
            json_objects_3d = assign_objects_to_voxels(
                json_objects=eval_response.json_objects,
                scene=scene,
                camera_origin=camera_origin,
                rotation_frame_camera=rotation_frame_camera,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                ray_max_distance=float(self.get_parameter('ray_max_distance').value),
            )

        except Exception as exc:
            return self.abort_generate_scene(
                goal_handle,
                f'Ray casting failed: {str(exc)}',
                result_json_matrix
            )

        self.get_logger().info(
            f'Objects 3D: {json_objects_3d}'
        )

        if self.get_parameter('publish_markers').value:
            try:
                self.publish_object_markers(
                    json.loads(json_objects_3d),
                    scene.voxel_size,
                    grid_frame
                )
            except Exception as exc:
                self.get_logger().warning(f'Failed to publish object markers: {exc}')

        result = GenerateScene.Result()
        result.success = True
        result.error_msg = ''
        result.json_matrix = result_json_matrix
        result.json_objects_3d = json_objects_3d

        # PlanAndExecute finds its target here
        self.latest_objects_3d = json.loads(json_objects_3d)
        self.latest_voxel_size = scene.voxel_size

        self.publish_feedback(goal_handle, 'done', 1.0)

        goal_handle.succeed()
        return result

    def publish_motion_feedback(self, goal_handle, action, current_step, progress):
        """Publish feedback of PlanAndExecute or MoveBetween, both have the same fields."""
        feedback = action.Feedback()
        feedback.current_step = current_step
        feedback.progress = float(progress)
        goal_handle.publish_feedback(feedback)

        self.get_logger().info(
            f'{action.__name__} feedback: {current_step} ({progress:.2f})'
        )

    def create_move_group_goal(self, joint_positions):
        """Create a joint space goal, every run ends in the same arm configuration."""
        speed_scaling = float(self.get_parameter('speed_scaling').value)

        constraints = Constraints()
        constraints.name = 'joint_goal'

        for name, position in zip(UR_JOINT_NAMES, joint_positions):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(position)
            constraint.tolerance_above = JOINT_TOLERANCE
            constraint.tolerance_below = JOINT_TOLERANCE
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)

        request = MotionPlanRequest()
        request.group_name = MOVE_GROUP_NAME
        request.num_planning_attempts = 10
        request.allowed_planning_time = 8.0
        request.max_velocity_scaling_factor = speed_scaling
        request.max_acceleration_scaling_factor = speed_scaling
        request.goal_constraints.append(constraints)

        planning_options = PlanningOptions()
        # Executed separately after the joint travel check
        planning_options.plan_only = True
        planning_options.look_around = False
        planning_options.replan = False

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options = planning_options

        return goal_msg

    def wait_for_future(self, future, timeout_sec, description):
        """Block until the future is done, the result arrives on another executor thread."""
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())

        if not done.wait(timeout_sec):
            raise TimeoutError(f'No {description} within {timeout_sec:.1f} s')

        return future.result()

    def run_motion(self, goal_handle, client, motion_goal, name):
        """
        Send a goal to MoveIt and wait until it finishes.

        Returns the MoveIt result, or None when PlanAndExecute was canceled, the
        goal is then canceled too. Raises on timeout and on any MoveIt error.
        """
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f'{name} action server not available')

        motion_handle = self.wait_for_future(
            client.send_goal_async(motion_goal), 10.0, f'{name} goal response'
        )

        if not motion_handle.accepted:
            raise RuntimeError(f'{name} goal was rejected')

        result_future = motion_handle.get_result_async()
        done = threading.Event()
        result_future.add_done_callback(lambda _: done.set())

        timeout_sec = float(self.get_parameter('motion_timeout').value)
        deadline = time.monotonic() + timeout_sec

        while not done.wait(0.1):
            if goal_handle.is_cancel_requested:
                motion_handle.cancel_goal_async()
                done.wait(5.0)
                return None

            if time.monotonic() > deadline:
                motion_handle.cancel_goal_async()
                raise TimeoutError(f'{name} did not finish within {timeout_sec:.1f} s')

        motion_result = result_future.result().result
        error_code = motion_result.error_code.val

        if error_code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'{name} failed with MoveIt error code {error_code}')

        return motion_result

    def create_tool0_pose(self, position, rotation):
        quaternion = rotation_matrix_to_quaternion(rotation)

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(quaternion[0])
        pose.orientation.y = float(quaternion[1])
        pose.orientation.z = float(quaternion[2])
        pose.orientation.w = float(quaternion[3])

        return pose

    def publish_tool0_poses(self, poses):
        pose_array = PoseArray()
        pose_array.header.frame_id = BASE_FRAME
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.poses = list(poses)

        self.tool0_pose_pub.publish(pose_array)

    def check_joint_travel(self, trajectory, name):
        """Reject a planned trajectory that turns one joint further than max_joint_travel."""
        limit = float(self.get_parameter('max_joint_travel').value)
        joint_trajectory = trajectory.joint_trajectory
        joint_name, travel = largest_joint_travel(
            joint_trajectory.joint_names,
            joint_trajectory.points
        )

        self.get_logger().info(
            f'{name}: {len(joint_trajectory.points)} points, '
            f'largest joint travel {math.degrees(travel):.1f} deg on {joint_name}'
        )

        if travel > limit:
            raise RuntimeError(
                f'{name} rejected: {joint_name} would travel {math.degrees(travel):.1f} deg, '
                f'max_joint_travel is {math.degrees(limit):.1f} deg'
            )

    def resolve_tcp_target(self, object_id, point, name):
        """Return the TCP target: hover_height above the object, or the point as given."""
        if object_id >= 0:
            approach_point = find_object_target(self.latest_objects_3d, object_id)

            return hover_point(
                approach_point,
                self.latest_voxel_size,
                float(self.get_parameter('hover_height').value)
            )

        target = (point.x, point.y, point.z)

        if not any(target):
            raise ValueError(f'Set {name}_object_id or {name}_point')

        return target

    def check_points_clear(self, samples, name):
        """Reject TCP samples that move into an occupied voxel of the last capture."""
        service_timeout = float(self.get_parameter('service_timeout').value)

        if not self.check_points_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'Service not available: {CHECK_POINTS_SERVICE}')

        request = CheckPoints.Request()
        request.points = [Point(x=x, y=y, z=z) for x, y, z in samples]
        response = self.call_service(self.check_points_client, request, service_timeout)

        if not response.success:
            raise RuntimeError(f'CheckPoints failed: {response.error_msg}')

        blocked = first_blocked_sample(list(response.occupied))

        if blocked is not None:
            point = [round(c, 3) for c in samples[blocked]]
            raise RuntimeError(
                f'{name} rejected: the TCP would enter an occupied voxel at {point}'
            )

        self.get_logger().info(f'{name} clear: {len(samples)} TCP samples checked')

    def plan_joint_move(self, goal_handle, target, rotation, tcp_offset, name):
        """
        Plan a checked joint move from the current state to the TCP target.

        Returns the trajectory, or None when the action was canceled.
        """
        tool0_target = tool0_position(target, rotation, tcp_offset)
        joints = inverse_kinematics(tool0_target, rotation)

        self.get_logger().info(
            f'{name}: TCP target {[round(c, 3) for c in target]}, '
            f'joints {[round(j, 4) for j in joints]}'
        )

        plan = self.run_motion(
            goal_handle,
            self.move_group_client,
            self.create_move_group_goal(joints),
            'MoveGroup planning'
        )

        if plan is None:
            return None

        trajectory = plan.planned_trajectory
        self.check_joint_travel(trajectory, name)

        joint_trajectory = trajectory.joint_trajectory
        self.check_points_clear(
            sample_tcp_path(joint_trajectory.joint_names, joint_trajectory.points, tcp_offset),
            name
        )

        return trajectory

    def plan_tcp_path(self, start, goal):
        """Return the PlanPath waypoints from start to goal, which must stay exactly in place."""
        service_timeout = float(self.get_parameter('service_timeout').value)

        if not self.plan_path_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'Service not available: {PLAN_PATH_SERVICE}')

        request = PlanPath.Request()
        request.start_point = Point(x=float(start[0]), y=float(start[1]), z=float(start[2]))
        request.goal_point = Point(x=float(goal[0]), y=float(goal[1]), z=float(goal[2]))

        response = self.call_service(self.plan_path_client, request, service_timeout)

        if not response.success:
            raise RuntimeError(f'PlanPath failed: {response.error_msg}')

        waypoints = [(w.x, w.y, w.z) for w in response.waypoints]

        # PlanPath moves a blocked point to a free voxel, that would not be above the object
        for name, requested, planned in (
                ('start', start, waypoints[0]),
                ('goal', goal, waypoints[-1])):
            if math.dist(requested, planned) > 0.001:
                raise RuntimeError(
                    f'The {name} {[round(c, 3) for c in requested]} is too close to an obstacle '
                    f'for the gripper (tool_radius of the path planner)'
                )

        self.get_logger().info(
            f'{len(waypoints)} TCP waypoints: '
            f'{[[round(c, 3) for c in waypoint] for waypoint in waypoints]}'
        )

        return waypoints

    def compute_cartesian_path(self, tool0_poses, start_trajectory):
        """Return a slowed down trajectory through the tool0 poses from start_trajectory's end."""
        service_timeout = float(self.get_parameter('service_timeout').value)

        if not self.cartesian_path_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'Service not available: {CARTESIAN_PATH_SERVICE}')

        start = start_trajectory.joint_trajectory

        request = GetCartesianPath.Request()
        request.header.frame_id = BASE_FRAME
        request.start_state.is_diff = True
        request.start_state.joint_state.name = list(start.joint_names)
        request.start_state.joint_state.position = list(start.points[-1].positions)
        request.group_name = MOVE_GROUP_NAME
        request.link_name = MOVEIT_TARGET_LINK
        request.max_step = CARTESIAN_MAX_STEP
        request.jump_threshold = 0.0
        request.revolute_jump_threshold = CARTESIAN_REVOLUTE_JUMP
        request.avoid_collisions = True
        request.waypoints = list(tool0_poses)

        response = self.call_service(self.cartesian_path_client, request, service_timeout)

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'Cartesian path failed with error code {response.error_code.val}')

        if response.fraction < 0.999:
            solution = response.solution.joint_trajectory

            if solution.points:
                last_joints = [round(position, 4) for position in solution.points[-1].positions]
                self.get_logger().error(
                    f'Cartesian path stopped after {len(solution.points)} points '
                    f'at {dict(zip(solution.joint_names, last_joints))}'
                )

            raise RuntimeError(
                f'Cartesian path covers only {response.fraction * 100.0:.1f} % of the waypoints'
            )

        rescale_trajectory(
            response.solution.joint_trajectory.points,
            float(self.get_parameter('speed_scaling').value)
        )

        return response.solution

    def execute_trajectory(self, goal_handle, trajectory):
        """Execute a planned trajectory, returns False when the action was canceled."""
        execute_goal = ExecuteTrajectory.Goal()
        execute_goal.trajectory = trajectory

        return self.run_motion(
            goal_handle,
            self.execute_trajectory_client,
            execute_goal,
            'ExecuteTrajectory'
        ) is not None

    def run_motion_action(self, goal_handle, action, run):
        """Run one motion action at a time and turn its outcome into the action result."""
        name = action.__name__
        self.get_logger().info(f'{name} action goal received')

        result = action.Result()
        result.success = False

        if not self.motion_lock.acquire(blocking=False):
            result.error_msg = 'Another motion is running'
            self.get_logger().error(f'{name} failed: {result.error_msg}')
            goal_handle.abort()
            return result

        try:
            completed = run(goal_handle)
        except Exception as exc:
            result.error_msg = str(exc)
            self.get_logger().error(f'{name} failed: {result.error_msg}')
            goal_handle.abort()
            return result
        finally:
            self.motion_lock.release()

        if not completed:
            result.error_msg = 'Canceled'
            self.get_logger().info(f'{name} canceled')
            goal_handle.canceled()
            return result

        self.publish_motion_feedback(goal_handle, action, 'done', 1.0)
        result.success = True
        result.error_msg = ''
        goal_handle.succeed()
        return result

    def execute_plan_and_execute_callback(self, goal_handle):
        return self.run_motion_action(goal_handle, PlanAndExecute, self.run_plan_and_execute)

    def execute_move_between_callback(self, goal_handle):
        return self.run_motion_action(goal_handle, MoveBetween, self.run_move_between)

    def motion_settings(self):
        """Return tcp_offset, dry_run and the fixed tool0 rotation from the parameters."""
        return (
            float(self.get_parameter('tcp_offset').value),
            bool(self.get_parameter('dry_run').value),
            pendant_to_tool0_rotation(list(self.get_parameter('tcp_rotation').value)),
        )

    def run_plan_and_execute(self, goal_handle):
        """Move above an object in one joint move, returns False when canceled."""
        goal = goal_handle.request
        tcp_offset, dry_run, rotation = self.motion_settings()

        target = self.resolve_tcp_target(goal.object_id, goal.goal_point, 'goal')
        self.publish_tool0_poses([
            self.create_tool0_pose(tool0_position(target, rotation, tcp_offset), rotation)
        ])

        self.publish_motion_feedback(goal_handle, PlanAndExecute, 'planning_motion', 0.20)
        trajectory = self.plan_joint_move(
            goal_handle, target, rotation, tcp_offset, 'Move to target'
        )

        if trajectory is None:
            return False

        if dry_run:
            self.get_logger().info('dry_run: plan is valid, the robot was not moved')
            return True

        self.publish_motion_feedback(goal_handle, PlanAndExecute, 'moving_to_target', 0.70)
        return self.execute_trajectory(goal_handle, trajectory)

    def run_move_between(self, goal_handle):
        """
        Move above the start, then along the path to above the goal without stopping.

        Everything is planned and checked before the robot moves. Returns False
        when canceled.
        """
        goal = goal_handle.request
        tcp_offset, dry_run, rotation = self.motion_settings()

        start = self.resolve_tcp_target(goal.start_object_id, goal.start_point, 'start')
        target = self.resolve_tcp_target(goal.goal_object_id, goal.goal_point, 'goal')

        self.publish_motion_feedback(goal_handle, MoveBetween, 'planning_path', 0.10)
        path_points = round_corners(
            self.plan_tcp_path(start, target),
            float(self.get_parameter('blend_radius').value)
        )
        self.check_points_clear(path_points, 'Path to goal')

        tool0_poses = [
            self.create_tool0_pose(tool0_position(point, rotation, tcp_offset), rotation)
            for point in path_points
        ]
        self.publish_tool0_poses(tool0_poses)

        self.publish_motion_feedback(goal_handle, MoveBetween, 'planning_move_to_start', 0.30)
        start_trajectory = self.plan_joint_move(
            goal_handle, start, rotation, tcp_offset, 'Move to start'
        )

        if start_trajectory is None:
            return False

        self.publish_motion_feedback(goal_handle, MoveBetween, 'planning_cartesian_path', 0.50)
        # The first pose is the start itself, where the start move ends
        path_trajectory = self.compute_cartesian_path(tool0_poses[1:], start_trajectory)
        self.check_joint_travel(path_trajectory, 'Path to goal')

        if dry_run:
            self.get_logger().info('dry_run: plans are valid, the robot was not moved')
            return True

        self.publish_motion_feedback(goal_handle, MoveBetween, 'moving_to_start', 0.60)

        if not self.execute_trajectory(goal_handle, start_trajectory):
            return False

        self.publish_motion_feedback(goal_handle, MoveBetween, 'moving_to_goal', 0.80)
        return self.execute_trajectory(goal_handle, path_trajectory)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrchestratorURServer()
    # Several threads: action callbacks block while waiting for responses of services and motions
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
