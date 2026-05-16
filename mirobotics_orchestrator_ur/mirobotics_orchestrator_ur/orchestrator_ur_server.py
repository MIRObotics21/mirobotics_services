import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.duration import Duration

from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory

from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from mirobotics_msg.action import GenerateScene, PlanAndExecute
from mirobotics_msg.srv import EvalScene, CaptureScene, PlanPath

from mirobotics_orchestrator_ur.ray_casting import assign_objects_to_voxels
from mirobotics_orchestrator_ur.moveit_execution import voxel_path_to_poses

COMPUTE_CARTESIAN_PATH_SERVICE = '/compute_cartesian_path'
EXECUTE_TRAJECTORY_ACTION = '/execute_trajectory'

MOVE_GROUP_NAME = 'ur_manipulator'
MOVEIT_BASE_FRAME = 'base_link'
MOVEIT_TIP_LINK = 'tool0'   # change later to 'rg2_tcp'
CARTESIAN_MAX_STEP = 0.01
CARTESIAN_JUMP_THRESHOLD = 0.0
MIN_WAYPOINT_DISTANCE = 0.03
TCP_Z_OFFSET = 0.1866

BASE_FRAME = 'base_link'

SCENE_EVAL_SERVICE = '/mirobotics_scene_eval/mirobotics_eval_scene'
CAPTURE_SCENE_SERVICE = '/capture_scene'
PLAN_PATH_SERVICE = '/plan_path'

RAY_STEP = 0.02
RAY_MAX_DISTANCE = 3.0
OCCUPIED_PASSABLE_VALUE = 0.0


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

        self.camera_info_received = False
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.image_width = None
        self.image_height = None
        self.camera_frame = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        self.eval_scene_client = self.create_client(
            EvalScene,
            SCENE_EVAL_SERVICE
        )

        self.capture_scene_client = self.create_client(
            CaptureScene,
            CAPTURE_SCENE_SERVICE
        )

        self.plan_path_client = self.create_client(
            PlanPath,
            PLAN_PATH_SERVICE
        )

        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            COMPUTE_CARTESIAN_PATH_SERVICE
        )

        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            EXECUTE_TRAJECTORY_ACTION
        )

        self.latest_json_matrix = ''
        self.latest_json_objects_3d = '[]'
        self.latest_voxel_size = None
        self.scene_ready = False

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
            self.execute_plan_and_execute_callback
        )

        self.get_logger().info(
            f'Using camera info topic: {self.camera_info_topic}'
        )
        self.get_logger().info(
            f'Using EvalScene service: {SCENE_EVAL_SERVICE}'
        )
        self.get_logger().info(
            f'Using CaptureScene service: {CAPTURE_SCENE_SERVICE}'
        )
        self.get_logger().info(
            'mirobotics_orchestrator_ur action server started: /generate_scene'
        )

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        self.camera_info_received = True

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.image_width = msg.width
        self.image_height = msg.height
        self.camera_frame = msg.header.frame_id

        self.get_logger().info(
            f'Camera intrinsics received: '
            f'fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}'
        )
        self.get_logger().info(
            f'Image size: {self.image_width}x{self.image_height}'
        )
        self.get_logger().info(
            f'Camera frame: {self.camera_frame}'
        )

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
        Converts quaternion into 3x3 rotation matrix.

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

    def get_camera_transform_to_base(self):
        """
        Returns:
            camera_origin_base: [x, y, z]
            rotation_base_camera: 3x3 rotation matrix
        """

        if self.camera_frame is None:
            raise RuntimeError('Camera frame is unknown because CameraInfo was not received')

        transform = self.tf_buffer.lookup_transform(
            BASE_FRAME,
            self.camera_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=2.0)
        )

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        camera_origin_base = [
            translation.x,
            translation.y,
            translation.z,
        ]

        rotation_base_camera = self.quaternion_to_rotation_matrix(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w
        )

        return camera_origin_base, rotation_base_camera

    async def execute_generate_scene_callback(self, goal_handle):
        self.get_logger().info('GenerateScene action goal received')

        goal = goal_handle.request
        result = GenerateScene.Result()

        self.get_logger().info(
            f'Goal model_path="{goal.model_path}", voxel_size={goal.voxel_size}'
        )

        if not self.camera_info_received:
            result.success = False
            result.error_msg = (
                f'No CameraInfo received from topic: {self.camera_info_topic}'
            )
            result.json_matrix = ''
            result.json_objects_3d = '[]'
            goal_handle.abort()
            return result

        self.publish_feedback(goal_handle, 'waiting_for_tf', 0.05)

        try:
            camera_origin_base, rotation_base_camera = self.get_camera_transform_to_base()
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
            RuntimeError,
            ValueError
        ) as exc:
            result.success = False
            result.error_msg = (
                f'Could not get TF {BASE_FRAME} <- {self.camera_frame}: {str(exc)}'
            )
            result.json_matrix = ''
            result.json_objects_3d = '[]'
            goal_handle.abort()
            return result

        self.get_logger().info(
            f'Camera origin in {BASE_FRAME}: {camera_origin_base}'
        )

        self.publish_feedback(goal_handle, 'waiting_for_eval_scene_service', 0.10)

        if not self.eval_scene_client.wait_for_service(timeout_sec=5.0):
            result.success = False
            result.error_msg = f'EvalScene service not available: {SCENE_EVAL_SERVICE}'
            result.json_matrix = ''
            result.json_objects_3d = '[]'
            goal_handle.abort()
            return result

        self.publish_feedback(goal_handle, 'calling_eval_scene', 0.25)

        eval_request = EvalScene.Request()
        eval_request.model_path = goal.model_path
        eval_request.annotated_image_path = ''

        eval_future = self.eval_scene_client.call_async(eval_request)
        eval_response = await eval_future

        if not eval_response.success:
            result.success = False
            result.error_msg = f'EvalScene failed: {eval_response.error_msg}'
            result.json_matrix = ''
            result.json_objects_3d = '[]'
            goal_handle.abort()
            return result

        self.get_logger().info(
            f'EvalScene objects: {eval_response.json_objects}'
        )

        self.publish_feedback(goal_handle, 'waiting_for_capture_scene_service', 0.45)

        if not self.capture_scene_client.wait_for_service(timeout_sec=5.0):
            result.success = False
            result.error_msg = f'CaptureScene service not available: {CAPTURE_SCENE_SERVICE}'
            result.json_matrix = ''
            result.json_objects_3d = '[]'
            goal_handle.abort()
            return result

        self.publish_feedback(goal_handle, 'calling_capture_scene', 0.60)

        capture_request = CaptureScene.Request()
        capture_request.voxel_size = goal.voxel_size

        capture_future = self.capture_scene_client.call_async(capture_request)
        capture_response = await capture_future

        if not capture_response.success:
            result.success = False
            result.error_msg = f'CaptureScene failed: {capture_response.error_msg}'
            result.json_matrix = ''
            result.json_objects_3d = '[]'
            goal_handle.abort()
            return result

        self.get_logger().info('CaptureScene returned voxel matrix')

        self.publish_feedback(goal_handle, 'ray_casting_objects_to_voxels', 0.80)

        try:
            json_objects_3d, updated_json_matrix = assign_objects_to_voxels(
                json_objects=eval_response.json_objects,
                json_matrix=capture_response.json_matrix,
                camera_origin_base=camera_origin_base,
                rotation_base_camera=rotation_base_camera,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                ray_step=RAY_STEP,
                ray_max_distance=RAY_MAX_DISTANCE,
                occupied_passable_value=OCCUPIED_PASSABLE_VALUE
            )

        except Exception as exc:
            result.success = False
            result.error_msg = f'Ray casting failed: {str(exc)}'
            result.json_matrix = capture_response.json_matrix
            result.json_objects_3d = '[]'
            goal_handle.abort()
            return result

        self.get_logger().info(
            f'Objects 3D: {json_objects_3d}'
        )

        result.success = True
        result.error_msg = ''
        result.json_matrix = updated_json_matrix
        result.json_objects_3d = json_objects_3d

        self.latest_json_matrix = capture_response.json_matrix
        self.latest_json_objects_3d = json_objects_3d
        self.latest_voxel_size = goal.voxel_size
        self.scene_ready = True
        self.publish_feedback(goal_handle, 'done', 1.0)

        goal_handle.succeed()
        return result

    async def execute_plan_and_execute_callback(self, goal_handle):
        self.get_logger().info('PlanAndExecute action goal received')

        goal = goal_handle.request
        result = PlanAndExecute.Result()

        self.get_logger().info(
            f'Goal start_voxel_id={goal.start_voxel_id}, '
            f'goal_voxel_id={goal.goal_voxel_id}'
        )

        feedback = PlanAndExecute.Feedback()
        feedback.current_step = 'checking_plan_path_service'
        feedback.progress = 0.1
        goal_handle.publish_feedback(feedback)

        if not self.plan_path_client.wait_for_service(timeout_sec=5.0):
            result.success = False
            result.error_msg = f'PlanPath service not available: {PLAN_PATH_SERVICE}'
            goal_handle.abort()
            return result

        feedback.current_step = 'calling_plan_path'
        feedback.progress = 0.4
        goal_handle.publish_feedback(feedback)

        request = PlanPath.Request()
        request.start_id = goal.start_voxel_id
        request.goal_id = goal.goal_voxel_id

        future = self.plan_path_client.call_async(request)
        response = await future

        if not response.success:
            result.success = False
            result.error_msg = f'PlanPath failed: {response.error_msg}'
            goal_handle.abort()
            return result

        self.get_logger().info(
            f'PlanPath returned json_path: {response.json_path}'
        )

        feedback.current_step = 'converting_path_to_waypoints'
        feedback.progress = 0.55
        goal_handle.publish_feedback(feedback)

        waypoints = voxel_path_to_poses(
            response.json_path,
            min_distance=MIN_WAYPOINT_DISTANCE,
        )

        if len(waypoints) < 2:
            result.success = False
            result.error_msg = f'Not enough waypoints for execution: {len(waypoints)}'
            goal_handle.abort()
            return result

        self.get_logger().info(
            f'Converted PlanPath result to {len(waypoints)} MoveIt waypoints'
        )

        feedback.current_step = 'waiting_for_moveit_cartesian_service'
        feedback.progress = 0.60
        goal_handle.publish_feedback(feedback)

        if not self.cartesian_path_client.wait_for_service(timeout_sec=5.0):
            result.success = False
            result.error_msg = 'MoveIt /compute_cartesian_path service not available'
            goal_handle.abort()
            return result

        feedback.current_step = 'computing_cartesian_path'
        feedback.progress = 0.70
        goal_handle.publish_feedback(feedback)

        cartesian_request = GetCartesianPath.Request()
        cartesian_request.header.frame_id = MOVEIT_BASE_FRAME
        cartesian_request.group_name = MOVE_GROUP_NAME
        cartesian_request.link_name = MOVEIT_TIP_LINK
        cartesian_request.waypoints = waypoints
        cartesian_request.max_step = CARTESIAN_MAX_STEP
        cartesian_request.jump_threshold = CARTESIAN_JUMP_THRESHOLD
        cartesian_request.avoid_collisions = True

        cartesian_future = self.cartesian_path_client.call_async(cartesian_request)
        cartesian_response = await cartesian_future

        self.get_logger().info(
            f'MoveIt Cartesian path fraction: {cartesian_response.fraction}'
        )

        if cartesian_response.fraction < 0.95:
            result.success = False
            result.error_msg = (
                f'MoveIt Cartesian path incomplete. '
                f'Fraction={cartesian_response.fraction:.3f}'
            )
            goal_handle.abort()
            return result

        feedback.current_step = 'waiting_for_execute_trajectory_action'
        feedback.progress = 0.80
        goal_handle.publish_feedback(feedback)

        if not self.execute_trajectory_client.wait_for_server(timeout_sec=5.0):
            result.success = False
            result.error_msg = 'MoveIt /execute_trajectory action not available'
            goal_handle.abort()
            return result

        feedback.current_step = 'executing_trajectory'
        feedback.progress = 0.90
        goal_handle.publish_feedback(feedback)

        execute_goal = ExecuteTrajectory.Goal()
        execute_goal.trajectory = cartesian_response.solution

        send_goal_future = self.execute_trajectory_client.send_goal_async(execute_goal)
        execute_goal_handle = await send_goal_future

        if not execute_goal_handle.accepted:
            result.success = False
            result.error_msg = 'MoveIt ExecuteTrajectory goal was rejected'
            goal_handle.abort()
            return result

        execute_result_future = execute_goal_handle.get_result_async()
        execute_result = await execute_result_future

        if execute_result.result.error_code.val != 1:
            result.success = False
            result.error_msg = (
                f'MoveIt execution failed with error code: '
                f'{execute_result.result.error_code.val}'
            )
            goal_handle.abort()
            return result

        feedback.current_step = 'done'
        feedback.progress = 1.0
        goal_handle.publish_feedback(feedback)

        result.success = True
        result.error_msg = ''

        goal_handle.succeed()
        return result

def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrchestratorURServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()