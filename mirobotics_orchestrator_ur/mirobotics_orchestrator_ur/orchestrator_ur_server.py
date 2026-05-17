import math
import json

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped,Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from shape_msgs.msg import SolidPrimitive

from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from mirobotics_msg.action import GenerateScene, PlanAndExecute
from mirobotics_msg.srv import EvalScene, CaptureScene, PlanPath

from mirobotics_orchestrator_ur.ray_casting import assign_objects_to_voxels

BASE_FRAME = 'base_link'
SCENE_EVAL_SERVICE = '/mirobotics_scene_eval/mirobotics_eval_scene'
CAPTURE_SCENE_SERVICE = '/capture_scene'
PLAN_PATH_SERVICE = '/plan_path'
RAY_STEP = 0.02
RAY_MAX_DISTANCE = 3.0
OCCUPIED_PASSABLE_VALUE = 0.0

MOVE_GROUP_ACTION = '/move_action'
MOVE_GROUP_NAME = 'ur_manipulator'
MOVEIT_TARGET_LINK = 'tool0'
TOOL0_TCP_OFFSET_Z = -0.1866
POSITION_TOLERANCE = 0.03

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

        self.latest_json_matrix = ''
        self.latest_voxel_size = None

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

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            MOVE_GROUP_ACTION
        )

        self.get_logger().info(
            f'Using camera info topic: {self.camera_info_topic}'
        )

        self.get_logger().info(
            'mirobotics_orchestrator_ur action server started.'
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

        self.latest_json_matrix = updated_json_matrix
        self.latest_voxel_size = goal.voxel_size

        self.publish_feedback(goal_handle, 'done', 1.0)

        goal_handle.succeed()
        return result

    def publish_plan_feedback(self, goal_handle, current_step, progress):
        feedback = PlanAndExecute.Feedback()
        feedback.current_step = current_step
        feedback.progress = float(progress)
        goal_handle.publish_feedback(feedback)

        self.get_logger().info(
            f'PlanAndExecute feedback: {current_step} ({progress:.2f})'
        )

    def find_voxel_by_id(self, voxel_id):
        if not self.latest_json_matrix:
            raise RuntimeError('No voxel matrix available. Run generate_scene first.')

        data = json.loads(self.latest_json_matrix)

        if not isinstance(data, dict):
            raise RuntimeError('latest_json_matrix must be a JSON object with key "matrix"')

        if 'matrix' not in data:
            raise RuntimeError('latest_json_matrix does not contain key "matrix"')

        matrix = data['matrix']
        target_id = int(voxel_id)

        if target_id < 0 or target_id >= len(matrix):
            raise RuntimeError(
                f'Voxel id {target_id} is outside matrix row range 0 -> {len(matrix) - 1}'
            )

        voxel = matrix[target_id]

        if not isinstance(voxel, list) or len(voxel) < 5:
            raise RuntimeError(f'Invalid voxel row at index {target_id}: {voxel}')

        row_id = int(float(voxel[0]))

        if row_id != target_id:
            self.get_logger().warn(
                f'Voxel row index {target_id} has row id {row_id}. '
                f'Using row index as requested.'
            )

        return {
            'id': row_id,
            'x': float(voxel[1]),
            'y': float(voxel[2]),
            'z': float(voxel[3]),
            'passable': float(voxel[4]),
        }

    def create_tool0_position_constraint(self, x, y, z):
        constraint = PositionConstraint()
        constraint.header.frame_id = BASE_FRAME
        constraint.link_name = MOVEIT_TARGET_LINK
        constraint.weight = 1.0

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [
            POSITION_TOLERANCE,
            POSITION_TOLERANCE,
            POSITION_TOLERANCE,
        ]

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.w = 1.0

        constraint.constraint_region.primitives.append(primitive)
        constraint.constraint_region.primitive_poses.append(pose)

        return constraint

    def create_move_group_goal(self, tool0_x, tool0_y, tool0_z):
        constraints = Constraints()
        constraints.name = 'tool0_position_only_goal'
        constraints.position_constraints.append(
            self.create_tool0_position_constraint(
                tool0_x,
                tool0_y,
                tool0_z
            )
        )

        request = MotionPlanRequest()
        request.group_name = MOVE_GROUP_NAME
        request.num_planning_attempts = 10
        request.allowed_planning_time = 8.0
        request.max_velocity_scaling_factor = 0.2
        request.max_acceleration_scaling_factor = 0.2
        request.goal_constraints.append(constraints)

        planning_options = PlanningOptions()
        planning_options.plan_only = False
        planning_options.look_around = False
        planning_options.replan = True
        planning_options.replan_attempts = 3

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options = planning_options

        return goal_msg

    async def execute_plan_and_execute_callback(self, goal_handle):
        self.get_logger().info('PlanAndExecute action goal received')

        result = PlanAndExecute.Result()
        goal = goal_handle.request

        try:
            self.publish_plan_feedback(goal_handle, 'checking_voxel_matrix', 0.05)

            start_voxel = self.find_voxel_by_id(goal.start_voxel_id)
            goal_voxel = self.find_voxel_by_id(goal.goal_voxel_id)

            if start_voxel['passable'] <= 0.0:
                result.success = False
                result.error_msg = f'Start voxel {goal.start_voxel_id} is not passable'
                goal_handle.abort()
                return result

            if goal_voxel['passable'] <= 0.0:
                result.success = False
                result.error_msg = f'Goal voxel {goal.goal_voxel_id} is not passable'
                goal_handle.abort()
                return result

            self.publish_plan_feedback(goal_handle, 'creating_tool0_goal', 0.20)

            tool0_x = goal_voxel['x']
            tool0_y = goal_voxel['y']
            tool0_z = goal_voxel['z'] - TOOL0_TCP_OFFSET_Z

            self.get_logger().info(
                f'Goal voxel center: '
                f'x={goal_voxel["x"]:.3f}, '
                f'y={goal_voxel["y"]:.3f}, '
                f'z={goal_voxel["z"]:.3f}'
            )

            self.get_logger().info(
                f'Tool0 target with TCP offset: '
                f'x={tool0_x:.3f}, y={tool0_y:.3f}, z={tool0_z:.3f}'
            )

            self.publish_plan_feedback(goal_handle, 'waiting_for_move_group', 0.30)

            if not self.move_group_client.wait_for_server(timeout_sec=10.0):
                result.success = False
                result.error_msg = f'MoveGroup action server not available: {MOVE_GROUP_ACTION}'
                goal_handle.abort()
                return result

            move_goal = self.create_move_group_goal(
                tool0_x,
                tool0_y,
                tool0_z
            )

            self.publish_plan_feedback(goal_handle, 'sending_goal_to_moveit', 0.45)

            send_goal_future = self.move_group_client.send_goal_async(move_goal)
            move_goal_handle = await send_goal_future

            if not move_goal_handle.accepted:
                result.success = False
                result.error_msg = 'MoveGroup goal was rejected'
                goal_handle.abort()
                return result

            self.publish_plan_feedback(goal_handle, 'planning_and_executing', 0.65)

            move_result_future = move_goal_handle.get_result_async()
            move_result_response = await move_result_future

            move_result = move_result_response.result
            error_code = move_result.error_code.val

            if error_code != 1:
                result.success = False
                result.error_msg = f'MoveGroup failed with error code: {error_code}'
                goal_handle.abort()
                return result

            self.publish_plan_feedback(goal_handle, 'done', 1.0)

            result.success = True
            result.error_msg = ''
            goal_handle.succeed()
            return result

        except Exception as exc:
            result.success = False
            result.error_msg = f'PlanAndExecute failed: {str(exc)}'
            goal_handle.abort()
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