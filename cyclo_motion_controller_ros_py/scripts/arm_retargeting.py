# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yeonguk Kim

"""ROS node that retargets arm poses into robot elbow and wrist goals."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from retargeting.robot_wrapper import RobotWrapper
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


QOS_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

@dataclass(frozen=True)
class RobotArmGeometry:
    """Robot limb lengths for one arm."""

    upper_arm_length: float
    forearm_length: float


@dataclass
class ArmPoseState:
    """Latest human shoulder, elbow, and wrist poses for one arm."""

    shoulder: Optional[PoseStamped] = None
    elbow: Optional[PoseStamped] = None
    wrist: Optional[PoseStamped] = None


class ArmRetargetingTeleop(Node):
    """Retarget human arm directions into robot elbow and wrist goals."""

    def __init__(self) -> None:
        """Initialize parameters, subscriptions, and publishers."""
        super().__init__('arm_retargeting_teleop')

        models_share_dir = get_package_share_directory(
            'cyclo_motion_controller_models'
        )
        default_urdf_path = os.path.join(
            models_share_dir,
            'models',
            'ai_worker',
            'ffw_sg2_follower.urdf',
        )

        urdf_path = self.declare_parameter(
            'urdf_path',
            default_urdf_path,
        ).value

        self.right_shoulder_link = self.declare_parameter(
            'right_shoulder_link',
            'arm_r_link2',
        ).value
        self.right_elbow_link = self.declare_parameter(
            'right_elbow_link',
            'arm_r_link4',
        ).value
        self.right_wrist_link = self.declare_parameter(
            'right_wrist_link',
            'arm_r_link7',
        ).value
        self.left_shoulder_link = self.declare_parameter(
            'left_shoulder_link',
            'arm_l_link2',
        ).value
        self.left_elbow_link = self.declare_parameter(
            'left_elbow_link',
            'arm_l_link4',
        ).value
        self.left_wrist_link = self.declare_parameter(
            'left_wrist_link',
            'arm_l_link7',
        ).value

        self.right_pose_state = ArmPoseState()
        self.left_pose_state = ArmPoseState()
        self.base_frame = self.declare_parameter(
            'base_frame',
            'base_link',
        ).value
        self.wrist_distance_priority = 1.0
        self.wrist_priority_reference_distance = 0.3
        self.wrist_priority_min_scale = 0.25
        self.wrist_priority_decay_rate = 4.0
        self.max_wrist_distance_correction = 0.3
        self.wrist_distance_smoothing_alpha = 0.2
        self.human_wrist_to_fingertip = 0.22
        self.robot_wrist_to_fingertip = 0.25
        self.wrist_distance_scale = self._compute_wrist_distance_scale(
            human_wrist_to_fingertip=self.human_wrist_to_fingertip,
            robot_wrist_to_fingertip=self.robot_wrist_to_fingertip,
        )
        self._filtered_right_wrist_target: Optional[np.ndarray] = None
        self._filtered_left_wrist_target: Optional[np.ndarray] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        robot = RobotWrapper(urdf_path)
        self.right_geometry = self._compute_robot_geometry(
            robot,
            shoulder_link=self.right_shoulder_link,
            elbow_link=self.right_elbow_link,
            wrist_link=self.right_wrist_link,
        )
        self.left_geometry = self._compute_robot_geometry(
            robot,
            shoulder_link=self.left_shoulder_link,
            elbow_link=self.left_elbow_link,
            wrist_link=self.left_wrist_link,
        )

        r_shoulder_pose_topic = self.declare_parameter(
            'r_shoulder_pose_topic',
            '/r_shoulder_pose',
        ).value
        l_shoulder_pose_topic = self.declare_parameter(
            'l_shoulder_pose_topic',
            '/l_shoulder_pose',
        ).value
        r_elbow_pose_topic = self.declare_parameter(
            'r_elbow_pose_topic',
            '/r_elbow_pose',
        ).value
        l_elbow_pose_topic = self.declare_parameter(
            'l_elbow_pose_topic',
            '/l_elbow_pose',
        ).value
        r_wrist_pose_topic = self.declare_parameter(
            'r_wrist_pose_topic',
            '/r_wrist_pose',
        ).value
        l_wrist_pose_topic = self.declare_parameter(
            'l_wrist_pose_topic',
            '/l_wrist_pose',
        ).value
        r_goal_pose_topic = self.declare_parameter(
            'r_goal_pose_topic',
            '/r_goal_pose',
        ).value
        l_goal_pose_topic = self.declare_parameter(
            'l_goal_pose_topic',
            '/l_goal_pose',
        ).value
        r_subgoal_pose_topic = self.declare_parameter(
            'r_subgoal_pose_topic',
            '/r_subgoal_pose',
        ).value
        l_subgoal_pose_topic = self.declare_parameter(
            'l_subgoal_pose_topic',
            '/l_subgoal_pose',
        ).value

        self.right_goal_publisher_ = self.create_publisher(
            PoseStamped,
            r_goal_pose_topic,
            QOS_BEST_EFFORT,
        )
        self.left_goal_publisher_ = self.create_publisher(
            PoseStamped,
            l_goal_pose_topic,
            QOS_BEST_EFFORT,
        )
        self.right_subgoal_publisher_ = self.create_publisher(
            PoseStamped,
            r_subgoal_pose_topic,
            QOS_BEST_EFFORT,
        )
        self.left_subgoal_publisher_ = self.create_publisher(
            PoseStamped,
            l_subgoal_pose_topic,
            QOS_BEST_EFFORT,
        )

        self.right_shoulder_subscriber_ = self.create_subscription(
            PoseStamped,
            r_shoulder_pose_topic,
            self._right_shoulder_callback,
            QOS_BEST_EFFORT,
        )
        self.left_shoulder_subscriber_ = self.create_subscription(
            PoseStamped,
            l_shoulder_pose_topic,
            self._left_shoulder_callback,
            QOS_BEST_EFFORT,
        )
        self.right_elbow_subscriber_ = self.create_subscription(
            PoseStamped,
            r_elbow_pose_topic,
            self._right_elbow_callback,
            QOS_BEST_EFFORT,
        )
        self.left_elbow_subscriber_ = self.create_subscription(
            PoseStamped,
            l_elbow_pose_topic,
            self._left_elbow_callback,
            QOS_BEST_EFFORT,
        )
        self.right_wrist_subscriber_ = self.create_subscription(
            PoseStamped,
            r_wrist_pose_topic,
            self._right_wrist_callback,
            QOS_BEST_EFFORT,
        )
        self.left_wrist_subscriber_ = self.create_subscription(
            PoseStamped,
            l_wrist_pose_topic,
            self._left_wrist_callback,
            QOS_BEST_EFFORT,
        )

        self.get_logger().info('Arm Retargeting Teleop Node Started')

    def _right_shoulder_callback(self, msg: PoseStamped) -> None:
        self._update_pose_state(self.right_pose_state, shoulder_msg=msg)
        self.run_teleop()

    def _left_shoulder_callback(self, msg: PoseStamped) -> None:
        self._update_pose_state(self.left_pose_state, shoulder_msg=msg)
        self.run_teleop()

    def _right_elbow_callback(self, msg: PoseStamped) -> None:
        self._update_pose_state(self.right_pose_state, elbow_msg=msg)
        self.run_teleop()

    def _left_elbow_callback(self, msg: PoseStamped) -> None:
        self._update_pose_state(self.left_pose_state, elbow_msg=msg)
        self.run_teleop()

    def _right_wrist_callback(self, msg: PoseStamped) -> None:
        self._update_pose_state(self.right_pose_state, wrist_msg=msg)
        self.run_teleop()

    def _left_wrist_callback(self, msg: PoseStamped) -> None:
        self._update_pose_state(self.left_pose_state, wrist_msg=msg)
        self.run_teleop()

    def run_teleop(self) -> None:
        """Retarget both arms and publish elbow/wrist goals."""
        retargeted_targets = self._retarget_bimanual_pose_states()
        if retargeted_targets is None:
            return

        right_elbow_goal, right_wrist_goal, left_elbow_goal, left_wrist_goal = (
            retargeted_targets
        )
        self.publish_targets_right(right_elbow_goal, right_wrist_goal)
        self.publish_targets_left(left_elbow_goal, left_wrist_goal)

    def _retarget_bimanual_pose_states(
        self,
    ) -> Optional[tuple[PoseStamped, PoseStamped, PoseStamped, PoseStamped]]:
        """Retarget both arms and prioritize wrist-to-wrist distance."""
        right_targets = self._retarget_pose_state(
            pose_state=self.right_pose_state,
            geometry=self.right_geometry,
            shoulder_link=self.right_shoulder_link,
        )
        if right_targets is None:
            return None

        left_targets = self._retarget_pose_state(
            pose_state=self.left_pose_state,
            geometry=self.left_geometry,
            shoulder_link=self.left_shoulder_link,
        )
        if left_targets is None:
            return None

        right_wrist_msg = self.right_pose_state.wrist
        left_wrist_msg = self.left_pose_state.wrist
        if right_wrist_msg is None or left_wrist_msg is None:
            return None

        right_elbow_goal, right_wrist_goal = right_targets
        left_elbow_goal, left_wrist_goal = left_targets
        right_wrist_target = self._pose_to_numpy(right_wrist_goal)
        left_wrist_target = self._pose_to_numpy(left_wrist_goal)

        right_elbow_target = self._pose_to_numpy(right_elbow_goal)
        left_elbow_target = self._pose_to_numpy(left_elbow_goal)
        human_wrist_distance = float(
            np.linalg.norm(
                self._pose_to_numpy(right_wrist_msg) - self._pose_to_numpy(left_wrist_msg)
            )
        )
        desired_wrist_distance = human_wrist_distance * self.wrist_distance_scale

        adjusted_right_wrist_target, adjusted_left_wrist_target = (
            self._apply_wrist_distance_priority(
                right_wrist_target=right_wrist_target,
                left_wrist_target=left_wrist_target,
                desired_wrist_distance=desired_wrist_distance,
            )
        )
        adjusted_right_wrist_target = self._project_wrist_to_forearm_length(
            elbow_target=right_elbow_target,
            wrist_target=adjusted_right_wrist_target,
            forearm_length=self.right_geometry.forearm_length,
            fallback_wrist_target=right_wrist_target,
        )
        adjusted_left_wrist_target = self._project_wrist_to_forearm_length(
            elbow_target=left_elbow_target,
            wrist_target=adjusted_left_wrist_target,
            forearm_length=self.left_geometry.forearm_length,
            fallback_wrist_target=left_wrist_target,
        )
        adjusted_right_wrist_target, adjusted_left_wrist_target = (
            self._smooth_wrist_targets(
                right_target=adjusted_right_wrist_target,
                left_target=adjusted_left_wrist_target,
            )
        )
        right_wrist_goal = self._copy_pose_with_new_position(
            right_wrist_goal,
            adjusted_right_wrist_target,
        )
        left_wrist_goal = self._copy_pose_with_new_position(
            left_wrist_goal,
            adjusted_left_wrist_target,
        )
        return right_elbow_goal, right_wrist_goal, left_elbow_goal, left_wrist_goal

    @staticmethod
    def _update_pose_state(
        pose_state: ArmPoseState,
        shoulder_msg: Optional[PoseStamped] = None,
        elbow_msg: Optional[PoseStamped] = None,
        wrist_msg: Optional[PoseStamped] = None,
    ) -> None:
        """Update the cached pose state for one arm."""
        if shoulder_msg is not None:
            pose_state.shoulder = shoulder_msg
        if elbow_msg is not None:
            pose_state.elbow = elbow_msg
        if wrist_msg is not None:
            pose_state.wrist = wrist_msg

    def _retarget_pose_state(
        self,
        pose_state: ArmPoseState,
        geometry: RobotArmGeometry,
        shoulder_link: str,
    ) -> Optional[tuple[PoseStamped, PoseStamped]]:
        """Retarget one arm pose state into elbow and wrist goals."""
        shoulder_msg = pose_state.shoulder
        elbow_msg = pose_state.elbow
        wrist_msg = pose_state.wrist
        if shoulder_msg is None or elbow_msg is None or wrist_msg is None:
            return
        if not self._poses_have_matching_stamps(
            shoulder_msg,
            elbow_msg,
            wrist_msg,
        ):
            return

        shoulder_pos = self._pose_to_numpy(shoulder_msg)
        elbow_pos = self._pose_to_numpy(elbow_msg)
        wrist_pos = self._pose_to_numpy(wrist_msg)

        upper_arm_direction = self._compute_unit_vector(
            elbow_pos - shoulder_pos
        )
        forearm_direction = self._compute_unit_vector(
            wrist_pos - elbow_pos
        )
        if upper_arm_direction is None or forearm_direction is None:
            return
        shoulder_anchor = self._lookup_link_position(shoulder_link)
        if shoulder_anchor is None:
            return

        elbow_target = (
            shoulder_anchor
            + geometry.upper_arm_length * upper_arm_direction
        )
        wrist_target = (
            elbow_target
            + geometry.forearm_length * forearm_direction
        )

        elbow_goal = self._copy_pose_with_new_position(elbow_msg, elbow_target)
        wrist_goal = self._copy_pose_with_new_position(wrist_msg, wrist_target)
        return elbow_goal, wrist_goal

    def publish_targets_left(
        self,
        elbow_goal: PoseStamped,
        wrist_goal: PoseStamped,
    ) -> None:
        """Publish left-arm elbow and wrist targets."""
        self._publish_targets(
            subgoal_publisher=self.left_subgoal_publisher_,
            goal_publisher=self.left_goal_publisher_,
            elbow_goal=elbow_goal,
            wrist_goal=wrist_goal,
        )

    def publish_targets_right(
        self,
        elbow_goal: PoseStamped,
        wrist_goal: PoseStamped,
    ) -> None:
        """Publish right-arm elbow and wrist targets."""
        self._publish_targets(
            subgoal_publisher=self.right_subgoal_publisher_,
            goal_publisher=self.right_goal_publisher_,
            elbow_goal=elbow_goal,
            wrist_goal=wrist_goal,
        )

    @staticmethod
    def _publish_targets(
        subgoal_publisher,
        goal_publisher,
        elbow_goal: PoseStamped,
        wrist_goal: PoseStamped,
    ) -> None:
        """Publish elbow and wrist goals."""
        subgoal_publisher.publish(elbow_goal)
        goal_publisher.publish(wrist_goal)

    def _compute_robot_geometry(
        self,
        robot: RobotWrapper,
        shoulder_link: str,
        elbow_link: str,
        wrist_link: str,
    ) -> RobotArmGeometry:
        """Compute robot arm limb lengths."""
        shoulder_idx = robot.get_link_index(shoulder_link)
        elbow_idx = robot.get_link_index(elbow_link)
        wrist_idx = robot.get_link_index(wrist_link)

        robot.compute_forward_kinematics(robot.q0)

        shoulder_pos = robot.get_link_pose(shoulder_idx)[:3, 3].astype(np.float64)
        elbow_pos = robot.get_link_pose(elbow_idx)[:3, 3].astype(np.float64)
        wrist_pos = robot.get_link_pose(wrist_idx)[:3, 3].astype(np.float64)

        return RobotArmGeometry(
            upper_arm_length=float(np.linalg.norm(elbow_pos - shoulder_pos)),
            forearm_length=float(np.linalg.norm(wrist_pos - elbow_pos)),
        )

    def _lookup_link_position(self, link_name: str) -> Optional[np.ndarray]:
        """Return the current link position in the base frame using TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                link_name,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Failed to lookup transform from {self.base_frame} to {link_name}: {exc}'
            )
            return None

        return np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _pose_to_numpy(msg: PoseStamped) -> np.ndarray:
        """Convert a pose message position into a NumPy vector."""
        return np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _copy_pose_with_new_position(
        source: PoseStamped,
        position: np.ndarray,
    ) -> PoseStamped:
        """Copy pose orientation and header while replacing the position."""
        msg = PoseStamped()
        msg.header = source.header
        msg.pose.position.x = source.pose.position.x
        msg.pose.position.y = source.pose.position.y
        msg.pose.position.z = source.pose.position.z
        msg.pose.orientation.x = source.pose.orientation.x
        msg.pose.orientation.y = source.pose.orientation.y
        msg.pose.orientation.z = source.pose.orientation.z
        msg.pose.orientation.w = source.pose.orientation.w
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        return msg

    @staticmethod
    def _compute_unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
        """Return a normalized vector or `None` if it is degenerate."""
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return None
        return vector / norm

    def _apply_wrist_distance_priority(
        self,
        right_wrist_target: np.ndarray,
        left_wrist_target: np.ndarray,
        desired_wrist_distance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shift both wrists to prioritize matching inter-wrist distance."""
        distance_vector = right_wrist_target - left_wrist_target
        current_distance = float(np.linalg.norm(distance_vector))
        if current_distance < 1e-6:
            return right_wrist_target, left_wrist_target

        direction = distance_vector / current_distance
        full_distance_error = desired_wrist_distance - current_distance
        dynamic_priority = self._compute_dynamic_wrist_priority(current_distance)
        weighted_distance_error = dynamic_priority * full_distance_error
        correction_per_wrist = 0.5 * weighted_distance_error
        correction_per_wrist = float(
            np.clip(
                correction_per_wrist,
                -self.max_wrist_distance_correction,
                self.max_wrist_distance_correction,
            )
        )
        correction = correction_per_wrist * direction
        return right_wrist_target + correction, left_wrist_target - correction

    def _compute_dynamic_wrist_priority(self, current_wrist_distance: float) -> float:
        """Return wrist-distance priority that decays past reference distance."""
        reference_distance = max(0.0, self.wrist_priority_reference_distance)
        if current_wrist_distance <= reference_distance:
            return self.wrist_distance_priority

        min_scale = float(np.clip(self.wrist_priority_min_scale, 0.0, 1.0))
        decay_rate = max(0.0, self.wrist_priority_decay_rate)
        distance_over_reference = current_wrist_distance - reference_distance
        decay_scale = float(np.exp(-decay_rate * distance_over_reference))
        scaled_priority = self.wrist_distance_priority * max(min_scale, decay_scale)
        return scaled_priority

    def _compute_wrist_distance_scale(
        self,
        human_wrist_to_fingertip: float,
        robot_wrist_to_fingertip: float,
    ) -> float:
        """Compute a stable wrist-distance scaling factor from hand lengths."""
        if human_wrist_to_fingertip <= 1e-6 or robot_wrist_to_fingertip <= 1e-6:
            self.get_logger().warn(
                'Invalid wrist-to-fingertip lengths; using wrist distance scale = 1.0'
            )
            return 1.0
        return float(robot_wrist_to_fingertip / human_wrist_to_fingertip)

    def _project_wrist_to_forearm_length(
        self,
        elbow_target: np.ndarray,
        wrist_target: np.ndarray,
        forearm_length: float,
        fallback_wrist_target: np.ndarray,
    ) -> np.ndarray:
        """Keep wrist target represented from elbow frame with forearm length."""
        elbow_to_wrist = wrist_target - elbow_target
        direction = self._compute_unit_vector(elbow_to_wrist)
        if direction is None:
            fallback_direction = self._compute_unit_vector(
                fallback_wrist_target - elbow_target
            )
            if fallback_direction is None:
                return wrist_target
            direction = fallback_direction
        return elbow_target + forearm_length * direction

    def _smooth_wrist_targets(
        self,
        right_target: np.ndarray,
        left_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Low-pass filter wrist targets to prevent frame-to-frame jumps."""
        alpha = float(np.clip(self.wrist_distance_smoothing_alpha, 0.0, 1.0))
        if self._filtered_right_wrist_target is None:
            self._filtered_right_wrist_target = right_target.copy()
        else:
            self._filtered_right_wrist_target = (
                (1.0 - alpha) * self._filtered_right_wrist_target + alpha * right_target
            )
        if self._filtered_left_wrist_target is None:
            self._filtered_left_wrist_target = left_target.copy()
        else:
            self._filtered_left_wrist_target = (
                (1.0 - alpha) * self._filtered_left_wrist_target + alpha * left_target
            )
        return (
            self._filtered_right_wrist_target.copy(),
            self._filtered_left_wrist_target.copy(),
        )

    def _poses_have_matching_stamps(self, *msgs: PoseStamped) -> bool:
        """Return whether all poses share the exact same ROS header stamp."""
        if not msgs:
            return False

        reference_stamp = self._pose_stamp_tuple(msgs[0])
        for msg in msgs[1:]:
            if self._pose_stamp_tuple(msg) != reference_stamp:
                return False
        return True

    @staticmethod
    def _pose_stamp_tuple(msg: PoseStamped) -> tuple[int, int]:
        """Convert a ROS pose header stamp into an equality-friendly tuple."""
        return int(msg.header.stamp.sec), int(msg.header.stamp.nanosec)

def main(args=None) -> None:
    """Run the arm retargeting teleoperation node."""
    rclpy.init(args=args)
    arm_retargeting_teleop = ArmRetargetingTeleop()
    rclpy.spin(arm_retargeting_teleop)
    arm_retargeting_teleop.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
