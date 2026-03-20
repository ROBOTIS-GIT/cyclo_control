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
# Author: Hyunwoo Nam

"""ROS node that retargets hand landmarks into joint trajectories."""

import os
from typing import Sequence

from ament_index_python.packages import get_package_share_directory

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from retargeting.seq_retarget import ROBOTISHandRetargeter

from robotis_interfaces.msg import HandJoints

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


QOS_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class RetargetingTeleop(Node):
    """Subscribe to hand landmarks and publish hand joint trajectories."""

    def __init__(self) -> None:
        """Initialize retargeters, publishers, and subscriptions."""
        super().__init__('retargeting_teleop')

        models_share_dir = get_package_share_directory(
            'robotis_motion_controller_models'
        )
        right_urdf_path = os.path.join(
            models_share_dir,
            'models',
            'hx5_d20',
            'hx5_d20_right.urdf',
        )
        left_urdf_path = os.path.join(
            models_share_dir,
            'models',
            'hx5_d20',
            'hx5_d20_left.urdf',
        )

        self.right_retargeter = ROBOTISHandRetargeter(
            path_to_urdf=right_urdf_path,
            hand_side='right',
        )
        self.left_retargeter = ROBOTISHandRetargeter(
            path_to_urdf=left_urdf_path,
            hand_side='left',
        )

        self.left_joint_names = [
            'finger_l_joint1',
            'finger_l_joint2',
            'finger_l_joint3',
            'finger_l_joint4',
            'finger_l_joint5',
            'finger_l_joint6',
            'finger_l_joint7',
            'finger_l_joint8',
            'finger_l_joint9',
            'finger_l_joint10',
            'finger_l_joint11',
            'finger_l_joint12',
            'finger_l_joint13',
            'finger_l_joint14',
            'finger_l_joint15',
            'finger_l_joint16',
            'finger_l_joint17',
            'finger_l_joint18',
            'finger_l_joint19',
            'finger_l_joint20',
        ]
        self.right_joint_names = [
            'finger_r_joint1',
            'finger_r_joint2',
            'finger_r_joint3',
            'finger_r_joint4',
            'finger_r_joint5',
            'finger_r_joint6',
            'finger_r_joint7',
            'finger_r_joint8',
            'finger_r_joint9',
            'finger_r_joint10',
            'finger_r_joint11',
            'finger_r_joint12',
            'finger_r_joint13',
            'finger_r_joint14',
            'finger_r_joint15',
            'finger_r_joint16',
            'finger_r_joint17',
            'finger_r_joint18',
            'finger_r_joint19',
            'finger_r_joint20',
        ]

        self.left_publisher_ = self.create_publisher(
            JointTrajectory,
            (
                '/leader/'
                'joint_trajectory_command_broadcaster_left_hand/'
                'joint_trajectory'
            ),
            QOS_BEST_EFFORT,
        )
        self.right_publisher_ = self.create_publisher(
            JointTrajectory,
            (
                '/leader/'
                'joint_trajectory_command_broadcaster_right_hand/'
                'joint_trajectory'
            ),
            QOS_BEST_EFFORT,
        )

        self.left_subscriber_ = self.create_subscription(
            HandJoints,
            '/left_hand/hand_joint_pos',
            self.run_teleop_left,
            QOS_BEST_EFFORT,
        )
        self.right_subscriber_ = self.create_subscription(
            HandJoints,
            '/right_hand/hand_joint_pos',
            self.run_teleop_right,
            QOS_BEST_EFFORT,
        )

        self.get_logger().info('Retargeting Teleop Node Started')

    def run_teleop_left(self, msg: HandJoints) -> None:
        """Retarget the left hand landmarks and publish a trajectory."""
        mediapipe_pos = self.convert_msg_to_numpy(msg)
        retargeting_result = self.left_retargeter.retarget(mediapipe_pos)
        self.publish_trajectory_left(retargeting_result.robot_qpos)

    def run_teleop_right(self, msg: HandJoints) -> None:
        """Retarget the right hand landmarks and publish a trajectory."""
        mediapipe_pos = self.convert_msg_to_numpy(msg)
        retargeting_result = self.right_retargeter.retarget(mediapipe_pos)
        self.publish_trajectory_right(retargeting_result.robot_qpos)

    def convert_msg_to_numpy(self, msg: HandJoints) -> np.ndarray:
        """Convert a `HandJoints` message into a `(21, 3)` NumPy array."""
        pose_array_np = np.zeros((21, 3), dtype=np.float32)
        for index, point in enumerate(msg.joints):
            pose_array_np[index, 0] = point.x
            pose_array_np[index, 1] = point.y
            pose_array_np[index, 2] = point.z
        return pose_array_np

    def publish_trajectory_left(
        self,
        goal: np.ndarray,
        duration: float = 0,
    ) -> None:
        """Publish a left-hand joint trajectory command."""
        self._publish_trajectory(
            self.left_publisher_,
            self.left_joint_names,
            goal,
            duration,
        )

    def publish_trajectory_right(
        self,
        goal: np.ndarray,
        duration: float = 0,
    ) -> None:
        """Publish a right-hand joint trajectory command."""
        self._publish_trajectory(
            self.right_publisher_,
            self.right_joint_names,
            goal,
            duration,
        )

    def _publish_trajectory(
        self,
        publisher,
        joint_names: Sequence[str],
        goal: np.ndarray,
        duration: float,
    ) -> None:
        """Build and publish a `JointTrajectory` message."""
        msg = JointTrajectory()
        msg.joint_names = list(joint_names)
        goal_point = JointTrajectoryPoint()
        goal_point.positions = goal.tolist()
        goal_point.time_from_start.sec = int(duration)
        goal_point.time_from_start.nanosec = 0
        msg.points.append(goal_point)
        publisher.publish(msg)


def main(args=None) -> None:
    """Run the retargeting teleoperation node."""
    rclpy.init(args=args)
    retargeting_teleop = RetargetingTeleop()
    rclpy.spin(retargeting_teleop)
    retargeting_teleop.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
