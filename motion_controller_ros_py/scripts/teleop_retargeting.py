import time
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from motion_controller_msgs.msg import HandJoints

from motion_controller_core.src.retargeting.retarget import ROBOTISHandRetargeter

# Create a profile that prioritizes speed over reliability
qos_best_effort = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

class RetargetingTeleop(Node):
    def __init__(self):
        super().__init__('retargeting_teleop')

        self.right_retargeter = ROBOTISHandRetargeter(hand_side="right")
        self.left_retargeter = ROBOTISHandRetargeter(hand_side="left")

        self.left_joint_names = [
            "finger_l_joint1", "finger_l_joint2", "finger_l_joint3", "finger_l_joint4",
            "finger_l_joint5", "finger_l_joint6", "finger_l_joint7", "finger_l_joint8",
            "finger_l_joint9", "finger_l_joint10", "finger_l_joint11", "finger_l_joint12",
            "finger_l_joint13", "finger_l_joint14", "finger_l_joint15", "finger_l_joint16",
            "finger_l_joint17", "finger_l_joint18", "finger_l_joint19", "finger_l_joint20"
        ]

        self.right_joint_names = [
            "finger_r_joint1", "finger_r_joint2", "finger_r_joint3", "finger_r_joint4",
            "finger_r_joint5", "finger_r_joint6", "finger_r_joint7", "finger_r_joint8",
            "finger_r_joint9", "finger_r_joint10", "finger_r_joint11", "finger_r_joint12",
            "finger_r_joint13", "finger_r_joint14", "finger_r_joint15", "finger_r_joint16",
            "finger_r_joint17", "finger_r_joint18", "finger_r_joint19", "finger_r_joint20"
        ]

        self.left_publisher_ = self.create_publisher(JointTrajectory, '/leader/joint_trajectory_command_broadcaster_left_hand/joint_trajectory', qos_best_effort)
        # self.left_publisher_ = self.create_publisher(JointTrajectory, '/left_hand_controller/joint_trajectory', qos_best_effort)
        # self.left_publisher_ = self.create_publisher(JointState, '/joint_states', qos_best_effort)

        self.right_publisher_ = self.create_publisher(JointTrajectory, '/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory', qos_best_effort)
        # self.right_publisher_ = self.create_publisher(JointTrajectory, '/right_hand_controller/joint_trajectory', qos_best_effort)
        # self.right_publisher_ = self.create_publisher(JointState, '/joint_states', qos_best_effort)

        self.left_subscriber_ = self.create_subscription(
            HandJoints,
            '/left_hand/hand_joint_pos',
            self.run_teleop_left,
            qos_best_effort
        )

        self.right_subscriber_ = self.create_subscription(
            HandJoints,
            '/right_hand/hand_joint_pos',
            self.run_teleop_right,
            qos_best_effort
        )

        self.get_logger().info('Retargeting Teleop Node Started')

    def run_teleop_left(self, msg):
        mediapipe_pos = self.convert_msg_to_numpy(msg)
        retargeting_result = self.left_retargeter.retarget(mediapipe_pos)
        hand_joint_positions = retargeting_result.robot_qpos
        self.publish_trajectory_left(hand_joint_positions)

    def run_teleop_right(self, msg):
        mediapipe_pos = self.convert_msg_to_numpy(msg)
        retargeting_result = self.right_retargeter.retarget(mediapipe_pos)
        hand_joint_positions = retargeting_result.robot_qpos
        self.publish_trajectory_right(hand_joint_positions)

    def convert_msg_to_numpy(self, msg):
        num_joints = 21
        pose_array_np = np.zeros((num_joints,3),dtype=np.float32)
        for i, point in enumerate(msg.joints):
            pose_array_np[i,0] = point.x
            pose_array_np[i,1] = point.y
            pose_array_np[i,2] = point.z
        return pose_array_np

    def publish_trajectory_left(self, goal, duration=0):
        msg = JointTrajectory()
        msg.joint_names = self.left_joint_names
        goal_point = JointTrajectoryPoint()
        goal_point.positions = goal.tolist()
        goal_point.time_from_start.sec = int(duration)
        goal_point.time_from_start.nanosec = 0
        msg.points.append(goal_point)
        self.left_publisher_.publish(msg)

    def publish_trajectory_right(self, goal, duration=0):
        msg = JointTrajectory()
        msg.joint_names = self.right_joint_names
        goal_point = JointTrajectoryPoint()
        goal_point.positions = goal.tolist()
        goal_point.time_from_start.sec = int(duration)
        goal_point.time_from_start.nanosec = 0
        msg.points.append(goal_point)
        self.right_publisher_.publish(msg)

    # def publish_trajectory_left(self, goal):
    #     msg = JointState()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.name = self.left_joint_names
    #     msg.position = goal.tolist()
    #     self.left_publisher_.publish(msg)

    # def publish_trajectory_right(self, goal):
    #     msg = JointState()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.name = self.right_joint_names
    #     msg.position = goal.tolist()
    #     self.right_publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    retargeting_teleop = RetargetingTeleop()
    rclpy.spin(retargeting_teleop)
    retargeting_teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
