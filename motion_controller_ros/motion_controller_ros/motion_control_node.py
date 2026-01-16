import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np
from rclpy.action import ActionServer
import time
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from motion_controller_msgs.action import MoveCartesian

from motion_controller_core.kinematics import PyrokiSolver, PinocchioSolver

class MotionControlNode(Node):
    def __init__(self):
        super().__init__('motion_control_node')

        # Declare Parameters
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_links', ['end_effector'])
        self.declare_parameter('command_topic', '/joint_commands')
        # self.declare_parameter('use_sim_time', False) # Already declared by Node in recent ROS 2 versions

        # Internal State
        self.urdf_content = None
        self.solver = None
        self.initialized = False

        # Robot State
        self.current_q = None
        self.joint_names = []

        self.base_link = self.get_parameter('base_link').get_parameter_value().string_value
        self.tip_links = self.get_parameter('tip_links').get_parameter_value().string_array_value
        self.command_topic = self.get_parameter('command_topic').get_parameter_value().string_value

        self.get_logger().info(f"Starting Motion Controller Node...")
        self.get_logger().info(f"Target Base: {self.base_link}, Tips: {self.tip_links}")

        # QoS Profile for /robot_description (Latched topic)
        # Must be Transient Local to receive the message if we join late
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Subscriber for robot description
        self.sub_robot_desc = self.create_subscription(
            String,
            '/robot_description',
            self.robot_description_callback,
            qos_profile
        )

        # Subscriber for Joint States
        self.sub_joint_states = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publisher for Joint Commands
        # Logic: We publish Float64MultiArray by default.
        self.pub_commands = self.create_publisher(
            JointTrajectory,
            self.command_topic,
            10
        )

        # Control Loop Timer (100Hz default)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Waiting for /robot_description...")

        # Action Server
        self._action_server = ActionServer(
            self,
            MoveCartesian,
            'move_cartesian',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        if not self.initialized or self.solver is None:
            self.get_logger().error("Solver not initialized!")
            goal_handle.abort()
            result = MoveCartesian.Result()
            result.success = False
            result.message = "Solver not initialized"
            return result

        if self.current_q is None:
            self.get_logger().error("Current Joint State unknown!")
            goal_handle.abort()
            result = MoveCartesian.Result()
            result.success = False
            result.message = "Current Joint State unknown"
            return result

        goal = goal_handle.request
        target_pose_msg = goal.target_pose

        # Convert PoseMsg to Matrix
        # TODO: Handle frame_id transforms if target is not in base_link

        t = target_pose_msg.pose.position
        q_rot = target_pose_msg.pose.orientation

        target_matrix = np.eye(4)
        target_matrix[:3, 3] = [t.x, t.y, t.z]
        r = Rotation.from_quat([q_rot.x, q_rot.y, q_rot.z, q_rot.w])
        target_matrix[:3, :3] = r.as_matrix()

        # Call IK
        # Use current_q as seed
        success, q_sol = self.solver.solve_ik(target_matrix, self.current_q)

        result = MoveCartesian.Result()

        if success:
            self.get_logger().info("IK Solution Found!")
            # Publish result immediately (Step 1: Check connectivity)
            # In Phase 3, we will interpolate.

            cmd = JointTrajectory()
            cmd.joint_names = self.joint_names
            cmd.points.append(JointTrajectoryPoint(positions=q_sol.tolist(), time_from_start=Duration(seconds=1)))
            self.pub_commands.publish(cmd)

            goal_handle.succeed()
            result.success = True
            result.message = "Moved to target"
        else:
            self.get_logger().error("IK Failed to find solution.")
            goal_handle.abort()
            result.success = False
            result.message = "IK Failed"

        return result


    def joint_state_callback(self, msg):
        # We need to map msg.position to our solver's expected order
        # For now, simplistic approach: assuming msg.position corresponds to solver dof order
        # TODO: Implement robust name-based mapping using self.solver.robot.joints

        if not self.initialized or self.solver is None:
            return

        # Store current q
        # Warning: This blindly takes positions. Real implementation needs mapping.
        if len(msg.position) > 0:
            self.current_q = np.array(msg.position)

    def robot_description_callback(self, msg):
        if self.initialized:
            return # Already initialized

        self.get_logger().info("Received /robot_description! Initializing Solver...")
        self.urdf_content = msg.data

        # Initialize Pyroki Solver
        # TODO: Make solver type configurable? defaults to Pyroki for now as requested.
        try:
            if PyrokiSolver is None:
                self.get_logger().error("PyrokiSolver could not be imported! Check dependencies.")
                return

            self.solver = PyrokiSolver()
            success = self.solver.init(self.urdf_content, self.base_link, self.tip_links)

            if success:
                self.initialized = True
                self.get_logger().info("Solver Initialized Successfully! Ready for commands.")
            else:
                self.get_logger().error("Failed to initialize PyrokiSolver with provided URDF.")

        except Exception as e:
            self.get_logger().error(f"Exception during solver init: {e}")

    def control_loop(self):
        if not self.initialized:
            return

        # Main control logic placeholder
        # Example: Just publish current q (hold) or computed trajectory point
        if self.current_q is not None:
            command_msg = Float64MultiArray()
            command_msg.data = self.current_q.tolist()
            self.pub_commands.publish(command_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MotionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
