"""High-level retargeting interface for ROBOTIS Hand using DexPilot algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retargeting.robot_wrapper import RobotWrapper
from retargeting.optimizer import DexPilotOptimizer


# Package root for URDF path resolution
# _PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

LOW_PASS_ALPHA = 0.5  # Low-pass filter alpha (smaller = smoother but more latency)

# MediaPipe landmark indices: wrist=0, thumb_tip=4, index_tip=8, middle_tip=12, ring_tip=16, pinky_tip=20
MP_WRIST_IDX = 0
MP_FINGER_TIP_INDICES = [4, 8, 12, 16, 20]  # Thumb to Pinky tip landmarks

@dataclass
class RetargetingResult:
    """Retargeting result containing robot joint positions and intermediate data."""
    robot_qpos: np.ndarray      # Robot joint positions (20,)
    mediapipe_pose: np.ndarray  # MediaPipe format hand pose (21, 3)
    reference: np.ndarray       # Reference vectors used in optimization


class ROBOTISHandRetargeter:
    """Retargeter for ROBOTIS Hand using DexPilot algorithm."""

    def __init__(self,
                 path_to_urdf: str,
                 hand_side: str = "right"):
        """
        Initialize retargeter for specified hand.

        Args:
            urdf_path
            hand_side: "right" or "left"
        """
        self.hand_side = hand_side.lower()
        if self.hand_side not in ["right", "left"]:
            raise ValueError(f"hand_side must be 'right' or 'left', got {hand_side}")

        if self.hand_side == "right":
            self.hx_wrist_link_name = "hx5_d20_right_base"
            self.hx_finger_tip_link_names = [
                "finger_end_r_link1",
                "finger_end_r_link2",
                "finger_end_r_link3",
                "finger_end_r_link4",
                "finger_end_r_link5"
                ]
        elif self.hand_side == "left":
            self.hx_wrist_link_name = "hx5_d20_left_base"
            self.hx_finger_tip_link_names = [
                "finger_end_l_link1",
                "finger_end_l_link2",
                "finger_end_l_link3",
                "finger_end_l_link4",
                "finger_end_l_link5"
                ]

        # Build URDF path (from package directory)
        # urdf_path = (_PACKAGE_ROOT / f"motion_controller_models/models/hx5_d20/hx5_d20_{self.hand_side}.urdf").resolve()
        urdf_path = Path(path_to_urdf)
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")

        # Load robot model
        robot = RobotWrapper(str(urdf_path))
        self.robot = robot

        # Compute robot finger lengths at neutral position (used for scaling)
        self.robot_finger_lengths = self._compute_robot_finger_lengths(robot)

        # Calibration state: will be set on first retarget call
        self.is_calibrated = False
        self.finger_scaling = np.ones(5, dtype=np.float32)  # Default 1:1 scaling

        # Build optimizer (scaling will be updated after calibration) and filter
        self.optimizer = DexPilotOptimizer(
            robot,
            robot.dof_joint_names,
            finger_tip_link_names=self.hx_finger_tip_link_names,
            wrist_link_name=self.hx_wrist_link_name,
            finger_scaling=self.finger_scaling.tolist(),
            hand_side=self.hand_side,
        )
        self.filter = LPFilter(LOW_PASS_ALPHA)

        # Joint limits (always enabled for ROBOTIS Hand)
        joint_limits = robot.joint_limits[self.optimizer.idx_pin2target]
        self.optimizer.set_joint_limit(joint_limits)
        self.joint_limits = joint_limits

        # Initialize last joint positions for warm start
        self.last_qpos = joint_limits.mean(1).astype(np.float32)

    def retarget(self, mediapipe_pose: np.ndarray) -> RetargetingResult:
        mediapipe_pose = np.asarray(mediapipe_pose, dtype=np.float64)
        if mediapipe_pose.shape != (21, 3):
            raise ValueError(f"Expected mediapipe_pose shape (21, 3), got {mediapipe_pose.shape}")

        # Calibrate on first frame if not already calibrated
        if not self.is_calibrated:
            self._calibrate_scaling(mediapipe_pose)

        # 1. Position Reference (Wrist-to-Tips)
        indices = self.optimizer.target_link_human_indices
        reference = mediapipe_pose[indices[1], :] - mediapipe_pose[indices[0], :]

        # 2. Orientation Reference (Tip Direction)
        # Using vector from DIP joint to Tip for each finger
        # Thumb: 3->4, Index: 7->8, Middle: 11->12, Ring: 15->16, Pinky: 19->20
        tip_ids = [4, 8, 12, 16, 20]
        dip_ids = [3, 7, 11, 15, 19]
        
        human_dir = mediapipe_pose[tip_ids, :] - mediapipe_pose[dip_ids, :]
        # Normalize vectors for the optimizer
        human_dir_norm = human_dir / (np.linalg.norm(human_dir, axis=1, keepdims=True) + 1e-6)

        # 3. Run Optimization
        robot_qpos = self._retarget_optimization(
            ref_value=reference, 
            target_dir=human_dir_norm
        )

        return RetargetingResult(
            robot_qpos=robot_qpos,
            mediapipe_pose=mediapipe_pose,
            reference=reference,
        )

    def _compute_robot_finger_lengths(self, robot: RobotWrapper) -> np.ndarray:
        """Compute robot wrist-to-fingertip distances at neutral pose.
        
        Returns:
            np.ndarray: Array of 5 finger lengths (thumb to pinky)
        """
        # Get link indices
        wrist_idx = robot.get_link_index(self.hx_wrist_link_name)
        tip_indices = [robot.get_link_index(name) for name in self.hx_finger_tip_link_names]
        
        # Compute FK at neutral (middle of joint limits)
        neutral_qpos = robot.joint_limits.mean(axis=1)
        robot.compute_forward_kinematics(neutral_qpos)
        
        # Get positions
        wrist_pos = robot.get_link_pose(wrist_idx)[:3, 3]
        tip_positions = np.array([robot.get_link_pose(idx)[:3, 3] for idx in tip_indices])
        
        # Compute distances
        finger_lengths = np.linalg.norm(tip_positions - wrist_pos, axis=1)
        return finger_lengths.astype(np.float32)

    def _compute_human_finger_lengths(self, mediapipe_pose: np.ndarray) -> np.ndarray:
        """Compute human wrist-to-fingertip distances from MediaPipe landmarks.

        Args:
            mediapipe_pose: MediaPipe hand landmarks (21, 3)

        Returns:
            np.ndarray: Array of 5 finger lengths (thumb to pinky)
        """
        wrist_pos = mediapipe_pose[MP_WRIST_IDX]
        tip_positions = mediapipe_pose[MP_FINGER_TIP_INDICES]
        finger_lengths = np.linalg.norm(tip_positions - wrist_pos, axis=1)
        return finger_lengths.astype(np.float32)

    def _calibrate_scaling(self, mediapipe_pose: np.ndarray):
        """Calibrate finger scaling based on human hand size.

        Computes scaling factors as human_length / robot_length so that
        the robot hand targets are scaled down to match human hand proportions.
        """
        human_lengths = self._compute_human_finger_lengths(mediapipe_pose)

        # Scaling = human / robot (this scales robot targets down to human size)
        self.finger_scaling = human_lengths / (self.robot_finger_lengths + 1e-6)

        # Update optimizer's scaling
        self.optimizer.finger_scaling = self.finger_scaling
        self.optimizer.vector_scaling = self.optimizer.build_vector_scaling()

        self.is_calibrated = True
        print(f"[Retargeter] Calibrated finger scaling: {self.finger_scaling}")
        print(f"[Retargeter] Human finger lengths: {human_lengths}")
        print(f"[Retargeter] Robot finger lengths: {self.robot_finger_lengths}")

    def _retarget_optimization(self, ref_value: np.ndarray, target_dir: np.ndarray) -> np.ndarray:
        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            target_dir=target_dir.astype(np.float32), # Pass the directions here
            last_qpos=np.clip(
                self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]
            ),
        )
        self.last_qpos = qpos
        return self.filter.next(qpos)

class LPFilter:
    """Low-pass filter for smoothing joint positions."""
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False

__all__ = [
    "ROBOTISHandRetargeter",
    "RetargetingResult",
]
