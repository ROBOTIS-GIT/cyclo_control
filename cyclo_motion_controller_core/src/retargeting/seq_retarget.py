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
# This file is derived from `dex_retargeting/seq_retarget.py` in the
# dexsuite/dex-retargeting project:
# https://github.com/dexsuite/dex-retargeting
#
# Original work Copyright (c) 2023 Yuzhe Qin, licensed under the MIT License.
# Modifications in this file are Copyright 2026 ROBOTIS CO., LTD.
#
# Author: Hyunwoo Nam

"""High-level DexPilot retargeting interface for the ROBOTIS hand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from retargeting.optimizer import DexPilotOptimizer
from retargeting.robot_wrapper import RobotWrapper


LOW_PASS_ALPHA = 0.5
MP_WRIST_IDX = 0
MP_FINGER_TIP_INDICES = [4, 8, 12, 16, 20]
MP_DIP_INDICES = [3, 7, 11, 15, 19]


@dataclass
class RetargetingResult:
    """Retargeting result containing robot joint positions and references."""

    robot_qpos: np.ndarray
    mediapipe_pose: np.ndarray
    reference: np.ndarray


class ROBOTISHandRetargeter:
    """Retarget MediaPipe hand landmarks to the ROBOTIS hand model."""

    def __init__(self, path_to_urdf: str, hand_side: str = 'right') -> None:
        """Initialize the retargeter for the requested hand side."""
        self.hand_side = hand_side.lower()
        if self.hand_side not in ['right', 'left']:
            raise ValueError(
                f"hand_side must be 'right' or 'left', got {hand_side}"
            )

        if self.hand_side == 'right':
            self.hx_wrist_link_name = 'hx5_d20_right_base'
            self.hx_finger_tip_link_names = [
                'finger_end_r_link1',
                'finger_end_r_link2',
                'finger_end_r_link3',
                'finger_end_r_link4',
                'finger_end_r_link5',
            ]
        else:
            self.hx_wrist_link_name = 'hx5_d20_left_base'
            self.hx_finger_tip_link_names = [
                'finger_end_l_link1',
                'finger_end_l_link2',
                'finger_end_l_link3',
                'finger_end_l_link4',
                'finger_end_l_link5',
            ]

        urdf_path = Path(path_to_urdf)
        if not urdf_path.exists():
            raise ValueError(f'URDF path {urdf_path} does not exist')

        robot = RobotWrapper(str(urdf_path))
        self.robot = robot
        self.robot_finger_lengths = self._compute_robot_finger_lengths(robot)

        self.is_calibrated = False
        self.finger_scaling = np.ones(5, dtype=np.float32)

        self.optimizer = DexPilotOptimizer(
            robot,
            robot.dof_joint_names,
            finger_tip_link_names=self.hx_finger_tip_link_names,
            wrist_link_name=self.hx_wrist_link_name,
            finger_scaling=self.finger_scaling.tolist(),
            hand_side=self.hand_side,
        )
        self.filter = LPFilter(LOW_PASS_ALPHA)

        joint_limits = robot.joint_limits[self.optimizer.idx_pin2target]
        self.optimizer.set_joint_limit(joint_limits)
        self.joint_limits = joint_limits
        self.last_qpos = joint_limits.mean(1).astype(np.float32)

    def retarget(self, mediapipe_pose: np.ndarray) -> RetargetingResult:
        """Convert a MediaPipe hand pose into robot joint targets."""
        mediapipe_pose = np.asarray(mediapipe_pose, dtype=np.float64)
        if mediapipe_pose.shape != (21, 3):
            raise ValueError(
                'Expected mediapipe_pose shape (21, 3), '
                f'got {mediapipe_pose.shape}'
            )

        if not self.is_calibrated:
            self._calibrate_scaling(mediapipe_pose)

        indices = self.optimizer.target_link_human_indices
        reference = (
            mediapipe_pose[indices[1], :] - mediapipe_pose[indices[0], :]
        )

        human_dir = (
            mediapipe_pose[MP_FINGER_TIP_INDICES, :]
            - mediapipe_pose[MP_DIP_INDICES, :]
        )
        human_dir_norm = human_dir / (
            np.linalg.norm(human_dir, axis=1, keepdims=True) + 1e-6
        )

        robot_qpos = self._retarget_optimization(
            ref_value=reference,
            target_dir=human_dir_norm,
        )

        return RetargetingResult(
            robot_qpos=robot_qpos,
            mediapipe_pose=mediapipe_pose,
            reference=reference,
        )

    def _compute_robot_finger_lengths(self, robot: RobotWrapper) -> np.ndarray:
        """Compute robot wrist-to-fingertip distances at a neutral pose."""
        wrist_idx = robot.get_link_index(self.hx_wrist_link_name)
        tip_indices = [
            robot.get_link_index(name)
            for name in self.hx_finger_tip_link_names
        ]

        neutral_qpos = robot.joint_limits.mean(axis=1)
        robot.compute_forward_kinematics(neutral_qpos)

        wrist_pos = robot.get_link_pose(wrist_idx)[:3, 3]
        tip_positions = np.array(
            [robot.get_link_pose(idx)[:3, 3] for idx in tip_indices]
        )
        finger_lengths = np.linalg.norm(tip_positions - wrist_pos, axis=1)
        return finger_lengths.astype(np.float32)

    def _compute_human_finger_lengths(
        self,
        mediapipe_pose: np.ndarray,
    ) -> np.ndarray:
        """Compute human wrist-to-fingertip distances from MediaPipe data."""
        wrist_pos = mediapipe_pose[MP_WRIST_IDX]
        tip_positions = mediapipe_pose[MP_FINGER_TIP_INDICES]
        finger_lengths = np.linalg.norm(tip_positions - wrist_pos, axis=1)
        return finger_lengths.astype(np.float32)

    def _calibrate_scaling(self, mediapipe_pose: np.ndarray) -> None:
        """Calibrate per-finger scaling from the first observed hand pose."""
        human_lengths = self._compute_human_finger_lengths(mediapipe_pose)
        self.finger_scaling = (
            human_lengths / (self.robot_finger_lengths + 1e-6)
        )

        self.optimizer.finger_scaling = self.finger_scaling
        self.optimizer.vector_scaling = self.optimizer.build_vector_scaling()

        self.is_calibrated = True
        print(f'[Retargeter] Calibrated finger scaling: {self.finger_scaling}')
        print(f'[Retargeter] Human finger lengths: {human_lengths}')
        print(
            '[Retargeter] Robot finger lengths: '
            f'{self.robot_finger_lengths}'
        )

    def _retarget_optimization(
        self,
        ref_value: np.ndarray,
        target_dir: np.ndarray,
    ) -> np.ndarray:
        """Run the optimizer and update the internal warm-start state."""
        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            target_dir=target_dir.astype(np.float32),
            last_qpos=np.clip(
                self.last_qpos,
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
            ),
        )
        self.last_qpos = qpos
        return self.filter.next(qpos)


class LPFilter:
    """Low-pass filter for smoothing joint positions."""

    def __init__(self, alpha: float) -> None:
        """Initialize the filter with the given smoothing factor."""
        self.alpha = alpha
        self.y: Optional[np.ndarray] = None
        self.is_init = False

    def next(self, x: np.ndarray) -> np.ndarray:  # noqa: A003
        """Return the next filtered output sample."""
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self) -> None:
        """Reset the filter state."""
        self.y = None
        self.is_init = False


__all__ = ['ROBOTISHandRetargeter', 'RetargetingResult']
