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
# This file is derived from `dex_retargeting/robot_wrapper.py` in the
# dexsuite/dex-retargeting project:
# https://github.com/dexsuite/dex-retargeting
#
# Original work Copyright (c) 2023 Yuzhe Qin, licensed under the MIT License.
# Modifications in this file are Copyright 2026 ROBOTIS CO., LTD.
#
# Author: Hyunwoo Nam

"""Thin Pinocchio-based robot wrapper used by the retargeting code."""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin


class RobotWrapper:
    """Wrap a Pinocchio model without handling mimic joints."""

    def __init__(
        self,
        urdf_path: str,
        use_collision: bool = False,
        use_visual: bool = False,
    ) -> None:
        """Create the robot model and allocate its runtime data."""
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        if use_visual or use_collision:
            raise NotImplementedError

        self.q0 = pin.neutral(self.model)
        if self.model.nv != self.model.nq:
            raise NotImplementedError(
                'Cannot handle robots with special joints.'
            )

    @property
    def joint_names(self) -> List[str]:
        """Return all joint names from the Pinocchio model."""
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        """Return the names of joints that contribute degrees of freedom."""
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        """Return the number of generalized joint coordinates."""
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        """Return all frame names exposed by the robot model."""
        return [frame.name for frame in self.model.frames]

    @property
    def joint_limits(self) -> npt.NDArray[np.floating]:
        """Return lower and upper joint limits as a stacked array."""
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    def get_joint_index(self, name: str) -> int:
        """Return the index of a controllable joint by name."""
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str) -> int:
        """Return the Pinocchio frame id for the named body link."""
        if name not in self.link_names:
            raise ValueError(
                f'{name} is not a link name. '
                f'Valid link names:\n{self.link_names}'
            )
        return self.model.getFrameId(name, pin.BODY)

    def get_joint_parent_child_frames(
        self,
        joint_name: str,
    ) -> Tuple[int, int]:
        """Return the parent and child frame ids for a joint frame."""
        joint_id = self.model.getFrameId(joint_name)
        parent_id = self.model.frames[joint_id].parent
        child_id = -1
        for idx, frame in enumerate(self.model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx
        if child_id == -1:
            raise ValueError(f'Cannot find child link of {joint_name}')

        return parent_id, child_id

    def compute_forward_kinematics(self, qpos: npt.NDArray) -> None:
        """Update internal kinematic state for the provided configuration."""
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray[np.floating]:
        """Return the homogeneous transform for a frame."""
        pose: pin.SE3 = pin.updateFramePlacement(
            self.model,
            self.data,
            link_id,
        )
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray[np.floating]:
        """Return the inverse homogeneous transform for a frame."""
        pose: pin.SE3 = pin.updateFramePlacement(
            self.model,
            self.data,
            link_id,
        )
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(
        self,
        qpos: npt.NDArray,
        link_id: int,
    ) -> npt.NDArray[np.floating]:
        """Return the local frame Jacobian for a single link."""
        return pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
