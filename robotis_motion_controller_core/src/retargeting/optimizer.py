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

"""DexPilot-based optimizer used for hand retargeting."""

from typing import Callable, List, Optional, Tuple

import nlopt
import numpy as np

from retargeting.robot_wrapper import RobotWrapper


class DexPilotOptimizer:
    """Optimize robot hand joint targets using the DexPilot formulation."""

    retargeting_type = 'DEXPILOT'

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        hand_side: str,
        finger_scaling: List[float],
        huber_delta: float = 0.03,
        norm_delta: float = 0.2,
        project_dist: float = 0.02,
        escape_dist: float = 0.03,
        max_iter: int = 100,
        eta1: float = 1e-4,
        eta2: float = 3e-2,
        orientation_weight: float = 0.5,
    ) -> None:
        """Initialize the optimizer and cache robot/link metadata."""
        self.robot = robot
        self.num_fingers = 5
        self.finger_scaling = np.array(finger_scaling, dtype=np.float32)

        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    (
                        f'Joint {target_joint_name} does not appear in the '
                        'robot XML.'
                    )
                )
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        origin_link_index, task_link_index = self.generate_link_indices(
            self.num_fingers
        )
        self.origin_finger_indices = np.array(origin_link_index, dtype=int)
        self.task_finger_indices = np.array(task_link_index, dtype=int)

        self.target_link_human_indices = (
            np.stack([origin_link_index, task_link_index], axis=0) * 4
        ).astype(int)

        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[i] for i in origin_link_index]
        target_task_link_names = [link_names[i] for i in task_link_index]
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names

        self.huber_delta = float(huber_delta)
        self.norm_delta = norm_delta
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2
        self.orientation_weight = orientation_weight

        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt.set_ftol_abs(1e-6)
        self.opt.set_maxeval(max_iter)
        self.opt_dof = len(idx_pin2target)

        if hand_side == 'right':
            self.proximal_link_names = [
                'finger_r_link4',
                'finger_r_link8',
                'finger_r_link12',
                'finger_r_link16',
                'finger_r_link20',
            ]
        elif hand_side == 'left':
            self.proximal_link_names = [
                'finger_l_link4',
                'finger_l_link8',
                'finger_l_link12',
                'finger_l_link16',
                'finger_l_link20',
            ]
        else:
            raise ValueError(f'Unsupported hand side: {hand_side}')

        self.computed_link_names = list(
            set(target_origin_link_names)
            .union(set(target_task_link_names))
            .union(set(self.proximal_link_names))
        )

        self.proximal_indices = np.array(
            [
                self.computed_link_names.index(name)
                for name in self.proximal_link_names
            ]
        )
        self.tip_indices = np.array(
            [
                self.computed_link_names.index(name)
                for name in finger_tip_link_names
            ]
        )
        self.origin_link_indices = np.array(
            [
                self.computed_link_names.index(name)
                for name in target_origin_link_names
            ],
            dtype=int,
        )
        self.task_link_indices = np.array(
            [
                self.computed_link_names.index(name)
                for name in target_task_link_names
            ],
            dtype=int,
        )
        self.computed_link_indices = [
            self.robot.get_link_index(name)
            for name in self.computed_link_names
        ]

        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self.set_dexpilot_cache(self.num_fingers, eta1, eta2)

        self.vector_scaling = self.build_vector_scaling()

    def set_joint_limit(
        self,
        joint_limits: np.ndarray,
        epsilon: float = 1e-3,
    ) -> None:
        """Set optimizer bounds from a `(dof, 2)` joint limit array."""
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f'Expected joint limits shape {(self.opt_dof, 2)}, '
                f'got {joint_limits.shape}'
            )
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    @staticmethod
    def generate_link_indices(num_fingers: int) -> Tuple[List[int], List[int]]:
        """Generate DexPilot origin/task index pairs for a hand."""
        origin, task = [], []
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin.append(j)
                task.append(i)
        for i in range(1, num_fingers + 1):
            origin.append(0)
            task.append(i)
        return origin, task

    @staticmethod
    def set_dexpilot_cache(
        num_fingers: int,
        eta1: float,
        eta2: float,
    ) -> Tuple[np.ndarray, List[int], List[int], np.ndarray]:
        """Precompute projection bookkeeping for the DexPilot loss."""
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array(
            [eta1] * (num_fingers - 1)
            + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2)
        )

        return (
            projected,
            s2_project_index_origin,
            s2_project_index_task,
            projected_dist,
        )

    def build_vector_scaling(self) -> np.ndarray:
        """Build per-vector scaling factors from per-finger multipliers."""
        factors = np.ones(len(self.origin_finger_indices), dtype=np.float32)
        for idx in range(len(self.origin_finger_indices)):
            indices = []
            if self.origin_finger_indices[idx] > 0:
                indices.append(self.origin_finger_indices[idx] - 1)
            if self.task_finger_indices[idx] > 0:
                indices.append(self.task_finger_indices[idx] - 1)
            if indices:
                factors[idx] = float(np.mean(self.finger_scaling[indices]))
        return factors

    def retarget(
        self,
        ref_value: np.ndarray,
        last_qpos: np.ndarray,
        target_dir: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for the next robot joint configuration."""
        objective_fn = self.get_objective_function(
            ref_value,
            np.array(last_qpos).astype(np.float32),
            target_dir,
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as error:
            print(error)
            return np.array(last_qpos, dtype=np.float32)

    def get_objective_function(
        self,
        target_vector: np.ndarray,
        last_qpos: np.ndarray,
        target_dir: Optional[np.ndarray] = None,
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        """Build the NLopt objective for the current hand target."""
        target_vector = np.asarray(target_vector, dtype=np.float64)

        len_proj = self.num_fingers * (self.num_fingers - 1) // 2
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][
            target_vec_dist[:len_s1] < self.project_dist
        ] = True
        self.projected[:len_s1][
            target_vec_dist[:len_s1] > self.escape_dist
        ] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj],
            target_vec_dist[len_s1:len_proj] <= 0.02,
        )

        normal_weight = np.ones(len_proj, dtype=np.float64)
        high_weight = np.array(
            [200] * len_s1 + [400] * len_s2,
            dtype=np.float64,
        )
        weight_proj = np.where(self.projected, high_weight, normal_weight)
        weight = np.concatenate(
            [
                weight_proj,
                np.ones(self.num_fingers, dtype=np.float32)
                * (len_proj + self.num_fingers),
            ]
        )

        normal_vec = target_vector * self.vector_scaling[:, None]
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)
        projected_vec = dir_vec * self.projected_dist[:, None]

        reference_vec_proj = np.where(
            self.projected[:, None],
            projected_vec,
            normal_vec[:len_proj],
        )
        reference_vec = np.concatenate(
            [reference_vec_proj, normal_vec[len_proj:]],
            axis=0,
        ).astype(np.float64)
        target_vec = reference_vec.astype(np.float32)

        num_vec = reference_vec.shape[0]
        weight_array = weight.astype(np.float32)
        target_dir_array = (
            np.asarray(target_dir, dtype=np.float32)
            if target_dir is not None
            else None
        )

        def huber_loss(x: np.ndarray, y: np.ndarray, delta: float) -> np.ndarray:
            """Compute Huber loss (Smooth L1 loss) element-wise."""
            diff = x - y
            abs_diff = np.abs(diff)
            quadratic = 0.5 * diff ** 2
            linear = delta * (abs_diff - 0.5 * delta)
            return np.where(abs_diff < delta, quadratic, linear)

        def objective(x: np.ndarray, grad_out: np.ndarray) -> float:
            qpos = np.asarray(x, dtype=np.float64)

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(idx)
                for idx in self.computed_link_indices
            ]
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses],
                axis=0,
            ).astype(np.float32)

            origin_pos = body_pos[self.origin_link_indices, :]
            task_pos = body_pos[self.task_link_indices, :]
            robot_vec = task_pos - origin_pos
            len_proj = self.num_fingers * (self.num_fingers - 1) // 2
            num_robot_vecs = len(self.origin_link_indices)
            vec_diff = robot_vec - target_vec[:num_robot_vecs]
            vec_dist = np.linalg.norm(vec_diff, axis=1)
            huber_per_vec = huber_loss(vec_dist, np.zeros_like(vec_dist), self.huber_delta)
            pos_loss = (huber_per_vec * weight_array[:num_robot_vecs]).sum() / num_vec

            if target_dir_array is not None:
                r_prox = body_pos[self.proximal_indices, :]
                r_tip = body_pos[self.tip_indices, :]
                r_dir = r_tip - r_prox
                r_dir_norm_val = np.linalg.norm(r_dir, axis=1, keepdims=True)
                r_dir_norm_val = np.clip(r_dir_norm_val, a_min=1e-6, a_max=None)
                r_dir_norm = r_dir / r_dir_norm_val
                cos_sim = (r_dir_norm * target_dir_array).sum(axis=1)
                dir_loss = (1.0 - cos_sim).sum() * self.orientation_weight
            else:
                dir_loss = 0.0

            reg_loss = self.norm_delta * ((qpos - last_qpos) ** 2).sum()
            total_loss = pos_loss + dir_loss + reg_loss
            result = float(total_loss)

            if grad_out.size > 0:
                # Compute gradient for position loss
                # Gradient of Huber loss w.r.t. vec_dist
                vec_dist_grad = np.zeros_like(vec_dist)
                abs_diff = np.abs(vec_dist)
                mask_quadratic = abs_diff < self.huber_delta
                vec_dist_grad[mask_quadratic] = vec_dist[mask_quadratic]
                vec_dist_grad[~mask_quadratic] = (
                    self.huber_delta * np.sign(vec_dist[~mask_quadratic])
                )
                
                # Gradient w.r.t. vec_diff
                vec_dist_normalized = vec_dist + 1e-8
                vec_diff_grad = (
                    vec_dist_grad[:, None] * vec_diff / vec_dist_normalized[:, None]
                )
                num_robot_vecs = len(self.origin_link_indices)
                vec_diff_grad *= weight_array[:num_robot_vecs, None] / num_vec
                
                # Gradient w.r.t. body_pos
                grad_pos = np.zeros_like(body_pos)
                for i, (origin_idx, task_idx) in enumerate(
                    zip(self.origin_link_indices, self.task_link_indices)
                ):
                    # task_pos contributes positively
                    grad_pos[task_idx, :] += vec_diff_grad[i, :]
                    # origin_pos contributes negatively
                    grad_pos[origin_idx, :] -= vec_diff_grad[i, :]

                # Compute gradient for direction loss
                if target_dir_array is not None:
                    for i, (prox_idx, tip_idx) in enumerate(
                        zip(self.proximal_indices, self.tip_indices)
                    ):
                        r_dir_i = r_dir[i, :]
                        r_dir_norm_i = r_dir_norm[i, :]
                        target_dir_i = target_dir_array[i, :]
                        
                        # Gradient of (1 - cos_sim) w.r.t. r_dir
                        # cos_sim = (r_dir_norm * target_dir).sum()
                        # d/dr_dir (1 - cos_sim) = -d/dr_dir cos_sim
                        r_dir_norm_val_i = r_dir_norm_val[i, 0]
                        
                        # Gradient of normalized direction
                        # d/dr_dir (r_dir / ||r_dir||) = (I - r_dir_norm @ r_dir_norm^T) / ||r_dir||
                        identity = np.eye(3)
                        proj_matrix = np.outer(r_dir_norm_i, r_dir_norm_i)
                        norm_grad = (identity - proj_matrix) / r_dir_norm_val_i
                        
                        # Gradient of cos_sim w.r.t. r_dir
                        cos_sim_grad = norm_grad @ target_dir_i
                        
                        # Gradient w.r.t. body_pos
                        grad_pos[tip_idx, :] -= cos_sim_grad * self.orientation_weight
                        grad_pos[prox_idx, :] += cos_sim_grad * self.orientation_weight

                # Reshape grad_pos for jacobian multiplication
                grad_pos = grad_pos[:, None, :]

                jacobians = []
                for index, link_id in enumerate(self.computed_link_indices):
                    link_jacobian = (
                        self.robot.compute_single_link_local_jacobian(
                            qpos,
                            link_id,
                        )[:3, ...]
                    )
                    link_pose = target_link_poses[index]
                    link_rot = link_pose[:3, :3]
                    jacobians.append(link_rot @ link_jacobian)

                jacobians = np.stack(jacobians, axis=0)
                if jacobians.shape[2] > self.opt_dof:
                    jacobians = jacobians[:, :, self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, jacobians).sum(axis=0).ravel()
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)
                grad_out[:] = np.asarray(grad_qpos, dtype=np.float64)

            return result

        return objective
