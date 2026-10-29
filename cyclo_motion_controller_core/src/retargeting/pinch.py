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

"""Precision pinch contact and lateral-axis alignment, without pad flattening."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PinchConfig:
    """Pinch weights; contact offsets are configured separately in meters."""

    enabled: bool = True
    contact_weight: float = 500.0
    axis_weight: float = 0.175
    direction_scale: float = 0.5

    def __post_init__(self) -> None:
        """Reject invalid weights and retain a human-direction preference."""
        weights = [self.contact_weight, self.axis_weight, self.direction_scale]
        if not np.isfinite(weights).all():
            raise ValueError('Pinch weights must be finite')
        if self.contact_weight < 0 or self.axis_weight < 0:
            raise ValueError('Pinch weights must be nonnegative')
        if not 0 < self.direction_scale <= 1:
            raise ValueError('Pinch direction_scale must be in (0, 1]')


def smoothstep(value: np.ndarray) -> np.ndarray:
    """Return a bounded ramp with zero derivative at either endpoint."""
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


class PrecisionPinch:
    """Keep one thumb partner until release, avoiding competing attachments."""

    def __init__(self, enter_distance: float, release_distance: float) -> None:
        """Set distances for full activation and release, in human meters."""
        if not 0 < enter_distance < release_distance:
            raise ValueError('Expected 0 < pinch distance < release distance')
        if not np.isfinite([enter_distance, release_distance]).all():
            raise ValueError('Pinch distances must be finite')
        self.enter_distance = enter_distance
        self.release_distance = release_distance
        self.partner: Optional[int] = None

    def update(self, distances: np.ndarray) -> Tuple[Optional[int], float]:
        """Return opponent index (1..4) and smooth activation for this frame."""
        distances = np.asarray(distances, dtype=np.float64)
        if distances.shape != (4,) or not np.isfinite(distances).all():
            raise ValueError('Expected four finite thumb-to-finger distances')
        if np.any(distances < 0):
            raise ValueError('Pinch distances must be nonnegative')
        if self.partner is not None:
            if distances[self.partner - 1] >= self.release_distance:
                self.partner = None
        if self.partner is None:
            closest = int(np.argmin(distances))
            if distances[closest] < self.release_distance:
                self.partner = closest + 1
        if self.partner is None:
            return None, 0.0
        activation = smoothstep(
            (self.release_distance - distances[self.partner - 1])
            / (self.release_distance - self.enter_distance)
        )
        return self.partner, float(activation)


def pinch_loss_and_gradient(
    tip_poses: np.ndarray,
    contact_offsets: np.ndarray,
    partner: Optional[int],
    activation: float,
    config: PinchConfig,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return loss and world linear/angular derivatives at the tip origins.

    Local X spans the thumb tip; local Y spans the other tips. Aligning these
    unoriented lateral axes leaves the pad opening angle free. Offset points
    move with the tip, so their moment arms contribute angular derivatives.
    Both fingers contribute to every term; no opponent pose is frozen.
    """
    linear = np.zeros((5, 3), dtype=np.float64)
    angular = np.zeros((5, 3), dtype=np.float64)
    if not config.enabled or partner is None or activation == 0:
        return 0.0, linear, angular

    rotations = tip_poses[:, :3, :3]
    arms = np.einsum('fij,fj->fi', rotations, contact_offsets)
    points = tip_poses[:, :3, 3] + arms
    difference = points[0] - points[partner]
    contact_weight = activation * config.contact_weight
    loss = 0.5 * contact_weight * float(difference @ difference)
    linear[0] = contact_weight * difference
    linear[partner] = -linear[0]
    angular[0] = np.cross(arms[0], linear[0])
    angular[partner] = np.cross(arms[partner], linear[partner])

    thumb_axis = rotations[0, :, 0]
    finger_axis = rotations[partner, :, 1]
    cosine = float(thumb_axis @ finger_axis)
    axis_weight = activation * config.axis_weight
    loss += axis_weight * (1.0 - cosine ** 2)
    axis_gradient = -2.0 * axis_weight * cosine * np.cross(
        thumb_axis, finger_axis
    )
    angular[0] += axis_gradient
    angular[partner] -= axis_gradient
    return loss, linear, angular
