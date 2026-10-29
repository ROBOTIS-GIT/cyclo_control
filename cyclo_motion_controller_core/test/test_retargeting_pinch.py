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

"""Offline pinch regressions using real HX5-D20 kinematics and numerical gradients."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

from retargeting.hand_geometry import endpoint_translations
from retargeting.pinch import PinchConfig, pinch_loss_and_gradient, PrecisionPinch
from retargeting.seq_retarget import ROBOTISHandRetargeter


MODELS = Path(__file__).resolve().parents[2] / 'cyclo_motion_controller_models/models/hx5_d20'


def make_retargeter(side='right', revision='rev3', **kwargs):
    """Load repository kinematics without ROS, hardware, or mesh loading."""
    return ROBOTISHandRetargeter(
        str(MODELS / f'hx5_d20_{side}.urdf'), side, revision, **kwargs,
    )


def poses_at(retargeter, q):
    """Read the optimizer's link poses at q."""
    retargeter.robot.compute_forward_kinematics(q)
    return np.stack([
        retargeter.robot.get_link_pose(i).copy()
        for i in retargeter.optimizer.computed_link_indices
    ])


def references_at(retargeter, q, partner=1, distance=0.01):
    """Create a synthetic pinch with consistent reference vectors."""
    opt = retargeter.optimizer
    poses = poses_at(retargeter, q)
    positions = poses[:, :3, 3].copy()
    directions = positions[opt.tip_indices] - positions[opt.proximal_indices]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions[opt.tip_indices[0]] = positions[opt.tip_indices[partner]] + [distance, 0, 0]
    vectors = positions[opt.task_link_indices] - positions[opt.origin_link_indices]
    return vectors, directions


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('revision', ['rev2', 'rev3'])
@pytest.mark.parametrize('partner', [1, 2, 3, 4])
@pytest.mark.parametrize('distance', [0.01, 0.025, 0.05])
def test_full_objective_gradient(side, revision, partner, distance):
    """Check all 20 derivatives, including both rotations and contact moment arms."""
    offsets = np.array([
        [0.002, 0.003, -0.001], [0.004, -0.002, 0.001],
        [-0.003, 0.001, 0.002], [0.001, 0.002, 0.003], [0.002, 0.003, 0.004],
    ])
    retargeter = make_retargeter(side, revision, contact_offsets=offsets)
    q = retargeter.last_qpos + np.linspace(-0.03, 0.03, 20)
    vectors, directions = references_at(retargeter, q, partner, distance)
    # Isolate the requested pairing even if other fingertips happen to be near.
    vectors[:4] = [0.1, 0, 0]
    vectors[partner - 1] = [distance, 0, 0]
    objective = retargeter.optimizer.get_objective_function(vectors, q + 0.02, directions)
    gradient = np.empty_like(q)
    assert np.isfinite(objective(q, gradient))
    numerical = np.zeros_like(q)
    for joint in range(q.size):
        step = np.zeros_like(q)
        step[joint] = 1e-6
        numerical[joint] = (
            objective(q + step, np.array([])) - objective(q - step, np.array([]))
        ) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_zero_residual_is_finite(side):
    """Exercise exact vector/direction matches, where norm derivatives vanish."""
    retargeter = make_retargeter(side, pinch_config=PinchConfig(enabled=False))
    opt = retargeter.optimizer
    q = retargeter.last_qpos.copy()
    poses = poses_at(retargeter, q)
    positions = poses[:, :3, 3]
    vectors = positions[opt.task_link_indices] - positions[opt.origin_link_indices]
    directions = positions[opt.tip_indices] - positions[opt.proximal_indices]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Keep this test on unprojected vectors.
    opt.project_dist = 1e-8
    opt.escape_dist = 2e-8
    objective = opt.get_objective_function(vectors, q, directions)
    gradient = np.zeros_like(q)
    assert abs(objective(q, gradient)) < 1e-12
    np.testing.assert_allclose(gradient, 0, atol=1e-12)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_revision_endpoints_and_no_double_extension(side, tmp_path):
    """Verify the measured 1.49 mm change and idempotent absolute overrides."""
    old, new = make_retargeter(side, 'rev2'), make_retargeter(side, 'rev3')
    p_old = poses_at(old, old.last_qpos)
    p_new = poses_at(new, new.last_qpos)
    for index, tip in enumerate(new.optimizer.tip_indices):
        delta_local = p_old[tip, :3, :3].T @ (p_new[tip, :3, 3] - p_old[tip, :3, 3])
        expected = [0, 0, 0.00149]
        if index == 0:
            expected = [0, 0.00149 if side == 'right' else -0.00149, 0]
        np.testing.assert_allclose(delta_local, expected, atol=1e-12)
    original = MODELS / f'hx5_d20_{side}.urdf'
    before = original.read_bytes()
    root = ET.fromstring(before)
    for name, xyz in endpoint_translations(side, 'rev3').items():
        root.find(f"joint[@name='{name}']/origin").set('xyz', ' '.join(map(str, xyz)))
    rev3_path = tmp_path / 'rev3.urdf'
    ET.ElementTree(root).write(rev3_path)
    already_rev3 = ROBOTISHandRetargeter(str(rev3_path), side, 'rev3')
    np.testing.assert_allclose(poses_at(already_rev3, new.last_qpos), p_new, atol=1e-12)
    preserved = ROBOTISHandRetargeter(str(rev3_path), side, 'urdf')
    np.testing.assert_allclose(poses_at(preserved, new.last_qpos), p_new, atol=1e-12)
    assert original.read_bytes() == before


def test_partner_persistence_and_release():
    """A nearer second finger must not steal an active precision pinch."""
    pinch = PrecisionPinch(0.02, 0.03)
    assert pinch.update(np.array([0.01, 0.04, 0.05, 0.06])) == (1, 1.0)
    assert pinch.update(np.array([0.025, 0.001, 0.05, 0.06])) == (1, pytest.approx(0.5))
    assert pinch.update(np.array([0.031, 0.001, 0.05, 0.06])) == (2, 1.0)
    assert pinch.update(np.full(4, 0.04)) == (None, 0.0)
    for threshold, expected in [(0.02, 1.0), (0.03, 0.0)]:
        for delta in [-1e-7, 0, 1e-7]:
            pinch.partner = None
            _, activation = pinch.update(np.array([threshold + delta, 0.1, 0.1, 0.1]))
            assert activation == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize('opening', [-1.0, -0.3, 0.0, 0.7, 1.3])
@pytest.mark.parametrize('sign', [-1, 1])
def test_axis_alignment_does_not_flatten_pads(opening, sign):
    """Aligned lateral axes have zero pinch loss at different opening angles."""
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    # Opponent local Y maps onto +/- thumb X. Rotate around that common
    # lateral axis to change pad opening without changing alignment.
    poses[1, :3, :3] = pin.exp3(np.array([opening, 0, 0])) @ pin.exp3(
        np.array([0, 0, sign * np.pi / 2])
    )
    loss, linear, angular = pinch_loss_and_gradient(
        poses, np.zeros((5, 3)), 1, 1.0, PinchConfig(),
    )
    assert abs(loss) < 1e-12
    np.testing.assert_allclose(linear, 0, atol=1e-12)
    np.testing.assert_allclose(angular, 0, atol=1e-12)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('revision', ['rev2', 'rev3'])
def test_synthetic_pinch_closes_without_losing_direction(side, revision):
    """Compare closure to legacy projection on a static synthetic request."""
    results = []
    for enabled in [False, True]:
        retargeter = make_retargeter(side, revision, pinch_config=PinchConfig(enabled=enabled))
        opt = retargeter.optimizer
        q = retargeter.last_qpos.copy()
        vectors, directions = references_at(retargeter, q, distance=0.005)
        for _ in range(30):
            q = opt.retarget(vectors, q, directions)
            assert np.isfinite(q).all()
            assert np.all(q >= retargeter.joint_limits[:, 0] - 1e-10)
            assert np.all(q <= retargeter.joint_limits[:, 1] + 1e-10)
        poses = poses_at(retargeter, q)
        tips = poses[opt.tip_indices]
        gap = np.linalg.norm(tips[0, :3, 3] - tips[1, :3, 3])
        lateral_error = np.arccos(np.clip(abs(tips[0, :3, 0] @ tips[1, :3, 1]), 0, 1))
        actual_dirs = tips[:, :3, 3] - poses[opt.proximal_indices, :3, 3]
        actual_dirs /= np.linalg.norm(actual_dirs, axis=1, keepdims=True)
        direction_errors = np.arccos(np.clip(np.sum(actual_dirs * directions, axis=1), -1, 1))
        results.append((gap, lateral_error, direction_errors))
    assert results[1][0] < results[0][0] * 0.25
    assert results[1][1] < results[0][1]
    assert np.max(results[1][2][:2]) < np.deg2rad(10)


@pytest.mark.parametrize('bad', ['nan', 'collapsed', 'shape'])
def test_bad_input_does_not_change_state(bad):
    """Reject invalid observations before calibration or pinch state changes."""
    retargeter = make_retargeter()
    q_before = retargeter.last_qpos.copy()
    pose = np.ones((21, 3))
    if bad == 'nan':
        pose[0, 0] = np.nan
    elif bad == 'shape':
        pose = pose[:20]
    with pytest.raises(ValueError):
        retargeter.retarget(pose)
    assert not retargeter.is_calibrated
    assert retargeter.optimizer.pinch.partner is None
    np.testing.assert_array_equal(retargeter.last_qpos, q_before)


def test_configuration_validation():
    """Reject weights that erase posture tracking or invalidate the objective."""
    for kwargs in [{'direction_scale': 0}, {'axis_weight': -1}, {'contact_weight': np.nan}]:
        with pytest.raises(ValueError):
            PinchConfig(**kwargs)
    with pytest.raises(ValueError):
        make_retargeter(revision='rev4')
    with pytest.raises(ValueError):
        make_retargeter(contact_offsets=np.zeros((4, 3)))


def test_scaling_maps_human_lengths_to_robot_lengths():
    """A larger human span must shrink to the robot span, not expand."""
    retargeter = make_retargeter()
    pose = np.zeros((21, 3))
    lengths = retargeter.robot_finger_lengths * np.array([1.2, 1.5, 1.4, 1.3, 1.1])
    pose[[4, 8, 12, 16, 20], 0] = lengths
    retargeter._calibrate_scaling(pose)
    np.testing.assert_allclose(
        lengths * retargeter.finger_scaling, retargeter.robot_finger_lengths,
    )


def test_scaled_vectors_form_consistent_triangle():
    """Unequal finger scales must preserve pairwise vector closure."""
    retargeter = make_retargeter()
    opt = retargeter.optimizer
    opt.finger_scaling = np.array([0.8, 1.2, 1.1, 0.9, 1.3])
    points = np.array([
        [0, 0, 0], [0.02, 0.03, 0.08], [0.04, 0.03, 0.15],
        [0.02, 0.03, 0.16], [0, 0.03, 0.15], [-0.02, 0.03, 0.13],
    ])
    vectors = points[opt.task_finger_indices] - points[opt.origin_finger_indices]
    scaled = opt.scale_reference_vectors(vectors)
    # finger1 - finger2 == (finger1 - wrist) - (finger2 - wrist)
    np.testing.assert_allclose(scaled[0], scaled[10] - scaled[11], atol=1e-15)
    np.testing.assert_allclose(scaled[4], scaled[11] - scaled[12], atol=1e-15)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_retarget_pipeline(side):
    """Exercise calibration, solving, and filtering together from 21 landmarks."""
    retargeter = make_retargeter(side)
    opt = retargeter.optimizer
    poses = poses_at(retargeter, retargeter.last_qpos)
    pose = np.zeros((21, 3))
    wrist = poses[opt.origin_link_indices[-1], :3, 3]
    pose[0] = wrist
    pose[[4, 8, 12, 16, 20]] = poses[opt.tip_indices, :3, 3]
    pose[[3, 7, 11, 15, 19]] = poses[opt.proximal_indices, :3, 3]
    translation = pose[8] + [0.005, 0, 0] - pose[4]
    pose[4] += translation
    pose[3] += translation
    for _ in range(20):
        result = retargeter.retarget(pose)
        assert np.isfinite(result.robot_qpos).all()
        assert np.all(result.robot_qpos >= retargeter.joint_limits[:, 0] - 1e-10)
        assert np.all(result.robot_qpos <= retargeter.joint_limits[:, 1] + 1e-10)
    assert retargeter.is_calibrated
    assert opt.pinch.partner == 1
