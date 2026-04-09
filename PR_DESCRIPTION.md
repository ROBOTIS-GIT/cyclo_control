Hello, thank you for sharing this great codebase.
I have been using the `retargeting/` package for robot teleoperation in an Isaac Sim environment and would like to share some information that may be helpful :)

---

# PICO VR Teleoperation Support 

A reference example for teleoperating the AI Worker ffw_sh5 model using VR. This code is also applicable to Isaac Sim.

The pipeline follows a two-stage coordinate transformation: VR hand tracking data → 3D hand model frame (MANO convention) → Robot URDF frame. Because the intermediate representation is MANO-compatible, any VR device that outputs hand tracking in MANO convention can be integrated without further changes to the retargeting logic.

## Changes

### 1. Added `examples/teleoperation_mujoco_pico/`

A minimal, self-contained example that uses only the `retargeting` module to teleoperate the HX5-D20 hand in MuJoCo via PICO VR.

| File | Description |
|---|---|
| `examples/teleoperation_mujoco_pico/main_control.py` | New |
| `examples/teleoperation_mujoco_pico/README.md` | Pipeline walkthrough |

### 2. Add `set_pre_calibration()` to `seq_retarget.py`

`teleop_retargeting.py` currently applies `finger_scaling` using the very first hand frame received. Any noise or inaccurate detection in that first frame propagates directly into the scaling and cannot be corrected without restarting. In my XRobotToolkit with PICO VR environment, I observed noticeable variation between runs.

This PR adds a `set_pre_calibration()` method to `seq_retarget.py` that accepts pre-measured finger lengths (`human_lengths: list[float]`) and immediately computes and synchronises `finger_scaling`, bypassing the first-frame auto-calibration.

In `teleop_retargeting.py`, if a `PRE_CALIBRATION` variable (= human finger lengths) is declared and not `None`, that value is used for scaling at startup. If it is `None`, the existing first-frame behaviour is preserved. Please refer to `teleop_retargeting.py`.

This allows a practical two-phase workflow:
- **First test run** — use first-frame auto-calibration to get started quickly.
- **Production run** — supply values measured by `wrist_to_tip_calibration.py` to guarantee consistent retargeting across every run.

`wrist_to_tip_calibration.py` is also included. It reads hand joint data from a PICO VR Ultra 4 via XRobotToolkit, accumulates samples while the user holds an open-palm pose, and outputs a ready-to-paste array for `PRE_CALIBRATION` / `set_pre_calibration()`.

| File | Description |
|---|---|
| `cyclo_motion_controller_core/src/retargeting/seq_retarget.py` | Added `set_pre_calibration()` |
| `cyclo_motion_controller_ros_py/scripts/wrist_to_tip_calibration.py` | New |

### 3. Tighten HX5-D20 Thumb Joint Limits

The HX5-D20 hand differs from most widely-used dexterous hands (Allegro, Shadow, LEAP, Inspire, and common anthropomorphic abstractions) in that the order of thumb joints 1 and 2 is reversed.

In my Isaac Sim teleoperation tests, depending on the hand pose at the start of teleoperation, the dex-pilot occasionally returned a solution where the nail-bed and nail positions of the thumb were swapped. This arises because the wide range of `joint_2` (joint_b) (`0` to `3.14` rad) allows multiple valid joint positions, and the solver may converge to an unintended one.

Constraining the range to something more human-like (`0` to `1.57` rad) eliminates the ambiguous solutions while still supporting all common grasp types (GRASP, PINCH, FIST) without any loss of functionality.

In my development environment I enforced this limit in code before passing values downstream, but I have reflected it directly in the URDF here so that it applies universally and confirmed that the hand operates correctly with this change.

* NOTICE: I downloaded [hx5_d20_left.urdf.xacro](https://github.com/ROBOTIS-GIT/robotis_hand/blob/main/robotis_hand_description/urdf/hx5_d20_rev2/hx5_d20_left.urdf.xacro) (also right) and [ffw_sh5.xml](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/blob/main/robotis_ffw/ffw_sh5.xml) from the official repositories and have been applying this correction manually. After checking the repositories again before submitting this PR, I found that the same correction has already been applied upstream. The same correction may also need to be reflected in the linked URDF and XML files. As this is part of my ongoing troubleshooting process, I have kept the proposal as-is and will attach a demonstration video.

| File | Description |
|---|---|
| `cyclo_motion_controller_models/models/hx5_d20/hx5_d20_left.urdf` | Thumb `joint_2` limit tightened |
| `cyclo_motion_controller_models/models/hx5_d20/hx5_d20_right.urdf` | Thumb `joint_2` limit tightened |

### Motivation

The `retargeting` package was designed with portability in mind, and I have been using it together with a CuRobo IK solver (for arm) in both MuJoCo and Isaac Sim environments for remote teleoperation.
The per-finger scaling approach in the provided hand retargeting code works remarkably well in practice.

By separating the coordinate transformation into two explicit steps — MANO convention → Robot URDF base frame (`_R_MANO_TO_ROBOT`) — swapping the VR device or replacing the robot hand requires updating only one matrix, leaving the rest of the pipeline intact.

---

# Out of Scope

No changes are made to ROS 2 node logic, controller behaviour, or the build system.

I have been running teleoperation in simulation using a CuRobo-based IK solver together with the provided retargeting package in an Isaac Sim environment, and I have ported the parts I thought would be worth sharing back into this repository. :)
