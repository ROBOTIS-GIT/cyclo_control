# cyclo_teleoperation

This package contains teleoperation and action-input applications grouped by follower robot.
The common `cyclo_teleoperation_node` loads exactly one numeric control-mode plugin and owns safe
transitions, the shared full-model QP, joint limits, and collision constraints.

The AI Worker example YAML uses these teleoperation mode IDs (IDs are configurable):

- `1`: MoveJ teleoperation. A stopped arm holds its measured stop position.
- `2+`: YAML-configured custom modes, such as `relative_pose`.

## SG2 rev1 configuration

AI Worker selects follower profiles in `config/ffw_sg2_rev1_follower/`:

- `ffw_sg2_teleoperation.yaml`: robot model, teleop modes, presets and initial/exit poses.
- `ffw_sg2_model_action.yaml`: model-action topics and controller modes.
- `ffw_sg2_controller_parameters.yaml`: controller and pose-sequence tuning.

A2 input/model settings remain in `config/ffw_a2_leader/ffw_a2_leader_reference.yaml`.
All four parameter files target the `cyclo_teleoperation` ROS node. The SG2 hardware
and Gazebo bringups declare this node directly, list its profiles, and resolve
`$(find-pkg-share ...)` model paths. Model paths are configured in YAML, not repeated
as launch arguments. `launch_utils.py` only shares command routing and successful
startup sequencing; it does not launch hidden nodes.
Only SG2 rev1 is integrated at present; other follower bringups are unchanged.

The follower bringup is the launch entry point. The intermediate
`ffw_control.launch.py`, `ffw_teleoperation.launch.py`, and
`ffw_follower_action_controller.launch.py` entry points have been removed.

`cyclo_control_node` and `cyclo_action_controller_node` remain compatibility executable
names for the same runtime, not additional controllers. Do not run them together.

## Shared teleoperation / action runtime

With AI Worker, launch the follower with `enable_control:=true` and start the A2
hardware bringup normally. The follower launch starts the common runtime once,
after its initial-position actions finish. No separate action node or topic mux
is needed. `enable_control:=false` (default) preserves the legacy LG2 connection.

```bash
ros2 launch ffw_bringup ffw_sg2_follower_ai.launch.py enable_control:=true
ros2 launch ffw_bringup ffw_a2_leader_ai.launch.py
```

Both input paths are available; only the selected source can command the robot.
The default `initial_source:=action` waits for model commands. Holding both A2
buttons toggles the source with both arms initially stopped; individual long
presses enable/pause each arm in teleop. Both short presses switch head/swerve
joystick mode. Joystick motion output is disabled in action source.

| Purpose | Topic / service |
| --- | --- |
| Raw hardware reference | `/reference/{left,right}/joint` |
| Model joint / absolute EEF input | `/command/{left,right}/{joint,pose}` |
| Applied joint action / EEF reference | `/action/{left,right}/{joint,pose}` |
| Measured EEF state | `/state/{left,right}/pose` |
| Hardware joint state | `/joint_states` |
| Selected source + heartbeat | `/source` (`std_msgs/msg/String`) |
| Select source | `/set_source` (`std_srvs/srv/SetBool`: true=teleop) |
| Toggle source | `/toggle_source` (`std_srvs/srv/Trigger`) |
| Select mode of the active source | `/set_mode` (`robotis_interfaces/srv/SetControlMode`) |

Modes are registered independently in `control_modes` and `action_modes`; action
modes also declare `reference_type: absolute_joint_position|absolute_eef_pose`.
EEF inputs are `PoseStamped` in the configured reference frame (`base_link` by
default). Gripper-only `JointTrajectory` input uses the same per-arm joint command
topic in EEF mode. Joint output always contains the arm and gripper and uses
`time_from_start=0`; interpolation happens inside the runtime.

Source switches discard old input, stop all groups and rebase the command on fresh
feedback. Requests are rejected while a preset or mode-pose movement is in progress.
A configured teleop exit pose finishes before action takes ownership; returning to
teleop runs its enabled initial pose even when the same mode was used previously.
`/source` reports `starting` or `switching` until the runtime is ready. A successful
source service response acknowledges the request, not completion of its pose sequence.
A failed handoff reports `none` and does not activate the target controller.

Inputs from the inactive source are ignored. Losing the source heartbeat
for 0.5 s disables A2 raw and joystick motion output. Base release sends one zero
Twist; it does not continually override another base controller.
Joystick motion also stops on incomplete or expired follower feedback; buttons remain
available, and fresh feedback rebases head/lift commands before motion resumes.

Stamped action inputs must be valid, recent and ordered within each input channel.
Zero-stamped joint commands remain supported using reception-time freshness. A single
fresh joint action is consumed immediately; teleop slow start still waits for its next
post-enable raw sample. Mode service replies and completion statuses share a transition ID.

The runtime takes a domain/source-topic process lock before creating robot outputs,
waits for ownership discovery, and stops if another source publisher is detected.
Do not launch the compatibility action entry point alongside an enabled follower runtime.

`CommandSourceManager` owns source transitions, `RuntimeOwnership` guards the output owner,
`ActionInput` validates and timestamps group
commands, `ModeRegistry` resolves YAML plugin IDs, and `ControlRuntime` executes the
shared QP. The robot plugin owns kinematics, ROS output and the existing
`robotis_interfaces` bridge. Controller plugins only contribute `ModeOutput`;
new plugins using the existing input types need no runtime or AI Worker C++ edits.

Teleoperation enable/disable independently selects `left`, `right`, or `both` arms. Preset motion is
a common per-arm overlay, not a control mode. Requesting a preset disables
teleoperation for the selected arm and moves it from current follower feedback.
Enabling that arm again cancels its preset and reconnects it to the active mode.

A disabled arm keeps tracking its captured stop position through a configurable
soft objective (`hold.tracking_weight`). Collision avoidance may move it only
when necessary, after which the same objective returns it to the stop position.

## Configure modes and presets

Mode IDs map to plugins in the follower YAML:

```yaml
available_control_modes: [1, 2, 3]

control_modes.3.name: movej_precise
control_modes.3.plugin: cyclo_teleoperation/MoveJMode
control_modes.3.kp_joint: 30.0
control_modes.3.tracking_weight: 15.0
```

Joint velocity limits are always enforced by the shared QP from the follower URDF.

Preset IDs are global to the AI Worker profile. Each arm independently selects
a preset, using that preset's duration. Joint velocity limits are
enforced only by the shared QP from the follower URDF:

```yaml
available_presets: [1, 2]

pose_sequence.kp_joint: 30.0
pose_sequence.tracking_weight: 10.0

presets.1.name: ready
presets.1.step_names: [ready]
presets.1.steps.ready.left.positions: [0.0, 0.3, 0.15, -2.45, -0.27, 0.69, -0.95]
presets.1.steps.ready.right.positions: [0.0, -0.3, -0.15, -2.45, 0.27, 0.69, 0.95]
presets.1.duration: 3.0
```

Start per-arm preset motion without changing the active mode:

```bash
ros2 service call /set_preset \
  robotis_interfaces/srv/SetPreset \
  "{target_arm: both, left_preset_id: 1, right_preset_id: 2}"
```

The request first disables teleoperation for every arm selected by `target_arm`,
then starts the selected preset. Repeating the same preset ID restarts it from
current follower feedback. A joystick enable for that arm cancels the preset.
The active mode objectives, preset objectives, and collision constraints are
solved together against the coupled AI Worker model.

## Add a custom controller

For a new control law:

1. Derive a class from `TeleoperationMode` under `controllers/common` for a reusable
   controller, or `robots/<robot>/controllers` for a robot-specific controller.
2. Implement `configure`, `activate`, `onGroupsEnabled`, and `update`.
3. Fill `ModeOutput`; do not publish commands or create another QP.
4. Add and export the plugin, then map any free positive numeric ID in the YAML.

The default `controlledGroups()` returns `context.enabled_groups`. The
runtime combines it with the active preset arms before applying soft hold,
so a preset arm is controlled by the overlay while any other disabled arm uses
the common soft-hold objective.

Only the selected mode plugin exists at runtime. The lightweight preset overlay
is shared by every mode and remains per-arm across mode changes.
