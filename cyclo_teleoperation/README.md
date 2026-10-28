# cyclo_teleoperation

This package contains teleoperation and model-action applications grouped by follower robot.
The common `cyclo_teleoperation_node` owns source transitions and loads the selected controller
plugin for teleoperation or model EEF control. Model absolute joint control bypasses the
runtime entirely: the model publishes directly to the follower's joint trajectory topics.

The AI Worker example YAML uses these teleoperation mode IDs (IDs are configurable):

- `1`: MoveJ teleoperation. A stopped arm holds its measured stop position.
- `2+`: YAML-configured custom modes, such as `relative_pose`.

## SG2 rev1 configuration

AI Worker selects follower profiles in `config/ffw_sg2_rev1_follower/`:

- `ffw_sg2_teleoperation.yaml`: the unified follower profile, including robot models,
  joints and topics, common controller settings, teleop modes and their gains,
  external control modes, arm presets, and initial/exit poses.

A2 input/model settings live in `config/ffw_a2_leader/ffw_a2_leader_input.yaml`.
Only the leader bringup loads this file into the common Leader JTCB. Follower
bringups load only their own `ffw_<model>_teleoperation.yaml`, declare the
Cyclo node directly, and resolve
`$(find-pkg-share ...)` model paths. Model paths are configured in YAML, not repeated
as launch arguments. `launch_utils.py` only shares command routing and successful
startup sequencing; it does not launch hidden nodes.
SG2 rev1, BG2 rev4, F1, F2, SH5 rev1 and BH5 rev1 hardware/Gazebo bringups are integrated.

The follower bringup is the launch entry point. The intermediate
`ffw_control.launch.py`, `ffw_teleoperation.launch.py`, and
`ffw_follower_action_controller.launch.py` entry points have been removed.

`cyclo_teleoperation_node` is the only executable. Both input paths are always available;
there is no model-only startup option. A follower starts without any leader model,
leader configuration or leader process. Leader FK runs outside the follower runtime.

## Shared teleoperation / model_action runtime

The integrated AI Worker follower bringups (including Gazebo) always start the
common runtime once, after controller activation and any initial-position actions.
There is no `enable_control` argument or separate action node/topic mux.

```bash
ros2 launch ffw_bringup ffw_sg2_follower_ai.launch.py
ros2 launch ffw_bringup ffw_a2_leader_ai.launch.py
```

Both input paths are available. Startup is fixed to `model_action` and the unique
mode with `reference_type: absolute_joint_position`, independent of its numeric ID.
The `initial_source` and `default_model_action_mode` parameters are removed.
Legacy LG2 publishers can use their existing topics: Cyclo does not publish or
replace joint commands at startup or feedback recovery. Holding both A2
buttons toggles the source with both arms initially stopped; individual long
presses enable/pause each arm in teleop. Both short presses switch head/swerve
joystick mode. Joystick motion output is disabled in `model_action` source.

| Purpose | Topic / service |
| --- | --- |
| Leader raw joint reference | `/leader/joint_trajectory_command_broadcaster_{left,right}/raw_joint_trajectory` |
| Model absolute joint input / final follower command | `/leader/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory` |
| Desired follower EEF action (leader-generated in relative teleop, external input in model_action) | `/action/{left,right}/pose` |
| Gripper-only EEF input | `/action/{left,right}/gripper` |
| Measured EEF state | `/state/{left,right}/pose` |
| Hardware joint state | `/joint_states` |
| Leader hardware joint state | `/leader/joint_states` |
| Selected source + heartbeat | `/source` (`std_msgs/msg/String`) |
| Select source | `/set_source` (`std_srvs/srv/SetBool`: true=teleop) |
| Toggle source | `/toggle_source` (`std_srvs/srv/Trigger`) |
| Select mode of the active source | `/set_mode` (`robotis_interfaces/srv/SetControlMode`) |

Modes are registered independently in `control_modes` and `model_action_modes`;
every mode declares `reference_type: absolute_joint_position|absolute_eef_pose`.
EEF inputs are `PoseStamped` in the configured reference frame (`base_link` by
default), using reliable, volatile, KeepLast(1) QoS on the shared action topic.
Gripper-only `JointTrajectory` input uses the same per-arm joint command
topic in EEF mode. Cyclo's joint output contains the arm and configured gripper and uses
`time_from_start=0`; interpolation happens inside the runtime. Direct model joint
messages are not altered, including their timestamps, durations and points.

For arm-only AI Worker followers, set `left_gripper_joint: ""` and
`right_gripper_joint: ""`. The robot plugin then requires no follower gripper
feedback and emits seven arm joints per side. Leader triggers remain optional
inputs (`leader_left_gripper_joint` / `leader_right_gripper_joint`); they are not
forwarded to a hand controller. Hands can be controlled independently on their
existing hand trajectory topics. Arm presets contain only their seven positions.
Keep the gripper names unchanged on SG2/BG2/F1/F2 to retain combined arm/gripper output.

`/action/{left,right}/pose` always represents a desired EEF target, while
`/state/{left,right}/pose` is computed from actual follower feedback. In joint teleop,
Cyclo publishes FK of its joint command. In relative EEF teleop, the common Leader
JTCB captures per-arm leader/follower anchors and publishes an absolute EEF goal.
Follower `AbsolutePoseMode` tracks this input, also used by external EEF control.
Cyclo owns EEF publication during preset/initial/exit overlays. Both joint actions and EEF actions continue
while arms are paused, representing the hold target; measured states continue
while valid follower feedback is received. Publication requires source setup to
complete and fresh feedback, not an enabled teleop arm.

In `model_action`, only the model publishes EEF actions. Cyclo consumes them in
absolute EEF mode without echoing them, including during input timeout/hold. In
direct joint mode Cyclo no longer publishes a derived FK action pose; measured
EEF states remain available. Model pose subscriptions reject local publications,
ignore inputs outside model EEF control and discard old references at handoff.
The former `/command/{left,right}/pose` input is replaced by the shared action
topic. Gripper-only input is `/action/{left,right}/gripper`.

### Lift and Cartesian frames

AI Worker relative teleop targets are held internally relative to `arm_base_link`.
The common leader broadcaster generates the reference using its calculation URDF.
It publishes a base-frame pose stamped with the follower EEF feedback sample.
The follower reconstructs the arm-relative target using the matching measured
frame history and follows the current arm frame. No leader TF or URDF is transferred.
The physical leader may keep wall time while the follower uses simulation time:
input freshness uses reception time, and TF/sample age is compared within the
follower's timestamp domain. A valid simulation `/clock` is still required by the follower.
The measured lift position updates kinematics every cycle, but lift is not an
arm QP command variable. In teleop, moving lift alone therefore does not make
either arm counter-move to hold an old `base_link` height. Arm commands remain open-loop.

Public EEF actions and states stay in `base_link`: teleop targets and measured
states include the actual lift height. External EEF commands remain absolute
`base_link` targets: if lift moves while that target is unchanged, the arm must
compensate to maintain the commanded EEF pose, subject to the configured QP
constraints. External action messages remain publisher-owned and are not echoed
or rewritten by Cyclo. Direct external joint commands are unchanged.

Robot plugins select `follower_control_frame` for each control group. A teleop
pose mode selects `pose_anchor_frame` from these frames, without robot names or
numeric mode IDs in the common controller. An empty anchor preserves base-frame tracking. Leader
deltas retain the shared base axes even when the control frame is rotated. Joints
outside all control groups are observed from feedback and locked in the arm QP;
disabled arms still use the existing soft hold.

New controllers reuse the synchronized `ModeContext` and shared QP runtime;
they should not implement their own lift feedback or joint ownership handling.
Only controllers with relative anchors use `followerControlFramePose()` to
store/reconstruct those anchors. Absolute pose controllers keep absolute targets.

Every external `absolute_joint_position` mode is direct, regardless of its numeric ID. Do not
specify a `plugin` or controller tuning for that mode. No plugin or QP is instantiated,
and Cyclo publishes no joint commands, including holds, while direct mode is active.
The old `/action/{left,right}/joint` output layer is removed. Follower arm, head and
lift input topics remain compatible with legacy bringups.

Direct commands bypass **all Cyclo filtering, interpolation, velocity limits and
collision constraints**. The model/follower integration must supply any required
validation, velocity limits, startup interpolation and input watchdog. A silent arm
retains the follower controller's previous trajectory; Cyclo does not soft-hold it.
Stopping publication does not cancel an already scheduled multi-point trajectory.

### Source handoff

No model start/stop service is required or called. Source and mode changes do not
check external command publishers or recent arm-trajectory traffic. Existing
mode/pose transition checks and the stopped-control-group requirement for mode
changes remain in effect. After departing direct mode, Cyclo sends one measured
hold to replace a pending follower trajectory before starting teleop or EEF control.

Models should publish newly generated actions only while `/source` is `model_action`
with a recent heartbeat, and stop on `starting`, `switching`, `none`, `teleop`, missing
heartbeat or missing follower feedback. Never resume cached actions after reconnecting.
Cyclo cannot block external direct publications, including during teleop, or cancel a
scheduled trajectory during lost feedback. Concurrent external and teleop commands can
conflict at the follower. Integration must honor ownership and bound scheduling horizons.
Follower timeout suspends Cyclo output; valid feedback rebases its references.

For Cyclo-controlled paths, source switches discard old input, stop all groups and rebase the command on fresh
feedback. Requests are rejected while a preset or mode-pose movement is in progress.
A configured teleop exit pose finishes before action takes ownership; returning to
teleop runs its enabled initial pose even when the same mode was used previously.
`/source` reports `starting` or `switching` until the runtime is ready. A successful
source service response acknowledges the request, not completion of its pose sequence.
A failed handoff reports `none` and does not activate the target controller.

Cyclo ignores its inputs from the inactive source. Losing the source heartbeat
for 0.5 s disables A2 raw and joystick motion output. Base release sends one zero
Twist; it does not continually override another base controller.
Joystick motion also stops on incomplete or expired follower feedback; buttons remain
available, and fresh feedback rebases head/lift commands before motion resumes.

Stamped EEF/gripper inputs must be valid, recent and ordered within each input channel.
Zero-stamped joint commands remain supported using reception-time freshness. A single
fresh EEF action is consumed immediately; teleop slow start still waits for its next
post-enable raw sample. Mode service replies and completion statuses share a transition ID.

The runtime takes a domain/source-topic process lock before creating robot outputs,
waits for ownership discovery, and stops if another source publisher is detected.
Do not launch a second runtime alongside an enabled follower runtime.

`TeleoperationNode` connects ROS callbacks and the periodic control loop.
`CommandSourceManager` owns source transitions and publication permissions, with
`RuntimeOwnership` guarding the output owner. `ModeManager` owns the selected plugin and exit/initial/activation
lifecycle; `ModeRegistry` resolves its YAML entries. `FeedbackState` owns feedback
freshness, integrated command state and hold targets; it rebases commands only at
explicit enable/recovery events, not every feedback callback.
`PoseSequenceManager` retains preset/initial/exit interpolation, and `ControlRuntime`
executes the shared QP. `ActionInput` validates EEF/gripper inputs and observes
direct joint commands and their external receive times. These are ordinary classes,
not additional ROS nodes.
The robot plugin owns kinematics, ROS output and the existing
`robotis_interfaces` bridge. Controller plugins only contribute `ModeOutput`;
new plugins using the existing input types need no runtime or AI Worker C++ edits.
Every mode declares `reference_type: absolute_joint_position|absolute_eef_pose`.
Teleop pose modes additionally select `pose_anchor_frame: arm_base_link`; external
EEF modes leave the goal fixed in the base frame. Numeric mode IDs have no special meaning.

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
control_modes.3.reference_type: absolute_joint_position
control_modes.3.kp_joint: 30.0
control_modes.3.tracking_weight: 15.0
```

For plugin-controlled paths, the shared QP enforces joint velocity limits from the follower URDF.

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

An external plugin package can use the exported CMake targets without linking the
AI Worker robot plugin or node runtime:

```cmake
find_package(cyclo_teleoperation REQUIRED)
target_link_libraries(my_controller_plugin PRIVATE
  cyclo_teleoperation::cyclo_teleoperation_runtime)
```

The runtime target includes the plugin interfaces and shared slow-start/constraint
helpers with their transitive dependencies. For interfaces only, link
`cyclo_teleoperation::cyclo_teleoperation_interfaces`. The shared QP is exported
as `cyclo_teleoperation::cyclo_teleoperation_qp`; controller plugins normally do
not need to instantiate or link their own QP.

The default `controlledGroups()` returns `context.enabled_groups`. The
runtime combines it with the active preset arms before applying soft hold,
so a preset arm is controlled by the overlay while any other disabled arm uses
the common soft-hold objective.

Only the selected mode plugin exists at runtime. The lightweight preset overlay
is shared by every mode and remains per-arm across mode changes.

## Use a different leader

The common `LeaderJointTrajectoryCommandBroadcaster` owns the optional pose bridge;
the A2 subclass adds no FK or relative-reference logic. Its input YAML configures
`leader_urdf_path`, optional `leader_base_frame`, `leader_left_eef` / `leader_right_eef`,
and `leader_left_model_joints` / `leader_right_model_joints`. Empty root/EEF settings
use the URDF root and the follower's EEF naming convention respectively. Explicit
joint mapping arrays map raw arm positions, in order, to joints in a differently
named calculation URDF. All movable joints between the selected base and EEF must
be supplied; incomplete mappings are rejected. Revolute, continuous and prismatic
joints are supported. FK uses no collision geometry or QP.

The exported `cyclo_teleoperation_leader_reference` target provides pure FK and
relative-to-absolute pose math without ROS connections. The leader bridge handles
ROS subscriptions, measured follower TF, mode metadata and per-arm anchors outside
the ros2_control update thread. Only the leader loads a leader calculation URDF.
F1/F2 may supply `input_reference_urdf_path` as a portable calculation-model hint;
the follower never constructs or executes a leader model from it. The A2 input
profile enables `leader_use_follower_reference_model` to preserve its existing
follower-sized geometry for these models. Such FK uses the configured arm anchor
as its root, so a non-arm joint such as lift is not treated as a leader input.
Set this flag to false when supplying a different leader calculation URDF.

LG2, A2 and A1 share optional Cartesian translation scaling in their leader input
YAML, under `/leader/joint_trajectory_command_broadcaster.ros__parameters`:

```yaml
cartesian_scaling:
  left_translation_gain: 1.0
  right_translation_gain: 1.0
```

Each positive, finite gain multiplies the selected FK model's displacement from
its per-arm start/resume anchor. The published `/action/left/pose` and
`/action/right/pose` already contain the scaled absolute target positions; the
follower must not apply this gain again. Rotation remains 1:1, and the measured
JointState, FK, gripper, joint-mode mapping/scaling and follower lift motion are
unchanged. A gain of 1.5 maps a 10 cm reference displacement to a 15 cm target
displacement. A2's follower-reference-model option scales that selected model's
displacement, not the physical A2 hand displacement. Defaults of 1.0 preserve
existing behavior. Gains are read-only after parameter declaration; configure
them before launching the leader. Existing hold and resume anchoring remain in
use, and the gains do not modify external model-action pose inputs.
