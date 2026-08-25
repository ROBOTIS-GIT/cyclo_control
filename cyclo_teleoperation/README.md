# cyclo_teleoperation

This package contains teleoperation applications grouped by follower robot.
The common runtime loads exactly one numeric control-mode plugin and owns safe
transitions, the shared full-model QP, joint limits, and collision constraints.

AI Worker reserves these mode IDs:

- `1`: MoveJ teleoperation. A stopped arm holds its measured stop position.
- `2+`: YAML-configured custom modes, such as `relative_pose`.

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

Preset IDs are global to the AI Worker profile. Left and right definitions are
independent, and each arm has its own duration. Joint velocity limits are
enforced only by the shared QP from the follower URDF:

```yaml
available_presets: [1, 2]

preset.kp_joint: 30.0
preset.tracking_weight: 10.0

presets.1.name: ready
presets.1.left.positions: [0.0, 0.3, 0.15, -2.45, -0.27, 0.69, -0.95]
presets.1.right.positions: [0.0, -0.3, -0.15, -2.45, 0.27, 0.69, 0.95]
presets.1.left.duration: 3.0
presets.1.right.duration: 3.0
```

Start per-arm preset motion without changing the active mode:

```bash
ros2 service call /leader/teleoperation/set_preset \
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
4. Add and export the plugin, then map a free numeric ID starting at two in
   a free positive numeric ID in the YAML.

The default `controlledGroups()` returns `context.enabled_groups`. The
runtime combines it with the active preset arms before applying soft hold,
so a preset arm is controlled by the overlay while any other disabled arm uses
the common soft-hold objective.

Only the selected mode plugin exists at runtime. The lightweight preset overlay
is shared by every mode and remains per-arm across mode changes.
