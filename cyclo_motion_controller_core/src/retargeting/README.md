# HX5-D20 precision pinch retargeting

The precision-pinch objective brings one thumb/finger pair's contact points
together and aligns their **lateral tip axes**. It does not align pad normals or
require the tactile surfaces to lie flat against each other. Human distal-finger
directions remain soft targets for both participants, including during a pinch.

## Behavior

- The nearest thumb partner is selected below 30 mm of human tip separation.
  That partner remains selected until its distance reaches 30 mm, preventing
  another nearby finger from stealing the pinch. Activation uses smoothstep
  between 30 mm (zero) and 20 mm (full). A partner switch on release can still
  change the target if another finger is already closed.
- At full activation, contact-point closure replaces that pair's original
  endpoint-vector projection. This avoids fighting a nonzero contact offset.
  Other fingers retain ordinary vector/direction tracking; they are not also
  attached to the same thumb point. Multi-finger contact planning is not included.
- The lateral-axis objective is `axis_weight * (1 - dot(axis_thumb, axis_finger)^2)`.
  Thumb local X and opponent local Y are unoriented lateral axes. Both parallel
  and antiparallel axes are accepted, leaving the pad opening angle free.
- Both participating fingers retain `direction_scale` of their original
  human-direction weight at full activation (default 0.5). This is a soft
  preference, not a hard angular-deviation bound or a guarantee of natural posture.
- Contact-point offsets rotate with their tips. The gradient includes both
  fingers' angular contributions and offset moment arms. Optimization uses
  float64 and the actual URDF joint limits, without widening the limits.
- Calibration now scales human spans by `robot_length / human_length`. Scaled
  wrist-relative tip positions generate all pairwise vectors, so unequal finger
  scales do not create inconsistent target triangles. Calibration still uses the
  first valid observation; start-pose sensitivity remains to be addressed with
  recorded data or an explicit calibration procedure.

## ROS parameters

The `retargeting_teleop` node accepts the following parameters at startup:

```yaml
retargeting_teleop:
  ros__parameters:
    hand_revision: rev3
    pinch_enabled: true
    pinch_contact_weight: 500.0
    pinch_axis_weight: 0.175
    pinch_direction_scale: 0.5
    right_contact_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    left_contact_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

`hand_revision` defaults to `rev2` in the ROS node to preserve the bundled model's
revision. Set it to `rev3` for the new endpoint positions. The Python retargeter
defaults to `urdf`, which preserves the supplied model. Explicit revision choices
replace the five fixed endpoint translations **in memory**, using absolute values
from ROBOTIS's rev2/rev3 descriptions. They are idempotent if the supplied URDF
already has those endpoint positions. Articulated-joint geometry is unchanged.
This does not replace visual/collision meshes or model the pad's changed slope.
The endpoint values were checked against both sides' `*.urdf.xacro` files in
`robotis_hand` at commit `acb125753e570c5f3c38cb37c914555a6ad9215b`.

Each contact-offset array contains 15 values: local `(x, y, z)` in meters for
thumb, index, middle, ring, and little finger, relative to the revision's
`finger_end_*` frames. Left and right offsets are independent. Zero means use the
kinematic endpoint. No unverified tactile-surface or distal-edge offsets are
enabled by default. Define and validate those points before interpreting numerical
closure as physical contact. The closure objective represents an empty pinch;
object thickness, table clearance, force regulation, and tactile feedback are not
modeled here.

Setting `pinch_enabled: false` uses the original DexPilot projection terms without
the precision-pinch attachment/alignment terms. Scaling, input validation,
precision, and strict-limit fixes still apply. Pinch weights are initial values
based on the earlier tip-alignment branch; hardware tuning remains necessary.

## Offline verification

From the repository root, in an environment containing NumPy, NLopt, Pinocchio,
and pytest:

```bash
PYTHONPATH=cyclo_motion_controller_core/src python3 -m pytest -q cyclo_motion_controller_core/test/test_retargeting_pinch.py
```

The tests load the repository's real hand kinematics, without ROS or a robot.
They cover all 20 objective derivatives for both hands, both revisions, all four
thumb partners, transition distances, and nonzero contact offsets. Additional
tests cover revision overrides, scale consistency, invalid inputs, partner
persistence, opening-angle freedom, strict limits, and synthetic solve behavior.
The suite is also registered with `ament_cmake_pytest` for colcon testing.

Synthetic closure/angle results establish numerical behavior only. Physical
contact location, tabletop pickup, tactile calibration, and operator preference
require subsequent experiments.
