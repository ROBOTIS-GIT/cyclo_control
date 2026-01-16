"""
Pinocchio Library Overview:
===========================

Pinocchio is a fast and flexible C++ library for rigid body dynamics and kinematics.
It provides efficient algorithms for robot motion computation.

Key Concepts:
-------------

1. Model vs Data:
   - Model: Static robot structure (joints, links, frames, inertias) - never changes
   - Data: Dynamic computation results (positions, velocities, accelerations) - updated during computation
   - Separation allows efficient memory management and parallel computation

2. SE3 Transform and Coordinate Frame Notation:
   - SE3 represents a 6D pose (3D position + 3D orientation) in Special Euclidean group
   - Contains rotation matrix (3x3) and translation vector (3x1)
   - Can be multiplied: A * B means "transform from frame B to frame A"
   - Inverse: A.inverse() gives transform from A back to origin

   Transform Notation (aMb format):
   - Format: aMb means "transform from frame B to frame A"
   - 'a' = target frame, 'M' = transform matrix, 'b' = source frame
   - Reading: "pose of frame b expressed in frame a"

   Common transforms in this code:
   - oMbase: transform from base frame to origin/world frame (o = origin)
   - oMbase_inv: inverse of oMbase = transform from origin to base frame
   - oMtip: transform from tip frame to origin/world frame
   - baseMtip: transform from tip frame to base frame
   - oMf[frame_id]: transform from frame to origin (stored in data.oMf array)

   Transform chain rule:
   - baseMtip = oMbase_inv * oMtip
   - This converts tip pose from world frame to base frame
   - Multiplication order: left-to-right means "apply transforms right-to-left"

3. Forward Kinematics (FK):
   - Given joint angles (q), compute end-effector pose
   - Process: q -> forwardKinematics() -> updateFramePlacements() -> get pose from data.oMf[frame_id]

4. Jacobian:
   - Matrix that maps joint velocities to end-effector velocity
   - Size: 6xN (6 for 6D velocity, N for number of joints)
   - LOCAL frame: velocity expressed in end-effector's own frame
   - WORLD frame: velocity expressed in world frame

5. Inverse Kinematics (IK):
   - Given desired end-effector pose, find joint angles
   - Uses iterative method (Newton-Raphson) with Jacobian
   - CLIK (Closed-Loop IK): continuously corrects error using feedback

6. Frame vs Joint:
   - Joint: connection between two links (revolute, prismatic, etc.)
   - Frame: coordinate system attached to a link (for pose queries)
   - A frame has a parentJoint that connects it to the kinematic tree

7. Reduced Model:
   - Full model contains all joints in URDF
   - Reduced model locks unused joints (not in kinematic chain)
   - Improves computation efficiency by reducing DOF (degrees of freedom)

8. SE3 Log Map:
   - Maps SE3 transform to 6D vector (tangent space)
   - Used for error computation in IK: log(current^-1 * target)
   - Jlog6: Jacobian of log map, needed for proper error propagation

9. Integration:
   - pin.integrate(model, q, v*dt): updates joint configuration by integrating velocity
   - Handles different joint types (revolute, prismatic, spherical, etc.)

Transform Matrix Variables in This Code:
-----------------------------------------

Variable naming convention (aMb format):
- 'a' = target frame (where we express the pose)
- 'M' = transform matrix (4x4 homogeneous transformation)
- 'b' = source frame (the frame whose pose we're describing)
- Reading: "pose of frame b expressed in frame a"

Specific variables used:

1. oMbase:
   - Transform from base frame to origin/world frame
   - Computed during init: oMbase = data.oMf[base_frame_id]
   - Represents: "where is the base located in the world?"

2. oMbase_inv:
   - Inverse of oMbase = transform from origin to base frame
   - Stored as: oMbase_inv = oMbase.inverse()
   - Used to convert poses from world frame to base frame
   - Example: baseMtip = oMbase_inv * oMtip

3. oMtip:
   - Transform from tip frame to origin/world frame
   - Computed by Pinocchio: oMtip = data.oMf[tip_frame_id]
   - Represents: "where is the tip located in the world?"

4. baseMtip:
   - Transform from tip frame to base frame
   - Computed by: baseMtip = oMbase_inv * oMtip
   - This is what users typically want (pose relative to base, not world)
   - Returned by solve_fk() as the result

5. oMtarget:
   - Target pose in world frame (used internally in IK)
   - Converted from user input: oMtarget = oMbase_inv.inverse() * baseMtarget
   - User provides baseMtarget, we convert to oMtarget for Pinocchio

6. data.oMf[frame_id]:
   - Array of transforms: "origin to Frame" for all frames
   - oMf[frame_id] = transform from frame to origin/world frame
   - Updated by: pin.updateFramePlacements(model, data, q)
   - Contains poses of all frames in the robot model

Why coordinate conversion?
- Pinocchio always computes in world/origin frame
- Users typically work in base frame (robot's local coordinate system)
- We convert between frames: base <-> world
"""

import pinocchio as pin  # Pinocchio library for robot kinematics
import numpy as np  # NumPy for numerical operations
import sys  # System-specific parameters and functions

class PinocchioSolver:
    def __init__(self):
        self.model = None  # Pinocchio robot model
        self.data = None  # Pinocchio data structure for computations
        self.tip_frame_id = None  # Frame ID of the tip link
        self.oMbase_inv = pin.SE3.Identity()  # Inverse transform from origin to base frame
        self.q_neutral = None  # Neutral joint configuration

    def init(self, urdf_content: str, base_link: str, tip_link: str) -> bool:
        """
        Initialize the solver with URDF content and chain limits.

        This method:
        1. Parses URDF to build full robot model
        2. Identifies kinematic chain from base to tip
        3. Creates reduced model by locking joints not in the chain
        4. Computes base frame transform for coordinate conversion

        Args:
            urdf_content: XML string of the URDF (Unified Robot Description Format).
            base_link: Name of the base link (starting point of kinematic chain).
            tip_link: Name of the tip link (end-effector, target of IK/FK).

        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            # 1. Build Full Model from URDF
            full_model = pin.buildModelFromXML(urdf_content)  # Parse URDF and build full robot model

            if not full_model.existFrame(tip_link):  # Check if tip link exists in the model
                print(f"[PinocchioSolver] Tip link {tip_link} not found.", file=sys.stderr)
                return False

            full_tip_id = full_model.getFrameId(tip_link)  # Get frame ID of tip link
            tip_joint_id = full_model.frames[full_tip_id].parentJoint  # Get parent joint of tip frame

            # 2. Identify Chain Joints (Tip -> Base)
            # Kinematic chain: sequence of joints connecting base to tip
            # Example: base -> joint1 -> joint2 -> ... -> jointN -> tip
            # We traverse from tip upward to base to find all joints in the chain

            base_joint_id = 0  # Default to Universe (joint 0 is universe/world joint, root of tree)
            if base_link != "universe" and base_link != "world" and full_model.existFrame(base_link):  # Check if base link is specified and exists
                base_joint_id = full_model.frames[full_model.getFrameId(base_link)].parentJoint  # Get parent joint of base frame

            chain_joints = []  # List to store joint IDs in the kinematic chain
            current_joint = tip_joint_id  # Start from tip joint

            # Trace up the kinematic tree
            # Pinocchio's parent array: parents[i] = parent joint ID of joint i
            # We traverse from tip to base by following parent pointers
            while current_joint > base_joint_id:  # Traverse from tip to base (stop when reaching base or universe)
                chain_joints.append(current_joint)  # Add current joint to chain
                current_joint = full_model.parents[current_joint]  # Move to parent joint (go up the tree)

            # 3. Identify joints to LOCK (all joints NOT in chain)
            # Why lock joints? If robot has multiple arms or redundant DOF,
            # we only want to control the kinematic chain from base to tip.
            # Other joints should be fixed at neutral position.

            joints_to_lock = []  # List of joint indices to lock (not in kinematic chain)
            is_chain_joint = [False] * (full_model.njoints)  # Boolean array marking chain joints
            for j in chain_joints:  # Mark all chain joints as True
                is_chain_joint[j] = True

            for i in range(1, full_model.njoints):  # Iterate through all joints (skip universe joint 0)
                if not is_chain_joint[i]:  # If joint is not in our kinematic chain
                    joints_to_lock.append(i)  # Add to lock list (will be fixed at neutral position)

            q_ref = pin.neutral(full_model)  # Get neutral configuration (all joints at zero/default position)

            # 4. Build Reduced Model
            # Reduced model: only contains joints in kinematic chain, other joints are locked
            # Benefits: smaller Jacobian matrices, faster computation, fewer DOF to optimize
            # Example: Full robot has 20 joints, but arm chain only has 7 joints
            #          Reduced model has 7 DOF, locked joints are fixed at q_ref values
            if joints_to_lock:  # If there are joints to lock
                self.model = pin.buildReducedModel(full_model, joints_to_lock, q_ref)  # Build reduced model with locked joints
            else:  # If no joints to lock (entire robot is the chain)
                self.model = full_model  # Use full model as-is

            self.data = self.model.createData()  # Create data structure for the model
            self.q_neutral = pin.neutral(self.model)  # Get neutral configuration of reduced model

            # Update tip frame ID in Reduced Model
            if self.model.existFrame(tip_link):  # Check if tip link exists in reduced model
                self.tip_frame_id = self.model.getFrameId(tip_link)  # Get tip frame ID in reduced model
            else:  # If tip link lost during reduction
                print("[PinocchioSolver] Tip link lost in reduced model.", file=sys.stderr)
                return False

            # Compute oMbase in Reduced Model
            # oMbase: transform from base frame to origin/world frame
            # oMbase_inv: inverse transform (from origin to base frame)
            # Why needed? User provides poses relative to base, but Pinocchio works in world frame
            # We store inverse to efficiently convert: baseMtip = oMbase_inv * oMtip

            pin.forwardKinematics(self.model, self.data, self.q_neutral)  # Compute forward kinematics with neutral config
            pin.updateFramePlacements(self.model, self.data)  # Update frame placements (fills data.oMf array)

            if self.model.existFrame(base_link):  # If base link exists in reduced model
                base_frame_id = self.model.getFrameId(base_link)  # Get base frame ID
                # data.oMf[frame_id] explanation:
                # - oMf = "origin to Frame" transforms
                # - oMf[frame_id] = oMframe = transform from frame to origin/world frame
                # - After updateFramePlacements(), oMf contains all frame poses in world frame
                # - oMbase = self.data.oMf[base_frame_id] = base pose in world frame
                # - oMbase_inv = inverse of oMbase = transform from world to base frame
                self.oMbase_inv = self.data.oMf[base_frame_id].inverse()  # Compute inverse transform from origin to base
            else:  # If base link doesn't exist (base is universe/world)
                self.oMbase_inv = pin.SE3.Identity()  # Use identity transform (base = origin)

            return True  # Initialization successful

        except Exception as e:  # Catch any exceptions during initialization
            print(f"[PinocchioSolver] Exception: {e}", file=sys.stderr)
            return False  # Return failure

    def solve_fk(self, q: np.ndarray) -> tuple[bool, pin.SE3]:
        """
        Compute Forward Kinematics (FK).

        Forward Kinematics: Given joint angles, compute end-effector pose.
        Process:
        1. Compute forward kinematics to update all joint positions
        2. Update frame placements to get tip frame pose in origin frame
        3. Transform from origin frame to base frame

        Args:
            q: Joint configuration (array of joint angles/positions).

        Returns:
            (success, pose): boolean success flag and pinocchio.SE3 pose of tip w.r.t base.
                            pose is a 4x4 homogeneous transformation matrix.
        """
        if self.model is None:  # Check if model is initialized
            return False, pin.SE3.Identity()  # Return failure with identity pose

        if q.size != self.model.nq:  # Check if joint configuration size matches model
            print(f"[PinocchioSolver] Joint size mismatch: {q.size} vs {self.model.nq}", file=sys.stderr)
            return False, pin.SE3.Identity()  # Return failure with identity pose

        pin.forwardKinematics(self.model, self.data, q)  # Compute forward kinematics (updates joint positions)
        pin.updateFramePlacements(self.model, self.data)  # Update frame placements (computes all frame poses)

        # Get tip pose in origin/world frame
        # oMtip = oMf[tip_frame_id] = transform from tip frame to origin frame
        # This is the tip's pose as computed by Pinocchio (always in world frame)
        oMtip = self.data.oMf[self.tip_frame_id]  # Get tip pose in origin/world frame (oMtip)

        # Transform from origin frame to base frame
        # Coordinate transformation chain:
        #   baseMtip = oMbase_inv * oMtip
        # Where:
        #   - oMtip: tip pose in world frame (from Pinocchio)
        #   - oMbase_inv: transform from world to base frame (stored during init)
        #   - baseMtip: tip pose in base frame (what user wants)
        #
        # Why this conversion?
        # - Pinocchio computes everything in world/origin frame
        # - User provides/expects poses relative to base frame
        # - We convert: world frame -> base frame
        baseMtip = self.oMbase_inv * oMtip  # Transform tip pose to base frame

        return True, baseMtip  # Return success with pose

    def solve_ik(self, target_pose: pin.SE3, q_init: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Compute Inverse Kinematics (IK) using CLIK (Closed-Loop Inverse Kinematics).

        Inverse Kinematics: Given desired end-effector pose, find joint angles.

        Algorithm (Newton-Raphson with CLIK):
        1. Compute current pose using forward kinematics
        2. Calculate error: log(current^-1 * target) -> 6D error vector
        3. Compute Jacobian (maps joint velocity to end-effector velocity)
        4. Apply Jlog6 correction for proper SE3 error propagation
        5. Solve for joint velocity: v = pinv(J_effective) * error
        6. Update joint angles: q_new = integrate(q_old, v * dt)
        7. Repeat until error < tolerance or max iterations

        Damped Pseudo-Inverse:
        - Regularizes singular configurations by adding damping: (J*J^T + damp*I)^-1
        - Prevents numerical instability when robot is near singularities

        Args:
            target_pose: Desired pose of tip w.r.t base (SE3 transform).
            q_init: Initial joint configuration (starting guess for optimization).

        Returns:
            (success, q_out): boolean success flag and resulting joint configuration.
                              success=True if converged within tolerance, False otherwise.
        """
        if self.model is None:  # Check if model is initialized
            return False, q_init  # Return failure with initial configuration

        if q_init.size != self.model.nq:  # Check if initial joint configuration size matches model
            print(f"[PinocchioSolver] Joint size mismatch: {q_init.size} vs {self.model.nq}", file=sys.stderr)
            return False, q_init  # Return failure with initial configuration

        q = q_init.copy()  # Copy initial configuration to working variable
        eps = 1e-4  # Convergence tolerance
        max_iter = 1000  # Maximum number of iterations
        dt = 0.1  # Integration time step
        damp = 1e-12  # Damping factor for pseudo-inverse

        # Transform target pose to world frame for solver
        # Input: target_pose = baseMtarget (user provides target pose relative to base frame)
        # Output: oMtarget (target pose in world frame, needed for Pinocchio computation)
        #
        # Transform chain:
        #   oMtarget = oMbase * baseMtarget
        # Where:
        #   - baseMtarget: target pose in base frame (user input)
        #   - oMbase: transform from base to world frame
        #   - oMtarget: target pose in world frame (for Pinocchio)
        #
        # Since we stored oMbase_inv during init, we get:
        #   oMbase = oMbase_inv.inverse()
        #   Therefore: oMtarget = oMbase_inv.inverse() * baseMtarget
        oMtarget = self.oMbase_inv.inverse() * target_pose  # Transform target pose from base frame to origin frame

        for i in range(max_iter):  # Iterate until convergence or max iterations
            pin.forwardKinematics(self.model, self.data, q)  # Compute forward kinematics with current configuration
            pin.updateFramePlacements(self.model, self.data)  # Update frame placements

            # Get current tip pose in world frame
            # current_pose = oMtip_current = current tip pose in origin/world frame
            # This is computed from current joint angles q using forward kinematics
            current_pose = self.data.oMf[self.tip_frame_id]  # Get current tip pose in origin frame

            # Error in Local Frame
            # Compute error between current pose and target pose:
            #   error_transform = current_pose.inverse() * oMtarget
            # This gives: "transform from current tip to target tip"
            #
            # SE3 log map converts transform difference to 6D vector:
            #   err = log(error_transform).vector
            # Result: 6D error vector [translation_error(3) + rotation_error(3)]
            # This error represents how much we need to move/rotate to reach target
            err = pin.log(current_pose.inverse() * oMtarget).vector  # Compute SE3 log error (6D vector)

            if np.linalg.norm(err) < eps:  # Check if error is below tolerance
                return True, q  # Return success with converged configuration

            # Compute Jacobian in Local Frame
            J = pin.computeFrameJacobian(self.model, self.data, q, self.tip_frame_id, pin.ReferenceFrame.LOCAL)  # Compute frame Jacobian in local frame

            # Jacobian correction for SE3 log mismatch
            # Why Jlog6 is needed:
            # - Standard Jacobian J maps: dq/dt -> d(pose)/dt (6D velocity in SE3)
            # - But our error is in log space: err = log(current^-1 * target)
            # - We need: d(log(pose))/dt, not d(pose)/dt
            # - Jlog6 corrects this: d(log(M))/dt = Jlog6 * dM/dt
            # - Therefore: J_effective = Jlog6 * J maps dq/dt -> d(log(pose))/dt
            # - This ensures Newton-Raphson works correctly in SE3 log space

            # Compute Jlog6: Jacobian of log map at the current error transform
            Jlog = pin.Jlog6(current_pose.inverse() * oMtarget)  # Compute Jlog6 for SE3 log mapping

            # Effective Jacobian: combines standard Jacobian with log space correction
            # This allows us to directly relate joint velocity to log-space error reduction
            J_effective = Jlog @ J  # Compute effective Jacobian with SE3 log correction

            # Damped Pseudo-Inverse (Levenberg-Marquardt style regularization)
            # Standard pseudo-inverse: v = J^T * (J*J^T)^-1 * err
            # Damped version: v = J^T * (J*J^T + damp*I)^-1 * err
            # Why damping?
            # - When J*J^T is near-singular (robot in singular configuration), inverse becomes unstable
            # - Adding damp*I to diagonal makes matrix invertible and stable
            # - Trade-off: slightly slower convergence but more robust

            JJt = J_effective @ J_effective.T  # Compute J * J^T (6x6 matrix)
            JJt[np.diag_indices_from(JJt)] += damp  # Add damping to diagonal (regularization)

            # Solve for joint velocity: v = J^T * (J*J^T + damp*I)^-1 * err
            # This gives the joint velocity that best reduces the error
            v = J_effective.T @ np.linalg.solve(JJt, err)  # Compute joint velocity using damped pseudo-inverse

            # Integrate velocity to update joint configuration
            # pin.integrate() handles different joint types correctly:
            # - Revolute joints: q_new = q_old + v*dt (simple addition)
            # - Prismatic joints: q_new = q_old + v*dt (simple addition)
            # - Spherical joints: uses exponential map for SO(3)
            # - Free-flyer: uses exponential map for SE(3)
            # dt is integration step size (smaller = more stable but slower convergence)
            q = pin.integrate(self.model, q, v * dt)  # Integrate velocity to update joint configuration

        return False, q  # Return failure if not converged within max iterations
