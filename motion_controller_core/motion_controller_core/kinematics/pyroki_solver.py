"""
Pyroki Library Overview:
=========================

Pyroki is a JAX-based robot kinematics library that uses optimization-based inverse kinematics.
Unlike Pinocchio which uses iterative Newton-Raphson, Pyroki formulates IK as a least-squares
optimization problem and solves it using JAX's automatic differentiation and JIT compilation.

Key Concepts:
-------------

1. JAX and JIT Compilation:
   - JAX: NumPy-like library with automatic differentiation and JIT compilation
   - JIT (Just-In-Time): Compiles Python code to optimized machine code for speed
   - JAX arrays: Similar to NumPy but can be compiled and differentiated
   - JIT functions run much faster but have restrictions (no dynamic Python features)

2. Optimization-Based IK:
   - Formulates IK as: minimize cost function subject to constraints
   - Cost function: weighted sum of pose error, rest pose penalty, joint limit penalty
   - Solver: Least-squares solver (jaxls) with trust region method
   - Advantages: Can handle multiple targets, constraints, and is differentiable

3. Pyroki Robot Model:
   - pk.Robot: Robot model loaded from URDF
   - robot.links: Link information (names, indices)
   - robot.joints: Joint information (actuated joints, limits)
   - robot.forward_kinematics(q): Computes all link poses from joint angles

4. Cost Functions:
   - pose_cost_analytic_jac: Penalizes distance between current and target pose
     * pos_weight: Weight for position error
     * ori_weight: Weight for orientation error
   - rest_cost: Penalizes deviation from rest/initial configuration
   - limit_constraint: Penalizes joints approaching limits

5. Multiple Targets:
   - Can solve IK for multiple end-effectors simultaneously
   - Example: Bimanual robot (two arms) reaching two different targets
   - All targets are optimized together in one problem

6. SE3 Representation:
   - jaxlie.SE3: Represents 6D pose (rotation + translation)
   - from_matrix(): Convert 4x4 homogeneous matrix to SE3
   - rotation().wxyz: Get quaternion (w, x, y, z) representation
   - translation(): Get 3D position vector
   - as_matrix(): Convert SE3 back to 4x4 matrix

7. Least Squares Problem:
   - jaxls.LeastSquaresProblem: Formulates optimization problem
   - variables: Joint variables to optimize (JointVar)
   - costs: List of cost functions to minimize
   - solve(): Solves using trust region method with Cholesky decomposition

8. Joint Variables:
   - JointVar(id): Represents a set of joint variables with ID
   - JointVar(0): Default joint variable (all actuated joints)
   - Used to link cost functions to the same set of joints

Differences from Pinocchio:
---------------------------
- Pinocchio: Iterative Newton-Raphson, single target, fast for single queries
- Pyroki: Optimization-based, multiple targets, differentiable, JIT-compiled
- Pinocchio: C++ backend, Python bindings
- Pyroki: Pure JAX/Python, can run on GPU/TPU
"""

import numpy as np  # NumPy for standard arrays
import sys  # System-specific parameters
import jax  # JAX library for automatic differentiation and JIT
import jax.numpy as jnp  # JAX NumPy (compatible with JIT)
import jax_dataclasses as jdc  # JAX-compatible dataclasses
import jaxlie  # JAX Lie groups (SO3, SE3)
import jaxls  # JAX Least Squares solver
import pyroki as pk  # Pyroki robot kinematics library
import yourdfpy  # URDF parser
import tempfile  # Temporary file handling
import os  # Operating system interface

class PyrokiSolver:
    def __init__(self):
        self.robot = None  # Pyroki robot model (pk.Robot)
        self.target_link_indices = None  # List of link indices for IK targets (end-effectors)
        self.jitted_solve = None  # JIT-compiled IK solver function (for performance)

    def init(self, urdf_content: str, base_link: str, tip_links: list[str] | str) -> bool:
        """
        Initialize the solver with URDF content.

        This method:
        1. Parses URDF using yourdfpy
        2. Creates Pyroki robot model
        3. Validates and stores target link indices
        4. JIT-compiles the IK solver for performance

        Args:
            urdf_content: XML string of the URDF (Unified Robot Description Format).
            base_link: Name of the base link (currently not used but kept for API compatibility).
            tip_links: Name(s) of the tip link(s) to control (can be single string or list).
                      Multiple links enable multi-target IK (e.g., bimanual robot).

        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            # Load URDF via temp file for yourdfpy compatibility
            # yourdfpy.URDF.load() requires a file path, not string content
            # We create a temporary file, write URDF content, then delete it
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as tmp:
                tmp.write(urdf_content)  # Write URDF XML content to temp file
                tmp_path = tmp.name  # Get temporary file path

            try:
                urdf_model = yourdfpy.URDF.load(tmp_path)  # Parse URDF file into yourdfpy model
            finally:
                os.remove(tmp_path)  # Clean up: delete temp file

            # Create Pyroki robot model from URDF
            # pk.Robot contains all kinematic information (links, joints, forward kinematics)
            self.robot = pk.Robot.from_urdf(urdf_model)

            # Normalize tip_links to list (handle both single string and list)
            if isinstance(tip_links, str):
                tip_links = [tip_links]  # Convert single string to list

            # Validate and store target link indices
            # Link indices are used internally by Pyroki (faster than name lookups)
            self.target_link_indices = []
            for link in tip_links:  # Iterate through all target links
                if link not in self.robot.links.names:  # Check if link exists in robot model
                    print(f"[PyrokiSolver] Tip link {link} not found.", file=sys.stderr)
                    return False
                # Convert link name to index (used for efficient access)
                self.target_link_indices.append(self.robot.links.names.index(link))

            # JIT-compile the IK solver function
            # This compiles the optimization problem once for better performance
            # JIT compilation happens at init time, not at solve time
            self._jit_solver()
            return True  # Initialization successful

        except Exception as e:
            print(f"[PyrokiSolver] Init Exception: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False

    def _jit_solver(self):
        """
        Pre-compile the JAX IK solver function using JIT.

        This method creates and JIT-compiles the IK solver function.
        JIT compilation converts Python/JAX code to optimized machine code,
        making subsequent IK solves much faster (10-100x speedup).

        The solver uses optimization-based IK:
        1. Formulates IK as least-squares problem
        2. Cost function: pose error + rest pose penalty + joint limit penalty
        3. Solves using trust region method with Cholesky decomposition
        4. Supports multiple targets (bimanual, multi-end-effector)

        Why JIT at init time?
        - JIT compilation is expensive (takes time)
        - We do it once at init, not every solve_ik() call
        - Compiled function is cached and reused
        """

        @jdc.jit  # JIT decorator: compile this function for performance
        def _solve_ik_jax(
            robot: pk.Robot,  # Pyroki robot model
            target_wxyzs: jax.Array,  # Target orientations as quaternions (N, 4) where N = number of targets
            target_positions: jax.Array,  # Target positions (N, 3)
            target_link_indices: jax.Array,  # Link indices for each target (N,)
            q_init: jax.Array  # Initial joint configuration (DoF,)
        ) -> jax.Array:  # Returns optimized joint configuration (DoF,)
            """
            JIT-compiled IK solver using optimization.

            This function solves IK by minimizing a cost function:
            - Pose cost: distance between current and target poses
            - Rest cost: deviation from initial/rest configuration
            - Limit cost: penalty for joints near limits

            Supports multiple targets: all targets are optimized simultaneously.
            For bimanual robots, this finds one configuration that satisfies all targets.
            """

            # Get joint variable class for this robot
            # JointVar is used to represent optimization variables in jaxls
            JointVar = robot.joint_var_cls

            # Construct target poses from quaternions and positions
            # target_wxyzs: (N, 4) - quaternions [w, x, y, z] for N targets
            # target_positions: (N, 3) - 3D positions for N targets
            # jaxlie.SE3.from_rotation_and_translation() creates SE3 transforms
            # Result: target_pose with shape () for single target, (N,) for N targets
            target_pose = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyzs),  # Convert quaternion to SO3 rotation
                target_positions  # Translation vector
            )

            # Get batch axes for broadcasting
            # If we have N targets, target_pose has batch dimension (N,)
            # batch_axes tells us the shape of the batch: () for single, (N,) for multiple
            # This is used to properly broadcast robot model and variables
            batch_axes = target_pose.get_batch_axes()

            # Build cost function list
            # The optimization minimizes the sum of all costs
            costs = []

            # Rest cost: Penalizes deviation from initial/rest configuration
            # This encourages solutions close to q_init (helps with continuity and avoids large jumps)
            # weight=1.0: How much to penalize deviation (lower = more flexible)
            costs.append(pk.costs.rest_cost(
                JointVar(0),  # Joint variable ID 0 (all actuated joints)
                rest_pose=q_init,  # Rest configuration = initial guess
                weight=1.0
            ))

            # Limit constraint: Penalizes joints approaching their limits
            # Prevents solutions that violate joint limits (soft constraint)
            # This is important for real robots with physical limits
            costs.append(pk.costs.limit_constraint(
                robot,  # Robot model (contains joint limits)
                JointVar(0)  # Joint variable to constrain
            ))

            # Pose cost: Main IK objective - minimize distance to target pose(s)
            # This is the primary cost that drives the solution toward target
            #
            # Broadcasting explanation for multiple targets:
            # - jax.tree.map(lambda x: x[None], robot): Adds batch dimension to robot (1, ...)
            #   This allows broadcasting with batched target_pose (N, ...)
            # - JointVar(jnp.full(batch_axes, 0)): Creates variable with batch shape
            #   All batch elements use the same variable ID 0 (same joint configuration)
            # - target_pose: Batched SE3 transforms (N,) for N targets
            # - target_link_indices: Which link each target applies to (N,)
            #
            # Result: One joint configuration (JointVar(0)) that minimizes error for ALL targets
            # For bimanual robots: finds one config where both hands reach their targets
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    jax.tree.map(lambda x: x[None], robot),  # Broadcast robot to (1, ...) for compatibility
                    JointVar(jnp.full(batch_axes, 0)),  # Use same variable ID 0 for all batch elements
                    target_pose,  # Batched target poses (N,)
                    target_link_indices,  # Link index for each target (N,)
                    pos_weight=50.0,  # Weight for position error (higher = more important)
                    ori_weight=10.0,  # Weight for orientation error
                )
            )

            # Create least-squares optimization problem
            # This formulates IK as: minimize sum of squared residuals (costs)
            # variables=[JointVar(0)]: We optimize one set of joint angles
            problem = jaxls.LeastSquaresProblem(
                costs=costs,  # List of cost functions to minimize
                variables=[JointVar(0)]  # Joint variables to optimize (all actuated joints)
            )

            # Solve the optimization problem
            # analyze(): Analyzes problem structure (computes Jacobians, Hessians)
            # solve(): Solves using iterative trust region method
            sol = problem.analyze().solve(
                verbose=False,  # Don't print solver progress
                linear_solver="dense_cholesky",  # Use Cholesky decomposition for linear system
                trust_region=jaxls.TrustRegionConfig(
                    lambda_initial=10.0  # Initial trust region size (larger = more conservative)
                ),
            )

            # Extract solution: optimized joint configuration
            # sol is a dictionary mapping variables to their optimized values
            # sol[JointVar(0)] gives the optimized joint angles
            return sol[JointVar(0)]

        self.jitted_solve = _solve_ik_jax

    def solve_ik(self, target_poses: list[np.ndarray] | np.ndarray, q_init: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Solve Inverse Kinematics (IK) for one or multiple targets.

        This method uses optimization-based IK to find joint angles that place
        the target link(s) at the desired pose(s). Supports multiple end-effectors
        (e.g., bimanual robots with two arms).

        Process:
        1. Convert target poses (4x4 matrices) to SE3 format
        2. Extract quaternions and positions
        3. Call JIT-compiled solver to optimize joint angles
        4. Return optimized configuration

        Args:
            target_poses: List of 4x4 homogeneous transformation matrices, or single 4x4 matrix.
                         Each matrix represents desired pose of one target link.
                         Number of poses must match number of tip_links from init().
            q_init: Initial joint configuration (starting guess for optimization).
                   Should be close to desired solution for better convergence.

        Returns:
            (success, q_out): Tuple of:
                - success: True if IK solved successfully, False otherwise
                - q_out: Optimized joint configuration (numpy array)
        """
        # Check if solver is initialized
        if self.robot is None or self.jitted_solve is None:
            return False, q_init  # Return failure with initial config

        try:
            # Normalize target_poses to list format
            # Handle both single 4x4 matrix and list of matrices
            if isinstance(target_poses, np.ndarray) and target_poses.shape == (4, 4):
                target_poses = [target_poses]  # Convert single matrix to list

            # Validate number of targets matches number of tip links
            num_targets = len(target_poses)
            if num_targets != len(self.target_link_indices):
                print(f"Mismatch targets {num_targets} vs links {len(self.target_link_indices)}", file=sys.stderr)
                return False, q_init  # Return failure if mismatch

            # Extract quaternions and positions from 4x4 transformation matrices
            # Each target pose is converted to SE3 format, then decomposed
            wxyzs = []  # List of quaternions [w, x, y, z] for each target
            positions = []  # List of 3D positions for each target
            for T in target_poses:  # Iterate through each target pose
                T_se3 = jaxlie.SE3.from_matrix(T)  # Convert 4x4 matrix to SE3
                wxyzs.append(T_se3.rotation().wxyz)  # Extract quaternion (w, x, y, z)
                positions.append(T_se3.translation())  # Extract 3D position

            # Convert to JAX arrays for JIT-compiled function
            # These arrays will be passed to the compiled solver
            target_wxyzs = jnp.array(wxyzs)  # Shape (N, 4) - N quaternions
            target_positions = jnp.array(positions)  # Shape (N, 3) - N positions
            target_indices = jnp.array(self.target_link_indices)  # Shape (N,) - link indices

            # Call JIT-compiled IK solver
            # This runs the optimized machine code (fast!)
            # Solver minimizes cost function to find optimal joint angles
            q_out = self.jitted_solve(
                self.robot,  # Robot model
                target_wxyzs,  # Target orientations (quaternions)
                target_positions,  # Target positions
                target_indices,  # Which links to control
                jnp.array(q_init)  # Initial joint configuration
            )

            # Convert result back to NumPy and ensure 1D shape
            # JAX arrays might have extra dimensions, flatten if needed
            q_out = np.array(q_out)  # Convert JAX array to NumPy
            if q_out.ndim > 1:  # If multi-dimensional
                q_out = q_out.flatten()  # Flatten to 1D array

            return True, q_out  # Return success with optimized configuration


        except Exception as e:
            print(f"[PyrokiSolver] IK Failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False, q_init

    def solve_fk(self, q: np.ndarray) -> tuple[bool, list[np.ndarray]]:
        """
        Solve Forward Kinematics (FK) for initialized tip links.

        Forward Kinematics: Given joint angles, compute end-effector poses.
        This method computes poses for all target links specified during init().

        Process:
        1. Compute forward kinematics for all links in robot
        2. Extract poses for target links only
        3. Convert to 4x4 homogeneous transformation matrices

        Args:
            q: Joint configuration (array of joint angles/positions).

        Returns:
            (success, poses): Tuple of:
                - success: True if FK computed successfully, False otherwise
                - poses: List of 4x4 homogeneous transformation matrices
                        One matrix per target link (in same order as tip_links from init())
        """
        # Check if robot is initialized
        if self.robot is None:
            return False, []  # Return failure with empty list

        try:
            # Compute forward kinematics for all links
            # robot.forward_kinematics() returns poses for all links
            # Each pose is in 7D format: [x, y, z, qw, qx, qy, qz] (position + quaternion)
            link_poses = self.robot.forward_kinematics(jnp.array(q))  # Convert to JAX array

            # Extract poses for target links only
            results = []
            for idx in self.target_link_indices:  # Iterate through target link indices
                pose_7 = link_poses[idx]  # Get 7D pose [x, y, z, qw, qx, qy, qz]
                # Convert 7D pose to SE3, then to 4x4 matrix
                # jaxlie.SE3(pose_7) creates SE3 from 7D format
                # .as_matrix() converts SE3 to 4x4 homogeneous transformation matrix
                results.append(np.array(jaxlie.SE3(pose_7).as_matrix()))

            return True, results  # Return success with list of 4x4 matrices

        except Exception as e:  # Catch any errors during computation
            print(f"[PyrokiSolver] FK Failed: {e}", file=sys.stderr)
            return False, []  # Return failure with empty list
