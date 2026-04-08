"""
Day 008: Forward and Inverse Kinematics for 2D Robotic Arms

A complete kinematics engine: given joint angles compute end-effector position
(forward kinematics), and given a target position compute joint angles that
reach it (inverse kinematics) — both analytically (2-link) and numerically
(n-link via Jacobian pseudo-inverse).

Usage: python3 solution.py
"""

import math
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RobotArm:
    """A planar (2D) robotic arm defined by link lengths."""

    def __init__(self, link_lengths: List[float]):
        """
        Args:
            link_lengths: Length of each link, from base outward.
                          e.g. [1.0, 0.8, 0.5] is a 3-link arm.
        """
        if not link_lengths or any(l <= 0 for l in link_lengths):
            raise ValueError("All link lengths must be positive")
        self.link_lengths = list(link_lengths)
        self.n_joints = len(link_lengths)

    @property
    def max_reach(self) -> float:
        """Maximum distance the end-effector can reach (all links extended)."""
        return sum(self.link_lengths)

    @property
    def min_reach(self) -> float:
        """Minimum distance from base the end-effector must be at.

        If the longest link exceeds the sum of all others, there's a hole
        in the center of the workspace. Otherwise the arm can fold to reach
        the origin (min_reach = 0 for 2+ links).
        """
        longest = max(self.link_lengths)
        rest = sum(self.link_lengths) - longest
        # If longest > rest, the arm can't fold back past (longest - rest)
        return max(0.0, longest - rest)


# ---------------------------------------------------------------------------
# Forward Kinematics
# ---------------------------------------------------------------------------

def forward_kinematics(
    arm: RobotArm,
    joint_angles: List[float]
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """Compute all joint positions and end-effector position.

    Each joint angle is relative to the previous link (local frame).
    The absolute angle of link i in the world frame is the cumulative
    sum theta_1 + theta_2 + ... + theta_i.

    Args:
        arm: The robot arm definition.
        joint_angles: Angle (radians) for each joint. Length must match n_joints.

    Returns:
        (joint_positions, end_effector):
            joint_positions: list of (x, y) for each joint including the base at (0,0)
            end_effector: (x, y) of the tip
    """
    if len(joint_angles) != arm.n_joints:
        raise ValueError(f"Expected {arm.n_joints} angles, got {len(joint_angles)}")

    # Start at the base (origin)
    x, y = 0.0, 0.0
    cumulative_angle = 0.0
    positions = [(x, y)]  # base position

    for i in range(arm.n_joints):
        # Accumulate angle: each joint rotates relative to the previous link
        cumulative_angle += joint_angles[i]
        # Translate along the link in the direction of the cumulative angle
        x += arm.link_lengths[i] * math.cos(cumulative_angle)
        y += arm.link_lengths[i] * math.sin(cumulative_angle)
        positions.append((x, y))

    # The last position is the end-effector
    end_effector = positions[-1]
    return positions, end_effector


# ---------------------------------------------------------------------------
# Analytical Inverse Kinematics (2-link only)
# ---------------------------------------------------------------------------

def inverse_kinematics_2link(
    arm: RobotArm,
    target: Tuple[float, float],
    elbow_up: bool = True
) -> Optional[Tuple[float, float]]:
    """Closed-form IK for a 2-link planar arm using the law of cosines.

    Args:
        arm: Must have exactly 2 links.
        target: Desired (x, y) position for end-effector.
        elbow_up: If True, return the elbow-up solution; else elbow-down.

    Returns:
        (theta1, theta2) in radians, or None if the target is unreachable.
    """
    if arm.n_joints != 2:
        raise ValueError("Analytical IK requires exactly 2 links")

    L1, L2 = arm.link_lengths
    tx, ty = target
    dist = math.sqrt(tx * tx + ty * ty)

    # Check reachability
    if dist > L1 + L2 + 1e-9:
        return None  # Too far
    if dist < abs(L1 - L2) - 1e-9:
        return None  # Too close (inside the hole)

    # Clamp for numerical safety at the boundary
    cos_theta2 = (tx * tx + ty * ty - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    # Two solutions: elbow up (+) or elbow down (-)
    if elbow_up:
        theta2 = math.atan2(math.sqrt(1 - cos_theta2 * cos_theta2), cos_theta2)
    else:
        theta2 = math.atan2(-math.sqrt(1 - cos_theta2 * cos_theta2), cos_theta2)

    # Theta1: angle to target minus the offset caused by link 2
    theta1 = math.atan2(ty, tx) - math.atan2(
        L2 * math.sin(theta2),
        L1 + L2 * math.cos(theta2)
    )

    return (theta1, theta2)


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def compute_jacobian(
    arm: RobotArm,
    joint_angles: List[float]
) -> List[List[float]]:
    """Compute the 2 x n Jacobian matrix for a planar arm.

    J[0][i] = partial(x)/partial(theta_i)
    J[1][i] = partial(y)/partial(theta_i)

    For a 2D arm, moving joint i affects the end-effector via all links
    from i onward. The partial derivatives are:

        dx/d(theta_i) = -sum_{k=i}^{n} L_k * sin(cumulative_angle_k)
        dy/d(theta_i) =  sum_{k=i}^{n} L_k * cos(cumulative_angle_k)

    where cumulative_angle_k = theta_1 + theta_2 + ... + theta_k.

    Args:
        arm: The robot arm.
        joint_angles: Current joint angles (radians).

    Returns:
        2 x n Jacobian matrix as list of lists.
    """
    n = arm.n_joints
    # Precompute cumulative angles
    cum_angles = []
    total = 0.0
    for a in joint_angles:
        total += a
        cum_angles.append(total)

    jacobian = [[0.0] * n, [0.0] * n]

    for i in range(n):
        # Joint i affects links i through n-1
        dx = 0.0
        dy = 0.0
        for k in range(i, n):
            dx -= arm.link_lengths[k] * math.sin(cum_angles[k])
            dy += arm.link_lengths[k] * math.cos(cum_angles[k])
        jacobian[0][i] = dx
        jacobian[1][i] = dy

    return jacobian


# ---------------------------------------------------------------------------
# Numerical Inverse Kinematics (n-link via Jacobian pseudo-inverse)
# ---------------------------------------------------------------------------

def _mat_mult_2xn_nx1(mat: List[List[float]], vec: List[float]) -> List[float]:
    """Multiply a 2xn matrix by an nx1 vector."""
    return [
        sum(mat[0][j] * vec[j] for j in range(len(vec))),
        sum(mat[1][j] * vec[j] for j in range(len(vec))),
    ]


def _transpose(mat: List[List[float]]) -> List[List[float]]:
    """Transpose a 2xn matrix to nx2."""
    n = len(mat[0])
    return [[mat[r][c] for r in range(2)] for c in range(n)]


def _pseudoinverse_2xn(J: List[List[float]], damping: float = 1e-4) -> List[List[float]]:
    """Compute the damped pseudo-inverse of a 2xn Jacobian.

    J+ = J^T (J J^T + lambda^2 I)^(-1)

    Damping prevents blow-up near singularities (fully extended/folded arm).
    Without damping, the pseudo-inverse gives infinite joint velocities
    at singular configurations.

    Returns n x 2 matrix.
    """
    n = len(J[0])
    JT = _transpose(J)

    # Compute JJT (2x2 matrix)
    # JJT[r][c] = sum_k J[r][k] * J[c][k]
    JJT = [[0.0, 0.0], [0.0, 0.0]]
    for r in range(2):
        for c in range(2):
            JJT[r][c] = sum(J[r][k] * J[c][k] for k in range(n))

    # Add damping: JJT + lambda^2 * I
    JJT[0][0] += damping * damping
    JJT[1][1] += damping * damping

    # Invert the 2x2 matrix analytically
    det = JJT[0][0] * JJT[1][1] - JJT[0][1] * JJT[1][0]
    if abs(det) < 1e-15:
        # Degenerate — return zero matrix
        return [[0.0, 0.0] for _ in range(n)]
    inv = [
        [ JJT[1][1] / det, -JJT[0][1] / det],
        [-JJT[1][0] / det,  JJT[0][0] / det],
    ]

    # J+ = JT * inv(JJT)   — (n x 2) = (n x 2) * (2 x 2)
    result = [[0.0, 0.0] for _ in range(n)]
    for i in range(n):
        for j in range(2):
            result[i][j] = JT[i][0] * inv[0][j] + JT[i][1] * inv[1][j]

    return result


def inverse_kinematics_numerical(
    arm: RobotArm,
    target: Tuple[float, float],
    initial_angles: Optional[List[float]] = None,
    max_iterations: int = 200,
    tolerance: float = 1e-4,
    damping: float = 1e-3,
    step_scale: float = 1.0
) -> Optional[Tuple[List[float], int, float]]:
    """Iterative IK using the damped Jacobian pseudo-inverse.

    Algorithm:
        1. Start from initial joint angles (or zeros).
        2. Compute current end-effector position via FK.
        3. Compute error = target - current.
        4. If ||error|| < tolerance, done.
        5. Compute Jacobian and its pseudo-inverse.
        6. d_theta = J_pseudoinverse * error * step_scale
        7. Update angles += d_theta. Go to step 2.

    Args:
        arm: The robot arm.
        target: Desired (x, y) position.
        initial_angles: Starting joint angles. Defaults to all zeros.
        max_iterations: Maximum iterations before giving up.
        tolerance: Position error threshold for convergence.
        damping: Damping factor for pseudo-inverse (prevents singularity blow-up).
        step_scale: Scales the step size (< 1.0 for more cautious steps).

    Returns:
        (joint_angles, iterations, final_error) or None if unreachable.
    """
    tx, ty = target
    dist = math.sqrt(tx * tx + ty * ty)

    # Quick reachability check
    if dist > arm.max_reach + tolerance:
        return None

    # Initialize angles — small random perturbation avoids starting at a singularity
    # (e.g., all zeros = fully extended, where the Jacobian is rank-deficient)
    if initial_angles:
        angles = list(initial_angles)
    else:
        angles = [0.1 * (i + 1) for i in range(arm.n_joints)]

    for iteration in range(max_iterations):
        # Forward kinematics to get current position
        _, (ex, ey) = forward_kinematics(arm, angles)

        # Error vector
        err_x = tx - ex
        err_y = ty - ey
        error = math.sqrt(err_x * err_x + err_y * err_y)

        if error < tolerance:
            return (angles, iteration + 1, error)

        # Compute Jacobian and pseudo-inverse
        J = compute_jacobian(arm, angles)
        J_pinv = _pseudoinverse_2xn(J, damping)

        # Compute angle updates: d_theta = J+ * error_vector
        for i in range(arm.n_joints):
            d_theta = (J_pinv[i][0] * err_x + J_pinv[i][1] * err_y) * step_scale
            angles[i] += d_theta

    # Check if we converged close enough
    _, (ex, ey) = forward_kinematics(arm, angles)
    final_error = math.sqrt((tx - ex) ** 2 + (ty - ey) ** 2)
    if final_error < tolerance * 10:  # Relaxed tolerance
        return (angles, max_iterations, final_error)

    return None  # Failed to converge — likely unreachable


# ---------------------------------------------------------------------------
# Workspace analysis
# ---------------------------------------------------------------------------

def compute_workspace(
    arm: RobotArm,
    samples_per_joint: int = 20
) -> List[Tuple[float, float]]:
    """Sample the reachable workspace by sweeping joint angles.

    For each combination of joint angles (uniformly sampled over [-pi, pi]),
    compute the end-effector position. The collection of all such points
    approximates the workspace boundary.

    Note: For arms with many joints, the combinatorial explosion makes
    exhaustive sampling impractical. We use random sampling instead for n > 3.

    Args:
        arm: The robot arm.
        samples_per_joint: Number of angle samples per joint.

    Returns:
        List of (x, y) end-effector positions.
    """
    import random

    points = []
    n = arm.n_joints

    if n <= 3:
        # Exhaustive grid sampling for small arms
        total = samples_per_joint ** n
        # Generate all combinations using iterative approach
        for idx in range(total):
            angles = []
            remainder = idx
            for _ in range(n):
                sample = remainder % samples_per_joint
                remainder //= samples_per_joint
                angle = -math.pi + 2 * math.pi * sample / samples_per_joint
                angles.append(angle)
            _, end = forward_kinematics(arm, angles)
            points.append(end)
    else:
        # Random sampling for high-DOF arms
        random.seed(42)
        for _ in range(samples_per_joint ** 3):
            angles = [random.uniform(-math.pi, math.pi) for _ in range(n)]
            _, end = forward_kinematics(arm, angles)
            points.append(end)

    return points


# ---------------------------------------------------------------------------
# Utility: normalize angle to [-pi, pi]
# ---------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 008: FORWARD AND INVERSE KINEMATICS")
    print("=" * 70)

    # --- 1. Forward Kinematics Demo ---
    print("\n--- Forward Kinematics ---\n")

    arm2 = RobotArm([1.0, 0.8])
    angles = [math.radians(45), math.radians(-30)]

    positions, end = forward_kinematics(arm2, angles)
    print(f"2-link arm: lengths = {arm2.link_lengths}")
    print(f"Joint angles: {[round(math.degrees(a), 1) for a in angles]} degrees")
    print(f"Joint positions:")
    for i, (x, y) in enumerate(positions):
        label = "base" if i == 0 else f"joint {i}" if i < len(positions) - 1 else "end-effector"
        print(f"  {label}: ({x:.4f}, {y:.4f})")
    print(f"End-effector: ({end[0]:.4f}, {end[1]:.4f})")

    # Verify: distance from base should be <= max_reach
    dist = math.sqrt(end[0] ** 2 + end[1] ** 2)
    print(f"Distance from base: {dist:.4f} (max reach: {arm2.max_reach})")

    # --- 2. Forward Kinematics for 3-link arm ---
    print("\n--- 3-link arm Forward Kinematics ---\n")

    arm3 = RobotArm([1.0, 0.7, 0.5])
    angles3 = [math.radians(30), math.radians(45), math.radians(-60)]
    positions3, end3 = forward_kinematics(arm3, angles3)
    print(f"3-link arm: lengths = {arm3.link_lengths}")
    print(f"Joint angles: {[round(math.degrees(a), 1) for a in angles3]} degrees")
    for i, (x, y) in enumerate(positions3):
        label = "base" if i == 0 else f"joint {i}" if i < len(positions3) - 1 else "end-effector"
        print(f"  {label}: ({x:.4f}, {y:.4f})")

    # --- 3. Analytical IK (2-link) ---
    print("\n--- Analytical Inverse Kinematics (2-link) ---\n")

    target = (1.2, 0.6)
    print(f"Target: {target}")
    print(f"Arm: lengths = {arm2.link_lengths}, max reach = {arm2.max_reach}")

    for elbow_label, elbow_up in [("elbow-up", True), ("elbow-down", False)]:
        result = inverse_kinematics_2link(arm2, target, elbow_up=elbow_up)
        if result:
            t1, t2 = result
            print(f"\n  {elbow_label} solution:")
            print(f"    theta1 = {math.degrees(t1):.2f} deg, theta2 = {math.degrees(t2):.2f} deg")
            # Verify by FK
            _, verify_end = forward_kinematics(arm2, [t1, t2])
            print(f"    FK verification: ({verify_end[0]:.4f}, {verify_end[1]:.4f})")
            err = math.sqrt((verify_end[0] - target[0]) ** 2 + (verify_end[1] - target[1]) ** 2)
            print(f"    Position error: {err:.2e}")

    # Unreachable target
    print(f"\n  Unreachable target (3.0, 0.0):")
    result = inverse_kinematics_2link(arm2, (3.0, 0.0))
    print(f"    Result: {result} (correctly None)")

    # --- 4. Numerical IK (3-link arm) ---
    print("\n--- Numerical Inverse Kinematics (Jacobian pseudo-inverse) ---\n")

    target3 = (1.5, 0.8)
    print(f"3-link arm: lengths = {arm3.link_lengths}")
    print(f"Target: {target3}")
    print(f"Max reach: {arm3.max_reach:.2f}")

    result = inverse_kinematics_numerical(arm3, target3)
    if result:
        angles_sol, iters, err = result
        print(f"\nSolution found in {iters} iterations:")
        print(f"  Joint angles: {[round(math.degrees(a), 2) for a in angles_sol]} deg")
        print(f"  Final error: {err:.2e}")
        # Verify
        _, verify = forward_kinematics(arm3, angles_sol)
        print(f"  FK verification: ({verify[0]:.4f}, {verify[1]:.4f})")

    # Try multiple targets
    print("\n  Testing various targets:")
    test_targets = [(1.0, 1.0), (0.5, 0.3), (2.0, 0.0), (-1.0, 0.5), (0.0, 2.1)]
    for t in test_targets:
        res = inverse_kinematics_numerical(arm3, t)
        if res:
            a, it, e = res
            _, v = forward_kinematics(arm3, a)
            status = f"OK (iters={it}, err={e:.2e})"
        else:
            status = "UNREACHABLE"
        print(f"    target={t} -> {status}")

    # --- 5. Jacobian Demo ---
    print("\n--- Jacobian Matrix ---\n")

    test_angles = [math.radians(30), math.radians(45), math.radians(-20)]
    J = compute_jacobian(arm3, test_angles)
    print(f"Arm: {arm3.link_lengths}, angles: {[round(math.degrees(a), 1) for a in test_angles]} deg")
    print(f"Jacobian (2x3):")
    print(f"  dx/d_theta: [{J[0][0]:.4f}, {J[0][1]:.4f}, {J[0][2]:.4f}]")
    print(f"  dy/d_theta: [{J[1][0]:.4f}, {J[1][1]:.4f}, {J[1][2]:.4f}]")
    print(f"\nInterpretation: J[0][0]={J[0][0]:.4f} means rotating joint 1 by 1 rad")
    print(f"  moves the end-effector {J[0][0]:.4f} units in x")

    # --- 6. Workspace Analysis ---
    print("\n--- Workspace Analysis ---\n")

    workspace = compute_workspace(arm2, samples_per_joint=36)
    xs = [p[0] for p in workspace]
    ys = [p[1] for p in workspace]
    print(f"2-link arm workspace ({len(workspace)} samples):")
    print(f"  x range: [{min(xs):.3f}, {max(xs):.3f}]")
    print(f"  y range: [{min(ys):.3f}, {max(ys):.3f}]")
    print(f"  Max distance: {max(math.sqrt(x*x+y*y) for x, y in workspace):.3f}")
    print(f"  Min distance: {min(math.sqrt(x*x+y*y) for x, y in workspace):.3f}")
    print(f"  Theoretical max reach: {arm2.max_reach:.3f}")
    print(f"  Theoretical min reach: {arm2.min_reach:.3f}")

    # --- 7. Round-trip verification ---
    print("\n--- Round-trip Verification: FK -> IK -> FK ---\n")

    original_angles = [math.radians(60), math.radians(-45)]
    _, target_pos = forward_kinematics(arm2, original_angles)
    print(f"Original angles: {[round(math.degrees(a), 1) for a in original_angles]} deg")
    print(f"FK gives target: ({target_pos[0]:.4f}, {target_pos[1]:.4f})")

    recovered = inverse_kinematics_2link(arm2, target_pos, elbow_up=True)
    if recovered:
        print(f"IK recovers: {[round(math.degrees(a), 1) for a in recovered]} deg")
        _, final_pos = forward_kinematics(arm2, list(recovered))
        print(f"FK verification: ({final_pos[0]:.4f}, {final_pos[1]:.4f})")
        error = math.sqrt((final_pos[0] - target_pos[0]) ** 2 + (final_pos[1] - target_pos[1]) ** 2)
        print(f"Round-trip error: {error:.2e}")

    print("\n" + "=" * 70)
    print("All kinematics demos completed successfully!")
    print("=" * 70)
