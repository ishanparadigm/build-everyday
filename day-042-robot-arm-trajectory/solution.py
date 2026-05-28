"""
Day 42: Robot Arm Trajectory Planning

A complete trajectory planner for a 2-link planar robot arm. Supports:
- Forward and inverse kinematics
- Trapezoidal velocity profile trajectories
- Cubic polynomial trajectories
- Multi-waypoint planning with velocity continuity
- Joint limit and velocity constraint validation
"""

import math
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# ARM MODEL
# ============================================================================

@dataclass
class JointLimits:
    """Min/max angle (radians) and max angular velocity (rad/s) for one joint."""
    min_angle: float
    max_angle: float
    max_velocity: float      # rad/s
    max_acceleration: float  # rad/s^2


@dataclass
class ArmConfig:
    """Configuration for a 2-link planar robot arm."""
    L1: float  # Length of link 1
    L2: float  # Length of link 2
    joint1_limits: JointLimits
    joint2_limits: JointLimits


@dataclass
class TrajectoryPoint:
    """A single point along a trajectory: position, velocity, acceleration for each joint."""
    time: float
    q1: float       # Joint 1 angle
    q2: float       # Joint 2 angle
    q1_dot: float   # Joint 1 velocity
    q2_dot: float   # Joint 2 velocity
    q1_ddot: float  # Joint 1 acceleration
    q2_ddot: float  # Joint 2 acceleration
    x: float        # End-effector x
    y: float        # End-effector y


# ============================================================================
# KINEMATICS
# ============================================================================

def forward_kinematics(arm: ArmConfig, q1: float, q2: float) -> tuple[float, float]:
    """
    Compute end-effector (x, y) from joint angles.

    The geometry is straightforward: the end of link 1 is at (L1*cos(q1), L1*sin(q1)),
    and the end of link 2 extends from there at angle (q1 + q2) from the horizontal.
    We use the absolute angle of link 2 because joint 2's angle is relative to link 1.
    """
    x = arm.L1 * math.cos(q1) + arm.L2 * math.cos(q1 + q2)
    y = arm.L1 * math.sin(q1) + arm.L2 * math.sin(q1 + q2)
    return x, y


def inverse_kinematics(
    arm: ArmConfig, x: float, y: float, elbow_up: bool = True
) -> Optional[tuple[float, float]]:
    """
    Compute joint angles (q1, q2) to reach target (x, y).

    Returns None if the target is unreachable (outside the workspace annulus).

    The math:
    - From the law of cosines applied to the triangle formed by the two links
      and the line from origin to target, we get cos(q2).
    - q2 has two solutions (elbow-up / elbow-down) based on the sign of sin(q2).
    - q1 is then determined by the geometry.

    Why elbow_up matters: choosing inconsistent elbow configurations between
    waypoints causes the arm to "flip" mid-trajectory, which is violent and dangerous.
    """
    dist_sq = x * x + y * y
    dist = math.sqrt(dist_sq)

    # Check reachability: target must be within the workspace annulus
    max_reach = arm.L1 + arm.L2
    min_reach = abs(arm.L1 - arm.L2)
    if dist > max_reach + 1e-9 or dist < min_reach - 1e-9:
        return None

    # Clamp for numerical stability at workspace boundary
    cos_q2 = (dist_sq - arm.L1**2 - arm.L2**2) / (2 * arm.L1 * arm.L2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))

    # Two solutions: elbow-up (q2 > 0) and elbow-down (q2 < 0)
    sin_q2 = math.sqrt(1 - cos_q2**2)
    if not elbow_up:
        sin_q2 = -sin_q2
    q2 = math.atan2(sin_q2, cos_q2)

    # Solve for q1
    # q1 = atan2(y, x) - atan2(L2*sin(q2), L1 + L2*cos(q2))
    q1 = math.atan2(y, x) - math.atan2(arm.L2 * sin_q2, arm.L1 + arm.L2 * cos_q2)

    return q1, q2


def is_reachable(arm: ArmConfig, x: float, y: float) -> bool:
    """Check if a Cartesian point is within the arm's workspace."""
    dist = math.sqrt(x * x + y * y)
    return abs(arm.L1 - arm.L2) - 1e-9 <= dist <= arm.L1 + arm.L2 + 1e-9


def check_joint_limits(arm: ArmConfig, q1: float, q2: float) -> list[str]:
    """Validate joint angles against limits. Returns list of violation descriptions."""
    violations = []
    if q1 < arm.joint1_limits.min_angle - 1e-9:
        violations.append(f"Joint 1 below min: {math.degrees(q1):.1f} < {math.degrees(arm.joint1_limits.min_angle):.1f} deg")
    if q1 > arm.joint1_limits.max_angle + 1e-9:
        violations.append(f"Joint 1 above max: {math.degrees(q1):.1f} > {math.degrees(arm.joint1_limits.max_angle):.1f} deg")
    if q2 < arm.joint2_limits.min_angle - 1e-9:
        violations.append(f"Joint 2 below min: {math.degrees(q2):.1f} < {math.degrees(arm.joint2_limits.min_angle):.1f} deg")
    if q2 > arm.joint2_limits.max_angle + 1e-9:
        violations.append(f"Joint 2 above max: {math.degrees(q2):.1f} > {math.degrees(arm.joint2_limits.max_angle):.1f} deg")
    return violations


# ============================================================================
# TRAJECTORY GENERATION: TRAPEZOIDAL VELOCITY PROFILE
# ============================================================================

def trapezoidal_profile(
    q_start: float, q_end: float, v_max: float, a_max: float, dt: float
) -> list[tuple[float, float, float, float]]:
    """
    Generate a trapezoidal velocity profile for a single joint.

    Returns list of (time, position, velocity, acceleration) tuples.

    The trapezoidal profile has three phases:
    1. Acceleration: ramp up from 0 to v_max (or peak velocity if distance is short)
    2. Cruise: constant velocity (may be zero-duration for short moves)
    3. Deceleration: ramp down from cruise velocity to 0

    Why trapezoidal? It's the time-optimal profile under a box constraint on
    velocity and acceleration. Industrial robots use this as a baseline.
    The downside: acceleration is discontinuous at phase transitions, causing
    "jerk" — a problem for high-precision applications.
    """
    d = q_end - q_start
    sign = 1.0 if d >= 0 else -1.0
    d = abs(d)

    if d < 1e-10:
        # No movement needed
        return [(0.0, q_start, 0.0, 0.0)]

    # Check if we can reach max velocity (trapezoidal) or not (triangular)
    # Time to accelerate to v_max
    t_accel = v_max / a_max
    # Distance covered during acceleration
    d_accel = 0.5 * a_max * t_accel**2

    if 2 * d_accel > d:
        # Triangular profile: we never reach v_max
        # Peak velocity: v_peak = sqrt(d * a_max)
        t_accel = math.sqrt(d / a_max)
        v_peak = a_max * t_accel
        t_cruise = 0.0
        t_total = 2 * t_accel
    else:
        # Full trapezoidal profile
        v_peak = v_max
        d_cruise = d - 2 * d_accel
        t_cruise = d_cruise / v_max
        t_total = 2 * t_accel + t_cruise

    points = []
    t = 0.0
    while t <= t_total + dt / 2:
        t_clamped = min(t, t_total)

        if t_clamped <= t_accel:
            # Acceleration phase
            acc = a_max
            vel = a_max * t_clamped
            pos = 0.5 * a_max * t_clamped**2
        elif t_clamped <= t_accel + t_cruise:
            # Cruise phase
            t_in_cruise = t_clamped - t_accel
            acc = 0.0
            vel = v_peak
            pos = 0.5 * a_max * t_accel**2 + v_peak * t_in_cruise
        else:
            # Deceleration phase
            t_in_decel = t_clamped - t_accel - t_cruise
            acc = -a_max
            vel = v_peak - a_max * t_in_decel
            pos = (0.5 * a_max * t_accel**2 + v_peak * t_cruise
                   + v_peak * t_in_decel - 0.5 * a_max * t_in_decel**2)

        # Apply direction sign and offset
        points.append((
            t_clamped,
            q_start + sign * pos,
            sign * vel,
            sign * acc
        ))
        t += dt

    return points


# ============================================================================
# TRAJECTORY GENERATION: CUBIC POLYNOMIAL
# ============================================================================

def cubic_trajectory(
    q_start: float, q_end: float,
    v_start: float, v_end: float,
    T: float, dt: float
) -> list[tuple[float, float, float, float]]:
    """
    Generate a cubic polynomial trajectory for a single joint.

    q(t) = a0 + a1*t + a2*t^2 + a3*t^3

    Boundary conditions: q(0)=q_start, q(T)=q_end, q_dot(0)=v_start, q_dot(T)=v_end

    Solving the 4 equations for 4 unknowns:
      a0 = q_start
      a1 = v_start
      a2 = (3*(q_end - q_start)/T^2) - (2*v_start + v_end)/T
      a3 = (-2*(q_end - q_start)/T^3) + (v_start + v_end)/T^2

    Why cubic? It's the lowest-degree polynomial that can satisfy position and
    velocity constraints at both endpoints. Velocity is continuous by construction,
    but acceleration is NOT zero at endpoints (only guaranteed continuous, not smooth).
    For zero-velocity endpoints, it reduces to the simpler form in the README.
    """
    a0 = q_start
    a1 = v_start
    a2 = (3 * (q_end - q_start) / T**2) - (2 * v_start + v_end) / T
    a3 = (-2 * (q_end - q_start) / T**3) + (v_start + v_end) / T**2

    points = []
    t = 0.0
    while t <= T + dt / 2:
        tc = min(t, T)
        pos = a0 + a1 * tc + a2 * tc**2 + a3 * tc**3
        vel = a1 + 2 * a2 * tc + 3 * a3 * tc**2
        acc = 2 * a2 + 6 * a3 * tc
        points.append((tc, pos, vel, acc))
        t += dt

    return points


# ============================================================================
# MULTI-WAYPOINT TRAJECTORY PLANNER
# ============================================================================

def compute_via_velocities(
    waypoints: list[tuple[float, float]], durations: list[float]
) -> list[float]:
    """
    Compute via-point velocities for smooth multi-segment trajectories.

    Uses the heuristic: velocity at via point = average of linear velocities
    of incoming and outgoing segments. Start and end velocities are zero.

    This is a simple but effective heuristic. More sophisticated approaches
    solve a tridiagonal system of equations for C2 continuity (continuous
    acceleration), but this gives C1 (continuous velocity) which is usually
    good enough.
    """
    n = len(waypoints)
    velocities = [0.0] * n  # Start and end at rest

    for i in range(1, n - 1):
        # Linear velocity of incoming segment
        v_in = (waypoints[i] - waypoints[i - 1]) / durations[i - 1]
        # Linear velocity of outgoing segment
        v_out = (waypoints[i + 1] - waypoints[i]) / durations[i]

        # If segments go in opposite directions, velocity should be zero
        # (the arm reverses direction at this point)
        if v_in * v_out < 0:
            velocities[i] = 0.0
        else:
            velocities[i] = (v_in + v_out) / 2.0

    return velocities


def plan_multi_waypoint_trajectory(
    arm: ArmConfig,
    cartesian_waypoints: list[tuple[float, float]],
    segment_durations: list[float],
    dt: float = 0.05,
    elbow_up: bool = True
) -> list[TrajectoryPoint]:
    """
    Plan a complete trajectory through multiple Cartesian waypoints.

    Steps:
    1. Convert all waypoints to joint space via inverse kinematics
    2. Compute via-point velocities for smooth transitions
    3. Generate cubic polynomial segments between consecutive waypoints
    4. Validate all points against joint limits and velocity constraints

    Returns a list of TrajectoryPoints with full state at each timestep.
    """
    # Step 1: Convert to joint space
    joint_waypoints = []
    for i, (x, y) in enumerate(cartesian_waypoints):
        ik_result = inverse_kinematics(arm, x, y, elbow_up=elbow_up)
        if ik_result is None:
            raise ValueError(f"Waypoint {i} ({x}, {y}) is unreachable!")
        q1, q2 = ik_result
        violations = check_joint_limits(arm, q1, q2)
        if violations:
            raise ValueError(f"Waypoint {i} violates joint limits: {violations}")
        joint_waypoints.append((q1, q2))

    # Step 2: Compute via velocities for each joint independently
    q1_waypoints = [jw[0] for jw in joint_waypoints]
    q2_waypoints = [jw[1] for jw in joint_waypoints]

    v1_via = compute_via_velocities(q1_waypoints, segment_durations)
    v2_via = compute_via_velocities(q2_waypoints, segment_durations)

    # Step 3: Generate cubic segments
    trajectory = []
    global_time = 0.0

    for seg in range(len(segment_durations)):
        T = segment_durations[seg]

        # Generate cubic trajectory for each joint
        traj_q1 = cubic_trajectory(
            q1_waypoints[seg], q1_waypoints[seg + 1],
            v1_via[seg], v1_via[seg + 1],
            T, dt
        )
        traj_q2 = cubic_trajectory(
            q2_waypoints[seg], q2_waypoints[seg + 1],
            v2_via[seg], v2_via[seg + 1],
            T, dt
        )

        # Combine joint trajectories into full trajectory points
        # Skip first point of subsequent segments to avoid duplicates
        start_idx = 1 if seg > 0 else 0

        for i in range(start_idx, min(len(traj_q1), len(traj_q2))):
            t_local = traj_q1[i][0]
            q1 = traj_q1[i][1]
            q1_dot = traj_q1[i][2]
            q1_ddot = traj_q1[i][3]
            q2 = traj_q2[i][1]
            q2_dot = traj_q2[i][2]
            q2_ddot = traj_q2[i][3]

            x, y = forward_kinematics(arm, q1, q2)

            trajectory.append(TrajectoryPoint(
                time=global_time + t_local,
                q1=q1, q2=q2,
                q1_dot=q1_dot, q2_dot=q2_dot,
                q1_ddot=q1_ddot, q2_ddot=q2_ddot,
                x=x, y=y
            ))

        global_time += T

    return trajectory


def validate_trajectory(arm: ArmConfig, trajectory: list[TrajectoryPoint]) -> list[str]:
    """
    Check every trajectory point for joint limit and velocity violations.

    Returns a list of violation descriptions. Empty list = valid trajectory.
    """
    violations = []
    for pt in trajectory:
        jv = check_joint_limits(arm, pt.q1, pt.q2)
        for v in jv:
            violations.append(f"t={pt.time:.3f}s: {v}")

        if abs(pt.q1_dot) > arm.joint1_limits.max_velocity + 1e-3:
            violations.append(
                f"t={pt.time:.3f}s: Joint 1 velocity {math.degrees(pt.q1_dot):.1f} "
                f"exceeds max {math.degrees(arm.joint1_limits.max_velocity):.1f} deg/s"
            )
        if abs(pt.q2_dot) > arm.joint2_limits.max_velocity + 1e-3:
            violations.append(
                f"t={pt.time:.3f}s: Joint 2 velocity {math.degrees(pt.q2_dot):.1f} "
                f"exceeds max {math.degrees(arm.joint2_limits.max_velocity):.1f} deg/s"
            )

    return violations


# ============================================================================
# DEMO & VISUALIZATION
# ============================================================================

def print_trajectory_table(trajectory: list[TrajectoryPoint], every_n: int = 1) -> None:
    """Print trajectory in a readable table format."""
    print(f"{'Time':>6s}  {'q1(deg)':>8s}  {'q2(deg)':>8s}  {'q1_dot':>8s}  {'q2_dot':>8s}  {'x':>7s}  {'y':>7s}")
    print("-" * 70)
    for i, pt in enumerate(trajectory):
        if i % every_n == 0 or i == len(trajectory) - 1:
            print(f"{pt.time:6.3f}  {math.degrees(pt.q1):8.2f}  {math.degrees(pt.q2):8.2f}  "
                  f"{math.degrees(pt.q1_dot):8.2f}  {math.degrees(pt.q2_dot):8.2f}  "
                  f"{pt.x:7.4f}  {pt.y:7.4f}")


if __name__ == "__main__":
    # ---- Setup: 2-link arm with realistic constraints ----
    arm = ArmConfig(
        L1=1.0,
        L2=0.8,
        joint1_limits=JointLimits(
            min_angle=math.radians(-180),
            max_angle=math.radians(180),
            max_velocity=math.radians(120),  # 120 deg/s
            max_acceleration=math.radians(300)  # 300 deg/s^2
        ),
        joint2_limits=JointLimits(
            min_angle=math.radians(-150),
            max_angle=math.radians(150),
            max_velocity=math.radians(150),  # 150 deg/s
            max_acceleration=math.radians(400)  # 400 deg/s^2
        )
    )

    print("=" * 70)
    print("ROBOT ARM TRAJECTORY PLANNER")
    print(f"Arm: L1={arm.L1}m, L2={arm.L2}m")
    print(f"Workspace: inner radius={abs(arm.L1 - arm.L2):.1f}m, outer radius={arm.L1 + arm.L2:.1f}m")
    print("=" * 70)

    # ---- Demo 1: Single-segment trapezoidal profile ----
    print("\n--- Demo 1: Trapezoidal Velocity Profile (single joint) ---")
    print("Moving joint 1 from 0 to 90 degrees with v_max=120 deg/s, a_max=300 deg/s^2")

    trap_points = trapezoidal_profile(
        q_start=0, q_end=math.radians(90),
        v_max=math.radians(120), a_max=math.radians(300),
        dt=0.1
    )

    print(f"\n{'Time':>6s}  {'Pos(deg)':>9s}  {'Vel(deg/s)':>11s}  {'Acc(deg/s2)':>12s}")
    print("-" * 45)
    for t, pos, vel, acc in trap_points:
        print(f"{t:6.2f}  {math.degrees(pos):9.2f}  {math.degrees(vel):11.2f}  {math.degrees(acc):12.2f}")

    # ---- Demo 2: Forward and Inverse Kinematics ----
    print("\n--- Demo 2: Kinematics Verification ---")
    test_angles = [(math.radians(45), math.radians(30)),
                   (math.radians(90), math.radians(-45)),
                   (math.radians(0), math.radians(90))]

    for q1, q2 in test_angles:
        x, y = forward_kinematics(arm, q1, q2)
        ik_result = inverse_kinematics(arm, x, y, elbow_up=(q2 >= 0))
        if ik_result:
            q1_recovered, q2_recovered = ik_result
            error = math.sqrt((q1 - q1_recovered)**2 + (q2 - q2_recovered)**2)
            print(f"  FK({math.degrees(q1):.0f}, {math.degrees(q2):.0f}) -> ({x:.4f}, {y:.4f}) "
                  f"-> IK -> ({math.degrees(q1_recovered):.1f}, {math.degrees(q2_recovered):.1f})  "
                  f"error={math.degrees(error):.4f} deg")

    # ---- Demo 3: Multi-waypoint trajectory ----
    print("\n--- Demo 3: Multi-Waypoint Cubic Trajectory ---")

    # Define a 4-waypoint path in Cartesian space
    waypoints = [
        (1.2, 0.5),   # Start
        (0.8, 1.0),   # Via point 1
        (0.4, 0.8),   # Via point 2
        (1.0, 0.3),   # End
    ]
    durations = [1.5, 1.0, 1.5]  # Time for each segment

    print(f"Waypoints (x, y): {waypoints}")
    print(f"Segment durations: {durations}s")

    # Check reachability
    for i, (x, y) in enumerate(waypoints):
        dist = math.sqrt(x**2 + y**2)
        print(f"  Waypoint {i}: ({x}, {y}), dist={dist:.3f}m, reachable={is_reachable(arm, x, y)}")

    # Plan trajectory
    trajectory = plan_multi_waypoint_trajectory(arm, waypoints, durations, dt=0.1)

    print(f"\nTrajectory: {len(trajectory)} points over {trajectory[-1].time:.1f}s")
    print("\nSampled trajectory (every 3rd point):")
    print_trajectory_table(trajectory, every_n=3)

    # ---- Demo 4: Validation ----
    print("\n--- Demo 4: Trajectory Validation ---")
    violations = validate_trajectory(arm, trajectory)
    if violations:
        print(f"Found {len(violations)} violations:")
        for v in violations[:5]:
            print(f"  {v}")
    else:
        print("Trajectory is valid! No joint limit or velocity violations.")

    # Show start and end positions match waypoints
    start_pt = trajectory[0]
    end_pt = trajectory[-1]
    print(f"\nStart: target=({waypoints[0][0]}, {waypoints[0][1]}), "
          f"actual=({start_pt.x:.4f}, {start_pt.y:.4f})")
    print(f"End:   target=({waypoints[-1][0]}, {waypoints[-1][1]}), "
          f"actual=({end_pt.x:.4f}, {end_pt.y:.4f})")

    # ---- Demo 5: Unreachable point handling ----
    print("\n--- Demo 5: Error Handling ---")
    unreachable = (2.5, 0.0)
    print(f"Attempting to reach ({unreachable[0]}, {unreachable[1]})...")
    print(f"  Reachable: {is_reachable(arm, *unreachable)}")
    result = inverse_kinematics(arm, *unreachable)
    print(f"  IK result: {result}")

    print("\nDone! All demos completed successfully.")
