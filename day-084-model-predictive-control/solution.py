"""
Day 84: Model Predictive Control (MPC) for Robot Navigation

A complete MPC implementation using a bicycle kinematic model to track a reference
trajectory. Demonstrates the receding horizon principle, cost function design,
constraint handling, and closed-loop simulation.

Key concepts:
- Bicycle kinematic model (nonlinear state dynamics)
- Receding horizon optimization (predict, optimize, apply first input, repeat)
- Multi-objective cost function (tracking + effort + smoothness)
- Input/state constraints via scipy.optimize.minimize with bounds
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from scipy.optimize import minimize


@dataclass
class VehicleParams:
    """Physical parameters of the bicycle-model vehicle."""
    wheelbase: float = 2.5          # Distance between front and rear axles [m]
    max_steering: float = 0.6       # Max steering angle [rad] (~34 degrees)
    max_steering_rate: float = 0.3  # Max steering change per step [rad]
    max_speed: float = 15.0         # Max forward speed [m/s]
    max_accel: float = 3.0          # Max acceleration [m/s^2]
    max_decel: float = 5.0          # Max braking deceleration [m/s^2]


@dataclass
class MPCParams:
    """Tuning parameters for the MPC controller."""
    horizon: int = 15               # Prediction horizon steps
    dt: float = 0.1                 # Timestep [s]

    # Cost function weights — these encode what we care about
    w_pos: float = 10.0             # Position tracking (high = follow path tightly)
    w_heading: float = 3.0          # Heading tracking (moderate = face the right way)
    w_vel: float = 1.0              # Velocity tracking (low = speed is secondary)
    w_steer: float = 0.5            # Steering effort penalty (smooth steering)
    w_accel: float = 0.5            # Acceleration effort penalty (smooth speed)
    w_steer_rate: float = 2.0       # Steering rate penalty (jerk reduction)
    w_terminal: float = 20.0        # Terminal position cost (pull toward goal)


def bicycle_model(state: np.ndarray, control: np.ndarray,
                  params: VehicleParams, dt: float) -> np.ndarray:
    """
    Discrete-time bicycle kinematic model.

    The bicycle model is the standard simplification for car-like robots:
    it collapses the two front wheels into one and the two rear wheels into one,
    giving a "bicycle" viewed from above. This captures the essential nonholonomic
    constraint (cars can't move sideways) while being simple enough for real-time MPC.

    State: [x, y, theta, v]
        x, y   — position in world frame [m]
        theta  — heading angle [rad], 0 = pointing right (east)
        v      — forward velocity [m/s]

    Control: [delta, a]
        delta  — front wheel steering angle [rad], positive = left turn
        a      — longitudinal acceleration [m/s^2]

    The key equation: theta_dot = v/L * tan(delta)
    This means turning rate depends on BOTH speed and steering angle.
    At zero speed, you can't turn — matching real car behavior.
    """
    x, y, theta, v = state
    delta, a = control

    # Euler integration of the continuous-time bicycle model
    # Why Euler and not RK4? For small dt (0.1s), Euler is accurate enough
    # and much simpler. Production MPC uses RK4 or exact discretization.
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + (v / params.wheelbase) * np.tan(delta) * dt
    v_next = v + a * dt

    # Enforce speed limits — velocity can't go negative (no reversing in this model)
    v_next = np.clip(v_next, 0.0, params.max_speed)

    # Normalize heading to [-pi, pi] to avoid discontinuities in cost function
    theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))

    return np.array([x_next, y_next, theta_next, v_next])


def predict_trajectory(state: np.ndarray, controls: np.ndarray,
                       vehicle: VehicleParams, dt: float) -> np.ndarray:
    """
    Roll out the bicycle model over a sequence of controls to get predicted states.

    This is the "shooting" part of our optimization: given initial state and a
    sequence of controls, simulate forward to get the resulting trajectory.
    The optimizer will then adjust the controls to minimize cost.

    Args:
        state: Initial state [x, y, theta, v]
        controls: Array of shape (N, 2) — N control pairs [delta, a]
        vehicle: Vehicle parameters
        dt: Timestep

    Returns:
        states: Array of shape (N+1, 4) — initial state + N predicted states
    """
    N = len(controls)
    states = np.zeros((N + 1, 4))
    states[0] = state

    for k in range(N):
        states[k + 1] = bicycle_model(states[k], controls[k], vehicle, dt)

    return states


def generate_reference_trajectory(n_points: int, trajectory_type: str = "figure8"
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a reference trajectory for the robot to follow.

    Returns positions, headings, and reference velocities at each point.
    The reference gives MPC its "goal" — what to track.

    Why figure-8? It tests both left and right turns, straight sections,
    and varying curvature — a good stress test for any controller.
    """
    t = np.linspace(0, 2 * np.pi, n_points)

    if trajectory_type == "figure8":
        # Lemniscate of Bernoulli (figure-8), scaled up
        scale = 30.0
        x_ref = scale * np.sin(t)
        y_ref = scale * np.sin(t) * np.cos(t)
    elif trajectory_type == "circle":
        radius = 25.0
        x_ref = radius * np.cos(t)
        y_ref = radius * np.sin(t)
    elif trajectory_type == "slalom":
        x_ref = t * 10.0  # Move forward
        y_ref = 10.0 * np.sin(t * 2)  # Weave side to side
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    # Compute reference headings from the path tangent
    # heading = atan2(dy, dx) gives the direction the path is going
    dx = np.gradient(x_ref, t)
    dy = np.gradient(y_ref, t)
    theta_ref = np.arctan2(dy, dx)

    # Reference velocity based on path curvature — slow down in tight turns
    # Curvature kappa = |d_theta/ds| where s is arc length
    ds = np.sqrt(dx**2 + dy**2)
    dtheta = np.gradient(theta_ref)
    # Unwrap heading differences to avoid discontinuities at +-pi
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    curvature = np.abs(dtheta) / (ds + 1e-6)

    # Higher curvature → lower speed. This is how humans drive too.
    v_max = 8.0
    v_min = 2.0
    v_ref = v_max - (v_max - v_min) * np.clip(curvature / 0.15, 0, 1)

    positions = np.column_stack([x_ref, y_ref])
    return positions, theta_ref, v_ref


def find_closest_reference_index(state: np.ndarray, ref_positions: np.ndarray,
                                 last_idx: int = 0, lookahead: int = 50) -> int:
    """
    Find the closest point on the reference trajectory to the current position.

    We search forward from the last known index to handle laps correctly
    (prevents "jumping back" to an earlier part of the path that happens
    to be geometrically close).

    The lookahead window prevents O(n) search every step — we only check
    nearby reference points.
    """
    pos = state[:2]
    # Search window: from last_idx forward (with wraparound)
    n = len(ref_positions)
    search_end = min(last_idx + lookahead, n)
    search_range = range(last_idx, search_end)

    if len(search_range) == 0:
        return last_idx

    distances = np.linalg.norm(ref_positions[search_range] - pos, axis=1)
    best_local = np.argmin(distances)
    return last_idx + best_local


def angle_diff(a: float, b: float) -> float:
    """
    Compute the shortest angular difference between two angles.

    This is critical for heading cost — without it, going from 179° to -179°
    looks like a 358° error instead of a 2° error.
    """
    diff = a - b
    return np.arctan2(np.sin(diff), np.cos(diff))


def mpc_cost(u_flat: np.ndarray, state: np.ndarray, ref_positions: np.ndarray,
             ref_headings: np.ndarray, ref_velocities: np.ndarray,
             ref_start_idx: int, prev_delta: float,
             vehicle: VehicleParams, mpc: MPCParams) -> float:
    """
    Compute the total MPC cost for a candidate control sequence.

    This is the function that scipy.optimize.minimize will try to minimize.
    The art of MPC is in designing this cost function — it implicitly defines
    the robot's behavior.

    Args:
        u_flat: Flattened control sequence [delta_0, a_0, delta_1, a_1, ...]
        state: Current robot state
        ref_positions: Reference positions (x, y) along the path
        ref_headings: Reference heading angles
        ref_velocities: Reference velocities
        ref_start_idx: Index into reference trajectory for current time
        prev_delta: Previous steering angle (for rate penalty)
        vehicle: Vehicle parameters
        mpc: MPC tuning parameters
    """
    N = mpc.horizon
    n_ref = len(ref_positions)

    # Reshape flat control vector into (N, 2) array of [delta, a] pairs
    controls = u_flat.reshape(N, 2)

    # Simulate forward to get predicted trajectory
    predicted = predict_trajectory(state, controls, vehicle, mpc.dt)

    cost = 0.0
    for k in range(N):
        # Which reference point should we be at time k steps from now?
        # We advance along the reference proportionally
        ref_idx = min(ref_start_idx + k, n_ref - 1)

        # Position tracking error — the primary objective
        pos_error = predicted[k + 1, :2] - ref_positions[ref_idx]
        cost += mpc.w_pos * np.sum(pos_error**2)

        # Heading error — use angle_diff to handle wraparound
        heading_err = angle_diff(predicted[k + 1, 2], ref_headings[ref_idx])
        cost += mpc.w_heading * heading_err**2

        # Velocity tracking — match the curvature-adapted reference speed
        vel_err = predicted[k + 1, 3] - ref_velocities[ref_idx]
        cost += mpc.w_vel * vel_err**2

        # Control effort — penalize large steering and acceleration
        cost += mpc.w_steer * controls[k, 0]**2
        cost += mpc.w_accel * controls[k, 1]**2

        # Steering rate — penalize sudden changes in steering
        # This is crucial for smooth, natural-looking motion
        if k == 0:
            steer_rate = controls[k, 0] - prev_delta
        else:
            steer_rate = controls[k, 0] - controls[k - 1, 0]
        cost += mpc.w_steer_rate * steer_rate**2

    # Terminal cost — extra penalty on final position to "pull" the trajectory
    # toward the reference. Without this, the optimizer might let the end of
    # the horizon drift because there's no future cost beyond it.
    ref_idx_terminal = min(ref_start_idx + N, n_ref - 1)
    terminal_error = predicted[N, :2] - ref_positions[ref_idx_terminal]
    cost += mpc.w_terminal * np.sum(terminal_error**2)

    return cost


class MPCController:
    """
    Model Predictive Controller for bicycle-model robot navigation.

    At each step:
    1. Find where we are on the reference trajectory
    2. Extract the relevant reference segment (horizon-length window)
    3. Optimize control inputs over the prediction horizon
    4. Apply only the first control input
    5. Store the solution as a warm start for next iteration

    The warm start is important: by shifting last solution forward, we give
    the optimizer a good initial guess, dramatically reducing solve time.
    """

    def __init__(self, vehicle: VehicleParams, mpc: MPCParams):
        self.vehicle = vehicle
        self.mpc = mpc
        self.prev_delta = 0.0  # Track previous steering for rate penalty
        self.last_ref_idx = 0  # Track position along reference
        self.prev_solution: Optional[np.ndarray] = None  # Warm start

    def compute_control(self, state: np.ndarray, ref_positions: np.ndarray,
                        ref_headings: np.ndarray, ref_velocities: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the MPC optimization and return the optimal first control input.

        Returns:
            control: [delta, a] — the steering angle and acceleration to apply NOW
            predicted: Predicted trajectory for visualization
        """
        N = self.mpc.horizon
        n_ref = len(ref_positions)

        # Step 1: Find closest reference point (with forward-search to avoid backtracking)
        self.last_ref_idx = find_closest_reference_index(
            state, ref_positions, self.last_ref_idx
        )

        # Step 2: Warm start — shift previous solution forward
        # If we solved [u0, u1, ..., u_{N-1}] last step and applied u0,
        # then [u1, u2, ..., u_{N-1}, u_{N-1}] is a good initial guess
        # (the last element is duplicated since we don't have a u_N).
        if self.prev_solution is not None:
            u0 = np.zeros(N * 2)
            prev = self.prev_solution.reshape(N, 2)
            shifted = np.vstack([prev[1:], prev[-1:]])  # Shift and duplicate last
            u0 = shifted.flatten()
        else:
            u0 = np.zeros(N * 2)  # Cold start: zero controls

        # Step 3: Set up bounds (constraint handling via box constraints)
        # Each control is [delta, a], so bounds alternate
        bounds = []
        for k in range(N):
            bounds.append((-self.vehicle.max_steering, self.vehicle.max_steering))
            bounds.append((-self.vehicle.max_decel, self.vehicle.max_accel))

        # Step 4: Optimize!
        # We use L-BFGS-B because it handles box constraints efficiently.
        # For production MPC, you'd use IPOPT or a dedicated QP solver.
        result = minimize(
            mpc_cost,
            u0,
            args=(state, ref_positions, ref_headings, ref_velocities,
                  self.last_ref_idx, self.prev_delta, self.vehicle, self.mpc),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-4}
        )

        # Extract optimal controls
        u_opt = result.x.reshape(N, 2)

        # Enforce steering rate constraint manually (L-BFGS-B only does box constraints)
        steer_change = u_opt[0, 0] - self.prev_delta
        if abs(steer_change) > self.vehicle.max_steering_rate:
            u_opt[0, 0] = self.prev_delta + np.sign(steer_change) * self.vehicle.max_steering_rate

        # Step 5: Extract first control, update state for next call
        control = u_opt[0]
        self.prev_delta = control[0]
        self.prev_solution = u_opt.flatten()

        # Predict trajectory for visualization
        predicted = predict_trajectory(state, u_opt, self.vehicle, self.mpc.dt)

        return control, predicted


def simulate_mpc(n_steps: int = 300, trajectory_type: str = "figure8"
                 ) -> dict:
    """
    Run a full closed-loop MPC simulation.

    This is where everything comes together: at each timestep, the MPC
    controller sees the current state, optimizes over the horizon, and
    applies the first control. The simulation records everything for
    analysis.

    Args:
        n_steps: Number of simulation steps
        trajectory_type: "figure8", "circle", or "slalom"

    Returns:
        Dictionary with recorded states, controls, costs, and reference data
    """
    vehicle = VehicleParams()
    mpc_params = MPCParams()

    # Generate reference trajectory with enough points for the simulation
    n_ref_points = n_steps + mpc_params.horizon + 50
    ref_positions, ref_headings, ref_velocities = generate_reference_trajectory(
        n_ref_points, trajectory_type
    )

    # Initial state: start at the beginning of the reference, facing the right way
    state = np.array([
        ref_positions[0, 0],  # x
        ref_positions[0, 1],  # y
        ref_headings[0],      # theta
        ref_velocities[0]     # v
    ])

    controller = MPCController(vehicle, mpc_params)

    # Recording arrays
    states = np.zeros((n_steps + 1, 4))
    controls = np.zeros((n_steps, 2))
    tracking_errors = np.zeros(n_steps)
    states[0] = state

    print(f"Simulating MPC with {trajectory_type} trajectory ({n_steps} steps)...")
    print(f"  Horizon: {mpc_params.horizon}, dt: {mpc_params.dt}s")
    print(f"  Vehicle: wheelbase={vehicle.wheelbase}m, max_steer={np.degrees(vehicle.max_steering):.0f}deg")
    print()

    for step in range(n_steps):
        # MPC computes optimal control
        control, predicted = controller.compute_control(
            state, ref_positions, ref_headings, ref_velocities
        )

        # Apply control to get next state (the "real" system)
        state = bicycle_model(state, control, vehicle, mpc_params.dt)

        # Record
        states[step + 1] = state
        controls[step] = control

        # Compute tracking error for analysis
        ref_idx = controller.last_ref_idx
        tracking_errors[step] = np.linalg.norm(state[:2] - ref_positions[ref_idx])

        # Progress report every 50 steps
        if (step + 1) % 50 == 0:
            avg_error = np.mean(tracking_errors[max(0, step-49):step+1])
            print(f"  Step {step+1:4d}/{n_steps}: "
                  f"pos=({state[0]:7.2f}, {state[1]:7.2f}), "
                  f"v={state[3]:5.2f} m/s, "
                  f"steer={np.degrees(control[0]):6.1f} deg, "
                  f"avg_tracking_err={avg_error:.3f} m")

    return {
        'states': states,
        'controls': controls,
        'tracking_errors': tracking_errors,
        'ref_positions': ref_positions[:n_steps],
        'ref_headings': ref_headings[:n_steps],
        'ref_velocities': ref_velocities[:n_steps],
        'vehicle': vehicle,
        'mpc_params': mpc_params,
    }


def analyze_performance(results: dict) -> None:
    """
    Print detailed performance analysis of the MPC controller.

    Good MPC should have:
    - Low average tracking error (< 1m for this scenario)
    - Smooth control inputs (no oscillation)
    - All constraints satisfied
    - No large transient errors after initial convergence
    """
    states = results['states']
    controls = results['controls']
    errors = results['tracking_errors']
    vehicle = results['vehicle']

    print("\n" + "="*60)
    print("MPC PERFORMANCE ANALYSIS")
    print("="*60)

    # Tracking accuracy
    # Skip first 10 steps (transient period while controller converges)
    steady_errors = errors[10:]
    print(f"\nTracking Error (after initial transient):")
    print(f"  Mean:  {np.mean(steady_errors):.4f} m")
    print(f"  Max:   {np.max(steady_errors):.4f} m")
    print(f"  Std:   {np.std(steady_errors):.4f} m")
    print(f"  99th%: {np.percentile(steady_errors, 99):.4f} m")

    # Control smoothness
    steer = controls[:, 0]
    accel = controls[:, 1]
    steer_rate = np.diff(steer)
    accel_rate = np.diff(accel)
    print(f"\nControl Smoothness:")
    print(f"  Steering — mean: {np.degrees(np.mean(np.abs(steer))):.2f} deg, "
          f"max: {np.degrees(np.max(np.abs(steer))):.2f} deg")
    print(f"  Steering rate — mean: {np.degrees(np.mean(np.abs(steer_rate))):.3f} deg/step, "
          f"max: {np.degrees(np.max(np.abs(steer_rate))):.3f} deg/step")
    print(f"  Acceleration — mean: {np.mean(np.abs(accel)):.3f} m/s^2, "
          f"max: {np.max(np.abs(accel)):.3f} m/s^2")

    # Constraint satisfaction
    steer_violations = np.sum(np.abs(steer) > vehicle.max_steering + 1e-6)
    speed_violations = np.sum(
        (states[1:, 3] > vehicle.max_speed + 1e-6) | (states[1:, 3] < -1e-6)
    )
    print(f"\nConstraint Satisfaction:")
    print(f"  Steering violations: {steer_violations}")
    print(f"  Speed violations: {speed_violations}")
    print(f"  All constraints met: {'YES' if steer_violations + speed_violations == 0 else 'NO'}")

    # Speed profile
    speeds = states[1:, 3]
    print(f"\nSpeed Profile:")
    print(f"  Mean: {np.mean(speeds):.2f} m/s ({np.mean(speeds)*3.6:.1f} km/h)")
    print(f"  Max:  {np.max(speeds):.2f} m/s ({np.max(speeds)*3.6:.1f} km/h)")
    print(f"  Min:  {np.min(speeds):.2f} m/s ({np.min(speeds)*3.6:.1f} km/h)")

    # Overall assessment
    print(f"\nOverall Assessment:")
    mean_err = np.mean(steady_errors)
    if mean_err < 0.5:
        print(f"  EXCELLENT — Mean tracking error {mean_err:.3f}m is well within tolerance")
    elif mean_err < 1.0:
        print(f"  GOOD — Mean tracking error {mean_err:.3f}m is acceptable")
    elif mean_err < 2.0:
        print(f"  FAIR — Mean tracking error {mean_err:.3f}m, consider tuning weights")
    else:
        print(f"  POOR — Mean tracking error {mean_err:.3f}m, needs significant tuning")


if __name__ == '__main__':
    print("="*60)
    print("MODEL PREDICTIVE CONTROL (MPC) — Robot Navigation")
    print("="*60)
    print()
    print("MPC is the 'look ahead and optimize' approach to control.")
    print("Unlike PID (react to error), MPC predicts the future, finds")
    print("the best control sequence, applies only the first step, and")
    print("re-plans. This receding horizon strategy handles constraints")
    print("and produces smooth, optimal trajectories.")
    print()

    # Run simulation on a figure-8 trajectory
    results = simulate_mpc(n_steps=300, trajectory_type="figure8")

    # Detailed analysis
    analyze_performance(results)

    # Show a sample of the trajectory
    states = results['states']
    print(f"\n{'='*60}")
    print("SAMPLE TRAJECTORY (every 30 steps)")
    print(f"{'='*60}")
    print(f"{'Step':>5} {'X':>8} {'Y':>8} {'Heading':>9} {'Speed':>7}")
    print(f"{'':>5} {'(m)':>8} {'(m)':>8} {'(deg)':>9} {'(m/s)':>7}")
    print("-" * 42)
    for i in range(0, len(states), 30):
        s = states[i]
        print(f"{i:5d} {s[0]:8.2f} {s[1]:8.2f} {np.degrees(s[2]):9.2f} {s[3]:7.2f}")

    print(f"\nSimulation complete. The robot successfully tracked a figure-8")
    print(f"trajectory using MPC with a {results['mpc_params'].horizon}-step prediction horizon.")
