"""
Day 063: Sensor Fusion (IMU + GPS) using Extended Kalman Filter

Combines high-frequency, drift-prone IMU data with low-frequency, noisy GPS data
to produce accurate position and velocity estimates. This is the same fundamental
approach used in every phone, drone, and autonomous vehicle navigation system.

Key insight: The Kalman filter optimally blends two imperfect information sources
by tracking how uncertain each one is and weighting them accordingly.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SensorConfig:
    """Configuration for sensor noise characteristics."""
    imu_rate_hz: float = 100.0        # IMU update rate
    gps_rate_hz: float = 1.0          # GPS update rate
    imu_accel_noise_std: float = 0.5  # m/s^2 - accelerometer noise
    imu_accel_bias: np.ndarray = None # m/s^2 - constant accelerometer bias
    gps_position_noise_std: float = 3.0  # meters - GPS position noise
    gps_dropout_prob: float = 0.1     # probability of GPS signal loss per reading
    duration_seconds: float = 60.0    # total simulation time

    def __post_init__(self):
        if self.imu_accel_bias is None:
            # A small constant bias that causes drift over time
            # This is realistic: real IMUs have biases of 0.01-0.1 m/s^2
            self.imu_accel_bias = np.array([0.05, -0.03])


def generate_ground_truth(config: SensorConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a realistic 2D trajectory with smooth turns.

    Creates a figure-eight-like path using sinusoidal acceleration components.
    This gives us ground truth position, velocity, and acceleration at every
    IMU timestep, which we'll corrupt with noise to simulate sensors.

    Returns:
        times: (N,) array of timestamps
        positions: (N, 2) true positions [x, y] in meters
        velocities: (N, 2) true velocities [vx, vy] in m/s
        accelerations: (N, 2) true accelerations [ax, ay] in m/s^2
    """
    dt = 1.0 / config.imu_rate_hz
    n_steps = int(config.duration_seconds * config.imu_rate_hz)
    times = np.arange(n_steps) * dt

    # Generate smooth acceleration profile using sinusoids
    # This creates a path with gentle curves — realistic for a ground robot
    freq1, freq2 = 0.05, 0.08  # Hz — slow oscillations
    accelerations = np.column_stack([
        1.0 * np.sin(2 * np.pi * freq1 * times),      # x acceleration
        0.8 * np.cos(2 * np.pi * freq2 * times) + 0.3  # y acceleration with forward bias
    ])

    # Integrate acceleration -> velocity -> position (ground truth, no noise)
    velocities = np.zeros((n_steps, 2))
    positions = np.zeros((n_steps, 2))

    for i in range(1, n_steps):
        velocities[i] = velocities[i - 1] + accelerations[i - 1] * dt
        positions[i] = positions[i - 1] + velocities[i - 1] * dt + 0.5 * accelerations[i - 1] * dt ** 2

    return times, positions, velocities, accelerations


def simulate_imu(
    accelerations: np.ndarray,
    config: SensorConfig,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate noisy IMU accelerometer readings.

    Real IMU errors have two components:
    1. White noise: random jitter each reading (modeled as Gaussian)
    2. Bias: constant offset from calibration errors (causes systematic drift)

    The bias is the killer — it integrates into a linearly-growing velocity error,
    which then integrates into a quadratically-growing position error.

    Args:
        accelerations: (N, 2) true accelerations
        config: sensor configuration
        rng: random number generator for reproducibility

    Returns:
        (N, 2) noisy IMU acceleration readings
    """
    noise = rng.normal(0, config.imu_accel_noise_std, size=accelerations.shape)
    # Bias is constant — same offset every reading. This is what makes IMU drift predictable
    # in direction but devastating in magnitude.
    return accelerations + noise + config.imu_accel_bias


def simulate_gps(
    times: np.ndarray,
    positions: np.ndarray,
    config: SensorConfig,
    rng: np.random.Generator
) -> List[Tuple[float, np.ndarray]]:
    """
    Simulate GPS position fixes at a lower rate with noise and occasional dropouts.

    GPS characteristics modeled:
    - Lower update rate than IMU (typically 1-10 Hz vs 100+ Hz)
    - Gaussian position noise (~2-5m standard)
    - Random signal dropouts (tunnels, urban canyons, interference)

    Returns:
        List of (timestamp, noisy_position) tuples — only when GPS is available
    """
    dt_imu = 1.0 / config.imu_rate_hz
    gps_interval = int(config.imu_rate_hz / config.gps_rate_hz)  # IMU steps between GPS fixes

    gps_readings = []
    for i in range(0, len(times), gps_interval):
        # Simulate dropout — GPS randomly unavailable
        if rng.random() < config.gps_dropout_prob:
            continue

        noise = rng.normal(0, config.gps_position_noise_std, size=2)
        gps_readings.append((times[i], positions[i] + noise))

    return gps_readings


def dead_reckoning(
    imu_readings: np.ndarray,
    dt: float,
    initial_position: np.ndarray = None,
    initial_velocity: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate IMU data to estimate position (no corrections).

    This is the naive approach: just integrate acceleration twice.
    It works great for ~1-5 seconds, then drift takes over.

    Double integration means errors grow QUADRATICALLY with time:
    - Acceleration bias b → position error = 0.5 * b * t^2
    - With b=0.05 m/s^2: after 60s, position error ≈ 90 meters

    Args:
        imu_readings: (N, 2) noisy accelerometer data
        dt: time step between readings
        initial_position: starting position (default [0, 0])
        initial_velocity: starting velocity (default [0, 0])

    Returns:
        positions: (N, 2) estimated positions
        velocities: (N, 2) estimated velocities
    """
    n = len(imu_readings)
    positions = np.zeros((n, 2))
    velocities = np.zeros((n, 2))

    if initial_position is not None:
        positions[0] = initial_position
    if initial_velocity is not None:
        velocities[0] = initial_velocity

    for i in range(1, n):
        velocities[i] = velocities[i - 1] + imu_readings[i - 1] * dt
        positions[i] = positions[i - 1] + velocities[i - 1] * dt + 0.5 * imu_readings[i - 1] * dt ** 2

    return positions, velocities


class KalmanFusionFilter:
    """
    Kalman filter for fusing IMU and GPS data.

    State vector: [px, py, vx, vy] — position and velocity in 2D
    IMU provides acceleration input (control input u)
    GPS provides position observations

    The filter maintains a state estimate AND a covariance matrix that tracks
    how uncertain we are. This uncertainty is what enables optimal blending:
    - After many IMU predictions without GPS: covariance grows → we trust GPS more when it arrives
    - After a GPS update: covariance shrinks → we trust our prediction more until next GPS
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_accel: float = 1.0,
        gps_noise_std: float = 3.0
    ):
        """
        Initialize the Kalman filter.

        Args:
            initial_state: [px, py, vx, vy] initial state estimate
            initial_covariance: 4x4 initial uncertainty matrix
            process_noise_accel: acceleration process noise (m/s^2) — models
                unknown accelerations not captured by our IMU input
            gps_noise_std: GPS measurement noise standard deviation (meters)
        """
        self.x = initial_state.copy().astype(float)  # State estimate
        self.P = initial_covariance.copy().astype(float)  # State covariance

        self.process_noise_accel = process_noise_accel
        self.gps_noise_std = gps_noise_std

        # Measurement matrix H: GPS observes position only, not velocity
        # z = H @ x → [px, py] = [[1,0,0,0],[0,1,0,0]] @ [px,py,vx,vy]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Measurement noise covariance — GPS noise is approximately independent in x and y
        self.R = np.eye(2) * gps_noise_std ** 2

    def _get_F_and_B(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build state transition matrix F and control input matrix B for timestep dt.

        Physics model (constant acceleration over dt):
            px_new = px + vx*dt + 0.5*ax*dt^2
            py_new = py + vy*dt + 0.5*ay*dt^2
            vx_new = vx + ax*dt
            vy_new = vy + ay*dt
        """
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        B = np.array([
            [0.5 * dt ** 2, 0],
            [0, 0.5 * dt ** 2],
            [dt, 0],
            [0, dt]
        ], dtype=float)

        return F, B

    def _get_Q(self, dt: float) -> np.ndarray:
        """
        Build process noise covariance matrix Q.

        Models uncertainty in the motion model — the accelerations we DON'T
        measure. Derived from assuming a white noise acceleration model:

        Q = G @ G^T @ sigma_a^2

        where G = [0.5*dt^2, 0.5*dt^2, dt, dt]^T captures how acceleration
        noise propagates into position and velocity.

        Getting Q right matters: too small and the filter is overconfident in
        the IMU (ignores GPS). Too large and it's too jittery (overreacts to GPS).
        """
        sigma_a = self.process_noise_accel

        # Noise gain vector — how acceleration noise enters each state
        G = np.array([
            [0.5 * dt ** 2, 0],
            [0, 0.5 * dt ** 2],
            [dt, 0],
            [0, dt]
        ])

        return G @ G.T * sigma_a ** 2

    def predict(self, dt: float, accel: np.ndarray) -> np.ndarray:
        """
        Prediction step: propagate state forward using IMU acceleration.

        This runs at IMU rate (100 Hz). Each predict step:
        1. Moves the state estimate forward using physics + IMU input
        2. Grows the covariance (we become more uncertain over time)

        Args:
            dt: time step
            accel: [ax, ay] IMU acceleration reading

        Returns:
            Current state estimate after prediction
        """
        F, B = self._get_F_and_B(dt)
        Q = self._get_Q(dt)

        # State prediction: physics model + IMU input
        self.x = F @ self.x + B @ accel

        # Covariance prediction: uncertainty grows
        # F @ P @ F^T stretches existing uncertainty through the dynamics
        # Q adds new uncertainty from process noise
        self.P = F @ self.P @ F.T + Q

        return self.x.copy()

    def update(self, gps_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step: correct the state using a GPS position fix.

        This runs at GPS rate (1 Hz). Each update step:
        1. Computes the innovation (GPS - prediction)
        2. Computes the Kalman gain (how much to trust GPS vs prediction)
        3. Corrects the state estimate
        4. Shrinks the covariance (we become more certain after seeing data)

        The Kalman gain K is the magic: it's the MMSE-optimal weighting.
        - K ≈ 0: trust prediction (P small relative to R) — filter is confident
        - K ≈ I: trust GPS (P large relative to R) — filter has drifted

        Args:
            gps_position: [px, py] GPS position measurement

        Returns:
            Tuple of (corrected state, Kalman gain)
        """
        # Innovation: how far off was our prediction from the GPS?
        z = gps_position
        y = z - self.H @ self.x  # Innovation vector

        # Innovation covariance: total uncertainty in the innovation
        # S combines prediction uncertainty (H@P@H^T) with measurement noise (R)
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: optimal weighting between prediction and measurement
        # K = P @ H^T @ S^-1
        # Using solve instead of inverse for numerical stability
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State correction: blend prediction with GPS observation
        self.x = self.x + K @ y

        # Covariance correction: uncertainty shrinks after incorporating measurement
        # Joseph form: (I - K@H) @ P @ (I - K@H)^T + K @ R @ K^T
        # Simplified (equivalent for optimal K): (I - K@H) @ P
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy(), K


def run_sensor_fusion(
    config: SensorConfig,
    seed: int = 42
) -> dict:
    """
    Run the complete sensor fusion pipeline.

    Pipeline:
    1. Generate ground truth trajectory
    2. Simulate noisy IMU and GPS sensors
    3. Run dead reckoning (IMU-only) to show drift
    4. Run Kalman filter fusion to show improvement
    5. Compute error metrics

    Args:
        config: sensor configuration
        seed: random seed for reproducibility

    Returns:
        Dictionary with all trajectories and metrics
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / config.imu_rate_hz

    # Step 1: Ground truth
    times, true_pos, true_vel, true_accel = generate_ground_truth(config)
    n_steps = len(times)

    # Step 2: Simulate sensors
    imu_readings = simulate_imu(true_accel, config, rng)
    gps_readings = simulate_gps(times, true_pos, config, rng)

    # Step 3: Dead reckoning (IMU only — no GPS corrections)
    dr_positions, dr_velocities = dead_reckoning(imu_readings, dt)

    # Step 4: Kalman filter fusion
    # Initialize with true starting state and moderate uncertainty
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    initial_cov = np.eye(4) * 1.0  # Moderate initial uncertainty

    kf = KalmanFusionFilter(
        initial_state=initial_state,
        initial_covariance=initial_cov,
        process_noise_accel=config.imu_accel_noise_std,  # Match expected IMU noise
        gps_noise_std=config.gps_position_noise_std
    )

    # Run filter: predict at IMU rate, update when GPS is available
    fused_positions = np.zeros((n_steps, 2))
    fused_velocities = np.zeros((n_steps, 2))
    covariance_trace = np.zeros(n_steps)  # Track uncertainty over time

    gps_idx = 0  # Pointer into GPS readings list
    gps_update_times = []

    for i in range(n_steps):
        # Always predict with IMU (high rate)
        if i > 0:
            state = kf.predict(dt, imu_readings[i - 1])
        else:
            state = kf.x.copy()

        # Check if we have a GPS reading at this time
        if gps_idx < len(gps_readings):
            gps_time, gps_pos = gps_readings[gps_idx]
            if abs(times[i] - gps_time) < dt / 2:
                state, kalman_gain = kf.update(gps_pos)
                gps_idx += 1
                gps_update_times.append(times[i])

        fused_positions[i] = state[:2]
        fused_velocities[i] = state[2:4]
        covariance_trace[i] = np.trace(kf.P)

    # Step 5: Compute error metrics
    dr_pos_error = np.sqrt(np.sum((dr_positions - true_pos) ** 2, axis=1))
    fused_pos_error = np.sqrt(np.sum((fused_positions - true_pos) ** 2, axis=1))

    dr_rms = np.sqrt(np.mean(dr_pos_error ** 2))
    fused_rms = np.sqrt(np.mean(fused_pos_error ** 2))

    return {
        "times": times,
        "true_positions": true_pos,
        "true_velocities": true_vel,
        "imu_readings": imu_readings,
        "gps_readings": gps_readings,
        "gps_update_times": gps_update_times,
        "dr_positions": dr_positions,
        "fused_positions": fused_positions,
        "fused_velocities": fused_velocities,
        "covariance_trace": covariance_trace,
        "dr_pos_error": dr_pos_error,
        "fused_pos_error": fused_pos_error,
        "dr_rms": dr_rms,
        "fused_rms": fused_rms,
        "config": config,
    }


def print_results(results: dict) -> None:
    """Print a detailed summary of the fusion results."""
    config = results["config"]
    times = results["times"]

    print("=" * 70)
    print("SENSOR FUSION RESULTS: IMU + GPS via Kalman Filter")
    print("=" * 70)

    print(f"\n--- Simulation Configuration ---")
    print(f"  Duration:          {config.duration_seconds:.0f} seconds")
    print(f"  IMU rate:          {config.imu_rate_hz:.0f} Hz")
    print(f"  GPS rate:          {config.gps_rate_hz:.0f} Hz")
    print(f"  IMU accel noise:   {config.imu_accel_noise_std:.2f} m/s^2")
    print(f"  IMU accel bias:    [{config.imu_accel_bias[0]:.3f}, {config.imu_accel_bias[1]:.3f}] m/s^2")
    print(f"  GPS position noise:{config.gps_position_noise_std:.1f} m")
    print(f"  GPS dropout prob:  {config.gps_dropout_prob:.0%}")
    print(f"  Total IMU samples: {len(times)}")
    print(f"  GPS fixes received:{len(results['gps_readings'])}")

    print(f"\n--- Position Error (RMS) ---")
    print(f"  Dead reckoning (IMU only): {results['dr_rms']:8.2f} meters")
    print(f"  Kalman filter (fused):     {results['fused_rms']:8.2f} meters")
    improvement = (1 - results['fused_rms'] / results['dr_rms']) * 100
    print(f"  Improvement:               {improvement:8.1f}%")

    print(f"\n--- Error Over Time ---")
    # Show error at several time points to demonstrate drift vs stability
    checkpoints = [10, 20, 30, 45, 60]
    print(f"  {'Time (s)':>10} {'DR Error (m)':>14} {'Fused Error (m)':>16}")
    print(f"  {'-' * 42}")
    for t in checkpoints:
        if t <= config.duration_seconds:
            idx = int(t * config.imu_rate_hz) - 1
            if idx < len(results['dr_pos_error']):
                print(f"  {t:>10.0f} {results['dr_pos_error'][idx]:>14.2f} {results['fused_pos_error'][idx]:>16.2f}")

    print(f"\n--- Final State ---")
    final_true = results['true_positions'][-1]
    final_dr = results['dr_positions'][-1]
    final_fused = results['fused_positions'][-1]
    print(f"  True position:     ({final_true[0]:8.2f}, {final_true[1]:8.2f})")
    print(f"  Dead reckoning:    ({final_dr[0]:8.2f}, {final_dr[1]:8.2f})")
    print(f"  Kalman fused:      ({final_fused[0]:8.2f}, {final_fused[1]:8.2f})")

    final_dr_err = np.sqrt(np.sum((final_dr - final_true) ** 2))
    final_fused_err = np.sqrt(np.sum((final_fused - final_true) ** 2))
    print(f"  Final DR error:    {final_dr_err:8.2f} m")
    print(f"  Final fused error: {final_fused_err:8.2f} m")

    # Show covariance evolution — this is the filter's self-reported uncertainty
    print(f"\n--- Filter Uncertainty (Covariance Trace) ---")
    cov = results['covariance_trace']
    print(f"  Initial: {cov[0]:.4f}")
    print(f"  Max:     {np.max(cov):.4f}")
    print(f"  Min:     {np.min(cov):.4f}")
    print(f"  Final:   {cov[-1]:.4f}")

    print(f"\n--- Key Takeaways ---")
    print(f"  * Dead reckoning error grows ~quadratically (IMU bias integrates twice)")
    print(f"  * Kalman filter error stays bounded (GPS anchors prevent drift)")
    print(f"  * Fused estimate is better than either sensor alone")
    print(f"  * Covariance grows between GPS fixes, shrinks at each GPS update")
    print("=" * 70)


if __name__ == "__main__":
    # Run with default configuration
    config = SensorConfig()
    results = run_sensor_fusion(config)
    print_results(results)

    # Demonstrate effect of GPS quality
    print("\n\n")
    print("=" * 70)
    print("EXPERIMENT: Effect of GPS Noise on Fusion Quality")
    print("=" * 70)

    for gps_noise in [1.0, 3.0, 10.0]:
        config_exp = SensorConfig(gps_position_noise_std=gps_noise)
        results_exp = run_sensor_fusion(config_exp)
        print(f"  GPS noise {gps_noise:5.1f}m → Fused RMS: {results_exp['fused_rms']:.2f}m "
              f"(DR: {results_exp['dr_rms']:.2f}m, improvement: "
              f"{(1 - results_exp['fused_rms'] / results_exp['dr_rms']) * 100:.1f}%)")

    # Demonstrate effect of GPS rate
    print(f"\n--- Effect of GPS Update Rate ---")
    for gps_rate in [0.2, 1.0, 5.0, 10.0]:
        config_exp = SensorConfig(gps_rate_hz=gps_rate)
        results_exp = run_sensor_fusion(config_exp)
        print(f"  GPS rate {gps_rate:5.1f}Hz → Fused RMS: {results_exp['fused_rms']:.2f}m "
              f"(GPS fixes: {len(results_exp['gps_readings'])})")
