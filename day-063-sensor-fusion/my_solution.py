"""
Day 063: Sensor Fusion (IMU + GPS) using Extended Kalman Filter

YOUR TASK: Implement sensor fusion combining noisy IMU acceleration data
with slow GPS position fixes using a Kalman filter.

Key concepts to implement:
- IMU noise simulation (white noise + constant bias)
- GPS noise simulation (Gaussian noise + dropouts)
- Dead reckoning via double integration
- Kalman filter predict/update cycle

Run this file to test your implementation:
    python3 my_solution.py
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SensorConfig:
    """Configuration for sensor noise characteristics."""
    imu_rate_hz: float = 100.0
    gps_rate_hz: float = 1.0
    imu_accel_noise_std: float = 0.5
    imu_accel_bias: np.ndarray = None
    gps_position_noise_std: float = 3.0
    gps_dropout_prob: float = 0.1
    duration_seconds: float = 60.0

    def __post_init__(self):
        if self.imu_accel_bias is None:
            self.imu_accel_bias = np.array([0.05, -0.03])


def generate_ground_truth(config: SensorConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a realistic 2D trajectory with smooth turns.

    Use sinusoidal accelerations to create a curved path, then integrate
    twice to get velocity and position.

    Returns:
        times: (N,) timestamps
        positions: (N, 2) true [x, y] positions
        velocities: (N, 2) true [vx, vy] velocities
        accelerations: (N, 2) true [ax, ay] accelerations
    """
    raise NotImplementedError("TODO: implement this")
    # Hint: Create sinusoidal accelerations, then use Euler integration
    # velocity[i] = velocity[i-1] + accel[i-1] * dt
    # position[i] = position[i-1] + velocity[i-1] * dt + 0.5 * accel[i-1] * dt^2


def simulate_imu(
    accelerations: np.ndarray,
    config: SensorConfig,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Add noise and bias to true accelerations to simulate IMU readings.

    Two noise components:
    1. White noise: rng.normal(0, std, size=shape)
    2. Constant bias: added to every reading (this causes drift!)

    Args:
        accelerations: (N, 2) true accelerations
        config: sensor configuration
        rng: random number generator

    Returns:
        (N, 2) noisy IMU readings
    """
    raise NotImplementedError("TODO: implement this")
    # Hint: noisy = true + gaussian_noise + constant_bias


def simulate_gps(
    times: np.ndarray,
    positions: np.ndarray,
    config: SensorConfig,
    rng: np.random.Generator
) -> List[Tuple[float, np.ndarray]]:
    """
    Simulate GPS position fixes at lower rate with noise and dropouts.

    Sample true position at GPS rate, add Gaussian noise, randomly drop readings.

    Returns:
        List of (timestamp, noisy_position) tuples
    """
    raise NotImplementedError("TODO: implement this")
    # Hint: Loop through times at GPS interval
    # Skip readings with probability gps_dropout_prob
    # Add Gaussian noise to position


def dead_reckoning(
    imu_readings: np.ndarray,
    dt: float,
    initial_position: np.ndarray = None,
    initial_velocity: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate IMU acceleration twice to get position (no GPS corrections).

    This shows how bad IMU-only navigation is — errors grow quadratically.

    Args:
        imu_readings: (N, 2) noisy accelerometer data
        dt: time step
        initial_position: starting position (default [0, 0])
        initial_velocity: starting velocity (default [0, 0])

    Returns:
        positions: (N, 2) estimated positions
        velocities: (N, 2) estimated velocities
    """
    raise NotImplementedError("TODO: implement this")
    # Hint: Same integration as generate_ground_truth, but with noisy data


class KalmanFusionFilter:
    """
    Kalman filter for fusing IMU and GPS data.

    State: [px, py, vx, vy]
    Input: IMU acceleration [ax, ay]
    Measurement: GPS position [px, py]
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_accel: float = 1.0,
        gps_noise_std: float = 3.0
    ):
        """
        Initialize the filter.

        Args:
            initial_state: [px, py, vx, vy]
            initial_covariance: 4x4 matrix
            process_noise_accel: process noise parameter
            gps_noise_std: GPS measurement noise std
        """
        raise NotImplementedError("TODO: implement this")
        # Hint: Store state x, covariance P, build measurement matrix H (2x4),
        # and measurement noise covariance R (2x2)

    def predict(self, dt: float, accel: np.ndarray) -> np.ndarray:
        """
        Prediction step using IMU acceleration.

        Build F (state transition) and B (control input) matrices for:
            px_new = px + vx*dt + 0.5*ax*dt^2
            py_new = py + vy*dt + 0.5*ay*dt^2
            vx_new = vx + ax*dt
            vy_new = vy + ay*dt

        Then: x = F @ x + B @ accel
              P = F @ P @ F^T + Q

        Returns:
            Current state estimate
        """
        raise NotImplementedError("TODO: implement this")
        # Hint: F is 4x4, B is 4x2. Q models process noise.

    def update(self, gps_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step using GPS position.

        Innovation:    y = z - H @ x
        Innov. cov:    S = H @ P @ H^T + R
        Kalman gain:   K = P @ H^T @ S^-1
        State update:  x = x + K @ y
        Cov. update:   P = (I - K @ H) @ P

        Returns:
            Tuple of (corrected state, Kalman gain)
        """
        raise NotImplementedError("TODO: implement this")
        # Hint: The Kalman gain K determines how much to trust GPS vs prediction


def run_sensor_fusion(config: SensorConfig, seed: int = 42) -> dict:
    """
    Run the complete sensor fusion pipeline.

    1. Generate ground truth
    2. Simulate IMU and GPS
    3. Run dead reckoning
    4. Run Kalman filter
    5. Compute errors

    Returns:
        Dict with trajectories and metrics
    """
    raise NotImplementedError("TODO: implement this")
    # Hint: Initialize KalmanFusionFilter, loop through timesteps:
    #   - Always call predict() with IMU data
    #   - Call update() when GPS is available at current time
    #   - Store results


if __name__ == "__main__":
    config = SensorConfig()

    print("Running sensor fusion pipeline...")
    results = run_sensor_fusion(config)

    print(f"\nDead reckoning RMS error: {results['dr_rms']:.2f} meters")
    print(f"Kalman fused RMS error:   {results['fused_rms']:.2f} meters")
    improvement = (1 - results['fused_rms'] / results['dr_rms']) * 100
    print(f"Improvement:              {improvement:.1f}%")

    print(f"\nGPS fixes received: {len(results['gps_readings'])}")
    print(f"Total IMU samples:  {len(results['times'])}")
