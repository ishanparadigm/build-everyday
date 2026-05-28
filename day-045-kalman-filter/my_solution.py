"""
Day 045: Kalman Filter Basics — Your Implementation

Implement a Kalman filter from scratch to track a 1D moving object
using noisy position measurements.

Key concepts to remember:
- The predict step uses the physics model to project state forward
- The update step uses sensor data to correct the prediction
- The Kalman gain balances trust between model and sensor
- Covariance P grows during prediction, shrinks during update

Hints:
- All operations are matrix multiplications (use @ operator)
- The innovation y = z - H @ x tells you "how surprised" the filter is
- If your filter diverges, check that P is growing in predict and shrinking in update
"""

import numpy as np
from typing import Tuple, List, Optional


class KalmanFilter:
    """
    Linear Kalman filter for state estimation.

    State model: x_{k+1} = F @ x_k + w_k,  w_k ~ N(0, Q)
    Measurement: z_k = H @ x_k + v_k,       v_k ~ N(0, R)
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> None:
        """
        Initialize the Kalman filter.

        Args:
            F: State transition matrix (n x n)
            H: Measurement matrix (m x n)
            Q: Process noise covariance (n x n)
            R: Measurement noise covariance (m x m)
            x0: Initial state estimate (n,)
            P0: Initial state covariance (n x n)
        """
        raise NotImplementedError("TODO: Store system matrices and initialize state")

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step (time update).

        Hint: Two equations:
          x_pred = F @ x
          P_pred = F @ P @ F^T + Q

        Returns:
            Tuple of (predicted state, predicted covariance)
        """
        raise NotImplementedError("TODO: Implement the predict step")

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step (measurement update).

        Hint: Five equations in order:
          1. Innovation: y = z - H @ x
          2. Innovation covariance: S = H @ P @ H^T + R
          3. Kalman gain: K = P @ H^T @ S^(-1)
          4. State update: x = x + K @ y
          5. Covariance update: P = (I - K @ H) @ P

        Args:
            z: Measurement vector (m,)

        Returns:
            Tuple of (updated state, updated covariance)
        """
        raise NotImplementedError("TODO: Implement the update step")

    def get_kalman_gain(self) -> Optional[np.ndarray]:
        """Return the most recent Kalman gain, or None if no updates yet."""
        raise NotImplementedError("TODO: Return the last Kalman gain from history")

    def get_uncertainty(self) -> np.ndarray:
        """
        Return standard deviations of each state variable.

        Hint: sqrt of diagonal of P
        """
        raise NotImplementedError("TODO: Extract uncertainty from covariance matrix")


def simulate_1d_tracking(
    n_steps: int = 100,
    dt: float = 0.1,
    true_velocity: float = 2.0,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 1.5,
    initial_position: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a 1D constant-velocity object with noisy measurements.

    Hint: At each step:
      - velocity changes by a small random acceleration
      - position updates via position += velocity * dt
      - measurement = true position + Gaussian noise

    Args:
        n_steps: Number of time steps
        dt: Time step duration
        true_velocity: Nominal velocity
        process_noise_std: Std dev of random acceleration
        measurement_noise_std: Std dev of sensor noise
        initial_position: Starting position
        seed: Random seed

    Returns:
        Tuple of (true_states, measurements, times)
        - true_states: (n_steps, 2) — [position, velocity] ground truth
        - measurements: (n_steps,) — noisy position readings
        - times: (n_steps,) — time values
    """
    raise NotImplementedError("TODO: Simulate the system")


def run_kalman_filter(
    measurements: np.ndarray,
    dt: float = 0.1,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 1.5,
) -> Tuple[KalmanFilter, np.ndarray, np.ndarray]:
    """
    Run a Kalman filter on a sequence of 1D position measurements.

    Hint: Set up the system matrices for a constant-velocity model:
      - F encodes "position += velocity * dt"
      - H selects position from the state vector
      - Q models random acceleration noise
      - R is measurement noise variance

    Then loop: predict() → update(z) for each measurement.

    Args:
        measurements: Array of noisy position readings
        dt: Time step
        process_noise_std: Assumed process noise
        measurement_noise_std: Assumed measurement noise

    Returns:
        Tuple of (kf, estimates, uncertainties)
    """
    raise NotImplementedError("TODO: Set up matrices, create filter, and run predict-update loop")


def compute_rmse(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute root mean squared error."""
    raise NotImplementedError("TODO: Implement RMSE calculation")


if __name__ == "__main__":
    print("Kalman Filter — 1D Object Tracking")
    print("=" * 50)

    # Generate test data
    true_states, measurements, times = simulate_1d_tracking(
        n_steps=100, dt=0.1, true_velocity=2.0,
        process_noise_std=0.3, measurement_noise_std=1.5,
    )
    print(f"Generated {len(measurements)} measurements")

    # Run your Kalman filter
    kf, estimates, uncertainties = run_kalman_filter(
        measurements, dt=0.1, process_noise_std=0.3, measurement_noise_std=1.5,
    )

    # Evaluate
    raw_rmse = compute_rmse(measurements, true_states[:, 0])
    kf_rmse = compute_rmse(estimates[:, 0], true_states[:, 0])

    print(f"\nPosition RMSE (raw):    {raw_rmse:.4f}")
    print(f"Position RMSE (filter): {kf_rmse:.4f}")
    print(f"Improvement: {(1 - kf_rmse / raw_rmse) * 100:.1f}%")

    # Check Kalman gain convergence
    gain = kf.get_kalman_gain()
    if gain is not None:
        print(f"\nFinal Kalman gain: {gain[0, 0]:.4f}")

    print(f"\nFinal uncertainty: pos ±{uncertainties[-1, 0]:.4f}, "
          f"vel ±{uncertainties[-1, 1]:.4f}")
