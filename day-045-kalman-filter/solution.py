"""
Day 045: Kalman Filter Basics

A from-scratch implementation of the linear Kalman filter for tracking
a 1D moving object using noisy position measurements. Demonstrates the
predict-update cycle, Kalman gain dynamics, and uncertainty evolution.

This builds on Day 044's SLAM concept — SLAM uses variants of the Kalman
filter (EKF/UKF) to simultaneously estimate robot pose and landmark positions.
"""

import numpy as np
from typing import Tuple, List, Optional


class KalmanFilter:
    """
    Linear Kalman filter for state estimation.

    The filter maintains a state estimate x and covariance P, and provides
    predict() and update() methods that implement the standard Kalman
    filter equations.

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
            F: State transition matrix (n x n). Encodes the physics model —
               how the state evolves from one time step to the next.
            H: Measurement matrix (m x n). Maps the state space to measurement
               space — what the sensor actually observes.
            Q: Process noise covariance (n x n). Represents uncertainty in the
               model. Larger Q = less trust in the model prediction.
            R: Measurement noise covariance (m x m). Represents sensor noise.
               Larger R = less trust in the sensor readings.
            x0: Initial state estimate (n x 1). Our best guess at time 0.
            P0: Initial state covariance (n x n). Our uncertainty at time 0.
                Large P0 means "I don't know where I am" — the filter will
                converge faster from measurements.
        """
        # Store system matrices — these define the linear dynamical system
        self.F = F.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()

        # Current state estimate and covariance
        # These are updated in-place by predict() and update()
        self.x = x0.copy()
        self.P = P0.copy()

        # Dimensionality: n = state dim, m = measurement dim
        self.n = F.shape[0]
        self.m = H.shape[0]

        # Store history for analysis and visualization
        # We track estimates, covariances, and Kalman gains over time
        self.history: List[dict] = []

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step (time update).

        Projects the state estimate and covariance forward using the
        physics model. After this step, uncertainty INCREASES because
        we're less sure about the model's accuracy.

        Equations:
            x_pred = F @ x
            P_pred = F @ P @ F^T + Q

        The F @ P @ F^T term propagates current uncertainty through the
        state transition. Q adds additional uncertainty from process noise.

        Returns:
            Tuple of (predicted state, predicted covariance)
        """
        # Project state forward: x_{k|k-1} = F @ x_{k-1|k-1}
        # This is pure physics — "where do I expect to be next?"
        self.x = self.F @ self.x

        # Project covariance forward: P_{k|k-1} = F @ P_{k-1|k-1} @ F^T + Q
        # The F @ P @ F^T transforms the uncertainty ellipse through the
        # state transition. Q adds model uncertainty on top.
        # Note: P always grows during prediction — we become less certain.
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step (measurement update).

        Incorporates a new sensor measurement to correct the prediction.
        After this step, uncertainty DECREASES (or stays the same) because
        we've gained information.

        The key computation is the Kalman gain K, which optimally blends
        the prediction with the measurement based on their relative
        uncertainties.

        Equations:
            y = z - H @ x           (innovation: how far off was our prediction?)
            S = H @ P @ H^T + R     (innovation covariance: total uncertainty)
            K = P @ H^T @ S^(-1)    (Kalman gain: optimal blending ratio)
            x = x + K @ y           (corrected state)
            P = (I - K @ H) @ P     (corrected covariance)

        Args:
            z: Measurement vector (m x 1)

        Returns:
            Tuple of (updated state, updated covariance)
        """
        # Innovation (measurement residual): how surprised are we?
        # y = z - H @ x is the difference between what we measured and
        # what we predicted we'd measure. Large y = big surprise.
        y = z - self.H @ self.x

        # Innovation covariance: total uncertainty in the innovation
        # S combines prediction uncertainty (H @ P @ H^T, projected into
        # measurement space) with measurement noise (R).
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: the optimal blending ratio
        # K = P @ H^T @ S^(-1)
        # If P is large (uncertain prediction) and R is small (good sensor):
        #   → K is large → trust the measurement more
        # If P is small (confident prediction) and R is large (noisy sensor):
        #   → K is small → trust the prediction more
        # This is the core insight: K automatically balances trust.
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: correct prediction with weighted innovation
        # x = x + K @ y
        # When K=1: x = x + (z - H@x) → snaps to measurement
        # When K=0: x = x → ignores measurement entirely
        self.x = self.x + K @ y

        # Covariance update: reduce uncertainty
        # P = (I - K @ H) @ P
        # The (I - K @ H) factor shrinks P — measurements always reduce
        # uncertainty (or leave it unchanged if K=0).
        # Using Joseph form would be more numerically stable for production:
        # P = (I-KH) @ P @ (I-KH)^T + K @ R @ K^T
        # But the simple form is fine for educational purposes.
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

        # Record this step for later analysis
        self.history.append({
            "x": self.x.copy(),
            "P": self.P.copy(),
            "K": K.copy(),
            "innovation": y.copy(),
            "S": S.copy(),
        })

        return self.x.copy(), self.P.copy()

    def get_kalman_gain(self) -> Optional[np.ndarray]:
        """Return the most recent Kalman gain, or None if no updates yet."""
        if self.history:
            return self.history[-1]["K"]
        return None

    def get_uncertainty(self) -> np.ndarray:
        """
        Return standard deviations of each state variable.

        The diagonal of P gives variances; sqrt gives standard deviations.
        This is the ±1σ uncertainty for each state component.
        """
        return np.sqrt(np.diag(self.P))


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

    Generates ground truth trajectory, process noise perturbations,
    and noisy sensor measurements.

    The object moves at roughly constant velocity, but with small random
    accelerations (process noise). The sensor measures position with
    Gaussian noise.

    Args:
        n_steps: Number of time steps to simulate
        dt: Time step duration in seconds
        true_velocity: Nominal velocity of the object
        process_noise_std: Std dev of random acceleration perturbations
        measurement_noise_std: Std dev of position sensor noise
        initial_position: Starting position
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
            true_states: (n_steps, 2) array of [position, velocity] ground truth
            measurements: (n_steps,) array of noisy position readings
            times: (n_steps,) array of time values
    """
    rng = np.random.default_rng(seed)

    true_states = np.zeros((n_steps, 2))  # [position, velocity]
    measurements = np.zeros(n_steps)
    times = np.arange(n_steps) * dt

    # Initialize
    true_states[0] = [initial_position, true_velocity]
    measurements[0] = initial_position + rng.normal(0, measurement_noise_std)

    for k in range(1, n_steps):
        # True dynamics with process noise (small random accelerations)
        # This is what happens in reality — the model isn't perfect
        acceleration_noise = rng.normal(0, process_noise_std)
        true_states[k, 1] = true_states[k - 1, 1] + acceleration_noise * dt
        true_states[k, 0] = (
            true_states[k - 1, 0] + true_states[k - 1, 1] * dt
        )

        # Noisy measurement of position only
        # The sensor can't see velocity — that's a hidden state
        measurements[k] = true_states[k, 0] + rng.normal(
            0, measurement_noise_std
        )

    return true_states, measurements, times


def run_kalman_filter(
    measurements: np.ndarray,
    dt: float = 0.1,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 1.5,
) -> Tuple[KalmanFilter, np.ndarray, np.ndarray]:
    """
    Run a Kalman filter on a sequence of 1D position measurements.

    Sets up the constant-velocity model and processes all measurements
    through the predict-update cycle.

    Args:
        measurements: Array of noisy position readings
        dt: Time step (must match simulation)
        process_noise_std: Assumed process noise (tuning parameter)
        measurement_noise_std: Assumed measurement noise (tuning parameter)

    Returns:
        Tuple of:
            kf: The KalmanFilter object (with full history)
            estimates: (n_steps, 2) array of [position, velocity] estimates
            uncertainties: (n_steps, 2) array of [pos_std, vel_std]
    """
    n_steps = len(measurements)

    # State transition matrix for constant-velocity model
    # x_new = [[1, dt], [0, 1]] @ x_old
    # position_new = position_old + velocity × dt
    # velocity_new = velocity_old (constant velocity assumption)
    F = np.array([[1, dt],
                  [0, 1]])

    # Measurement matrix: we only observe position
    # z = [1, 0] @ [pos, vel]^T = pos
    H = np.array([[1, 0]])

    # Process noise covariance
    # Models random accelerations. The standard discrete-time noise model
    # for constant-velocity assumes acceleration is white noise with
    # variance q. This gives:
    # Q = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
    # Simplified version (works well in practice):
    q = process_noise_std ** 2
    Q = q * np.array([[dt**4 / 4, dt**3 / 2],
                       [dt**3 / 2, dt**2]])

    # Measurement noise covariance (scalar wrapped in 1x1 matrix)
    R = np.array([[measurement_noise_std ** 2]])

    # Initial state: start at first measurement with zero velocity
    # We're honest that we don't know the velocity
    x0 = np.array([measurements[0], 0.0])

    # Initial covariance: large uncertainty (we're unsure)
    # Large P0 means the filter will quickly adapt to measurements
    # rather than stubbornly holding its initial estimate
    P0 = np.array([[10.0, 0.0],
                    [0.0, 10.0]])

    # Create and run filter
    kf = KalmanFilter(F, H, Q, R, x0, P0)

    estimates = np.zeros((n_steps, 2))
    uncertainties = np.zeros((n_steps, 2))

    # First step: just use initial state
    estimates[0] = kf.x
    uncertainties[0] = kf.get_uncertainty()

    # Process each measurement through predict → update cycle
    for k in range(1, n_steps):
        # Predict: "where do I think I'll be next?"
        kf.predict()

        # Update: "now that I see the sensor, let me correct"
        z = np.array([measurements[k]])
        kf.update(z)

        estimates[k] = kf.x
        uncertainties[k] = kf.get_uncertainty()

    return kf, estimates, uncertainties


def compute_rmse(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute root mean squared error between estimates and ground truth.

    RMSE is the standard metric for evaluating filter performance.
    Lower is better. Comparing filter RMSE to raw measurement RMSE
    shows the filter's value.
    """
    return float(np.sqrt(np.mean((estimates - ground_truth) ** 2)))


if __name__ == "__main__":
    print("=" * 70)
    print("Day 045: Kalman Filter — 1D Object Tracking")
    print("=" * 70)

    # --- Simulation parameters ---
    N_STEPS = 100
    DT = 0.1
    TRUE_VEL = 2.0
    PROCESS_NOISE = 0.3
    MEASUREMENT_NOISE = 1.5

    print(f"\nSimulation: {N_STEPS} steps, dt={DT}s, true velocity={TRUE_VEL} m/s")
    print(f"Process noise σ={PROCESS_NOISE}, Measurement noise σ={MEASUREMENT_NOISE}")

    # --- Generate ground truth and noisy measurements ---
    true_states, measurements, times = simulate_1d_tracking(
        n_steps=N_STEPS,
        dt=DT,
        true_velocity=TRUE_VEL,
        process_noise_std=PROCESS_NOISE,
        measurement_noise_std=MEASUREMENT_NOISE,
    )

    print(f"\nGround truth position range: [{true_states[:, 0].min():.2f}, "
          f"{true_states[:, 0].max():.2f}]")
    print(f"Measurement noise range: [{(measurements - true_states[:, 0]).min():.2f}, "
          f"{(measurements - true_states[:, 0]).max():.2f}]")

    # --- Run Kalman filter ---
    kf, estimates, uncertainties = run_kalman_filter(
        measurements,
        dt=DT,
        process_noise_std=PROCESS_NOISE,
        measurement_noise_std=MEASUREMENT_NOISE,
    )

    # --- Evaluate performance ---
    pos_rmse_raw = compute_rmse(measurements, true_states[:, 0])
    pos_rmse_kf = compute_rmse(estimates[:, 0], true_states[:, 0])
    vel_rmse_kf = compute_rmse(estimates[:, 1], true_states[:, 1])

    print("\n" + "-" * 50)
    print("PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"Position RMSE (raw measurements): {pos_rmse_raw:.4f}")
    print(f"Position RMSE (Kalman filter):    {pos_rmse_kf:.4f}")
    print(f"Improvement:                      {(1 - pos_rmse_kf / pos_rmse_raw) * 100:.1f}%")
    print(f"Velocity RMSE (Kalman filter):    {vel_rmse_kf:.4f}")
    print(f"  (Raw measurements can't estimate velocity at all!)")

    # --- Show Kalman gain convergence ---
    print("\n" + "-" * 50)
    print("KALMAN GAIN CONVERGENCE")
    print("-" * 50)
    gains = [h["K"][0, 0] for h in kf.history]
    print(f"Initial Kalman gain (position): {gains[0]:.4f}")
    print(f"Final Kalman gain (position):   {gains[-1]:.4f}")
    print(f"Steady-state reached after ~{next(i for i in range(len(gains)-1) if abs(gains[i+1]-gains[i]) < 0.001) + 1} steps")
    print("  High initial gain → filter trusts sensors early (uncertain prediction)")
    print("  Low steady-state gain → filter trusts its model more as it converges")

    # --- Show uncertainty evolution ---
    print("\n" + "-" * 50)
    print("UNCERTAINTY EVOLUTION (±1σ)")
    print("-" * 50)
    for step in [0, 1, 5, 10, 50, N_STEPS - 1]:
        print(f"Step {step:3d}: position ±{uncertainties[step, 0]:.4f}, "
              f"velocity ±{uncertainties[step, 1]:.4f}")
    print("  Uncertainty shrinks as the filter processes more measurements")

    # --- Show a few predict-update steps in detail ---
    print("\n" + "-" * 50)
    print("DETAILED TRACE: Steps 1-5")
    print("-" * 50)
    for k in range(min(5, len(kf.history))):
        h = kf.history[k]
        print(f"\nStep {k + 1}:")
        print(f"  Measurement:     z = {measurements[k + 1]:.3f}")
        print(f"  Innovation:      y = {h['innovation'][0]:.3f} "
              f"(predicted vs actual)")
        print(f"  Kalman gain:     K = [{h['K'][0, 0]:.4f}, {h['K'][1, 0]:.4f}]")
        print(f"  State estimate:  pos = {h['x'][0]:.3f}, vel = {h['x'][1]:.3f}")
        print(f"  Uncertainty:     σ_pos = {np.sqrt(h['P'][0, 0]):.4f}, "
              f"σ_vel = {np.sqrt(h['P'][1, 1]):.4f}")

    # --- Verify filter consistency ---
    # The normalized innovation squared should be ~chi-squared(1)
    # In other words, innovations should be within ±2σ about 95% of the time
    innovations = [h["innovation"][0] for h in kf.history]
    innov_stds = [np.sqrt(h["S"][0, 0]) for h in kf.history]
    normalized = [inn / std for inn, std in zip(innovations, innov_stds)]
    within_2sigma = sum(1 for n in normalized if abs(n) < 2) / len(normalized)

    print("\n" + "-" * 50)
    print("FILTER CONSISTENCY CHECK")
    print("-" * 50)
    print(f"Innovations within ±2σ: {within_2sigma * 100:.1f}% (expected ~95%)")
    print(f"Mean normalized innovation: {np.mean(normalized):.3f} (expected ~0)")
    print(f"Std normalized innovation:  {np.std(normalized):.3f} (expected ~1)")

    if within_2sigma > 0.85:
        print("✓ Filter is consistent — noise parameters match reality")
    else:
        print("⚠ Filter may be inconsistent — check Q and R tuning")

    print("\n" + "=" * 70)
    print("Key takeaway: The Kalman filter reduced position RMSE by "
          f"{(1 - pos_rmse_kf / pos_rmse_raw) * 100:.0f}%")
    print("AND estimated velocity (a hidden state) from position-only measurements!")
    print("=" * 70)
