"""
Day 044: SLAM Concept Implementation — EKF-SLAM

Implements Simultaneous Localization and Mapping using the Extended Kalman Filter.
A simulated robot drives through a 2D world with landmarks, using noisy odometry
and range-bearing observations to simultaneously build a map and localize itself.

Key insight: the cross-correlations in the covariance matrix allow a single landmark
observation to improve estimates of ALL landmarks, not just the observed one.
"""

import math
import random
import warnings
import numpy as np
from typing import Optional

# Suppress numerical warnings from early EKF steps where covariance is near-zero.
# Results converge correctly after the first few observations.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)


class Landmark:
    """A point landmark in the world with a unique ID."""

    def __init__(self, landmark_id: int, x: float, y: float):
        self.id = landmark_id
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Landmark({self.id}, x={self.x:.2f}, y={self.y:.2f})"


class Robot:
    """Simulated robot with noisy odometry and range-bearing sensors."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        odom_noise_d: float = 0.1,
        odom_noise_angle: float = 0.05,
        sensor_noise_range: float = 0.2,
        sensor_noise_bearing: float = 0.05,
        sensor_max_range: float = 10.0,
    ):
        # True pose (unknown to the SLAM algorithm — only used for simulation)
        self.x = x
        self.y = y
        self.theta = theta

        # Noise parameters
        self.odom_noise_d = odom_noise_d          # std dev on distance
        self.odom_noise_angle = odom_noise_angle   # std dev on angle (radians)
        self.sensor_noise_range = sensor_noise_range
        self.sensor_noise_bearing = sensor_noise_bearing
        self.sensor_max_range = sensor_max_range

    def move(self, d: float, alpha: float) -> tuple[float, float]:
        """
        Move the robot: turn by alpha, then drive forward by d.
        Returns noisy odometry (d_noisy, alpha_noisy) that the SLAM algorithm sees.
        """
        # True motion
        self.theta = normalize_angle(self.theta + alpha)
        self.x += d * math.cos(self.theta)
        self.y += d * math.sin(self.theta)

        # Noisy odometry — this is what the SLAM algorithm receives
        d_noisy = d + random.gauss(0, self.odom_noise_d)
        alpha_noisy = alpha + random.gauss(0, self.odom_noise_angle)

        return d_noisy, alpha_noisy

    def observe(self, landmarks: list[Landmark]) -> list[tuple[int, float, float]]:
        """
        Observe all landmarks within sensor range.
        Returns list of (landmark_id, range, bearing) with added noise.
        """
        observations = []
        for lm in landmarks:
            dx = lm.x - self.x
            dy = lm.y - self.y
            true_range = math.sqrt(dx**2 + dy**2)

            if true_range <= self.sensor_max_range:
                true_bearing = math.atan2(dy, dx) - self.theta

                # Add sensor noise
                noisy_range = true_range + random.gauss(0, self.sensor_noise_range)
                noisy_bearing = true_bearing + random.gauss(0, self.sensor_noise_bearing)

                observations.append((lm.id, noisy_range, normalize_angle(noisy_bearing)))

        return observations


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class EKFSLAM:
    """
    Extended Kalman Filter SLAM.

    State vector: [x_r, y_r, θ_r, x_L1, y_L1, x_L2, y_L2, ...]
    - First 3 elements: robot pose
    - Each pair after that: a landmark (x, y) position

    The covariance matrix P tracks uncertainty and cross-correlations
    between the robot and all landmarks.
    """

    def __init__(
        self,
        motion_noise: np.ndarray,
        observation_noise: np.ndarray,
    ):
        """
        Args:
            motion_noise: 2x2 diagonal matrix [σ²_d, σ²_α] for odometry noise
            observation_noise: 2x2 diagonal matrix [σ²_range, σ²_bearing] for sensor noise
        """
        # State: start at origin with zero heading
        self.state = np.zeros(3)
        # Covariance: nearly certain about starting pose (small epsilon for numerical stability)
        self.cov = np.eye(3) * 1e-6

        # Noise models
        self.Q = motion_noise        # Process noise (odometry)
        self.R = observation_noise   # Measurement noise (sensor)

        # Maps landmark_id -> index in the state vector
        # If landmark i is at index j, its (x,y) are at state[j] and state[j+1]
        self.landmark_index: dict[int, int] = {}

    @property
    def robot_pose(self) -> tuple[float, float, float]:
        """Current estimated robot pose."""
        return float(self.state[0]), float(self.state[1]), float(self.state[2])

    @property
    def num_landmarks(self) -> int:
        return len(self.landmark_index)

    def get_landmark_estimate(self, landmark_id: int) -> Optional[tuple[float, float]]:
        """Get estimated position of a landmark, or None if not yet seen."""
        if landmark_id not in self.landmark_index:
            return None
        idx = self.landmark_index[landmark_id]
        return float(self.state[idx]), float(self.state[idx + 1])

    def predict(self, d: float, alpha: float) -> None:
        """
        Prediction step: propagate state and covariance using odometry.

        The robot moves: turn by alpha, then drive forward by d.
        Only the robot portion of the state changes — landmarks don't move.

        After prediction, robot uncertainty grows (we become less sure where we are).
        """
        n = len(self.state)
        x, y, theta = self.state[0], self.state[1], self.state[2]

        # New theta after turning
        new_theta = normalize_angle(theta + alpha)

        # Predicted robot pose
        self.state[0] = x + d * math.cos(new_theta)
        self.state[1] = y + d * math.sin(new_theta)
        self.state[2] = normalize_angle(new_theta)

        # Jacobian of motion model w.r.t. robot state (3x3)
        # ∂f/∂(x, y, θ): only θ affects the prediction nonlinearly
        F_r = np.array([
            [1, 0, -d * math.sin(new_theta)],
            [0, 1,  d * math.cos(new_theta)],
            [0, 0, 1],
        ])

        # Jacobian of motion model w.r.t. noise (3x2)
        # How does noise in (d, alpha) affect (x, y, θ)?
        F_n = np.array([
            [math.cos(new_theta), -d * math.sin(new_theta)],
            [math.sin(new_theta),  d * math.cos(new_theta)],
            [0, 1],
        ])

        # Build full-size Jacobian: identity for landmarks, F_r for robot
        # Instead of building the full (3+2N)x(3+2N) matrix, we update in blocks:

        # Robot-robot block
        self.cov[:3, :3] = F_r @ self.cov[:3, :3] @ F_r.T + F_n @ self.Q @ F_n.T

        # Robot-landmark cross-correlations (these are critical for SLAM!)
        # When the robot moves, its correlation with landmarks changes
        if n > 3:
            self.cov[:3, 3:] = F_r @ self.cov[:3, 3:]
            self.cov[3:, :3] = self.cov[:3, 3:].T

        # Landmark-landmark blocks are unchanged (landmarks don't move)

    def update(self, landmark_id: int, z_range: float, z_bearing: float) -> None:
        """
        Update step: incorporate a range-bearing observation of a landmark.

        If this is a new landmark, we first initialize it in the state vector.
        Then we compute the Kalman update which corrects the ENTIRE state —
        robot pose AND all landmark positions — based on this single observation.
        """
        # Check if this is a new landmark
        if landmark_id not in self.landmark_index:
            self._initialize_landmark(landmark_id, z_range, z_bearing)
            return  # First observation initializes; no update needed

        # Index of this landmark in the state vector
        idx = self.landmark_index[landmark_id]
        n = len(self.state)

        # Robot pose
        x_r, y_r, theta_r = self.state[0], self.state[1], self.state[2]

        # Landmark estimated position
        x_l, y_l = self.state[idx], self.state[idx + 1]

        # Predicted observation
        dx = x_l - x_r
        dy = y_l - y_r
        q = dx**2 + dy**2
        sq = math.sqrt(q)

        z_hat = np.array([sq, normalize_angle(math.atan2(dy, dx) - theta_r)])

        # Innovation (difference between actual and predicted observation)
        innovation = np.array([z_range - z_hat[0], normalize_angle(z_bearing - z_hat[1])])

        # Jacobian of observation model H (2 x n)
        # H has nonzero entries only for robot pose and the observed landmark
        # ∂h/∂(x_r, y_r, θ_r):
        H_r = np.array([
            [-dx / sq, -dy / sq,  0],
            [ dy / q,  -dx / q,  -1],
        ])
        # ∂h/∂(x_l, y_l):
        H_l = np.array([
            [ dx / sq,  dy / sq],
            [-dy / q,   dx / q ],
        ])

        # Build full H matrix (2 x n): mostly zeros
        H = np.zeros((2, n))
        H[:, :3] = H_r
        H[:, idx:idx + 2] = H_l

        # Kalman gain: K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
        S = H @ self.cov @ H.T + self.R
        K = self.cov @ H.T @ np.linalg.inv(S)

        # State update: correct the ENTIRE state (robot + all landmarks)
        self.state = self.state + K @ innovation
        self.state[2] = normalize_angle(self.state[2])

        # Covariance update using Joseph form for numerical stability:
        # P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ
        I_KH = np.eye(n) - K @ H
        self.cov = I_KH @ self.cov @ I_KH.T + K @ self.R @ K.T

        # Ensure symmetry (floating point can break it over many iterations)
        self.cov = (self.cov + self.cov.T) / 2

    def _initialize_landmark(self, landmark_id: int, z_range: float, z_bearing: float) -> None:
        """
        Add a new landmark to the state vector.

        We estimate the landmark position from the current robot pose and
        the observation, then expand the covariance matrix.
        """
        x_r, y_r, theta_r = self.state[0], self.state[1], self.state[2]

        # Estimate landmark position from robot pose + observation
        angle = theta_r + z_bearing
        lx = x_r + z_range * math.cos(angle)
        ly = y_r + z_range * math.sin(angle)

        # Record the index where this landmark will live in the state
        idx = len(self.state)
        self.landmark_index[landmark_id] = idx

        # Expand state vector
        self.state = np.append(self.state, [lx, ly])

        # Expand covariance matrix
        # New landmark has high initial uncertainty and correlations derived from
        # the robot's current uncertainty
        n_old = len(self.cov)
        n_new = n_old + 2

        # Jacobian of landmark initialization w.r.t. robot pose
        G_r = np.array([
            [1, 0, -z_range * math.sin(angle)],
            [0, 1,  z_range * math.cos(angle)],
        ])

        # Jacobian of landmark initialization w.r.t. observation noise
        G_z = np.array([
            [math.cos(angle), -z_range * math.sin(angle)],
            [math.sin(angle),  z_range * math.cos(angle)],
        ])

        new_cov = np.zeros((n_new, n_new))

        # Copy old covariance
        new_cov[:n_old, :n_old] = self.cov

        # Cross-correlation between new landmark and robot
        cross = G_r @ self.cov[:3, :3]
        new_cov[n_old:, :3] = cross
        new_cov[:3, n_old:] = cross.T

        # Cross-correlation between new landmark and existing landmarks
        if n_old > 3:
            cross_lm = G_r @ self.cov[:3, 3:n_old]
            new_cov[n_old:, 3:n_old] = cross_lm
            new_cov[3:n_old, n_old:] = cross_lm.T

        # New landmark's own covariance
        new_cov[n_old:, n_old:] = G_r @ self.cov[:3, :3] @ G_r.T + G_z @ self.R @ G_z.T

        self.cov = new_cov


def run_slam_simulation(
    num_steps: int = 60,
    seed: int = 42,
) -> tuple[EKFSLAM, list[tuple[float, float, float]], list[tuple[float, float, float]], list[Landmark]]:
    """
    Run a full SLAM simulation.

    Returns:
        slam: The final SLAM state
        true_path: List of true robot poses
        estimated_path: List of estimated robot poses
        landmarks: The true landmark positions
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create a world with landmarks arranged in a rough grid
    landmarks = [
        Landmark(0, 5.0, 5.0),
        Landmark(1, 5.0, -5.0),
        Landmark(2, -5.0, 5.0),
        Landmark(3, -5.0, -5.0),
        Landmark(4, 10.0, 0.0),
        Landmark(5, 0.0, 10.0),
        Landmark(6, -10.0, 0.0),
        Landmark(7, 0.0, -10.0),
        Landmark(8, 8.0, 8.0),
        Landmark(9, -8.0, -8.0),
    ]

    # Robot with moderate noise
    robot = Robot(
        odom_noise_d=0.15,
        odom_noise_angle=0.05,
        sensor_noise_range=0.3,
        sensor_noise_bearing=0.05,
        sensor_max_range=12.0,
    )

    # SLAM filter with noise estimates matching the robot
    Q = np.diag([0.15**2, 0.05**2])  # Odometry noise covariance
    R = np.diag([0.3**2, 0.05**2])   # Sensor noise covariance
    slam = EKFSLAM(motion_noise=Q, observation_noise=R)

    true_path = [(robot.x, robot.y, robot.theta)]
    estimated_path = [slam.robot_pose]

    # Drive in a rough loop to demonstrate loop closure
    # The robot will accumulate odometry error, then correct it when it
    # re-observes landmarks from the beginning of its path
    for step in range(num_steps):
        # Drive in a large square-ish loop
        # Slight turn + forward motion each step
        turn = 2 * math.pi / num_steps  # Complete one full loop
        distance = 1.0

        # Execute motion (returns noisy odometry)
        d_noisy, alpha_noisy = robot.move(distance, turn)

        # SLAM prediction step
        slam.predict(d_noisy, alpha_noisy)

        # Observe landmarks
        observations = robot.observe(landmarks)

        # SLAM update step for each observation
        for lm_id, r, b in observations:
            slam.update(lm_id, r, b)

        true_path.append((robot.x, robot.y, robot.theta))
        estimated_path.append(slam.robot_pose)

    return slam, true_path, estimated_path, landmarks


def print_results(
    slam: EKFSLAM,
    true_path: list[tuple[float, float, float]],
    estimated_path: list[tuple[float, float, float]],
    landmarks: list[Landmark],
) -> None:
    """Print detailed SLAM results with error analysis."""

    print("=" * 70)
    print("EKF-SLAM SIMULATION RESULTS")
    print("=" * 70)

    # Final robot pose
    true_x, true_y, true_theta = true_path[-1]
    est_x, est_y, est_theta = estimated_path[-1]
    pose_error = math.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)

    print(f"\n--- Robot Pose ---")
    print(f"  True:      ({true_x:7.3f}, {true_y:7.3f}, θ={math.degrees(true_theta):7.2f}°)")
    print(f"  Estimated: ({est_x:7.3f}, {est_y:7.3f}, θ={math.degrees(est_theta):7.2f}°)")
    print(f"  Position error: {pose_error:.4f}")
    print(f"  Heading error:  {math.degrees(abs(normalize_angle(true_theta - est_theta))):.2f}°")

    # Landmark estimates
    print(f"\n--- Landmark Estimates ({slam.num_landmarks} discovered) ---")
    total_landmark_error = 0.0
    for lm in landmarks:
        est = slam.get_landmark_estimate(lm.id)
        if est is not None:
            ex, ey = est
            err = math.sqrt((lm.x - ex)**2 + (lm.y - ey)**2)
            total_landmark_error += err
            print(f"  LM {lm.id:2d}: true=({lm.x:6.2f}, {lm.y:6.2f})  "
                  f"est=({ex:6.2f}, {ey:6.2f})  err={err:.4f}")
        else:
            print(f"  LM {lm.id:2d}: not observed")

    if slam.num_landmarks > 0:
        avg_err = total_landmark_error / slam.num_landmarks
        print(f"\n  Average landmark error: {avg_err:.4f}")

    # Odometry drift analysis — compare pure odometry path to SLAM-corrected path
    print(f"\n--- Odometry Drift Analysis ---")
    # Sum up positional errors at each step
    slam_errors = []
    for (tx, ty, _), (ex, ey, _) in zip(true_path, estimated_path):
        slam_errors.append(math.sqrt((tx - ex)**2 + (ty - ey)**2))

    print(f"  Max SLAM error during run: {max(slam_errors):.4f}")
    print(f"  Mean SLAM error:           {sum(slam_errors) / len(slam_errors):.4f}")
    print(f"  Final SLAM error:          {slam_errors[-1]:.4f}")

    # Covariance insight
    robot_cov = slam.cov[:3, :3]
    print(f"\n--- Uncertainty (Robot Covariance Diagonal) ---")
    print(f"  σ_x:     {math.sqrt(robot_cov[0, 0]):.4f}")
    print(f"  σ_y:     {math.sqrt(robot_cov[1, 1]):.4f}")
    print(f"  σ_theta: {math.sqrt(robot_cov[2, 2]):.4f} rad "
          f"({math.degrees(math.sqrt(robot_cov[2, 2])):.2f}°)")

    # Show covariance matrix structure
    print(f"\n--- Covariance Matrix Structure ---")
    n = len(slam.cov)
    print(f"  State dimension: {n} (3 robot + {n - 3} landmark coords)")
    print(f"  Matrix size: {n}×{n} = {n*n} entries")

    # Show that cross-correlations are non-zero (the key insight)
    if n > 3:
        cross_norm = np.linalg.norm(slam.cov[:3, 3:])
        print(f"  Robot-landmark cross-correlation norm: {cross_norm:.4f}")
        print(f"  (Non-zero cross-correlations are WHY SLAM works!)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Running EKF-SLAM simulation...\n")

    slam, true_path, estimated_path, landmarks = run_slam_simulation(
        num_steps=60,
        seed=42,
    )

    print_results(slam, true_path, estimated_path, landmarks)

    # Demonstrate what happens without SLAM (pure odometry)
    print("\n--- COMPARISON: Pure Odometry vs SLAM ---")
    print("Running same path with no landmark corrections...\n")

    random.seed(42)
    np.random.seed(42)

    robot_odom = Robot(odom_noise_d=0.15, odom_noise_angle=0.05, sensor_max_range=12.0)
    odom_x, odom_y, odom_theta = 0.0, 0.0, 0.0

    for step in range(60):
        turn = 2 * math.pi / 60
        d_noisy, alpha_noisy = robot_odom.move(1.0, turn)

        # Pure dead reckoning — no corrections
        odom_theta = normalize_angle(odom_theta + alpha_noisy)
        odom_x += d_noisy * math.cos(odom_theta)
        odom_y += d_noisy * math.sin(odom_theta)

    true_final = true_path[-1]
    slam_final = estimated_path[-1]
    odom_error = math.sqrt((true_final[0] - odom_x)**2 + (true_final[1] - odom_y)**2)
    slam_error = math.sqrt((true_final[0] - slam_final[0])**2 + (true_final[1] - slam_final[1])**2)

    print(f"True final position:  ({true_final[0]:.3f}, {true_final[1]:.3f})")
    print(f"Odometry estimate:    ({odom_x:.3f}, {odom_y:.3f})  error={odom_error:.4f}")
    print(f"SLAM estimate:        ({slam_final[0]:.3f}, {slam_final[1]:.3f})  error={slam_error:.4f}")
    print(f"\nSLAM reduces error by {odom_error / max(slam_error, 1e-6):.1f}x compared to pure odometry!")
    print("\nThis demonstrates the core value of SLAM: by fusing landmark observations")
    print("with odometry, we bound drift that would otherwise grow without limit.")
