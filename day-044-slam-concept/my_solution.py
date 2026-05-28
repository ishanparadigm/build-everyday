"""
Day 044: SLAM Concept Implementation — EKF-SLAM (Your Implementation)

Implement Simultaneous Localization and Mapping using the Extended Kalman Filter.

Key concepts to implement:
1. State vector: [x_r, y_r, θ_r, x_L1, y_L1, ..., x_Ln, y_Ln]
2. Prediction step: propagate robot pose + grow uncertainty
3. Update step: correct entire state from a single landmark observation
4. Landmark initialization: expand state when new landmarks are discovered
"""

import math
import random
import numpy as np
from typing import Optional


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
        self.x = x
        self.y = y
        self.theta = theta
        self.odom_noise_d = odom_noise_d
        self.odom_noise_angle = odom_noise_angle
        self.sensor_noise_range = sensor_noise_range
        self.sensor_noise_bearing = sensor_noise_bearing
        self.sensor_max_range = sensor_max_range

    def move(self, d: float, alpha: float) -> tuple[float, float]:
        """
        Move the robot: turn by alpha, then drive forward by d.
        Returns noisy odometry (d_noisy, alpha_noisy).
        """
        self.theta = normalize_angle(self.theta + alpha)
        self.x += d * math.cos(self.theta)
        self.y += d * math.sin(self.theta)

        d_noisy = d + random.gauss(0, self.odom_noise_d)
        alpha_noisy = alpha + random.gauss(0, self.odom_noise_angle)
        return d_noisy, alpha_noisy

    def observe(self, landmarks: list[Landmark]) -> list[tuple[int, float, float]]:
        """
        Observe all landmarks within sensor range.
        Returns list of (landmark_id, range, bearing) with noise.
        """
        observations = []
        for lm in landmarks:
            dx = lm.x - self.x
            dy = lm.y - self.y
            true_range = math.sqrt(dx**2 + dy**2)

            if true_range <= self.sensor_max_range:
                true_bearing = math.atan2(dy, dx) - self.theta
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

    Hints:
    - The state grows dynamically as new landmarks are discovered
    - Jacobians are the key to propagating uncertainty correctly
    - Cross-correlations between robot and landmarks are what make SLAM work
    """

    def __init__(
        self,
        motion_noise: np.ndarray,
        observation_noise: np.ndarray,
    ):
        """
        Args:
            motion_noise: 2x2 diagonal matrix [σ²_d, σ²_α]
            observation_noise: 2x2 diagonal matrix [σ²_range, σ²_bearing]
        """
        raise NotImplementedError("TODO: Initialize state vector, covariance, noise models, and landmark index")

    @property
    def robot_pose(self) -> tuple[float, float, float]:
        """Current estimated robot pose."""
        raise NotImplementedError("TODO: Return (x, y, theta) from state vector")

    @property
    def num_landmarks(self) -> int:
        raise NotImplementedError("TODO: Return number of discovered landmarks")

    def get_landmark_estimate(self, landmark_id: int) -> Optional[tuple[float, float]]:
        """Get estimated position of a landmark, or None if not yet seen."""
        raise NotImplementedError("TODO: Look up landmark in state vector by its ID")

    def predict(self, d: float, alpha: float) -> None:
        """
        Prediction step: propagate state and covariance using odometry.

        Hints:
        - Update robot pose: turn by alpha, then drive forward by d
        - Compute the Jacobian F_r (3x3) of the motion model w.r.t. robot state
        - Compute the Jacobian F_n (3x2) of the motion model w.r.t. noise
        - Update covariance: robot-robot block, robot-landmark cross-correlations
        - Landmarks don't move, so their blocks stay the same
        """
        raise NotImplementedError("TODO: Implement EKF prediction step")

    def update(self, landmark_id: int, z_range: float, z_bearing: float) -> None:
        """
        Update step: incorporate a range-bearing observation.

        Hints:
        - If new landmark, call _initialize_landmark and return
        - Compute predicted observation (expected range and bearing)
        - Innovation = actual - predicted (don't forget to normalize the bearing!)
        - Build the Jacobian H (2×n) with nonzero blocks for robot and observed landmark
        - Kalman gain K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
        - State update: x = x + K·innovation
        - Covariance update: P = (I - K·H)·P
        """
        raise NotImplementedError("TODO: Implement EKF update step")

    def _initialize_landmark(self, landmark_id: int, z_range: float, z_bearing: float) -> None:
        """
        Add a new landmark to the state vector and expand covariance.

        Hints:
        - Estimate landmark (x, y) from robot pose + observation
        - Expand state vector with np.append
        - Expand covariance: compute Jacobians G_r and G_z for the initialization
        - New landmark covariance = G_r·P_robot·G_rᵀ + G_z·R·G_zᵀ
        - Cross-correlations: G_r·P[:3, :] connects new landmark to everything
        """
        raise NotImplementedError("TODO: Initialize new landmark in state and covariance")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    landmarks = [
        Landmark(0, 5.0, 5.0),
        Landmark(1, 5.0, -5.0),
        Landmark(2, -5.0, 5.0),
        Landmark(3, -5.0, -5.0),
        Landmark(4, 10.0, 0.0),
    ]

    robot = Robot(odom_noise_d=0.15, odom_noise_angle=0.05,
                  sensor_noise_range=0.3, sensor_noise_bearing=0.05,
                  sensor_max_range=12.0)

    Q = np.diag([0.15**2, 0.05**2])
    R = np.diag([0.3**2, 0.05**2])
    slam = EKFSLAM(motion_noise=Q, observation_noise=R)

    print("Running EKF-SLAM...")
    for step in range(60):
        turn = 2 * math.pi / 60
        d_noisy, alpha_noisy = robot.move(1.0, turn)
        slam.predict(d_noisy, alpha_noisy)

        observations = robot.observe(landmarks)
        for lm_id, r, b in observations:
            slam.update(lm_id, r, b)

        if step % 10 == 0:
            ex, ey, etheta = slam.robot_pose
            print(f"  Step {step:3d}: est=({ex:.2f}, {ey:.2f}), "
                  f"true=({robot.x:.2f}, {robot.y:.2f}), "
                  f"landmarks found: {slam.num_landmarks}")

    print(f"\nFinal robot pose: {slam.robot_pose}")
    print(f"True robot pose: ({robot.x:.3f}, {robot.y:.3f}, {robot.theta:.3f})")
    print(f"\nLandmark estimates:")
    for lm in landmarks:
        est = slam.get_landmark_estimate(lm.id)
        if est:
            print(f"  LM {lm.id}: true=({lm.x:.2f}, {lm.y:.2f}), est=({est[0]:.2f}, {est[1]:.2f})")
