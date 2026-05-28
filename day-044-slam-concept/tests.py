"""
Tests for EKF-SLAM implementation.

Run with: python3 -m pytest tests.py -v
    or:   python3 tests.py
"""

import math
import random
import unittest
import numpy as np
from my_solution import EKFSLAM, Robot, Landmark, normalize_angle


class TestNormalizeAngle(unittest.TestCase):
    """Test angle normalization utility."""

    def test_already_normalized(self):
        self.assertAlmostEqual(normalize_angle(0.5), 0.5)

    def test_positive_wrap(self):
        result = normalize_angle(3 * math.pi)
        self.assertGreaterEqual(result, -math.pi)
        self.assertLessEqual(result, math.pi)
        self.assertAlmostEqual(result, math.pi, places=5)

    def test_negative_wrap(self):
        result = normalize_angle(-3 * math.pi)
        self.assertGreaterEqual(result, -math.pi)
        self.assertLessEqual(result, math.pi)


class TestEKFSLAMInit(unittest.TestCase):
    """Test SLAM initialization."""

    def setUp(self):
        Q = np.diag([0.1**2, 0.05**2])
        R = np.diag([0.2**2, 0.05**2])
        self.slam = EKFSLAM(motion_noise=Q, observation_noise=R)

    def test_initial_pose_at_origin(self):
        x, y, theta = self.slam.robot_pose
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(theta, 0.0)

    def test_no_initial_landmarks(self):
        self.assertEqual(self.slam.num_landmarks, 0)

    def test_unknown_landmark_returns_none(self):
        self.assertIsNone(self.slam.get_landmark_estimate(99))


class TestPrediction(unittest.TestCase):
    """Test the EKF prediction (motion) step."""

    def setUp(self):
        Q = np.diag([0.1**2, 0.05**2])
        R = np.diag([0.2**2, 0.05**2])
        self.slam = EKFSLAM(motion_noise=Q, observation_noise=R)

    def test_straight_line_motion(self):
        """Moving forward should increase x, not y."""
        self.slam.predict(1.0, 0.0)  # Drive 1m forward, no turn
        x, y, theta = self.slam.robot_pose
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)
        self.assertAlmostEqual(theta, 0.0, places=5)

    def test_turn_then_drive(self):
        """Turn 90° left then drive should increase y."""
        self.slam.predict(1.0, math.pi / 2)
        x, y, theta = self.slam.robot_pose
        self.assertAlmostEqual(x, 0.0, places=4)
        self.assertAlmostEqual(y, 1.0, places=4)

    def test_covariance_grows(self):
        """Uncertainty should increase after motion (no observations)."""
        cov_before = self.slam.cov.copy()
        self.slam.predict(1.0, 0.1)
        # Robot uncertainty should grow
        self.assertGreater(self.slam.cov[0, 0], cov_before[0, 0])


class TestLandmarkInitialization(unittest.TestCase):
    """Test adding new landmarks."""

    def setUp(self):
        Q = np.diag([0.1**2, 0.05**2])
        R = np.diag([0.2**2, 0.05**2])
        self.slam = EKFSLAM(motion_noise=Q, observation_noise=R)

    def test_first_landmark_added(self):
        """Observing a new landmark should add it to the state."""
        # Observe landmark at range=5, bearing=0 (straight ahead)
        self.slam.update(0, 5.0, 0.0)
        self.assertEqual(self.slam.num_landmarks, 1)
        est = self.slam.get_landmark_estimate(0)
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est[0], 5.0, places=3)
        self.assertAlmostEqual(est[1], 0.0, places=3)

    def test_multiple_landmarks(self):
        """Should handle multiple landmarks correctly."""
        self.slam.update(0, 5.0, 0.0)
        self.slam.update(1, 3.0, math.pi / 2)
        self.assertEqual(self.slam.num_landmarks, 2)

    def test_state_vector_grows(self):
        """State vector should grow by 2 for each new landmark."""
        initial_size = len(self.slam.cov)
        self.slam.update(0, 5.0, 0.0)
        self.assertEqual(len(self.slam.cov), initial_size + 2)


class TestUpdate(unittest.TestCase):
    """Test the EKF update (observation) step."""

    def setUp(self):
        Q = np.diag([0.1**2, 0.05**2])
        R = np.diag([0.2**2, 0.05**2])
        self.slam = EKFSLAM(motion_noise=Q, observation_noise=R)

    def test_repeated_observations_reduce_uncertainty(self):
        """Observing the same landmark repeatedly should reduce its uncertainty."""
        # Initialize landmark
        self.slam.update(0, 5.0, 0.0)
        idx = self.slam.landmark_index[0]

        # Drive a tiny bit and re-observe
        self.slam.predict(0.1, 0.0)
        cov_before = self.slam.cov[idx, idx]
        self.slam.update(0, 4.9, 0.0)
        cov_after = self.slam.cov[idx, idx]

        self.assertLess(cov_after, cov_before)

    def test_slam_corrects_drift(self):
        """SLAM should maintain better pose estimates than pure odometry."""
        random.seed(123)
        np.random.seed(123)

        landmarks = [Landmark(0, 5.0, 0.0), Landmark(1, 0.0, 5.0)]
        robot = Robot(odom_noise_d=0.2, odom_noise_angle=0.1,
                      sensor_noise_range=0.3, sensor_noise_bearing=0.05,
                      sensor_max_range=15.0)

        Q = np.diag([0.2**2, 0.1**2])
        R = np.diag([0.3**2, 0.05**2])
        slam = EKFSLAM(motion_noise=Q, observation_noise=R)

        # Run 30 steps in a circle
        for _ in range(30):
            d_noisy, a_noisy = robot.move(0.5, 2 * math.pi / 30)
            slam.predict(d_noisy, a_noisy)
            for lm_id, r, b in robot.observe(landmarks):
                slam.update(lm_id, r, b)

        ex, ey, _ = slam.robot_pose
        slam_err = math.sqrt((robot.x - ex)**2 + (robot.y - ey)**2)

        # SLAM error should be reasonable (< 2m for this scenario)
        self.assertLess(slam_err, 2.0,
                        f"SLAM error too large: {slam_err:.3f}")


class TestCrossCorrelations(unittest.TestCase):
    """Test that cross-correlations are properly maintained."""

    def test_cross_correlations_nonzero(self):
        """After motion + observations, robot-landmark cross-correlations should be nonzero."""
        Q = np.diag([0.1**2, 0.05**2])
        R = np.diag([0.2**2, 0.05**2])
        slam = EKFSLAM(motion_noise=Q, observation_noise=R)

        # Add a landmark, move, re-observe
        slam.update(0, 5.0, 0.0)
        slam.predict(1.0, 0.1)
        slam.update(0, 4.2, -0.1)

        # Cross-correlation block should be nonzero
        cross = slam.cov[:3, 3:]
        self.assertGreater(np.linalg.norm(cross), 0.0,
                           "Cross-correlations should be non-zero after update")


if __name__ == "__main__":
    unittest.main()
