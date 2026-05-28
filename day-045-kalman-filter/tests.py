"""
Day 045: Kalman Filter Tests

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    KalmanFilter,
    simulate_1d_tracking,
    run_kalman_filter,
    compute_rmse,
)


class TestKalmanFilter(unittest.TestCase):
    """Test the Kalman filter implementation."""

    def setUp(self):
        """Set up a simple 1D position-only filter for basic tests."""
        # Simplest possible system: static position with noisy measurements
        self.F = np.array([[1.0]])  # state doesn't change
        self.H = np.array([[1.0]])  # measure the state directly
        self.Q = np.array([[0.01]])  # tiny process noise
        self.R = np.array([[1.0]])  # measurement noise variance = 1
        self.x0 = np.array([0.0])
        self.P0 = np.array([[100.0]])  # very uncertain initially

    def test_predict_increases_uncertainty(self):
        """After prediction, covariance should grow (P_pred > P_prior)."""
        kf = KalmanFilter(self.F, self.H, self.Q, self.R, self.x0, self.P0)
        P_before = kf.P[0, 0]
        kf.predict()
        P_after = kf.P[0, 0]
        self.assertGreater(P_after, P_before - 0.001,
                          "Prediction should not decrease uncertainty significantly")

    def test_update_decreases_uncertainty(self):
        """After update, covariance should shrink (P_post < P_pred)."""
        kf = KalmanFilter(self.F, self.H, self.Q, self.R, self.x0, self.P0)
        kf.predict()
        P_predicted = kf.P[0, 0]
        kf.update(np.array([1.0]))
        P_updated = kf.P[0, 0]
        self.assertLess(P_updated, P_predicted,
                       "Update should decrease uncertainty")

    def test_filter_converges_to_measurements(self):
        """With repeated measurements at a fixed value, estimate should converge."""
        kf = KalmanFilter(self.F, self.H, self.Q, self.R, self.x0, self.P0)
        target = 5.0
        for _ in range(50):
            kf.predict()
            kf.update(np.array([target]))
        # After many measurements, should be close to the true value
        self.assertAlmostEqual(kf.x[0], target, places=1,
                              msg="Filter should converge to repeated measurements")

    def test_kalman_gain_decreases_over_time(self):
        """Kalman gain should decrease as the filter becomes more confident."""
        kf = KalmanFilter(self.F, self.H, self.Q, self.R, self.x0, self.P0)
        gains = []
        for _ in range(20):
            kf.predict()
            kf.update(np.array([1.0]))
            K = kf.get_kalman_gain()
            gains.append(K[0, 0])
        # First gain should be larger than last (converging)
        self.assertGreater(gains[0], gains[-1],
                          "Kalman gain should decrease as filter gains confidence")

    def test_high_measurement_noise_reduces_gain(self):
        """Higher measurement noise → lower Kalman gain (trust model more)."""
        # Low noise sensor
        kf_low = KalmanFilter(self.F, self.H, self.Q,
                              np.array([[0.1]]), self.x0, self.P0)
        kf_low.predict()
        kf_low.update(np.array([1.0]))
        K_low = kf_low.get_kalman_gain()[0, 0]

        # High noise sensor
        kf_high = KalmanFilter(self.F, self.H, self.Q,
                               np.array([[100.0]]), self.x0, self.P0)
        kf_high.predict()
        kf_high.update(np.array([1.0]))
        K_high = kf_high.get_kalman_gain()[0, 0]

        self.assertGreater(K_low, K_high,
                          "Lower measurement noise should give higher Kalman gain")

    def test_2d_constant_velocity(self):
        """Test the full constant-velocity model tracks position and velocity."""
        dt = 0.1
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = 0.01 * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
        R = np.array([[1.0]])
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * 10.0

        kf = KalmanFilter(F, H, Q, R, x0, P0)

        # Feed measurements from object moving at velocity=3
        true_vel = 3.0
        for k in range(1, 100):
            true_pos = true_vel * k * dt
            kf.predict()
            kf.update(np.array([true_pos + np.random.normal(0, 0.5)]))

        # Position should be close to true value
        expected_pos = true_vel * 100 * dt
        self.assertAlmostEqual(kf.x[0], expected_pos, delta=2.0,
                              msg="Position estimate should be close to truth")
        # Velocity should be estimated (even though we only measure position!)
        self.assertAlmostEqual(kf.x[1], true_vel, delta=1.0,
                              msg="Velocity should be inferred from position changes")


class TestSimulation(unittest.TestCase):
    """Test the simulation and end-to-end pipeline."""

    def test_simulation_shapes(self):
        """Simulation should return correct array shapes."""
        states, meas, times = simulate_1d_tracking(n_steps=50, dt=0.1)
        self.assertEqual(states.shape, (50, 2))
        self.assertEqual(meas.shape, (50,))
        self.assertEqual(times.shape, (50,))

    def test_simulation_deterministic(self):
        """Same seed should give same results."""
        s1, m1, _ = simulate_1d_tracking(seed=123)
        s2, m2, _ = simulate_1d_tracking(seed=123)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(m1, m2)

    def test_kalman_filter_improves_over_raw(self):
        """Kalman filter RMSE should be lower than raw measurement RMSE."""
        states, meas, _ = simulate_1d_tracking(
            n_steps=200, measurement_noise_std=2.0, seed=42
        )
        _, estimates, _ = run_kalman_filter(
            meas, process_noise_std=0.3, measurement_noise_std=2.0
        )
        raw_rmse = compute_rmse(meas, states[:, 0])
        kf_rmse = compute_rmse(estimates[:, 0], states[:, 0])
        self.assertLess(kf_rmse, raw_rmse,
                       "Kalman filter should outperform raw measurements")

    def test_compute_rmse(self):
        """RMSE of identical arrays should be 0."""
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(compute_rmse(a, a), 0.0)
        # Known RMSE
        b = np.array([2.0, 3.0, 4.0])
        self.assertAlmostEqual(compute_rmse(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()
