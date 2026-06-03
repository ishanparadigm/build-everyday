"""
Tests for Day 063: Sensor Fusion (IMU + GPS)

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    SensorConfig,
    generate_ground_truth,
    simulate_imu,
    simulate_gps,
    dead_reckoning,
    KalmanFusionFilter,
    run_sensor_fusion,
)


class TestGroundTruth(unittest.TestCase):
    """Test ground truth trajectory generation."""

    def test_output_shapes(self):
        """Ground truth arrays should have correct shapes."""
        config = SensorConfig(duration_seconds=10.0, imu_rate_hz=100.0)
        times, pos, vel, accel = generate_ground_truth(config)
        n = int(config.duration_seconds * config.imu_rate_hz)
        self.assertEqual(times.shape, (n,))
        self.assertEqual(pos.shape, (n, 2))
        self.assertEqual(vel.shape, (n, 2))
        self.assertEqual(accel.shape, (n, 2))

    def test_starts_at_origin(self):
        """Trajectory should start at position (0, 0) with zero velocity."""
        config = SensorConfig(duration_seconds=5.0)
        _, pos, vel, _ = generate_ground_truth(config)
        np.testing.assert_array_almost_equal(pos[0], [0, 0])
        np.testing.assert_array_almost_equal(vel[0], [0, 0])

    def test_position_changes(self):
        """Robot should actually move — final position should differ from start."""
        config = SensorConfig(duration_seconds=10.0)
        _, pos, _, _ = generate_ground_truth(config)
        displacement = np.linalg.norm(pos[-1] - pos[0])
        self.assertGreater(displacement, 1.0)


class TestIMUSimulation(unittest.TestCase):
    """Test IMU noise simulation."""

    def test_output_shape(self):
        """IMU readings should match input shape."""
        accel = np.zeros((1000, 2))
        config = SensorConfig()
        rng = np.random.default_rng(42)
        imu = simulate_imu(accel, config, rng)
        self.assertEqual(imu.shape, accel.shape)

    def test_noise_is_added(self):
        """IMU readings should differ from true accelerations."""
        accel = np.zeros((1000, 2))
        config = SensorConfig(imu_accel_noise_std=1.0)
        rng = np.random.default_rng(42)
        imu = simulate_imu(accel, config, rng)
        # With zero true accel, readings should be noise + bias
        self.assertGreater(np.std(imu), 0.1)

    def test_bias_present(self):
        """Mean of IMU readings should reflect the constant bias."""
        accel = np.zeros((10000, 2))
        config = SensorConfig(imu_accel_noise_std=0.1, imu_accel_bias=np.array([1.0, -0.5]))
        rng = np.random.default_rng(42)
        imu = simulate_imu(accel, config, rng)
        # Mean should be close to bias (noise averages out)
        np.testing.assert_almost_equal(np.mean(imu[:, 0]), 1.0, decimal=1)
        np.testing.assert_almost_equal(np.mean(imu[:, 1]), -0.5, decimal=1)


class TestGPSSimulation(unittest.TestCase):
    """Test GPS noise simulation."""

    def test_readings_at_lower_rate(self):
        """GPS should produce fewer readings than IMU."""
        config = SensorConfig(duration_seconds=10.0, imu_rate_hz=100.0, gps_rate_hz=1.0, gps_dropout_prob=0.0)
        times = np.arange(1000) * 0.01
        positions = np.zeros((1000, 2))
        rng = np.random.default_rng(42)
        gps = simulate_gps(times, positions, config, rng)
        # Should get ~10 readings for 10s at 1Hz (no dropouts)
        self.assertGreaterEqual(len(gps), 8)
        self.assertLessEqual(len(gps), 12)

    def test_noise_added_to_gps(self):
        """GPS readings should be noisy versions of true position."""
        config = SensorConfig(duration_seconds=100.0, gps_rate_hz=1.0, gps_dropout_prob=0.0,
                              gps_position_noise_std=5.0)
        n = int(config.duration_seconds * config.imu_rate_hz)
        times = np.arange(n) / config.imu_rate_hz
        positions = np.ones((n, 2)) * 100.0  # Constant position
        rng = np.random.default_rng(42)
        gps = simulate_gps(times, positions, config, rng)
        # Check noise std is approximately correct
        gps_positions = np.array([g[1] for g in gps])
        errors = gps_positions - 100.0
        self.assertAlmostEqual(np.std(errors), 5.0, delta=2.0)


class TestDeadReckoning(unittest.TestCase):
    """Test IMU-only integration."""

    def test_zero_acceleration(self):
        """With zero acceleration, position should stay at origin."""
        imu = np.zeros((1000, 2))
        pos, vel = dead_reckoning(imu, dt=0.01)
        np.testing.assert_array_almost_equal(pos[-1], [0, 0])

    def test_constant_acceleration(self):
        """With constant acceleration, position should follow s = 0.5*a*t^2."""
        n = 1000
        dt = 0.01
        accel = np.ones((n, 2)) * 2.0  # 2 m/s^2 in both axes
        pos, vel = dead_reckoning(accel, dt)
        t = (n - 1) * dt
        expected_pos = 0.5 * 2.0 * t ** 2
        # Allow small numerical error from discrete integration
        np.testing.assert_almost_equal(pos[-1, 0], expected_pos, decimal=0)

    def test_drift_grows_with_bias(self):
        """A constant bias should cause quadratically growing error."""
        n = 5000
        dt = 0.01
        true_accel = np.zeros((n, 2))
        biased_accel = true_accel + np.array([0.1, 0.0])  # 0.1 m/s^2 bias
        pos, _ = dead_reckoning(biased_accel, dt)
        # Error at end should be ~0.5 * 0.1 * 50^2 = 125m
        self.assertGreater(np.abs(pos[-1, 0]), 50)


class TestKalmanFilter(unittest.TestCase):
    """Test the Kalman filter implementation."""

    def test_predict_moves_state(self):
        """Prediction with acceleration should change state."""
        kf = KalmanFusionFilter(
            initial_state=np.array([0, 0, 1, 0]),
            initial_covariance=np.eye(4),
            gps_noise_std=3.0
        )
        state = kf.predict(0.1, np.array([1.0, 0.0]))
        # Position should have moved: px = 0 + 1*0.1 + 0.5*1*0.01 = 0.105
        self.assertGreater(state[0], 0)

    def test_update_corrects_state(self):
        """GPS update should pull state toward measurement."""
        kf = KalmanFusionFilter(
            initial_state=np.array([0, 0, 0, 0]),
            initial_covariance=np.eye(4) * 100,  # Very uncertain
            gps_noise_std=1.0  # Precise GPS
        )
        state, K = kf.update(np.array([10.0, 5.0]))
        # State should move significantly toward GPS reading
        self.assertGreater(state[0], 5.0)
        self.assertGreater(state[1], 2.0)

    def test_covariance_shrinks_on_update(self):
        """Covariance should decrease after GPS update."""
        kf = KalmanFusionFilter(
            initial_state=np.zeros(4),
            initial_covariance=np.eye(4) * 10,
            gps_noise_std=3.0
        )
        trace_before = np.trace(kf.P)
        kf.update(np.array([1.0, 1.0]))
        trace_after = np.trace(kf.P)
        self.assertLess(trace_after, trace_before)


class TestFullPipeline(unittest.TestCase):
    """Test the complete sensor fusion pipeline."""

    def test_fusion_beats_dead_reckoning(self):
        """Kalman fused estimate should be more accurate than IMU-only."""
        config = SensorConfig(duration_seconds=30.0)
        results = run_sensor_fusion(config)
        self.assertLess(results['fused_rms'], results['dr_rms'])

    def test_result_keys(self):
        """Pipeline should return all expected keys."""
        config = SensorConfig(duration_seconds=10.0)
        results = run_sensor_fusion(config)
        expected_keys = ['times', 'true_positions', 'dr_positions',
                         'fused_positions', 'dr_rms', 'fused_rms',
                         'gps_readings']
        for key in expected_keys:
            self.assertIn(key, results)


if __name__ == "__main__":
    unittest.main()
