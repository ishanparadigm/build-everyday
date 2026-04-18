"""
Test suite for Sensor Reading Simulator.

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import random
import unittest

from my_solution import (
    Vec2,
    RobotState,
    LineSegment,
    LidarSensor,
    IMUSensor,
    OdometrySensor,
    fuse_gaussian_measurements,
)


class TestVec2(unittest.TestCase):
    def test_distance(self):
        a = Vec2(0, 0)
        b = Vec2(3, 4)
        self.assertAlmostEqual(a.distance_to(b), 5.0, places=5)

    def test_add_sub(self):
        a = Vec2(1, 2)
        b = Vec2(3, 4)
        c = a + b
        self.assertAlmostEqual(c.x, 4.0)
        self.assertAlmostEqual(c.y, 6.0)
        d = b - a
        self.assertAlmostEqual(d.x, 2.0)
        self.assertAlmostEqual(d.y, 2.0)


class TestSensorFusion(unittest.TestCase):
    """Tests for the Gaussian sensor fusion function."""

    def test_single_measurement(self):
        """Single measurement should return itself."""
        val, sigma = fuse_gaussian_measurements([(5.0, 0.1)])
        self.assertAlmostEqual(val, 5.0, places=5)
        self.assertAlmostEqual(sigma, 0.1, places=5)

    def test_equal_sensors(self):
        """Two identical-precision sensors: fused value = average, sigma = sigma/sqrt(2)."""
        val, sigma = fuse_gaussian_measurements([(4.0, 1.0), (6.0, 1.0)])
        self.assertAlmostEqual(val, 5.0, places=5)
        self.assertAlmostEqual(sigma, 1.0 / math.sqrt(2), places=5)

    def test_unequal_sensors(self):
        """Fused value should be closer to the more precise sensor."""
        val, sigma = fuse_gaussian_measurements([(5.0, 0.1), (6.0, 1.0)])
        # With 0.1 vs 1.0 sigma, the precise sensor has 100x the weight
        self.assertAlmostEqual(val, 5.0 + 1.0 / 101.0, places=3)
        # Fused sigma should be less than the best individual
        self.assertLess(sigma, 0.1)

    def test_three_sensors(self):
        """Fusion of three sensors reduces variance below all individuals."""
        val, sigma = fuse_gaussian_measurements([(5.0, 0.1), (5.0, 0.2), (5.0, 0.3)])
        self.assertLess(sigma, 0.1)

    def test_empty_raises(self):
        """Empty measurement list should raise ValueError."""
        with self.assertRaises(ValueError):
            fuse_gaussian_measurements([])


class TestLidarSensor(unittest.TestCase):
    """Tests for the LIDAR range sensor."""

    def test_straight_wall(self):
        """Robot facing a wall should measure correct distance (within noise)."""
        random.seed(123)
        lidar = LidarSensor(noise_sigma=0.01, max_range=10.0, rate_hz=100.0)
        state = RobotState(Vec2(2.0, 5.0), heading=0.0, linear_vel=0, angular_vel=0, timestamp=0.0)
        obstacles = [LineSegment(Vec2(7.0, 0.0), Vec2(7.0, 10.0))]  # Wall at x=7

        reading = lidar.update(state, obstacles=obstacles)
        self.assertIsNotNone(reading)
        # True distance is 5.0m, should be close
        self.assertAlmostEqual(reading, 5.0, delta=0.1)

    def test_no_obstacle_returns_max(self):
        """No obstacles should return max_range (plus noise)."""
        random.seed(456)
        lidar = LidarSensor(noise_sigma=0.01, max_range=8.0, rate_hz=100.0)
        state = RobotState(Vec2(5.0, 5.0), heading=0.0, linear_vel=0, angular_vel=0, timestamp=0.0)

        reading = lidar.update(state, obstacles=[])
        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading, 8.0, delta=0.1)

    def test_rate_limiting(self):
        """Sensor should only fire at its configured rate."""
        lidar = LidarSensor(noise_sigma=0.0, max_range=10.0, rate_hz=10.0)
        state = RobotState(Vec2(0, 0), heading=0, linear_vel=0, angular_vel=0, timestamp=0.0)
        obstacles = [LineSegment(Vec2(5, -1), Vec2(5, 1))]

        # First call should return a reading
        r1 = lidar.update(state, obstacles=obstacles)
        self.assertIsNotNone(r1)

        # Call 50ms later (period is 100ms) — should NOT fire
        state.timestamp = 0.05
        r2 = lidar.update(state, obstacles=obstacles)
        self.assertIsNone(r2)

        # Call 100ms later — should fire
        state.timestamp = 0.10
        r3 = lidar.update(state, obstacles=obstacles)
        self.assertIsNotNone(r3)

    def test_scan_returns_correct_count(self):
        """LIDAR scan should return the correct number of beams."""
        lidar = LidarSensor(noise_sigma=0.01, max_range=10.0)
        state = RobotState(Vec2(5, 5), heading=0, linear_vel=0, angular_vel=0, timestamp=0.0)
        obstacles = [LineSegment(Vec2(0, 0), Vec2(10, 0))]

        scan = lidar.scan(state, num_beams=36, fov=2 * math.pi, obstacles=obstacles)
        self.assertEqual(len(scan), 36)
        for angle, dist in scan:
            self.assertIsInstance(angle, float)
            self.assertIsInstance(dist, float)
            self.assertGreater(dist, 0)


class TestIMUSensor(unittest.TestCase):
    """Tests for the IMU sensor."""

    def test_gyro_reading_near_true(self):
        """Gyro should read close to true angular velocity."""
        random.seed(789)
        imu = IMUSensor(gyro_noise=0.01, gyro_bias_drift=0.0, rate_hz=100.0)
        state = RobotState(Vec2(0, 0), heading=0, linear_vel=0, angular_vel=1.5, timestamp=0.0)

        reading = imu.update(state)
        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading["angular_vel"], 1.5, delta=0.05)

    def test_bias_drift_accumulates(self):
        """After many samples, gyro bias should drift noticeably from zero."""
        random.seed(101)
        imu = IMUSensor(gyro_noise=0.001, gyro_bias_drift=0.01, rate_hz=100.0)
        state = RobotState(Vec2(0, 0), heading=0, linear_vel=0, angular_vel=0.0, timestamp=0.0)

        # Run 1000 samples
        readings = []
        for i in range(1000):
            state.timestamp = i * 0.01
            r = imu.update(state)
            if r is not None:
                readings.append(r["angular_vel"])

        # With true_vel=0, any nonzero mean comes from bias drift
        mean_reading = sum(readings) / len(readings)
        # Bias should have drifted — mean reading won't be zero
        # (This test verifies drift exists, not a specific value)
        self.assertGreater(abs(imu._gyro_bias), 0.01)


class TestOdometrySensor(unittest.TestCase):
    """Tests for the odometry sensor."""

    def test_straight_line_accuracy(self):
        """Odometry on a straight line should be reasonably close to truth."""
        random.seed(202)
        odom = OdometrySensor(noise_per_meter=0.001, rate_hz=50.0)
        state = RobotState(Vec2(0, 0), heading=0, linear_vel=1.0, angular_vel=0.0, timestamp=0.0)

        dt = 0.02  # 50 Hz
        for i in range(250):  # 5 seconds = 5 meters
            state.timestamp = i * dt
            state.position.x += state.linear_vel * dt
            odom.update(state, dt=dt)

        # After 5m straight travel with very low noise, should be close
        error = math.sqrt((5.0 - odom.est_x) ** 2 + odom.est_y ** 2)
        self.assertLess(error, 0.1)  # Within 10cm

    def test_drift_grows_with_distance(self):
        """Longer travel should produce more odometry error."""
        errors = []
        for distance in [5, 10, 20]:
            random.seed(303)
            odom = OdometrySensor(noise_per_meter=0.01, rate_hz=50.0)
            state = RobotState(Vec2(0, 0), heading=0, linear_vel=1.0, angular_vel=0.0, timestamp=0.0)
            dt = 0.02
            steps = int(distance / (state.linear_vel * dt))
            for i in range(steps):
                state.timestamp = i * dt
                state.position.x += state.linear_vel * dt
                odom.update(state, dt=dt)
            err = math.sqrt((state.position.x - odom.est_x) ** 2 + odom.est_y ** 2)
            errors.append(err)

        # Error should generally increase with distance
        self.assertLess(errors[0], errors[2])

    def test_reset_clears_state(self):
        """Reset should zero out all accumulated state."""
        odom = OdometrySensor()
        odom.est_x = 5.0
        odom.est_y = 3.0
        odom.est_heading = 1.0
        odom.reset()
        self.assertEqual(odom.est_x, 0.0)
        self.assertEqual(odom.est_y, 0.0)
        self.assertEqual(odom.est_heading, 0.0)


if __name__ == "__main__":
    unittest.main()
