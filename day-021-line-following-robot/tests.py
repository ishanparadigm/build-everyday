"""
Day 021: Line-Following Robot Logic — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import unittest

from my_solution import (
    SensorArray,
    BangBangController,
    PController,
    PIDController,
    DiffDriveRobot,
    build_track,
    closest_point_on_track,
    run_simulation,
    analyze_performance,
)


class TestTrackBuilder(unittest.TestCase):
    """Test track construction from segment definitions."""

    def test_straight_segment(self):
        """A straight segment should produce evenly spaced points."""
        segments = [{"type": "straight", "start": (0.0, 0.0), "end": (1.0, 0.0)}]
        track = build_track(segments)
        self.assertGreater(len(track), 5)
        # First and last points should match start and end
        self.assertAlmostEqual(track[0][0], 0.0, places=3)
        self.assertAlmostEqual(track[0][1], 0.0, places=3)
        self.assertAlmostEqual(track[-1][0], 1.0, places=3)
        self.assertAlmostEqual(track[-1][1], 0.0, places=3)

    def test_arc_segment(self):
        """An arc segment should produce points on a circle."""
        segments = [{
            "type": "arc", "center": (0.0, 0.0), "radius": 1.0,
            "start_angle": 0.0, "end_angle": math.pi / 2
        }]
        track = build_track(segments)
        # All points should be at radius 1.0 from center
        for x, y in track:
            dist = math.hypot(x, y)
            self.assertAlmostEqual(dist, 1.0, places=3)

    def test_multi_segment_track(self):
        """Multiple segments should produce a continuous track."""
        segments = [
            {"type": "straight", "start": (0.0, 0.0), "end": (1.0, 0.0)},
            {"type": "arc", "center": (1.0, -0.5), "radius": 0.5,
             "start_angle": math.pi / 2, "end_angle": 0.0},
        ]
        track = build_track(segments)
        self.assertGreater(len(track), 10)


class TestClosestPoint(unittest.TestCase):
    """Test the closest-point-on-track computation."""

    def test_point_on_track(self):
        """A point exactly on the track should have distance ~0."""
        track = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        cx, cy, dist = closest_point_on_track(track, 0.5, 0.0)
        self.assertAlmostEqual(dist, 0.0, places=5)
        self.assertAlmostEqual(cx, 0.5, places=3)

    def test_point_off_track(self):
        """A point perpendicular to the track should have correct distance."""
        track = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        cx, cy, dist = closest_point_on_track(track, 1.0, 0.3)
        self.assertAlmostEqual(dist, 0.3, places=3)
        self.assertAlmostEqual(cx, 1.0, places=3)
        self.assertAlmostEqual(cy, 0.0, places=3)

    def test_point_past_end(self):
        """A point beyond the track end should snap to the endpoint."""
        track = [(0.0, 0.0), (1.0, 0.0)]
        cx, cy, dist = closest_point_on_track(track, 1.5, 0.0)
        self.assertAlmostEqual(cx, 1.0, places=3)
        self.assertAlmostEqual(dist, 0.5, places=3)


class TestSensorArray(unittest.TestCase):
    """Test the reflectance sensor simulation."""

    def test_sensor_positions_symmetric(self):
        """Sensor positions should be symmetric around center."""
        sa = SensorArray(n_sensors=5, array_width=0.1)
        positions = sa.sensor_positions
        self.assertEqual(len(positions), 5)
        self.assertAlmostEqual(positions[0], -0.05, places=5)
        self.assertAlmostEqual(positions[-1], 0.05, places=5)
        self.assertAlmostEqual(positions[2], 0.0, places=5)

    def test_on_line_reading(self):
        """A sensor directly on the line should read near 0."""
        sa = SensorArray(n_sensors=1, array_width=0.0, line_sigma=0.01)
        track = [(0.0, 0.0), (1.0, 0.0)]
        readings = sa.read(0.5, 0.0, 0.0, track)
        self.assertLess(readings[0], 0.1)

    def test_off_line_reading(self):
        """A sensor far from the line should read near 1."""
        sa = SensorArray(n_sensors=1, array_width=0.0, line_sigma=0.01)
        track = [(0.0, 0.0), (1.0, 0.0)]
        readings = sa.read(0.5, 0.5, 0.0, track)
        self.assertGreater(readings[0], 0.9)

    def test_line_position_centered(self):
        """When robot is centered on line, estimated position should be ~0."""
        sa = SensorArray(n_sensors=5, array_width=0.06, line_sigma=0.012)
        track = [(0.0, 0.0), (1.0, 0.0)]
        readings = sa.read(0.5, 0.0, 0.0, track)
        pos = sa.estimate_line_position(readings)
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos, 0.0, places=3)

    def test_line_position_offset(self):
        """When robot is offset, estimated position should reflect offset direction."""
        sa = SensorArray(n_sensors=5, array_width=0.06, line_sigma=0.012)
        track = [(0.0, 0.0), (1.0, 0.0)]
        # Robot offset to the right of the line (line is to its left)
        readings = sa.read(0.5, -0.01, 0.0, track)
        pos = sa.estimate_line_position(readings)
        self.assertIsNotNone(pos)
        # Line should appear on the left (negative position means left-side sensors see it more)
        self.assertLess(pos, 0.0)


class TestControllers(unittest.TestCase):
    """Test controller output correctness."""

    def test_bangbang_positive_error(self):
        bb = BangBangController(strength=0.08)
        self.assertAlmostEqual(bb.compute(0.5, 0.01), 0.08)

    def test_bangbang_negative_error(self):
        bb = BangBangController(strength=0.08)
        self.assertAlmostEqual(bb.compute(-0.5, 0.01), -0.08)

    def test_p_proportional(self):
        p = PController(kp=3.0)
        self.assertAlmostEqual(p.compute(0.1, 0.01), 0.3)
        self.assertAlmostEqual(p.compute(-0.2, 0.01), -0.6)

    def test_pid_integral_accumulation(self):
        """PID integral term should accumulate over repeated calls."""
        pid = PIDController(kp=0.0, ki=10.0, kd=0.0, integral_limit=0.5)
        pid.reset()
        # Constant error of 0.1 for 10 steps of dt=0.01
        outputs = [pid.compute(0.1, 0.01) for _ in range(10)]
        # Integral grows: 10.0 * (0.1 * 0.01 * n) for step n
        # Output should increase each step
        self.assertGreater(outputs[-1], outputs[0])


class TestDiffDriveRobot(unittest.TestCase):
    """Test differential drive kinematics."""

    def test_straight_line(self):
        """Equal wheel speeds should drive straight."""
        robot = DiffDriveRobot(x=0.0, y=0.0, theta=0.0)
        for _ in range(100):
            robot.update(0.1, 0.1, 0.01)
        self.assertGreater(robot.x, 0.09)
        self.assertAlmostEqual(robot.y, 0.0, places=3)
        self.assertAlmostEqual(robot.theta, 0.0, places=3)

    def test_turn_in_place(self):
        """Opposite wheel speeds should rotate without translation."""
        robot = DiffDriveRobot(x=0.0, y=0.0, theta=0.0, wheel_base=0.1)
        for _ in range(100):
            robot.update(-0.1, 0.1, 0.01)
        # Should have rotated significantly but stayed near origin
        self.assertAlmostEqual(robot.x, 0.0, places=2)
        self.assertAlmostEqual(robot.y, 0.0, places=2)
        self.assertNotAlmostEqual(robot.theta, 0.0, places=1)


class TestSimulation(unittest.TestCase):
    """Test the full simulation pipeline."""

    def test_pid_tracks_straight_line(self):
        """PID should track a straight line with low error."""
        track = build_track([
            {"type": "straight", "start": (0.0, 0.0), "end": (2.0, 0.0)}
        ])
        ctrl = PIDController(kp=3.0, ki=0.5, kd=0.8)
        result = run_simulation(track, ctrl, dt=0.005, max_time=15.0)
        perf = analyze_performance(result)

        # On a straight line, PID should achieve sub-millimeter MAE
        self.assertLess(perf.mae, 0.005, "PID MAE too high on straight line")
        self.assertEqual(perf.line_lost_count, 0, "Should not lose line on straight")

    def test_bangbang_more_oscillation_than_pid(self):
        """Bang-bang should oscillate more than PID."""
        track = build_track([
            {"type": "straight", "start": (0.0, 0.0), "end": (2.0, 0.0)}
        ])
        bb_result = run_simulation(track, BangBangController(), dt=0.005, max_time=15.0, start_offset=0.005)
        pid_result = run_simulation(track, PIDController(), dt=0.005, max_time=15.0, start_offset=0.005)

        bb_perf = analyze_performance(bb_result)
        pid_perf = analyze_performance(pid_result)

        self.assertGreater(bb_perf.error_std, pid_perf.error_std)


if __name__ == "__main__":
    unittest.main()
