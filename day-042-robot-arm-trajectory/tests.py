"""
Day 42: Robot Arm Trajectory Planning - Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import unittest

from my_solution import (
    ArmConfig,
    JointLimits,
    TrajectoryPoint,
    forward_kinematics,
    inverse_kinematics,
    is_reachable,
    check_joint_limits,
    trapezoidal_profile,
    cubic_trajectory,
    compute_via_velocities,
    plan_multi_waypoint_trajectory,
    validate_trajectory,
)


def make_arm() -> ArmConfig:
    """Standard test arm."""
    return ArmConfig(
        L1=1.0,
        L2=0.8,
        joint1_limits=JointLimits(
            min_angle=math.radians(-180),
            max_angle=math.radians(180),
            max_velocity=math.radians(120),
            max_acceleration=math.radians(300),
        ),
        joint2_limits=JointLimits(
            min_angle=math.radians(-150),
            max_angle=math.radians(150),
            max_velocity=math.radians(150),
            max_acceleration=math.radians(400),
        ),
    )


class TestForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.arm = make_arm()

    def test_straight_arm(self):
        """Both joints at 0: end-effector at (L1+L2, 0)."""
        x, y = forward_kinematics(self.arm, 0, 0)
        self.assertAlmostEqual(x, 1.8, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)

    def test_folded_arm(self):
        """Joint 2 at 180 degrees: end-effector at (L1-L2, 0)."""
        x, y = forward_kinematics(self.arm, 0, math.pi)
        self.assertAlmostEqual(x, 0.2, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)

    def test_right_angle(self):
        """Joint 1 at 90 deg, joint 2 at 0: end-effector at (0, L1+L2)."""
        x, y = forward_kinematics(self.arm, math.pi / 2, 0)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.8, places=5)


class TestInverseKinematics(unittest.TestCase):
    def setUp(self):
        self.arm = make_arm()

    def test_roundtrip(self):
        """FK then IK should recover original angles."""
        q1, q2 = math.radians(45), math.radians(30)
        x, y = forward_kinematics(self.arm, q1, q2)
        result = inverse_kinematics(self.arm, x, y, elbow_up=True)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], q1, places=4)
        self.assertAlmostEqual(result[1], q2, places=4)

    def test_unreachable_point(self):
        """Point beyond workspace returns None."""
        result = inverse_kinematics(self.arm, 2.5, 0.0)
        self.assertIsNone(result)

    def test_elbow_configurations(self):
        """Elbow-up and elbow-down give different solutions to the same point."""
        x, y = 1.0, 0.5
        up = inverse_kinematics(self.arm, x, y, elbow_up=True)
        down = inverse_kinematics(self.arm, x, y, elbow_up=False)
        self.assertIsNotNone(up)
        self.assertIsNotNone(down)
        # Both should reach the same point
        x_up, y_up = forward_kinematics(self.arm, *up)
        x_down, y_down = forward_kinematics(self.arm, *down)
        self.assertAlmostEqual(x_up, x, places=4)
        self.assertAlmostEqual(y_up, y, places=4)
        self.assertAlmostEqual(x_down, x, places=4)
        self.assertAlmostEqual(y_down, y, places=4)
        # But angles should differ
        self.assertNotAlmostEqual(up[1], down[1], places=2)


class TestReachability(unittest.TestCase):
    def setUp(self):
        self.arm = make_arm()

    def test_reachable_point(self):
        self.assertTrue(is_reachable(self.arm, 1.0, 0.5))

    def test_unreachable_far(self):
        self.assertFalse(is_reachable(self.arm, 2.5, 0.0))

    def test_boundary_point(self):
        """Point at max reach should be reachable."""
        self.assertTrue(is_reachable(self.arm, 1.8, 0.0))


class TestTrapezoidalProfile(unittest.TestCase):
    def test_reaches_target(self):
        """Profile should start at q_start and end at q_end."""
        points = trapezoidal_profile(0, math.radians(90), math.radians(120), math.radians(300), 0.01)
        self.assertAlmostEqual(points[0][1], 0.0, places=3)
        self.assertAlmostEqual(points[-1][1], math.radians(90), places=2)

    def test_starts_and_ends_at_rest(self):
        """Velocity should be ~0 at start and end."""
        points = trapezoidal_profile(0, math.radians(90), math.radians(120), math.radians(300), 0.01)
        self.assertAlmostEqual(points[0][2], 0.0, places=3)
        self.assertAlmostEqual(points[-1][2], 0.0, places=1)

    def test_respects_velocity_limit(self):
        """No point should exceed v_max."""
        v_max = math.radians(120)
        points = trapezoidal_profile(0, math.radians(90), v_max, math.radians(300), 0.01)
        for _, _, vel, _ in points:
            self.assertLessEqual(abs(vel), v_max + 1e-6)


class TestCubicTrajectory(unittest.TestCase):
    def test_boundary_conditions(self):
        """Cubic should satisfy position and velocity at endpoints."""
        q_start, q_end = 0.0, math.radians(60)
        v_start, v_end = 0.0, 0.0
        T = 2.0
        points = cubic_trajectory(q_start, q_end, v_start, v_end, T, 0.01)
        # Start
        self.assertAlmostEqual(points[0][1], q_start, places=4)
        self.assertAlmostEqual(points[0][2], v_start, places=4)
        # End
        self.assertAlmostEqual(points[-1][1], q_end, places=3)
        self.assertAlmostEqual(points[-1][2], v_end, places=2)

    def test_monotonic_zero_velocity_endpoints(self):
        """With zero start/end velocity and positive displacement, trajectory should be monotonic."""
        points = cubic_trajectory(0, 1.0, 0, 0, 2.0, 0.01)
        for i in range(1, len(points)):
            self.assertGreaterEqual(points[i][1], points[i - 1][1] - 1e-9)


class TestMultiWaypointTrajectory(unittest.TestCase):
    def setUp(self):
        self.arm = make_arm()

    def test_reaches_all_waypoints(self):
        """Trajectory should start and end at the specified Cartesian positions."""
        waypoints = [(1.2, 0.5), (0.8, 1.0), (1.0, 0.3)]
        durations = [1.5, 1.5]
        traj = plan_multi_waypoint_trajectory(self.arm, waypoints, durations, dt=0.05)
        # Check start
        self.assertAlmostEqual(traj[0].x, waypoints[0][0], places=3)
        self.assertAlmostEqual(traj[0].y, waypoints[0][1], places=3)
        # Check end
        self.assertAlmostEqual(traj[-1].x, waypoints[-1][0], places=3)
        self.assertAlmostEqual(traj[-1].y, waypoints[-1][1], places=3)

    def test_valid_trajectory(self):
        """Generated trajectory should have no violations."""
        waypoints = [(1.2, 0.5), (0.8, 1.0), (1.0, 0.3)]
        durations = [1.5, 1.5]
        traj = plan_multi_waypoint_trajectory(self.arm, waypoints, durations, dt=0.05)
        violations = validate_trajectory(self.arm, traj)
        self.assertEqual(len(violations), 0, f"Violations found: {violations[:3]}")

    def test_unreachable_waypoint_raises(self):
        """Should raise ValueError for unreachable waypoint."""
        waypoints = [(1.2, 0.5), (2.5, 0.0)]
        durations = [1.5]
        with self.assertRaises(ValueError):
            plan_multi_waypoint_trajectory(self.arm, waypoints, durations)


if __name__ == "__main__":
    unittest.main()
