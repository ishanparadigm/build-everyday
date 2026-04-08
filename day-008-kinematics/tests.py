"""
Tests for Day 008: Forward and Inverse Kinematics

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import math
import unittest
from my_solution import (
    RobotArm,
    forward_kinematics,
    inverse_kinematics_2link,
    compute_jacobian,
    inverse_kinematics_numerical,
    compute_workspace,
)


class TestRobotArm(unittest.TestCase):
    """Tests for the RobotArm data structure."""

    def test_max_reach(self):
        arm = RobotArm([1.0, 0.8])
        self.assertAlmostEqual(arm.max_reach, 1.8)

    def test_min_reach_equal_links(self):
        # Two equal links can fold to reach the origin
        arm = RobotArm([1.0, 1.0])
        self.assertAlmostEqual(arm.min_reach, 0.0)

    def test_min_reach_unequal_links(self):
        # L1=2, L2=0.5 -> can't reach closer than 1.5
        arm = RobotArm([2.0, 0.5])
        self.assertAlmostEqual(arm.min_reach, 1.5)

    def test_invalid_links(self):
        with self.assertRaises(ValueError):
            RobotArm([])
        with self.assertRaises(ValueError):
            RobotArm([-1.0, 0.5])


class TestForwardKinematics(unittest.TestCase):
    """Tests for forward kinematics."""

    def test_straight_arm(self):
        """All angles zero -> arm extends along x-axis."""
        arm = RobotArm([1.0, 0.8])
        positions, end = forward_kinematics(arm, [0.0, 0.0])
        self.assertAlmostEqual(end[0], 1.8, places=6)
        self.assertAlmostEqual(end[1], 0.0, places=6)

    def test_right_angle(self):
        """First link along x, second link turns 90 degrees up."""
        arm = RobotArm([1.0, 1.0])
        positions, end = forward_kinematics(arm, [0.0, math.pi / 2])
        self.assertAlmostEqual(end[0], 1.0, places=6)
        self.assertAlmostEqual(end[1], 1.0, places=6)

    def test_folded_back(self):
        """Second link folds back 180 degrees -> end at (L1-L2, 0)."""
        arm = RobotArm([1.0, 0.5])
        positions, end = forward_kinematics(arm, [0.0, math.pi])
        self.assertAlmostEqual(end[0], 0.5, places=6)
        self.assertAlmostEqual(end[1], 0.0, places=6)

    def test_base_is_origin(self):
        arm = RobotArm([1.0, 0.8])
        positions, _ = forward_kinematics(arm, [0.5, -0.3])
        self.assertEqual(positions[0], (0.0, 0.0))

    def test_three_link(self):
        """3-link arm, all angles zero -> straight along x."""
        arm = RobotArm([1.0, 0.7, 0.5])
        positions, end = forward_kinematics(arm, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(end[0], 2.2, places=6)
        self.assertAlmostEqual(end[1], 0.0, places=6)
        self.assertEqual(len(positions), 4)  # base + 3 joints/end

    def test_wrong_angle_count(self):
        arm = RobotArm([1.0, 0.8])
        with self.assertRaises(ValueError):
            forward_kinematics(arm, [0.0])


class TestAnalyticalIK(unittest.TestCase):
    """Tests for 2-link analytical inverse kinematics."""

    def test_round_trip(self):
        """FK -> IK -> FK should recover the same position."""
        arm = RobotArm([1.0, 0.8])
        original = [math.radians(45), math.radians(-30)]
        _, target = forward_kinematics(arm, original)

        result = inverse_kinematics_2link(arm, target, elbow_up=True)
        self.assertIsNotNone(result)
        _, recovered = forward_kinematics(arm, list(result))
        self.assertAlmostEqual(recovered[0], target[0], places=5)
        self.assertAlmostEqual(recovered[1], target[1], places=5)

    def test_max_reach(self):
        """Target exactly at max reach -> one solution (theta2 = 0)."""
        arm = RobotArm([1.0, 0.8])
        result = inverse_kinematics_2link(arm, (1.8, 0.0))
        self.assertIsNotNone(result)
        t1, t2 = result
        self.assertAlmostEqual(abs(t2), 0.0, places=3)

    def test_unreachable_far(self):
        arm = RobotArm([1.0, 0.8])
        result = inverse_kinematics_2link(arm, (3.0, 0.0))
        self.assertIsNone(result)

    def test_unreachable_close(self):
        """Target inside the inner hole of unequal-length arm."""
        arm = RobotArm([2.0, 0.5])
        result = inverse_kinematics_2link(arm, (0.1, 0.0))
        self.assertIsNone(result)

    def test_elbow_up_vs_down(self):
        """Two solutions should both reach the target but differ in angles."""
        arm = RobotArm([1.0, 0.8])
        target = (1.0, 0.5)
        up = inverse_kinematics_2link(arm, target, elbow_up=True)
        down = inverse_kinematics_2link(arm, target, elbow_up=False)
        self.assertIsNotNone(up)
        self.assertIsNotNone(down)
        # Both reach the target
        _, end_up = forward_kinematics(arm, list(up))
        _, end_down = forward_kinematics(arm, list(down))
        self.assertAlmostEqual(end_up[0], target[0], places=4)
        self.assertAlmostEqual(end_down[0], target[0], places=4)
        # But angles differ
        self.assertFalse(
            abs(up[1] - down[1]) < 1e-6,
            "Elbow-up and elbow-down should give different theta2"
        )


class TestJacobian(unittest.TestCase):
    """Tests for Jacobian computation."""

    def test_numerical_gradient(self):
        """Jacobian should match finite-difference approximation."""
        arm = RobotArm([1.0, 0.7, 0.5])
        angles = [0.5, -0.3, 0.8]
        J = compute_jacobian(arm, angles)
        eps = 1e-6

        for i in range(arm.n_joints):
            # Perturb angle i
            angles_plus = list(angles)
            angles_plus[i] += eps
            angles_minus = list(angles)
            angles_minus[i] -= eps

            _, end_plus = forward_kinematics(arm, angles_plus)
            _, end_minus = forward_kinematics(arm, angles_minus)

            dx_numerical = (end_plus[0] - end_minus[0]) / (2 * eps)
            dy_numerical = (end_plus[1] - end_minus[1]) / (2 * eps)

            self.assertAlmostEqual(J[0][i], dx_numerical, places=4,
                                   msg=f"dx/dtheta{i} mismatch")
            self.assertAlmostEqual(J[1][i], dy_numerical, places=4,
                                   msg=f"dy/dtheta{i} mismatch")


class TestNumericalIK(unittest.TestCase):
    """Tests for numerical inverse kinematics."""

    def test_reachable_target(self):
        arm = RobotArm([1.0, 0.7, 0.5])
        target = (1.5, 0.5)
        result = inverse_kinematics_numerical(arm, target)
        self.assertIsNotNone(result)
        angles, iters, err = result
        self.assertLess(err, 1e-3)
        # Verify with FK
        _, end = forward_kinematics(arm, angles)
        self.assertAlmostEqual(end[0], target[0], places=2)
        self.assertAlmostEqual(end[1], target[1], places=2)

    def test_unreachable_target(self):
        arm = RobotArm([1.0, 0.7, 0.5])
        result = inverse_kinematics_numerical(arm, (5.0, 0.0))
        self.assertIsNone(result)

    def test_matches_analytical(self):
        """For a 2-link arm, numerical should find same position as analytical."""
        arm = RobotArm([1.0, 0.8])
        target = (1.0, 0.5)
        analytical = inverse_kinematics_2link(arm, target)
        numerical = inverse_kinematics_numerical(arm, target)
        self.assertIsNotNone(analytical)
        self.assertIsNotNone(numerical)
        # Both should reach the target
        _, end_a = forward_kinematics(arm, list(analytical))
        _, end_n = forward_kinematics(arm, numerical[0])
        self.assertAlmostEqual(end_a[0], end_n[0], places=2)
        self.assertAlmostEqual(end_a[1], end_n[1], places=2)


class TestWorkspace(unittest.TestCase):
    """Tests for workspace analysis."""

    def test_workspace_bounds(self):
        arm = RobotArm([1.0, 0.8])
        points = compute_workspace(arm, samples_per_joint=18)
        self.assertGreater(len(points), 0)
        max_dist = max(math.sqrt(x * x + y * y) for x, y in points)
        # Max distance should be close to max_reach
        self.assertAlmostEqual(max_dist, arm.max_reach, places=1)


if __name__ == "__main__":
    unittest.main()
