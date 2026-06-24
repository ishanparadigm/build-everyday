"""
Tests for Day 84: Model Predictive Control (MPC)

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    VehicleParams, MPCParams, bicycle_model, predict_trajectory,
    generate_reference_trajectory, find_closest_reference_index,
    angle_diff, mpc_cost, MPCController, simulate_mpc
)


class TestBicycleModel(unittest.TestCase):
    """Test the discrete-time bicycle kinematic model."""

    def setUp(self):
        self.vehicle = VehicleParams()
        self.dt = 0.1

    def test_straight_line(self):
        """With zero steering and constant speed, robot moves in a straight line."""
        state = np.array([0.0, 0.0, 0.0, 10.0])  # Heading east at 10 m/s
        control = np.array([0.0, 0.0])  # No steer, no accel
        next_state = bicycle_model(state, control, self.vehicle, self.dt)
        # Should move 1m east (10 m/s * 0.1s)
        np.testing.assert_allclose(next_state[0], 1.0, atol=0.01)
        np.testing.assert_allclose(next_state[1], 0.0, atol=0.01)
        np.testing.assert_allclose(next_state[2], 0.0, atol=0.01)
        np.testing.assert_allclose(next_state[3], 10.0, atol=0.01)

    def test_acceleration(self):
        """Acceleration increases velocity."""
        state = np.array([0.0, 0.0, 0.0, 5.0])
        control = np.array([0.0, 2.0])  # Accelerate at 2 m/s^2
        next_state = bicycle_model(state, control, self.vehicle, self.dt)
        self.assertAlmostEqual(next_state[3], 5.2, places=1)  # v + a*dt

    def test_steering_turns(self):
        """Positive steering causes left turn (increasing theta)."""
        state = np.array([0.0, 0.0, 0.0, 5.0])
        control = np.array([0.3, 0.0])  # Steer left
        next_state = bicycle_model(state, control, self.vehicle, self.dt)
        self.assertGreater(next_state[2], 0.0)  # Heading should increase

    def test_speed_clipping(self):
        """Speed should be clipped to [0, max_speed]."""
        # Test max speed
        state = np.array([0.0, 0.0, 0.0, 14.5])
        control = np.array([0.0, 3.0])  # Strong acceleration
        next_state = bicycle_model(state, control, self.vehicle, self.dt)
        self.assertLessEqual(next_state[3], self.vehicle.max_speed)

        # Test min speed (no negative velocity)
        state = np.array([0.0, 0.0, 0.0, 0.5])
        control = np.array([0.0, -5.0])  # Strong braking
        next_state = bicycle_model(state, control, self.vehicle, self.dt)
        self.assertGreaterEqual(next_state[3], 0.0)

    def test_heading_normalization(self):
        """Heading should stay in [-pi, pi]."""
        state = np.array([0.0, 0.0, 3.1, 5.0])  # Near pi
        control = np.array([0.3, 0.0])  # Turn left
        next_state = bicycle_model(state, control, self.vehicle, self.dt)
        self.assertGreaterEqual(next_state[2], -np.pi)
        self.assertLessEqual(next_state[2], np.pi)


class TestPredictTrajectory(unittest.TestCase):
    """Test forward trajectory prediction."""

    def test_output_shape(self):
        """Predicted trajectory has correct shape."""
        vehicle = VehicleParams()
        state = np.array([0.0, 0.0, 0.0, 5.0])
        N = 10
        controls = np.zeros((N, 2))
        result = predict_trajectory(state, controls, vehicle, 0.1)
        self.assertEqual(result.shape, (N + 1, 4))

    def test_initial_state_preserved(self):
        """First element of predicted trajectory is the initial state."""
        vehicle = VehicleParams()
        state = np.array([1.0, 2.0, 0.5, 3.0])
        controls = np.zeros((5, 2))
        result = predict_trajectory(state, controls, vehicle, 0.1)
        np.testing.assert_allclose(result[0], state)


class TestAngleDiff(unittest.TestCase):
    """Test angle difference handling."""

    def test_simple_diff(self):
        """Basic angle difference."""
        self.assertAlmostEqual(angle_diff(0.5, 0.3), 0.2, places=5)

    def test_wraparound(self):
        """Difference across the +-pi boundary should be small."""
        diff = angle_diff(3.1, -3.1)
        self.assertAlmostEqual(abs(diff), 2 * np.pi - 6.2, atol=0.1)
        # More precisely, it should be about 0.083 (the short way around)
        self.assertLess(abs(diff), np.pi)

    def test_opposite_directions(self):
        """Difference between 0 and pi should be pi."""
        diff = angle_diff(np.pi, 0.0)
        self.assertAlmostEqual(abs(diff), np.pi, places=5)


class TestReferenceTrajectory(unittest.TestCase):
    """Test reference trajectory generation."""

    def test_output_shapes(self):
        """All outputs have correct shapes."""
        n = 100
        pos, headings, vels = generate_reference_trajectory(n, "figure8")
        self.assertEqual(pos.shape, (n, 2))
        self.assertEqual(headings.shape, (n,))
        self.assertEqual(vels.shape, (n,))

    def test_velocities_positive(self):
        """Reference velocities should be positive."""
        _, _, vels = generate_reference_trajectory(100, "figure8")
        self.assertTrue(np.all(vels > 0))

    def test_figure8_crosses_origin(self):
        """Figure-8 should cross near the origin."""
        pos, _, _ = generate_reference_trajectory(200, "figure8")
        min_dist_to_origin = np.min(np.linalg.norm(pos, axis=1))
        self.assertLess(min_dist_to_origin, 2.0)


class TestMPCController(unittest.TestCase):
    """Test the MPC controller end-to-end."""

    def test_controller_returns_valid_control(self):
        """Controller should return control within bounds."""
        vehicle = VehicleParams()
        mpc_params = MPCParams(horizon=5)  # Short horizon for speed
        controller = MPCController(vehicle, mpc_params)

        ref_pos, ref_head, ref_vel = generate_reference_trajectory(100, "figure8")
        state = np.array([ref_pos[0, 0], ref_pos[0, 1], ref_head[0], ref_vel[0]])

        control, predicted = controller.compute_control(
            state, ref_pos, ref_head, ref_vel
        )

        self.assertEqual(control.shape, (2,))
        self.assertLessEqual(abs(control[0]), vehicle.max_steering + 0.01)
        self.assertLessEqual(control[1], vehicle.max_accel + 0.01)
        self.assertGreaterEqual(control[1], -vehicle.max_decel - 0.01)

    def test_simulation_tracks_reference(self):
        """Full simulation should achieve reasonable tracking error."""
        results = simulate_mpc(n_steps=100, trajectory_type="figure8")
        # After transient, mean error should be under 3m (generous for short sim)
        mean_error = np.mean(results['tracking_errors'][10:])
        self.assertLess(mean_error, 3.0,
                        f"Mean tracking error {mean_error:.2f}m is too large")


class TestFindClosestReference(unittest.TestCase):
    """Test reference point lookup."""

    def test_finds_closest(self):
        """Should find the geometrically closest point in the search window."""
        ref_positions = np.array([[i, 0] for i in range(100)], dtype=float)
        state = np.array([25.3, 0.1, 0.0, 0.0])
        idx = find_closest_reference_index(state, ref_positions, last_idx=20)
        self.assertEqual(idx, 25)

    def test_respects_last_idx(self):
        """Should not search behind last_idx."""
        ref_positions = np.array([[i, 0] for i in range(100)], dtype=float)
        state = np.array([5.0, 0.0, 0.0, 0.0])
        idx = find_closest_reference_index(state, ref_positions, last_idx=30)
        self.assertGreaterEqual(idx, 30)


if __name__ == '__main__':
    unittest.main()
