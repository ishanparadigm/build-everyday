"""
Day 85: Occupancy Grid Mapping — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import unittest
import numpy as np

from my_solution import bresenham, OccupancyGrid, Environment


class TestBresenham(unittest.TestCase):
    """Test Bresenham's line algorithm for correctness."""

    def test_horizontal_line(self):
        """Horizontal line should return all cells along x-axis."""
        cells = bresenham(0, 0, 5, 0)
        self.assertEqual(cells, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])

    def test_vertical_line(self):
        """Vertical line should return all cells along y-axis."""
        cells = bresenham(0, 0, 0, 4)
        self.assertEqual(cells, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])

    def test_diagonal_line(self):
        """45-degree diagonal should step in both x and y each time."""
        cells = bresenham(0, 0, 3, 3)
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0], (0, 0))
        self.assertEqual(cells[-1], (3, 3))

    def test_single_point(self):
        """Line from a cell to itself should return just that cell."""
        cells = bresenham(5, 5, 5, 5)
        self.assertEqual(cells, [(5, 5)])

    def test_negative_direction(self):
        """Line going in negative direction should still work."""
        cells = bresenham(5, 5, 0, 5)
        self.assertEqual(len(cells), 6)
        self.assertEqual(cells[0], (5, 5))
        self.assertEqual(cells[-1], (0, 5))

    def test_steep_line(self):
        """Steep line (dy > dx) should work correctly."""
        cells = bresenham(0, 0, 1, 5)
        # Should include start and end
        self.assertEqual(cells[0], (0, 0))
        self.assertEqual(cells[-1], (1, 5))
        # Should have 6 cells (steps along the major axis = max(dx,dy) + 1)
        self.assertEqual(len(cells), 6)


class TestOccupancyGrid(unittest.TestCase):
    """Test the occupancy grid fundamentals."""

    def setUp(self):
        self.grid = OccupancyGrid(
            width=10.0, height=10.0, resolution=0.5, origin=(0.0, 0.0)
        )

    def test_grid_dimensions(self):
        """Grid size should match width/height divided by resolution."""
        self.assertEqual(self.grid.grid_w, 20)
        self.assertEqual(self.grid.grid_h, 20)

    def test_initial_state_unknown(self):
        """All cells should start as unknown (log-odds = 0, P = 0.5)."""
        self.assertTrue(np.all(self.grid.log_odds == 0.0))
        prob = self.grid.get_probability_map()
        np.testing.assert_allclose(prob, 0.5)

    def test_world_to_grid_conversion(self):
        """World coordinates should map to correct grid indices."""
        gx, gy = self.grid.world_to_grid(5.0, 5.0)
        self.assertEqual(gx, 10)
        self.assertEqual(gy, 10)

    def test_world_to_grid_origin(self):
        """Origin should map to grid (0, 0)."""
        gx, gy = self.grid.world_to_grid(0.0, 0.0)
        self.assertEqual(gx, 0)
        self.assertEqual(gy, 0)

    def test_grid_to_world_roundtrip(self):
        """Converting to grid and back should land near the original point."""
        wx, wy = 3.7, 6.2
        gx, gy = self.grid.world_to_grid(wx, wy)
        wx2, wy2 = self.grid.grid_to_world(gx, gy)
        # Should be within one cell width of the original
        self.assertAlmostEqual(wx2, wx, delta=self.grid.resolution)
        self.assertAlmostEqual(wy2, wy, delta=self.grid.resolution)

    def test_update_cell_clamping(self):
        """Cell updates should be clamped to [l_min, l_max]."""
        # Apply many occupied updates to exceed l_max
        for _ in range(100):
            self.grid.update_cell(5, 5, self.grid.l_occ)
        self.assertLessEqual(self.grid.log_odds[5, 5], self.grid.l_max)

        # Apply many free updates to go below l_min
        for _ in range(200):
            self.grid.update_cell(5, 5, self.grid.l_free)
        self.assertGreaterEqual(self.grid.log_odds[5, 5], self.grid.l_min)

    def test_in_bounds(self):
        """Bounds checking should correctly identify valid and invalid cells."""
        self.assertTrue(self.grid.in_bounds(0, 0))
        self.assertTrue(self.grid.in_bounds(19, 19))
        self.assertFalse(self.grid.in_bounds(-1, 0))
        self.assertFalse(self.grid.in_bounds(0, 20))
        self.assertFalse(self.grid.in_bounds(20, 0))


class TestInverseSensorModel(unittest.TestCase):
    """Test the sensor model and scan processing."""

    def setUp(self):
        self.grid = OccupancyGrid(
            width=10.0, height=10.0, resolution=0.2, origin=(0.0, 0.0)
        )

    def test_single_beam_marks_free_and_occupied(self):
        """A single beam should mark cells along the ray as free and the endpoint as occupied."""
        # Beam from (5,5) pointing right, hitting something at range 3.0
        self.grid.inverse_sensor_model(5.0, 5.0, 0.0, 0.0, 3.0, 10.0)

        # Endpoint cell should be occupied (positive log-odds)
        gx_end, gy_end = self.grid.world_to_grid(8.0, 5.0)
        self.assertGreater(self.grid.log_odds[gy_end, gx_end], 0.0)

        # A cell along the ray should be free (negative log-odds)
        gx_mid, gy_mid = self.grid.world_to_grid(6.5, 5.0)
        self.assertLess(self.grid.log_odds[gy_mid, gx_mid], 0.0)

    def test_max_range_no_occupied(self):
        """A beam at max range should not mark any cell as occupied."""
        max_range = 10.0
        self.grid.inverse_sensor_model(5.0, 5.0, 0.0, 0.0, max_range, max_range)

        # No cell should have positive log-odds (no occupied marking)
        self.assertTrue(np.all(self.grid.log_odds <= 0.0))

    def test_multiple_scans_increase_confidence(self):
        """Repeated observations should increase confidence (higher |log-odds|)."""
        # Single scan
        self.grid.inverse_sensor_model(5.0, 5.0, 0.0, 0.0, 3.0, 10.0)
        gx, gy = self.grid.world_to_grid(8.0, 5.0)
        lo_after_one = self.grid.log_odds[gy, gx]

        # Second scan at same pose
        self.grid.inverse_sensor_model(5.0, 5.0, 0.0, 0.0, 3.0, 10.0)
        lo_after_two = self.grid.log_odds[gy, gx]

        # Confidence should increase (log-odds should be more positive)
        self.assertGreater(lo_after_two, lo_after_one)


class TestEnvironment(unittest.TestCase):
    """Test the simulated environment."""

    def setUp(self):
        # Simple box: 10x10 room
        self.env = Environment([
            (0, 0, 10, 0), (10, 0, 10, 10),
            (10, 10, 0, 10), (0, 10, 0, 0),
        ])

    def test_ray_hits_wall(self):
        """Ray from center of box should hit a wall at the expected distance."""
        # From (5, 5) pointing right → should hit wall at x=10, distance=5
        dist = self.env.cast_ray(5.0, 5.0, 0.0, 20.0)
        self.assertAlmostEqual(dist, 5.0, places=2)

    def test_ray_max_range(self):
        """Ray that doesn't hit anything within max_range returns max_range."""
        # Point the ray in a direction with a nearby wall, but set max_range very small
        dist = self.env.cast_ray(5.0, 5.0, 0.0, 2.0)
        # Wall is 5m away but max_range is 2m
        self.assertAlmostEqual(dist, 2.0, places=2)

    def test_lidar_scan_shape(self):
        """Simulated lidar should return correct array shapes."""
        ranges, angles = self.env.simulate_lidar(5.0, 5.0, 0.0, num_beams=36)
        self.assertEqual(len(ranges), 36)
        self.assertEqual(len(angles), 36)

    def test_lidar_range_bounds(self):
        """All lidar ranges should be between 0 and max_range."""
        max_r = 8.0
        ranges, _ = self.env.simulate_lidar(5.0, 5.0, 0.0, max_range=max_r)
        self.assertTrue(np.all(ranges >= 0))
        self.assertTrue(np.all(ranges <= max_r))


class TestEndToEnd(unittest.TestCase):
    """Integration test: full mapping pipeline."""

    def test_mapping_produces_correct_map(self):
        """After mapping a simple room, free/occupied cells should be reasonable."""
        env = Environment([
            (0, 0, 10, 0), (10, 0, 10, 10),
            (10, 10, 0, 10), (0, 10, 0, 0),
        ])
        grid = OccupancyGrid(width=10.0, height=10.0, resolution=0.2)

        np.random.seed(42)
        # Take several scans from the center
        for theta in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            ranges, angles = env.simulate_lidar(
                5.0, 5.0, theta, num_beams=90, max_range=8.0, noise_std=0.02
            )
            grid.update_scan(5.0, 5.0, theta, ranges, angles, 8.0)

        stats = grid.get_map_stats()
        # Should have meaningful free and occupied cells
        self.assertGreater(stats["free_pct"], 10.0, "Should have significant free space")
        self.assertGreater(stats["occupied_pct"], 0.5, "Should detect walls")
        # Free should dominate (it's mostly open space)
        self.assertGreater(stats["free_pct"], stats["occupied_pct"])


if __name__ == "__main__":
    unittest.main()
