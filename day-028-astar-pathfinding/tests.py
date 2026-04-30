"""
Tests for A* Pathfinding implementation.

Run with:
    python3 -m pytest tests.py
    python3 tests.py
"""

import math
import unittest

from my_solution import Grid, astar, manhattan, euclidean, chebyshev, octile


class TestHeuristics(unittest.TestCase):
    """Test that heuristic functions compute correct distances."""

    def test_manhattan_same_point(self):
        self.assertEqual(manhattan((5, 5), (5, 5)), 0)

    def test_manhattan_cardinal(self):
        self.assertEqual(manhattan((0, 0), (3, 4)), 7)

    def test_euclidean_3_4_5_triangle(self):
        self.assertAlmostEqual(euclidean((0, 0), (3, 4)), 5.0)

    def test_chebyshev_diagonal(self):
        # Chebyshev from (0,0) to (3,3) = max(3,3) = 3
        self.assertEqual(chebyshev((0, 0), (3, 3)), 3)

    def test_octile_cardinal(self):
        # Pure horizontal: octile((0,0), (0,5)) = 5 (no diagonals needed)
        self.assertAlmostEqual(octile((0, 0), (0, 5)), 5.0)

    def test_octile_diagonal(self):
        # Pure diagonal: octile((0,0), (3,3)) = 3*sqrt(2)
        self.assertAlmostEqual(octile((0, 0), (3, 3)), 3 * math.sqrt(2))


class TestGrid(unittest.TestCase):
    """Test grid boundary and neighbor logic."""

    def setUp(self):
        self.grid = Grid(5, 5, obstacles={(2, 2)})

    def test_in_bounds(self):
        self.assertTrue(self.grid.in_bounds((0, 0)))
        self.assertTrue(self.grid.in_bounds((4, 4)))
        self.assertFalse(self.grid.in_bounds((-1, 0)))
        self.assertFalse(self.grid.in_bounds((5, 0)))

    def test_obstacle_not_passable(self):
        self.assertFalse(self.grid.is_passable((2, 2)))
        self.assertTrue(self.grid.is_passable((0, 0)))

    def test_4dir_neighbors_corner(self):
        """Corner cell should have exactly 2 neighbors."""
        neighbors = self.grid.neighbors_4dir((0, 0))
        positions = [pos for pos, _ in neighbors]
        self.assertEqual(len(positions), 2)
        self.assertIn((0, 1), positions)
        self.assertIn((1, 0), positions)

    def test_4dir_neighbors_skip_obstacle(self):
        """Neighbors of (2,1) should not include obstacle at (2,2)."""
        neighbors = self.grid.neighbors_4dir((2, 1))
        positions = [pos for pos, _ in neighbors]
        self.assertNotIn((2, 2), positions)


class TestAstar(unittest.TestCase):
    """Test A* algorithm correctness."""

    def test_trivial_path(self):
        """Start == goal should return path of length 1."""
        grid = Grid(5, 5)
        result = astar(grid, (2, 2), (2, 2))
        self.assertIsNotNone(result)
        path, cost, _ = result
        self.assertEqual(path, [(2, 2)])
        self.assertAlmostEqual(cost, 0.0)

    def test_straight_line_4dir(self):
        """Open grid, 4-dir: path from (0,0) to (0,4) should cost 4."""
        grid = Grid(5, 5)
        result = astar(grid, (0, 0), (0, 4), heuristic_name="manhattan")
        self.assertIsNotNone(result)
        path, cost, _ = result
        self.assertAlmostEqual(cost, 4.0)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (0, 4))

    def test_path_around_wall(self):
        """A* should find a path around a wall obstacle."""
        # Wall from row 0-3 at col 3
        obstacles = {(r, 3) for r in range(4)}
        grid = Grid(7, 5, obstacles)
        result = astar(grid, (2, 1), (2, 5), heuristic_name="manhattan")
        self.assertIsNotNone(result)
        path, cost, _ = result
        self.assertEqual(path[0], (2, 1))
        self.assertEqual(path[-1], (2, 5))
        # Path should not go through any obstacle
        for pos in path:
            self.assertNotIn(pos, obstacles)

    def test_no_path(self):
        """Complete wall should result in no path."""
        obstacles = {(r, 5) for r in range(10)}
        grid = Grid(10, 10, obstacles)
        result = astar(grid, (5, 2), (5, 8))
        self.assertIsNone(result)

    def test_optimal_cost_4dir(self):
        """Verify A* finds the optimal cost on a known grid."""
        # Open 5x5 grid: optimal 4-dir cost from (0,0) to (4,4) is 8
        grid = Grid(5, 5)
        result = astar(grid, (0, 0), (4, 4), heuristic_name="manhattan")
        self.assertIsNotNone(result)
        _, cost, _ = result
        self.assertAlmostEqual(cost, 8.0)

    def test_optimal_cost_8dir(self):
        """8-dir on open grid: (0,0) to (4,4) should cost 4*sqrt(2)."""
        grid = Grid(5, 5)
        result = astar(grid, (0, 0), (4, 4), heuristic_name="octile", eight_directional=True)
        self.assertIsNotNone(result)
        _, cost, _ = result
        self.assertAlmostEqual(cost, 4 * math.sqrt(2), places=5)

    def test_different_heuristics_same_optimal_cost(self):
        """All admissible heuristics should yield the same optimal path cost."""
        grid = Grid(10, 10, obstacles={(r, 5) for r in range(8)})
        costs = []
        for h_name in ["manhattan", "euclidean"]:
            result = astar(grid, (4, 2), (4, 8), heuristic_name=h_name)
            self.assertIsNotNone(result)
            costs.append(result[1])
        # Both should find the same optimal cost
        self.assertAlmostEqual(costs[0], costs[1], places=5)

    def test_manhattan_explores_fewer_than_euclidean(self):
        """Manhattan (tighter heuristic) should explore fewer nodes than Euclidean for 4-dir."""
        grid = Grid(15, 15)
        r1 = astar(grid, (0, 0), (14, 14), heuristic_name="manhattan")
        r2 = astar(grid, (0, 0), (14, 14), heuristic_name="euclidean")
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        # Manhattan should explore <= Euclidean (tighter heuristic = fewer nodes)
        self.assertLessEqual(len(r1[2]), len(r2[2]))


if __name__ == "__main__":
    unittest.main()
