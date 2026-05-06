"""
Day 034: RRT Path Planning — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import random
import unittest
from my_solution import (
    Point, CircleObstacle, TreeNode, Environment, PlannerConfig,
    is_edge_collision_free, sample_random_point, find_nearest,
    steer, find_near_nodes, extract_path, smooth_path, path_length,
    rrt, rrt_star, create_demo_environment,
)


class TestPoint(unittest.TestCase):
    """Test the Point data structure."""

    def test_distance_to_same(self):
        p = Point(3, 4)
        self.assertAlmostEqual(p.distance_to(p), 0.0)

    def test_distance_to_origin(self):
        p = Point(3, 4)
        o = Point(0, 0)
        self.assertAlmostEqual(p.distance_to(o), 5.0)


class TestCircleObstacle(unittest.TestCase):
    """Test obstacle collision detection."""

    def test_point_inside(self):
        obs = CircleObstacle(10, 10, 5)
        self.assertTrue(obs.contains(Point(10, 10)))  # Center
        self.assertTrue(obs.contains(Point(12, 10)))  # Inside

    def test_point_outside(self):
        obs = CircleObstacle(10, 10, 5)
        self.assertFalse(obs.contains(Point(20, 20)))  # Far outside

    def test_point_on_boundary(self):
        obs = CircleObstacle(0, 0, 5)
        self.assertTrue(obs.contains(Point(5, 0)))  # On boundary = inside


class TestEnvironment(unittest.TestCase):
    """Test environment free-space checking."""

    def setUp(self):
        self.env = Environment(0, 0, 50, 50, [CircleObstacle(25, 25, 5)])

    def test_in_bounds(self):
        self.assertTrue(self.env.in_bounds(Point(0, 0)))
        self.assertTrue(self.env.in_bounds(Point(50, 50)))
        self.assertFalse(self.env.in_bounds(Point(-1, 25)))

    def test_is_free(self):
        self.assertTrue(self.env.is_free(Point(0, 0)))      # Free corner
        self.assertFalse(self.env.is_free(Point(25, 25)))    # In obstacle
        self.assertFalse(self.env.is_free(Point(-1, -1)))    # Out of bounds


class TestCollisionDetection(unittest.TestCase):
    """Test edge collision checking."""

    def setUp(self):
        self.env = Environment(0, 0, 50, 50, [CircleObstacle(25, 25, 5)])

    def test_free_edge(self):
        # Edge far from obstacle
        self.assertTrue(is_edge_collision_free(
            self.env, Point(0, 0), Point(10, 0)))

    def test_blocked_edge(self):
        # Edge goes straight through obstacle
        self.assertFalse(is_edge_collision_free(
            self.env, Point(0, 25), Point(50, 25)))

    def test_edge_around_obstacle(self):
        # Edge that goes around the obstacle (above it)
        self.assertTrue(is_edge_collision_free(
            self.env, Point(0, 45), Point(50, 45)))


class TestSteer(unittest.TestCase):
    """Test the steering function."""

    def test_steer_within_step(self):
        """If target is closer than step_size, go directly there."""
        result = steer(Point(0, 0), Point(1, 0), step_size=5.0)
        self.assertAlmostEqual(result.x, 1.0, places=5)
        self.assertAlmostEqual(result.y, 0.0, places=5)

    def test_steer_beyond_step(self):
        """If target is farther than step_size, move exactly step_size."""
        result = steer(Point(0, 0), Point(10, 0), step_size=3.0)
        self.assertAlmostEqual(result.x, 3.0, places=5)
        self.assertAlmostEqual(result.y, 0.0, places=5)

    def test_steer_diagonal(self):
        """Steering diagonally should maintain correct distance."""
        result = steer(Point(0, 0), Point(10, 10), step_size=2.0)
        dist = Point(0, 0).distance_to(result)
        self.assertAlmostEqual(dist, 2.0, places=5)


class TestFindNearest(unittest.TestCase):
    """Test nearest neighbor search."""

    def test_finds_closest(self):
        nodes = [
            TreeNode(Point(0, 0)),
            TreeNode(Point(10, 10)),
            TreeNode(Point(5, 5)),
        ]
        idx = find_nearest(nodes, Point(4, 4))
        self.assertEqual(idx, 2)  # (5,5) is closest to (4,4)

    def test_single_node(self):
        nodes = [TreeNode(Point(7, 7))]
        self.assertEqual(find_nearest(nodes, Point(100, 100)), 0)


class TestExtractPath(unittest.TestCase):
    """Test path extraction from tree."""

    def test_simple_path(self):
        nodes = [
            TreeNode(Point(0, 0), parent=None),
            TreeNode(Point(1, 1), parent=0),
            TreeNode(Point(2, 2), parent=1),
        ]
        path = extract_path(nodes, 2)
        self.assertEqual(len(path), 3)
        self.assertAlmostEqual(path[0].x, 0.0)
        self.assertAlmostEqual(path[-1].x, 2.0)

    def test_root_only(self):
        nodes = [TreeNode(Point(5, 5), parent=None)]
        path = extract_path(nodes, 0)
        self.assertEqual(len(path), 1)


class TestRRT(unittest.TestCase):
    """Test the full RRT planner."""

    def test_rrt_finds_path_easy(self):
        """RRT should find a path in an open environment."""
        random.seed(42)
        env = Environment(0, 0, 50, 50, [])  # No obstacles
        start, goal = Point(5, 5), Point(45, 45)
        config = PlannerConfig(step_size=3.0, goal_threshold=3.0, goal_bias=0.1)
        path, tree, iters = rrt(env, start, goal, config)
        self.assertTrue(len(path) > 0, "RRT should find path in open space")
        self.assertAlmostEqual(path[0].x, start.x, places=1)
        self.assertAlmostEqual(path[-1].x, goal.x, places=1)

    def test_rrt_finds_path_with_obstacles(self):
        """RRT should find a path in the demo environment."""
        random.seed(42)
        env, start, goal = create_demo_environment()
        config = PlannerConfig(
            step_size=2.0, goal_threshold=2.0,
            goal_bias=0.1, max_iterations=8000
        )
        path, tree, iters = rrt(env, start, goal, config)
        self.assertTrue(len(path) > 0, "RRT should find path in demo env")

    def test_rrt_path_is_collision_free(self):
        """The returned path must not pass through obstacles."""
        random.seed(42)
        env, start, goal = create_demo_environment()
        config = PlannerConfig(
            step_size=2.0, goal_threshold=2.0,
            goal_bias=0.1, max_iterations=8000
        )
        path, _, _ = rrt(env, start, goal, config)
        if path:
            for i in range(len(path) - 1):
                self.assertTrue(
                    is_edge_collision_free(env, path[i], path[i + 1]),
                    f"Path segment {i} collides with obstacle"
                )


class TestRRTStar(unittest.TestCase):
    """Test the RRT* planner."""

    def test_rrt_star_finds_path(self):
        """RRT* should find a path in the demo environment."""
        random.seed(42)
        env, start, goal = create_demo_environment()
        config = PlannerConfig(
            step_size=2.0, goal_threshold=2.0,
            goal_bias=0.1, max_iterations=8000,
            rewire_radius=5.0
        )
        path, tree, iters = rrt_star(env, start, goal, config)
        self.assertTrue(len(path) > 0, "RRT* should find path")


if __name__ == "__main__":
    unittest.main()
