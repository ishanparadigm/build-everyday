"""
Tests for Day 70: Autonomous Drone Path Planning

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import random
import unittest

from my_solution import (
    Point3D,
    RRTNode,
    BoxObstacle,
    CylinderObstacle,
    NoFlyZone,
    DronePhysics,
    DroneEnvironment,
    RRTStarPlanner,
    smooth_path,
    analyze_path,
)


class TestPoint3D(unittest.TestCase):
    """Test 3D point operations."""

    def test_distance_same_point(self):
        p = Point3D(1, 2, 3)
        self.assertAlmostEqual(p.distance_to(p), 0.0)

    def test_distance_known(self):
        a = Point3D(0, 0, 0)
        b = Point3D(3, 4, 0)
        self.assertAlmostEqual(a.distance_to(b), 5.0)

    def test_distance_3d(self):
        a = Point3D(1, 2, 3)
        b = Point3D(4, 6, 3)
        self.assertAlmostEqual(a.distance_to(b), 5.0)

    def test_horizontal_distance_ignores_z(self):
        a = Point3D(0, 0, 0)
        b = Point3D(3, 4, 100)  # z difference should be ignored
        self.assertAlmostEqual(a.horizontal_distance_to(b), 5.0)


class TestBoxObstacle(unittest.TestCase):
    """Test box obstacle containment."""

    def test_point_inside(self):
        box = BoxObstacle(0, 10, 0, 10, 0, 10, safety_margin=0)
        self.assertTrue(box.contains_point(Point3D(5, 5, 5)))

    def test_point_outside(self):
        box = BoxObstacle(0, 10, 0, 10, 0, 10, safety_margin=0)
        self.assertFalse(box.contains_point(Point3D(15, 5, 5)))

    def test_safety_margin(self):
        box = BoxObstacle(0, 10, 0, 10, 0, 10, safety_margin=2)
        # Point at (11, 5, 5) is outside the box but inside safety margin
        self.assertTrue(box.contains_point(Point3D(11, 5, 5)))
        # Point at (13, 5, 5) is outside even the safety margin
        self.assertFalse(box.contains_point(Point3D(13, 5, 5)))


class TestCylinderObstacle(unittest.TestCase):
    """Test cylinder obstacle containment."""

    def test_point_inside(self):
        cyl = CylinderObstacle(5, 5, 3, 0, 10, safety_margin=0)
        self.assertTrue(cyl.contains_point(Point3D(5, 5, 5)))

    def test_point_outside_radially(self):
        cyl = CylinderObstacle(5, 5, 3, 0, 10, safety_margin=0)
        self.assertFalse(cyl.contains_point(Point3D(5, 9, 5)))

    def test_point_above(self):
        cyl = CylinderObstacle(5, 5, 3, 0, 10, safety_margin=0)
        self.assertFalse(cyl.contains_point(Point3D(5, 5, 15)))


class TestDronePhysics(unittest.TestCase):
    """Test energy cost model."""

    def test_horizontal_cost(self):
        physics = DronePhysics(drag_coeff=1.0, wind_x=0, wind_y=0)
        a = Point3D(0, 0, 10)
        b = Point3D(10, 0, 10)  # Pure horizontal, 10m
        cost = physics.segment_cost(a, b)
        self.assertAlmostEqual(cost, 10.0)  # 10m * 1.0 drag

    def test_climb_costs_more_than_descent(self):
        physics = DronePhysics(drag_coeff=0, wind_x=0, wind_y=0)
        a = Point3D(0, 0, 0)
        up = Point3D(0, 0, 10)
        down = Point3D(0, 0, -10)
        climb_cost = physics.segment_cost(a, up)
        descend_cost = physics.segment_cost(a, down)
        self.assertGreater(climb_cost, descend_cost,
                           "Climbing should cost more than descending")

    def test_headwind_increases_cost(self):
        physics_no_wind = DronePhysics(wind_x=0, wind_y=0)
        physics_headwind = DronePhysics(wind_x=5.0, wind_y=0, wind_penalty=0.5)
        a = Point3D(0, 0, 10)
        # Flying in negative x direction (into the wind from +x)
        b = Point3D(-10, 0, 10)
        cost_no_wind = physics_no_wind.segment_cost(a, b)
        cost_headwind = physics_headwind.segment_cost(a, b)
        self.assertGreater(cost_headwind, cost_no_wind,
                           "Headwind should increase energy cost")


class TestDroneEnvironment(unittest.TestCase):
    """Test environment collision checking."""

    def setUp(self):
        self.env = DroneEnvironment(
            x_range=(0, 50), y_range=(0, 50), z_range=(0, 30)
        )
        self.env.add_obstacle(BoxObstacle(10, 20, 10, 20, 0, 15, safety_margin=0))

    def test_valid_point_in_free_space(self):
        self.assertTrue(self.env.is_valid_point(Point3D(5, 5, 5)))

    def test_invalid_point_in_obstacle(self):
        self.assertFalse(self.env.is_valid_point(Point3D(15, 15, 5)))

    def test_invalid_point_out_of_bounds(self):
        self.assertFalse(self.env.is_valid_point(Point3D(-5, 5, 5)))

    def test_collision_free_segment(self):
        a = Point3D(0, 0, 5)
        b = Point3D(5, 5, 5)  # Both in free space, no obstacle between
        self.assertTrue(self.env.is_collision_free_segment(a, b))

    def test_segment_through_obstacle(self):
        a = Point3D(5, 15, 5)
        b = Point3D(25, 15, 5)  # Passes through the box obstacle
        self.assertFalse(self.env.is_collision_free_segment(a, b))


class TestRRTStarPlanner(unittest.TestCase):
    """Test the RRT* planner end-to-end."""

    def test_simple_path_no_obstacles(self):
        """RRT* should find a path in an empty environment."""
        random.seed(42)
        env = DroneEnvironment(
            x_range=(0, 50), y_range=(0, 50), z_range=(0, 30)
        )
        physics = DronePhysics(wind_x=0, wind_y=0)
        planner = RRTStarPlanner(
            env=env, physics=physics,
            step_size=10.0, goal_threshold=5.0,
            max_iterations=500, goal_bias=0.15,
        )
        path = planner.plan(Point3D(5, 5, 10), Point3D(45, 45, 10))
        self.assertIsNotNone(path, "Should find a path in empty environment")
        self.assertGreaterEqual(len(path), 2)

    def test_path_avoids_obstacle(self):
        """Path should not pass through obstacles."""
        random.seed(42)
        env = DroneEnvironment(
            x_range=(0, 50), y_range=(0, 50), z_range=(0, 30)
        )
        env.add_obstacle(BoxObstacle(20, 30, 0, 50, 0, 30, safety_margin=0))
        physics = DronePhysics(wind_x=0, wind_y=0)
        planner = RRTStarPlanner(
            env=env, physics=physics,
            step_size=8.0, goal_threshold=5.0,
            max_iterations=2000, goal_bias=0.15,
        )
        path = planner.plan(Point3D(5, 25, 15), Point3D(45, 25, 15))
        self.assertIsNotNone(path)
        # Verify no waypoint is inside the obstacle
        for wp in path:
            self.assertFalse(
                env.obstacles[0].contains_point(wp),
                f"Waypoint {wp} is inside obstacle!"
            )


class TestPathSmoothing(unittest.TestCase):
    """Test path smoothing."""

    def test_smoothing_reduces_waypoints(self):
        """Smoothing should reduce unnecessary waypoints."""
        env = DroneEnvironment(
            x_range=(0, 50), y_range=(0, 50), z_range=(0, 30)
        )
        # Zigzag path that could be simplified
        path = [
            Point3D(0, 0, 10),
            Point3D(5, 3, 10),
            Point3D(10, 1, 10),
            Point3D(15, 4, 10),
            Point3D(20, 2, 10),
            Point3D(25, 5, 10),
            Point3D(30, 3, 10),
        ]
        smoothed = smooth_path(path, env, iterations=50)
        self.assertLessEqual(len(smoothed), len(path))
        # Start and end should be preserved
        self.assertAlmostEqual(smoothed[0].x, path[0].x)
        self.assertAlmostEqual(smoothed[-1].x, path[-1].x)


class TestPathAnalysis(unittest.TestCase):
    """Test path analysis computations."""

    def test_pure_horizontal(self):
        physics = DronePhysics(drag_coeff=1.0, wind_x=0, wind_y=0)
        path = [Point3D(0, 0, 10), Point3D(10, 0, 10)]
        stats = analyze_path(path, physics)
        self.assertAlmostEqual(stats["total_distance"], 10.0)
        self.assertAlmostEqual(stats["total_climb"], 0.0)
        self.assertAlmostEqual(stats["total_descent"], 0.0)
        self.assertEqual(stats["n_waypoints"], 2)

    def test_climb_and_descent(self):
        physics = DronePhysics(drag_coeff=0, wind_x=0, wind_y=0)
        path = [
            Point3D(0, 0, 0),
            Point3D(0, 0, 10),  # Climb 10m
            Point3D(0, 0, 5),   # Descend 5m
        ]
        stats = analyze_path(path, physics)
        self.assertAlmostEqual(stats["total_climb"], 10.0)
        self.assertAlmostEqual(stats["total_descent"], 5.0)
        self.assertAlmostEqual(stats["max_altitude"], 10.0)
        self.assertAlmostEqual(stats["min_altitude"], 0.0)


if __name__ == "__main__":
    unittest.main()
