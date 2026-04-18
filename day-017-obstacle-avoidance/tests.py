"""
Tests for Day 17: Obstacle Avoidance (VFH)

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import unittest

from my_solution import (
    Obstacle,
    Robot,
    VFHConfig,
    World,
    angular_distance,
    build_histogram,
    find_valleys,
    normalize_angle,
    run_simulation,
    sector_to_angle,
    select_direction,
)


class TestNormalizeAngle(unittest.TestCase):
    """Test angle normalization to [-pi, pi]."""

    def test_already_normalized(self):
        self.assertAlmostEqual(normalize_angle(0.5), 0.5)

    def test_positive_overflow(self):
        self.assertAlmostEqual(normalize_angle(3 * math.pi), math.pi, places=5)

    def test_negative_overflow(self):
        self.assertAlmostEqual(normalize_angle(-3 * math.pi), math.pi, places=5)

    def test_two_pi(self):
        result = normalize_angle(2 * math.pi)
        self.assertAlmostEqual(result, 0.0, places=5)


class TestAngularDistance(unittest.TestCase):
    """Test angular distance computation with wraparound."""

    def test_same_angle(self):
        self.assertAlmostEqual(angular_distance(1.0, 1.0), 0.0)

    def test_opposite_angles(self):
        self.assertAlmostEqual(angular_distance(0, math.pi), math.pi)

    def test_wraparound(self):
        # Almost the same direction but expressed differently
        d = angular_distance(0.1, 2 * math.pi - 0.1)
        self.assertAlmostEqual(d, 0.2, places=5)

    def test_symmetric(self):
        self.assertAlmostEqual(
            angular_distance(0.5, 1.5), angular_distance(1.5, 0.5)
        )


class TestRayCast(unittest.TestCase):
    """Test ray-obstacle intersection."""

    def test_ray_hits_obstacle(self):
        world = World(width=50, height=50, obstacles=[Obstacle(10, 0, 2.0)])
        dist = world.ray_cast(0, 0, 0, 20.0)  # Ray goes east, obstacle at x=10
        # Should hit the circle at x=8 (center=10, radius=2)
        self.assertAlmostEqual(dist, 8.0, places=1)

    def test_ray_misses_obstacle(self):
        world = World(width=50, height=50, obstacles=[Obstacle(10, 10, 1.0)])
        dist = world.ray_cast(0, 0, 0, 20.0)  # Ray goes east, obstacle far above
        # Should hit right wall or max range, not the obstacle
        self.assertGreater(dist, 15.0)

    def test_ray_hits_wall(self):
        world = World(width=20, height=20, obstacles=[])
        dist = world.ray_cast(10, 10, 0, 50.0)  # East from center
        self.assertAlmostEqual(dist, 10.0, places=1)

    def test_max_range_returned(self):
        world = World(width=100, height=100, obstacles=[])
        dist = world.ray_cast(50, 50, 0, 5.0)  # Short range, far from walls
        self.assertAlmostEqual(dist, 5.0, places=1)


class TestBuildHistogram(unittest.TestCase):
    """Test polar histogram construction."""

    def test_close_obstacle_high_weight(self):
        config = VFHConfig(num_sectors=72, a_const=5.0, b_const=0.5)
        # One reading at angle 0, very close
        readings = [(0.0, 1.0)]
        hist = build_histogram(readings, config, 8.0)
        # Weight should be 5.0 - 0.5*1.0 = 4.5, sector 0 should be high
        self.assertGreater(hist[0], 3.0)

    def test_far_obstacle_low_weight(self):
        config = VFHConfig(num_sectors=72, a_const=5.0, b_const=0.5)
        readings = [(0.0, 9.0)]
        hist = build_histogram(readings, config, 8.0)
        # Beyond max range, should be zero
        self.assertAlmostEqual(hist[0], 0.0)

    def test_empty_readings(self):
        config = VFHConfig(num_sectors=72)
        hist = build_histogram([], config, 8.0)
        self.assertTrue(all(h == 0.0 for h in hist))


class TestFindValleys(unittest.TestCase):
    """Test valley detection in histograms."""

    def test_all_free(self):
        config = VFHConfig(num_sectors=10, threshold=5.0, min_valley_width=1)
        histogram = [0.0] * 10
        valleys = find_valleys(histogram, config)
        # One big valley covering all sectors
        self.assertTrue(len(valleys) >= 1)

    def test_all_blocked(self):
        config = VFHConfig(num_sectors=10, threshold=5.0, min_valley_width=1)
        histogram = [10.0] * 10
        valleys = find_valleys(histogram, config)
        self.assertEqual(len(valleys), 0)

    def test_single_gap(self):
        config = VFHConfig(num_sectors=10, threshold=5.0, min_valley_width=3)
        histogram = [10.0] * 10
        histogram[3] = 0.0
        histogram[4] = 0.0
        histogram[5] = 0.0
        valleys = find_valleys(histogram, config)
        self.assertTrue(len(valleys) >= 1)
        # Valley should include sectors 3-5
        start, end = valleys[0]
        self.assertIn(4, range(start, end + 1))


class TestSelectDirection(unittest.TestCase):
    """Test steering direction selection."""

    def test_selects_toward_goal(self):
        config = VFHConfig(num_sectors=72, mu_target=5.0, mu_current=1.0, mu_previous=1.0)
        # One valley centered around sector 9 (45 degrees) and goal is at 45 degrees
        valleys = [(7, 11)]
        goal_angle = math.pi / 4  # 45 degrees
        direction = select_direction(valleys, goal_angle, 0.0, 0.0, config)
        self.assertIsNotNone(direction)
        # Should be close to goal direction
        self.assertLess(angular_distance(direction, goal_angle), math.pi / 4)

    def test_returns_none_when_no_valleys(self):
        config = VFHConfig()
        direction = select_direction([], 0.0, 0.0, 0.0, config)
        self.assertIsNone(direction)


class TestSectorToAngle(unittest.TestCase):
    """Test sector-to-angle conversion."""

    def test_first_sector(self):
        angle = sector_to_angle(0, 72)
        expected = (2 * math.pi / 72) / 2  # Center of first 5-degree sector
        self.assertAlmostEqual(angle, expected, places=5)

    def test_quarter_turn(self):
        angle = sector_to_angle(18, 72)  # 18 * 5 = 90 degrees + half sector
        expected = math.radians(90 + 2.5)
        self.assertAlmostEqual(angle, expected, places=3)


class TestEndToEnd(unittest.TestCase):
    """Test the full simulation runs and reaches the goal."""

    def test_reaches_goal_in_open_world(self):
        import random
        random.seed(42)
        world = World(width=30, height=30, obstacles=[])
        robot = Robot(x=5.0, y=5.0, heading=0.0)
        config = VFHConfig()
        path = run_simulation(world, robot, 25.0, 25.0, config, max_steps=200)
        # Should reach near the goal
        final_x, final_y = path[-1]
        dist = math.sqrt((final_x - 25.0) ** 2 + (final_y - 25.0) ** 2)
        self.assertLess(dist, 2.0)


if __name__ == "__main__":
    unittest.main()
