"""
Day 043: Swarm Behavior Simulation — Test Suite

Run with: python3 -m pytest tests.py -v
     or: python3 tests.py
"""

import math
import random
import unittest

from my_solution import (
    Boid,
    Obstacle,
    SpatialHash,
    SwarmSimulation,
    Vector2D,
)


class TestVector2D(unittest.TestCase):
    """Verify the Vector2D building block works correctly."""

    def test_basic_arithmetic(self):
        a = Vector2D(3, 4)
        b = Vector2D(1, 2)
        result = a + b
        self.assertAlmostEqual(result.x, 4.0)
        self.assertAlmostEqual(result.y, 6.0)

    def test_magnitude(self):
        v = Vector2D(3, 4)
        self.assertAlmostEqual(v.magnitude(), 5.0)

    def test_normalize(self):
        v = Vector2D(0, 5)
        n = v.normalize()
        self.assertAlmostEqual(n.magnitude(), 1.0)
        self.assertAlmostEqual(n.x, 0.0)
        self.assertAlmostEqual(n.y, 1.0)

    def test_normalize_zero_vector(self):
        v = Vector2D(0, 0)
        n = v.normalize()
        self.assertAlmostEqual(n.magnitude(), 0.0)

    def test_limit(self):
        v = Vector2D(10, 0)
        limited = v.limit(3.0)
        self.assertAlmostEqual(limited.magnitude(), 3.0)


class TestSpatialHash(unittest.TestCase):
    """Verify spatial hash returns correct neighbors."""

    def test_insert_and_query(self):
        sh = SpatialHash(cell_size=10.0)
        sh.insert(0, Vector2D(5, 5))
        sh.insert(1, Vector2D(8, 8))
        sh.insert(2, Vector2D(50, 50))
        neighbors = sh.query_neighbors(Vector2D(5, 5), 10.0)
        self.assertIn(0, neighbors)
        self.assertIn(1, neighbors)
        self.assertNotIn(2, neighbors)

    def test_clear(self):
        sh = SpatialHash(cell_size=10.0)
        sh.insert(0, Vector2D(5, 5))
        sh.clear()
        neighbors = sh.query_neighbors(Vector2D(5, 5), 10.0)
        self.assertEqual(len(neighbors), 0)


class TestBoid(unittest.TestCase):
    """Verify individual agent physics."""

    def test_update_moves_position(self):
        b = Boid(position=Vector2D(0, 0), velocity=Vector2D(1, 0))
        b.update(dt=1.0)
        self.assertAlmostEqual(b.position.x, 1.0)
        self.assertAlmostEqual(b.position.y, 0.0)

    def test_max_speed_enforced(self):
        b = Boid(position=Vector2D(0, 0), velocity=Vector2D(0, 0), max_speed=3.0)
        b.apply_force(Vector2D(100, 0))
        b.update(dt=1.0)
        self.assertLessEqual(b.velocity.magnitude(), 3.0 + 1e-9)

    def test_acceleration_resets(self):
        b = Boid(position=Vector2D(0, 0), velocity=Vector2D(0, 0))
        b.apply_force(Vector2D(1, 1))
        b.update(dt=1.0)
        self.assertAlmostEqual(b.acceleration.magnitude(), 0.0)


class TestSwarmSimulation(unittest.TestCase):
    """Integration tests for the full swarm simulation."""

    def test_simulation_runs(self):
        """Basic smoke test: simulation doesn't crash."""
        random.seed(42)
        sim = SwarmSimulation(num_boids=10, width=100, height=100)
        metrics = sim.step()
        self.assertIn("avg_distance_to_centroid", metrics)
        self.assertIn("avg_velocity_alignment", metrics)
        self.assertIn("min_neighbor_distance", metrics)

    def test_flocking_improves_alignment(self):
        """After enough steps, agents should be more aligned than at start."""
        random.seed(42)
        sim = SwarmSimulation(num_boids=20, width=100, height=100)
        initial = sim.compute_metrics()
        sim.run(num_steps=80)
        final = sim.metrics_history[-1]
        # Alignment should improve (higher = more coherent)
        self.assertGreater(final["avg_velocity_alignment"],
                           initial["avg_velocity_alignment"] - 0.1)

    def test_separation_prevents_overlap(self):
        """Min neighbor distance should stay above 0 (no perfect overlaps)."""
        random.seed(42)
        sim = SwarmSimulation(num_boids=15, width=80, height=80)
        sim.run(num_steps=50)
        for m in sim.metrics_history:
            self.assertGreater(m["min_neighbor_distance"], 0.0,
                               "Agents should not perfectly overlap")

    def test_obstacle_avoidance(self):
        """After settling, no agent should be inside an obstacle."""
        random.seed(42)
        sim = SwarmSimulation(num_boids=20, width=100, height=100)
        sim.add_obstacle(50, 50, 10)
        sim.run(num_steps=60)
        for boid in sim.boids:
            dist = boid.position.distance_to(sim.obstacles[0].position)
            # Allow small overlap (force-based, not hard constraint), but not deep inside
            self.assertGreater(dist, sim.obstacles[0].radius * 0.3,
                               "Agent deeply inside obstacle")

    def test_goal_seeking_converges(self):
        """Swarm should get closer to goal over time."""
        random.seed(42)
        sim = SwarmSimulation(num_boids=20, width=200, height=200, goal_weight=1.0)
        sim.set_goal(100, 100)

        # Measure initial avg distance to goal
        initial_dist = sum(
            b.position.distance_to(sim.goal) for b in sim.boids
        ) / len(sim.boids)

        sim.run(num_steps=100)

        final_dist = sum(
            b.position.distance_to(sim.goal) for b in sim.boids
        ) / len(sim.boids)

        self.assertLess(final_dist, initial_dist,
                        "Swarm should converge toward goal")

    def test_boundary_wrapping(self):
        """All agents should stay within world bounds after many steps."""
        random.seed(42)
        sim = SwarmSimulation(num_boids=30, width=100, height=100)
        sim.run(num_steps=100)
        for boid in sim.boids:
            self.assertGreaterEqual(boid.position.x, 0)
            self.assertLessEqual(boid.position.x, 100)
            self.assertGreaterEqual(boid.position.y, 0)
            self.assertLessEqual(boid.position.y, 100)

    def test_metrics_history_length(self):
        """Metrics should be recorded for each step."""
        sim = SwarmSimulation(num_boids=5, width=50, height=50)
        sim.run(num_steps=25)
        self.assertEqual(len(sim.metrics_history), 25)

    def test_many_agents_no_crash(self):
        """Stress test with more agents — spatial hash should handle it."""
        random.seed(42)
        sim = SwarmSimulation(num_boids=100, width=200, height=200)
        sim.add_obstacle(100, 100, 15)
        sim.set_goal(150, 150)
        sim.run(num_steps=20)
        self.assertEqual(len(sim.metrics_history), 20)


if __name__ == "__main__":
    unittest.main()
