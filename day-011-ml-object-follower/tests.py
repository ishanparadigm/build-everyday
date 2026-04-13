"""
Day 011: Tests for Robot That Follows ML-Detected Objects

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import random
import unittest

from my_solution import (
    Vec2,
    KNNClassifier,
    ObjectDetector,
    TrackedTarget,
    PIDController,
    Detection,
    WorldObject,
    ObjectType,
    Robot,
    RobotState,
    FollowerSimulation,
    normalize_angle,
)


class TestKNNClassifier(unittest.TestCase):
    """Test the KNN classifier used for object detection."""

    def setUp(self):
        self.knn = KNNClassifier(k=3)
        # Simple 2-class problem: targets are high in feature 0, obstacles high in feature 1
        self.knn.fit(
            [[0.9, 0.1], [0.85, 0.15], [0.95, 0.05],
             [0.1, 0.9], [0.15, 0.85], [0.05, 0.95]],
            ["target", "target", "target",
             "obstacle", "obstacle", "obstacle"],
        )

    def test_classify_target(self):
        """Points near target cluster should be classified as target."""
        self.assertEqual(self.knn.predict([0.88, 0.12]), "target")

    def test_classify_obstacle(self):
        """Points near obstacle cluster should be classified as obstacle."""
        self.assertEqual(self.knn.predict([0.12, 0.88]), "obstacle")

    def test_boundary_case(self):
        """Point equidistant from both clusters — should still return a valid label."""
        result = self.knn.predict([0.5, 0.5])
        self.assertIn(result, ["target", "obstacle"])

    def test_euclidean_distance(self):
        """Euclidean distance computation should be correct."""
        dist = self.knn._euclidean_distance([0, 0], [3, 4])
        self.assertAlmostEqual(dist, 5.0)


class TestObjectDetector(unittest.TestCase):
    """Test the noisy ML detector."""

    def setUp(self):
        knn = KNNClassifier(k=3)
        knn.fit(
            [[0.9, 0.1, 0.1, 0.5, 0.9]] * 10 + [[0.1, 0.1, 0.9, 0.8, 0.2]] * 10,
            ["target"] * 10 + ["obstacle"] * 10,
        )
        self.detector = ObjectDetector(
            classifier=knn,
            detection_range=15.0,
            position_noise_std=0.3,
            false_negative_rate=0.0,  # Disable for deterministic tests
            false_positive_rate=0.0,
        )

    def test_detects_objects_in_range(self):
        """Should detect objects within detection range."""
        robot_pos = Vec2(0, 0)
        obj = WorldObject(0, ObjectType.TARGET, Vec2(5, 0),
                          features=[0.9, 0.1, 0.1, 0.5, 0.9])
        random.seed(42)
        detections = self.detector.detect(robot_pos, [obj])
        self.assertEqual(len(detections), 1)

    def test_ignores_objects_out_of_range(self):
        """Should not detect objects beyond detection range."""
        robot_pos = Vec2(0, 0)
        obj = WorldObject(0, ObjectType.TARGET, Vec2(100, 0),
                          features=[0.9, 0.1, 0.1, 0.5, 0.9])
        detections = self.detector.detect(robot_pos, [obj])
        self.assertEqual(len(detections), 0)

    def test_detection_has_noise(self):
        """Detected position should differ from true position (noise added)."""
        robot_pos = Vec2(0, 0)
        obj = WorldObject(0, ObjectType.TARGET, Vec2(5, 0),
                          features=[0.9, 0.1, 0.1, 0.5, 0.9])
        random.seed(123)
        det = self.detector.detect(robot_pos, [obj])[0]
        # Position should be close but not exactly (5, 0)
        self.assertNotAlmostEqual(det.position.x, 5.0, places=5)


class TestTrackedTarget(unittest.TestCase):
    """Test the EMA tracking filter."""

    def test_first_detection_initializes(self):
        """First detection should set position directly, not blend."""
        tracker = TrackedTarget(alpha=0.4)
        det = Detection(Vec2(10, 5), "target", 0.9, 0)
        tracker.update_with_detection(det)
        self.assertAlmostEqual(tracker.position.x, 10.0)
        self.assertAlmostEqual(tracker.position.y, 5.0)

    def test_ema_smoothing(self):
        """EMA should smooth noisy detections toward the true position."""
        tracker = TrackedTarget(alpha=0.4)
        # Initialize
        tracker.update_with_detection(Detection(Vec2(10, 0), "target", 0.9, 0))
        # New detection at (12, 0) — EMA should blend
        tracker.update_with_detection(Detection(Vec2(12, 0), "target", 0.9, 0))
        # Expected: 0.4 * 12 + 0.6 * 10 = 10.8
        self.assertAlmostEqual(tracker.position.x, 10.8)

    def test_confidence_decay(self):
        """Confidence should decay when no detections arrive."""
        tracker = TrackedTarget(alpha=0.4)
        tracker.update_with_detection(Detection(Vec2(10, 0), "target", 0.9, 0))
        initial_conf = tracker.confidence
        tracker.update_no_detection()
        self.assertLess(tracker.confidence, initial_conf)

    def test_consecutive_tracking(self):
        """consecutive_detections should increment, consecutive_misses should reset."""
        tracker = TrackedTarget(alpha=0.4)
        for i in range(5):
            tracker.update_with_detection(Detection(Vec2(10, 0), "target", 0.9, 0))
        self.assertEqual(tracker.consecutive_detections, 5)
        self.assertEqual(tracker.consecutive_misses, 0)


class TestPIDController(unittest.TestCase):
    """Test PID controller computation."""

    def test_proportional_only(self):
        """With Ki=Kd=0, output should be proportional to error."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
        output = pid.compute(5.0, 0.1)
        self.assertAlmostEqual(output, 10.0)

    def test_output_clamping(self):
        """Output should be clamped to [output_min, output_max]."""
        pid = PIDController(kp=10.0, ki=0.0, kd=0.0, output_min=-5, output_max=5)
        output = pid.compute(100.0, 0.1)
        self.assertEqual(output, 5.0)

    def test_integral_accumulation(self):
        """Integral should accumulate over multiple steps."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        pid.compute(1.0, 0.1)  # integral = 0.1
        output = pid.compute(1.0, 0.1)  # integral = 0.2
        self.assertAlmostEqual(output, 0.2)

    def test_reset(self):
        """Reset should clear integral and previous error."""
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
        pid.compute(5.0, 0.1)
        pid.reset()
        self.assertEqual(pid.integral, 0.0)
        self.assertIsNone(pid.prev_error)


class TestFollowerSimulation(unittest.TestCase):
    """Integration tests for the full simulation."""

    def test_simulation_runs(self):
        """Simulation should run without errors for 10 seconds."""
        sim = FollowerSimulation(seed=42)
        results = sim.run(10.0)
        self.assertEqual(len(results), 100)  # 10s at 10Hz

    def test_robot_eventually_follows(self):
        """Robot should enter FOLLOWING state within 10 seconds."""
        sim = FollowerSimulation(seed=42)
        results = sim.run(10.0)
        states = [r["state"] for r in results]
        self.assertIn("FOLLOWING", states)

    def test_follow_distance_reasonable(self):
        """When following, distance error should be < 5 units on average."""
        sim = FollowerSimulation(seed=42)
        sim.run(30.0)
        if sim.distance_errors:
            avg_err = sum(sim.distance_errors) / len(sim.distance_errors)
            self.assertLess(avg_err, 5.0)


class TestNormalizeAngle(unittest.TestCase):
    """Test angle normalization utility."""

    def test_already_normalized(self):
        self.assertAlmostEqual(normalize_angle(1.0), 1.0)

    def test_wrap_positive(self):
        result = normalize_angle(3 * math.pi)
        self.assertAlmostEqual(result, math.pi, places=5)

    def test_wrap_negative(self):
        result = normalize_angle(-3 * math.pi)
        self.assertAlmostEqual(result, -math.pi, places=5)


if __name__ == "__main__":
    unittest.main()
