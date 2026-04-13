"""
Day 011: Robot That Follows ML-Detected Objects

YOUR TASK: Implement a simulated robot that uses ML-based object detection
to identify and follow a target in a 2D environment.

This integrates three concepts:
- KNN classification (Day 009) for detecting objects
- PID control (Day 006) for smooth following
- State machines (Day 007) for robust behavior

Work through the classes in order — each builds on the previous.
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# =============================================================================
# Vec2 — 2D vector helper (provided, no changes needed)
# =============================================================================

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).length()


# =============================================================================
# World objects (provided, no changes needed)
# =============================================================================

class ObjectType(Enum):
    TARGET = "target"
    OBSTACLE = "obstacle"
    DISTRACTOR = "distractor"


@dataclass
class WorldObject:
    obj_id: int
    obj_type: ObjectType
    position: Vec2
    features: list[float] = field(default_factory=list)
    path_center: Vec2 = field(default_factory=Vec2)
    path_radius: float = 0.0
    path_speed: float = 0.0
    path_phase: float = 0.0

    def update(self, t: float) -> None:
        if self.path_radius > 0:
            angle = self.path_phase + self.path_speed * t
            self.position = Vec2(
                self.path_center.x + self.path_radius * math.cos(angle),
                self.path_center.y + self.path_radius * math.sin(angle),
            )


@dataclass
class Detection:
    position: Vec2
    predicted_class: str
    confidence: float
    obj_id: int


# =============================================================================
# TODO 1: KNN Classifier
# Hint: This is the same algorithm from Day 009 — euclidean distance,
# K nearest neighbors, majority vote. Nothing new here, just apply it.
# =============================================================================

class KNNClassifier:
    """Classify objects by their feature vectors using K-Nearest Neighbors."""

    def __init__(self, k: int = 3):
        self.k = k
        self.training_data: list[tuple[list[float], str]] = []

    def fit(self, features: list[list[float]], labels: list[str]) -> None:
        """Store training data. KNN is lazy — no model built at fit time.

        Args:
            features: List of feature vectors
            labels: Corresponding class labels
        """
        raise NotImplementedError("TODO: implement this")

    def _euclidean_distance(self, a: list[float], b: list[float]) -> float:
        """Compute euclidean distance between two feature vectors."""
        raise NotImplementedError("TODO: implement this")

    def predict(self, features: list[float]) -> str:
        """Predict the class label for a feature vector.

        Steps:
        1. Compute distance to all training examples
        2. Find K nearest
        3. Return majority vote

        Returns:
            Predicted class label string
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# TODO 2: Object Detector
# Hint: Loop through objects in range, classify with KNN, add noise.
# The noise simulation makes this realistic — positions jitter,
# detections are missed, phantoms appear.
# =============================================================================

class ObjectDetector:
    """Simulates an ML-based object detector with realistic noise."""

    def __init__(
        self,
        classifier: KNNClassifier,
        detection_range: float = 15.0,
        position_noise_std: float = 0.3,
        false_negative_rate: float = 0.1,
        false_positive_rate: float = 0.05,
    ):
        self.classifier = classifier
        self.detection_range = detection_range
        self.position_noise_std = position_noise_std
        self.false_negative_rate = false_negative_rate
        self.false_positive_rate = false_positive_rate

    def detect(
        self, robot_pos: Vec2, objects: list[WorldObject]
    ) -> list[Detection]:
        """Run detection on all objects within range.

        For each object in range:
        1. Check if within detection_range of robot_pos
        2. Randomly skip (false negative) with probability false_negative_rate
        3. Classify using self.classifier.predict(obj.features)
        4. Add gaussian noise to position (scale with distance)
        5. Compute confidence (decreases with distance)

        Also: with probability false_positive_rate, add a phantom detection
        at a random nearby position.

        Returns:
            List of Detection objects
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# TODO 3: Tracker (EMA Filter)
# Hint: Exponential moving average smooths noisy detections.
# new = alpha * measurement + (1 - alpha) * old
# Track confidence and consecutive detections/misses for the state machine.
# =============================================================================

@dataclass
class TrackedTarget:
    """Maintains a smoothed position estimate using EMA filtering."""
    position: Vec2 = field(default_factory=Vec2)
    confidence: float = 0.0
    consecutive_detections: int = 0
    consecutive_misses: int = 0
    alpha: float = 0.4

    def update_with_detection(self, detection: Detection) -> None:
        """Update with a new detection. Apply EMA to smooth position.

        First detection: initialize position directly.
        Subsequent: blend with EMA formula.
        Increase confidence (cap at 1.0), increment consecutive_detections,
        reset consecutive_misses.
        """
        raise NotImplementedError("TODO: implement this")

    def update_no_detection(self) -> None:
        """No detection this frame. Decay confidence, increment misses.

        Don't move the position — hold at last known location.
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# TODO 4: PID Controller
# Hint: Same as Day 006. P = Kp*error, I = Ki*integral, D = Kd*derivative.
# Don't forget integral windup clamping and output clamping.
# =============================================================================

class PIDController:
    """PID controller for heading and distance control."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -float("inf"),
        output_max: float = float("inf"),
        integral_max: float = 10.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max
        self.integral: float = 0.0
        self.prev_error: Optional[float] = None

    def compute(self, error: float, dt: float) -> float:
        """Compute PID output.

        Args:
            error: Current error (setpoint - measurement)
            dt: Timestep in seconds

        Returns:
            Control output (clamped to [output_min, output_max])
        """
        raise NotImplementedError("TODO: implement this")

    def reset(self) -> None:
        """Reset integral and derivative state."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Robot state machine and robot model (provided)
# =============================================================================

class RobotState(Enum):
    SEARCHING = auto()
    ACQUIRING = auto()
    FOLLOWING = auto()
    LOST = auto()


@dataclass
class Robot:
    position: Vec2 = field(default_factory=Vec2)
    heading: float = 0.0
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    max_linear_vel: float = 3.0
    max_angular_vel: float = 2.0
    search_angular_vel: float = 0.8

    def update(self, dt: float) -> None:
        self.linear_vel = max(-self.max_linear_vel,
                              min(self.max_linear_vel, self.linear_vel))
        self.angular_vel = max(-self.max_angular_vel,
                               min(self.max_angular_vel, self.angular_vel))
        self.heading += self.angular_vel * dt
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        self.position.x += self.linear_vel * math.cos(self.heading) * dt
        self.position.y += self.linear_vel * math.sin(self.heading) * dt


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# =============================================================================
# TODO 5: FollowerSimulation — the main integration
# Hint: Wire together all the pieces. The step() method is the core loop:
# move objects -> detect -> track -> state machine -> PID -> move robot
# =============================================================================

class FollowerSimulation:
    """Complete perception-action loop simulation."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.dt = 0.1
        self.time = 0.0
        self.follow_distance = 3.0

        self.objects = self._create_world()
        self.robot = Robot(position=Vec2(0, 0), heading=0)

        classifier = self._train_classifier()
        self.detector = ObjectDetector(classifier=classifier)
        self.tracker = TrackedTarget(alpha=0.4)

        self.heading_pid = PIDController(kp=2.0, ki=0.0, kd=0.5,
                                          output_min=-2.0, output_max=2.0)
        self.distance_pid = PIDController(kp=0.8, ki=0.1, kd=0.3,
                                           output_min=-1.0, output_max=3.0)

        self.state = RobotState.SEARCHING
        self.acquire_threshold = 3
        self.lost_grace_frames = 15

        self.state_log: list[tuple[float, str]] = []
        self.distance_errors: list[float] = []

    def _create_world(self) -> list[WorldObject]:
        """Create world with 1 target, 2 distractors, 2 obstacles.

        Target: red, medium, round — moves in a circle around (5, 5)
        Distractors: warm colors (similar to target — tests classifier)
        Obstacles: cool colors, large, angular (stationary)
        """
        raise NotImplementedError("TODO: implement this — create 5 WorldObjects")

    def _train_classifier(self) -> KNNClassifier:
        """Train KNN on synthetic data for 3 classes.

        Generate ~20 examples per class with gaussian noise around
        the class centroids. Target: red/medium/round. Distractor: warm/varied.
        Obstacle: cool/large/angular.
        """
        raise NotImplementedError("TODO: implement this — generate training data and fit KNN")

    def _find_target_detection(self, detections: list[Detection]) -> Optional[Detection]:
        """Find the highest-confidence 'target' detection, or None."""
        raise NotImplementedError("TODO: implement this")

    def _run_state_machine(self, target_detection: Optional[Detection]) -> None:
        """Update behavioral state based on detection.

        Transitions:
        - SEARCHING -> ACQUIRING: first detection
        - ACQUIRING -> FOLLOWING: N consecutive detections
        - ACQUIRING -> SEARCHING: confidence drops to 0
        - FOLLOWING -> LOST: 3+ consecutive misses
        - LOST -> FOLLOWING: detection recovers
        - LOST -> SEARCHING: grace period expires

        Remember to reset PIDs on appropriate transitions!
        """
        raise NotImplementedError("TODO: implement this")

    def _compute_commands(self) -> tuple[float, float]:
        """Compute (linear_vel, angular_vel) based on current state.

        SEARCHING: rotate in place (0, search_angular_vel)
        ACQUIRING: slowly turn toward target
        FOLLOWING: heading PID + distance PID
        LOST: same as following but at 50% speed
        """
        raise NotImplementedError("TODO: implement this")

    def step(self) -> dict:
        """One simulation timestep: perceive -> decide -> act.

        Returns dict with time, state, positions, distance, detection info.
        """
        raise NotImplementedError("TODO: implement this")

    def run(self, duration: float = 30.0) -> list[dict]:
        """Run simulation for the specified duration."""
        steps = int(duration / self.dt)
        results = []
        for _ in range(steps):
            results.append(self.step())
        return results


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    print("Day 011: Robot That Follows ML-Detected Objects")
    print("=" * 50)

    # Test 1: KNN Classifier
    print("\n[Test 1] KNN Classifier")
    knn = KNNClassifier(k=3)
    knn.fit(
        [[0.9, 0.1, 0.1], [0.85, 0.15, 0.05], [0.1, 0.1, 0.9], [0.15, 0.05, 0.85]],
        ["target", "target", "obstacle", "obstacle"],
    )
    print(f"  Predict [0.88, 0.12, 0.08]: {knn.predict([0.88, 0.12, 0.08])}")  # should be 'target'
    print(f"  Predict [0.12, 0.08, 0.88]: {knn.predict([0.12, 0.08, 0.88])}")  # should be 'obstacle'

    # Test 2: PID Controller
    print("\n[Test 2] PID Controller")
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05, output_min=-5, output_max=5)
    for i in range(5):
        error = 10.0 - i * 2.5
        output = pid.compute(error, 0.1)
        print(f"  Error={error:5.1f}  Output={output:6.3f}")

    # Test 3: Tracker
    print("\n[Test 3] EMA Tracker")
    tracker = TrackedTarget(alpha=0.4)
    for i in range(5):
        det = Detection(Vec2(10 + random.gauss(0, 0.5), 5 + random.gauss(0, 0.5)),
                        "target", 0.9, 0)
        tracker.update_with_detection(det)
        print(f"  Detection {i+1}: smoothed=({tracker.position.x:.2f}, {tracker.position.y:.2f}), "
              f"confidence={tracker.confidence:.2f}")

    # Test 4: Full simulation
    print("\n[Test 4] Full Simulation (30s)")
    sim = FollowerSimulation(seed=42)
    results = sim.run(30.0)
    print(f"  Final state: {results[-1]['state']}")
    print(f"  State transitions: {len(sim.state_log)}")
    following_time = sum(sim.dt for r in results if r["state"] == "FOLLOWING")
    print(f"  Time following: {following_time:.1f}s / 30.0s")
    if sim.distance_errors:
        avg_err = sum(sim.distance_errors) / len(sim.distance_errors)
        print(f"  Avg distance error: {avg_err:.3f}")

    print("\nAll tests passed!" if sim.distance_errors else "\nSimulation ran but check results.")
