"""
Day 011: Robot That Follows ML-Detected Objects

Integration challenge combining:
- KNN classification (Day 009) for object detection
- PID control (Day 006) for target following
- State machines (Day 007) for robust behavior

Architecture:
  Environment (ground truth)
    -> Detector (KNN + noise)
    -> Tracker (EMA filter)
    -> State Machine (behavior logic)
    -> PID Controllers (heading + distance)
    -> Robot (actuators)
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# =============================================================================
# Vector2D helper — simplifies all the position/velocity math
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
        """Angle from origin to this point in radians."""
        return math.atan2(self.y, self.x)

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).length()


# =============================================================================
# World objects — things that exist in the simulation
# =============================================================================

class ObjectType(Enum):
    """Each object in the world has a type. The robot's job is to detect
    and follow TARGET objects while ignoring the rest."""
    TARGET = "target"
    OBSTACLE = "obstacle"
    DISTRACTOR = "distractor"


@dataclass
class WorldObject:
    """An object in the 2D world with features for ML classification."""
    obj_id: int
    obj_type: ObjectType
    position: Vec2
    # Features used for KNN classification (simulating visual features):
    # [color_r, color_g, color_b, size, shape_roundness]
    # These are normalized to [0, 1] range
    features: list[float] = field(default_factory=list)
    # Movement: objects follow circular or linear paths
    path_center: Vec2 = field(default_factory=Vec2)
    path_radius: float = 0.0
    path_speed: float = 0.0  # radians per second
    path_phase: float = 0.0  # starting angle on circular path

    def update(self, t: float) -> None:
        """Move the object along its path. Circular motion around path_center."""
        if self.path_radius > 0:
            angle = self.path_phase + self.path_speed * t
            self.position = Vec2(
                self.path_center.x + self.path_radius * math.cos(angle),
                self.path_center.y + self.path_radius * math.sin(angle),
            )


# =============================================================================
# KNN Classifier — reused concept from Day 009
# =============================================================================

class KNNClassifier:
    """Simple KNN for classifying detected objects by their features.

    In a real system, this would be a CNN operating on image patches.
    Here we classify based on feature vectors (color, size, shape) to
    demonstrate the same concept: spatial classification drives detection.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.training_data: list[tuple[list[float], str]] = []

    def fit(self, features: list[list[float]], labels: list[str]) -> None:
        """Store training examples. KNN is a lazy learner — no model is
        actually built. All computation happens at prediction time."""
        self.training_data = list(zip(features, labels))

    def _euclidean_distance(self, a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def predict(self, features: list[float]) -> str:
        """Classify a feature vector by majority vote of K nearest neighbors.

        Returns the predicted class label. In our case: 'target', 'obstacle',
        or 'distractor'.
        """
        # Compute distances to all training examples
        distances = [
            (self._euclidean_distance(features, train_feat), label)
            for train_feat, label in self.training_data
        ]
        # Sort by distance, take K closest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[: self.k]

        # Majority vote
        votes: dict[str, int] = {}
        for _, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1

        return max(votes, key=votes.get)  # type: ignore


# =============================================================================
# Detection — the perception layer
# =============================================================================

@dataclass
class Detection:
    """A single detection from the ML detector. Includes position (with noise),
    predicted class, and confidence score."""
    position: Vec2
    predicted_class: str
    confidence: float
    obj_id: int  # Ground truth ID for evaluation (not available in real systems)


class ObjectDetector:
    """Simulates an ML-based object detector.

    In reality, this would be a neural network processing camera frames.
    We simulate it by:
    1. Taking ground-truth object positions
    2. Classifying them with KNN
    3. Adding realistic noise (position jitter, false negatives, false positives)

    This captures the essential challenge: your detector is IMPERFECT, and
    your control system must handle that gracefully.
    """

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
        """Run 'detection' on all objects within range of the robot.

        This simulates what a real detector does:
        - Only sees objects within sensor range (camera FOV)
        - Adds noise to detected positions (pixel-to-world projection error)
        - Sometimes misses objects (false negatives)
        - Sometimes hallucinates objects (false positives)
        """
        detections: list[Detection] = []

        for obj in objects:
            dist = robot_pos.distance_to(obj.position)
            if dist > self.detection_range:
                continue  # Object outside sensor range

            # False negative: randomly miss this object
            if random.random() < self.false_negative_rate:
                continue

            # Classify the object using KNN on its feature vector
            predicted_class = self.classifier.predict(obj.features)

            # Add position noise — simulates real-world sensor inaccuracy
            # Noise increases with distance (farther objects are harder to localize)
            noise_scale = self.position_noise_std * (1 + dist / self.detection_range)
            noisy_pos = Vec2(
                obj.position.x + random.gauss(0, noise_scale),
                obj.position.y + random.gauss(0, noise_scale),
            )

            # Confidence decreases with distance (farther = less certain)
            confidence = max(0.3, 1.0 - (dist / self.detection_range) * 0.5)

            detections.append(Detection(
                position=noisy_pos,
                predicted_class=predicted_class,
                confidence=confidence,
                obj_id=obj.obj_id,
            ))

        # False positives: occasionally hallucinate a detection
        if random.random() < self.false_positive_rate:
            phantom_pos = Vec2(
                robot_pos.x + random.uniform(-5, 5),
                robot_pos.y + random.uniform(-5, 5),
            )
            detections.append(Detection(
                position=phantom_pos,
                predicted_class=random.choice(["target", "obstacle", "distractor"]),
                confidence=random.uniform(0.2, 0.5),
                obj_id=-1,  # Not a real object
            ))

        return detections


# =============================================================================
# Tracker — smooths noisy detections over time
# =============================================================================

@dataclass
class TrackedTarget:
    """Maintains a smoothed estimate of the target's position.

    Uses Exponential Moving Average (EMA) — a simple but effective filter.
    In production, you'd use a Kalman filter which also estimates velocity
    and provides uncertainty bounds.
    """
    position: Vec2 = field(default_factory=Vec2)
    confidence: float = 0.0
    consecutive_detections: int = 0
    consecutive_misses: int = 0
    alpha: float = 0.4  # EMA smoothing factor

    def update_with_detection(self, detection: Detection) -> None:
        """Update tracker with a new detection.

        The EMA formula: new = alpha * measurement + (1-alpha) * previous
        Alpha controls the tradeoff:
        - High alpha (0.8): responsive but noisy — good for fast targets
        - Low alpha (0.2): smooth but laggy — good for slow targets
        """
        if self.consecutive_detections == 0:
            # First detection — initialize directly instead of blending
            self.position = detection.position
        else:
            # EMA update — blend new detection with previous estimate
            self.position = Vec2(
                self.alpha * detection.position.x + (1 - self.alpha) * self.position.x,
                self.alpha * detection.position.y + (1 - self.alpha) * self.position.y,
            )
        self.confidence = min(1.0, self.confidence + 0.3)
        self.consecutive_detections += 1
        self.consecutive_misses = 0

    def update_no_detection(self) -> None:
        """No detection this frame. Decay confidence.

        We don't move the position estimate — it stays at the last known
        location. A Kalman filter would predict forward using velocity.
        """
        self.confidence = max(0.0, self.confidence - 0.15)
        self.consecutive_misses += 1
        # Don't reset consecutive_detections — that tracks the acquisition count


# =============================================================================
# PID Controller — reused from Day 006
# =============================================================================

class PIDController:
    """PID controller with integral windup protection and derivative filtering.

    Computes: output = Kp*e + Ki*integral(e) + Kd*de/dt

    Two instances are used:
    - Heading PID: controls angular velocity to point at target
    - Distance PID: controls linear velocity to maintain follow distance
    """

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
        """Compute PID output given current error and timestep.

        The three terms:
        - P (proportional): reacts to current error. Bigger error = bigger correction.
        - I (integral): accumulates past errors. Fixes steady-state offset.
        - D (derivative): reacts to rate of change. Dampens oscillation.
        """
        # Proportional term
        p = self.kp * error

        # Integral term with anti-windup clamping
        # Without clamping, the integral can grow unbounded during sustained
        # errors (e.g., searching state), causing massive overshoot later
        self.integral += error * dt
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        i = self.ki * self.integral

        # Derivative term
        if self.prev_error is not None:
            d = self.kd * (error - self.prev_error) / dt
        else:
            d = 0.0
        self.prev_error = error

        # Clamp output
        output = p + i + d
        return max(self.output_min, min(self.output_max, output))

    def reset(self) -> None:
        """Reset integral and derivative state. Called on state transitions
        to prevent stale accumulated error from causing control jumps."""
        self.integral = 0.0
        self.prev_error = None


# =============================================================================
# Robot State Machine — behavior management
# =============================================================================

class RobotState(Enum):
    """Behavioral states for the following robot.

    The state machine prevents jittery behavior. Without it, a single
    missed detection would cause the robot to immediately stop and spin,
    then jerk back into following when the next detection arrives.
    """
    SEARCHING = auto()   # No target, rotating to scan
    ACQUIRING = auto()   # Target detected, waiting for confirmation
    FOLLOWING = auto()   # Confirmed target, actively following
    LOST = auto()        # Target was being followed, temporarily lost


# =============================================================================
# Robot — the agent in the environment
# =============================================================================

@dataclass
class Robot:
    """A differential-drive robot with position, heading, and velocity.

    The robot receives linear and angular velocity commands and integrates
    them using simple Euler integration (fine for our discrete timestep).
    """
    position: Vec2 = field(default_factory=Vec2)
    heading: float = 0.0  # radians, 0 = facing right (+x)
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    max_linear_vel: float = 3.0
    max_angular_vel: float = 2.0  # rad/s
    search_angular_vel: float = 0.8  # rad/s when searching

    def update(self, dt: float) -> None:
        """Integrate velocity to update position and heading.

        Uses Euler integration: p(t+dt) = p(t) + v*dt
        This is first-order and accumulates error, but for small dt
        (0.1s) it's accurate enough for our simulation.
        """
        # Clamp velocities to physical limits
        self.linear_vel = max(-self.max_linear_vel,
                              min(self.max_linear_vel, self.linear_vel))
        self.angular_vel = max(-self.max_angular_vel,
                               min(self.max_angular_vel, self.angular_vel))

        # Update heading
        self.heading += self.angular_vel * dt
        # Wrap to [-pi, pi] to prevent floating point drift over long simulations
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))

        # Update position based on current heading
        self.position.x += self.linear_vel * math.cos(self.heading) * dt
        self.position.y += self.linear_vel * math.sin(self.heading) * dt


# =============================================================================
# Simulation — ties everything together
# =============================================================================

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]. Essential for PID heading control —
    without this, the robot might spin 350 degrees instead of -10 degrees."""
    return math.atan2(math.sin(angle), math.cos(angle))


class FollowerSimulation:
    """The complete perception-action loop simulation.

    Each timestep:
    1. Move world objects along their paths
    2. Run detector (KNN + noise) from robot's perspective
    3. Update tracker with new detections
    4. Run state machine to determine behavior
    5. Compute PID commands based on state
    6. Apply commands to robot
    7. Log metrics
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.dt = 0.1  # 10 Hz update rate — typical for robot control
        self.time = 0.0
        self.follow_distance = 3.0  # Desired distance to maintain from target

        # Build the world
        self.objects = self._create_world()
        self.robot = Robot(position=Vec2(0, 0), heading=0)

        # ML detector
        classifier = self._train_classifier()
        self.detector = ObjectDetector(
            classifier=classifier,
            detection_range=15.0,
            position_noise_std=0.3,
            false_negative_rate=0.10,
            false_positive_rate=0.05,
        )

        # Tracker
        self.tracker = TrackedTarget(alpha=0.4)

        # PID controllers
        # Heading PID: moderate gains, no integral (heading shouldn't have steady-state error)
        self.heading_pid = PIDController(kp=2.0, ki=0.0, kd=0.5,
                                          output_min=-2.0, output_max=2.0)
        # Distance PID: lower gains for smooth approach, some integral for steady-state
        self.distance_pid = PIDController(kp=0.8, ki=0.1, kd=0.3,
                                           output_min=-1.0, output_max=3.0)

        # State machine
        self.state = RobotState.SEARCHING
        self.acquire_threshold = 3  # Consecutive detections to confirm
        self.lost_grace_frames = 15  # Frames to wait before giving up

        # Metrics
        self.state_log: list[tuple[float, str]] = []
        self.distance_errors: list[float] = []
        self.positions: list[tuple[float, float, float, float]] = []  # robot_x, robot_y, target_x, target_y

    def _create_world(self) -> list[WorldObject]:
        """Create a world with one target and several distractors/obstacles.

        The target moves in a circle. Distractors have similar but not
        identical features — this tests the classifier's ability to
        distinguish the target from look-alikes.
        """
        objects = []

        # TARGET: red-ish, medium sized, round — moves in a circle
        objects.append(WorldObject(
            obj_id=0,
            obj_type=ObjectType.TARGET,
            position=Vec2(8, 0),
            features=[0.9, 0.1, 0.1, 0.5, 0.9],  # red, medium, round
            path_center=Vec2(5, 5),
            path_radius=6.0,
            path_speed=0.3,
            path_phase=0.0,
        ))

        # DISTRACTOR 1: orange-ish (similar to target!) — tests classifier
        objects.append(WorldObject(
            obj_id=1,
            obj_type=ObjectType.DISTRACTOR,
            position=Vec2(-5, 3),
            features=[0.8, 0.3, 0.1, 0.4, 0.7],  # orange, slightly smaller
            path_center=Vec2(-5, 3),
            path_radius=3.0,
            path_speed=-0.2,
            path_phase=1.0,
        ))

        # DISTRACTOR 2: pink-ish
        objects.append(WorldObject(
            obj_id=2,
            obj_type=ObjectType.DISTRACTOR,
            position=Vec2(10, -5),
            features=[0.7, 0.2, 0.5, 0.6, 0.8],  # pink, medium-large
            path_center=Vec2(10, -5),
            path_radius=2.0,
            path_speed=0.4,
            path_phase=2.0,
        ))

        # OBSTACLE 1: blue, large, square
        objects.append(WorldObject(
            obj_id=3,
            obj_type=ObjectType.OBSTACLE,
            position=Vec2(3, -4),
            features=[0.1, 0.1, 0.9, 0.8, 0.2],  # blue, large, square
            path_center=Vec2(3, -4),
            path_radius=0.0,  # Stationary
            path_speed=0.0,
            path_phase=0.0,
        ))

        # OBSTACLE 2: green, large, square
        objects.append(WorldObject(
            obj_id=4,
            obj_type=ObjectType.OBSTACLE,
            position=Vec2(-3, 8),
            features=[0.1, 0.8, 0.1, 0.7, 0.3],  # green, large, angular
            path_center=Vec2(-3, 8),
            path_radius=0.0,
            path_speed=0.0,
            path_phase=0.0,
        ))

        return objects

    def _train_classifier(self) -> KNNClassifier:
        """Train a KNN classifier on example object features.

        In a real system, this would be a pre-trained CNN. Here we generate
        synthetic training data that captures the distribution of features
        for each class. The classifier must learn that targets are red,
        medium-sized, and round — even when distractors are similar.
        """
        features = []
        labels = []

        # Generate training data for each class
        # Target class: red, medium, round
        for _ in range(20):
            features.append([
                0.85 + random.gauss(0, 0.08),  # red
                0.15 + random.gauss(0, 0.08),  # low green
                0.12 + random.gauss(0, 0.08),  # low blue
                0.5 + random.gauss(0, 0.1),    # medium size
                0.85 + random.gauss(0, 0.08),  # round
            ])
            labels.append("target")

        # Distractor class: warm colors but different proportions
        for _ in range(20):
            features.append([
                0.7 + random.gauss(0, 0.1),
                0.3 + random.gauss(0, 0.1),
                0.3 + random.gauss(0, 0.15),
                0.5 + random.gauss(0, 0.15),
                0.6 + random.gauss(0, 0.15),
            ])
            labels.append("distractor")

        # Obstacle class: cool colors, larger, angular
        for _ in range(20):
            features.append([
                0.15 + random.gauss(0, 0.1),
                0.4 + random.gauss(0, 0.2),
                0.5 + random.gauss(0, 0.2),
                0.75 + random.gauss(0, 0.1),
                0.25 + random.gauss(0, 0.1),
            ])
            labels.append("obstacle")

        classifier = KNNClassifier(k=5)
        classifier.fit(features, labels)
        return classifier

    def _find_target_detection(self, detections: list[Detection]) -> Optional[Detection]:
        """Find the best target detection from the detection list.

        If multiple objects are classified as 'target', pick the one with
        highest confidence. This is a simple strategy — production systems
        use more sophisticated data association (Hungarian algorithm, etc).
        """
        target_detections = [d for d in detections if d.predicted_class == "target"]
        if not target_detections:
            return None
        return max(target_detections, key=lambda d: d.confidence)

    def _run_state_machine(self, target_detection: Optional[Detection]) -> None:
        """Update the behavioral state based on detection status.

        State transitions prevent jittery behavior:
        - SEARCHING -> ACQUIRING: first detection appears
        - ACQUIRING -> FOLLOWING: N consecutive detections confirm target
        - ACQUIRING -> SEARCHING: not enough consecutive detections
        - FOLLOWING -> LOST: detection drops out
        - LOST -> FOLLOWING: detection recovers within grace period
        - LOST -> SEARCHING: grace period expires
        """
        prev_state = self.state

        if self.state == RobotState.SEARCHING:
            if target_detection is not None:
                self.state = RobotState.ACQUIRING
                self.tracker = TrackedTarget(alpha=0.4)
                self.tracker.update_with_detection(target_detection)

        elif self.state == RobotState.ACQUIRING:
            if target_detection is not None:
                self.tracker.update_with_detection(target_detection)
                if self.tracker.consecutive_detections >= self.acquire_threshold:
                    self.state = RobotState.FOLLOWING
                    # Reset PIDs to prevent stale integral from causing a lurch
                    self.heading_pid.reset()
                    self.distance_pid.reset()
            else:
                self.tracker.update_no_detection()
                if self.tracker.confidence <= 0:
                    self.state = RobotState.SEARCHING

        elif self.state == RobotState.FOLLOWING:
            if target_detection is not None:
                self.tracker.update_with_detection(target_detection)
            else:
                self.tracker.update_no_detection()
                if self.tracker.consecutive_misses >= 3:
                    self.state = RobotState.LOST

        elif self.state == RobotState.LOST:
            if target_detection is not None:
                self.tracker.update_with_detection(target_detection)
                self.state = RobotState.FOLLOWING
            else:
                self.tracker.update_no_detection()
                if self.tracker.consecutive_misses >= self.lost_grace_frames:
                    self.state = RobotState.SEARCHING
                    self.heading_pid.reset()
                    self.distance_pid.reset()

        # Log state change
        if self.state != prev_state:
            self.state_log.append((self.time, f"{prev_state.name} -> {self.state.name}"))

    def _compute_commands(self) -> tuple[float, float]:
        """Compute linear and angular velocity commands based on current state.

        Returns (linear_vel, angular_vel) tuple.
        """
        if self.state == RobotState.SEARCHING:
            # Rotate in place to scan for targets
            return 0.0, self.robot.search_angular_vel

        elif self.state == RobotState.ACQUIRING:
            # Slowly turn toward detected target while confirming
            dx = self.tracker.position.x - self.robot.position.x
            dy = self.tracker.position.y - self.robot.position.y
            target_angle = math.atan2(dy, dx)
            angle_error = normalize_angle(target_angle - self.robot.heading)
            return 0.0, 1.0 * angle_error  # Simple proportional turn

        elif self.state in (RobotState.FOLLOWING, RobotState.LOST):
            # Full PID following
            dx = self.tracker.position.x - self.robot.position.x
            dy = self.tracker.position.y - self.robot.position.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            target_angle = math.atan2(dy, dx)

            # Heading PID: minimize angle to target
            angle_error = normalize_angle(target_angle - self.robot.heading)
            angular_vel = self.heading_pid.compute(angle_error, self.dt)

            # Distance PID: maintain follow distance
            distance_error = distance - self.follow_distance
            linear_vel = self.distance_pid.compute(distance_error, self.dt)

            # If we're in LOST state, slow down (we're less confident)
            if self.state == RobotState.LOST:
                linear_vel *= 0.5
                angular_vel *= 0.5

            return linear_vel, angular_vel

        return 0.0, 0.0

    def step(self) -> dict:
        """Execute one simulation timestep.

        This is the core perception-action loop:
        1. Update world (move objects)
        2. Perceive (run detector)
        3. Track (filter detections)
        4. Decide (state machine)
        5. Act (PID -> velocity commands)
        6. Move (integrate robot position)
        """
        # 1. Update world objects
        for obj in self.objects:
            obj.update(self.time)

        # 2. Run detector
        detections = self.detector.detect(self.robot.position, self.objects)
        target_detection = self._find_target_detection(detections)

        # 3-4. Update state machine (which also updates tracker)
        self._run_state_machine(target_detection)

        # 5. Compute velocity commands
        linear_vel, angular_vel = self._compute_commands()
        self.robot.linear_vel = linear_vel
        self.robot.angular_vel = angular_vel

        # 6. Move robot
        self.robot.update(self.dt)

        # 7. Log metrics
        target_obj = self.objects[0]  # Ground truth target
        actual_distance = self.robot.position.distance_to(target_obj.position)
        if self.state == RobotState.FOLLOWING:
            self.distance_errors.append(abs(actual_distance - self.follow_distance))

        self.positions.append((
            self.robot.position.x, self.robot.position.y,
            target_obj.position.x, target_obj.position.y,
        ))

        self.time += self.dt

        return {
            "time": self.time,
            "state": self.state.name,
            "robot_pos": (self.robot.position.x, self.robot.position.y),
            "target_pos": (target_obj.position.x, target_obj.position.y),
            "distance": actual_distance,
            "n_detections": len(detections),
            "target_detected": target_detection is not None,
            "tracker_confidence": self.tracker.confidence,
        }

    def run(self, duration: float = 30.0) -> list[dict]:
        """Run simulation for the specified duration."""
        steps = int(duration / self.dt)
        results = []
        for _ in range(steps):
            results.append(self.step())
        return results


# =============================================================================
# Main — run simulation and analyze results
# =============================================================================

def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print_section("Day 011: Robot That Follows ML-Detected Objects")
    print("Integration: KNN detection + PID control + State machine\n")

    # Run simulation
    sim = FollowerSimulation(seed=42)
    print(f"World: {len(sim.objects)} objects (1 target, 2 distractors, 2 obstacles)")
    print(f"Follow distance: {sim.follow_distance} units")
    print(f"Detection range: {sim.detector.detection_range} units")
    print(f"Update rate: {1/sim.dt:.0f} Hz")
    print(f"False negative rate: {sim.detector.false_negative_rate:.0%}")
    print(f"False positive rate: {sim.detector.false_positive_rate:.0%}")

    duration = 30.0
    results = sim.run(duration)

    # Print trajectory samples
    print_section("Simulation Trajectory (sampled every 2s)")
    print(f"{'Time':>6}  {'State':<12}  {'Robot Pos':>16}  {'Target Pos':>16}  {'Dist':>6}  {'Detected':>8}")
    print("-" * 75)
    for r in results:
        if abs(r["time"] % 2.0) < sim.dt:  # Sample every 2 seconds
            rx, ry = r["robot_pos"]
            tx, ty = r["target_pos"]
            print(f"{r['time']:6.1f}  {r['state']:<12}  ({rx:6.2f}, {ry:6.2f})  "
                  f"({tx:6.2f}, {ty:6.2f})  {r['distance']:6.2f}  {'yes' if r['target_detected'] else 'no':>8}")

    # State transition log
    print_section("State Transitions")
    for t, transition in sim.state_log:
        print(f"  t={t:6.2f}s  {transition}")
    print(f"\n  Total transitions: {len(sim.state_log)}")

    # Time in each state
    print_section("Time in Each State")
    state_times: dict[str, float] = {}
    for r in results:
        state_times[r["state"]] = state_times.get(r["state"], 0) + sim.dt
    for state, t in sorted(state_times.items()):
        pct = t / duration * 100
        bar = "#" * int(pct / 2)
        print(f"  {state:<12}  {t:5.1f}s  ({pct:5.1f}%)  {bar}")

    # Following performance
    print_section("Following Performance")
    if sim.distance_errors:
        avg_err = sum(sim.distance_errors) / len(sim.distance_errors)
        max_err = max(sim.distance_errors)
        min_err = min(sim.distance_errors)
        print(f"  Average distance error: {avg_err:.3f} units")
        print(f"  Max distance error:     {max_err:.3f} units")
        print(f"  Min distance error:     {min_err:.3f} units")
        print(f"  Time spent following:   {len(sim.distance_errors) * sim.dt:.1f}s / {duration:.1f}s")

        # Track how many times target was lost during following
        lost_count = sum(1 for _, t in sim.state_log if "LOST" in t and "FOLLOWING" in t)
        print(f"  Times target lost:      {lost_count}")
    else:
        print("  Robot never entered FOLLOWING state!")

    # Detection accuracy simulation
    print_section("ML Detector Analysis")
    # Re-run detections to compute accuracy stats
    random.seed(42)
    correct = 0
    total = 0
    for obj in sim.objects:
        for _ in range(100):
            predicted = sim.detector.classifier.predict(obj.features)
            if predicted == obj.obj_type.value:
                correct += 1
            total += 1
    print(f"  KNN classification accuracy: {correct/total:.1%}")
    print(f"  K value: {sim.detector.classifier.k}")
    print(f"  Training examples: {len(sim.detector.classifier.training_data)}")

    print_section("Summary")
    following_pct = state_times.get("FOLLOWING", 0) / duration * 100
    print(f"  The robot spent {following_pct:.1f}% of the time actively following the target.")
    print(f"  Average tracking error of {avg_err:.3f} units vs desired {sim.follow_distance} unit follow distance.")
    print(f"  The state machine handled {len(sim.state_log)} transitions, keeping behavior stable")
    print(f"  despite {sim.detector.false_negative_rate:.0%} false negative rate from the detector.")
    print("\n  Key insight: Even with imperfect ML detection, the combination of")
    print("  EMA filtering + state machine + PID control produces smooth, robust following.")


if __name__ == "__main__":
    main()
