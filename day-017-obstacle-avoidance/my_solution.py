"""
Day 17: Obstacle Avoidance using Vector Field Histogram (VFH)

YOUR TASK: Implement the VFH obstacle avoidance algorithm from scratch.

The algorithm pipeline:
  sensors → polar histogram → valleys → steering direction → move

Key math you'll need:
- Ray-circle intersection (quadratic formula)
- Angular arithmetic with wraparound (normalize to [-pi, pi])
- Cost function minimization over candidate directions

Hints:
- Start with ray_cast — get that working with a simple test before moving on
- The histogram bins aggregate nearby sensor readings — think of it as a smoothed
  density estimate of obstacle proximity
- Valleys wrap around: sector 71 and sector 0 can be part of the same valley
- The cost function has three terms: goal-seeking, momentum, and smoothness
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Obstacle:
    """Circular obstacle defined by center and radius."""
    x: float
    y: float
    radius: float


@dataclass
class World:
    """2D world with circular obstacles and boundaries."""
    width: float
    height: float
    obstacles: list[Obstacle] = field(default_factory=list)

    def ray_cast(self, ox: float, oy: float, angle: float, max_range: float) -> float:
        """
        Cast a ray from (ox, oy) at the given angle. Return distance to the
        nearest obstacle or wall. Return max_range if nothing is hit.

        Hint: For each circular obstacle, solve the quadratic equation for
        ray-circle intersection. For walls, compute where the ray crosses
        each boundary line.
        """
        raise NotImplementedError("TODO: implement ray casting")


@dataclass
class Robot:
    """Mobile robot with range sensors arranged in a ring."""
    x: float
    y: float
    heading: float
    num_sensors: int = 36
    max_sensor_range: float = 8.0
    sensor_noise_std: float = 0.2
    step_size: float = 0.5

    def get_sensor_angles(self) -> list[float]:
        """
        Return the global angle of each sensor beam.
        Sensors are evenly distributed around the robot.
        """
        raise NotImplementedError("TODO: implement sensor angle calculation")

    def read_sensors(self, world: World) -> list[tuple[float, float]]:
        """
        Simulate range sensor readings with Gaussian noise.
        Returns list of (angle, distance) pairs.

        Hint: Use get_sensor_angles() and world.ray_cast() for each beam.
        Add noise with random.gauss(). Clamp to valid range.
        """
        raise NotImplementedError("TODO: implement sensor reading simulation")

    def move(self, direction: float) -> None:
        """
        Move robot one step in the given direction.
        Update position and heading.
        """
        raise NotImplementedError("TODO: implement robot movement")


@dataclass
class VFHConfig:
    """Configuration parameters for the VFH algorithm."""
    num_sectors: int = 72
    threshold: float = 3.0
    a_const: float = 5.0
    b_const: float = 0.5
    mu_target: float = 5.0
    mu_current: float = 2.0
    mu_previous: float = 2.0
    min_valley_width: int = 3
    wide_valley_threshold: int = 12


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi].

    Hint: Use a while loop or modular arithmetic.
    This is critical — angular math breaks without normalization.
    """
    raise NotImplementedError("TODO: implement angle normalization")


def angular_distance(a: float, b: float) -> float:
    """
    Compute the absolute angular distance between two angles.

    Hint: normalize_angle(a - b) then take absolute value.
    """
    raise NotImplementedError("TODO: implement angular distance")


def build_histogram(
    readings: list[tuple[float, float]],
    config: VFHConfig,
    max_range: float,
) -> list[float]:
    """
    Build a polar obstacle density histogram from sensor readings.

    Hint: For each reading, compute weight = a - b * distance.
    Map the reading's angle to a sector index. Add weight to that sector
    and half-weight to neighbors for smoothing.
    """
    raise NotImplementedError("TODO: implement histogram construction")


def find_valleys(histogram: list[float], config: VFHConfig) -> list[tuple[int, int]]:
    """
    Find contiguous runs of free (below-threshold) sectors.

    Returns list of (start_sector, end_sector) tuples.

    Hint: Iterate through sectors, tracking when you enter/exit free regions.
    Don't forget wraparound — the valley might span from sector 70 to sector 2.
    Filter out valleys narrower than min_valley_width.
    """
    raise NotImplementedError("TODO: implement valley detection")


def sector_to_angle(sector: int, num_sectors: int) -> float:
    """Convert sector index to the angle at the center of that sector."""
    raise NotImplementedError("TODO: implement sector-to-angle conversion")


def select_direction(
    valleys: list[tuple[int, int]],
    goal_angle: float,
    current_heading: float,
    previous_direction: float,
    config: VFHConfig,
) -> Optional[float]:
    """
    Select the best steering direction from candidate valleys.

    Hint: For each valley, compute a candidate angle (center for narrow valleys,
    edge-nearest-to-goal for wide valleys). Evaluate the cost function:
      cost = mu1 * delta_goal + mu2 * delta_heading + mu3 * delta_previous
    Pick the candidate with lowest cost. Return None if no valleys.
    """
    raise NotImplementedError("TODO: implement direction selection")


def run_simulation(
    world: World,
    robot: Robot,
    goal_x: float,
    goal_y: float,
    config: VFHConfig,
    max_steps: int = 200,
    goal_tolerance: float = 1.5,
) -> list[tuple[float, float]]:
    """
    Run the full VFH obstacle avoidance loop.

    Hint: Each step: read sensors → build histogram → find valleys →
    select direction → move. Check for goal reached. Handle stuck case
    (no free valleys) with a recovery rotation.
    """
    raise NotImplementedError("TODO: implement simulation loop")


if __name__ == "__main__":
    random.seed(42)

    world = World(
        width=40.0,
        height=30.0,
        obstacles=[
            Obstacle(15, 15, 3.0),
            Obstacle(18, 13, 2.5),
            Obstacle(20, 16, 2.0),
            Obstacle(8, 10, 1.5),
            Obstacle(25, 20, 2.0),
            Obstacle(30, 12, 2.5),
            Obstacle(12, 22, 1.8),
            Obstacle(28, 8, 1.5),
            Obstacle(33, 18, 2.0),
        ],
    )

    robot = Robot(x=3.0, y=3.0, heading=math.pi / 4)
    goal_x, goal_y = 37.0, 27.0
    config = VFHConfig()

    print("VFH Obstacle Avoidance - Your Implementation")
    print(f"Start: ({robot.x}, {robot.y}) → Goal: ({goal_x}, {goal_y})")
    print()

    path = run_simulation(world, robot, goal_x, goal_y, config)

    total_dist = sum(
        math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
        for i in range(len(path) - 1)
    )
    print(f"\nPath length: {total_dist:.2f}")
    print(f"Steps: {len(path) - 1}")
