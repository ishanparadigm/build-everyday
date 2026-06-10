"""
Day 70: Autonomous Drone Path Planning with 3D RRT*

YOUR TASK: Implement a 3D path planner for autonomous drones using RRT*
with energy-aware cost functions, obstacle avoidance, and path smoothing.

Key concepts to implement:
- 3D configuration space with obstacles and no-fly zones
- RRT* with near-neighbor rewiring for asymptotic optimality
- Energy cost model (climbing ≠ descending ≠ horizontal flight)
- Path smoothing via shortcutting

Hints:
- Start with Point3D and collision detection — get those right first
- Implement basic RRT before adding the * (rewiring) — it's easier to debug
- The energy cost function is the key differentiator from Day 34's RRT
- Test collision detection independently before running the full planner
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List, Tuple


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Point3D:
    """A point in 3D space."""
    x: float
    y: float
    z: float

    def distance_to(self, other: 'Point3D') -> float:
        """Euclidean distance to another point."""
        raise NotImplementedError("TODO: implement 3D Euclidean distance")

    def horizontal_distance_to(self, other: 'Point3D') -> float:
        """Distance projected onto the xy-plane (ignoring altitude)."""
        raise NotImplementedError("TODO: implement horizontal distance")

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


@dataclass
class RRTNode:
    """A node in the RRT* tree.

    Each node stores its position, parent pointer, cost-from-start, and children.
    The cost is energy-based, not Euclidean distance.
    """
    position: Point3D
    parent: Optional['RRTNode'] = None
    cost: float = 0.0
    children: list = field(default_factory=list)


# =============================================================================
# Obstacle Types
# =============================================================================

@dataclass
class BoxObstacle:
    """Axis-aligned rectangular prism (buildings, walls).

    Hint: contains_point just checks if all 3 coordinates are within
    the min/max range (plus safety margin).
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    safety_margin: float = 1.0

    def contains_point(self, p: Point3D) -> bool:
        """Check if a point is inside the obstacle (including safety margin)."""
        raise NotImplementedError("TODO: implement box containment check")


@dataclass
class CylinderObstacle:
    """Vertical cylinder (trees, poles, towers).

    Hint: Check horizontal distance to center < radius AND z is in range.
    """
    cx: float
    cy: float
    radius: float
    z_min: float
    z_max: float
    safety_margin: float = 1.0

    def contains_point(self, p: Point3D) -> bool:
        """Check if point is inside cylinder + safety margin."""
        raise NotImplementedError("TODO: implement cylinder containment check")


@dataclass
class NoFlyZone:
    """Restricted airspace volume — any path through here is forbidden."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains_point(self, p: Point3D) -> bool:
        raise NotImplementedError("TODO: implement no-fly zone check")


# =============================================================================
# Energy Cost Model
# =============================================================================

@dataclass
class DronePhysics:
    """Energy cost model for a quadrotor drone.

    Hint: The key insight is that cost is NOT symmetric:
    - Climbing costs mass * gravity * height (fighting gravity)
    - Descending costs much less (just controlled braking)
    - Horizontal flight costs drag * distance
    - Headwind adds extra energy cost

    Think about WHY each component exists physically.
    """
    mass: float = 2.0
    gravity: float = 9.81
    drag_coeff: float = 0.5
    climb_coeff: float = 1.5
    descend_coeff: float = 0.3
    wind_x: float = 0.0
    wind_y: float = 0.0
    wind_penalty: float = 0.1

    def segment_cost(self, a: Point3D, b: Point3D) -> float:
        """Compute energy cost to fly from point a to point b.

        Three components to compute:
        1. Horizontal energy = horizontal_distance * drag_coeff
        2. Vertical energy = depends on whether climbing or descending
        3. Wind energy = headwind component * distance * wind_penalty

        Returns energy in Joules.
        """
        raise NotImplementedError("TODO: implement energy cost function")


# =============================================================================
# 3D Environment
# =============================================================================

class DroneEnvironment:
    """The 3D world the drone operates in.

    Hint: is_valid_point checks bounds + all obstacles + all no-fly zones.
    is_collision_free_segment samples points along the line and checks each.
    """

    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.obstacles: list[BoxObstacle | CylinderObstacle] = []
        self.no_fly_zones: list[NoFlyZone] = []

    def add_obstacle(self, obstacle: BoxObstacle | CylinderObstacle) -> None:
        self.obstacles.append(obstacle)

    def add_no_fly_zone(self, zone: NoFlyZone) -> None:
        self.no_fly_zones.append(zone)

    def is_valid_point(self, p: Point3D) -> bool:
        """Check if a point is in valid free space (in bounds, not in obstacle/NFZ)."""
        raise NotImplementedError("TODO: implement point validity check")

    def is_collision_free_segment(
        self, a: Point3D, b: Point3D, resolution: float = 0.5
    ) -> bool:
        """Check if a straight-line segment from a to b is collision-free.

        Hint: Sample points along the segment at the given resolution
        and check each one with is_valid_point.
        """
        raise NotImplementedError("TODO: implement segment collision check")

    def random_point(self) -> Point3D:
        """Sample a uniformly random point in the world bounds."""
        raise NotImplementedError("TODO: implement random sampling")


# =============================================================================
# RRT* Planner
# =============================================================================

class RRTStarPlanner:
    """3D RRT* path planner with energy-aware cost.

    Implement in stages:
    1. First get basic RRT working (just nearest neighbor, no rewiring)
    2. Then add the near-neighbor search and best parent selection
    3. Finally add rewiring — this is what makes it RRT*

    Hint: The rewiring step checks if routing through the new node
    gives a cheaper path to any nearby existing node.
    """

    def __init__(
        self,
        env: DroneEnvironment,
        physics: DronePhysics,
        step_size: float = 5.0,
        goal_threshold: float = 3.0,
        max_iterations: int = 3000,
        goal_bias: float = 0.1,
        gamma: float = 50.0,
    ):
        self.env = env
        self.physics = physics
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.gamma = gamma

    def _steer(self, from_pt: Point3D, to_pt: Point3D) -> Point3D:
        """Steer from from_pt toward to_pt, limited by step_size.

        Hint: If distance < step_size, return to_pt. Otherwise,
        move step_size in the direction of to_pt.
        """
        raise NotImplementedError("TODO: implement steering")

    def _near_radius(self, n_nodes: int) -> float:
        """Compute the near-neighbor radius for RRT*.

        Formula: r = min(gamma * (log(n) / n)^(1/3), step_size)
        """
        raise NotImplementedError("TODO: implement adaptive radius")

    def _find_near_nodes(
        self, nodes: list[RRTNode], point: Point3D, radius: float
    ) -> list[RRTNode]:
        """Find all nodes within radius of point."""
        raise NotImplementedError("TODO: implement near-neighbor search")

    def _find_nearest(self, nodes: list[RRTNode], point: Point3D) -> RRTNode:
        """Find the single nearest node to point."""
        raise NotImplementedError("TODO: implement nearest-neighbor search")

    def plan(self, start: Point3D, goal: Point3D) -> Optional[list[Point3D]]:
        """Run RRT* to find an energy-optimal path from start to goal.

        The main loop:
        1. Sample (with goal bias)
        2. Find nearest node
        3. Steer toward sample
        4. Check collision
        5. Find near nodes, choose cheapest parent
        6. Add node
        7. Rewire neighbors through new node if cheaper
        8. Check if goal reached

        Returns list of waypoints or None.
        """
        raise NotImplementedError("TODO: implement RRT* planning loop")

    def _propagate_cost(self, node: RRTNode, new_cost: float) -> None:
        """Recursively update costs after rewiring.

        Hint: When a node gets cheaper, ALL its descendants also get cheaper
        by the same delta. Use recursion on children.
        """
        raise NotImplementedError("TODO: implement cost propagation")

    def _extract_path(self, goal_parent: RRTNode, goal: Point3D) -> list[Point3D]:
        """Trace back from goal to start through parent pointers."""
        raise NotImplementedError("TODO: implement path extraction")


# =============================================================================
# Path Smoothing
# =============================================================================

def smooth_path(
    path: list[Point3D],
    env: DroneEnvironment,
    iterations: int = 100,
) -> list[Point3D]:
    """Smooth an RRT* path using random shortcutting.

    Hint: Pick two random indices i < j (with gap >= 2).
    If the direct segment from path[i] to path[j] is collision-free,
    remove all points between them.
    """
    raise NotImplementedError("TODO: implement path smoothing")


# =============================================================================
# Path Analysis
# =============================================================================

def analyze_path(
    path: list[Point3D], physics: DronePhysics
) -> dict[str, float]:
    """Compute detailed energy breakdown and path statistics.

    Returns dict with: total_energy, total_distance, horizontal_distance,
    total_climb, total_descent, max_altitude, min_altitude, n_waypoints.
    """
    raise NotImplementedError("TODO: implement path analysis")


def print_path_analysis(stats: dict[str, float], label: str = "Path") -> None:
    """Pretty-print path analysis results."""
    print(f"\n{'=' * 50}")
    print(f"  {label} Analysis")
    print(f"{'=' * 50}")
    print(f"  Total energy:        {stats['total_energy']:.1f} J")
    print(f"  Total distance:      {stats['total_distance']:.1f} m")
    print(f"  Horizontal distance: {stats['horizontal_distance']:.1f} m")
    print(f"  Total climb:         {stats['total_climb']:.1f} m")
    print(f"  Total descent:       {stats['total_descent']:.1f} m")
    print(f"  Max altitude:        {stats['max_altitude']:.1f} m")
    print(f"  Min altitude:        {stats['min_altitude']:.1f} m")
    print(f"  Waypoints:           {stats['n_waypoints']}")
    print(f"{'=' * 50}")


# =============================================================================
# Demo Environment (provided — use this to test your implementation)
# =============================================================================

def create_urban_environment() -> DroneEnvironment:
    """Create a city-like environment with buildings, trees, and a no-fly zone."""
    env = DroneEnvironment(
        x_range=(0, 100),
        y_range=(0, 100),
        z_range=(2, 50),
    )

    buildings = [
        BoxObstacle(20, 30, 20, 35, 0, 25, safety_margin=2.0),
        BoxObstacle(50, 65, 10, 25, 0, 15, safety_margin=2.0),
        BoxObstacle(40, 50, 50, 60, 0, 30, safety_margin=2.0),
        BoxObstacle(70, 80, 60, 75, 0, 20, safety_margin=2.0),
        BoxObstacle(10, 20, 60, 70, 0, 12, safety_margin=2.0),
    ]
    for b in buildings:
        env.add_obstacle(b)

    trees = [
        CylinderObstacle(35, 45, 3, 0, 10, safety_margin=1.5),
        CylinderObstacle(60, 45, 3, 0, 12, safety_margin=1.5),
        CylinderObstacle(25, 80, 4, 0, 14, safety_margin=1.5),
        CylinderObstacle(85, 30, 3, 0, 8, safety_margin=1.5),
    ]
    for t in trees:
        env.add_obstacle(t)

    env.add_no_fly_zone(NoFlyZone(80, 100, 80, 100, 0, 50))

    return env


if __name__ == "__main__":
    random.seed(42)

    print("Day 70: Autonomous Drone Path Planning (3D RRT*)")
    print("=" * 50)

    # Build environment
    env = create_urban_environment()
    print(f"Environment: {env.x_range} x {env.y_range} x {env.z_range}")
    print(f"Obstacles: {len(env.obstacles)}, No-fly zones: {len(env.no_fly_zones)}")

    # Define mission
    start = Point3D(5.0, 5.0, 10.0)
    goal = Point3D(75.0, 90.0, 10.0)
    print(f"Start: ({start.x}, {start.y}, {start.z})")
    print(f"Goal:  ({goal.x}, {goal.y}, {goal.z})")

    # Configure physics
    physics = DronePhysics(wind_x=3.0)

    # Plan
    planner = RRTStarPlanner(
        env=env, physics=physics,
        step_size=8.0, goal_threshold=5.0,
        max_iterations=3000, goal_bias=0.1,
    )
    raw_path = planner.plan(start, goal)

    if raw_path:
        raw_stats = analyze_path(raw_path, physics)
        print_path_analysis(raw_stats, "Raw Path")

        smoothed = smooth_path(raw_path, env, iterations=200)
        smooth_stats = analyze_path(smoothed, physics)
        print_path_analysis(smooth_stats, "Smoothed Path")

        savings = (1 - smooth_stats["total_energy"] / raw_stats["total_energy"]) * 100
        print(f"\nEnergy savings from smoothing: {savings:.1f}%")
    else:
        print("No path found!")
