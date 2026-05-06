"""
Day 034: RRT Path Planning — Your Implementation

Implement the RRT and RRT* algorithms for 2D path planning.

Hints:
- Start with the data structures (Point, TreeNode, Environment)
- Build the collision checker first — everything else depends on it
- The core RRT loop is: sample → nearest → steer → check → add → goal?
- Goal biasing is just "sometimes return goal instead of random point"
- For RRT*, the key additions are: choose best parent + rewire neighbors
- Path smoothing is independent of the planner — apply it after
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Point:
    """A 2D point in the workspace."""
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        """Euclidean distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"


@dataclass
class CircleObstacle:
    """A circular obstacle defined by center and radius."""
    cx: float
    cy: float
    radius: float

    def contains(self, p: Point) -> bool:
        """Check if a point is inside this obstacle."""
        raise NotImplementedError("TODO: implement point-in-circle check")


@dataclass
class TreeNode:
    """A node in the RRT tree.

    Stores position, parent index (for path extraction), and cost-from-root
    (needed for RRT* rewiring).
    """
    point: Point
    parent: Optional[int] = None  # Index into the tree's node list
    cost: float = 0.0             # Cost from root to this node


@dataclass
class Environment:
    """2D workspace with boundaries and obstacles."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    obstacles: list[CircleObstacle] = field(default_factory=list)

    def in_bounds(self, p: Point) -> bool:
        """Check if point is within workspace boundaries."""
        raise NotImplementedError("TODO: implement bounds check")

    def is_free(self, p: Point) -> bool:
        """Check if a point is in free space (in bounds and not in any obstacle)."""
        raise NotImplementedError("TODO: implement free space check")


@dataclass
class PlannerConfig:
    """Configuration parameters for RRT/RRT*."""
    step_size: float = 2.0
    goal_threshold: float = 1.5
    goal_bias: float = 0.08
    max_iterations: int = 5000
    collision_step: float = 0.5
    rewire_radius: float = 5.0
    use_rrt_star: bool = False


# =============================================================================
# Collision Detection
# =============================================================================

def is_edge_collision_free(env: Environment, p1: Point, p2: Point,
                           step: float = 0.5) -> bool:
    """Check if the straight-line edge from p1 to p2 is collision-free.

    Hint: Discretize the edge into small steps. At each step, interpolate
    the position and check if it's in free space. The number of checks
    should be ceil(distance / step).
    """
    raise NotImplementedError("TODO: implement edge collision checking")


# =============================================================================
# Core RRT Functions
# =============================================================================

def sample_random_point(env: Environment, goal: Point,
                        goal_bias: float) -> Point:
    """Sample a random point in the workspace, with goal biasing.

    Hint: With probability goal_bias, return the goal. Otherwise, return
    a uniformly random point in [x_min, x_max] x [y_min, y_max].
    Use random.random() for the bias check and random.uniform() for coordinates.
    """
    raise NotImplementedError("TODO: implement random sampling with goal bias")


def find_nearest(nodes: list[TreeNode], target: Point) -> int:
    """Find the index of the nearest node to the target point.

    Hint: Linear scan through all nodes, track the index with minimum distance.
    """
    raise NotImplementedError("TODO: implement nearest neighbor search")


def steer(from_point: Point, to_point: Point, step_size: float) -> Point:
    """Steer from from_point toward to_point, moving at most step_size.

    Hint: Compute direction vector, normalize it, scale by min(step_size, dist).
    q_new = q_near + min(δ, d) * (q_rand - q_near) / d
    """
    raise NotImplementedError("TODO: implement steering function")


def find_near_nodes(nodes: list[TreeNode], point: Point,
                    radius: float) -> list[int]:
    """Find all nodes within radius of point (for RRT* rewiring).

    Hint: Linear scan, return indices of all nodes within distance <= radius.
    """
    raise NotImplementedError("TODO: implement near-neighbor search")


def extract_path(nodes: list[TreeNode], goal_idx: int) -> list[Point]:
    """Trace parent pointers from goal back to root to extract the path.

    Hint: Follow parent indices until you reach None (root). Reverse the result.
    """
    raise NotImplementedError("TODO: implement path extraction")


def smooth_path(path: list[Point], env: Environment,
                collision_step: float = 0.5,
                max_attempts: int = 200) -> list[Point]:
    """Smooth the path by shortcutting.

    Hint: Randomly pick two non-adjacent indices i, j. If the direct line
    from path[i] to path[j] is collision-free, remove everything in between.
    Repeat max_attempts times.
    """
    raise NotImplementedError("TODO: implement path smoothing")


# =============================================================================
# RRT Planner
# =============================================================================

def rrt(env: Environment, start: Point, goal: Point,
        config: PlannerConfig) -> tuple[list[Point], list[TreeNode], int]:
    """Basic RRT path planner.

    Returns:
        path: List of points from start to goal (empty if no path found)
        tree: All tree nodes (for analysis)
        iterations: Number of iterations used

    Hint: The main loop is:
    1. Sample random point (with goal bias)
    2. Find nearest node in tree
    3. Steer from nearest toward sample
    4. Check edge for collisions
    5. Add new node to tree
    6. Check if we reached the goal
    """
    raise NotImplementedError("TODO: implement the RRT algorithm")


def rrt_star(env: Environment, start: Point, goal: Point,
             config: PlannerConfig) -> tuple[list[Point], list[TreeNode], int]:
    """RRT* path planner — asymptotically optimal variant.

    Hint: Same as RRT but with two additions:
    1. Choose parent: Before adding q_new, check all nearby nodes — connect
       to the one that gives lowest cost-from-root (not just nearest).
    2. Rewire: After adding q_new, check if routing nearby nodes through
       q_new gives them a shorter path. Update their parents if so.
    """
    raise NotImplementedError("TODO: implement the RRT* algorithm")


# =============================================================================
# Utility
# =============================================================================

def path_length(path: list[Point]) -> float:
    """Compute total Euclidean length of a path."""
    total = 0.0
    for i in range(len(path) - 1):
        total += path[i].distance_to(path[i + 1])
    return total


# =============================================================================
# Main — Test your implementation
# =============================================================================

def create_demo_environment() -> tuple[Environment, Point, Point]:
    """Create the demo environment (same as solution.py)."""
    env = Environment(x_min=0, y_min=0, x_max=50, y_max=50)
    env.obstacles = [
        CircleObstacle(20, 25, 4),
        CircleObstacle(25, 25, 4),
        CircleObstacle(30, 25, 4),
        CircleObstacle(15, 38, 3),
        CircleObstacle(35, 38, 3),
        CircleObstacle(15, 12, 3),
        CircleObstacle(35, 12, 3),
        CircleObstacle(8, 8, 2.5),
        CircleObstacle(42, 42, 2.5),
        CircleObstacle(25, 15, 2),
        CircleObstacle(25, 35, 2),
    ]
    start = Point(5, 5)
    goal = Point(45, 45)
    return env, start, goal


if __name__ == "__main__":
    random.seed(42)
    env, start, goal = create_demo_environment()

    print("Testing your RRT implementation...")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Obstacles: {len(env.obstacles)}")

    # Test basic RRT
    config = PlannerConfig(step_size=2.0, goal_threshold=2.0, goal_bias=0.08)
    path, tree, iters = rrt(env, start, goal, config)

    if path:
        print(f"\nRRT found path in {iters} iterations!")
        print(f"  Path length: {path_length(path):.2f}")
        print(f"  Path nodes:  {len(path)}")
        print(f"  Tree size:   {len(tree)}")

        smoothed = smooth_path(path, env)
        print(f"  After smoothing: {path_length(smoothed):.2f} ({len(smoothed)} nodes)")
    else:
        print(f"\nRRT failed to find path after {iters} iterations")

    # Test RRT*
    random.seed(42)
    config_star = PlannerConfig(
        step_size=2.0, goal_threshold=2.0, goal_bias=0.08,
        rewire_radius=5.0, use_rrt_star=True
    )
    path_star, tree_star, iters_star = rrt_star(env, start, goal, config_star)

    if path_star:
        print(f"\nRRT* found path in {iters_star} iterations!")
        print(f"  Path length: {path_length(path_star):.2f}")
        smoothed_star = smooth_path(path_star, env)
        print(f"  After smoothing: {path_length(smoothed_star):.2f}")
    else:
        print(f"\nRRT* failed to find path after {iters_star} iterations")
