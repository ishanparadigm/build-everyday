"""
Day 034: RRT and RRT* Path Planning

A complete implementation of Rapidly-exploring Random Trees (RRT) and the
asymptotically optimal variant RRT* for 2D path planning with circular obstacles.

Builds on Day 028's A* by moving from grid-based to sampling-based planning —
the approach that dominates real-world robotics where configuration spaces are
continuous and high-dimensional.
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
        """Check if a point is inside this obstacle (with small margin)."""
        dx = p.x - self.cx
        dy = p.y - self.cy
        return (dx * dx + dy * dy) <= (self.radius * self.radius)


@dataclass
class TreeNode:
    """A node in the RRT tree.

    Stores position, parent index (for path extraction), and cost-from-root
    (needed for RRT* rewiring). The tree is stored as a flat list — parent
    indices reference positions in that list. This is more cache-friendly and
    simpler than pointer-based trees.
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
        return self.x_min <= p.x <= self.x_max and self.y_min <= p.y <= self.y_max

    def is_free(self, p: Point) -> bool:
        """Check if a point is in free space (in bounds and not in any obstacle)."""
        if not self.in_bounds(p):
            return False
        return not any(obs.contains(p) for obs in self.obstacles)


@dataclass
class PlannerConfig:
    """Configuration parameters for RRT/RRT*."""
    step_size: float = 2.0          # Maximum extension distance per step
    goal_threshold: float = 1.5     # How close to goal counts as "reached"
    goal_bias: float = 0.08         # Probability of sampling the goal (8%)
    max_iterations: int = 5000      # Maximum tree expansion steps
    collision_step: float = 0.5     # Step size for collision checking along edges
    # RRT* specific
    rewire_radius: float = 5.0     # Radius for near-neighbor rewiring
    use_rrt_star: bool = False      # Whether to use RRT* extensions


# =============================================================================
# Collision Detection
# =============================================================================

def is_edge_collision_free(env: Environment, p1: Point, p2: Point,
                           step: float = 0.5) -> bool:
    """Check if the straight-line edge from p1 to p2 is collision-free.

    We discretize the edge into small steps and check each intermediate point.
    The step size must be smaller than the smallest obstacle radius to prevent
    "tunneling" through thin obstacles.

    Why discretize? Because checking a continuous segment analytically against
    arbitrary obstacle shapes is complex. For circles, there IS an analytical
    solution (line-circle intersection), but discretization generalizes to any
    obstacle shape and is simpler to implement correctly.
    """
    dist = p1.distance_to(p2)
    if dist < 1e-9:
        return env.is_free(p1)

    # Number of checks: enough that no gap exceeds step size
    n_checks = max(2, int(math.ceil(dist / step)))

    for i in range(n_checks + 1):
        t = i / n_checks  # Interpolation parameter [0, 1]
        check_point = Point(
            p1.x + t * (p2.x - p1.x),
            p1.y + t * (p2.y - p1.y)
        )
        if not env.is_free(check_point):
            return False

    return True


# =============================================================================
# Core RRT Algorithm
# =============================================================================

def sample_random_point(env: Environment, goal: Point,
                        goal_bias: float) -> Point:
    """Sample a random point in the workspace, with goal biasing.

    With probability goal_bias, return the goal point. Otherwise, return a
    uniformly random point in the workspace bounds. Goal biasing is critical
    for convergence — without it, the tree explores uniformly and may take
    very long to stumble upon the goal.
    """
    if random.random() < goal_bias:
        return Point(goal.x, goal.y)

    x = random.uniform(env.x_min, env.x_max)
    y = random.uniform(env.y_min, env.y_max)
    return Point(x, y)


def find_nearest(nodes: list[TreeNode], target: Point) -> int:
    """Find the index of the nearest node to the target point.

    Brute-force O(n) search. For trees with >10K nodes, a KD-tree would
    reduce this to O(log n), but for our purposes linear search is fine
    and avoids the complexity of maintaining a balanced spatial index.
    """
    best_idx = 0
    best_dist = float('inf')

    for i, node in enumerate(nodes):
        d = node.point.distance_to(target)
        if d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx


def steer(from_point: Point, to_point: Point, step_size: float) -> Point:
    """Steer from from_point toward to_point, moving at most step_size.

    This is the "local planner" — it assumes straight-line motion between
    points. For robots with dynamics constraints (e.g., cars that can't
    move sideways), this function would need to respect those constraints,
    producing curved paths instead of straight lines.

    Math: q_new = q_near + min(δ, d) * (q_rand - q_near) / d
    where δ = step_size and d = distance(q_near, q_rand)
    """
    dist = from_point.distance_to(to_point)

    if dist < 1e-9:
        return Point(from_point.x, from_point.y)

    # If target is within step_size, go directly there
    if dist <= step_size:
        return Point(to_point.x, to_point.y)

    # Otherwise, move step_size in the direction of to_point
    ratio = step_size / dist
    new_x = from_point.x + ratio * (to_point.x - from_point.x)
    new_y = from_point.y + ratio * (to_point.y - from_point.y)
    return Point(new_x, new_y)


def find_near_nodes(nodes: list[TreeNode], point: Point,
                    radius: float) -> list[int]:
    """Find all nodes within radius of point (for RRT* rewiring).

    In RRT*, the rewiring radius should shrink as the tree grows:
    r = min(γ * (log(n)/n)^(1/d), step_size)
    where γ is a constant, n is tree size, d is dimension.

    For simplicity, we use a fixed radius here. In practice, the adaptive
    radius ensures the algorithm remains efficient as the tree densifies.
    """
    near = []
    for i, node in enumerate(nodes):
        if node.point.distance_to(point) <= radius:
            near.append(i)
    return near


def extract_path(nodes: list[TreeNode], goal_idx: int) -> list[Point]:
    """Trace parent pointers from goal back to root to extract the path.

    This is why we store parent indices — path extraction is O(path_length),
    just following the chain of parent pointers back to root (parent=None).
    """
    path = []
    idx: Optional[int] = goal_idx

    while idx is not None:
        path.append(nodes[idx].point)
        idx = nodes[idx].parent

    path.reverse()  # Root-to-goal order
    return path


def smooth_path(path: list[Point], env: Environment,
                collision_step: float = 0.5,
                max_attempts: int = 200) -> list[Point]:
    """Smooth the RRT path by shortcutting.

    RRT paths are jagged because each segment points toward a random sample.
    Smoothing tries to "cut corners" by connecting non-adjacent path nodes
    directly, skipping the intermediate zigzag.

    Algorithm:
    1. Pick two random indices i < j on the path
    2. If the direct line from path[i] to path[j] is collision-free,
       replace path[i+1..j-1] with nothing (direct connection)
    3. Repeat until no improvement or budget exhausted

    This is a greedy heuristic — it doesn't find the globally shortest
    collision-free path, but it dramatically improves practical quality.
    """
    if len(path) <= 2:
        return path

    smoothed = list(path)

    for _ in range(max_attempts):
        if len(smoothed) <= 2:
            break

        # Pick two random non-adjacent indices
        i = random.randint(0, len(smoothed) - 3)
        j = random.randint(i + 2, len(smoothed) - 1)

        # Try direct connection
        if is_edge_collision_free(env, smoothed[i], smoothed[j], collision_step):
            # Remove intermediate points
            smoothed = smoothed[:i + 1] + smoothed[j:]

    return smoothed


# =============================================================================
# RRT and RRT* Planners
# =============================================================================

def rrt(env: Environment, start: Point, goal: Point,
        config: PlannerConfig) -> tuple[list[Point], list[TreeNode], int]:
    """Basic RRT path planner.

    Returns:
        path: List of points from start to goal (empty if no path found)
        tree: All tree nodes (for visualization/analysis)
        iterations: Number of iterations used

    The algorithm is probabilistically complete: if a path exists, RRT will
    find it given enough iterations. But the path is generally suboptimal —
    it's whatever the random exploration happened to produce.
    """
    # Initialize tree with start node
    root = TreeNode(point=start, parent=None, cost=0.0)
    nodes: list[TreeNode] = [root]

    for iteration in range(config.max_iterations):
        # Step 1: Sample random point (with goal bias)
        q_rand = sample_random_point(env, goal, config.goal_bias)

        # Step 2: Find nearest node in tree
        nearest_idx = find_nearest(nodes, q_rand)
        q_near = nodes[nearest_idx].point

        # Step 3: Steer toward sample
        q_new = steer(q_near, q_rand, config.step_size)

        # Step 4: Check collision
        if not is_edge_collision_free(env, q_near, q_new, config.collision_step):
            continue  # This sample doesn't work, try another

        # Step 5: Add new node to tree
        new_cost = nodes[nearest_idx].cost + q_near.distance_to(q_new)
        new_node = TreeNode(point=q_new, parent=nearest_idx, cost=new_cost)
        new_idx = len(nodes)
        nodes.append(new_node)

        # Step 6: Check if we reached the goal
        if q_new.distance_to(goal) <= config.goal_threshold:
            # Add goal node explicitly for a clean path endpoint
            goal_node = TreeNode(
                point=Point(goal.x, goal.y),
                parent=new_idx,
                cost=new_cost + q_new.distance_to(goal)
            )
            nodes.append(goal_node)
            path = extract_path(nodes, len(nodes) - 1)
            return path, nodes, iteration + 1

    # No path found within iteration budget
    return [], nodes, config.max_iterations


def rrt_star(env: Environment, start: Point, goal: Point,
             config: PlannerConfig) -> tuple[list[Point], list[TreeNode], int]:
    """RRT* path planner — asymptotically optimal variant.

    Two key additions over basic RRT:
    1. Choose parent: Connect q_new to the nearby node that minimizes
       cost-from-root, not just the nearest node.
    2. Rewire: Check if routing through q_new gives nearby nodes a shorter
       path, and update their parents if so.

    These operations add O(k) work per iteration (k = nearby nodes) but
    guarantee the path converges to optimal as iterations → ∞.
    """
    root = TreeNode(point=start, parent=None, cost=0.0)
    nodes: list[TreeNode] = [root]
    goal_idx: Optional[int] = None

    for iteration in range(config.max_iterations):
        # Sample
        q_rand = sample_random_point(env, goal, config.goal_bias)

        # Find nearest
        nearest_idx = find_nearest(nodes, q_rand)
        q_near = nodes[nearest_idx].point

        # Steer
        q_new = steer(q_near, q_rand, config.step_size)

        # Collision check
        if not is_edge_collision_free(env, q_near, q_new, config.collision_step):
            continue

        # === RRT* Addition 1: Choose best parent from nearby nodes ===
        # Instead of always connecting to nearest, check all nodes within
        # rewire_radius and pick the one giving lowest cost-to-q_new
        near_indices = find_near_nodes(nodes, q_new, config.rewire_radius)

        best_parent = nearest_idx
        best_cost = nodes[nearest_idx].cost + q_near.distance_to(q_new)

        for ni in near_indices:
            candidate_cost = nodes[ni].cost + nodes[ni].point.distance_to(q_new)
            if candidate_cost < best_cost:
                # Verify this edge is collision-free
                if is_edge_collision_free(env, nodes[ni].point, q_new,
                                         config.collision_step):
                    best_parent = ni
                    best_cost = candidate_cost

        # Add new node with best parent
        new_node = TreeNode(point=q_new, parent=best_parent, cost=best_cost)
        new_idx = len(nodes)
        nodes.append(new_node)

        # === RRT* Addition 2: Rewire nearby nodes through q_new ===
        # For each nearby node, check if routing through q_new gives a
        # shorter path from root. If so, change its parent to q_new.
        for ni in near_indices:
            if ni == best_parent:
                continue  # Skip the parent we just chose

            new_potential_cost = best_cost + q_new.distance_to(nodes[ni].point)

            if new_potential_cost < nodes[ni].cost:
                if is_edge_collision_free(env, q_new, nodes[ni].point,
                                         config.collision_step):
                    # Rewire: this node gets a shorter path through q_new
                    nodes[ni].parent = new_idx
                    # Propagate cost update (simplified — full implementation
                    # would recursively update all descendants)
                    nodes[ni].cost = new_potential_cost

        # Check goal
        if q_new.distance_to(goal) <= config.goal_threshold:
            goal_cost = best_cost + q_new.distance_to(goal)

            if goal_idx is None or goal_cost < nodes[goal_idx].cost:
                goal_node = TreeNode(
                    point=Point(goal.x, goal.y),
                    parent=new_idx,
                    cost=goal_cost
                )
                goal_idx = len(nodes)
                nodes.append(goal_node)

            # Don't return immediately — RRT* keeps improving
            # But we'll stop if we've found a path and done enough iterations
            if iteration > config.max_iterations * 0.3:
                path = extract_path(nodes, goal_idx)
                return path, nodes, iteration + 1

    # Return best path found (if any)
    if goal_idx is not None:
        path = extract_path(nodes, goal_idx)
        return path, nodes, config.max_iterations

    return [], nodes, config.max_iterations


# =============================================================================
# Path Analysis Utilities
# =============================================================================

def path_length(path: list[Point]) -> float:
    """Compute total Euclidean length of a path."""
    total = 0.0
    for i in range(len(path) - 1):
        total += path[i].distance_to(path[i + 1])
    return total


def print_path_stats(label: str, path: list[Point], iterations: int,
                     tree_size: int) -> None:
    """Print statistics about a planned path."""
    if not path:
        print(f"  {label}: NO PATH FOUND after {iterations} iterations "
              f"({tree_size} nodes)")
        return

    length = path_length(path)
    print(f"  {label}:")
    print(f"    Path length:  {length:.2f} units")
    print(f"    Path nodes:   {len(path)}")
    print(f"    Iterations:   {iterations}")
    print(f"    Tree size:    {tree_size} nodes")


def render_ascii_map(env: Environment, tree: list[TreeNode],
                     path: list[Point], start: Point, goal: Point,
                     width: int = 60, height: int = 30) -> str:
    """Render the environment, tree, and path as ASCII art.

    Legend: S=start, G=goal, #=obstacle, .=tree node, *=path, ' '=free space
    """
    # Create empty grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    def to_grid(p: Point) -> tuple[int, int]:
        """Convert workspace coordinates to grid coordinates."""
        gx = int((p.x - env.x_min) / (env.x_max - env.x_min) * (width - 1))
        gy = int((p.y - env.y_min) / (env.y_max - env.y_min) * (height - 1))
        gy = height - 1 - gy  # Flip Y (grid row 0 = top)
        gx = max(0, min(width - 1, gx))
        gy = max(0, min(height - 1, gy))
        return gx, gy

    # Draw obstacles
    for row in range(height):
        for col in range(width):
            # Convert grid back to workspace
            wx = env.x_min + col / (width - 1) * (env.x_max - env.x_min)
            wy = env.y_max - row / (height - 1) * (env.y_max - env.y_min)
            wp = Point(wx, wy)
            if any(obs.contains(wp) for obs in env.obstacles):
                grid[row][col] = '#'

    # Draw tree nodes (sparse — only every Nth node to avoid clutter)
    step = max(1, len(tree) // 200)
    for i in range(0, len(tree), step):
        gx, gy = to_grid(tree[i].point)
        if grid[gy][gx] == ' ':
            grid[gy][gx] = '.'

    # Draw path
    if path:
        for p in path:
            gx, gy = to_grid(p)
            grid[gy][gx] = '*'

    # Draw start and goal
    sx, sy = to_grid(start)
    gx, gy = to_grid(goal)
    grid[sy][sx] = 'S'
    grid[gy][gx] = 'G'

    # Render
    border = '+' + '-' * width + '+'
    lines = [border]
    for row in grid:
        lines.append('|' + ''.join(row) + '|')
    lines.append(border)
    return '\n'.join(lines)


# =============================================================================
# Demo Environment and Main
# =============================================================================

def create_demo_environment() -> tuple[Environment, Point, Point]:
    """Create a challenging 2D environment with obstacles forming corridors.

    The obstacle layout creates a scenario where the planner must navigate
    around barriers — a straight-line path from start to goal is blocked,
    forcing the tree to explore around obstacles.
    """
    env = Environment(x_min=0, y_min=0, x_max=50, y_max=50)

    # Wall-like obstacle arrangements
    env.obstacles = [
        # Central barrier (forces path around)
        CircleObstacle(20, 25, 4),
        CircleObstacle(25, 25, 4),
        CircleObstacle(30, 25, 4),
        # Upper passage obstacles
        CircleObstacle(15, 38, 3),
        CircleObstacle(35, 38, 3),
        # Lower passage obstacles
        CircleObstacle(15, 12, 3),
        CircleObstacle(35, 12, 3),
        # Corner obstacles
        CircleObstacle(8, 8, 2.5),
        CircleObstacle(42, 42, 2.5),
        # Narrow passage guardians
        CircleObstacle(25, 15, 2),
        CircleObstacle(25, 35, 2),
    ]

    start = Point(5, 5)
    goal = Point(45, 45)

    return env, start, goal


if __name__ == "__main__":
    # Fix seed for reproducible demo (remove for true randomness)
    random.seed(42)

    print("=" * 65)
    print("  Day 034: RRT Path Planning")
    print("  Rapidly-exploring Random Trees for Collision-Free Navigation")
    print("=" * 65)

    # Create environment
    env, start, goal = create_demo_environment()
    print(f"\nEnvironment: {env.x_max}x{env.y_max} workspace")
    print(f"  Obstacles:  {len(env.obstacles)} circles")
    print(f"  Start:      {start}")
    print(f"  Goal:       {goal}")
    print(f"  Straight-line distance: {start.distance_to(goal):.2f}")

    # --- Run basic RRT ---
    print("\n" + "-" * 65)
    print("Running basic RRT...")
    config_rrt = PlannerConfig(
        step_size=2.0,
        goal_threshold=2.0,
        goal_bias=0.08,
        max_iterations=5000,
    )
    path_rrt, tree_rrt, iters_rrt = rrt(env, start, goal, config_rrt)
    print_path_stats("Basic RRT", path_rrt, iters_rrt, len(tree_rrt))

    # Smooth the RRT path
    if path_rrt:
        smoothed_rrt = smooth_path(path_rrt, env, config_rrt.collision_step)
        print(f"    After smoothing: {path_length(smoothed_rrt):.2f} units "
              f"({len(smoothed_rrt)} nodes)")

    # --- Run RRT* ---
    print("\n" + "-" * 65)
    print("Running RRT* (asymptotically optimal)...")
    random.seed(42)  # Reset for fair comparison
    config_star = PlannerConfig(
        step_size=2.0,
        goal_threshold=2.0,
        goal_bias=0.08,
        max_iterations=5000,
        rewire_radius=5.0,
        use_rrt_star=True,
    )
    path_star, tree_star, iters_star = rrt_star(env, start, goal, config_star)
    print_path_stats("RRT*", path_star, iters_star, len(tree_star))

    if path_star:
        smoothed_star = smooth_path(path_star, env, config_star.collision_step)
        print(f"    After smoothing: {path_length(smoothed_star):.2f} units "
              f"({len(smoothed_star)} nodes)")

    # --- Visualize ---
    print("\n" + "-" * 65)
    print("ASCII Visualization (Basic RRT):")
    print("  S=start, G=goal, #=obstacle, .=tree node, *=path\n")
    best_path = smoothed_rrt if path_rrt else []
    ascii_map = render_ascii_map(env, tree_rrt, best_path, start, goal)
    print(ascii_map)

    # --- Comparison ---
    print("\n" + "-" * 65)
    print("RRT vs RRT* Comparison:")
    if path_rrt and path_star:
        rrt_len = path_length(smoothed_rrt)
        star_len = path_length(smoothed_star)
        improvement = (rrt_len - star_len) / rrt_len * 100
        print(f"  RRT  smoothed path: {rrt_len:.2f} units")
        print(f"  RRT* smoothed path: {star_len:.2f} units")
        if improvement > 0:
            print(f"  RRT* improvement:   {improvement:.1f}% shorter")
        else:
            print(f"  RRT* was {-improvement:.1f}% longer (randomness — "
                  f"run more iterations for convergence)")
    print(f"  RRT  tree size:     {len(tree_rrt)} nodes")
    print(f"  RRT* tree size:     {len(tree_star)} nodes")

    # --- Key insight ---
    print("\n" + "-" * 65)
    print("Key Insight:")
    print("  RRT finds A path quickly (probabilistically complete).")
    print("  RRT* finds THE BEST path given enough time (asymptotically optimal).")
    print("  In practice, RRT + smoothing is often 'good enough' and faster.")
    print("  RRT* shines when path quality matters (surgical robots, tight spaces).")
    print("=" * 65)
