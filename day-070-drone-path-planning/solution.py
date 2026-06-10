"""
Day 70: Autonomous Drone Path Planning with 3D RRT*

A complete 3D path planner for autonomous drones using the RRT* algorithm
with energy-aware cost functions, no-fly zones, and path smoothing.

This builds on Day 34's RRT by adding:
- 3D configuration space (x, y, z)
- Asymptotically optimal rewiring (RRT*)
- Physics-based energy cost instead of Euclidean distance
- Multiple obstacle types (boxes, cylinders, no-fly zones)
- Path smoothing via shortcutting
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
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def horizontal_distance_to(self, other: 'Point3D') -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2
        )

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
    cost: float = 0.0  # Total energy cost from start to this node
    children: list = field(default_factory=list)


# =============================================================================
# Obstacle Types
# =============================================================================

@dataclass
class BoxObstacle:
    """Axis-aligned rectangular prism (buildings, walls).

    Defined by min/max corners. The safety_margin adds a buffer zone
    around the obstacle that the drone must also avoid.
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float  # Usually 0 (ground level)
    z_max: float
    safety_margin: float = 1.0

    def contains_point(self, p: Point3D) -> bool:
        """Check if a point is inside the obstacle (including safety margin)."""
        m = self.safety_margin
        return (self.x_min - m <= p.x <= self.x_max + m and
                self.y_min - m <= p.y <= self.y_max + m and
                self.z_min - m <= p.z <= self.z_max + m)


@dataclass
class CylinderObstacle:
    """Vertical cylinder (trees, poles, towers).

    Defined by center (x,y), radius, and height range.
    """
    cx: float
    cy: float
    radius: float
    z_min: float
    z_max: float
    safety_margin: float = 1.0

    def contains_point(self, p: Point3D) -> bool:
        """Check if point is inside cylinder + safety margin."""
        r = self.radius + self.safety_margin
        dx = p.x - self.cx
        dy = p.y - self.cy
        horizontal_dist = math.sqrt(dx * dx + dy * dy)
        return (horizontal_dist <= r and
                self.z_min - self.safety_margin <= p.z <= self.z_max + self.safety_margin)


@dataclass
class NoFlyZone:
    """Restricted airspace volume — any path through here is forbidden.

    Modeled as a rectangular prism in airspace, e.g. near airports.
    Unlike obstacles which are physical, no-fly zones are regulatory.
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains_point(self, p: Point3D) -> bool:
        return (self.x_min <= p.x <= self.x_max and
                self.y_min <= p.y <= self.y_max and
                self.z_min <= p.z <= self.z_max)


# =============================================================================
# Energy Cost Model
# =============================================================================

@dataclass
class DronePhysics:
    """Energy cost model for a quadrotor drone.

    Key insight: energy cost != distance. Climbing is expensive because the
    drone must do work against gravity (F = mg). Descending recovers some
    energy but still has drag. Horizontal flight has aerodynamic drag
    proportional to distance.

    Wind is modeled as a constant vector. Flying into headwind costs more
    energy; tailwind reduces cost.
    """
    mass: float = 2.0           # kg
    gravity: float = 9.81       # m/s^2
    drag_coeff: float = 0.5     # Energy per meter of horizontal flight (J/m)
    climb_coeff: float = 1.5    # Extra energy per meter of altitude gain (J/m)
    descend_coeff: float = 0.3  # Energy per meter of descent (braking, J/m)
    wind_x: float = 0.0         # Wind velocity in x direction (m/s)
    wind_y: float = 0.0         # Wind velocity in y direction (m/s)
    wind_penalty: float = 0.1   # Energy cost multiplier for fighting wind (J*s/m^2)

    def segment_cost(self, a: Point3D, b: Point3D) -> float:
        """Compute energy cost to fly from point a to point b.

        Three components:
        1. Horizontal drag: proportional to horizontal distance
        2. Vertical cost: asymmetric — climbing costs much more than descending
        3. Wind penalty: flying against the wind costs additional energy

        Returns energy in Joules (simplified model).
        """
        # Horizontal component
        dx = b.x - a.x
        dy = b.y - a.y
        dz = b.z - a.z
        d_horiz = math.sqrt(dx * dx + dy * dy)
        horizontal_energy = d_horiz * self.drag_coeff

        # Vertical component — the key asymmetry
        # Climbing: drone must produce thrust > weight, doing work against gravity
        # Descending: some energy for controlled descent (can't just freefall)
        if dz > 0:
            vertical_energy = dz * self.climb_coeff * self.mass * self.gravity
        else:
            vertical_energy = abs(dz) * self.descend_coeff

        # Wind penalty: compute component of wind opposing the direction of travel
        # If wind is aligned with travel → negative penalty (benefit)
        # If wind opposes travel → positive penalty (extra energy needed)
        if d_horiz > 0.001:
            # Normalize travel direction
            travel_dir_x = dx / d_horiz
            travel_dir_y = dy / d_horiz
            # Wind component against travel direction (dot product)
            headwind = -(self.wind_x * travel_dir_x + self.wind_y * travel_dir_y)
            wind_energy = max(0, headwind * d_horiz * self.wind_penalty)
        else:
            wind_energy = 0.0

        return horizontal_energy + vertical_energy + wind_energy


# =============================================================================
# 3D Environment
# =============================================================================

class DroneEnvironment:
    """The 3D world the drone operates in.

    Contains obstacles, no-fly zones, and altitude constraints.
    Provides collision checking for both points and line segments.
    """

    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],  # (min_altitude, max_altitude)
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
        """Check if a point is in valid free space.

        A point is valid if:
        1. It's within the world bounds
        2. It's not inside any obstacle
        3. It's not inside any no-fly zone
        """
        # Bounds check
        if not (self.x_range[0] <= p.x <= self.x_range[1] and
                self.y_range[0] <= p.y <= self.y_range[1] and
                self.z_range[0] <= p.z <= self.z_range[1]):
            return False

        # Obstacle check
        for obs in self.obstacles:
            if obs.contains_point(p):
                return False

        # No-fly zone check
        for nfz in self.no_fly_zones:
            if nfz.contains_point(p):
                return False

        return True

    def is_collision_free_segment(
        self, a: Point3D, b: Point3D, resolution: float = 0.5
    ) -> bool:
        """Check if a straight-line segment from a to b is collision-free.

        Uses discrete sampling along the segment. The resolution parameter
        controls the step size — smaller = more accurate but slower.

        Why discrete sampling instead of analytical intersection?
        - Simpler to implement for multiple obstacle types
        - Fast enough for RRT* (most segments are short)
        - In practice, 0.5m resolution catches all real obstacles
        """
        dist = a.distance_to(b)
        if dist < 0.001:
            return self.is_valid_point(a)

        n_checks = max(2, int(dist / resolution) + 1)
        for i in range(n_checks + 1):
            t = i / n_checks
            p = Point3D(
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
            )
            if not self.is_valid_point(p):
                return False
        return True

    def random_point(self) -> Point3D:
        """Sample a uniformly random point in the world bounds."""
        return Point3D(
            random.uniform(self.x_range[0], self.x_range[1]),
            random.uniform(self.y_range[0], self.y_range[1]),
            random.uniform(self.z_range[0], self.z_range[1]),
        )


# =============================================================================
# RRT* Planner
# =============================================================================

class RRTStarPlanner:
    """3D RRT* path planner with energy-aware cost.

    RRT* improves over basic RRT by:
    1. Finding all near neighbors within a shrinking radius
    2. Choosing the cheapest parent among them
    3. Rewiring the tree through the new node if it provides shortcuts

    This gives asymptotic optimality — as iterations → ∞, the path
    converges to the true optimum. In practice, 2000-5000 iterations
    gives good paths for most environments.
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
        self.goal_bias = goal_bias  # Probability of sampling the goal directly
        # gamma controls the rewiring radius: r = gamma * (log(n)/n)^(1/3)
        # Larger gamma = more rewiring = better paths but slower
        self.gamma = gamma

    def _steer(self, from_pt: Point3D, to_pt: Point3D) -> Point3D:
        """Steer from from_pt toward to_pt, limited by step_size.

        If to_pt is within step_size, return to_pt directly.
        Otherwise, return the point step_size away in the direction of to_pt.
        This prevents the tree from making huge jumps.
        """
        dist = from_pt.distance_to(to_pt)
        if dist <= self.step_size:
            return to_pt

        ratio = self.step_size / dist
        return Point3D(
            from_pt.x + ratio * (to_pt.x - from_pt.x),
            from_pt.y + ratio * (to_pt.y - from_pt.y),
            from_pt.z + ratio * (to_pt.z - from_pt.z),
        )

    def _near_radius(self, n_nodes: int) -> float:
        """Compute the near-neighbor radius for RRT*.

        r = min(gamma * (log(n) / n)^(1/d), step_size)

        This radius shrinks as the tree grows, but not too fast —
        the log(n)/n term ensures we always check enough neighbors
        for optimality guarantees. d=3 for 3D space.
        """
        if n_nodes < 2:
            return self.step_size
        r = self.gamma * (math.log(n_nodes) / n_nodes) ** (1.0 / 3.0)
        return min(r, self.step_size)

    def _find_near_nodes(
        self, nodes: list[RRTNode], point: Point3D, radius: float
    ) -> list[RRTNode]:
        """Find all nodes within radius of point.

        In production, you'd use a KD-tree for O(log n) queries.
        For our problem sizes (<5000 nodes), brute force is fine.
        """
        return [
            node for node in nodes
            if node.position.distance_to(point) <= radius
        ]

    def _find_nearest(self, nodes: list[RRTNode], point: Point3D) -> RRTNode:
        """Find the single nearest node to point. O(n) brute force."""
        return min(nodes, key=lambda n: n.position.distance_to(point))

    def plan(self, start: Point3D, goal: Point3D) -> Optional[list[Point3D]]:
        """Run RRT* to find an energy-optimal path from start to goal.

        Returns a list of waypoints or None if no path found.

        The algorithm:
        1. Sample a random point (with goal bias)
        2. Find nearest node in tree
        3. Steer toward the sample to get a new candidate point
        4. If collision-free, find all near nodes
        5. Choose the parent that gives minimum cost
        6. Add the new node
        7. Rewire: check if routing through the new node is cheaper for neighbors
        8. Repeat until goal is reached or max iterations
        """
        # Validate start and goal
        if not self.env.is_valid_point(start):
            print("ERROR: Start position is not in free space!")
            return None
        if not self.env.is_valid_point(goal):
            print("ERROR: Goal position is not in free space!")
            return None

        # Initialize tree with start node
        start_node = RRTNode(position=start, cost=0.0)
        nodes: list[RRTNode] = [start_node]
        best_goal_node: Optional[RRTNode] = None
        best_goal_cost = float('inf')

        for iteration in range(self.max_iterations):
            # --- Step 1: Sample ---
            # With probability goal_bias, sample the goal directly.
            # This speeds up convergence dramatically — without it, the tree
            # might explore forever in 3D space before stumbling on the goal.
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self.env.random_point()
                # Rejection sampling: skip invalid points
                if not self.env.is_valid_point(sample):
                    continue

            # --- Step 2: Find nearest node ---
            nearest = self._find_nearest(nodes, sample)

            # --- Step 3: Steer ---
            new_point = self._steer(nearest.position, sample)

            # Check collision for the segment
            if not self.env.is_collision_free_segment(nearest.position, new_point):
                continue

            # --- Step 4: Find near nodes for rewiring ---
            radius = self._near_radius(len(nodes))
            near_nodes = self._find_near_nodes(nodes, new_point, radius)

            # --- Step 5: Choose best parent ---
            # Among all near nodes, pick the one that gives minimum cost-to-come.
            # This is what makes RRT* better than RRT — we don't just connect
            # to the nearest node, we connect to the CHEAPEST node.
            best_parent = nearest
            best_cost = nearest.cost + self.physics.segment_cost(
                nearest.position, new_point
            )

            for near_node in near_nodes:
                candidate_cost = near_node.cost + self.physics.segment_cost(
                    near_node.position, new_point
                )
                if (candidate_cost < best_cost and
                        self.env.is_collision_free_segment(
                            near_node.position, new_point
                        )):
                    best_parent = near_node
                    best_cost = candidate_cost

            # --- Step 6: Add new node ---
            new_node = RRTNode(
                position=new_point,
                parent=best_parent,
                cost=best_cost,
            )
            best_parent.children.append(new_node)
            nodes.append(new_node)

            # --- Step 7: Rewire ---
            # For each near node, check if going through new_node is cheaper.
            # This propagates improvements through the tree — a key feature
            # of RRT* that standard RRT lacks.
            for near_node in near_nodes:
                if near_node is best_parent:
                    continue
                rewire_cost = new_node.cost + self.physics.segment_cost(
                    new_point, near_node.position
                )
                if (rewire_cost < near_node.cost and
                        self.env.is_collision_free_segment(
                            new_point, near_node.position
                        )):
                    # Rewire: change near_node's parent to new_node
                    if near_node.parent is not None:
                        near_node.parent.children.remove(near_node)
                    near_node.parent = new_node
                    new_node.children.append(near_node)
                    # Propagate cost improvement to all descendants
                    self._propagate_cost(near_node, rewire_cost)

            # --- Step 8: Check goal ---
            dist_to_goal = new_point.distance_to(goal)
            if dist_to_goal <= self.goal_threshold:
                # Can we connect directly to the goal?
                if self.env.is_collision_free_segment(new_point, goal):
                    goal_cost = new_node.cost + self.physics.segment_cost(
                        new_point, goal
                    )
                    if goal_cost < best_goal_cost:
                        best_goal_cost = goal_cost
                        best_goal_node = new_node

            # Progress logging every 500 iterations
            if (iteration + 1) % 500 == 0:
                status = f"  Iteration {iteration + 1}/{self.max_iterations}, "
                status += f"nodes: {len(nodes)}, "
                if best_goal_node:
                    status += f"best cost: {best_goal_cost:.1f} J"
                else:
                    status += "no path yet"
                print(status)

        # Extract path
        if best_goal_node is None:
            print("No path found!")
            return None

        path = self._extract_path(best_goal_node, goal)
        print(f"  Path found! Nodes: {len(path)}, Energy: {best_goal_cost:.1f} J")
        return path

    def _propagate_cost(self, node: RRTNode, new_cost: float) -> None:
        """Recursively update costs after rewiring.

        When we rewire a node, all its descendants also get cheaper paths.
        We must propagate this improvement down the tree.
        """
        cost_delta = node.cost - new_cost
        node.cost = new_cost
        for child in node.children:
            self._propagate_cost(child, child.cost - cost_delta)

    def _extract_path(self, goal_parent: RRTNode, goal: Point3D) -> list[Point3D]:
        """Trace back from goal to start through parent pointers."""
        path = [goal]
        current = goal_parent
        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path


# =============================================================================
# Path Smoothing
# =============================================================================

def smooth_path(
    path: list[Point3D],
    env: DroneEnvironment,
    iterations: int = 100,
) -> list[Point3D]:
    """Smooth an RRT* path using random shortcutting.

    RRT* paths have unnecessary zigzags because the tree is built from
    random samples. Shortcutting picks two random points on the path and
    tries to connect them directly — if the shortcut is collision-free,
    we remove all intermediate waypoints.

    This is simple but effective. More sophisticated approaches (B-spline,
    minimum-snap trajectory) give smoother curves but require solving
    optimization problems.
    """
    if len(path) <= 2:
        return path

    smoothed = list(path)

    for _ in range(iterations):
        if len(smoothed) <= 2:
            break

        # Pick two random indices (ensuring i < j with gap >= 2)
        i = random.randint(0, len(smoothed) - 3)
        j = random.randint(i + 2, len(smoothed) - 1)

        # Try direct connection
        if env.is_collision_free_segment(smoothed[i], smoothed[j]):
            # Remove intermediate waypoints
            smoothed = smoothed[:i + 1] + smoothed[j:]

    return smoothed


# =============================================================================
# Path Analysis
# =============================================================================

def analyze_path(
    path: list[Point3D], physics: DronePhysics
) -> dict[str, float]:
    """Compute detailed energy breakdown and path statistics.

    Returns a dictionary with:
    - total_energy: Total energy in Joules
    - total_distance: Euclidean path length in meters
    - horizontal_distance: Distance projected onto the xy-plane
    - total_climb: Total altitude gained (m)
    - total_descent: Total altitude lost (m)
    - max_altitude: Peak altitude reached
    - min_altitude: Lowest altitude
    - n_waypoints: Number of waypoints
    """
    total_energy = 0.0
    total_distance = 0.0
    horiz_distance = 0.0
    total_climb = 0.0
    total_descent = 0.0
    max_alt = path[0].z
    min_alt = path[0].z

    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        total_energy += physics.segment_cost(a, b)
        total_distance += a.distance_to(b)
        horiz_distance += a.horizontal_distance_to(b)

        dz = b.z - a.z
        if dz > 0:
            total_climb += dz
        else:
            total_descent += abs(dz)

        max_alt = max(max_alt, b.z)
        min_alt = min(min_alt, b.z)

    return {
        "total_energy": total_energy,
        "total_distance": total_distance,
        "horizontal_distance": horiz_distance,
        "total_climb": total_climb,
        "total_descent": total_descent,
        "max_altitude": max_alt,
        "min_altitude": min_alt,
        "n_waypoints": len(path),
    }


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
# Demo: Build a realistic urban environment
# =============================================================================

def create_urban_environment() -> DroneEnvironment:
    """Create a city-like environment with buildings, trees, and a no-fly zone.

    Layout (100m x 100m x 50m):
    - Several buildings of varying heights
    - A cluster of trees (cylinders)
    - One no-fly zone (simulating restricted airspace)
    """
    env = DroneEnvironment(
        x_range=(0, 100),
        y_range=(0, 100),
        z_range=(2, 50),  # Minimum 2m altitude (above ground clutter)
    )

    # Buildings (rectangular prisms)
    buildings = [
        BoxObstacle(20, 30, 20, 35, 0, 25, safety_margin=2.0),   # Tall office building
        BoxObstacle(50, 65, 10, 25, 0, 15, safety_margin=2.0),   # Wide warehouse
        BoxObstacle(40, 50, 50, 60, 0, 30, safety_margin=2.0),   # Apartment tower
        BoxObstacle(70, 80, 60, 75, 0, 20, safety_margin=2.0),   # Medium building
        BoxObstacle(10, 20, 60, 70, 0, 12, safety_margin=2.0),   # Small building
    ]
    for b in buildings:
        env.add_obstacle(b)

    # Trees (cylinders) — shorter but spread around
    trees = [
        CylinderObstacle(35, 45, 3, 0, 10, safety_margin=1.5),
        CylinderObstacle(60, 45, 3, 0, 12, safety_margin=1.5),
        CylinderObstacle(25, 80, 4, 0, 14, safety_margin=1.5),
        CylinderObstacle(85, 30, 3, 0, 8, safety_margin=1.5),
    ]
    for t in trees:
        env.add_obstacle(t)

    # No-fly zone — restricted airspace near a helipad
    env.add_no_fly_zone(NoFlyZone(80, 100, 80, 100, 0, 50))

    return env


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    random.seed(42)  # Reproducible results

    print("=" * 60)
    print("  Day 70: Autonomous Drone Path Planning (3D RRT*)")
    print("=" * 60)

    # --- Build environment ---
    print("\n[1] Creating urban environment...")
    env = create_urban_environment()
    print(f"    World: {env.x_range} x {env.y_range} x {env.z_range}")
    print(f"    Obstacles: {len(env.obstacles)} (buildings + trees)")
    print(f"    No-fly zones: {len(env.no_fly_zones)}")

    # --- Define mission ---
    start = Point3D(5.0, 5.0, 10.0)     # Takeoff point, low altitude
    goal = Point3D(75.0, 90.0, 10.0)    # Delivery point across the city

    print(f"\n[2] Mission: fly from ({start.x}, {start.y}, {start.z}) "
          f"to ({goal.x}, {goal.y}, {goal.z})")

    # --- Configure physics ---
    # Light headwind from the east (positive x direction)
    physics = DronePhysics(
        mass=2.0,
        drag_coeff=0.5,
        climb_coeff=1.5,
        descend_coeff=0.3,
        wind_x=3.0,   # 3 m/s headwind from east
        wind_y=0.0,
        wind_penalty=0.1,
    )
    print(f"    Wind: ({physics.wind_x}, {physics.wind_y}) m/s")

    # --- Run RRT* ---
    print("\n[3] Running RRT* planner...")
    planner = RRTStarPlanner(
        env=env,
        physics=physics,
        step_size=8.0,
        goal_threshold=5.0,
        max_iterations=3000,
        goal_bias=0.1,
        gamma=50.0,
    )

    raw_path = planner.plan(start, goal)

    if raw_path is None:
        print("Planning failed! Try increasing max_iterations or adjusting obstacles.")
        exit(1)

    # --- Analyze raw path ---
    raw_stats = analyze_path(raw_path, physics)
    print_path_analysis(raw_stats, "Raw RRT* Path")

    # --- Smooth the path ---
    print("\n[4] Smoothing path...")
    smoothed_path = smooth_path(raw_path, env, iterations=200)
    smooth_stats = analyze_path(smoothed_path, physics)
    print_path_analysis(smooth_stats, "Smoothed Path")

    # --- Compare ---
    if raw_stats["total_energy"] > 0:
        savings = (1 - smooth_stats["total_energy"] / raw_stats["total_energy"]) * 100
        print(f"\n  Energy savings from smoothing: {savings:.1f}%")
        dist_savings = (1 - smooth_stats["total_distance"] / raw_stats["total_distance"]) * 100
        print(f"  Distance savings from smoothing: {dist_savings:.1f}%")
        wp_reduction = (1 - smooth_stats["n_waypoints"] / raw_stats["n_waypoints"]) * 100
        print(f"  Waypoint reduction: {wp_reduction:.1f}%")

    # --- Print waypoints ---
    print(f"\n[5] Smoothed path waypoints ({len(smoothed_path)} points):")
    for i, wp in enumerate(smoothed_path):
        # Show segment cost to next waypoint
        if i < len(smoothed_path) - 1:
            cost = physics.segment_cost(wp, smoothed_path[i + 1])
            print(f"    WP{i:2d}: ({wp.x:6.1f}, {wp.y:6.1f}, {wp.z:6.1f})  "
                  f"-> next: {cost:.1f} J")
        else:
            print(f"    WP{i:2d}: ({wp.x:6.1f}, {wp.y:6.1f}, {wp.z:6.1f})  [GOAL]")

    # --- Energy comparison: straight line vs planned path ---
    print("\n[6] Comparison with straight-line (if no obstacles):")
    straight_cost = physics.segment_cost(start, goal)
    print(f"    Straight-line energy:  {straight_cost:.1f} J")
    print(f"    Planned path energy:   {smooth_stats['total_energy']:.1f} J")
    overhead = (smooth_stats["total_energy"] / straight_cost - 1) * 100
    print(f"    Obstacle overhead:     {overhead:.1f}%")

    print("\n  Planning complete!")
