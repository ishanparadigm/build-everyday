"""
Day 078: Multi-Robot Coordination

A complete implementation of multi-robot task allocation and formation control:
1. Hungarian algorithm for optimal task assignment
2. Auction-based decentralized task allocation
3. Potential field formation controller
4. Velocity obstacle collision avoidance
5. Full simulation coordinator

Run: python3 solution.py
"""

import math
import copy
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Vec2:
    """2D vector with basic operations."""
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self.__mul__(scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self) -> "Vec2":
        m = self.magnitude()
        if m < 1e-10:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / m, self.y / m)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).magnitude()

    def __repr__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f})"


@dataclass
class Task:
    """A task at a location that needs to be performed by a robot."""
    id: int
    position: Vec2
    priority: float = 1.0  # Higher = more urgent
    completed: bool = False

    def __repr__(self) -> str:
        status = "done" if self.completed else "pending"
        return f"Task{self.id}@{self.position} [{status}]"


@dataclass
class Robot:
    """An autonomous robot with position, velocity, and assigned tasks."""
    id: int
    position: Vec2
    velocity: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    max_speed: float = 2.0
    radius: float = 0.5
    assigned_task: Optional[Task] = None
    tasks_completed: int = 0

    def __repr__(self) -> str:
        return f"Robot{self.id}@{self.position}"


@dataclass
class Obstacle:
    """Static circular obstacle in the environment."""
    position: Vec2
    radius: float


# =============================================================================
# Task Allocation — Hungarian Algorithm
# =============================================================================

def compute_cost_matrix(robots: List[Robot], tasks: List[Task]) -> List[List[float]]:
    """
    Build cost matrix where C[i][j] = cost for robot i to do task j.

    Cost is Euclidean distance divided by task priority — higher priority tasks
    have lower effective cost, making them more attractive.

    We handle rectangular matrices (N robots != M tasks) by padding with large
    values. This is necessary because the Hungarian algorithm requires a square matrix.
    """
    n = max(len(robots), len(tasks))
    # Initialize with large cost (effectively "don't assign")
    INF = 1e9
    cost = [[INF] * n for _ in range(n)]

    for i, robot in enumerate(robots):
        for j, task in enumerate(tasks):
            if not task.completed:
                # Distance weighted by inverse priority — urgent tasks are "cheaper"
                dist = robot.position.distance_to(task.position)
                cost[i][j] = dist / task.priority
    return cost


def hungarian_algorithm(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Solve the assignment problem using the Hungarian (Kuhn-Munkres) algorithm.

    Returns optimal assignment as list of (robot_index, task_index) pairs.

    The algorithm works by finding a system of zero-cost entries in a modified
    cost matrix that allows a perfect matching. It alternates between:
    - Covering all zeros with minimum lines (König's theorem)
    - Adjusting the matrix to create new zeros where needed

    Time complexity: O(n³) — cubic in the matrix dimension.
    This is optimal for the assignment problem; no algorithm can do better
    in the worst case for general cost matrices.
    """
    n = len(cost_matrix)
    # Deep copy so we don't modify the original
    C = [row[:] for row in cost_matrix]

    # Step 1: Row reduction — subtract row minimum from each row.
    # This ensures each row has at least one zero, meaning each robot
    # has at least one "free" task relative to its other options.
    for i in range(n):
        min_val = min(C[i])
        if min_val < 1e8:  # Don't reduce rows that are all INF
            for j in range(n):
                C[i][j] -= min_val

    # Step 2: Column reduction — same logic for tasks.
    for j in range(n):
        min_val = min(C[i][j] for i in range(n))
        if min_val < 1e8:
            for i in range(n):
                C[i][j] -= min_val

    # Now we need to find maximum matching using zeros.
    # We use a simplified approach: augmenting path method on the zero entries.

    # Track assignments: row_assign[i] = column assigned to row i (-1 if none)
    row_assign = [-1] * n
    col_assign = [-1] * n

    def try_augment(row: int, visited_cols: List[bool]) -> bool:
        """Try to find an augmenting path from this row through zero entries."""
        for col in range(n):
            if C[row][col] < 1e-9 and not visited_cols[col]:
                visited_cols[col] = True
                # If column is unassigned, or we can reassign its current row
                if col_assign[col] == -1 or try_augment(col_assign[col], visited_cols):
                    row_assign[row] = col
                    col_assign[col] = row
                    return True
        return False

    # Iteratively adjust matrix and find matchings
    for iteration in range(n * 2):  # Safety bound on iterations
        # Try to find a complete matching with current zeros
        # Reset and rebuild matching
        row_assign = [-1] * n
        col_assign = [-1] * n
        matched = 0
        for i in range(n):
            visited = [False] * n
            if try_augment(i, visited):
                matched += 1

        if matched == n:
            break  # Found perfect matching!

        # Not all rows matched — need to create more zeros.
        # Find minimum uncovered value and adjust matrix.

        # Determine which rows are matched and which aren't
        row_covered = [row_assign[i] != -1 for i in range(n)]
        col_covered = [False] * n

        # Mark columns covered by matched rows' zeros
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not row_covered[i]:
                    for j in range(n):
                        if C[i][j] < 1e-9 and not col_covered[j]:
                            col_covered[j] = True
                            changed = True
            for j in range(n):
                if col_covered[j]:
                    for i in range(n):
                        if col_assign[j] == i and row_covered[i]:
                            row_covered[i] = False
                            changed = True

        # Invert row coverage for the standard algorithm interpretation
        row_lines = [not rc for rc in row_covered]
        col_lines = col_covered

        # Find minimum uncovered element
        min_uncovered = float('inf')
        for i in range(n):
            for j in range(n):
                if not row_lines[i] and not col_lines[j]:
                    min_uncovered = min(min_uncovered, C[i][j])

        if min_uncovered >= 1e8 or min_uncovered == float('inf'):
            break  # Can't improve further

        # Subtract from uncovered, add to doubly-covered
        for i in range(n):
            for j in range(n):
                if not row_lines[i] and not col_lines[j]:
                    C[i][j] -= min_uncovered
                elif row_lines[i] and col_lines[j]:
                    C[i][j] += min_uncovered

    # Extract assignment pairs (only valid robot-task pairs)
    assignments = []
    for i in range(n):
        if row_assign[i] != -1:
            assignments.append((i, row_assign[i]))
    return assignments


def allocate_tasks_hungarian(
    robots: List[Robot], tasks: List[Task]
) -> Dict[int, int]:
    """
    Optimally assign tasks to robots using the Hungarian algorithm.

    Returns mapping of robot_id -> task_id.

    Why Hungarian over greedy? A greedy approach (each robot picks nearest task)
    can produce assignments that are globally suboptimal. Example: if two robots
    are close to the same task, greedy assigns one arbitrarily, potentially forcing
    the other to travel far. Hungarian finds the global optimum.
    """
    pending_tasks = [t for t in tasks if not t.completed]
    if not pending_tasks or not robots:
        return {}

    cost = compute_cost_matrix(robots, pending_tasks)
    assignments = hungarian_algorithm(cost)

    result = {}
    for robot_idx, task_idx in assignments:
        if robot_idx < len(robots) and task_idx < len(pending_tasks):
            result[robots[robot_idx].id] = pending_tasks[task_idx].id
    return result


# =============================================================================
# Task Allocation — Auction-Based (Decentralized)
# =============================================================================

def auction_based_allocation(
    robots: List[Robot], tasks: List[Task], rounds: int = 10
) -> Dict[int, int]:
    """
    Decentralized task allocation using an auction mechanism.

    Each round:
    1. Each unassigned robot bids on its best remaining task
    2. Tasks are awarded to the lowest bidder (cheapest to execute)
    3. Outbid robots re-enter the next round

    This converges to near-optimal in O(rounds * N * M) time.
    The beauty of auction-based approaches is resilience: if a robot fails,
    its tasks naturally get re-auctioned to remaining robots.

    The epsilon increment prevents cycling — each bid must improve by at least
    epsilon, guaranteeing termination. This is the key insight from Bertsekas'
    auction algorithm.
    """
    pending = [t for t in tasks if not t.completed]
    if not pending or not robots:
        return {}

    # Track current prices (how much each task "costs" beyond distance)
    prices = {t.id: 0.0 for t in pending}
    assignments: Dict[int, int] = {}  # robot_id -> task_id
    epsilon = 0.1  # Minimum bid increment — prevents cycling

    for round_num in range(rounds):
        # Find unassigned robots
        assigned_robots = set(assignments.keys())
        unassigned = [r for r in robots if r.id not in assigned_robots]

        if not unassigned:
            break

        for robot in unassigned:
            # Calculate value of each task: negative distance (closer = better)
            # minus current price. Robot bids on the task with highest net value.
            best_task_id = None
            best_value = -float('inf')
            second_value = -float('inf')

            for task in pending:
                if task.completed:
                    continue
                value = -robot.position.distance_to(task.position) / task.priority - prices[task.id]
                if value > best_value:
                    second_value = best_value
                    best_value = value
                    best_task_id = task.id
                elif value > second_value:
                    second_value = value

            if best_task_id is None:
                continue

            # Bid = current price + (value difference + epsilon)
            # The value difference ensures the bid reflects how much this robot
            # specifically wants this task vs its next-best option
            bid_increment = best_value - second_value + epsilon
            prices[best_task_id] += bid_increment

            # Check if task was already assigned to another robot
            prev_owner = None
            for rid, tid in assignments.items():
                if tid == best_task_id:
                    prev_owner = rid
                    break

            # Reassign: kick out previous owner (they'll rebid next round)
            if prev_owner is not None:
                del assignments[prev_owner]
            assignments[robot.id] = best_task_id

    return assignments


# =============================================================================
# Formation Control — Potential Fields
# =============================================================================

def compute_formation_positions(
    center: Vec2, num_robots: int, formation: str = "circle", spacing: float = 3.0
) -> List[Vec2]:
    """
    Compute desired positions for a given formation shape.

    Formations:
    - "circle": Robots equally spaced on a circle of radius `spacing`
    - "line": Robots in a horizontal line with `spacing` between them
    - "wedge": V-shape formation (like migrating geese — reduces drag for followers)

    The formation is centered on `center`. In practice, the center would track
    a virtual leader or the centroid of task locations.
    """
    positions = []
    if formation == "circle":
        for i in range(num_robots):
            angle = 2 * math.pi * i / num_robots
            pos = Vec2(
                center.x + spacing * math.cos(angle),
                center.y + spacing * math.sin(angle),
            )
            positions.append(pos)

    elif formation == "line":
        start_x = center.x - spacing * (num_robots - 1) / 2
        for i in range(num_robots):
            positions.append(Vec2(start_x + i * spacing, center.y))

    elif formation == "wedge":
        # V-shape: leader at front, others spread behind
        positions.append(Vec2(center.x, center.y + spacing))
        for i in range(1, num_robots):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2
            positions.append(Vec2(
                center.x + side * row * spacing * 0.7,
                center.y - row * spacing * 0.5,
            ))
    else:
        raise ValueError(f"Unknown formation: {formation}")

    return positions


def formation_force(
    robot: Robot,
    goal_pos: Vec2,
    all_robots: List[Robot],
    obstacles: List[Obstacle],
    k_attract: float = 1.0,
    k_repel_robot: float = 2.0,
    k_repel_obstacle: float = 3.0,
    safety_dist_robot: float = 2.0,
    safety_dist_obstacle: float = 3.0,
) -> Vec2:
    """
    Compute the net force on a robot from potential fields.

    Three force components, inspired by physics but tuned for robotics:

    1. ATTRACTION to goal: Linear spring pulling toward formation position.
       Linear (not quadratic) because we want steady approach, not acceleration
       that overshoots. F = -k * (pos - goal)

    2. REPULSION from robots: Inverse-distance force pushing away from nearby robots.
       Only active within safety_dist — beyond that, robots don't interact.
       The 1/d² falloff means force grows rapidly as collision approaches.
       F = k * (1/d - 1/d0) * (1/d²) * direction

    3. REPULSION from obstacles: Same math as robot repulsion but stronger
       (k_repel_obstacle > k_repel_robot) because obstacles don't move.

    The balance of these forces creates stable formations: attraction pulls
    robots toward their spots, repulsion prevents collisions, and the
    equilibrium point IS the formation.
    """
    # Attraction toward formation goal
    diff = goal_pos - robot.position
    f_attract = diff * k_attract

    # Repulsion from other robots
    f_repel = Vec2(0.0, 0.0)
    for other in all_robots:
        if other.id == robot.id:
            continue
        d = robot.position.distance_to(other.position)
        if 0 < d < safety_dist_robot:
            # Force magnitude: stronger as distance decreases
            direction = (robot.position - other.position).normalized()
            magnitude = k_repel_robot * (1.0 / d - 1.0 / safety_dist_robot) * (1.0 / (d * d))
            f_repel = f_repel + direction * magnitude

    # Repulsion from obstacles
    f_obstacle = Vec2(0.0, 0.0)
    for obs in obstacles:
        d = robot.position.distance_to(obs.position) - obs.radius
        d = max(d, 0.01)  # Avoid division by zero
        if d < safety_dist_obstacle:
            direction = (robot.position - obs.position).normalized()
            magnitude = k_repel_obstacle * (1.0 / d - 1.0 / safety_dist_obstacle) * (1.0 / (d * d))
            f_obstacle = f_obstacle + direction * magnitude

    return f_attract + f_repel + f_obstacle


# =============================================================================
# Collision Avoidance — Velocity Obstacles (Simplified ORCA)
# =============================================================================

def compute_safe_velocity(
    robot: Robot,
    desired_velocity: Vec2,
    all_robots: List[Robot],
    time_horizon: float = 5.0,
) -> Vec2:
    """
    Adjust desired velocity to avoid collisions using velocity obstacles.

    The key idea: given robot A's desired velocity and robot B's current velocity,
    we check if A would collide with B within `time_horizon` seconds. If so,
    we project A's velocity onto the nearest safe direction.

    ORCA (Optimal Reciprocal Collision Avoidance) insight: each robot takes
    HALF the responsibility for avoidance. This prevents the "hallway dance"
    where both agents dodge the same way. By splitting responsibility 50/50,
    the system is provably collision-free if all agents comply.

    Simplified here: we use a velocity damping approach when collision is imminent.
    Full ORCA would compute half-planes and solve a linear program — correct but
    complex for a learning exercise.
    """
    safe_vel = Vec2(desired_velocity.x, desired_velocity.y)

    for other in all_robots:
        if other.id == robot.id:
            continue

        # Relative position and velocity
        rel_pos = other.position - robot.position
        rel_vel = desired_velocity - other.velocity
        dist = rel_pos.magnitude()
        combined_radius = robot.radius + other.radius + 0.5  # Safety margin

        if dist < combined_radius:
            # Already too close — push apart
            escape_dir = (robot.position - other.position).normalized()
            safe_vel = safe_vel + escape_dir * robot.max_speed * 0.5
            continue

        # Check if relative velocity points toward collision
        # Time to closest approach
        closing_speed = rel_vel.dot(rel_pos.normalized())
        if closing_speed <= 0:
            continue  # Moving apart, no collision

        time_to_closest = dist / closing_speed
        if time_to_closest > time_horizon:
            continue  # Collision too far in the future

        # Closest approach distance
        closest_dist = math.sqrt(max(0, dist ** 2 - (closing_speed * time_to_closest) ** 2))
        if closest_dist > combined_radius:
            continue  # Will miss each other

        # Collision imminent — deflect velocity perpendicular to line between robots
        # Each robot takes half the responsibility (ORCA principle)
        perp = Vec2(-rel_pos.y, rel_pos.x).normalized()
        deflection_strength = (combined_radius - closest_dist) / combined_radius
        # Add perpendicular component, scale by urgency (closer = stronger)
        safe_vel = safe_vel + perp * (deflection_strength * robot.max_speed * 0.5)

    # Clamp to max speed
    speed = safe_vel.magnitude()
    if speed > robot.max_speed:
        safe_vel = safe_vel.normalized() * robot.max_speed

    return safe_vel


# =============================================================================
# Simulation Coordinator
# =============================================================================

class MultiRobotCoordinator:
    """
    Orchestrates multi-robot coordination: allocation, formation, navigation.

    The coordinator runs a loop:
    1. Allocate/reallocate tasks (when tasks complete or new ones arrive)
    2. Compute formation positions around task centroid
    3. For each robot: compute formation force -> desired velocity -> safe velocity
    4. Update positions
    5. Check task completion
    """

    def __init__(
        self,
        robots: List[Robot],
        tasks: List[Task],
        obstacles: List[Obstacle],
        formation: str = "circle",
        task_completion_dist: float = 1.0,
    ):
        self.robots = robots
        self.tasks = tasks
        self.obstacles = obstacles
        self.formation = formation
        self.task_completion_dist = task_completion_dist
        self.time = 0.0
        self.allocation: Dict[int, int] = {}
        self.history: List[Dict] = []  # For tracking simulation state

    def allocate(self, method: str = "hungarian") -> Dict[int, int]:
        """Run task allocation and assign results to robots."""
        pending = [t for t in self.tasks if not t.completed]
        if not pending:
            return {}

        if method == "hungarian":
            self.allocation = allocate_tasks_hungarian(self.robots, pending)
        elif method == "auction":
            self.allocation = auction_based_allocation(self.robots, pending)
        else:
            raise ValueError(f"Unknown allocation method: {method}")

        # Update robot assigned_task references
        task_map = {t.id: t for t in self.tasks}
        for robot in self.robots:
            if robot.id in self.allocation:
                robot.assigned_task = task_map.get(self.allocation[robot.id])
            else:
                robot.assigned_task = None

        return self.allocation

    def step(self, dt: float = 0.1) -> bool:
        """
        Advance simulation by dt seconds. Returns True if all tasks complete.

        Each timestep:
        1. Each robot computes its goal (assigned task or formation position)
        2. Potential field gives desired velocity
        3. ORCA adjusts for collision avoidance
        4. Position updates with velocity * dt
        5. Check if any robot reached its task
        """
        # Determine goal positions — go directly to assigned task
        goals: Dict[int, Vec2] = {}
        for robot in self.robots:
            if robot.assigned_task and not robot.assigned_task.completed:
                goals[robot.id] = robot.assigned_task.position
            else:
                goals[robot.id] = robot.position  # Stay put if no task

        # Compute forces and velocities
        for robot in self.robots:
            goal = goals[robot.id]
            force = formation_force(
                robot, goal, self.robots, self.obstacles
            )
            # Force -> desired velocity (with speed limit)
            desired_vel = force
            speed = desired_vel.magnitude()
            if speed > robot.max_speed:
                desired_vel = desired_vel.normalized() * robot.max_speed

            # Collision avoidance
            safe_vel = compute_safe_velocity(robot, desired_vel, self.robots)
            robot.velocity = safe_vel

        # Update positions
        for robot in self.robots:
            robot.position = robot.position + robot.velocity * dt

        # Check task completion
        task_map = {t.id: t for t in self.tasks}
        for robot in self.robots:
            if robot.assigned_task and not robot.assigned_task.completed:
                dist = robot.position.distance_to(robot.assigned_task.position)
                if dist < self.task_completion_dist:
                    robot.assigned_task.completed = True
                    robot.tasks_completed += 1

        self.time += dt

        # Record state for analysis
        self.history.append({
            "time": self.time,
            "positions": [(r.id, r.position.x, r.position.y) for r in self.robots],
            "completed": sum(1 for t in self.tasks if t.completed),
        })

        return all(t.completed for t in self.tasks)

    def run(
        self, max_steps: int = 1000, dt: float = 0.1, method: str = "hungarian",
        reallocate_interval: int = 20, verbose: bool = True
    ) -> float:
        """
        Run full simulation until all tasks done or max_steps reached.

        Reallocates tasks every `reallocate_interval` steps to handle the case
        where a closer robot becomes available after completing its own task.

        Returns total time elapsed.
        """
        self.allocate(method)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Multi-Robot Coordination Simulation")
            print(f"  {len(self.robots)} robots, {len(self.tasks)} tasks, method={method}")
            print(f"{'='*60}")
            print(f"\nInitial allocation:")
            task_map = {t.id: t for t in self.tasks}
            for rid, tid in sorted(self.allocation.items()):
                r = next(r for r in self.robots if r.id == rid)
                t = task_map[tid]
                dist = r.position.distance_to(t.position)
                print(f"  Robot {rid} -> Task {tid} (distance: {dist:.1f})")

        for step_num in range(max_steps):
            # Periodically reallocate (handles completed tasks, dynamic scenarios)
            if step_num > 0 and step_num % reallocate_interval == 0:
                old_alloc = dict(self.allocation)
                self.allocate(method)
                if verbose and self.allocation != old_alloc:
                    print(f"\n  [t={self.time:.1f}] Reallocated tasks")

            done = self.step(dt)

            # Print progress at milestones
            if verbose and step_num % 50 == 0 and step_num > 0:
                completed = sum(1 for t in self.tasks if t.completed)
                print(f"  [t={self.time:.1f}] {completed}/{len(self.tasks)} tasks complete")

            if done:
                if verbose:
                    print(f"\n  All tasks completed at t={self.time:.1f}!")
                    print(f"\n  Robot performance:")
                    for r in self.robots:
                        print(f"    Robot {r.id}: {r.tasks_completed} tasks completed")
                return self.time

        if verbose:
            completed = sum(1 for t in self.tasks if t.completed)
            print(f"\n  Simulation ended: {completed}/{len(self.tasks)} tasks done at t={self.time:.1f}")
        return self.time


# =============================================================================
# Demo and Analysis
# =============================================================================

def demo_hungarian_vs_auction():
    """Compare Hungarian and auction-based allocation on the same scenario."""
    print("\n" + "=" * 60)
    print("  DEMO 1: Hungarian vs Auction Allocation")
    print("=" * 60)

    random.seed(42)

    # Create robots and tasks
    robots = [
        Robot(0, Vec2(0, 0)),
        Robot(1, Vec2(10, 0)),
        Robot(2, Vec2(0, 10)),
        Robot(3, Vec2(10, 10)),
    ]
    tasks = [
        Task(0, Vec2(9, 1), priority=1.0),
        Task(1, Vec2(1, 9), priority=2.0),  # High priority
        Task(2, Vec2(5, 5), priority=1.0),
        Task(3, Vec2(8, 8), priority=1.5),
    ]

    # Show cost matrix
    cost = compute_cost_matrix(robots, tasks)
    print("\nCost matrix (distance / priority):")
    print(f"{'':>10}", end="")
    for t in tasks:
        print(f"  Task{t.id}(p={t.priority})", end="")
    print()
    for i, r in enumerate(robots):
        print(f"  Robot{r.id}  ", end="")
        for j in range(len(tasks)):
            if cost[i][j] < 1e8:
                print(f"  {cost[i][j]:>10.2f}  ", end="")
            else:
                print(f"  {'INF':>10}  ", end="")
        print()

    # Hungarian
    h_assign = allocate_tasks_hungarian(robots, tasks)
    total_h = sum(
        robots[rid].position.distance_to(tasks[tid].position)
        for rid, tid in h_assign.items()
        if tid < len(tasks)
    )
    print(f"\nHungarian assignment (total distance: {total_h:.2f}):")
    for rid, tid in sorted(h_assign.items()):
        d = robots[rid].position.distance_to(tasks[tid].position)
        print(f"  Robot {rid} -> Task {tid} (dist={d:.2f})")

    # Reset tasks
    for t in tasks:
        t.completed = False

    # Auction
    a_assign = auction_based_allocation(robots, tasks)
    total_a = sum(
        robots[rid].position.distance_to(tasks[tid].position)
        for rid, tid in a_assign.items()
        if tid < len(tasks)
    )
    print(f"\nAuction assignment (total distance: {total_a:.2f}):")
    for rid, tid in sorted(a_assign.items()):
        d = robots[rid].position.distance_to(tasks[tid].position)
        print(f"  Robot {rid} -> Task {tid} (dist={d:.2f})")

    improvement = (total_a - total_h) / total_a * 100 if total_a > 0 else 0
    print(f"\nHungarian saves {improvement:.1f}% distance vs auction")


def demo_formation_control():
    """Show how potential fields create and maintain formations."""
    print("\n" + "=" * 60)
    print("  DEMO 2: Formation Control with Potential Fields")
    print("=" * 60)

    # 4 robots that need to form a circle
    robots = [
        Robot(0, Vec2(0, 0)),
        Robot(1, Vec2(1, 0)),
        Robot(2, Vec2(0, 1)),
        Robot(3, Vec2(1, 1)),
    ]
    center = Vec2(5, 5)
    goals = compute_formation_positions(center, 4, "circle", spacing=3.0)

    print(f"\nTarget formation (circle, radius=3.0, center={center}):")
    for i, g in enumerate(goals):
        print(f"  Robot {i} target: {g}")

    print(f"\nSimulating formation convergence...")
    obstacles = [Obstacle(Vec2(3, 3), 1.0)]
    dt = 0.1

    for step in range(200):
        for i, robot in enumerate(robots):
            force = formation_force(robot, goals[i], robots, obstacles)
            vel = force
            speed = vel.magnitude()
            if speed > robot.max_speed:
                vel = vel.normalized() * robot.max_speed
            vel = compute_safe_velocity(robot, vel, robots)
            robot.velocity = vel
            robot.position = robot.position + vel * dt

        if step % 50 == 0:
            errors = [robots[i].position.distance_to(goals[i]) for i in range(4)]
            avg_error = sum(errors) / len(errors)
            print(f"  Step {step:3d}: avg position error = {avg_error:.3f}")

    print(f"\nFinal positions vs targets:")
    for i, robot in enumerate(robots):
        err = robot.position.distance_to(goals[i])
        print(f"  Robot {i}: {robot.position} -> target {goals[i]} (error: {err:.3f})")


def demo_full_coordination():
    """Full scenario: robots coordinate to complete scattered tasks."""
    print("\n" + "=" * 60)
    print("  DEMO 3: Full Multi-Robot Coordination")
    print("=" * 60)

    random.seed(123)

    # 5 robots, 8 tasks scattered in a 20x20 area
    robots = [
        Robot(i, Vec2(random.uniform(0, 5), random.uniform(0, 5)))
        for i in range(5)
    ]
    tasks = [
        Task(i, Vec2(random.uniform(2, 18), random.uniform(2, 18)),
             priority=random.choice([1.0, 1.5, 2.0]))
        for i in range(8)
    ]
    obstacles = [
        Obstacle(Vec2(10, 3), 1.5),
        Obstacle(Vec2(3, 12), 1.0),
    ]

    print(f"\nRobots: {robots}")
    print(f"Tasks: {tasks}")
    print(f"Obstacles: {len(obstacles)} static obstacles")

    # Run with Hungarian allocation
    coordinator = MultiRobotCoordinator(
        robots=robots, tasks=tasks, obstacles=obstacles,
        formation="circle", task_completion_dist=1.0,
    )
    time_h = coordinator.run(
        max_steps=2000, dt=0.1, method="hungarian",
        reallocate_interval=20, verbose=True,
    )

    # Reset and run with auction
    for t in tasks:
        t.completed = False
    for r in robots:
        r.tasks_completed = 0
    robots2 = [
        Robot(i, Vec2(robots[i].position.x, robots[i].position.y))
        for i in range(len(robots))
    ]
    # Restore original positions
    random.seed(123)
    robots2 = [
        Robot(i, Vec2(random.uniform(0, 5), random.uniform(0, 5)))
        for i in range(5)
    ]
    for t in tasks:
        t.completed = False

    coordinator2 = MultiRobotCoordinator(
        robots=robots2, tasks=tasks, obstacles=obstacles,
        formation="circle", task_completion_dist=1.0,
    )
    time_a = coordinator2.run(
        max_steps=2000, dt=0.1, method="auction",
        reallocate_interval=20, verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Hungarian: {time_h:.1f}s")
    print(f"  Auction:   {time_a:.1f}s")
    if time_h < time_a:
        print(f"  Hungarian was {(time_a-time_h)/time_a*100:.1f}% faster")
    else:
        print(f"  Auction was {(time_h-time_a)/time_h*100:.1f}% faster")


if __name__ == "__main__":
    demo_hungarian_vs_auction()
    demo_formation_control()
    demo_full_coordination()
    print("\n" + "=" * 60)
    print("  All demos completed successfully!")
    print("=" * 60)
