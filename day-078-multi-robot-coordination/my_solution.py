"""
Day 078: Multi-Robot Coordination — Your Implementation

Implement multi-robot task allocation and formation control from scratch.

Run tests: python3 -m pytest tests.py
Run this file: python3 my_solution.py
"""

import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# =============================================================================
# Data Structures (provided — do not modify)
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
    priority: float = 1.0
    completed: bool = False


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


@dataclass
class Obstacle:
    """Static circular obstacle in the environment."""
    position: Vec2
    radius: float


# =============================================================================
# Task 1: Cost Matrix
# =============================================================================

def compute_cost_matrix(robots: List[Robot], tasks: List[Task]) -> List[List[float]]:
    """
    Build a square cost matrix where C[i][j] = cost for robot i to do task j.

    Cost = Euclidean distance / task priority.
    Pad with 1e9 for non-existent robot-task pairs (matrix must be square).

    Hint: Think about why we divide by priority — a priority-2 task at distance 10
    should be as attractive as a priority-1 task at distance 5.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Task 2: Hungarian Algorithm
# =============================================================================

def hungarian_algorithm(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Solve the assignment problem optimally.

    Steps:
    1. Row reduction (subtract row min from each row)
    2. Column reduction (subtract col min from each column)
    3. Find maximum matching using augmenting paths on zero entries
    4. If not perfect matching, adjust matrix and repeat

    Returns list of (row, col) assignment pairs.

    Hint: The augmenting path search is like a bipartite matching — try to
    assign each row, and if there's a conflict, recursively try to reassign
    the conflicting row to a different column.
    """
    raise NotImplementedError("TODO: implement this")


def allocate_tasks_hungarian(
    robots: List[Robot], tasks: List[Task]
) -> Dict[int, int]:
    """
    Assign tasks to robots optimally using Hungarian algorithm.
    Returns {robot_id: task_id} mapping.

    Hint: Filter out completed tasks, build cost matrix, run Hungarian,
    then map indices back to robot/task IDs.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Task 3: Auction-Based Allocation
# =============================================================================

def auction_based_allocation(
    robots: List[Robot], tasks: List[Task], rounds: int = 10
) -> Dict[int, int]:
    """
    Decentralized task allocation via auction mechanism.

    Each round:
    1. Unassigned robots bid on their best task
    2. Bid value = -(distance/priority) - current_price
    3. Winner pays: price += (best_value - second_best_value + epsilon)
    4. Previous task owner gets unassigned (rebids next round)

    Hint: The epsilon increment (use 0.1) prevents infinite cycling.
    Track prices per task and assignments per robot separately.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Task 4: Formation Control
# =============================================================================

def compute_formation_positions(
    center: Vec2, num_robots: int, formation: str = "circle", spacing: float = 3.0
) -> List[Vec2]:
    """
    Compute target positions for a formation shape centered at `center`.

    Formations:
    - "circle": Equally spaced on circle of radius `spacing`
    - "line": Horizontal line with `spacing` between robots
    - "wedge": V-shape, leader at front

    Hint: For circle, angle_i = 2*pi*i/n. For line, center the line on `center`.
    """
    raise NotImplementedError("TODO: implement this")


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
    Compute net potential field force on a robot.

    Three components:
    1. Attraction to goal: F = k_attract * (goal - position)
    2. Repulsion from robots: F = k * (1/d - 1/d0) * (1/d^2) * direction, when d < d0
    3. Repulsion from obstacles: Same as robot repulsion but with obstacle params

    Hint: Direction always points AWAY from the repelling object (robot.pos - other.pos).
    Only apply repulsion when distance < safety_dist.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Task 5: Collision Avoidance
# =============================================================================

def compute_safe_velocity(
    robot: Robot,
    desired_velocity: Vec2,
    all_robots: List[Robot],
    time_horizon: float = 5.0,
) -> Vec2:
    """
    Adjust velocity to avoid collisions using velocity obstacles.

    For each other robot:
    1. Compute relative position and velocity
    2. Check if closing (dot product of rel_vel with direction > 0)
    3. Estimate time to closest approach and miss distance
    4. If collision predicted, add perpendicular deflection (half responsibility)
    5. Clamp final velocity to max_speed

    Hint: Perpendicular to rel_pos is Vec2(-rel_pos.y, rel_pos.x).
    Combined radius = robot.radius + other.radius + 0.5 (safety margin).
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    import random
    random.seed(42)

    # Test 1: Cost matrix
    print("=== Test 1: Cost Matrix ===")
    robots = [Robot(0, Vec2(0, 0)), Robot(1, Vec2(10, 0))]
    tasks = [Task(0, Vec2(1, 0)), Task(1, Vec2(9, 0))]
    cost = compute_cost_matrix(robots, tasks)
    print(f"Cost matrix: {cost}")
    print(f"Expected: Robot 0 closer to Task 0, Robot 1 closer to Task 1")

    # Test 2: Hungarian allocation
    print("\n=== Test 2: Hungarian Allocation ===")
    assignment = allocate_tasks_hungarian(robots, tasks)
    print(f"Assignment: {assignment}")
    print(f"Expected: {{0: 0, 1: 1}} (each robot gets nearest task)")

    # Test 3: Auction allocation
    print("\n=== Test 3: Auction Allocation ===")
    for t in tasks:
        t.completed = False
    a_assign = auction_based_allocation(robots, tasks)
    print(f"Assignment: {a_assign}")

    # Test 4: Formation positions
    print("\n=== Test 4: Formation Positions ===")
    positions = compute_formation_positions(Vec2(5, 5), 4, "circle", 3.0)
    print(f"Circle formation: {positions}")

    # Test 5: Formation force
    print("\n=== Test 5: Formation Force ===")
    robot = Robot(0, Vec2(0, 0))
    force = formation_force(robot, Vec2(5, 5), [robot], [])
    print(f"Force toward (5,5): {force}")
    print(f"Should point toward (5,5) with magnitude ~7.07")

    # Test 6: Full coordination
    print("\n=== Test 6: Full Coordination ===")
    robots = [Robot(i, Vec2(random.uniform(0, 5), random.uniform(0, 5))) for i in range(3)]
    tasks = [Task(i, Vec2(random.uniform(5, 15), random.uniform(5, 15))) for i in range(3)]

    alloc = allocate_tasks_hungarian(robots, tasks)
    print(f"Allocation: {alloc}")
    print("Simulating navigation...")

    task_map = {t.id: t for t in tasks}
    for robot in robots:
        if robot.id in alloc:
            robot.assigned_task = task_map[alloc[robot.id]]

    for step in range(500):
        for robot in robots:
            if robot.assigned_task and not robot.assigned_task.completed:
                force = formation_force(robot, robot.assigned_task.position, robots, [])
                vel = force
                speed = vel.magnitude()
                if speed > robot.max_speed:
                    vel = vel.normalized() * robot.max_speed
                vel = compute_safe_velocity(robot, vel, robots)
                robot.velocity = vel
                robot.position = robot.position + vel * 0.1

                if robot.position.distance_to(robot.assigned_task.position) < 1.0:
                    robot.assigned_task.completed = True
                    robot.tasks_completed += 1

    completed = sum(1 for t in tasks if t.completed)
    print(f"Completed: {completed}/{len(tasks)} tasks")
    print("Done!" if completed == len(tasks) else "Some tasks incomplete — check your implementation")
