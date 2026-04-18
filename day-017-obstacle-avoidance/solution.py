"""
Day 17: Obstacle Avoidance using Vector Field Histogram (VFH)

A reactive obstacle avoidance system that:
1. Simulates range sensors around a mobile robot
2. Builds a polar histogram of obstacle density
3. Identifies free valleys (safe steering directions)
4. Selects the best direction balancing goal-seeking, momentum, and smoothness
5. Navigates through an obstacle field to reach a goal

This is a simplified but faithful implementation of the VFH algorithm
as described by Borenstein & Koren (1991).
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ─── World representation ─────────────────────────────────────────────

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
        Cast a ray from (ox, oy) at the given angle and return distance to the
        nearest obstacle intersection. Returns max_range if nothing is hit.

        Uses the geometric ray-circle intersection test:
        A ray from point O in direction D hits a circle with center C and radius r
        when the quadratic  |O + tD - C|^2 = r^2  has a positive real solution.

        Expanding: t^2(D.D) + 2t(D.(O-C)) + (O-C).(O-C) - r^2 = 0
        We want the smallest positive t, then distance = t (since |D| = 1).
        """
        dx = math.cos(angle)
        dy = math.sin(angle)
        closest = max_range

        for obs in self.obstacles:
            # Vector from ray origin to circle center
            fx = ox - obs.x
            fy = oy - obs.y

            # Quadratic coefficients: a*t^2 + b*t + c = 0
            # a = dx^2 + dy^2 = 1 (unit direction vector)
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - obs.radius * obs.radius

            discriminant = b * b - 4.0 * c  # a=1 so 4ac = 4c

            if discriminant < 0:
                continue  # Ray misses this circle entirely

            sqrt_disc = math.sqrt(discriminant)
            # Two intersection distances (could be negative = behind ray origin)
            t1 = (-b - sqrt_disc) / 2.0
            t2 = (-b + sqrt_disc) / 2.0

            # We want the nearest positive intersection
            if t1 > 0:
                closest = min(closest, t1)
            elif t2 > 0:
                # Ray starts inside the circle — we're already colliding
                closest = min(closest, t2)

        # Also check world boundaries (4 walls)
        # For each wall, compute ray parameter t where the ray hits it
        for wall_dist, wall_dir in [
            (0.0 - ox, dx),      # left wall:   x = 0
            (self.width - ox, dx),   # right wall:  x = width
            (0.0 - oy, dy),      # bottom wall: y = 0
            (self.height - oy, dy),  # top wall:    y = height
        ]:
            if abs(wall_dir) > 1e-10:
                t = wall_dist / wall_dir
                if t > 0:
                    closest = min(closest, t)

        return closest


# ─── Robot with simulated sensors ──────────────────────────────────────

@dataclass
class Robot:
    """Mobile robot with range sensors arranged in a ring."""
    x: float
    y: float
    heading: float  # radians, 0 = east, pi/2 = north
    num_sensors: int = 36  # One sensor every 10 degrees
    max_sensor_range: float = 8.0
    sensor_noise_std: float = 0.2  # Gaussian noise on range readings
    step_size: float = 0.5  # Distance moved per timestep

    def get_sensor_angles(self) -> list[float]:
        """
        Return the global angle of each sensor beam.
        Sensors are evenly distributed around the robot.
        """
        return [
            self.heading + (2 * math.pi * i / self.num_sensors)
            for i in range(self.num_sensors)
        ]

    def read_sensors(self, world: World) -> list[tuple[float, float]]:
        """
        Simulate range sensor readings. Returns list of (angle, distance) pairs.
        Adds Gaussian noise to simulate real sensor imperfections.
        Noise is clamped so we never get negative distances.
        """
        readings = []
        for angle in self.get_sensor_angles():
            true_dist = world.ray_cast(self.x, self.y, angle, self.max_sensor_range)
            # Add noise — real sensors are noisy, especially at long range
            noisy_dist = true_dist + random.gauss(0, self.sensor_noise_std)
            noisy_dist = max(0.1, min(noisy_dist, self.max_sensor_range))
            readings.append((angle, noisy_dist))
        return readings

    def move(self, direction: float) -> None:
        """Move robot one step in the given direction (global angle in radians)."""
        self.x += self.step_size * math.cos(direction)
        self.y += self.step_size * math.sin(direction)
        self.heading = direction  # Robot faces the direction it's moving


# ─── VFH Algorithm ─────────────────────────────────────────────────────

@dataclass
class VFHConfig:
    """Configuration parameters for the VFH algorithm."""
    num_sectors: int = 72          # 360/72 = 5 degrees per sector
    threshold: float = 3.0         # Histogram threshold for blocked/free
    a_const: float = 5.0           # Weight constant: weight = a - b*dist
    b_const: float = 0.5           # Must satisfy: a - b*max_range > 0 ideally
    mu_target: float = 5.0         # Cost weight: goal direction
    mu_current: float = 2.0        # Cost weight: current heading (momentum)
    mu_previous: float = 2.0       # Cost weight: previous chosen direction (smoothness)
    min_valley_width: int = 3      # Minimum sectors in a valley for safe passage
    wide_valley_threshold: int = 12  # Valleys wider than this use edge-nearest-to-goal


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]. Essential for angular arithmetic."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angular_distance(a: float, b: float) -> float:
    """Absolute angular distance between two angles, handling wraparound."""
    return abs(normalize_angle(a - b))


def build_histogram(
    readings: list[tuple[float, float]],
    config: VFHConfig,
    max_range: float,
) -> list[float]:
    """
    Build a polar obstacle density histogram from sensor readings.

    Each sector covers (360 / num_sectors) degrees. For each reading,
    we compute a weight inversely proportional to distance (closer = more dangerous)
    and add it to the corresponding sector bin.

    Why aggregate into bins instead of using raw readings?
    - Reduces noise: multiple readings in similar directions reinforce each other
    - Handles sensor sparsity: we might have 36 sensors but 72 sectors,
      so interpolation happens naturally
    - Provides a compact representation for fast processing
    """
    sector_size = 2 * math.pi / config.num_sectors
    histogram = [0.0] * config.num_sectors

    for angle, dist in readings:
        if dist >= max_range:
            continue  # No obstacle detected in this direction

        # Weight: closer obstacles get much higher weight
        # This is the core of VFH — it creates a "repulsive force" that
        # scales with proximity, making nearby obstacles dominate the histogram
        weight = config.a_const - config.b_const * dist
        if weight <= 0:
            continue  # Obstacle too far to matter

        # Determine which sector this reading falls in
        # Normalize angle to [0, 2pi] for sector mapping
        norm_angle = angle % (2 * math.pi)
        sector = int(norm_angle / sector_size) % config.num_sectors

        # Add weight to this sector AND neighbors for smoothing
        # This prevents thin obstacles from being "invisible" between sectors
        histogram[sector] += weight
        left = (sector - 1) % config.num_sectors
        right = (sector + 1) % config.num_sectors
        histogram[left] += weight * 0.5
        histogram[right] += weight * 0.5

    return histogram


def find_valleys(histogram: list[float], config: VFHConfig) -> list[tuple[int, int]]:
    """
    Find contiguous runs of free (below-threshold) sectors.

    Returns list of (start_sector, end_sector) tuples.
    Handles wraparound: if the histogram is free at both sector 0 and sector N-1,
    they form one valley.

    Why require minimum valley width?
    A single free sector between two blocked ones is too narrow for the robot
    to safely pass through. The min_valley_width parameter ensures we only
    consider valleys wide enough for the robot's physical size plus safety margin.
    """
    n = config.num_sectors
    free = [h <= config.threshold for h in histogram]

    # Find runs of consecutive free sectors
    # Handle wraparound by doubling the array and deduplicating
    valleys = []
    in_valley = False
    start = -1

    for i in range(n):
        if free[i] and not in_valley:
            start = i
            in_valley = True
        elif not free[i] and in_valley:
            width = i - start
            if width >= config.min_valley_width:
                valleys.append((start, i - 1))
            in_valley = False

    # Handle valley that wraps around from end to start
    if in_valley:
        # Valley extends to end of array — check if it connects to a valley at the start
        if valleys and valleys[0][0] == 0:
            # Merge: the first valley and this one are actually connected
            first_start, first_end = valleys[0]
            width = (n - start) + (first_end - first_start + 1)
            if width >= config.min_valley_width:
                valleys[0] = (start, first_end)  # Merged valley
            else:
                valleys.pop(0)  # Remove too-narrow merged valley
        else:
            width = n - start
            if width >= config.min_valley_width:
                valleys.append((start, n - 1))

    return valleys


def sector_to_angle(sector: int, num_sectors: int) -> float:
    """Convert sector index to the angle at the center of that sector."""
    sector_size = 2 * math.pi / num_sectors
    return sector * sector_size + sector_size / 2


def select_direction(
    valleys: list[tuple[int, int]],
    goal_angle: float,
    current_heading: float,
    previous_direction: float,
    config: VFHConfig,
) -> Optional[float]:
    """
    Select the best steering direction from candidate valleys.

    For each valley, we generate a candidate direction:
    - Narrow valleys: use the center
    - Wide valleys: use the edge closest to the goal (we don't need to go
      through the middle of a wide opening — cut toward the goal)

    Then evaluate a cost function that balances:
    - Getting closer to the goal (dominant term)
    - Maintaining current heading (momentum — avoids erratic steering)
    - Staying near the previous chosen direction (smoothness — avoids oscillation)

    Returns None if no valley is wide enough (robot is trapped).
    """
    if not valleys:
        return None

    best_cost = float("inf")
    best_angle = None
    n = config.num_sectors

    for start, end in valleys:
        # Compute valley width handling wraparound
        if start <= end:
            width = end - start + 1
        else:
            width = (n - start) + end + 1

        center_sector = (start + width // 2) % n
        center_angle = sector_to_angle(center_sector, n)

        if width > config.wide_valley_threshold:
            # Wide valley: steer toward the edge nearest to the goal
            # This is more efficient — no need to go to the center of a huge opening
            left_angle = sector_to_angle(start, n)
            right_angle = sector_to_angle(end, n)

            # Pick the edge closest to goal, but offset inward by a few sectors for safety
            offset = config.min_valley_width // 2
            left_candidate = sector_to_angle((start + offset) % n, n)
            right_candidate = sector_to_angle((end - offset) % n, n)

            candidates = [left_candidate, right_candidate, center_angle]
        else:
            candidates = [center_angle]

        for candidate in candidates:
            cost = (
                config.mu_target * angular_distance(candidate, goal_angle)
                + config.mu_current * angular_distance(candidate, current_heading)
                + config.mu_previous * angular_distance(candidate, previous_direction)
            )
            if cost < best_cost:
                best_cost = cost
                best_angle = candidate

    return best_angle


# ─── Simulation ────────────────────────────────────────────────────────

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
    Run the full VFH obstacle avoidance simulation.

    Returns the path taken by the robot as a list of (x, y) positions.
    """
    path = [(robot.x, robot.y)]
    previous_direction = robot.heading

    print(f"{'Step':>4} | {'Robot X':>7} | {'Robot Y':>7} | {'Heading':>7} | {'Goal Dist':>9} | {'Status'}")
    print("-" * 70)

    for step in range(max_steps):
        # Check if we've reached the goal
        dist_to_goal = math.sqrt((robot.x - goal_x) ** 2 + (robot.y - goal_y) ** 2)

        if dist_to_goal < goal_tolerance:
            print(f"{step:4d} | {robot.x:7.2f} | {robot.y:7.2f} | {math.degrees(robot.heading):7.1f} | {dist_to_goal:9.2f} | GOAL REACHED!")
            return path

        # Step 1: Read sensors
        readings = robot.read_sensors(world)

        # Step 2: Build polar histogram
        histogram = build_histogram(readings, config, robot.max_sensor_range)

        # Step 3: Find free valleys
        valleys = find_valleys(histogram, config)

        # Step 4: Compute goal angle
        goal_angle = math.atan2(goal_y - robot.y, goal_x - robot.x)

        # Step 5: Select best steering direction
        direction = select_direction(
            valleys, goal_angle, robot.heading, previous_direction, config
        )

        # Step 6: Handle case where robot is stuck (no free valleys)
        if direction is None:
            # Recovery: turn 90 degrees — this is a simple escape heuristic.
            # More sophisticated: reverse, or do a full 360 scan.
            direction = normalize_angle(robot.heading + math.pi / 2)
            status = "STUCK - rotating"
        else:
            # Determine if we're going roughly toward goal or detouring
            if angular_distance(direction, goal_angle) < math.pi / 6:
                status = "-> goal"
            else:
                status = "avoiding"

        # Step 7: Move
        robot.move(direction)
        previous_direction = direction
        path.append((robot.x, robot.y))

        if step % 5 == 0 or status != "-> goal":
            print(f"{step:4d} | {robot.x:7.2f} | {robot.y:7.2f} | {math.degrees(robot.heading):7.1f} | {dist_to_goal:9.2f} | {status}")

    print(f"  ** Did not reach goal in {max_steps} steps")
    return path


def print_ascii_map(
    world: World,
    path: list[tuple[float, float]],
    goal_x: float,
    goal_y: float,
    scale: float = 2.0,
) -> None:
    """
    Print a simple ASCII visualization of the world, obstacles, and robot path.
    Scale controls how many world-units each character represents.
    """
    cols = int(world.width / scale)
    rows = int(world.height / scale)

    # Initialize grid
    grid = [["." for _ in range(cols)] for _ in range(rows)]

    # Draw obstacles
    for obs in world.obstacles:
        for r in range(rows):
            for c in range(cols):
                wx = c * scale + scale / 2
                wy = (rows - 1 - r) * scale + scale / 2  # Flip y for display
                if math.sqrt((wx - obs.x) ** 2 + (wy - obs.y) ** 2) <= obs.radius:
                    grid[r][c] = "#"

    # Draw path
    for px, py in path:
        c = int(px / scale)
        r = rows - 1 - int(py / scale)
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] == ".":
            grid[r][c] = "*"

    # Draw start and goal
    sc = int(path[0][0] / scale)
    sr = rows - 1 - int(path[0][1] / scale)
    if 0 <= sr < rows and 0 <= sc < cols:
        grid[sr][sc] = "S"

    gc = int(goal_x / scale)
    gr = rows - 1 - int(goal_y / scale)
    if 0 <= gr < rows and 0 <= gc < cols:
        grid[gr][gc] = "G"

    # Print
    print("\n--- World Map ---")
    print("S=start  G=goal  #=obstacle  *=path  .=free")
    print("+" + "-" * cols + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * cols + "+")


# ─── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)  # Reproducible results

    # Create a world with various obstacles
    # The layout creates a scenario where the robot can't just go straight to the goal
    world = World(
        width=40.0,
        height=30.0,
        obstacles=[
            # Wall-like barrier in the middle
            Obstacle(15, 15, 3.0),
            Obstacle(18, 13, 2.5),
            Obstacle(20, 16, 2.0),

            # Scattered obstacles
            Obstacle(8, 10, 1.5),
            Obstacle(25, 20, 2.0),
            Obstacle(30, 12, 2.5),
            Obstacle(12, 22, 1.8),
            Obstacle(28, 8, 1.5),
            Obstacle(33, 18, 2.0),
        ],
    )

    # Robot starts bottom-left, goal is top-right
    robot = Robot(x=3.0, y=3.0, heading=math.pi / 4)  # Facing northeast
    goal_x, goal_y = 37.0, 27.0

    config = VFHConfig()

    print("=" * 70)
    print("VFH Obstacle Avoidance Simulation")
    print("=" * 70)
    print(f"Robot start: ({robot.x:.1f}, {robot.y:.1f})")
    print(f"Goal:        ({goal_x:.1f}, {goal_y:.1f})")
    print(f"Obstacles:   {len(world.obstacles)}")
    print(f"Sensors:     {robot.num_sensors} beams, {robot.max_sensor_range:.1f} max range")
    print(f"VFH sectors: {config.num_sectors} ({360/config.num_sectors:.0f} deg each)")
    print()

    # Run the simulation
    path = run_simulation(world, robot, goal_x, goal_y, config)

    # Print summary
    total_distance = sum(
        math.sqrt((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2)
        for i in range(len(path) - 1)
    )
    straight_line = math.sqrt((goal_x - path[0][0]) ** 2 + (goal_y - path[0][1]) ** 2)

    print(f"\n--- Summary ---")
    print(f"Steps taken:       {len(path) - 1}")
    print(f"Path length:       {total_distance:.2f}")
    print(f"Straight-line:     {straight_line:.2f}")
    print(f"Efficiency ratio:  {straight_line / total_distance:.2%}")

    # ASCII map
    print_ascii_map(world, path, goal_x, goal_y)

    # --- Demonstrate the histogram for one reading ---
    print("\n--- Example Polar Histogram (initial position) ---")
    demo_robot = Robot(x=3.0, y=3.0, heading=math.pi / 4)
    readings = demo_robot.read_sensors(world)
    histogram = build_histogram(readings, config, demo_robot.max_sensor_range)

    # Show histogram as a bar chart
    max_h = max(histogram) if max(histogram) > 0 else 1
    bar_width = 40
    print(f"{'Sector':>6} | {'Angle':>6} | {'Value':>6} | Bar")
    for i in range(0, config.num_sectors, 3):  # Every 3rd sector to save space
        angle_deg = (i * 360 / config.num_sectors)
        bar_len = int(histogram[i] / max_h * bar_width)
        blocked = "X" if histogram[i] > config.threshold else " "
        print(f"{i:6d} | {angle_deg:5.0f}° | {histogram[i]:6.2f} | {'█' * bar_len} {blocked}")
