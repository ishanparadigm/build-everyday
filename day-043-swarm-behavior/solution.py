"""
Day 043: Swarm Behavior Simulation

A complete implementation of Reynolds' Boids flocking algorithm with spatial hashing,
obstacle avoidance, goal-seeking, and swarm quality metrics.

Each agent follows only local rules (separation, alignment, cohesion) yet the swarm
exhibits complex emergent behavior — flocking, splitting around obstacles, and
converging on goals — all without any central controller.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Vector2D — lightweight 2D vector for all position/velocity math
# ============================================================================

@dataclass
class Vector2D:
    """Simple 2D vector. We avoid numpy to keep dependencies zero and make
    the math explicit — every operation maps directly to the physics."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x / scalar, self.y / scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> "Vector2D":
        """Return unit vector. Returns zero vector if magnitude is ~0 to avoid
        division by zero — this happens when two agents overlap exactly."""
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / mag, self.y / mag)

    def limit(self, max_val: float) -> "Vector2D":
        """Clamp magnitude to max_val. Critical for realistic motion — without this,
        forces can produce infinite acceleration."""
        mag = self.magnitude()
        if mag > max_val and mag > 1e-10:
            return self.normalize() * max_val
        return Vector2D(self.x, self.y)

    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: "Vector2D") -> float:
        return (self - other).magnitude()

    def __repr__(self) -> str:
        return f"V({self.x:.2f}, {self.y:.2f})"


# ============================================================================
# Obstacle — circular obstacles in the world
# ============================================================================

@dataclass
class Obstacle:
    position: Vector2D
    radius: float


# ============================================================================
# SpatialHash — O(n) neighbor lookups instead of O(n²)
# ============================================================================

class SpatialHash:
    """Grid-based spatial index for fast neighbor queries.

    The world is divided into cells of size `cell_size`. Each agent is placed
    in exactly one cell based on its position. To find neighbors within radius r,
    we only need to check the 9 cells surrounding the agent's cell (in 2D).

    This turns O(n²) all-pairs neighbor search into O(n * k) where k is the
    average number of agents per cell — typically constant for uniform distributions.
    """

    def __init__(self, cell_size: float):
        # Cell size must be >= the largest interaction radius.
        # If it's smaller, neighbors in non-adjacent cells get missed.
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[int]] = {}

    def clear(self) -> None:
        """Wipe the grid. Called every frame because agents move."""
        self.cells.clear()

    def _key(self, pos: Vector2D) -> tuple[int, int]:
        """Hash a position to a grid cell. floor() ensures consistent bucketing
        for negative coordinates too."""
        return (int(math.floor(pos.x / self.cell_size)),
                int(math.floor(pos.y / self.cell_size)))

    def insert(self, idx: int, pos: Vector2D) -> None:
        key = self._key(pos)
        if key not in self.cells:
            self.cells[key] = []
        self.cells[key].append(idx)

    def query_neighbors(self, pos: Vector2D, radius: float) -> list[int]:
        """Return indices of all agents within `radius` of `pos`.

        We check the 9 surrounding cells (3x3 grid centered on pos's cell).
        This guarantees we find all neighbors as long as cell_size >= radius.
        """
        cx, cy = self._key(pos)
        result = []
        # Check 3x3 neighborhood of cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell_key = (cx + dx, cy + dy)
                if cell_key in self.cells:
                    for idx in self.cells[cell_key]:
                        result.append(idx)
        return result


# ============================================================================
# Boid — an individual swarm agent
# ============================================================================

@dataclass
class Boid:
    """A single agent in the swarm. Has position, velocity, and accumulated
    acceleration (reset each frame). The max_speed and max_force constraints
    model physical limitations of real robots — motors have torque limits,
    and there's a top speed."""
    position: Vector2D
    velocity: Vector2D
    acceleration: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    max_speed: float = 4.0
    max_force: float = 0.3  # Maximum steering force per frame

    def apply_force(self, force: Vector2D) -> None:
        """Accumulate a steering force. In a real system this maps to motor commands."""
        self.acceleration = self.acceleration + force

    def update(self, dt: float = 1.0) -> None:
        """Euler integration: velocity += acceleration, position += velocity.
        Then reset acceleration for next frame.

        Euler integration is sufficient here because:
        1. Our timestep is fixed and small
        2. We're not simulating precise physics (no energy conservation needed)
        3. The steering forces are continuously recomputed, self-correcting errors
        """
        self.velocity = self.velocity + self.acceleration * dt
        self.velocity = self.velocity.limit(self.max_speed)
        self.position = self.position + self.velocity * dt
        # Reset acceleration — forces are recomputed each frame
        self.acceleration = Vector2D(0, 0)


# ============================================================================
# SwarmSimulation — the main simulation engine
# ============================================================================

class SwarmSimulation:
    """Orchestrates the swarm: manages agents, obstacles, spatial indexing,
    force computation, and metrics collection.

    The simulation loop each frame:
    1. Rebuild spatial hash (agents moved since last frame)
    2. For each agent, compute all steering forces using local neighbor info
    3. Update all agent positions via Euler integration
    4. Handle boundary conditions (wrapping)
    5. Collect metrics
    """

    def __init__(
        self,
        num_boids: int = 50,
        width: float = 200.0,
        height: float = 200.0,
        # Interaction radii — how far each rule "sees"
        separation_radius: float = 15.0,
        alignment_radius: float = 30.0,
        cohesion_radius: float = 40.0,
        # Force weights — the main tuning knobs
        separation_weight: float = 2.0,    # Highest: collision avoidance is priority #1
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        obstacle_weight: float = 3.0,      # Very high: hitting obstacles is catastrophic
        goal_weight: float = 0.5,          # Low: gentle pull, doesn't override flocking
    ):
        self.width = width
        self.height = height
        self.separation_radius = separation_radius
        self.alignment_radius = alignment_radius
        self.cohesion_radius = cohesion_radius
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.obstacle_weight = obstacle_weight
        self.goal_weight = goal_weight

        # Cell size = max interaction radius so spatial hash catches all neighbors
        max_radius = max(separation_radius, alignment_radius, cohesion_radius)
        self.spatial_hash = SpatialHash(cell_size=max_radius)

        self.obstacles: list[Obstacle] = []
        self.goal: Optional[Vector2D] = None

        # Initialize boids with random positions and velocities
        self.boids: list[Boid] = []
        for _ in range(num_boids):
            pos = Vector2D(
                random.uniform(0, width),
                random.uniform(0, height)
            )
            # Random initial velocity — magnitude ~ 2.0, random direction
            angle = random.uniform(0, 2 * math.pi)
            vel = Vector2D(math.cos(angle) * 2.0, math.sin(angle) * 2.0)
            self.boids.append(Boid(position=pos, velocity=vel))

        # Metrics history for analysis
        self.metrics_history: list[dict[str, float]] = []

    def add_obstacle(self, x: float, y: float, radius: float) -> None:
        self.obstacles.append(Obstacle(Vector2D(x, y), radius))

    def set_goal(self, x: float, y: float) -> None:
        self.goal = Vector2D(x, y)

    # --- The Three Flocking Rules ---

    def _separation(self, boid: Boid, neighbor_indices: list[int], boid_idx: int) -> Vector2D:
        """Separation: steer AWAY from neighbors that are too close.

        The force is inversely proportional to distance squared — this models
        the urgency of collision avoidance. A neighbor at distance 1 exerts
        100x more repulsion than one at distance 10.

        Why squared? Linear falloff is too gentle — agents would clip through
        each other before the repulsion force overcomes their momentum.
        """
        steer = Vector2D(0, 0)
        count = 0

        for j in neighbor_indices:
            if j == boid_idx:
                continue
            other = self.boids[j]
            dist = boid.position.distance_to(other.position)
            if 0 < dist < self.separation_radius:
                # Vector pointing AWAY from neighbor, scaled by 1/dist
                diff = boid.position - other.position
                diff = diff.normalize() / dist  # Closer = stronger repulsion
                steer = steer + diff
                count += 1

        if count > 0:
            steer = steer / count
            # Desired velocity = away from neighbors at max speed
            if steer.magnitude() > 0:
                steer = steer.normalize() * boid.max_speed - boid.velocity
                steer = steer.limit(boid.max_force)
        return steer

    def _alignment(self, boid: Boid, neighbor_indices: list[int], boid_idx: int) -> Vector2D:
        """Alignment: steer toward the average heading of neighbors.

        This is what creates the "flock moves as one" behavior. Without it,
        you get a cohesive cluster that tumbles randomly instead of flowing
        in a direction.

        We compute the average velocity of neighbors, then steer toward it.
        The steering force is the difference between desired and current velocity,
        clamped to max_force — this means alignment happens gradually, not instantly.
        """
        avg_velocity = Vector2D(0, 0)
        count = 0

        for j in neighbor_indices:
            if j == boid_idx:
                continue
            other = self.boids[j]
            dist = boid.position.distance_to(other.position)
            if 0 < dist < self.alignment_radius:
                avg_velocity = avg_velocity + other.velocity
                count += 1

        if count > 0:
            avg_velocity = avg_velocity / count
            # Steer toward average velocity
            avg_velocity = avg_velocity.normalize() * boid.max_speed
            steer = avg_velocity - boid.velocity
            steer = steer.limit(boid.max_force)
            return steer
        return Vector2D(0, 0)

    def _cohesion(self, boid: Boid, neighbor_indices: list[int], boid_idx: int) -> Vector2D:
        """Cohesion: steer toward the center of mass of neighbors.

        This counters separation's scattering effect. Without cohesion, the swarm
        would fly apart. With too much cohesion, agents pile up in the center.

        The force is a "seek" behavior: desired velocity points at the centroid,
        steering force = desired - current velocity.
        """
        center = Vector2D(0, 0)
        count = 0

        for j in neighbor_indices:
            if j == boid_idx:
                continue
            other = self.boids[j]
            dist = boid.position.distance_to(other.position)
            if 0 < dist < self.cohesion_radius:
                center = center + other.position
                count += 1

        if count > 0:
            center = center / count
            # Seek toward center of mass
            desired = center - boid.position
            if desired.magnitude() > 0:
                desired = desired.normalize() * boid.max_speed
                steer = desired - boid.velocity
                steer = steer.limit(boid.max_force)
                return steer
        return Vector2D(0, 0)

    def _obstacle_avoidance(self, boid: Boid) -> Vector2D:
        """Steer away from obstacles. The force increases sharply at close range
        (inverse square) to guarantee avoidance even at high speed.

        We check all obstacles — in a real system with many obstacles, you'd
        put these in a spatial hash too. For our simulation with a handful
        of obstacles, brute force is fine.
        """
        steer = Vector2D(0, 0)
        detection_range = 30.0  # How far ahead the agent "sees" obstacles

        for obs in self.obstacles:
            dist = boid.position.distance_to(obs.position) - obs.radius
            if 0 < dist < detection_range:
                # Repulsion vector: away from obstacle center
                away = boid.position - obs.position
                # Inverse square: very strong when close, gentle when far
                force = away.normalize() / max(dist * dist, 0.01)
                steer = steer + force

        if steer.magnitude() > 0:
            steer = steer.normalize() * boid.max_speed - boid.velocity
            steer = steer.limit(boid.max_force * 2.0)  # Allow extra force for obstacle avoidance
        return steer

    def _goal_seeking(self, boid: Boid) -> Vector2D:
        """Gentle pull toward the goal position. Uses "arrival" behavior —
        the agent slows down as it approaches the goal to prevent oscillation.

        The slowing radius means agents within 50 units start decelerating,
        preventing the whole swarm from overshooting and circling the goal.
        """
        if self.goal is None:
            return Vector2D(0, 0)

        desired = self.goal - boid.position
        dist = desired.magnitude()

        if dist < 1e-10:
            return Vector2D(0, 0)

        # Arrival behavior: scale speed by distance when close
        slowing_radius = 50.0
        if dist < slowing_radius:
            speed = boid.max_speed * (dist / slowing_radius)
        else:
            speed = boid.max_speed

        desired = desired.normalize() * speed
        steer = desired - boid.velocity
        steer = steer.limit(boid.max_force)
        return steer

    def _wrap_boundaries(self, boid: Boid) -> None:
        """Toroidal wrapping: agents leaving one side appear on the opposite side.

        Alternative: bouncing (reflect velocity at walls). Wrapping is preferred
        for swarm research because it eliminates wall-following artifacts where
        agents cluster at boundaries.
        """
        if boid.position.x < 0:
            boid.position.x += self.width
        elif boid.position.x > self.width:
            boid.position.x -= self.width
        if boid.position.y < 0:
            boid.position.y += self.height
        elif boid.position.y > self.height:
            boid.position.y -= self.height

    def compute_metrics(self) -> dict[str, float]:
        """Measure swarm quality. These three metrics tell you if your weights
        are well-tuned:

        - avg_distance_to_centroid: Lower = tighter swarm. Too low = collisions.
        - avg_velocity_alignment: Higher = more coherent movement (1.0 = perfect).
          Computed as average cosine similarity between each agent's velocity and
          the swarm's mean velocity.
        - min_neighbor_distance: If this hits 0, you have collisions. Should stay
          above some minimum safe distance.
        """
        if not self.boids:
            return {"avg_distance_to_centroid": 0, "avg_velocity_alignment": 0, "min_neighbor_distance": 0}

        n = len(self.boids)

        # Centroid
        cx = sum(b.position.x for b in self.boids) / n
        cy = sum(b.position.y for b in self.boids) / n
        centroid = Vector2D(cx, cy)

        # Average distance to centroid (cohesion metric)
        avg_dist = sum(b.position.distance_to(centroid) for b in self.boids) / n

        # Average velocity alignment (cosine similarity with mean velocity)
        mean_vel = Vector2D(
            sum(b.velocity.x for b in self.boids) / n,
            sum(b.velocity.y for b in self.boids) / n
        )
        mean_mag = mean_vel.magnitude()

        if mean_mag > 1e-10:
            alignment_sum = 0.0
            for b in self.boids:
                b_mag = b.velocity.magnitude()
                if b_mag > 1e-10:
                    cos_sim = b.velocity.dot(mean_vel) / (b_mag * mean_mag)
                    alignment_sum += cos_sim
            avg_alignment = alignment_sum / n
        else:
            avg_alignment = 0.0

        # Minimum inter-agent distance (collision indicator)
        min_dist = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                d = self.boids[i].position.distance_to(self.boids[j].position)
                if d < min_dist:
                    min_dist = d

        return {
            "avg_distance_to_centroid": avg_dist,
            "avg_velocity_alignment": avg_alignment,
            "min_neighbor_distance": min_dist if min_dist != float("inf") else 0.0,
        }

    def step(self, dt: float = 1.0) -> dict[str, float]:
        """Advance the simulation by one timestep.

        The order matters:
        1. Rebuild spatial hash (positions changed last frame)
        2. Compute ALL forces for ALL agents (read-only pass over current state)
        3. Apply forces and update positions (write pass)

        Computing all forces before updating any positions is critical —
        otherwise early-updated agents see a partially-updated world,
        creating subtle asymmetries and instabilities.
        """
        # 1. Rebuild spatial hash
        self.spatial_hash.clear()
        for i, boid in enumerate(self.boids):
            self.spatial_hash.insert(i, boid.position)

        # 2. Compute forces for all boids
        forces: list[Vector2D] = []
        for i, boid in enumerate(self.boids):
            # Get candidate neighbors from spatial hash (O(1) average per cell)
            neighbor_indices = self.spatial_hash.query_neighbors(
                boid.position, max(self.separation_radius, self.alignment_radius, self.cohesion_radius)
            )

            # Compute the three flocking forces
            sep = self._separation(boid, neighbor_indices, i) * self.separation_weight
            ali = self._alignment(boid, neighbor_indices, i) * self.alignment_weight
            coh = self._cohesion(boid, neighbor_indices, i) * self.cohesion_weight

            # Additional forces
            obs = self._obstacle_avoidance(boid) * self.obstacle_weight
            goal = self._goal_seeking(boid) * self.goal_weight

            total_force = sep + ali + coh + obs + goal
            forces.append(total_force)

        # 3. Apply forces and update positions
        for i, boid in enumerate(self.boids):
            boid.apply_force(forces[i])
            boid.update(dt)
            self._wrap_boundaries(boid)

        # 4. Collect metrics
        metrics = self.compute_metrics()
        self.metrics_history.append(metrics)
        return metrics

    def run(self, num_steps: int = 100, dt: float = 1.0) -> list[dict[str, float]]:
        """Run the simulation for multiple steps, returning metrics for each."""
        for _ in range(num_steps):
            self.step(dt)
        return self.metrics_history

    def get_state_snapshot(self) -> list[dict]:
        """Return current positions and velocities for visualization/debugging."""
        return [
            {
                "id": i,
                "x": b.position.x,
                "y": b.position.y,
                "vx": b.velocity.x,
                "vy": b.velocity.y,
                "speed": b.velocity.magnitude(),
            }
            for i, b in enumerate(self.boids)
        ]


# ============================================================================
# Demonstration
# ============================================================================

def print_swarm_state(sim: SwarmSimulation, step_num: int, metrics: dict) -> None:
    """Print a compact summary of swarm state for one timestep."""
    snapshot = sim.get_state_snapshot()
    speeds = [s["speed"] for s in snapshot]
    avg_speed = sum(speeds) / len(speeds)

    print(f"\n--- Step {step_num:3d} ---")
    print(f"  Cohesion (avg dist to centroid): {metrics['avg_distance_to_centroid']:.2f}")
    print(f"  Alignment (velocity correlation): {metrics['avg_velocity_alignment']:.4f}")
    print(f"  Min neighbor distance:            {metrics['min_neighbor_distance']:.2f}")
    print(f"  Avg speed:                        {avg_speed:.2f}")


def render_ascii(sim: SwarmSimulation, grid_w: int = 60, grid_h: int = 25) -> str:
    """Render the swarm as ASCII art. Obstacles are 'O', boids are '*', goal is 'G'."""
    grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]

    # Draw obstacles
    for obs in sim.obstacles:
        gx = int(obs.position.x / sim.width * (grid_w - 1))
        gy = int(obs.position.y / sim.height * (grid_h - 1))
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            grid[gy][gx] = "O"

    # Draw goal
    if sim.goal:
        gx = int(sim.goal.x / sim.width * (grid_w - 1))
        gy = int(sim.goal.y / sim.height * (grid_h - 1))
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            grid[gy][gx] = "G"

    # Draw boids
    for boid in sim.boids:
        gx = int(boid.position.x / sim.width * (grid_w - 1))
        gy = int(boid.position.y / sim.height * (grid_h - 1))
        gx = max(0, min(grid_w - 1, gx))
        gy = max(0, min(grid_h - 1, gy))
        grid[gy][gx] = "*"

    border = "+" + "-" * grid_w + "+"
    lines = [border]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append(border)
    return "\n".join(lines)


if __name__ == "__main__":
    random.seed(42)  # Reproducible results

    print("=" * 65)
    print("  SWARM BEHAVIOR SIMULATION")
    print("  Reynolds' Boids with spatial hashing, obstacles, and goals")
    print("=" * 65)

    # --- Scenario 1: Basic flocking (no obstacles, no goal) ---
    print("\n\n>>> SCENARIO 1: Basic Flocking (30 agents)")
    print("    Watching separation + alignment + cohesion self-organize...\n")

    sim1 = SwarmSimulation(num_boids=30, width=100, height=100)

    # Show initial state
    print("Initial state (random positions and velocities):")
    print(render_ascii(sim1, grid_w=50, grid_h=20))

    # Run and show evolution
    for step in range(1, 51):
        metrics = sim1.step()
        if step in (1, 10, 25, 50):
            print_swarm_state(sim1, step, metrics)

    print(f"\nAfter 50 steps:")
    print(render_ascii(sim1, grid_w=50, grid_h=20))

    # --- Scenario 2: Obstacle avoidance ---
    print("\n\n>>> SCENARIO 2: Obstacle Avoidance (40 agents, 2 obstacles)")
    print("    Swarm splits around obstacles and re-merges...\n")

    sim2 = SwarmSimulation(num_boids=40, width=150, height=150)
    sim2.add_obstacle(75, 60, 15)   # Center-ish obstacle
    sim2.add_obstacle(50, 100, 10)  # Left obstacle

    # Run for a bit to let flock form, then show
    for _ in range(30):
        sim2.step()

    print("After 30 steps with obstacles (O = obstacle):")
    print(render_ascii(sim2, grid_w=50, grid_h=20))

    metrics = sim2.compute_metrics()
    print(f"  Cohesion: {metrics['avg_distance_to_centroid']:.2f}")
    print(f"  Alignment: {metrics['avg_velocity_alignment']:.4f}")

    # --- Scenario 3: Goal seeking ---
    print("\n\n>>> SCENARIO 3: Goal Seeking (50 agents, goal at center)")
    print("    Swarm converges toward goal while maintaining flock structure...\n")

    sim3 = SwarmSimulation(num_boids=50, width=200, height=200, goal_weight=0.8)
    sim3.set_goal(100, 100)
    sim3.add_obstacle(80, 70, 12)

    print("Initial state (goal = G, obstacle = O):")
    print(render_ascii(sim3))

    # Run in phases
    for phase, steps in enumerate([(20, "forming"), (20, "converging"), (20, "settled")], 1):
        for _ in range(steps[0]):
            metrics = sim3.step()
        print(f"\nPhase {phase} ({steps[1]}) — after {phase * 20} steps:")
        print(render_ascii(sim3))
        print(f"  Cohesion: {metrics['avg_distance_to_centroid']:.2f}")
        print(f"  Alignment: {metrics['avg_velocity_alignment']:.4f}")
        print(f"  Min distance: {metrics['min_neighbor_distance']:.2f}")

    # --- Metrics summary ---
    print("\n\n>>> METRICS EVOLUTION (Scenario 3)")
    print("-" * 55)
    history = sim3.metrics_history
    for i, m in enumerate(history):
        if i % 10 == 0 or i == len(history) - 1:
            print(f"  Step {i:3d}: cohesion={m['avg_distance_to_centroid']:6.2f}  "
                  f"alignment={m['avg_velocity_alignment']:.4f}  "
                  f"min_dist={m['min_neighbor_distance']:.2f}")

    # --- Spatial hash performance demonstration ---
    print("\n\n>>> SPATIAL HASH: Performance Check")
    large_sim = SwarmSimulation(num_boids=200, width=500, height=500)
    import time
    start = time.time()
    for _ in range(50):
        large_sim.step()
    elapsed = time.time() - start
    print(f"  200 agents x 50 steps = {200*50:,} agent-updates in {elapsed:.3f}s")
    print(f"  That's {200*50/elapsed:,.0f} agent-updates/second")
    print(f"  Without spatial hashing, neighbor search alone would be O(200²×50) = {200*200*50:,} comparisons")

    print("\n" + "=" * 65)
    print("  Simulation complete. Key takeaway: complex flocking behavior")
    print("  emerges from three simple local rules — no central controller.")
    print("=" * 65)
