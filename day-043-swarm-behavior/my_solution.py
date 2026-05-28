"""
Day 043: Swarm Behavior Simulation — Your Implementation

Implement Reynolds' Boids flocking algorithm. Each agent follows three local rules
(separation, alignment, cohesion) that produce emergent flocking behavior.

Key concepts to keep in mind:
- Forces are VECTORS — direction matters as much as magnitude
- Separation must be strongest to prevent collisions
- The spatial hash is what makes this scale beyond ~50 agents
- All forces are computed BEFORE any positions update (avoid read-write conflicts)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Vector2D:
    """2D vector for position, velocity, and force calculations."""
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
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / mag, self.y / mag)

    def limit(self, max_val: float) -> "Vector2D":
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


@dataclass
class Obstacle:
    position: Vector2D
    radius: float


class SpatialHash:
    """Grid-based spatial index for fast neighbor queries.

    Hint: The cell size should be >= the largest interaction radius.
    Hint: Use floor(pos / cell_size) to compute the cell key.
    Hint: Check 3x3 neighborhood of cells for neighbor queries.
    """

    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[int]] = {}

    def clear(self) -> None:
        """Wipe the grid for a new frame."""
        raise NotImplementedError("TODO: implement this")

    def _key(self, pos: Vector2D) -> tuple[int, int]:
        """Hash a position to a grid cell coordinate."""
        raise NotImplementedError("TODO: implement this")

    def insert(self, idx: int, pos: Vector2D) -> None:
        """Insert an agent index into the grid at its position."""
        raise NotImplementedError("TODO: implement this")

    def query_neighbors(self, pos: Vector2D, radius: float) -> list[int]:
        """Return indices of all agents in cells near `pos`.

        Hint: Check the 3x3 grid of cells centered on pos's cell.
        """
        raise NotImplementedError("TODO: implement this")


@dataclass
class Boid:
    """A single swarm agent with position, velocity, and acceleration.

    Hint: max_speed prevents infinite acceleration from alignment forces.
    Hint: max_force ensures steering is gradual (realistic for physical robots).
    """
    position: Vector2D
    velocity: Vector2D
    acceleration: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    max_speed: float = 4.0
    max_force: float = 0.3

    def apply_force(self, force: Vector2D) -> None:
        """Accumulate a steering force into acceleration."""
        raise NotImplementedError("TODO: implement this")

    def update(self, dt: float = 1.0) -> None:
        """Euler integration: update velocity and position, then reset acceleration.

        Hint: velocity += acceleration * dt, then limit to max_speed
        Hint: position += velocity * dt
        Hint: Reset acceleration to zero after updating
        """
        raise NotImplementedError("TODO: implement this")


class SwarmSimulation:
    """Main simulation engine for the Boids swarm.

    Hint: The simulation loop order matters:
    1. Rebuild spatial hash
    2. Compute ALL forces (read-only pass)
    3. Apply forces and update ALL positions (write pass)
    4. Handle boundaries
    """

    def __init__(
        self,
        num_boids: int = 50,
        width: float = 200.0,
        height: float = 200.0,
        separation_radius: float = 15.0,
        alignment_radius: float = 30.0,
        cohesion_radius: float = 40.0,
        separation_weight: float = 2.0,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        obstacle_weight: float = 3.0,
        goal_weight: float = 0.5,
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

        max_radius = max(separation_radius, alignment_radius, cohesion_radius)
        self.spatial_hash = SpatialHash(cell_size=max_radius)

        self.obstacles: list[Obstacle] = []
        self.goal: Optional[Vector2D] = None

        self.boids: list[Boid] = []
        for _ in range(num_boids):
            pos = Vector2D(random.uniform(0, width), random.uniform(0, height))
            angle = random.uniform(0, 2 * math.pi)
            vel = Vector2D(math.cos(angle) * 2.0, math.sin(angle) * 2.0)
            self.boids.append(Boid(position=pos, velocity=vel))

        self.metrics_history: list[dict[str, float]] = []

    def add_obstacle(self, x: float, y: float, radius: float) -> None:
        self.obstacles.append(Obstacle(Vector2D(x, y), radius))

    def set_goal(self, x: float, y: float) -> None:
        self.goal = Vector2D(x, y)

    def _separation(self, boid: Boid, neighbor_indices: list[int], boid_idx: int) -> Vector2D:
        """Compute separation force: steer AWAY from neighbors within separation_radius.

        Hint: For each neighbor within radius, compute a vector pointing AWAY from it
        Hint: Weight by 1/distance — closer neighbors push harder
        Hint: Average the repulsion vectors, then compute steering = desired - current velocity
        Hint: Limit the result to max_force
        """
        raise NotImplementedError("TODO: implement this")

    def _alignment(self, boid: Boid, neighbor_indices: list[int], boid_idx: int) -> Vector2D:
        """Compute alignment force: steer toward average heading of neighbors.

        Hint: Average the velocity vectors of neighbors within alignment_radius
        Hint: The desired velocity is that average, normalized and scaled to max_speed
        Hint: Steering = desired - current velocity, limited to max_force
        """
        raise NotImplementedError("TODO: implement this")

    def _cohesion(self, boid: Boid, neighbor_indices: list[int], boid_idx: int) -> Vector2D:
        """Compute cohesion force: steer toward center of mass of neighbors.

        Hint: Average the positions of neighbors within cohesion_radius
        Hint: The desired velocity points from current position toward that centroid
        Hint: Steering = desired - current velocity, limited to max_force
        """
        raise NotImplementedError("TODO: implement this")

    def _obstacle_avoidance(self, boid: Boid) -> Vector2D:
        """Compute obstacle avoidance force.

        Hint: For each obstacle, check distance minus obstacle radius
        Hint: If within detection range (~30 units), compute repulsion away from obstacle
        Hint: Use inverse square distance for sharp close-range avoidance
        Hint: Allow extra force (2x max_force) for obstacle avoidance
        """
        raise NotImplementedError("TODO: implement this")

    def _goal_seeking(self, boid: Boid) -> Vector2D:
        """Compute goal-seeking force with arrival behavior.

        Hint: Desired velocity = direction to goal * max_speed
        Hint: Within a slowing radius (~50 units), scale speed proportional to distance
        Hint: This prevents the swarm from overshooting the goal
        """
        raise NotImplementedError("TODO: implement this")

    def _wrap_boundaries(self, boid: Boid) -> None:
        """Toroidal wrapping: agents exiting one side appear on the opposite side.

        Hint: If x < 0, add width; if x > width, subtract width. Same for y.
        """
        raise NotImplementedError("TODO: implement this")

    def compute_metrics(self) -> dict[str, float]:
        """Compute swarm quality metrics.

        Returns dict with:
        - avg_distance_to_centroid: cohesion measure (lower = tighter)
        - avg_velocity_alignment: heading agreement (higher = more coherent, max 1.0)
        - min_neighbor_distance: collision indicator (0 = collision)

        Hint: Centroid = average of all positions
        Hint: Alignment = average cosine similarity of each velocity with mean velocity
        Hint: Min distance requires checking all pairs (O(n²) but only for metrics)
        """
        raise NotImplementedError("TODO: implement this")

    def step(self, dt: float = 1.0) -> dict[str, float]:
        """Advance simulation by one timestep.

        IMPORTANT: Compute ALL forces before updating ANY positions!
        This prevents early-updated agents from seeing partially-updated state.

        Steps:
        1. Rebuild spatial hash
        2. Compute forces for all boids (store in a list)
        3. Apply forces and update all boids
        4. Wrap boundaries
        5. Compute and store metrics
        """
        raise NotImplementedError("TODO: implement this")

    def run(self, num_steps: int = 100, dt: float = 1.0) -> list[dict[str, float]]:
        """Run the simulation for multiple steps, returning metrics for each."""
        raise NotImplementedError("TODO: implement this")

    def get_state_snapshot(self) -> list[dict]:
        """Return current positions and velocities for all boids."""
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


if __name__ == "__main__":
    random.seed(42)

    print("Testing your swarm implementation...")

    # Test 1: Basic simulation runs
    sim = SwarmSimulation(num_boids=20, width=100, height=100)
    metrics = sim.step()
    print(f"Step 1 metrics: {metrics}")

    # Test 2: Run for 50 steps, check metrics improve
    history = sim.run(num_steps=50)
    print(f"\nAfter 50 steps:")
    print(f"  Cohesion: {history[-1]['avg_distance_to_centroid']:.2f}")
    print(f"  Alignment: {history[-1]['avg_velocity_alignment']:.4f}")

    # Test 3: Obstacle avoidance
    sim2 = SwarmSimulation(num_boids=30, width=100, height=100)
    sim2.add_obstacle(50, 50, 10)
    for _ in range(30):
        sim2.step()
    # Check no agent is inside the obstacle
    for boid in sim2.boids:
        dist = boid.position.distance_to(sim2.obstacles[0].position)
        assert dist > sim2.obstacles[0].radius, f"Agent inside obstacle! dist={dist:.2f}"
    print("\nObstacle avoidance: PASS")

    # Test 4: Goal seeking
    sim3 = SwarmSimulation(num_boids=20, width=200, height=200, goal_weight=1.0)
    sim3.set_goal(100, 100)
    for _ in range(100):
        sim3.step()
    avg_dist_to_goal = sum(
        b.position.distance_to(sim3.goal) for b in sim3.boids
    ) / len(sim3.boids)
    print(f"Goal seeking — avg distance to goal after 100 steps: {avg_dist_to_goal:.2f}")

    print("\nAll basic tests passed!")
