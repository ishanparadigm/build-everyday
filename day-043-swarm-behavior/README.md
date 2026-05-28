# Day 043: Swarm Behavior Simulation

## Overview

Build a multi-agent swarm simulation where dozens of simple robots exhibit complex emergent behavior through local-only interaction rules. Each agent follows three basic rules — separation, alignment, and cohesion — yet the swarm self-organizes into flocking patterns, obstacle avoidance, and goal-seeking behavior without any central controller.

This matters because swarm robotics is how we'll coordinate drone delivery fleets, search-and-rescue teams, and warehouse robots at scale. No single point of failure, no communication bottleneck — just local rules producing global intelligence.

## Core Concepts

### Boids Algorithm (Reynolds, 1987)

Craig Reynolds showed that realistic flocking emerges from three steering behaviors applied to each agent independently:

**1. Separation** — Steer away from nearby neighbors to avoid collisions.

Each agent looks at neighbors within a "separation radius" r_s. For each neighbor j within r_s, compute a repulsion vector pointing away from j, weighted inversely by distance:

```
F_sep(i) = Σ (pos_i - pos_j) / |pos_i - pos_j|²    for all j where |pos_i - pos_j| < r_s
```

The 1/d² weighting means closer neighbors exert much stronger repulsion — this prevents pile-ups.

**2. Alignment** — Steer toward the average heading of nearby neighbors.

Within an "alignment radius" r_a, compute the mean velocity of all neighbors and steer toward it:

```
F_align(i) = (1/N) Σ vel_j - vel_i    for all j where |pos_i - pos_j| < r_a
```

This is what makes the flock move as a coherent unit rather than a random cloud.

**3. Cohesion** — Steer toward the average position (center of mass) of nearby neighbors.

Within a "cohesion radius" r_c, compute the centroid and steer toward it:

```
F_coh(i) = (1/N) Σ pos_j - pos_i    for all j where |pos_i - pos_j| < r_c
```

Without cohesion, separation would scatter the swarm indefinitely.

### The Key Insight: Emergent Behavior

None of these rules reference global state. No agent knows where the whole swarm is heading. Yet the combination produces:
- **Flocking**: agents cluster and move together
- **Splitting and merging**: groups dynamically form and dissolve around obstacles
- **Lane formation**: in bidirectional flows, agents self-organize into lanes

This is a fundamental principle in robotics and distributed systems: **complex global behavior from simple local rules**.

### Force Combination and Tuning

The final steering force is a weighted sum:

```
F_total = w_sep * F_sep + w_align * F_align + w_coh * F_coh + w_obs * F_obstacle + w_goal * F_goal
```

Weight tuning is where the art lives:
- High w_sep → scattered, cautious swarm
- High w_align → rigid formation, slow to turn
- High w_coh → tight cluster, collision-prone
- The sweet spot depends on your application (delivery drones want different behavior than search robots)

### Spatial Hashing for Neighbor Lookup

Naive neighbor search is O(n²) per step — checking every agent against every other. With 100+ agents running at 60 FPS, this kills performance.

**Spatial hashing** divides the world into grid cells. Each agent hashes its position to a cell key. To find neighbors within radius r, you only check agents in the 9 (2D) surrounding cells. This gives O(n) average-case performance — a critical optimization for real swarm systems.

```
cell_x = floor(pos_x / cell_size)
cell_y = floor(pos_y / cell_size)
key = (cell_x, cell_y)
```

Cell size should be >= the largest interaction radius so all potential neighbors are in adjacent cells.

### Obstacle Avoidance in Swarms

Each agent casts a "look-ahead" vector in its velocity direction. If an obstacle is within detection range, compute a repulsion force perpendicular to the velocity (steering away from the obstacle surface). This is independent of the flocking forces and simply adds to F_total.

## Step-by-Step Breakdown

1. **Define the Agent (Boid) class**: Position, velocity, acceleration vectors. Max speed and max force constraints to keep motion realistic — without max speed, alignment forces would accelerate agents infinitely. Without max force, steering would be instantaneous (unrealistic for physical robots).

2. **Implement spatial hashing**: Build a grid-based spatial index that's rebuilt each frame. This is essential because without it, neighbor queries dominate runtime and the simulation can't scale past ~50 agents.

3. **Implement the three flocking rules**: Each returns a steering force vector. Normalize and weight them. The separation force must be computed first and given highest priority — collision avoidance is non-negotiable.

4. **Add obstacle avoidance**: Place circular obstacles in the world. Agents detect them via distance checks and steer away with a force that increases sharply at close range.

5. **Add goal-seeking**: A target point that attracts all agents. The force is a simple vector toward the goal, scaled down so it doesn't override flocking behavior. This simulates real scenarios like "all drones converge on this area."

6. **Simulation loop**: Each timestep: rebuild spatial hash, compute forces for all agents, update velocities and positions, wrap or bounce at boundaries. Use Euler integration (sufficient for this simulation fidelity).

7. **Metrics collection**: Track swarm cohesion (average distance to centroid), alignment (average velocity correlation), and separation (minimum inter-agent distance). These metrics tell you if your weights are tuned correctly.

## Learning Objectives

- Understand emergent behavior from local interaction rules
- Implement the Boids flocking algorithm with separation, alignment, and cohesion
- Build a spatial hash for efficient O(n) neighbor queries
- Tune multi-objective force weights for desired swarm behavior
- Measure swarm quality metrics (cohesion, alignment, collision avoidance)
- Connect to real applications: drone swarms, warehouse robots, search-and-rescue

## Going Deeper

- **Reynolds steering behaviors** extend well beyond flocking: pursuit, evasion, path following, leader following, and unaligned collision avoidance all compose with the same force-combination framework.
- **Predator-prey dynamics**: Add "predator" agents that scatter the flock. The flock exhibits realistic escape behavior without any explicit escape programming.
- **Communication-limited swarms**: What happens when agents can only sense neighbors within a very small radius? The swarm fragments. This models real RF-limited robot teams.
- **Formation control**: Instead of free-form flocking, assign agents to slots in a formation (V-shape, grid, circle). The challenge becomes balancing formation-keeping with obstacle avoidance.
- **Real-world deployment**: Production swarm systems (e.g., Crazyswarm for Crazyflie drones) use these exact algorithms with added constraints for 3D, communication latency, and battery life.
