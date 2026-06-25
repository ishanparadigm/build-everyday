# Day 85: Occupancy Grid Mapping

## What You're Building

An **occupancy grid mapping** system that takes noisy range sensor readings from a mobile robot and builds a probabilistic 2D map of the environment. This is the foundational representation used by virtually every autonomous robot — from warehouse AGVs to self-driving cars — to understand what space is free, what's occupied, and what's unknown.

In the real world, a robot exploring a building doesn't get a neat blueprint. It gets thousands of noisy laser or sonar readings from different poses. Occupancy grid mapping fuses all of these into a single coherent map where each cell holds a probability of being occupied. This map then drives downstream tasks: path planning (days 28, 34), obstacle avoidance (day 17), and full SLAM (day 44).

## Core Concepts

### 1. The Occupancy Grid Representation

The environment is discretized into a 2D grid of cells, each storing the probability `P(occupied)`:

- `P = 0.0` → definitely free
- `P = 0.5` → unknown (no information)
- `P = 1.0` → definitely occupied

A grid with resolution `r` (meters per cell) covering a `W × H` meter area has `(W/r) × (H/r)` cells. Finer resolution = more detail but more memory and computation.

**Key tradeoff**: Resolution vs. computational cost. A 100m × 100m environment at 5cm resolution = 4 million cells. At 10cm = 1 million cells. Production systems typically use 5-10cm for indoor and 10-25cm for outdoor.

### 2. Log-Odds Representation

Working directly with probabilities is numerically unstable — repeated multiplication of small numbers causes underflow. Instead, we use **log-odds** (also called logit):

```
l(x) = log(P(x) / (1 - P(x)))
```

The inverse mapping (log-odds back to probability):

```
P(x) = 1 - 1 / (1 + exp(l(x)))
```

Why this is brilliant:
- **Unknown → 0**: `P = 0.5` maps to `l = log(0.5/0.5) = 0`. Initialization is just zeroing the array.
- **Additive updates**: Instead of multiplying probabilities (Bayes' rule), we simply add log-odds. This is numerically stable and computationally cheap.
- **Symmetric**: Positive log-odds = likely occupied, negative = likely free.

The update rule becomes:

```
l(cell | z₁:t) = l(cell | z₁:t-1) + l(cell | zₜ) - l₀
```

where `l₀ = log(0.5/0.5) = 0` is the prior (which vanishes, making the update just addition).

### 3. The Inverse Sensor Model

Given a single range reading `z` from pose `(x, y, θ)`, which cells should be updated and how?

For a ray cast from the sensor at angle `α`:
- **Cells along the ray before the endpoint**: These are **free** — the beam passed through them. Apply a negative log-odds update (e.g., `l_free = -0.4`).
- **Cells at/near the endpoint**: These are **occupied** — the beam hit something. Apply a positive log-odds update (e.g., `l_occ = 0.85`).
- **Cells beyond the endpoint**: No information — leave unchanged.

The asymmetry in magnitudes (`|l_occ| > |l_free|`) is intentional: we want occupied cells to "stick" more readily because missing an obstacle is more dangerous than being overly cautious. A single hit should outweigh a single pass-through.

### 4. Bresenham's Line Algorithm for Ray Casting

To find which cells a ray passes through, we use **Bresenham's line algorithm** — an efficient integer-arithmetic method that traces a line between two grid cells without floating-point operations.

Given start cell `(x₀, y₀)` and end cell `(x₁, y₁)`:
1. Compute `dx = |x₁ - x₀|`, `dy = |y₁ - y₀|`
2. Step along the major axis (whichever of dx, dy is larger)
3. Accumulate the minor axis error; when it exceeds 0.5, step in the minor axis

This gives us the exact set of cells the ray traverses — critical for correctly marking free space.

### 5. Multi-Beam Sensor Updates

A real lidar produces hundreds of beams per scan (e.g., 360 beams for 1° resolution). Each beam is processed independently through the inverse sensor model. The log-odds representation makes this efficient — each beam just adds its contribution to affected cells.

**Important subtlety**: Beams from the same scan should all use the same robot pose. If the robot moves during a scan (common at high speeds), you need motion compensation — but for this challenge, we assume instantaneous scans.

### 6. Clamping Log-Odds

Without bounds, log-odds can grow unboundedly large. A cell observed as occupied 1000 times would need 1000 free observations to become uncertain again. This makes the map too "rigid" — it can't adapt to changes (e.g., a door opening).

Solution: clamp log-odds to `[l_min, l_max]` (e.g., `[-5, 5]`). This bounds the "confidence" and allows the map to eventually update if the environment changes.

## Step-by-Step Breakdown

### Step 1: Initialize the Grid
Create a 2D numpy array of zeros (log-odds = 0 → P = 0.5 → unknown). Define the grid resolution, world bounds, and coordinate transforms between world and grid frames.

### Step 2: Implement Coordinate Transforms
- `world_to_grid(wx, wy)`: Convert world coordinates to grid indices
- `grid_to_world(gx, gy)`: Convert grid indices back to world coordinates

These must handle the offset (grid origin isn't necessarily at world origin) and scaling (by resolution).

### Step 3: Implement Bresenham's Line Algorithm
Given two grid cells, return all cells along the line between them. This is the workhorse for ray casting — it determines which cells each sensor beam passes through.

### Step 4: Build the Inverse Sensor Model
Given a robot pose and a single range measurement + angle, compute:
- The endpoint in world coordinates: `(x + z·cos(θ+α), y + z·sin(θ+α))`
- Convert to grid coordinates
- Use Bresenham to get cells along the ray
- Mark traversed cells as free, endpoint cell as occupied

### Step 5: Process Full Scans
For each lidar scan (array of ranges at known angles), iterate through all beams and apply the inverse sensor model. Clamp log-odds after each full scan.

### Step 6: Simulate a Robot Exploring
Create a simple environment with walls and obstacles. Simulate a robot following a predefined path, taking lidar scans at each pose. Feed all scans into the mapping system.

### Step 7: Visualize the Map
Convert log-odds to probabilities for display. Use a colormap where white = free, black = occupied, gray = unknown. Overlay the robot's trajectory.

## Learning Objectives

- Understand probabilistic spatial representations and why they dominate robotics
- Master log-odds as a numerically stable alternative to raw probability updates
- Implement Bresenham's algorithm for efficient grid-based ray casting
- Build an inverse sensor model that correctly handles free/occupied/unknown space
- See how individual noisy measurements fuse into a coherent, confident map
- Connect to downstream systems: this map is what path planners (A*, RRT) operate on

## Going Deeper

- **Dynamic environments**: Use a recency-weighted decay so old observations fade, allowing the map to track changes (opening doors, moving furniture)
- **Multi-resolution grids**: Octrees (3D) or quadtrees (2D) save memory in large, sparse environments by using coarse cells in open areas and fine cells near obstacles
- **3D occupancy**: Extend to voxel grids or OctoMap for full 3D mapping — essential for drone navigation and manipulation
- **GPU acceleration**: Each ray is independent → embarrassingly parallel. CUDA implementations process millions of rays per second
- **Integration with SLAM**: In real systems, the robot's pose is uncertain too. SLAM (day 44) jointly estimates the map and the trajectory. Graph-based SLAM maintains multiple map hypotheses weighted by pose graph optimization
- **Semantic occupancy**: Modern systems label cells not just as occupied/free but with semantic classes (wall, furniture, person) using neural network predictions fused into the grid
