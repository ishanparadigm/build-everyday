# Day 70: Autonomous Drone Path Planning

## Overview

Build a 3D path planning system for an autonomous drone navigating through obstacle-filled environments. Unlike ground robots that plan in 2D, drones operate in full 3D space with additional constraints: altitude limits, battery life (energy-aware planning), wind effects, and no-fly zones. This challenge implements **3D RRT\*** (Rapidly-exploring Random Tree Star) with energy-aware cost functions — the same class of algorithms used in real drone delivery systems, search-and-rescue operations, and agricultural surveying.

**Why it matters:** Autonomous drones are deployed for package delivery (Wing, Amazon Prime Air), infrastructure inspection (power lines, bridges), precision agriculture, and disaster response. The path planner must find collision-free paths that minimize energy consumption while respecting airspace constraints — a problem that's fundamentally different from 2D robot navigation.

## Core Concepts

### 1. 3D Configuration Space

A drone's position is described by `(x, y, z)` in continuous 3D space. The configuration space (C-space) includes all possible positions, and obstacles create "forbidden" regions. Unlike 2D planning:

- Obstacles are 3D volumes (buildings, trees, terrain)
- The drone can fly over obstacles if altitude permits
- Altitude itself is constrained: `z_min ≤ z ≤ z_max`

The free configuration space `C_free = C_total - C_obstacle` is what we search through.

### 2. RRT* Algorithm (Optimal RRT)

Standard RRT builds a tree by randomly sampling points and connecting them to the nearest existing node. RRT* improves on this with two key additions:

**Near-neighbor rewiring:** After adding a new node, check all nodes within a radius `r` and see if routing through the new node gives a cheaper path. If so, rewire.

**Asymptotic optimality:** As the number of samples → ∞, RRT* converges to the optimal path. Standard RRT does not have this guarantee.

The radius shrinks as nodes grow:
```
r = min(γ * (log(n) / n)^(1/d), η)
```
where `γ` is a constant depending on the free space volume, `n` is the number of nodes, `d` is the dimension (3 for us), and `η` is the maximum step size.

### 3. Energy-Aware Cost Function

A drone's energy consumption is NOT proportional to Euclidean distance. Key factors:

- **Climbing costs more than descending:** Gaining altitude requires fighting gravity. Energy ∝ `m * g * Δz` for ascent.
- **Horizontal flight has drag:** Energy ∝ `distance * drag_coefficient`
- **Hovering costs energy:** Unlike ground robots, staying still still drains battery
- **Wind affects energy:** Flying into headwind costs more; tailwind helps

Our cost function:
```
cost(a, b) = horizontal_energy + vertical_energy + wind_penalty
           = d_horiz * C_drag + max(0, Δz) * C_climb + min(0, Δz) * C_descend + wind_component
```

where `C_climb > C_drag > C_descend` reflecting real physics.

### 4. No-Fly Zones and Geofencing

Real drones must respect airspace restrictions:
- **Static no-fly zones:** Airports, government buildings, national parks
- **Dynamic restrictions:** Temporary flight restrictions (TFRs)
- **Altitude bands:** Different rules at different altitudes

We model these as 3D volumes that the planner must avoid entirely.

### 5. Path Smoothing

RRT* paths have unnecessary zigzags. Post-processing smooths them:
- **Shortcutting:** Try connecting non-adjacent waypoints directly; keep if collision-free
- **B-spline smoothing:** Fit a smooth curve through waypoints for flyable trajectories
- **Minimum-snap trajectory:** The gold standard for quadrotors — minimize the 4th derivative of position for smooth, energy-efficient flight

## Step-by-Step Breakdown

### Step 1: Define the 3D Environment
Create a world with ground plane, 3D obstacles (buildings as rectangular prisms, trees as cylinders), no-fly zones, and altitude constraints. Each obstacle has position, dimensions, and a safety margin (buffer zone).

### Step 2: Implement Collision Detection
For each candidate path segment, check if it intersects any obstacle. We use line-segment vs. axis-aligned bounding box (AABB) intersection tests. For cylindrical obstacles, check line-segment vs. cylinder intersection. This is the performance bottleneck — keep it fast.

### Step 3: Build the RRT* Tree
Sample random 3D points in the free space, find nearest neighbors, extend the tree, check for collisions, and rewire for optimality. Use a KD-tree for efficient nearest-neighbor queries in 3D.

### Step 4: Energy-Aware Cost
Replace Euclidean distance with the energy cost function. This changes which paths RRT* considers "optimal" — it will prefer paths that avoid unnecessary climbing and exploit favorable wind.

### Step 5: Path Extraction and Smoothing
Once the goal is reached, trace back through the tree to get the raw path. Apply shortcutting to remove unnecessary waypoints, then smooth the result.

### Step 6: Visualization and Analysis
Display the 3D environment, obstacles, RRT* tree, raw path, and smoothed path. Show energy consumption breakdown along the path.

## Learning Objectives

- Implement sampling-based path planning in 3D (extending Day 34's RRT to RRT*)
- Design physics-informed cost functions beyond simple distance
- Handle complex 3D collision detection with multiple obstacle types
- Apply path smoothing for real-world flyable trajectories
- Understand the tradeoffs between planning time, path quality, and computational cost

## Going Deeper

- **Kinodynamic RRT*:** Incorporate drone dynamics (velocity, acceleration limits) directly into the planner so the tree only contains dynamically feasible motions
- **Informed RRT*:** Once an initial path is found, focus sampling in a prolate hyperspheroid to converge faster
- **Multi-drone planning:** Coordinate multiple drones with inter-drone collision avoidance
- **Replanning:** When new obstacles appear mid-flight, efficiently update the plan rather than restarting
- **Real-time constraints:** Production systems need paths in <100ms. Anytime algorithms return the best path found so far when time runs out
