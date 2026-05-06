# Day 034: RRT Path Planning (Rapidly-exploring Random Trees)

## Overview

Build a Rapidly-exploring Random Tree (RRT) path planner from scratch — the algorithm that self-driving cars, robotic arms, and drone systems use to find collision-free paths through complex environments. Unlike grid-based planners like A* (Day 028), RRT works in continuous space and scales to high-dimensional configuration spaces where discretization would be computationally impossible.

**Why it matters:** A 6-DOF robot arm has a 6-dimensional configuration space. Discretizing each dimension into just 100 bins creates 10^12 cells — impossible for grid search. RRT samples this space randomly and builds a tree incrementally, finding feasible paths in seconds. It's the backbone of motion planning in robotics, from warehouse robots to surgical systems.

## Core Concepts

### 1. Configuration Space vs. Workspace

The robot moves in **workspace** (physical 2D/3D space), but planning happens in **configuration space** (C-space) — the space of all possible robot states. For a point robot in 2D, C-space = workspace. For a robot arm, each joint angle is a dimension.

An obstacle in workspace maps to a **C-space obstacle** — the set of all configurations where the robot collides. The free space C_free is everything else. Our job: find a path through C_free from start to goal.

### 2. The RRT Algorithm

RRT builds a tree rooted at the start configuration by repeatedly:

1. **Sample** a random point q_rand in C-space
2. **Find nearest** node q_near in the tree (by distance metric)
3. **Steer** from q_near toward q_rand by a fixed step size δ, producing q_new
4. **Check** if the edge (q_near → q_new) is collision-free
5. **Add** q_new to the tree with q_near as its parent
6. **Check** if q_new is close enough to the goal

The key insight is **Voronoi bias**: nodes with larger Voronoi regions (i.e., nodes on the frontier of the tree) are more likely to be selected as q_near. This naturally drives exploration toward unexplored space without any explicit exploration heuristic.

**Mathematical guarantee:** RRT is *probabilistically complete* — given infinite samples, the probability of finding a path (if one exists) approaches 1. Formally: P(failure after n samples) → 0 as n → ∞. However, the path found is generally NOT optimal.

### 3. Goal Biasing

Pure random sampling can be slow to reach the goal. **Goal biasing** samples the goal location with some probability p_goal (typically 5-10%) instead of a random point. This creates a pull toward the goal while maintaining exploration.

Too much bias (p_goal > 0.2) causes the tree to repeatedly slam into obstacles near the goal. Too little, and convergence is slow. The sweet spot depends on the environment's obstacle density.

### 4. Collision Detection

For each candidate edge, we must verify it doesn't pass through obstacles. The standard approach: **discretize the edge** into small steps and check each point against all obstacles. For circular obstacles, check if point-to-center distance < radius. For polygonal obstacles, use point-in-polygon tests.

The step size for collision checking should be smaller than the thinnest obstacle — otherwise the path can "tunnel" through. A common choice: step = min(obstacle_dimension) / 2.

### 5. Path Extraction and Smoothing

Once the goal is reached, trace parent pointers back to the root to get the path. RRT paths are typically jagged due to random sampling. **Path smoothing** improves quality:

- **Shortcutting:** Pick two random non-adjacent nodes on the path. If the straight line between them is collision-free, replace the intermediate path with that line.
- Repeat until no more shortcuts are found or a time budget expires.

### 6. RRT* — The Optimal Variant

Standard RRT finds *a* path but not the *best* path. **RRT*** adds two operations:
- **Near-neighbor search:** Instead of just the nearest node, consider all nodes within radius r = γ(log(n)/n)^(1/d) where d is dimension
- **Rewiring:** If connecting q_new to a different nearby node gives a shorter total path, rewire the tree

RRT* is *asymptotically optimal* — the path cost converges to the true optimum as samples → ∞. The cost: O(n log n) per iteration vs O(n) for basic RRT.

## Step-by-Step Breakdown

### Step 1: Define the Environment
Create a 2D workspace with rectangular bounds and circular obstacles. Each obstacle has a center (x, y) and radius r. Store the start and goal positions.

### Step 2: Implement the Tree Data Structure
Each node stores: position (x, y), parent index, and cost-from-root. The tree is a list of nodes with parent pointers forming an implicit tree structure.

### Step 3: Random Sampling with Goal Bias
Generate uniform random points in the workspace bounds. With probability p_goal, return the goal position instead. This is where the exploration/exploitation tradeoff lives.

### Step 4: Nearest Neighbor Search
Find the tree node closest to the sample point using Euclidean distance. For large trees, a KD-tree accelerates this to O(log n) — but for our purposes, brute force O(n) is fine for trees under ~10K nodes.

### Step 5: Steering Function
Given q_near and q_rand, compute q_new by moving from q_near toward q_rand by at most step_size δ. If q_rand is closer than δ, just use q_rand. This is simply: q_new = q_near + min(δ, dist) * (q_rand - q_near) / dist.

### Step 6: Collision Checking
Discretize the segment from q_near to q_new into small steps. At each step, check if the point lies inside any obstacle (distance to center < radius). Also check bounds.

### Step 7: Path Extraction
When q_new is within goal_threshold of the goal, trace parent pointers from q_new back to root. Reverse to get start-to-goal path.

### Step 8: Path Smoothing
Iteratively try to shortcut: pick two random path nodes, check if direct connection is collision-free, replace the subpath if so. This dramatically reduces path length and jaggedness.

### Step 9: RRT* Extensions
Implement near-neighbor search and rewiring for asymptotically optimal paths. Compare path quality between RRT and RRT*.

## Learning Objectives

- Understand sampling-based motion planning and why it dominates high-dimensional spaces
- Implement RRT with proper collision detection and goal biasing
- Grasp probabilistic completeness vs. asymptotic optimality (RRT vs. RRT*)
- Build intuition for exploration/exploitation tradeoffs in path planning
- Apply path smoothing as a post-processing optimization
- Connect to Day 028's A* — understand when grid-based vs. sampling-based planning is appropriate

## Going Deeper

- **RRT-Connect:** Grow two trees (from start and goal) simultaneously; they meet in the middle. Much faster for narrow-passage problems.
- **Informed RRT*:** After finding an initial path, restrict sampling to an ellipsoidal region that could actually improve the path. Converges to optimal much faster.
- **Kinodynamic RRT:** Incorporate robot dynamics (velocity, acceleration limits) into the steering function. The tree explores state space (position + velocity), not just configuration space.
- **Narrow passages:** RRT struggles with narrow corridors between obstacles. Bridge sampling and Gaussian sampling address this by biasing samples toward obstacle boundaries.
- **Real-time replanning:** In dynamic environments, maintain the tree and prune/regrow as obstacles move. This is how autonomous vehicles handle moving traffic.
