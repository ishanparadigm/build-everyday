# Day 17: Obstacle Avoidance Algorithm

## What You're Building

A reactive obstacle avoidance system using the **Vector Field Histogram (VFH)** algorithm — one of the most widely used approaches in real mobile robots. Your robot will take noisy range sensor readings, build a polar histogram of obstacle density around it, find safe steering directions, and navigate toward a goal while avoiding collisions.

This matters because obstacle avoidance is the foundational capability that separates a remote-controlled toy from an autonomous robot. Every self-driving car, warehouse robot, and delivery drone runs some form of this algorithm. VFH specifically is popular because it handles noisy sensors gracefully and runs in real-time on limited hardware.

## Core Concepts

### 1. The Configuration Space Problem

A robot moving in a 2D world has three degrees of freedom: (x, y, theta). The space of all possible robot positions is called **configuration space** (C-space). Obstacles in the physical world map to regions in C-space that the robot cannot enter.

Naive approach: plan a complete path through C-space. Problem: this requires a full map, is computationally expensive, and breaks when the environment changes.

**Reactive approaches** like VFH instead make local steering decisions based on current sensor data. They trade global optimality for real-time responsiveness. In practice, you combine reactive avoidance with a higher-level planner — the planner says "go northeast," and the reactive layer figures out how to do that without hitting anything.

### 2. Range Sensors and the Polar Histogram

A typical mobile robot has range sensors (LIDAR, sonar, IR) arranged in a ring, each measuring distance to the nearest obstacle along its beam direction. Raw readings are noisy — a single bad reading shouldn't cause the robot to swerve.

VFH's key insight: **aggregate sensor readings into angular bins** to create a polar histogram. Each bin covers a sector of angles (e.g., 5 degrees) and accumulates an **obstacle density** value:

```
h_k = sum over all readings in sector k of: (c_i)^2 * d_i * (a - b * dist_i)
```

Where:
- `k` is the sector index
- `c_i` is a certainty value for each cell (how confident we are there's an obstacle)
- `d_i` is a distance-based weight
- `a` and `b` are constants ensuring that closer obstacles get higher weight
- `dist_i` is the distance to the obstacle

The simplification we'll use: for each sensor reading, compute `weight = a - b * distance`. Closer obstacles contribute more. Readings beyond a threshold contribute nothing.

### 3. Finding Free Sectors and Candidate Valleys

Once we have the histogram, we apply a **threshold** to classify each sector as blocked or free:

- `h_k > threshold` → sector k is blocked
- `h_k <= threshold` → sector k is free

Consecutive free sectors form **valleys** — candidate steering directions. We want wide valleys (the robot needs enough clearance) and we prefer valleys whose center is closest to the goal direction.

### 4. Steering Direction Selection

Given candidate valleys, the algorithm picks the best steering direction using a cost function:

```
cost = mu_1 * delta_target + mu_2 * delta_current + mu_3 * delta_prev
```

Where:
- `delta_target` = angular difference from goal direction
- `delta_current` = angular difference from current heading  
- `delta_prev` = angular difference from previous selected direction (for smoothness)
- `mu_1, mu_2, mu_3` are weights (typically mu_1 is largest — goal-seeking dominates)

This balances goal-seeking, momentum, and smooth steering. Without the smoothness term, the robot oscillates between valleys. Without the goal term, it wanders.

### 5. The Full Loop

Each timestep:
1. Read sensors → get distances at known angles
2. Build polar histogram from readings
3. Apply threshold → identify blocked/free sectors
4. Find valleys (runs of free sectors)
5. Pick best steering direction via cost function
6. Move robot in that direction
7. Repeat

## Step-by-Step Breakdown

### Step 1: Represent the Environment
Create a 2D grid world with obstacles (circles or polygons). Place a robot with a position, heading, and goal location. The robot has N simulated range sensors evenly spaced around it.

### Step 2: Simulate Range Sensors
For each sensor beam, cast a ray from the robot and find the nearest intersection with any obstacle. Add Gaussian noise to simulate real sensor imperfections. Clamp readings to a max range.

### Step 3: Build the Polar Histogram
Divide 360 degrees into sectors (e.g., 72 sectors of 5 degrees each). For each sensor reading, compute its weight based on distance and add it to the appropriate sector bin. Closer obstacles = higher weight.

### Step 4: Threshold and Find Valleys
Apply a threshold to the histogram. Scan for contiguous runs of below-threshold sectors — these are valleys. Track their start, end, and center angles.

### Step 5: Select Steering Direction
For each valley, compute the candidate steering angle (center of the valley, or the edge nearest the goal for wide valleys). Evaluate the cost function for each candidate. Pick the lowest cost.

### Step 6: Move the Robot
Update the robot's position by moving a step in the selected direction. Re-check if we've reached the goal (within a tolerance). If stuck (no free valleys), implement a recovery behavior (e.g., turn in place).

### Step 7: Run the Simulation
Loop through timesteps, printing robot state and decisions. Visualize the path taken.

## Learning Objectives

- Understand reactive vs. deliberative navigation and when each is appropriate
- Implement the Vector Field Histogram algorithm from first principles
- Work with polar coordinates, angular arithmetic (wrapping!), and sensor models
- Handle noisy sensor data through aggregation and thresholding
- Design cost functions that balance competing objectives
- Build a simulation loop that demonstrates emergent intelligent behavior from simple rules

## Going Deeper

- **VFH+** adds a second stage that considers robot kinematics (turning radius) to reject directions the robot physically can't achieve
- **VFH*** adds a third stage with A* lookahead to avoid local minima (e.g., U-shaped traps)
- **Dynamic Window Approach (DWA)** is an alternative that searches velocity space instead of direction space
- **Potential Fields** are simpler but suffer from local minima — VFH was designed to overcome this
- In production, obstacle avoidance runs as one layer in a navigation stack (e.g., ROS navigation stack: global planner → local planner → obstacle avoidance → motor commands)
- Real LIDAR produces thousands of points per scan — the histogram compression is what makes real-time processing feasible
