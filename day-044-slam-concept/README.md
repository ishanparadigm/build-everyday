# Day 044: SLAM Concept Implementation

## Overview

**Simultaneous Localization and Mapping (SLAM)** is one of the most important problems in robotics: a robot must build a map of an unknown environment while simultaneously figuring out where it is within that map. It's a chicken-and-egg problem — you need a map to localize, but you need to know your location to build a map. SLAM solves both at once.

This challenge implements **EKF-SLAM** (Extended Kalman Filter SLAM), the classical approach that fuses noisy odometry and landmark observations into a single probabilistic state estimate. Every self-driving car, autonomous drone, and warehouse robot relies on some variant of SLAM.

## Core Concepts

### The SLAM Problem

A robot moves through the world with **odometry** (wheel encoders, IMU) that tells it roughly how far it moved and turned. But odometry drifts — small errors accumulate into large ones. The robot also has **sensors** (lidar, camera, sonar) that detect **landmarks** — distinguishable features in the environment.

The key insight: if the robot re-observes a landmark it saw before, it can correct its position estimate. And correcting the robot's position also corrects the estimated positions of all other landmarks. Everything is correlated.

### State Vector

In EKF-SLAM, we maintain a single large state vector:

```
x = [x_r, y_r, θ_r, x_L1, y_L1, x_L2, y_L2, ..., x_Ln, y_Ln]
```

- `(x_r, y_r, θ_r)`: Robot pose (position + heading)
- `(x_Li, y_Li)`: Position of landmark i

The state has dimension `3 + 2N` where N is the number of landmarks discovered so far.

### Covariance Matrix

The covariance matrix `P` is `(3+2N) × (3+2N)` and encodes:
- **Robot uncertainty**: How confident we are about the robot's pose
- **Landmark uncertainty**: How confident we are about each landmark's position
- **Cross-correlations**: How the robot's uncertainty relates to each landmark's uncertainty

These cross-correlations are what make SLAM work. When the robot re-observes a landmark, the correction propagates through the covariance to improve estimates of ALL landmarks, not just the one observed.

### Motion Model (Prediction Step)

When the robot moves, we predict the new state using the odometry:

```
x_r' = x_r + d·cos(θ_r + α)
y_r' = y_r + d·sin(θ_r + α)
θ_r' = θ_r + α + β
```

Where `d` is distance traveled, `α` is the turn before moving, `β` is the turn after. The Jacobian of this motion model (matrix `F`) tells us how uncertainty propagates. After prediction:

```
P' = F·P·Fᵀ + Q
```

Where `Q` is the process noise (how much we distrust odometry). Landmarks don't move, so only the robot rows/columns of P change.

### Observation Model (Update Step)

When the robot observes landmark i at range `r` and bearing `φ`:

```
Expected range:   r̂ = √((x_Li - x_r)² + (y_Li - y_r)²)
Expected bearing: φ̂ = atan2(y_Li - y_r, x_Li - x_r) - θ_r
```

The **innovation** (surprise) is:

```
z - ẑ = [r - r̂, φ - φ̂]
```

The Jacobian `H` of this observation model tells us how sensitive the expected observation is to changes in the state. The Kalman gain:

```
K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
```

Where `R` is measurement noise. Then we update:

```
x = x + K·(z - ẑ)
P = (I - K·H)·P
```

This is where the magic happens: the correction from one landmark observation flows through the cross-correlations to update the entire state.

### Data Association

Before updating, we need to know WHICH landmark we're observing. This implementation uses a simple known-correspondence model (each landmark has a unique ID). In production, you'd use nearest-neighbor matching or more sophisticated techniques like JCBB.

## Step-by-Step Breakdown

1. **Initialize**: Start with the robot at origin, empty landmark set, small initial covariance
2. **Predict**: For each odometry command, propagate the state and grow the covariance
3. **Observe**: For each sensor reading, check if it's a known landmark or a new one
4. **New landmark**: Expand the state vector and covariance matrix to include it
5. **Known landmark**: Compute innovation, Kalman gain, and update the full state
6. **Repeat**: Each cycle improves the map and the robot's localization

## Learning Objectives

- Understand the SLAM problem and why it's fundamental to autonomous robots
- Implement the Extended Kalman Filter for joint robot-landmark estimation
- Work with Jacobian matrices for nonlinear state estimation
- See how cross-correlations enable map-wide corrections from single observations
- Appreciate the computational tradeoffs (EKF-SLAM is O(N²) per update)

## Going Deeper

- **Sparsity**: EKF-SLAM's O(N²) cost is prohibitive for large maps. Graph-based SLAM and particle filter SLAM (FastSLAM) scale better.
- **Loop closure**: The dramatic correction when a robot returns to a previously-visited area. This is where SLAM really shines vs. pure odometry.
- **3D SLAM**: Extend to 6-DOF pose with 3D landmarks. Used in visual SLAM (ORB-SLAM, LSD-SLAM).
- **Sensor types**: Lidar SLAM vs visual SLAM vs radar SLAM — each has different noise characteristics and data association challenges.
- **Connection to Day 028 (A\*)**: SLAM builds the map that path planning algorithms like A* operate on. In a real system, SLAM runs continuously while the planner uses the current best map.
- **Connection to Day 043 (Swarm)**: Multi-robot SLAM where robots share map information to build a collaborative map faster.
