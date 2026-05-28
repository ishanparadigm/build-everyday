# Day 42: Robot Arm Trajectory Planning

## Overview

You're building a **trajectory planner for a 2D robotic arm** that smoothly moves its end-effector from one point to another while respecting joint limits and velocity constraints. This is the core problem behind every industrial robot, surgical robot, and robotic manipulator in the world. Without good trajectory planning, a robot arm either jerks violently (destroying hardware or injuring people) or moves too slowly to be useful.

Real trajectory planners must solve two coupled problems: **path planning** (what sequence of joint configurations connects start to goal?) and **trajectory generation** (how do we move along that path with smooth velocities and accelerations?). Today we tackle both.

## Core Concepts

### Joint Space vs. Task Space

A robot arm can be described in two coordinate systems:
- **Task space (Cartesian)**: Where the end-effector is in (x, y) coordinates — intuitive for humans
- **Joint space**: The angles of each joint (theta_1, theta_2, ...) — what the motors actually control

Planning in joint space is often simpler because joint limits are box constraints, and interpolation between joint configurations is straightforward. But the *goal* is usually specified in task space ("move the gripper to position (0.5, 0.3)"), so we need **inverse kinematics** to convert.

For a 2-link planar arm with link lengths L1 and L2, the inverse kinematics are:

```
cos(theta_2) = (x^2 + y^2 - L1^2 - L2^2) / (2 * L1 * L2)
theta_2 = atan2(+/- sqrt(1 - cos^2(theta_2)), cos(theta_2))
theta_1 = atan2(y, x) - atan2(L2 * sin(theta_2), L1 + L2 * cos(theta_2))
```

Note the +/- in theta_2: most target positions have **two valid solutions** (elbow-up and elbow-down). The trajectory planner must choose consistently.

### Trajectory Profiles

Once we have waypoints in joint space, we need to generate smooth motion profiles. The key constraint: **acceleration must be finite** (infinite acceleration = infinite force = broken robot).

**Linear interpolation** is the simplest but worst approach — constant velocity with instantaneous acceleration at start/stop. Never used in practice.

**Trapezoidal velocity profile** (bang-bang control):
- Phase 1: Constant acceleration from rest to max velocity
- Phase 2: Cruise at max velocity
- Phase 3: Constant deceleration to rest

For a move of distance `d` with max velocity `v_max` and max acceleration `a_max`:
- Acceleration time: `t_a = v_max / a_max`
- Acceleration distance: `d_a = 0.5 * a_max * t_a^2`
- If `2 * d_a > d`: triangular profile (never reaches max velocity)
  - `t_a = sqrt(d / a_max)`, total time = `2 * t_a`
- Otherwise: trapezoidal
  - Cruise distance: `d_c = d - 2 * d_a`
  - Cruise time: `t_c = d_c / v_max`
  - Total time: `2 * t_a + t_c`

**Cubic polynomial trajectory** provides smooth position and velocity:

```
q(t) = a0 + a1*t + a2*t^2 + a3*t^3
```

Given boundary conditions q(0)=q_start, q(T)=q_end, q_dot(0)=0, q_dot(T)=0:
```
a0 = q_start
a1 = 0
a2 = 3*(q_end - q_start) / T^2
a3 = -2*(q_end - q_start) / T^3
```

**Quintic polynomial** also constrains acceleration at boundaries (smoother but harder to compute).

### Multi-Segment Trajectories (Via Points)

Real tasks require moving through multiple waypoints. At each via point, we must ensure **velocity continuity** — the arm doesn't stop at each point. This means solving for intermediate velocities that balance:
- Smoothness (no sudden velocity changes)
- Timing (respect the desired duration for each segment)

A common heuristic: set via-point velocity to the average of the velocities of the incoming and outgoing linear segments.

### Workspace and Joint Limits

Not every (x, y) point is reachable. The workspace of a 2-link arm is the annular region:
- Outer radius: `L1 + L2`
- Inner radius: `|L1 - L2|`

Joint limits further restrict this. A good trajectory planner validates reachability and checks joint limits at every timestep, not just at waypoints.

## Step-by-Step Breakdown

1. **Build the arm model**: Define link lengths, joint limits, and max velocity/acceleration per joint. Implement forward kinematics (joint angles -> end-effector position) and inverse kinematics (position -> joint angles).

2. **Implement trajectory profiles**: Build both trapezoidal and cubic polynomial generators. Each takes start/end configurations and timing, and returns a function of time that gives position, velocity, and acceleration.

3. **Multi-waypoint planner**: Given a list of task-space waypoints, convert each to joint space, compute via-point velocities, and generate a continuous multi-segment trajectory.

4. **Validation**: At each timestep, verify joint positions are within limits and velocities don't exceed maximums. Flag any violations.

5. **Visualization**: Print the trajectory state at sampled time steps so you can trace the arm's motion.

## Learning Objectives

- Understand the difference between path planning and trajectory generation
- Implement inverse kinematics for a 2-link planar arm
- Build trapezoidal and cubic polynomial trajectory generators
- Handle multi-segment trajectories with velocity continuity at via points
- Validate trajectories against joint and velocity constraints

## Going Deeper

- **Quintic splines**: Add acceleration boundary conditions for even smoother motion
- **Obstacle avoidance**: Combine with configuration-space obstacle mapping (builds on Day 28 A* and Day 34 RRT)
- **Minimum-time trajectories**: Optimize timing to move as fast as possible while respecting all constraints — this is a nonlinear optimization problem used in real industrial robots
- **Jerk limits**: Real servos also have maximum jerk (derivative of acceleration). S-curve profiles handle this
- **Redundant arms**: With 3+ joints for 2D (or 6+ for 3D), there are infinite IK solutions — null-space optimization becomes important
- **Dynamic trajectory generation**: Account for the arm's mass and inertia, not just kinematic limits — leads to computed torque control
