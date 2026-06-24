# Day 84: Model Predictive Control (MPC) for Robot Navigation

## Overview

Model Predictive Control (MPC) is the gold standard for control in modern robotics and autonomous vehicles. Unlike PID controllers (Day 6), which react to errors *after* they happen, MPC looks ahead. It simulates the robot's future trajectory over a prediction horizon, optimizes control inputs to minimize a cost function, applies only the first control action, then re-plans at the next timestep. This "receding horizon" strategy lets robots handle constraints (max speed, obstacle boundaries, actuator limits) while following optimal paths — something PID fundamentally cannot do.

MPC powers Tesla's Autopilot, Boston Dynamics' Atlas, industrial process control, and virtually every modern autonomous vehicle stack. Understanding it deeply gives you the core framework behind predictive robot control.

## Core Concepts

### 1. The Receding Horizon Principle

MPC solves an optimization problem at every timestep:
- Predict the system state over N future timesteps (the **prediction horizon**)
- Find the sequence of N control inputs that minimizes a cost function
- Apply **only the first** control input
- Shift the horizon forward by one step and repeat

Why only apply the first input? Because the model is imperfect. By re-solving at every step, MPC self-corrects for model errors, disturbances, and changing conditions. This is what makes it robust.

### 2. System Dynamics Model

MPC requires a model of how the robot moves. For a differential-drive or bicycle-model robot:

**State vector:** `x = [x_pos, y_pos, theta, velocity]`

**Bicycle model (continuous):**
```
x_dot = v * cos(theta)
y_dot = v * sin(theta)
theta_dot = v / L * tan(delta)
v_dot = a
```

Where `delta` is the steering angle, `a` is acceleration, and `L` is the wheelbase.

**Discretized (Euler method with timestep dt):**
```
x[k+1] = x[k] + v[k] * cos(theta[k]) * dt
y[k+1] = y[k] + v[k] * sin(theta[k]) * dt
theta[k+1] = theta[k] + v[k] / L * tan(delta[k]) * dt
v[k+1] = v[k] + a[k] * dt
```

The discretized model is what MPC uses to "simulate forward" and predict future states.

### 3. Cost Function Design

The cost function J encodes what we want the robot to do:

```
J = sum_{k=0}^{N-1} [ w_pos * ||p[k] - p_ref[k]||^2     # Track reference path
                     + w_vel * (v[k] - v_ref)^2            # Maintain desired speed
                     + w_heading * (theta[k] - theta_ref[k])^2  # Face the right direction
                     + w_steer * delta[k]^2                 # Minimize steering effort
                     + w_accel * a[k]^2                     # Minimize acceleration effort
                     + w_steer_rate * (delta[k] - delta[k-1])^2 ]  # Smooth steering
    + w_terminal * ||p[N] - p_ref[N]||^2                    # Terminal cost
```

**Why each term matters:**
- **Position tracking** — the primary goal, follow the reference
- **Velocity tracking** — maintain desired speed
- **Heading tracking** — face the right direction at each point
- **Control effort** — penalize large inputs to avoid aggressive/jerky behavior
- **Control rate** — penalize *changes* in control to ensure smoothness
- **Terminal cost** — heavier weight on final state to ensure the horizon "pulls" toward the goal

The weights (w_pos, w_steer, etc.) are crucial tuning parameters. Too much position weight → aggressive, jerky control. Too much control penalty → sluggish, overshooting response.

### 4. Constraints

The real power of MPC over PID: explicit constraint handling.

**Input constraints** (actuator limits):
```
-delta_max <= delta[k] <= delta_max    # Max steering angle
-a_max <= a[k] <= a_max                # Max acceleration/braking
```

**State constraints** (safety boundaries):
```
0 <= v[k] <= v_max                     # Speed limits
obstacle_distance(x[k], y[k]) >= d_safe  # Obstacle avoidance
```

**Rate constraints** (smoothness):
```
|delta[k] - delta[k-1]| <= delta_rate_max  # Steering rate limit
```

### 5. Optimization

At each timestep, MPC solves a **nonlinear optimization problem** (because the bicycle model is nonlinear). Common approaches:

- **Sequential Quadratic Programming (SQP)** — linearize, solve QP, repeat
- **Interior Point Methods** — handle constraints with barrier functions
- **Shooting methods** — optimize over control inputs only, simulate states forward

For this challenge, we use a simplified shooting method with scipy.optimize.minimize, which is transparent and educational. Production systems use specialized solvers like IPOPT, ACADO, or CasADi for real-time performance.

## Step-by-Step Breakdown

1. **Define the bicycle kinematic model** — discrete-time state update function
2. **Design the reference trajectory** — a path the robot should follow (figure-8, slalom, etc.)
3. **Implement the cost function** — weighted sum of tracking error + control effort
4. **Set up constraints** — steering limits, speed limits, steering rate limits
5. **Build the MPC controller** — at each timestep: predict forward, optimize, apply first input
6. **Simulate the closed-loop system** — run MPC in a loop, recording state history
7. **Visualize and analyze** — plot trajectories, control inputs, cost evolution, constraint satisfaction

## Learning Objectives

- Understand the receding horizon control paradigm and why it outperforms reactive controllers
- Implement a nonlinear bicycle kinematic model for vehicle/robot simulation
- Design multi-objective cost functions with proper weight tuning
- Handle input and state constraints in an optimization framework
- Build a complete MPC control loop from scratch
- Analyze controller performance through trajectory tracking, smoothness, and constraint satisfaction

## Going Deeper

- **Linearized MPC (Linear MPC):** Linearize the model around the operating point → solve a QP instead of NLP. Much faster, used when nonlinearity is mild.
- **Adaptive MPC:** Update the internal model online as you learn more about the system.
- **Robust MPC / Tube MPC:** Handle bounded model uncertainty by optimizing over worst-case disturbances.
- **MPC with learned dynamics:** Replace the analytical model with a neural network learned from data (connects to Days 57-58 on RL).
- **Real-time MPC:** Production systems need to solve MPC in <10ms. Techniques: warm-starting, explicit MPC (precompute solutions offline), RTI (Real-Time Iteration).
- **Comparison to RL:** MPC is model-based and optimal-per-step; RL is model-free and optimal-over-episodes. Many modern systems combine both.
