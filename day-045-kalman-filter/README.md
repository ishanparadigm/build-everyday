# Day 045: Kalman Filter Basics

## Overview

Build a Kalman filter from scratch to track a moving object using noisy sensor measurements. The Kalman filter is arguably the most important algorithm in estimation theory — it's the backbone of GPS navigation, autonomous vehicle localization, drone flight controllers, financial time series filtering, and spacecraft guidance systems. Apollo 11 used a Kalman filter to navigate to the moon.

The core insight: you have two sources of information about a system's state — a **prediction** from physics (which drifts over time) and a **measurement** from sensors (which is noisy). The Kalman filter optimally combines these two imperfect sources to produce an estimate that's better than either one alone.

## Core Concepts

### The State Estimation Problem

Imagine tracking a car moving along a road. You have:
- A **model** of how the car moves (position += velocity × dt)
- A **GPS sensor** that gives noisy position readings

Neither source is perfect:
- The model doesn't account for bumps, wind, or driver behavior → prediction drifts
- The GPS has measurement noise → readings jitter around the true position

The Kalman filter answers: **Given my model prediction AND my sensor reading, what's my best estimate of where the car actually is?**

### State Space Representation

We represent the system as:

**State vector** `x`: Everything we want to estimate. For 1D tracking:
```
x = [position, velocity]^T
```

**State transition model** `F`: How the state evolves over one time step (from physics):
```
F = [[1, dt],
     [0,  1]]
```
This encodes: `new_position = old_position + velocity × dt`, `new_velocity = old_velocity`.

**Measurement model** `H`: What the sensor actually observes. If we only measure position:
```
H = [[1, 0]]
```
This means: sensor reading = position (we can't directly measure velocity).

### The Two-Step Dance: Predict and Update

The Kalman filter alternates between two steps every time step:

#### Step 1: Predict (Time Update)
Use the physics model to predict where we think the state is:
```
x_predicted = F @ x_previous
P_predicted = F @ P_previous @ F^T + Q
```

Where:
- `P` is the **covariance matrix** — our uncertainty about the state estimate
- `Q` is the **process noise covariance** — how much we expect the model to be wrong

After prediction, our uncertainty *grows* (we're less sure because the model isn't perfect).

#### Step 2: Update (Measurement Update)
Incorporate the new sensor reading to correct our prediction:
```
y = z - H @ x_predicted          # Innovation (measurement residual)
S = H @ P_predicted @ H^T + R    # Innovation covariance
K = P_predicted @ H^T @ S^(-1)   # Kalman gain
x_updated = x_predicted + K @ y   # Corrected state estimate
P_updated = (I - K @ H) @ P_predicted  # Corrected covariance
```

Where:
- `z` is the actual sensor measurement
- `R` is the **measurement noise covariance** — how noisy the sensor is
- `K` is the **Kalman gain** — the magic ratio that decides how much to trust the sensor vs. the model

### The Kalman Gain: Intuition

The Kalman gain `K` is the heart of the algorithm. Think of it as a trust dial between 0 and 1:

- **K ≈ 1**: "I trust the sensor more than my prediction" → the estimate snaps to measurements
- **K ≈ 0**: "I trust my prediction more than the sensor" → the estimate ignores noisy readings

The filter automatically adjusts K based on the relative uncertainties:
- High process noise (bad model) → K increases → lean on sensors
- High measurement noise (bad sensor) → K decreases → lean on model
- Over time, K converges to a steady-state value

### Why It's Optimal

For linear systems with Gaussian noise, the Kalman filter produces the **minimum variance unbiased estimate**. This means no other linear estimator can do better. The proof relies on the fact that for Gaussian distributions, the mean minimizes mean squared error, and the Kalman filter recursively computes the conditional mean.

### Covariance Matrix P: Tracking Uncertainty

The covariance matrix P doesn't just track point estimates — it maintains a full picture of uncertainty. For our 2D state [position, velocity]:
```
P = [[σ²_pos,       σ_pos_vel],
     [σ_pos_vel,    σ²_vel   ]]
```

- Diagonal: variance of each state variable
- Off-diagonal: correlation between position and velocity uncertainty
- P always decreases (or stays the same) after the update step — measurements always help

## Step-by-Step Breakdown

1. **Initialize**: Set initial state estimate `x₀` and initial uncertainty `P₀`. The filter is robust to poor initialization — it will converge, just takes a few more steps.

2. **Define system matrices**: Set up F (state transition), H (measurement), Q (process noise), and R (measurement noise) based on your physical system.

3. **Predict**: Project state and covariance forward using the physics model. This is your "prior" — what you expect before seeing new data.

4. **Update**: When a measurement arrives, compute the Kalman gain, then blend the prediction with the measurement. This is your "posterior."

5. **Repeat**: Each predict-update cycle refines the estimate. The covariance P shrinks as the filter gains confidence.

## Learning Objectives

- Understand state-space representation of dynamical systems
- Implement the predict-update cycle of a Kalman filter from scratch using NumPy
- Gain intuition for the Kalman gain and how it balances model vs. sensor trust
- Visualize how the filter reduces uncertainty over time
- Track both observable (position) and hidden (velocity) state variables

## Going Deeper

- **Extended Kalman Filter (EKF)**: Handles nonlinear systems by linearizing around the current estimate. Used in most real robotics applications.
- **Unscented Kalman Filter (UKF)**: Better handles nonlinearity by using sigma points instead of Jacobians. More accurate than EKF for highly nonlinear systems.
- **Sensor fusion**: Combine multiple sensors (IMU + GPS + magnetometer) by stacking measurement models. This is exactly what Day 044's SLAM builds upon.
- **Observability**: Can you estimate velocity from position-only measurements? Yes — because the state transition couples them. But not all states are always observable.
- **Tuning Q and R**: In practice, these are tuning knobs. Too-small Q makes the filter sluggish; too-small R makes it jumpy. Adaptive Kalman filters estimate these online.
- **Connection to Bayesian inference**: The Kalman filter is recursive Bayesian estimation for Gaussians. Predict = prior, Update = posterior via Bayes' rule.
