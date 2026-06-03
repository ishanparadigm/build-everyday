# Day 063: Sensor Fusion (IMU + GPS)

## Overview

Build a sensor fusion system that combines noisy Inertial Measurement Unit (IMU) and GPS readings to produce accurate position and velocity estimates for a moving robot.

**Why this matters:** No single sensor is perfect. GPS gives absolute position but updates slowly (1-10 Hz) and jumps around. IMUs give smooth, fast acceleration data (100+ Hz) but drift over time because you're integrating noise. Every autonomous vehicle, drone, and phone navigation system solves this exact problem: combine fast-but-drifty IMU data with slow-but-absolute GPS data to get the best of both worlds.

## Core Concepts

### 1. IMU (Inertial Measurement Unit)

An IMU measures acceleration and angular velocity using accelerometers and gyroscopes. To get position from acceleration, you integrate twice:

```
velocity(t) = velocity(t-1) + acceleration * dt
position(t) = position(t-1) + velocity * dt
```

The problem: every tiny error in acceleration gets integrated into velocity, and then integrated again into position. A 0.01 m/s^2 bias becomes 0.01*t m/s velocity error and 0.005*t^2 m position error. After 60 seconds, you're off by 18 meters — and it gets worse quadratically.

This is called **dead reckoning drift**, and it's the fundamental reason IMU-only navigation fails over time.

### 2. GPS (Global Positioning System)

GPS gives absolute position by triangulating satellite signals. Key characteristics:
- **Accuracy:** ~2-5m civilian, ~0.3m with RTK corrections
- **Update rate:** Typically 1-10 Hz (slow compared to IMU's 100+ Hz)
- **Noise model:** Approximately Gaussian with occasional outliers (multipath, urban canyons)
- **No drift:** Errors don't accumulate over time — each fix is independent

GPS is the "anchor" that prevents long-term drift, but it's too slow and noisy for real-time control.

### 3. The Extended Kalman Filter (EKF)

The Kalman filter is the mathematical framework for optimally combining predictions (from IMU) with observations (from GPS). It maintains:

- **State vector x:** What we're estimating [position_x, position_y, velocity_x, velocity_y]
- **Covariance matrix P:** How uncertain we are about each state (and correlations between them)
- **Process noise Q:** How much the state changes unpredictably between steps
- **Measurement noise R:** How noisy the GPS observations are

The filter alternates between two steps:

**Predict (IMU update, every dt):**
```
x_predicted = F @ x + B @ u        # State transition with IMU input
P_predicted = F @ P @ F^T + Q      # Uncertainty grows
```

Where F is the state transition matrix and u is the IMU acceleration input.

**Update (GPS correction, when available):**
```
y = z - H @ x_predicted            # Innovation: GPS minus prediction
S = H @ P_predicted @ H^T + R      # Innovation covariance
K = P_predicted @ H^T @ S^-1       # Kalman gain: how much to trust GPS
x_corrected = x_predicted + K @ y  # Blend prediction with GPS
P_corrected = (I - K @ H) @ P_predicted  # Uncertainty shrinks
```

The **Kalman gain K** is the key insight: it automatically balances trust between the IMU prediction and the GPS measurement based on their respective uncertainties. When IMU drift has grown large (big P), K is large and we trust GPS more. When GPS is noisy (big R), K is small and we trust the IMU more.

### 4. Why "Extended"?

The standard Kalman filter assumes linear dynamics. Real IMU data involves rotations and nonlinear motion models. The Extended Kalman Filter (EKF) handles this by linearizing the dynamics at each timestep using Jacobians. For our 2D case with constant-acceleration motion, the dynamics are actually linear, so we get the standard KF — but the code structure generalizes to the nonlinear case.

### 5. State Space Model

Our state vector: `x = [px, py, vx, vy]^T`

**State transition (constant velocity + acceleration input):**
```
px_new = px + vx*dt + 0.5*ax*dt^2
py_new = py + vy*dt + 0.5*ay*dt^2
vx_new = vx + ax*dt
vy_new = vy + ay*dt
```

**Measurement model (GPS observes position only):**
```
z = [px, py] + noise
```

## Step-by-Step Breakdown

1. **Simulate ground truth trajectory** — Generate a realistic 2D path with known positions, velocities, and accelerations. This is our reference for evaluating accuracy.

2. **Simulate IMU readings** — Add Gaussian noise and a constant bias to the true acceleration. The bias models real IMU calibration errors and is the main source of drift.

3. **Simulate GPS readings** — Sample the true position at a lower rate (e.g., 1 Hz vs 100 Hz) and add Gaussian noise. Occasionally drop readings to simulate signal loss.

4. **Implement dead reckoning** — Integrate the noisy IMU data to show how quickly position estimates diverge without correction. This is the "before" picture.

5. **Implement the Kalman filter** — Build the predict/update cycle. Predict at IMU rate, update at GPS rate. Track the covariance to see uncertainty grow and shrink.

6. **Compare results** — Plot all three trajectories (ground truth, IMU-only, fused) and compute RMS errors to quantify the improvement.

## Learning Objectives

- Understand why sensor fusion is necessary (complementary error characteristics)
- Implement the Kalman filter predict/update cycle from scratch using matrix math
- Model IMU noise, bias, and integration drift
- Combine sensors with different update rates (asynchronous fusion)
- Evaluate filter performance with RMS error metrics
- Build intuition for how the Kalman gain balances sensor trust

## Going Deeper

- **Nonlinear dynamics:** Add heading/orientation to make it a true EKF with Jacobian linearization
- **Outlier rejection:** Implement Mahalanobis distance gating to reject GPS outliers (multipath)
- **Adaptive noise:** Estimate Q and R online using innovation sequences
- **Other filters:** Compare with Unscented Kalman Filter (UKF) or particle filter for highly nonlinear systems
- **Real applications:** This exact architecture runs in every phone (Google Fused Location Provider), every drone (PX4/ArduPilot EKF2), and every self-driving car
- **Connection to Day 045 (Kalman filter basics):** This builds directly on those fundamentals, adding multi-sensor fusion and realistic noise modeling
