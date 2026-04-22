# Day 021: Line-Following Robot Logic

## Overview

Build a complete line-following robot simulation from scratch. A line-following robot uses optical sensors to detect a contrasting line on the ground and adjusts its motor outputs to stay centered on that path. This is one of the most fundamental problems in mobile robotics — it's how warehouse robots follow magnetic tape paths, how agricultural robots track crop rows, and how autonomous vehicles stay in lane. Despite its simplicity, it touches sensor processing, control theory, and real-time decision-making.

We'll simulate a differential-drive robot with an array of reflectance sensors, implement multiple control strategies (bang-bang, proportional, and full PID), and compare their tracking performance on curves of varying difficulty.

## Core Concepts

### Reflectance Sensor Arrays

A line-following robot typically uses an array of infrared reflectance sensors mounted on its underside. Each sensor emits IR light and measures how much bounces back — dark surfaces (the line) absorb more light and return lower values, while light surfaces (the background) reflect more.

**Sensor model**: Each sensor reads a value in [0, 1] where 0 = fully on line (dark) and 1 = fully off line (light). In reality, the reading depends on the distance from sensor to the line center:

```
reading(d) = 1 - exp(-d^2 / (2 * sigma^2))
```

where `d` is the lateral distance from sensor to line center and `sigma` controls the line's optical width. This Gaussian model captures how a sensor gradually transitions between "on line" and "off line" — there's no hard edge.

**Weighted position estimation**: With N sensors at known positions x_i across the robot's width, we compute the estimated line position as a weighted average:

```
line_position = sum((1 - r_i) * x_i) / sum(1 - r_i)
```

where `r_i` is the reading from sensor i. This gives us a continuous estimate of where the line is relative to the robot's center. When the line is centered, `line_position = 0`. When it drifts left, `line_position < 0`. This is our **error signal** for the controller.

### Differential Drive Kinematics

A differential-drive robot has two independently driven wheels. The robot's motion is determined by the speeds of left and right wheels:

```
v = (v_right + v_left) / 2        # forward velocity
omega = (v_right - v_left) / L     # angular velocity (L = wheel base)
```

The robot's pose (x, y, theta) updates as:

```
x_new = x + v * cos(theta) * dt
y_new = y + v * sin(theta) * dt
theta_new = theta + omega * dt
```

**Key insight**: To turn, we create a speed difference between wheels. To turn left, slow the left wheel (or speed up the right). The tighter the turn we need, the larger the speed difference. This is where the controller output maps directly to physical action.

### Control Strategies

#### Bang-Bang Control
The simplest approach: if the line is left of center, turn left at full power. If right, turn right. Mathematically:

```
correction = +MAX if error > 0
correction = -MAX if error < 0
```

**Problem**: The robot oscillates wildly around the line because it's always applying maximum correction. There's no concept of "a little off" vs "way off." This works on straight paths but fails on curves.

#### Proportional Control (P)
Scale the correction proportionally to the error:

```
correction = Kp * error
```

**Improvement**: Small errors get small corrections. The robot smoothly tracks the line on gentle curves. **Problem**: On sharp curves, the robot needs a sustained correction. P-control only reacts to current error — it can't "lean into" a curve. The result is a steady-state offset on curves (the robot tracks slightly outside the curve).

#### PID Control
Add integral (accumulated error) and derivative (rate of change) terms:

```
correction = Kp * error + Ki * integral(error) + Kd * d(error)/dt
```

- **Integral**: Eliminates steady-state offset by accumulating error over time. On a sustained curve, the integral builds up and pushes the robot closer to the line.
- **Derivative**: Anticipates future error by looking at the rate of change. If the error is growing quickly (line curving away), the derivative kicks in before the proportional term would. This provides damping and reduces overshoot.

**PID tuning tradeoffs**: High Kp = fast response but oscillation. High Ki = eliminates offset but can cause integral windup and overshoot. High Kd = smooth response but amplifies sensor noise. Real robots need careful tuning — we'll measure this quantitatively.

### Performance Metrics

We evaluate controllers on:
- **Mean Absolute Error (MAE)**: Average absolute deviation from line center. Lower = better tracking.
- **Max Error**: Worst-case deviation. Important for safety — how far does the robot ever stray?
- **Oscillation**: Standard deviation of error. High oscillation means uncomfortable/inefficient movement.
- **Line Loss Events**: How often the line completely leaves the sensor array. A lost line is a critical failure.

## Step-by-Step Breakdown

1. **Build the track**: Define a 2D path as a sequence of (x, y) waypoints. Support straight segments and circular arcs with configurable radius. The track is the "ground truth" line position.

2. **Simulate the sensor array**: Place N sensors at fixed offsets across the robot's width. For each sensor, compute its world position based on robot pose, find the closest point on the track, and generate a reading using the Gaussian model.

3. **Estimate line position**: Apply the weighted-average formula to convert raw sensor readings into a single error value.

4. **Implement controllers**: Build bang-bang, P, and PID controllers that take error as input and output a steering correction.

5. **Map correction to wheel speeds**: Convert the base speed and steering correction into left/right wheel velocities. Clamp to physical limits.

6. **Run the simulation loop**: At each timestep, read sensors -> estimate error -> compute correction -> update wheel speeds -> update robot pose. Log everything for analysis.

7. **Compare performance**: Run all three controllers on the same track and compare MAE, max error, oscillation, and smoothness.

## Learning Objectives

- Understand reflectance sensor modeling and weighted position estimation
- Implement and compare bang-bang, P, and PID control strategies quantitatively
- Build a differential-drive kinematic simulation
- Analyze control performance metrics: tracking accuracy, oscillation, and robustness
- Connect to Day 006 (PID control fundamentals) and Day 014 (motor dynamics) — this is PID applied to a real navigation task

## Going Deeper

- **Adaptive PID**: Adjust gains based on estimated curvature. Increase Kd on straights for stability, increase Ki on curves for tighter tracking.
- **Sensor noise and filtering**: Add Gaussian noise to sensor readings and implement a low-pass filter or Kalman filter (Day 016) to smooth the error signal.
- **Predictive control**: Use the derivative term to estimate where the line *will be* and steer proactively — connects to Model Predictive Control (MPC).
- **Junction handling**: What happens when the sensor array sees two lines (a fork)? Production line-followers need state machines (Day 007) to handle intersections.
- **Hardware mapping**: The sensor model and control loop map directly to Arduino/Pololu QTR sensor arrays. The simulation parameters are chosen to match real hardware.
