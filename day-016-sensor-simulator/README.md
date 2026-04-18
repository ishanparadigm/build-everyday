# Day 16: Sensor Reading Simulator

## Overview

Build a realistic sensor simulation system that models how robots perceive the world — noisy, delayed, and imperfect. In real robotics, sensors never give you the truth. A LIDAR returns distance estimates corrupted by Gaussian noise. An IMU drifts over time. A camera's depth sensor has quantization error. Understanding sensor models is foundational because every downstream algorithm (SLAM, path planning, control) must reason about uncertainty.

You'll implement multiple sensor types with realistic noise models, simulate sensor fusion basics, and build a framework for testing robot perception algorithms without physical hardware.

## Core Concepts

### Sensor Noise Models

Every physical sensor measurement is a random variable. The true value `x` is corrupted:

```
z = h(x) + v
```

Where `z` is the measurement, `h(x)` is the ideal sensor model (often just identity), and `v` is noise drawn from some distribution.

**Gaussian (Normal) Noise** — The most common model. If `v ~ N(0, sigma^2)`, then measurements cluster around the true value with standard deviation `sigma`. This arises from the Central Limit Theorem: many small independent error sources sum to a Gaussian.

**Why sigma matters**: A LIDAR with sigma=0.01m is precise to ~1cm (68% of readings within 1cm of truth). A cheap ultrasonic with sigma=0.05m is 5x worse. This directly affects what algorithms you can run — you can't do cm-precision mapping with a 5cm-noise sensor.

**Bias (Systematic Error)** — A constant or slowly-drifting offset. IMU gyroscopes are notorious for this: even sitting still, they report a small nonzero rotation rate. Bias doesn't average out with more samples (unlike zero-mean Gaussian noise), making it especially dangerous.

```
z = h(x) + bias + v
bias(t+1) = bias(t) + w,  where w ~ N(0, sigma_bias^2)
```

### Sensor Types and Their Characteristics

**Range sensors (LIDAR, Ultrasonic, IR)**:
- Measure distance to nearest obstacle
- LIDAR: low noise (sigma ~1-3cm), high rate (10-40Hz), narrow beam
- Ultrasonic: higher noise (sigma ~3-5cm), lower rate (10Hz), wide cone
- Key failure mode: specular reflection (smooth surfaces at angles bounce the signal away)

**Inertial Measurement Unit (IMU)**:
- Accelerometer (linear acceleration) + Gyroscope (angular velocity)
- High rate (100-1000Hz) but drifts over time
- The drift problem: integrating noisy angular velocity to get orientation means errors accumulate. After minutes, a cheap IMU can be off by degrees.

**Wheel Encoders (Odometry)**:
- Count wheel rotations → estimate distance traveled
- Subject to slip (wheels spin without moving) and drift
- Dead reckoning: position = integral of velocity. Errors compound quadratically with distance.

### Sensor Fusion Preview

No single sensor is reliable enough alone. Sensor fusion combines multiple sensors to get a better estimate. The key insight: if two independent sensors measure the same thing, their combined estimate is better than either alone.

For two Gaussian sensors with variances `sigma_1^2` and `sigma_2^2`:
```
combined_variance = 1 / (1/sigma_1^2 + 1/sigma_2^2)
combined_mean = (z1/sigma_1^2 + z2/sigma_2^2) * combined_variance
```

The combined variance is always smaller than either individual variance. This is why robots carry multiple sensors — redundancy isn't just for safety, it's for accuracy.

## Step-by-Step Breakdown

### Step 1: Base Sensor Class
Create an abstract sensor with configurable noise parameters (sigma, bias, rate, range limits). Every sensor shares this interface — it takes a true world state and returns a noisy measurement. This abstraction lets downstream code work with any sensor type uniformly.

### Step 2: Range Sensor (LIDAR)
Simulate a rotating LIDAR that casts rays into a 2D world of line-segment obstacles. For each ray: compute intersection with all obstacles, take the nearest, add Gaussian noise, and clamp to the sensor's min/max range. Without the range clamp, you'd get impossible readings that crash your algorithms.

### Step 3: IMU Simulator
Model an accelerometer and gyroscope with bias drift. The bias random walk means the sensor slowly "wanders" — you'll see the error grow as sqrt(t), which is why IMU-only navigation degrades so fast.

### Step 4: Wheel Encoder / Odometry
Simulate a differential-drive robot's wheel encoders. Convert commanded velocity to encoder ticks, add noise proportional to distance traveled (odometry noise scales with motion, not time). Integrate to get position — and watch the position estimate drift from truth.

### Step 5: Sensor Fusion (Weighted Average)
Combine two range sensors measuring the same distance using the optimal Gaussian weighting formula above. Demonstrate that the fused estimate has lower error than either sensor alone.

### Step 6: Simulation Loop
Run a robot moving through a 2D world, collecting sensor readings at each timestep. Log true state vs. sensor readings to visualize how noise and drift affect perception.

## Learning Objectives

- Understand Gaussian noise models and how sensor parameters affect measurement quality
- Implement realistic sensor simulations for LIDAR, IMU, and wheel encoders
- See how bias drift causes unbounded error growth in dead reckoning
- Apply basic sensor fusion to combine redundant measurements
- Build the foundation for Day 6's PID controller and future SLAM/Kalman filter work

## Going Deeper

- **Kalman Filter**: The principled way to do sensor fusion — maintains a full probability distribution over state, not just point estimates. This is weeks 5-8 material.
- **Non-Gaussian noise**: Real sensors have outliers (e.g., LIDAR multipath). Robust estimators like median filters or RANSAC handle these.
- **Sensor placement optimization**: Where you mount sensors on the robot matters — overlapping fields of view, vibration isolation, and occlusion all affect performance.
- **Hardware-in-the-loop**: This simulator could be extended to generate synthetic data for testing real perception algorithms before deploying on hardware.
