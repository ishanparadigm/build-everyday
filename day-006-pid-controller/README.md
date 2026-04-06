# Day 006: PID Controller Simulation

## What You're Building

A complete PID (Proportional-Integral-Derivative) controller from scratch -- the single most important control algorithm in engineering. You'll implement it, simulate it controlling a real system, and build deep intuition for how each term shapes the response. No control theory library, just NumPy and math.

PID controllers are everywhere. Your car's cruise control, your home thermostat, the altitude hold on a drone, the temperature regulation in a 3D printer, the balance system on a Segway -- they all use PID or some close variant. Understanding PID deeply means understanding the bridge between software and the physical world.

## Core Concepts

### The Control Problem

You have a system (a "plant") with some measurable output -- temperature, position, speed, angle. You want that output to match a desired value (the **setpoint**). The difference between the setpoint and the current measurement is the **error**:

```
error(t) = setpoint - measurement(t)
```

The PID controller computes a **control output** that drives the error toward zero. It does this by combining three terms, each responding to a different aspect of the error.

### The PID Equation

```
u(t) = Kp * e(t) + Ki * integral(e(t)dt) + Kd * de(t)/dt
```

In discrete time (which is what we implement):

```
u[n] = Kp * e[n] + Ki * sum(e[0..n]) * dt + Kd * (e[n] - e[n-1]) / dt
```

Where:
- `u[n]` is the control output at timestep n
- `e[n]` is the error at timestep n
- `Kp`, `Ki`, `Kd` are the tuning gains
- `dt` is the timestep duration

### The Three Terms

**Proportional (P) -- React to the present**

```
P = Kp * e(t)
```

The proportional term is the simplest: output is directly proportional to error. Big error, big correction. Zero error, zero correction.

The problem: P-only control almost always leaves a **steady-state error**. Imagine cruise control on a hill -- the car slows down, the error increases, the controller pushes harder, but it settles at a speed slightly below the setpoint. The controller needs *some* error to produce *some* output. This residual error is called **droop** or **offset**.

Increasing Kp reduces steady-state error but makes the system more aggressive. Too much Kp and the system oscillates -- it overshoots the target, then overcorrects, then overcorrects again.

**Integral (I) -- Learn from the past**

```
I = Ki * integral(e(t)dt)
```

The integral term accumulates error over time. Even a tiny persistent error will eventually build up a large integral, which drives the output to eliminate the offset.

This is what kills steady-state error. The integral keeps growing until the error reaches zero on average. It's the "memory" of the controller -- it remembers all past mistakes and compensates.

The danger: **integral windup**. If the system can't respond fast enough (e.g., an actuator is saturated), the integral keeps growing. When the system finally does respond, the huge integral causes massive overshoot. Practical controllers use anti-windup techniques like clamping the integral term.

**Derivative (D) -- Anticipate the future**

```
D = Kd * de(t)/dt
```

The derivative term responds to the *rate of change* of the error. If the error is decreasing quickly, the derivative term reduces the output to prevent overshoot. If the error is increasing quickly, it adds extra correction.

Think of it as a damper -- it resists rapid changes. Without it, the system can ring (oscillate around the setpoint). With it, the system settles smoothly.

The danger: derivative is sensitive to noise. If the measurement is noisy, the derivative amplifies that noise into wild control swings. In practice, the derivative is often applied to the measurement rather than the error, and a low-pass filter is used.

### Tuning Intuition

| Parameter | Too Low | Too High |
|-----------|---------|----------|
| Kp | Sluggish response, large steady-state error | Oscillation, instability |
| Ki | Steady-state error persists | Overshoot, integral windup, slow oscillation |
| Kd | Overshoot, ringing | Amplifies noise, jerky response |

**Ziegler-Nichols method** (a classic tuning heuristic):
1. Set Ki = 0, Kd = 0
2. Increase Kp until the system oscillates with constant amplitude (this Kp is called Ku, the "ultimate gain")
3. Measure the oscillation period Tu
4. Set: Kp = 0.6*Ku, Ki = 2*Kp/Tu, Kd = Kp*Tu/8

### Real-World Applications

- **Drone flight controllers**: Multiple nested PID loops (attitude, rate, position)
- **Thermostats**: PI control (derivative not needed for slow thermal systems)
- **Cruise control**: PID with feedforward for hills and wind
- **Robotic arms**: PID at each joint, often with gravity compensation
- **Self-balancing robots**: PD control with fast update rates
- **Industrial process control**: 95%+ of all industrial controllers are PID

## Step-by-Step Approach

### Step 1: Implement the PIDController class
Build the controller with configurable gains, setpoint, and timestep. Track the integral sum and previous error for the derivative calculation.

### Step 2: Implement a simple plant model
A first-order system (like temperature control) where the plant responds to the control output with some lag.

### Step 3: Run a step response simulation
Apply a step change in setpoint and observe how the system responds. Measure overshoot, settling time, and steady-state error.

### Step 4: Compare controller configurations
Run P-only, PI, PD, and full PID controllers on the same system. Show how each term contributes to the response.

### Step 5: Demonstrate tuning effects
Vary Kp, Ki, Kd individually and show the effect on step response. Build intuition for what each knob does.

## Learning Objectives

- Implement a discrete-time PID controller from the mathematical definition
- Understand how proportional, integral, and derivative terms each contribute to control behavior
- Simulate a closed-loop control system with a plant model
- Analyze step response characteristics: overshoot, settling time, steady-state error
- Develop tuning intuition by comparing different gain configurations
- Understand practical issues like integral windup

## Going Deeper

- **Anti-windup**: Clamp the integral term when the output saturates. More advanced: back-calculation anti-windup.
- **Derivative filtering**: Apply a low-pass filter to the derivative term to reduce noise sensitivity.
- **Cascaded PID**: Use the output of one PID as the setpoint of another (common in drone controllers).
- **Feedforward control**: Combine PID feedback with a model-based feedforward term for faster response.
- **Adaptive PID**: Automatically tune gains based on system identification during operation.
- **Model Predictive Control (MPC)**: The "next level" beyond PID -- optimizes over a future horizon using a system model. Used in self-driving cars and advanced robotics.
