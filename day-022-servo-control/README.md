# Day 022: Servo Control Patterns

## Overview

Servos are the muscles of robotics. Every robotic arm, walking robot, drone gimbal, and RC vehicle relies on servos to convert electrical signals into precise angular motion. Today you'll build a servo control system from scratch — implementing the math behind PWM signal generation, multi-servo coordination, and smooth motion profiles that prevent the jerky, mechanical movements that separate toy robots from professional ones.

**Why this matters:** In production robotics, naive servo control (instantly jumping to target angles) causes mechanical stress, overshooting, and power spikes. Real systems use motion profiles — trapezoidal velocity curves, S-curves, and synchronized multi-servo trajectories — to move smoothly, predictably, and safely.

## Core Concepts

### PWM (Pulse Width Modulation) and Servo Positioning

A standard hobby servo expects a PWM signal with a 20ms period (50Hz). The pulse width within that period determines the angle:

```
  |<--- 20ms period (50Hz) --->|
  |                             |
  |-----|                       |-----|
  | pw  |                       | pw  |
  |     |_______________________|     |___...
```

- **1.0ms pulse** -> 0 degrees (full left)
- **1.5ms pulse** -> 90 degrees (center)
- **2.0ms pulse** -> 180 degrees (full right)

The mapping is linear:

```
pulse_width(angle) = MIN_PULSE + (angle / MAX_ANGLE) * (MAX_PULSE - MIN_PULSE)
```

where MIN_PULSE = 1.0ms, MAX_PULSE = 2.0ms, MAX_ANGLE = 180 degrees.

**Duty cycle** is the ratio of pulse width to period:

```
duty_cycle = pulse_width / period = pulse_width / 20ms
```

For a 0-180 degree range, duty cycle ranges from 5% (1/20) to 10% (2/20).

### Motion Profiles

Moving a servo instantly to a target angle is called a **step command**. This is bad because:
1. Mechanical stress on gears and linkages
2. Power spikes that can brown-out your controller
3. Overshooting and oscillation
4. Unpredictable behavior in multi-servo systems

**Linear interpolation** is the simplest improvement — move at constant velocity:

```
angle(t) = start + (end - start) * (t / duration)
velocity(t) = (end - start) / duration  (constant)
```

Problem: instantaneous acceleration at start and end (velocity jumps from 0 to max).

**Trapezoidal velocity profile** fixes this by adding acceleration/deceleration ramps:

```
Velocity
  ^
  |    ___________
  |   /           \
  |  /             \
  | /               \
  |/                 \
  +---+---+---+---+---> time
    accel cruise decel
```

Three phases:
1. **Acceleration phase** (0 to t_a): velocity ramps linearly from 0 to v_max
2. **Cruise phase** (t_a to t_a + t_c): constant velocity v_max
3. **Deceleration phase**: velocity ramps linearly from v_max to 0

The math:
- Given: total distance `d`, max velocity `v_max`, acceleration `a`
- Accel time: `t_a = v_max / a`
- Distance during accel/decel: `d_ramp = v_max^2 / a` (both ramps combined)
- If `d_ramp > d`: triangle profile (never reaches v_max), reduce `v_max = sqrt(d * a)`
- Cruise distance: `d_c = d - d_ramp`
- Cruise time: `t_c = d_c / v_max`

**S-curve profile** goes further by smoothing the acceleration itself (adding jerk limits), but trapezoidal is the standard for most servo applications.

### Multi-Servo Synchronization

When moving multiple servos simultaneously (e.g., a robotic arm), you need them to start and finish together. Without synchronization, the fastest servo finishes first, causing uncoordinated motion.

**Time synchronization**: find the servo that needs the most time, then slow all others to match:

```
t_total = max(time_needed(servo_i) for all servos)
for each servo:
    adjust velocity so it takes exactly t_total
```

### Easing Functions

Beyond trapezoidal profiles, easing functions from animation provide smooth, natural-looking motion:

- **Ease-in** (slow start): `f(t) = t^2`
- **Ease-out** (slow end): `f(t) = 1 - (1-t)^2`
- **Ease-in-out** (slow start and end): `f(t) = 3t^2 - 2t^3` (Hermite interpolation)
- **Sine easing**: `f(t) = (1 - cos(pi * t)) / 2`

These map a normalized time `t in [0,1]` to a normalized position `f(t) in [0,1]`.

## Step-by-Step Breakdown

1. **Servo model**: Create a `Servo` class that tracks current angle, min/max limits, and converts angles to PWM pulse widths. This is the foundation — without accurate angle-to-PWM mapping, nothing else works.

2. **Motion profiles**: Implement `LinearProfile`, `TrapezoidalProfile`, and `EasingProfile` classes. Each takes start/end angles and duration, and returns the target angle at any time `t`. Separating profiles from servos follows the Strategy pattern — you can swap motion styles without changing servo logic.

3. **Servo controller**: Build a `ServoController` that manages multiple servos, assigns motion profiles, and steps through time. This is the orchestrator — it handles the simulation loop and synchronization.

4. **Synchronized movement**: Implement multi-servo coordination where all servos in a group start and finish movement simultaneously, regardless of how far each needs to travel.

5. **Sequence execution**: Create a system for defining sequences of movements (like a robot arm picking up an object) as a series of keyframes, then executing them with smooth transitions.

## Learning Objectives

- Understand PWM signal generation and servo positioning math
- Implement trapezoidal and easing motion profiles
- Build synchronized multi-servo coordination
- Design a keyframe-based motion sequence system
- Apply the Strategy pattern to swap motion profiles at runtime

## Going Deeper

- **Real hardware**: These patterns map directly to libraries like `pigpio` (Raspberry Pi) or Arduino's `Servo.h`. The math is identical; only the PWM output mechanism changes.
- **Bezier curves**: For complex robotic arm paths, cubic Bezier curves in joint space provide smooth, tunable trajectories with control points.
- **Inverse kinematics integration**: Combine with Day 008's kinematics to compute target angles from desired end-effector positions, then use these motion profiles to reach them smoothly.
- **Torque considerations**: Real servos have torque limits. A proper controller monitors load and adjusts speed to prevent stalling.
- **PID feedback**: Combine with Day 006's PID controller for closed-loop servo control — the motion profile generates setpoints, and PID corrects for real-world error.
