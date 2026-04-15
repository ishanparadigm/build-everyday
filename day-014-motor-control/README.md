# Day 14: Motor Control Simulation

## What You're Building

A complete DC motor control simulator — modeling the electrical and mechanical dynamics of a DC motor, implementing PWM (Pulse Width Modulation) speed control, and building closed-loop velocity and position controllers on top of it.

**Why it matters:** Every robot that moves uses motors, and every motor needs a controller. Whether it's a drone stabilizing in wind, a robotic arm reaching for an object, or an electric vehicle managing torque — the physics and control principles are identical. Understanding motor dynamics from first principles lets you tune controllers that are responsive without being unstable, and predict behavior before deploying on real hardware.

This challenge builds directly on Day 6 (PID controller) by applying PID to a realistic plant model instead of an abstract system.

---

## Core Concepts

### 1. DC Motor Physics

A DC motor converts electrical energy to mechanical energy via electromagnetism. The key equations come from two coupled domains:

**Electrical domain** (Kirchhoff's voltage law around the armature circuit):

```
V = L * (di/dt) + R * i + K_e * omega
```

Where:
- `V` = applied voltage (volts)
- `L` = armature inductance (henries) — usually small, often neglected
- `R` = armature resistance (ohms)
- `i` = armature current (amps)
- `K_e` = back-EMF constant (V/(rad/s))
- `omega` = angular velocity (rad/s)

The back-EMF term `K_e * omega` is crucial: as the motor spins faster, it generates a voltage that *opposes* the applied voltage. This is the natural speed-limiting mechanism — it's why a motor doesn't accelerate forever.

**Mechanical domain** (Newton's second law for rotation):

```
J * (d_omega/dt) = K_t * i - B * omega - T_load
```

Where:
- `J` = moment of inertia (kg*m^2)
- `K_t` = torque constant (N*m/A) — numerically equal to K_e in SI units
- `B` = viscous friction coefficient (N*m*s/rad)
- `T_load` = external load torque (N*m)

**The intuition:** Apply voltage -> current flows -> current creates torque -> torque accelerates the rotor -> spinning rotor creates back-EMF -> back-EMF reduces current -> system reaches steady state. This negative feedback loop is inherent to the motor physics.

### 2. PWM (Pulse Width Modulation)

Instead of varying voltage continuously (expensive in power electronics), we rapidly switch between full voltage and zero. The motor's inductance acts as a low-pass filter, so it "sees" the average voltage:

```
V_effective = duty_cycle * V_supply
```

Where `duty_cycle` is between 0.0 (off) and 1.0 (full power).

**Why PWM instead of analog voltage control?** Efficiency. A transistor switching fully on/off dissipates almost no power. A transistor operating in its linear region (partial on) dissipates significant heat. PWM gives us fine-grained control with minimal power loss.

**PWM frequency tradeoff:**
- Too low (< 1 kHz): audible whine, jerky motion, current ripple damages motor
- Too high (> 50 kHz): switching losses in transistor increase, diminishing returns
- Sweet spot: 5-25 kHz for most DC motors

### 3. Closed-Loop Motor Control

Open-loop control (apply fixed PWM duty cycle) fails because:
- Load changes alter speed (pick up a heavy object, motor slows)
- Voltage supply varies (battery drains)
- Friction changes with temperature

**Velocity control:** Use a PID controller where:
- Setpoint = desired angular velocity
- Process variable = measured angular velocity (from encoder)
- Output = PWM duty cycle (clamped to [0, 1])

**Position control (cascaded):** An outer PID loop generates a velocity *setpoint*, and the inner velocity loop tracks it. This cascade structure is standard in robotics because:
1. The inner loop rejects disturbances faster
2. You can limit velocity independently of position error
3. Each loop can be tuned independently

### 4. Numerical Integration (Euler vs RK4)

We simulate continuous dynamics in discrete time steps. The choice of integrator matters:

**Euler method:** `x(t+dt) = x(t) + f(x,t) * dt`
- Simple but accumulates error as O(dt)
- Can go unstable with large dt or stiff systems

**Runge-Kutta 4 (RK4):** Evaluates the derivative at 4 points per step
- Error is O(dt^4) — dramatically more accurate
- Stable for much larger dt values
- Standard choice for robotics simulation

We use RK4 in this challenge for accurate motor dynamics.

---

## Step-by-Step Breakdown

### Step 1: Model the DC Motor

Define the motor parameters (resistance, inductance, torque constant, inertia, friction) and implement the state derivative function. The state vector is `[current, angular_velocity, angle]`.

**Why angle too?** Position control needs it, and integrating velocity gives us angle for free.

### Step 2: Implement RK4 Integration

Write a general-purpose RK4 stepper that can integrate any ODE system. This keeps the motor model clean and the integrator reusable.

### Step 3: Simulate Open-Loop Response

Apply a step voltage and watch the motor spin up. Verify the steady-state speed matches the analytical prediction: `omega_ss = V / K_e` (when friction is zero) or `omega_ss = (V - R*B*V/(K_t*K_e + R*B)) / K_e` in the general case.

### Step 4: Add PWM Modulation

Replace continuous voltage with a PWM signal. Verify that the motor's response to a PWM signal matches the response to the equivalent average voltage (due to the inductance filtering).

### Step 5: Build Velocity PID Controller

Implement a discrete PID controller with:
- Anti-windup (clamp integral term)
- Derivative filtering (low-pass on D term to avoid noise amplification)
- Output clamping to valid PWM range [0, 1]

Test with step changes in desired velocity and with disturbance torques.

### Step 6: Build Cascaded Position Controller

Add an outer position PID loop whose output is a velocity setpoint for the inner loop. Tune the outer loop to be slower than the inner loop (~3-5x lower bandwidth).

### Step 7: Analyze and Visualize

Plot motor response curves: voltage, current, velocity, position vs time. Compare open-loop vs closed-loop performance under load disturbances.

---

## Learning Objectives

- Model electromechanical systems from first principles (coupled ODEs)
- Implement RK4 numerical integration for accurate simulation
- Understand PWM and its role in power electronics
- Build and tune cascaded PID controllers for velocity and position
- Analyze transient response: rise time, overshoot, settling time, steady-state error

---

## Going Deeper

- **Field-Oriented Control (FOC):** For brushless motors (BLDC/PMSM), you control current in a rotating reference frame. This is what modern drones and EVs use.
- **Model Predictive Control (MPC):** Instead of PID, optimize the control trajectory over a future horizon. Better performance but computationally expensive.
- **System identification:** In real systems, you don't know motor parameters exactly. You apply test signals and fit the model to measured data.
- **Current limiting:** Real motor controllers have an inner current loop to protect the motor from overcurrent. This makes a 3-level cascade: position -> velocity -> current.
- **Regenerative braking:** When a motor decelerates, it acts as a generator. Capturing that energy (instead of dissipating it as heat) is critical for EVs and battery-powered robots.
- **Encoder quantization:** Real encoders have finite resolution. At low speeds, velocity estimation becomes noisy — requiring observer-based estimation (Luenberger observer, Kalman filter from a future challenge).
