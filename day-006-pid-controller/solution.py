"""
Day 006: PID Controller Simulation

A complete PID controller implementation with:
1. Generic PIDController class with configurable gains
2. First-order plant model (thermal system simulation)
3. Step response analysis with overshoot, settling time, steady-state error
4. Comparison of P-only, PI, PD, and full PID controllers
5. Tuning parameter exploration

No matplotlib -- just NumPy and clear printed results.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Dict, Optional


# =============================================================================
# PID Controller
# =============================================================================

class PIDController:
    """
    Discrete-time PID controller.

    u[n] = Kp * e[n] + Ki * sum(e[0..n]) * dt + Kd * (e[n] - e[n-1]) / dt

    The controller tracks integral accumulation and previous error internally.
    Call reset() to clear state (e.g., when changing setpoints discontinuously).
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float,
        dt: float = 0.01,
        output_limits: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.output_limits = output_limits  # (min, max) for anti-windup clamping

        # Internal state
        self._integral: float = 0.0
        self._prev_error: Optional[float] = None

    def update(self, measurement: float) -> float:
        """
        Compute PID output given the current measurement.

        This is the core control loop step:
        1. Compute error = setpoint - measurement
        2. Accumulate integral (error * dt)
        3. Compute derivative (change in error / dt)
        4. Combine: u = Kp*e + Ki*integral + Kd*derivative
        5. Apply output limits if set (anti-windup)
        """
        error = self.setpoint - measurement

        # Proportional term: react to current error
        p_term = self.kp * error

        # Integral term: accumulate past errors
        self._integral += error * self.dt
        i_term = self.ki * self._integral

        # Derivative term: rate of change of error
        if self._prev_error is None:
            derivative = 0.0  # no previous error on first call
        else:
            derivative = (error - self._prev_error) / self.dt
        d_term = self.kd * derivative

        self._prev_error = error

        # Sum all terms
        output = p_term + i_term + d_term

        # Anti-windup: clamp output and prevent integral from growing further
        if self.output_limits is not None:
            lo, hi = self.output_limits
            if output > hi:
                # Back off the integral to prevent windup
                self._integral -= error * self.dt
                output = hi
            elif output < lo:
                self._integral -= error * self.dt
                output = lo

        return output

    def reset(self) -> None:
        """Clear internal state (integral accumulator and previous error)."""
        self._integral = 0.0
        self._prev_error = None


# =============================================================================
# Plant Model: First-Order Thermal System
# =============================================================================

class ThermalPlant:
    """
    Simple first-order thermal system model.

    Models a heater/cooler controlling the temperature of a mass:
        dT/dt = (1/tau) * (K * u - (T - T_ambient))

    Where:
        T = current temperature
        u = control input (heater power, 0-100%)
        tau = thermal time constant (how sluggish the system is)
        K = system gain (how much temperature changes per unit input)
        T_ambient = ambient temperature

    This is a classic first-order lag system -- the simplest nontrivial plant
    for testing PID controllers.
    """

    def __init__(
        self,
        initial_temp: float = 20.0,
        ambient_temp: float = 20.0,
        tau: float = 10.0,
        gain: float = 1.0,
        dt: float = 0.01,
        noise_std: float = 0.0,
    ) -> None:
        self.temperature = initial_temp
        self.ambient_temp = ambient_temp
        self.tau = tau
        self.gain = gain
        self.dt = dt
        self.noise_std = noise_std
        self._rng = np.random.default_rng(42)

    def step(self, control_input: float) -> float:
        """
        Advance the plant by one timestep and return the (noisy) temperature.

        Uses Euler integration: T[n+1] = T[n] + dT/dt * dt
        """
        dT = (1.0 / self.tau) * (
            self.gain * control_input - (self.temperature - self.ambient_temp)
        )
        self.temperature += dT * self.dt

        # Return measurement with optional noise
        if self.noise_std > 0:
            return self.temperature + self._rng.normal(0, self.noise_std)
        return self.temperature

    def reset(self, initial_temp: float = 20.0) -> None:
        """Reset plant to initial conditions."""
        self.temperature = initial_temp


# =============================================================================
# Simulation Engine
# =============================================================================

def simulate(
    controller: PIDController,
    plant: ThermalPlant,
    duration: float,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a closed-loop simulation.

    Returns:
        times: array of time values
        temperatures: array of plant temperatures at each timestep
        control_outputs: array of control signals at each timestep
    """
    n_steps = int(duration / dt)
    times = np.linspace(0, duration, n_steps)
    temperatures = np.zeros(n_steps)
    control_outputs = np.zeros(n_steps)

    for i in range(n_steps):
        measurement = plant.temperature
        control = controller.update(measurement)
        plant.step(control)
        temperatures[i] = plant.temperature
        control_outputs[i] = control

    return times, temperatures, control_outputs


# =============================================================================
# Step Response Analysis
# =============================================================================

def analyze_step_response(
    times: np.ndarray,
    values: np.ndarray,
    setpoint: float,
    tolerance: float = 0.02,
) -> Dict[str, float]:
    """
    Analyze a step response for key performance metrics.

    Returns:
        overshoot: maximum percentage overshoot above setpoint
        settling_time: time to stay within tolerance band around setpoint
        steady_state_error: final error (averaged over last 10% of simulation)
        rise_time: time from 10% to 90% of the step change
    """
    initial = values[0]
    step_size = setpoint - initial

    # Overshoot (percentage of step size)
    if abs(step_size) < 1e-10:
        overshoot = 0.0
    else:
        if step_size > 0:
            peak = np.max(values)
            overshoot = max(0.0, (peak - setpoint) / step_size * 100.0)
        else:
            peak = np.min(values)
            overshoot = max(0.0, (setpoint - peak) / abs(step_size) * 100.0)

    # Settling time (last time the signal exits the tolerance band)
    band = abs(step_size) * tolerance
    within_band = np.abs(values - setpoint) <= band
    if np.any(within_band):
        # Find the last time it was outside the band
        outside = np.where(~within_band)[0]
        if len(outside) == 0:
            settling_time = 0.0
        else:
            settling_time = times[outside[-1]]
    else:
        settling_time = times[-1]  # never settled

    # Steady-state error (average over last 10% of simulation)
    tail = max(1, len(values) // 10)
    steady_state_value = np.mean(values[-tail:])
    steady_state_error = abs(setpoint - steady_state_value)

    # Rise time (10% to 90% of step change)
    target_10 = initial + 0.1 * step_size
    target_90 = initial + 0.9 * step_size
    if step_size > 0:
        crossed_10 = np.where(values >= target_10)[0]
        crossed_90 = np.where(values >= target_90)[0]
    else:
        crossed_10 = np.where(values <= target_10)[0]
        crossed_90 = np.where(values <= target_90)[0]

    if len(crossed_10) > 0 and len(crossed_90) > 0:
        rise_time = times[crossed_90[0]] - times[crossed_10[0]]
    else:
        rise_time = times[-1]

    return {
        "overshoot_pct": overshoot,
        "settling_time": settling_time,
        "steady_state_error": steady_state_error,
        "rise_time": rise_time,
    }


def print_ascii_response(
    times: np.ndarray,
    values: np.ndarray,
    setpoint: float,
    label: str,
    width: int = 60,
    height: int = 15,
) -> None:
    """Print an ASCII art plot of the step response."""
    # Downsample to fit width
    indices = np.linspace(0, len(values) - 1, width, dtype=int)
    sampled = values[indices]

    # Determine y-axis range
    y_min = min(np.min(sampled), setpoint - 1) - 1
    y_max = max(np.max(sampled), setpoint + 1) + 1

    print(f"\n  {label}")
    print(f"  Setpoint: {setpoint:.1f}")

    for row in range(height, -1, -1):
        y_val = y_min + (y_max - y_min) * row / height
        line = f"  {y_val:6.1f} |"
        for col in range(width):
            val = sampled[col]
            val_row = int((val - y_min) / (y_max - y_min) * height + 0.5)
            sp_row = int((setpoint - y_min) / (y_max - y_min) * height + 0.5)

            if val_row == row:
                line += "*"
            elif sp_row == row:
                line += "-"
            else:
                line += " "
        print(line)

    print(f"         +{''.join(['-'] * width)}")
    print(f"          0{' ' * (width // 2 - 1)}time{' ' * (width // 2 - 4)}{times[-1]:.0f}s")


def print_metrics(metrics: Dict[str, float], label: str) -> None:
    """Print step response metrics in a clean format."""
    print(f"\n  {label} - Metrics:")
    print(f"    Overshoot:          {metrics['overshoot_pct']:6.2f}%")
    print(f"    Settling time:      {metrics['settling_time']:6.2f}s")
    print(f"    Steady-state error: {metrics['steady_state_error']:6.4f}")
    print(f"    Rise time:          {metrics['rise_time']:6.2f}s")


# =============================================================================
# Main: Demonstrate Everything
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 006: PID CONTROLLER SIMULATION")
    print("=" * 70)

    # Parameters
    SETPOINT = 50.0     # target temperature
    INITIAL = 20.0      # starting temperature
    AMBIENT = 20.0      # ambient temperature
    TAU = 10.0          # plant time constant
    GAIN = 1.0          # plant gain
    DT = 0.05           # simulation timestep
    DURATION = 60.0     # simulation duration in seconds

    # --- Compare P, PI, PD, and PID controllers ---
    print("\n" + "=" * 70)
    print("COMPARISON: P vs PI vs PD vs PID CONTROLLERS")
    print("=" * 70)
    print(f"\nSystem: Thermal plant (tau={TAU}s, gain={GAIN})")
    print(f"Goal: Heat from {INITIAL}C to {SETPOINT}C")

    configs = {
        "P-only  (Kp=2.0)":                 {"kp": 2.0, "ki": 0.0,  "kd": 0.0},
        "PI      (Kp=2.0, Ki=0.1)":         {"kp": 2.0, "ki": 0.1,  "kd": 0.0},
        "PD      (Kp=2.0, Kd=1.0)":         {"kp": 2.0, "ki": 0.0,  "kd": 1.0},
        "PID     (Kp=2.0, Ki=0.1, Kd=1.0)": {"kp": 2.0, "ki": 0.1,  "kd": 1.0},
    }

    all_results = {}

    for label, gains in configs.items():
        controller = PIDController(
            kp=gains["kp"], ki=gains["ki"], kd=gains["kd"],
            setpoint=SETPOINT, dt=DT,
        )
        plant = ThermalPlant(
            initial_temp=INITIAL, ambient_temp=AMBIENT,
            tau=TAU, gain=GAIN, dt=DT,
        )

        times, temps, controls = simulate(controller, plant, DURATION, DT)
        metrics = analyze_step_response(times, temps, SETPOINT)
        all_results[label] = {"times": times, "temps": temps, "metrics": metrics}

        print_ascii_response(times, temps, SETPOINT, label)
        print_metrics(metrics, label)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY: CONTROLLER COMPARISON")
    print("=" * 70)

    print(f"\n  {'Controller':<35} {'Overshoot':>10} {'Settling':>10} {'SS Error':>10} {'Rise':>8}")
    print(f"  {'':<35} {'(%)':>10} {'(s)':>10} {'':>10} {'(s)':>8}")
    print("  " + "-" * 75)

    for label, result in all_results.items():
        m = result["metrics"]
        print(
            f"  {label:<35} {m['overshoot_pct']:>9.2f}% "
            f"{m['settling_time']:>9.2f}s "
            f"{m['steady_state_error']:>9.4f} "
            f"{m['rise_time']:>7.2f}s"
        )

    print("\n  Key observations:")
    print("  - P-only has significant steady-state error (it can't fully reach setpoint)")
    print("  - Adding I (integral) eliminates steady-state error but adds overshoot")
    print("  - Adding D (derivative) reduces overshoot and oscillation")
    print("  - Full PID combines all benefits: no SS error, less overshoot, smooth settling")

    # --- Effect of Kp tuning ---
    print("\n" + "=" * 70)
    print("TUNING: EFFECT OF Kp (Ki=0.1, Kd=0.5 fixed)")
    print("=" * 70)

    kp_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n  {'Kp':<8} {'Overshoot':>10} {'Settling':>10} {'SS Error':>10} {'Rise':>8}")
    print("  " + "-" * 48)

    for kp in kp_values:
        controller = PIDController(kp=kp, ki=0.1, kd=0.5, setpoint=SETPOINT, dt=DT)
        plant = ThermalPlant(initial_temp=INITIAL, ambient_temp=AMBIENT, tau=TAU, gain=GAIN, dt=DT)
        times, temps, controls = simulate(controller, plant, DURATION, DT)
        m = analyze_step_response(times, temps, SETPOINT)
        print(
            f"  {kp:<8.1f} {m['overshoot_pct']:>9.2f}% "
            f"{m['settling_time']:>9.2f}s "
            f"{m['steady_state_error']:>9.4f} "
            f"{m['rise_time']:>7.2f}s"
        )

    print("\n  Observation: Higher Kp gives faster rise but more overshoot.")
    print("  Too high and the system oscillates wildly.")

    # --- Effect of Ki tuning ---
    print("\n" + "=" * 70)
    print("TUNING: EFFECT OF Ki (Kp=2.0, Kd=0.5 fixed)")
    print("=" * 70)

    ki_values = [0.0, 0.05, 0.1, 0.5, 1.0]

    print(f"\n  {'Ki':<8} {'Overshoot':>10} {'Settling':>10} {'SS Error':>10} {'Rise':>8}")
    print("  " + "-" * 48)

    for ki in ki_values:
        controller = PIDController(kp=2.0, ki=ki, kd=0.5, setpoint=SETPOINT, dt=DT)
        plant = ThermalPlant(initial_temp=INITIAL, ambient_temp=AMBIENT, tau=TAU, gain=GAIN, dt=DT)
        times, temps, controls = simulate(controller, plant, DURATION, DT)
        m = analyze_step_response(times, temps, SETPOINT)
        print(
            f"  {ki:<8.2f} {m['overshoot_pct']:>9.2f}% "
            f"{m['settling_time']:>9.2f}s "
            f"{m['steady_state_error']:>9.4f} "
            f"{m['rise_time']:>7.2f}s"
        )

    print("\n  Observation: Ki eliminates steady-state error but too much causes oscillation.")

    # --- Effect of Kd tuning ---
    print("\n" + "=" * 70)
    print("TUNING: EFFECT OF Kd (Kp=2.0, Ki=0.1 fixed)")
    print("=" * 70)

    kd_values = [0.0, 0.5, 1.0, 3.0, 5.0]

    print(f"\n  {'Kd':<8} {'Overshoot':>10} {'Settling':>10} {'SS Error':>10} {'Rise':>8}")
    print("  " + "-" * 48)

    for kd in kd_values:
        controller = PIDController(kp=2.0, ki=0.1, kd=kd, setpoint=SETPOINT, dt=DT)
        plant = ThermalPlant(initial_temp=INITIAL, ambient_temp=AMBIENT, tau=TAU, gain=GAIN, dt=DT)
        times, temps, controls = simulate(controller, plant, DURATION, DT)
        m = analyze_step_response(times, temps, SETPOINT)
        print(
            f"  {kd:<8.1f} {m['overshoot_pct']:>9.2f}% "
            f"{m['settling_time']:>9.2f}s "
            f"{m['steady_state_error']:>9.4f} "
            f"{m['rise_time']:>7.2f}s"
        )

    print("\n  Observation: Kd damps overshoot but too much slows the response.")

    # --- Demonstrate reset ---
    print("\n" + "=" * 70)
    print("RESET DEMONSTRATION")
    print("=" * 70)

    controller = PIDController(kp=2.0, ki=0.1, kd=1.0, setpoint=SETPOINT, dt=DT)
    plant = ThermalPlant(initial_temp=INITIAL, ambient_temp=AMBIENT, tau=TAU, gain=GAIN, dt=DT)

    # Run for a while
    for _ in range(200):
        m = plant.temperature
        u = controller.update(m)
        plant.step(u)

    print(f"\n  After 200 steps: temp={plant.temperature:.2f}, integral={controller._integral:.4f}")
    print(f"  Resetting controller...")

    controller.reset()
    print(f"  After reset: integral={controller._integral:.4f}, prev_error={controller._prev_error}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("- P gives proportional response but leaves steady-state error")
    print("- I eliminates steady-state error by accumulating past mistakes")
    print("- D damps oscillation by reacting to the rate of change")
    print("- Tuning is a tradeoff: fast response vs stability vs accuracy")
    print("- In practice, start with P, add I to kill offset, add D to reduce overshoot")
