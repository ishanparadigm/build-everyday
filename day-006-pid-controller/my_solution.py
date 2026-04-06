"""
Day 006: PID Controller Simulation — Your Implementation

Implement a PID controller and simulate it controlling a thermal system.

Key equations:
    error = setpoint - measurement
    P = Kp * error
    I = Ki * sum(errors) * dt
    D = Kd * (error - prev_error) / dt
    output = P + I + D
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional


class PIDController:
    """
    Discrete-time PID controller.

    TODO: Implement the three-term PID control law:
        u[n] = Kp * e[n] + Ki * sum(e[0..n]) * dt + Kd * (e[n] - e[n-1]) / dt

    Hints:
    - Track _integral (running sum of error * dt) and _prev_error
    - On the first call, derivative should be 0 (no previous error yet)
    - Optional: implement output_limits for anti-windup clamping
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
        self.output_limits = output_limits

        # TODO: Initialize internal state
        # Hint: you need _integral (float) and _prev_error (Optional[float])
        raise NotImplementedError

    def update(self, measurement: float) -> float:
        """
        Compute PID output given the current measurement.

        Steps:
        1. error = setpoint - measurement
        2. P term = Kp * error
        3. Update integral: _integral += error * dt; I term = Ki * _integral
        4. Derivative: if prev_error exists, d = (error - prev_error) / dt, else 0
           D term = Kd * derivative
        5. Store current error as prev_error
        6. output = P + I + D
        7. (Optional) Clamp output to output_limits, undo integral if clamped

        Returns: control output (float)
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal state: set _integral to 0 and _prev_error to None."""
        raise NotImplementedError


class ThermalPlant:
    """
    First-order thermal system: dT/dt = (1/tau) * (K * u - (T - T_ambient))

    TODO: Implement Euler integration to advance the temperature by one timestep.

    Hints:
    - dT = (1/tau) * (gain * control_input - (temperature - ambient_temp))
    - temperature += dT * dt
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
        Advance plant by one timestep, return (possibly noisy) temperature.

        Hint: Use Euler integration: T_new = T_old + dT/dt * dt
        """
        raise NotImplementedError

    def reset(self, initial_temp: float = 20.0) -> None:
        """Reset plant temperature."""
        raise NotImplementedError


def simulate(
    controller: PIDController,
    plant: ThermalPlant,
    duration: float,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a closed-loop simulation.

    TODO: For each timestep:
        1. Read plant temperature
        2. Pass to controller.update() to get control output
        3. Apply control output to plant.step()
        4. Record temperature and control output

    Returns: (times, temperatures, control_outputs)
    """
    raise NotImplementedError


def analyze_step_response(
    times: np.ndarray,
    values: np.ndarray,
    setpoint: float,
    tolerance: float = 0.02,
) -> Dict[str, float]:
    """
    Analyze step response for overshoot, settling time, steady-state error, rise time.

    TODO:
    - Overshoot: max percentage the signal exceeds the setpoint (relative to step size)
    - Settling time: last time the signal is outside the tolerance band
    - Steady-state error: average error over the last 10% of the simulation
    - Rise time: time from 10% to 90% of the step change

    Returns: dict with keys 'overshoot_pct', 'settling_time', 'steady_state_error', 'rise_time'
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("Implement the PIDController, ThermalPlant, simulate, and analyze_step_response.")
    print("Then run: python3 tests.py")
