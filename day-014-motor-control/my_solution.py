"""
Day 14: Motor Control Simulation — Your Implementation

Build a DC motor simulator from first principles:
1. Model the coupled electrical-mechanical dynamics
2. Implement RK4 numerical integration
3. Add PWM voltage control
4. Build PID velocity controller
5. Build cascaded position controller

Hints:
- The motor has 3 state variables: current, angular velocity, angle
- The key physics: V = R*i + L*di/dt + K_e*omega (electrical)
                    J*d_omega/dt = K_t*i - B*omega - T_load (mechanical)
- RK4 uses 4 derivative evaluations per step for O(dt^4) accuracy
- PID anti-windup prevents integral term from growing unboundedly
- Cascaded control: outer loop (position) feeds inner loop (velocity)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import math


# =============================================================================
# Motor Model
# =============================================================================

@dataclass
class DCMotorParams:
    """Physical parameters for a small DC motor.

    Defaults model a typical 12V brushed DC motor.
    """
    R: float = 4.0           # Armature resistance (ohms)
    L: float = 0.01          # Armature inductance (henries)
    K_t: float = 0.023       # Torque constant (N*m/A)
    K_e: float = 0.023       # Back-EMF constant (V/(rad/s))
    J: float = 0.0001        # Rotor moment of inertia (kg*m^2)
    B: float = 0.0001        # Viscous friction coefficient (N*m*s/rad)
    V_supply: float = 12.0   # Supply voltage (V)


@dataclass
class MotorState:
    """State vector of the DC motor."""
    current: float = 0.0     # Armature current (A)
    omega: float = 0.0       # Angular velocity (rad/s)
    theta: float = 0.0       # Angular position (rad)

    def to_list(self) -> List[float]:
        return [self.current, self.omega, self.theta]

    @staticmethod
    def from_list(vals: List[float]) -> "MotorState":
        return MotorState(current=vals[0], omega=vals[1], theta=vals[2])


def motor_derivatives(
    state: List[float],
    voltage: float,
    load_torque: float,
    params: DCMotorParams,
) -> List[float]:
    """Compute time derivatives of the motor state [di/dt, d_omega/dt, d_theta/dt].

    Hint: Apply Kirchhoff's voltage law for electrical, Newton's 2nd law
    (rotational) for mechanical, and omega = d_theta/dt for kinematics.
    """
    raise NotImplementedError("TODO: implement motor dynamics equations")


# =============================================================================
# Numerical Integration (RK4)
# =============================================================================

def rk4_step(
    deriv_func: Callable[[List[float]], List[float]],
    state: List[float],
    dt: float,
) -> List[float]:
    """Single step of 4th-order Runge-Kutta integration.

    Hint: Evaluate derivative at 4 points (start, two midpoints, end)
    and combine with weights (1/6, 1/3, 1/3, 1/6).

    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    new_state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    """
    raise NotImplementedError("TODO: implement RK4 integration step")


# =============================================================================
# PWM Signal
# =============================================================================

def pwm_voltage(
    duty_cycle: float,
    t: float,
    v_supply: float,
    pwm_freq: float = 10000.0,
) -> float:
    """Generate averaged PWM voltage.

    Hint: For simulation, the motor's inductance filters the PWM switching.
    The effective voltage is simply duty_cycle * v_supply.
    Clamp duty_cycle to [-1, 1] (negative = reverse via H-bridge).
    """
    raise NotImplementedError("TODO: implement PWM voltage averaging")


# =============================================================================
# PID Controller
# =============================================================================

@dataclass
class PIDController:
    """Discrete PID controller with anti-windup and derivative filtering.

    Key features to implement:
    - Anti-windup: clamp integral to [-integral_max, integral_max]
    - Derivative on measurement (not error) to avoid derivative kick
    - Output clamping to [output_min, output_max]
    """
    Kp: float = 0.0
    Kd: float = 0.0
    Ki: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    integral_max: float = 10.0

    _integral: float = field(default=0.0, repr=False)
    _prev_measurement: float = field(default=0.0, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._prev_measurement = 0.0
        self._initialized = False

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Compute PID output for one time step.

        Hint:
        - P term: Kp * error
        - I term: Ki * integral(error * dt), clamped
        - D term: -Kd * d(measurement)/dt  (negative because derivative on measurement)
        - Clamp final output to [output_min, output_max]
        """
        raise NotImplementedError("TODO: implement PID update")


# =============================================================================
# Motor Simulator
# =============================================================================

@dataclass
class SimResult:
    """Container for simulation results."""
    time: List[float] = field(default_factory=list)
    current: List[float] = field(default_factory=list)
    omega: List[float] = field(default_factory=list)
    theta: List[float] = field(default_factory=list)
    voltage: List[float] = field(default_factory=list)
    duty_cycle: List[float] = field(default_factory=list)
    setpoint: List[float] = field(default_factory=list)


class MotorSimulator:
    """Simulates a DC motor with optional PID control."""

    def __init__(self, params: Optional[DCMotorParams] = None):
        self.params = params or DCMotorParams()
        self.state = MotorState()

    def reset(self) -> None:
        self.state = MotorState()

    def _step(self, voltage: float, load_torque: float, dt: float) -> None:
        """Advance motor state by one time step using RK4.

        Hint: Create a closure that calls motor_derivatives with the
        current voltage and load, then pass it to rk4_step.
        """
        raise NotImplementedError("TODO: implement simulation step")

    def run_open_loop(
        self,
        voltage: float = 12.0,
        duration: float = 0.5,
        dt: float = 0.0001,
        load_torque: float = 0.0,
        load_start_time: float = -1.0,
    ) -> SimResult:
        """Run motor with constant voltage (no feedback control).

        Hint: For each time step, record state, then call self._step().
        Handle load_start_time to apply load only after that time.
        """
        raise NotImplementedError("TODO: implement open-loop simulation")

    def run_velocity_control(
        self,
        target_omega: float = 50.0,
        duration: float = 1.0,
        dt: float = 0.0001,
        pid: Optional[PIDController] = None,
        load_torque: float = 0.0,
        load_start_time: float = -1.0,
    ) -> SimResult:
        """Run motor with PID velocity control.

        Hint: Each step, use PID to compute duty cycle from velocity error,
        convert to voltage via pwm_voltage, then step the motor.
        """
        raise NotImplementedError("TODO: implement velocity control simulation")

    def run_position_control(
        self,
        target_theta: float = math.pi,
        duration: float = 2.0,
        dt: float = 0.0001,
        pos_pid: Optional[PIDController] = None,
        vel_pid: Optional[PIDController] = None,
        max_velocity: float = 100.0,
    ) -> SimResult:
        """Run motor with cascaded position -> velocity control.

        Hint: Outer PID (position) outputs velocity setpoint.
        Inner PID (velocity) outputs duty cycle.
        This is two PID controllers running each step.
        """
        raise NotImplementedError("TODO: implement cascaded position control")


# =============================================================================
# Analysis Helpers
# =============================================================================

def compute_steady_state(params: DCMotorParams, voltage: float) -> dict:
    """Analytically compute the steady-state operating point.

    Hint: At steady state, all derivatives = 0. Solve the two equations:
    V = R*i + K_e*omega  and  K_t*i = B*omega  simultaneously.
    """
    raise NotImplementedError("TODO: implement steady-state analysis")


def find_settling_time(
    times: List[float],
    values: List[float],
    target: float,
    tolerance: float = 0.02,
) -> float:
    """Find the time at which signal settles within tolerance of target.

    Hint: Search backward from the end for the last time the signal
    exits the tolerance band [target*(1-tol), target*(1+tol)].
    """
    raise NotImplementedError("TODO: implement settling time calculation")


def find_rise_time(
    times: List[float],
    values: List[float],
    target: float,
) -> float:
    """Find 10%-90% rise time.

    Hint: Find first time value >= 10% of target, then first time >= 90%.
    """
    raise NotImplementedError("TODO: implement rise time calculation")


def find_overshoot(values: List[float], target: float) -> float:
    """Find peak overshoot as percentage of target.

    Hint: overshoot = (max(values) - target) / target * 100
    """
    raise NotImplementedError("TODO: implement overshoot calculation")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    print("Testing your motor control implementation...\n")

    params = DCMotorParams()
    sim = MotorSimulator(params)

    # Test 1: Steady-state analysis
    print("Test 1: Analytical steady state at 12V")
    ss = compute_steady_state(params, 12.0)
    print(f"  Speed: {ss['omega_ss']:.1f} rad/s, Current: {ss['i_ss']:.4f} A")

    # Test 2: Open-loop step response
    print("\nTest 2: Open-loop step response")
    result = sim.run_open_loop(voltage=12.0, duration=0.3)
    print(f"  Final speed: {result.omega[-1]:.2f} rad/s")
    print(f"  Should approach: {ss['omega_ss']:.2f} rad/s")

    # Test 3: Velocity control
    print("\nTest 3: Velocity control (target=50 rad/s)")
    result = sim.run_velocity_control(target_omega=50.0, duration=0.5)
    print(f"  Final speed: {result.omega[-1]:.2f} rad/s")

    # Test 4: Position control
    print("\nTest 4: Position control (target=pi rad)")
    result = sim.run_position_control(target_theta=math.pi, duration=1.5)
    print(f"  Final angle: {math.degrees(result.theta[-1]):.2f} deg (target: 180.00 deg)")

    print("\nAll tests passed!" if True else "")
