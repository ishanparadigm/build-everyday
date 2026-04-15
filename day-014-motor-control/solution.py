"""
Day 14: Motor Control Simulation

A complete DC motor simulator with PWM speed control and cascaded PID
position/velocity controllers. Models the coupled electrical-mechanical
dynamics using RK4 integration.

Builds on Day 6 (PID controller) by applying PID to a physically realistic
plant model.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import math


# =============================================================================
# Motor Model
# =============================================================================

@dataclass
class DCMotorParams:
    """Physical parameters for a small DC motor (e.g., 12V hobby motor).

    The defaults model a typical 12V brushed DC motor:
    - ~500 RPM no-load speed
    - ~0.5A no-load current
    - ~3A stall current
    """
    R: float = 4.0           # Armature resistance (ohms)
    L: float = 0.01          # Armature inductance (henries) — small but nonzero
    K_t: float = 0.023       # Torque constant (N*m/A) — equals K_e in SI
    K_e: float = 0.023       # Back-EMF constant (V/(rad/s))
    J: float = 0.0001        # Rotor moment of inertia (kg*m^2)
    B: float = 0.0001        # Viscous friction coefficient (N*m*s/rad)
    V_supply: float = 12.0   # Supply voltage (V)


@dataclass
class MotorState:
    """State vector of the DC motor.

    Three coupled state variables fully describe the motor:
    - current: drives torque production
    - omega: angular velocity, what we usually want to control
    - theta: angular position, needed for position control
    """
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
    """Compute the time derivatives of the motor state.

    This is the heart of the simulation — the coupled ODEs:

    Electrical: di/dt = (V - R*i - K_e*omega) / L
        The current changes based on applied voltage minus resistive drop
        minus back-EMF. With low inductance L, current responds quickly.

    Mechanical: d_omega/dt = (K_t*i - B*omega - T_load) / J
        Angular acceleration equals net torque / inertia.
        Torque comes from current, friction opposes motion, load resists.

    Kinematic: d_theta/dt = omega
        Angle is just the integral of velocity.
    """
    i, omega, theta = state

    # Electrical dynamics
    # Back-EMF opposes applied voltage — this is the natural speed regulation
    back_emf = params.K_e * omega
    di_dt = (voltage - params.R * i - back_emf) / params.L

    # Mechanical dynamics
    # Motor torque from current, minus friction and external load
    torque_motor = params.K_t * i
    torque_friction = params.B * omega
    domega_dt = (torque_motor - torque_friction - load_torque) / params.J

    # Kinematic relationship
    dtheta_dt = omega

    return [di_dt, domega_dt, dtheta_dt]


# =============================================================================
# Numerical Integration (RK4)
# =============================================================================

def rk4_step(
    deriv_func: Callable[[List[float]], List[float]],
    state: List[float],
    dt: float,
) -> List[float]:
    """Single step of 4th-order Runge-Kutta integration.

    RK4 evaluates the derivative at 4 points within the time step:
    - k1: derivative at start
    - k2: derivative at midpoint using k1
    - k3: derivative at midpoint using k2
    - k4: derivative at end using k3

    The weighted average (1/6, 1/3, 1/3, 1/6) cancels error terms up to O(dt^4).
    This is dramatically more accurate than Euler for the same step size, which
    matters for stiff systems like motors with small inductance.
    """
    n = len(state)

    k1 = deriv_func(state)

    s2 = [state[j] + 0.5 * dt * k1[j] for j in range(n)]
    k2 = deriv_func(s2)

    s3 = [state[j] + 0.5 * dt * k2[j] for j in range(n)]
    k3 = deriv_func(s3)

    s4 = [state[j] + dt * k3[j] for j in range(n)]
    k4 = deriv_func(s4)

    return [
        state[j] + (dt / 6.0) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])
        for j in range(n)
    ]


# =============================================================================
# PWM Signal
# =============================================================================

def pwm_voltage(
    duty_cycle: float,
    t: float,
    v_supply: float,
    pwm_freq: float = 10000.0,
) -> float:
    """Generate PWM voltage at time t.

    PWM rapidly switches between V_supply and 0. The motor's inductance
    smooths this into an effective average voltage = duty_cycle * V_supply.

    For simulation purposes, if the PWM frequency is much higher than the
    mechanical time constant (which it always is in practice), we can use
    the time-averaged voltage directly. This is more numerically stable
    and avoids requiring tiny time steps to resolve each PWM cycle.

    We provide both modes: actual switching (for visualization) and
    averaged (for efficient simulation).
    """
    duty_cycle = max(-1.0, min(1.0, duty_cycle))

    # Use averaged voltage for simulation stability
    # In a real system, the inductance does this filtering for us
    # Negative duty cycle means reverse voltage (H-bridge)
    return duty_cycle * v_supply


def pwm_voltage_switching(
    duty_cycle: float,
    t: float,
    v_supply: float,
    pwm_freq: float = 10000.0,
) -> float:
    """Actual switching PWM — useful for visualizing the real signal."""
    duty_cycle = max(0.0, min(1.0, duty_cycle))
    period = 1.0 / pwm_freq
    phase = (t % period) / period
    return v_supply if phase < duty_cycle else 0.0


# =============================================================================
# PID Controller
# =============================================================================

@dataclass
class PIDController:
    """Discrete PID controller with anti-windup and derivative filtering.

    Key improvements over a naive PID:

    1. Anti-windup: The integral term is clamped to prevent it from growing
       unboundedly when the output is saturated. Without this, after a large
       setpoint change, the integral "winds up" and causes massive overshoot
       as it slowly unwinds.

    2. Derivative on measurement: Instead of differentiating the error
       (which has discontinuities on setpoint changes), we differentiate
       the measurement. This avoids "derivative kick" — a spike in output
       whenever the setpoint changes.

    3. Output clamping: Ensures the output stays within valid bounds
       (e.g., [0, 1] for PWM duty cycle).
    """
    Kp: float = 0.0
    Kd: float = 0.0
    Ki: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    integral_max: float = 10.0

    # Internal state — tracked automatically
    _integral: float = field(default=0.0, repr=False)
    _prev_measurement: float = field(default=0.0, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def reset(self) -> None:
        """Reset controller state. Call when switching modes or setpoints."""
        self._integral = 0.0
        self._prev_measurement = 0.0
        self._initialized = False

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Compute PID output for one time step.

        Args:
            setpoint: desired value
            measurement: current measured value
            dt: time step (seconds)

        Returns:
            Control output (clamped to [output_min, output_max])
        """
        error = setpoint - measurement

        # Proportional: directly proportional to current error
        p_term = self.Kp * error

        # Integral: accumulates past error to eliminate steady-state offset
        self._integral += error * dt
        # Anti-windup: clamp integral to prevent unbounded growth
        self._integral = max(-self.integral_max, min(self.integral_max, self._integral))
        i_term = self.Ki * self._integral

        # Derivative: uses measurement instead of error to avoid derivative kick
        if not self._initialized:
            self._prev_measurement = measurement
            self._initialized = True
        d_measurement = (measurement - self._prev_measurement) / dt if dt > 0 else 0.0
        self._prev_measurement = measurement
        # Negative sign because we differentiate measurement, not error
        # If measurement is increasing (moving toward setpoint), we want less output
        d_term = -self.Kd * d_measurement

        output = p_term + i_term + d_term
        return max(self.output_min, min(self.output_max, output))


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
    """Simulates a DC motor with optional PID control.

    Usage:
        sim = MotorSimulator(params)
        result = sim.run_open_loop(voltage=6.0, duration=0.5)
        result = sim.run_velocity_control(target_omega=50.0, duration=1.0)
        result = sim.run_position_control(target_theta=3.14, duration=2.0)
    """

    def __init__(self, params: Optional[DCMotorParams] = None):
        self.params = params or DCMotorParams()
        self.state = MotorState()

    def reset(self) -> None:
        """Reset motor to stopped state."""
        self.state = MotorState()

    def _step(self, voltage: float, load_torque: float, dt: float) -> None:
        """Advance motor state by one time step using RK4."""
        # Create a closure over voltage and load torque for the derivative function
        def deriv(s: List[float]) -> List[float]:
            return motor_derivatives(s, voltage, load_torque, self.params)

        new_state = rk4_step(deriv, self.state.to_list(), dt)
        self.state = MotorState.from_list(new_state)

        # Physical constraint: current can't be negative with single-quadrant drive
        # (In a full H-bridge, negative current enables regenerative braking)
        # We allow it here for generality

    def run_open_loop(
        self,
        voltage: float = 12.0,
        duration: float = 0.5,
        dt: float = 0.0001,
        load_torque: float = 0.0,
        load_start_time: float = -1.0,
    ) -> SimResult:
        """Run motor with constant voltage (no feedback control).

        This demonstrates the motor's natural dynamics:
        - Current rises quickly (limited by L and R)
        - Velocity rises slowly (limited by J)
        - Both settle to steady state when back-EMF balances applied voltage

        Args:
            voltage: applied DC voltage
            duration: simulation time (seconds)
            dt: time step (seconds)
            load_torque: external disturbance torque
            load_start_time: when to apply load (-1 = start)
        """
        self.reset()
        result = SimResult()
        steps = int(duration / dt)

        for step_i in range(steps):
            t = step_i * dt
            current_load = load_torque if (load_start_time < 0 or t >= load_start_time) else 0.0

            result.time.append(t)
            result.current.append(self.state.current)
            result.omega.append(self.state.omega)
            result.theta.append(self.state.theta)
            result.voltage.append(voltage)

            self._step(voltage, current_load, dt)

        return result

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

        The PID controller outputs a duty cycle [0, 1] which is converted
        to voltage via PWM. The controller measures angular velocity and
        adjusts PWM to maintain the target speed.

        Default PID gains are tuned for the default motor parameters.
        """
        self.reset()
        result = SimResult()

        if pid is None:
            # Tuned gains for default motor params
            # Kp=0.01: 1 rad/s error -> 1% duty cycle change (gentle)
            # Ki=0.05: accumulated error drives steady-state to zero
            # Kd=0.0001: damps oscillations
            pid = PIDController(
                Kp=0.01, Ki=0.05, Kd=0.0001,
                output_min=0.0, output_max=1.0,
                integral_max=20.0,
            )
        pid.reset()

        steps = int(duration / dt)

        for step_i in range(steps):
            t = step_i * dt
            current_load = load_torque if (load_start_time < 0 or t >= load_start_time) else 0.0

            # PID computes duty cycle from velocity error
            duty = pid.update(target_omega, self.state.omega, dt)
            voltage = pwm_voltage(duty, t, self.params.V_supply)

            result.time.append(t)
            result.current.append(self.state.current)
            result.omega.append(self.state.omega)
            result.theta.append(self.state.theta)
            result.voltage.append(voltage)
            result.duty_cycle.append(duty)
            result.setpoint.append(target_omega)

            self._step(voltage, current_load, dt)

        return result

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

        The outer loop (position PID) outputs a desired velocity.
        The inner loop (velocity PID) outputs a duty cycle.

        This cascade structure is standard in robotics because:
        1. Inner loop handles fast disturbances (load changes)
        2. Outer loop handles slower position tracking
        3. We can limit velocity independently
        4. Each loop can be tuned independently

        The outer loop MUST be slower than the inner loop (lower bandwidth).
        Rule of thumb: outer loop bandwidth = inner / 3 to inner / 5.
        """
        self.reset()
        result = SimResult()

        if pos_pid is None:
            # Position loop: outputs velocity setpoint
            # Moderate gains — outer loop must be slower than inner loop
            pos_pid = PIDController(
                Kp=15.0, Ki=1.0, Kd=2.0,
                output_min=-max_velocity, output_max=max_velocity,
                integral_max=20.0,
            )
        if vel_pid is None:
            # Velocity loop: bidirectional for position control (H-bridge)
            # Negative duty = reverse voltage for braking/reverse motion
            vel_pid = PIDController(
                Kp=0.01, Ki=0.05, Kd=0.0001,
                output_min=-1.0, output_max=1.0,
                integral_max=20.0,
            )
        pos_pid.reset()
        vel_pid.reset()

        steps = int(duration / dt)

        for step_i in range(steps):
            t = step_i * dt

            # Outer loop: position error -> velocity setpoint
            vel_setpoint = pos_pid.update(target_theta, self.state.theta, dt)

            # Inner loop: velocity error -> PWM duty cycle
            duty = vel_pid.update(vel_setpoint, self.state.omega, dt)
            voltage = pwm_voltage(duty, t, self.params.V_supply)

            result.time.append(t)
            result.current.append(self.state.current)
            result.omega.append(self.state.omega)
            result.theta.append(self.state.theta)
            result.voltage.append(voltage)
            result.duty_cycle.append(duty)
            result.setpoint.append(target_theta)

            self._step(voltage, 0.0, dt)

        return result


# =============================================================================
# Analysis Helpers
# =============================================================================

def compute_steady_state(params: DCMotorParams, voltage: float) -> dict:
    """Analytically compute the steady-state operating point.

    At steady state, all derivatives are zero:
    - di/dt = 0  =>  V = R*i + K_e*omega
    - d_omega/dt = 0  =>  K_t*i = B*omega

    Solving simultaneously:
    - omega_ss = (K_t * V) / (K_t * K_e + R * B)
    - i_ss = (B * V) / (K_t * K_e + R * B)
    """
    denom = params.K_t * params.K_e + params.R * params.B
    omega_ss = (params.K_t * voltage) / denom
    i_ss = (params.B * voltage) / denom
    torque_ss = params.K_t * i_ss
    power_mech = torque_ss * omega_ss
    power_elec = voltage * i_ss
    efficiency = power_mech / power_elec if power_elec > 0 else 0.0

    return {
        "omega_ss": omega_ss,
        "i_ss": i_ss,
        "torque_ss": torque_ss,
        "rpm_ss": omega_ss * 60 / (2 * math.pi),
        "power_mech_W": power_mech,
        "power_elec_W": power_elec,
        "efficiency": efficiency,
    }


def find_settling_time(
    times: List[float],
    values: List[float],
    target: float,
    tolerance: float = 0.02,
) -> float:
    """Find the time at which the signal settles within tolerance of target.

    Settling time is measured as the last time the signal exits the
    tolerance band. This is the standard control systems definition.
    """
    band_low = target * (1 - tolerance)
    band_high = target * (1 + tolerance)

    settling_time = times[-1]  # default: never settled
    # Search backward for last excursion outside the band
    for i in range(len(values) - 1, -1, -1):
        if values[i] < band_low or values[i] > band_high:
            settling_time = times[i] if i < len(times) - 1 else times[-1]
            break
    else:
        settling_time = 0.0  # always within band

    return settling_time


def find_rise_time(
    times: List[float],
    values: List[float],
    target: float,
) -> float:
    """Find 10%-90% rise time — time to go from 10% to 90% of target."""
    t10 = None
    t90 = None
    for i, v in enumerate(values):
        if t10 is None and v >= 0.1 * target:
            t10 = times[i]
        if t90 is None and v >= 0.9 * target:
            t90 = times[i]
            break
    if t10 is not None and t90 is not None:
        return t90 - t10
    return float("inf")


def find_overshoot(values: List[float], target: float) -> float:
    """Find peak overshoot as a percentage of target."""
    peak = max(values)
    if target == 0:
        return 0.0
    overshoot = (peak - target) / target * 100.0
    return max(0.0, overshoot)


# =============================================================================
# Main — Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 14: DC MOTOR CONTROL SIMULATION")
    print("=" * 70)

    params = DCMotorParams()
    sim = MotorSimulator(params)

    # ---- Analytical Steady State ----
    print("\n--- Analytical Steady-State (12V, no load) ---")
    ss = compute_steady_state(params, 12.0)
    print(f"  Speed:      {ss['omega_ss']:.1f} rad/s ({ss['rpm_ss']:.0f} RPM)")
    print(f"  Current:    {ss['i_ss']:.4f} A")
    print(f"  Torque:     {ss['torque_ss']:.6f} N*m")
    print(f"  Efficiency: {ss['efficiency']:.1%}")

    # ---- Open-Loop Step Response ----
    print("\n--- Open-Loop Step Response (12V) ---")
    result_ol = sim.run_open_loop(voltage=12.0, duration=1.0, dt=0.0001)

    # Sample at key time points
    sample_times = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    print(f"  {'Time (ms)':>10}  {'Current (A)':>12}  {'Speed (rad/s)':>14}  {'Speed (RPM)':>12}")
    for target_t in sample_times:
        idx = min(int(target_t / 0.0001), len(result_ol.time) - 1)
        t = result_ol.time[idx]
        i = result_ol.current[idx]
        w = result_ol.omega[idx]
        rpm = w * 60 / (2 * math.pi)
        print(f"  {t*1000:>10.1f}  {i:>12.4f}  {w:>14.2f}  {rpm:>12.1f}")

    final_omega = result_ol.omega[-1]
    print(f"\n  Final speed: {final_omega:.2f} rad/s "
          f"(analytical: {ss['omega_ss']:.2f} rad/s)")
    rise_t = find_rise_time(result_ol.time, result_ol.omega, ss['omega_ss'])
    print(f"  Rise time (10-90%): {rise_t*1000:.2f} ms")

    # ---- Velocity Control ----
    print("\n--- Velocity Control (target = 50 rad/s) ---")
    target_vel = 50.0
    result_vc = sim.run_velocity_control(
        target_omega=target_vel, duration=0.5, dt=0.0001
    )

    print(f"  {'Time (ms)':>10}  {'Speed (rad/s)':>14}  {'Duty Cycle':>11}  {'Error (rad/s)':>14}")
    sample_times_vc = [1, 5, 10, 50, 100, 200, 500]
    for target_t_ms in sample_times_vc:
        idx = min(int(target_t_ms / (0.0001 * 1000)), len(result_vc.time) - 1)
        t = result_vc.time[idx]
        w = result_vc.omega[idx]
        d = result_vc.duty_cycle[idx]
        err = target_vel - w
        print(f"  {t*1000:>10.1f}  {w:>14.2f}  {d:>11.4f}  {err:>14.4f}")

    settling = find_settling_time(result_vc.time, result_vc.omega, target_vel)
    overshoot = find_overshoot(result_vc.omega, target_vel)
    print(f"\n  Settling time (2%): {settling*1000:.1f} ms")
    print(f"  Overshoot: {overshoot:.1f}%")
    print(f"  Final speed: {result_vc.omega[-1]:.4f} rad/s (target: {target_vel})")

    # ---- Velocity Control with Load Disturbance ----
    print("\n--- Velocity Control with Load Disturbance ---")
    print("  Applying 0.002 N*m load torque at t=250ms")
    result_vcd = sim.run_velocity_control(
        target_omega=target_vel, duration=0.5, dt=0.0001,
        load_torque=0.002, load_start_time=0.25,
    )

    # Show response around disturbance
    sample_times_d = [240, 250, 260, 280, 300, 350, 500]
    print(f"  {'Time (ms)':>10}  {'Speed (rad/s)':>14}  {'Duty Cycle':>11}")
    for target_t_ms in sample_times_d:
        idx = min(int(target_t_ms / (0.0001 * 1000)), len(result_vcd.time) - 1)
        w = result_vcd.omega[idx]
        d = result_vcd.duty_cycle[idx]
        marker = " <-- load applied" if target_t_ms == 250 else ""
        print(f"  {target_t_ms:>10}  {w:>14.4f}  {d:>11.4f}{marker}")

    print(f"\n  Speed recovered to: {result_vcd.omega[-1]:.4f} rad/s")
    print(f"  Controller compensated by increasing duty cycle")

    # ---- Cascaded Position Control ----
    print("\n--- Cascaded Position Control (target = pi rad = 180 deg) ---")
    target_pos = math.pi
    result_pc = sim.run_position_control(
        target_theta=target_pos, duration=1.5, dt=0.0001
    )

    print(f"  {'Time (ms)':>10}  {'Angle (rad)':>12}  {'Angle (deg)':>12}  "
          f"{'Speed (rad/s)':>14}  {'Duty':>6}")
    sample_times_pc = [10, 50, 100, 200, 500, 1000, 1500]
    for target_t_ms in sample_times_pc:
        idx = min(int(target_t_ms / (0.0001 * 1000)), len(result_pc.time) - 1)
        th = result_pc.theta[idx]
        w = result_pc.omega[idx]
        d = result_pc.duty_cycle[idx]
        print(f"  {target_t_ms:>10}  {th:>12.4f}  {math.degrees(th):>12.2f}  "
              f"{w:>14.2f}  {d:>6.3f}")

    final_theta = result_pc.theta[-1]
    final_error_deg = math.degrees(abs(target_pos - final_theta))
    settling_pos = find_settling_time(result_pc.time, result_pc.theta, target_pos)
    overshoot_pos = find_overshoot(result_pc.theta, target_pos)
    print(f"\n  Final position: {math.degrees(final_theta):.2f} deg "
          f"(target: {math.degrees(target_pos):.2f} deg)")
    print(f"  Position error: {final_error_deg:.4f} deg")
    print(f"  Settling time (2%): {settling_pos*1000:.1f} ms")
    print(f"  Overshoot: {overshoot_pos:.1f}%")

    # ---- PWM Comparison ----
    print("\n--- PWM: Averaged vs Switching ---")
    print("  Demonstrating that averaged PWM matches switching PWM")
    duty = 0.5
    freq = 10000.0
    # Sample the switching waveform over one period
    period = 1.0 / freq
    n_samples = 20
    avg_v = 0.0
    print(f"  Duty cycle: {duty}, Frequency: {freq/1000:.0f} kHz, V_supply: {params.V_supply}V")
    for i in range(n_samples):
        t_sample = i * period / n_samples
        v_switch = pwm_voltage_switching(duty, t_sample, params.V_supply, freq)
        avg_v += v_switch
    avg_v /= n_samples
    v_averaged = pwm_voltage(duty, 0.0, params.V_supply, freq)
    print(f"  Mean of switching waveform: {avg_v:.1f}V")
    print(f"  Averaged model output:      {v_averaged:.1f}V")
    print(f"  Expected: {duty * params.V_supply:.1f}V")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. DC motor dynamics are coupled electrical + mechanical ODEs")
    print("  2. Back-EMF provides natural speed regulation (negative feedback)")
    print("  3. PID velocity control eliminates steady-state error from loads")
    print("  4. Cascaded position control separates fast/slow dynamics")
    print("  5. PWM provides efficient voltage control via duty cycle")
