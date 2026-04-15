"""
Tests for Day 14: Motor Control Simulation

Run with: python3 -m pytest tests.py -v
     or: python3 tests.py
"""

import unittest
import math
from my_solution import (
    DCMotorParams,
    MotorState,
    motor_derivatives,
    rk4_step,
    pwm_voltage,
    PIDController,
    MotorSimulator,
    SimResult,
    compute_steady_state,
    find_settling_time,
    find_rise_time,
    find_overshoot,
)


class TestMotorDerivatives(unittest.TestCase):
    """Test the motor dynamics equations."""

    def setUp(self):
        self.params = DCMotorParams()

    def test_stall_condition(self):
        """At stall (omega=0), current should increase (di/dt > 0) with applied voltage."""
        state = [0.0, 0.0, 0.0]  # i=0, omega=0, theta=0
        derivs = motor_derivatives(state, voltage=12.0, load_torque=0.0, params=self.params)
        # di/dt = (V - R*i - K_e*omega) / L = 12 / 0.01 = 1200
        self.assertAlmostEqual(derivs[0], 12.0 / self.params.L, places=1)
        # d_omega/dt = 0 since current is 0
        self.assertAlmostEqual(derivs[1], 0.0, places=5)
        # d_theta/dt = omega = 0
        self.assertAlmostEqual(derivs[2], 0.0, places=5)

    def test_steady_state_derivatives_near_zero(self):
        """At the analytical steady state, derivatives should be ~0."""
        ss = compute_steady_state(self.params, 12.0)
        state = [ss['i_ss'], ss['omega_ss'], 0.0]
        derivs = motor_derivatives(state, voltage=12.0, load_torque=0.0, params=self.params)
        self.assertAlmostEqual(derivs[0], 0.0, places=1)
        self.assertAlmostEqual(derivs[1], 0.0, places=1)

    def test_back_emf_reduces_current_growth(self):
        """Higher omega should reduce di/dt due to back-EMF."""
        state_slow = [0.0, 10.0, 0.0]
        state_fast = [0.0, 100.0, 0.0]
        derivs_slow = motor_derivatives(state_slow, 12.0, 0.0, self.params)
        derivs_fast = motor_derivatives(state_fast, 12.0, 0.0, self.params)
        # di/dt should be smaller (or negative) at higher speed
        self.assertGreater(derivs_slow[0], derivs_fast[0])


class TestRK4(unittest.TestCase):
    """Test the RK4 integrator."""

    def test_constant_derivative(self):
        """With constant derivatives, RK4 should give exact linear result."""
        # dx/dt = 2, dy/dt = -1
        deriv = lambda s: [2.0, -1.0]
        state = [0.0, 10.0]
        new_state = rk4_step(deriv, state, dt=0.1)
        self.assertAlmostEqual(new_state[0], 0.2, places=10)
        self.assertAlmostEqual(new_state[1], 9.9, places=10)

    def test_quadratic_accuracy(self):
        """RK4 should handle x' = t (quadratic solution) exactly."""
        # Approximate by making derivative depend on state: x' = 2*x, x(0) = 1
        # Solution: x(t) = e^(2t)
        deriv = lambda s: [2.0 * s[0]]
        state = [1.0]
        dt = 0.01
        for _ in range(100):  # 100 steps of 0.01 = t=1.0
            state = rk4_step(deriv, state, dt)
        expected = math.exp(2.0)  # e^2 ≈ 7.389
        self.assertAlmostEqual(state[0], expected, places=4)


class TestPWM(unittest.TestCase):
    """Test PWM voltage generation."""

    def test_full_duty(self):
        """100% duty cycle should give full supply voltage."""
        v = pwm_voltage(1.0, 0.0, 12.0)
        self.assertAlmostEqual(v, 12.0, places=5)

    def test_zero_duty(self):
        """0% duty cycle should give zero voltage."""
        v = pwm_voltage(0.0, 0.0, 12.0)
        self.assertAlmostEqual(v, 0.0, places=5)

    def test_half_duty(self):
        """50% duty cycle should give half supply voltage."""
        v = pwm_voltage(0.5, 0.0, 12.0)
        self.assertAlmostEqual(v, 6.0, places=5)

    def test_clamping(self):
        """Duty cycle should be clamped to [0, 1]."""
        v_over = pwm_voltage(1.5, 0.0, 12.0)
        v_under = pwm_voltage(-0.5, 0.0, 12.0)
        self.assertAlmostEqual(v_over, 12.0, places=5)
        self.assertAlmostEqual(v_under, 0.0, places=5)


class TestPIDController(unittest.TestCase):
    """Test PID controller behavior."""

    def test_proportional_only(self):
        """P-only controller output should be proportional to error."""
        pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0, output_min=-100, output_max=100)
        pid.reset()
        output = pid.update(setpoint=10.0, measurement=6.0, dt=0.01)
        self.assertAlmostEqual(output, 8.0, places=3)  # 2.0 * (10-6)

    def test_integral_accumulation(self):
        """Integral should accumulate over time to reduce steady-state error."""
        pid = PIDController(Kp=0.0, Ki=10.0, Kd=0.0, output_min=-100, output_max=100)
        pid.reset()
        # Apply constant error of 1.0 for 10 steps of dt=0.1
        for _ in range(10):
            output = pid.update(setpoint=1.0, measurement=0.0, dt=0.1)
        # Integral = 10 * 1.0 * 0.1 = 1.0, Ki * integral = 10.0
        self.assertAlmostEqual(output, 10.0, places=3)

    def test_output_clamping(self):
        """Output should be clamped to [output_min, output_max]."""
        pid = PIDController(Kp=100.0, Ki=0.0, Kd=0.0, output_min=0.0, output_max=1.0)
        pid.reset()
        output = pid.update(setpoint=100.0, measurement=0.0, dt=0.01)
        self.assertEqual(output, 1.0)

    def test_anti_windup(self):
        """Integral should be clamped to prevent windup."""
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0,
                           output_min=-100, output_max=100, integral_max=5.0)
        pid.reset()
        # Apply large error for many steps
        for _ in range(1000):
            pid.update(setpoint=100.0, measurement=0.0, dt=0.1)
        # Integral should be clamped to integral_max
        output = pid.update(setpoint=100.0, measurement=0.0, dt=0.1)
        self.assertLessEqual(output, 5.0 * 1.0 + 1e-6)  # Ki * integral_max


class TestMotorSimulator(unittest.TestCase):
    """Test the full motor simulator."""

    def setUp(self):
        self.params = DCMotorParams()
        self.sim = MotorSimulator(self.params)

    def test_open_loop_reaches_steady_state(self):
        """Open-loop should approach analytical steady-state speed."""
        ss = compute_steady_state(self.params, 12.0)
        result = self.sim.run_open_loop(voltage=12.0, duration=0.5)
        # Should be within 1% of analytical steady state
        self.assertAlmostEqual(
            result.omega[-1], ss['omega_ss'], delta=ss['omega_ss'] * 0.01
        )

    def test_velocity_control_tracks_setpoint(self):
        """Velocity control should reach and maintain target speed."""
        result = self.sim.run_velocity_control(target_omega=50.0, duration=0.5)
        # Final speed should be within 2% of target
        self.assertAlmostEqual(result.omega[-1], 50.0, delta=1.0)

    def test_velocity_control_rejects_disturbance(self):
        """Controller should recover speed after load disturbance."""
        result = self.sim.run_velocity_control(
            target_omega=50.0, duration=0.8,
            load_torque=0.002, load_start_time=0.3,
        )
        # Should recover to within 5% of target by end
        self.assertAlmostEqual(result.omega[-1], 50.0, delta=2.5)

    def test_position_control_reaches_target(self):
        """Position control should reach the target angle."""
        result = self.sim.run_position_control(
            target_theta=math.pi, duration=2.0
        )
        # Should be within 2% of target
        self.assertAlmostEqual(
            result.theta[-1], math.pi, delta=math.pi * 0.02
        )

    def test_zero_voltage_no_motion(self):
        """With zero voltage, motor should not move."""
        result = self.sim.run_open_loop(voltage=0.0, duration=0.1)
        self.assertAlmostEqual(result.omega[-1], 0.0, places=5)
        self.assertAlmostEqual(result.theta[-1], 0.0, places=5)


class TestAnalysis(unittest.TestCase):
    """Test analysis helper functions."""

    def test_steady_state_calculation(self):
        """Steady-state values should be physically reasonable."""
        params = DCMotorParams()
        ss = compute_steady_state(params, 12.0)
        self.assertGreater(ss['omega_ss'], 0)
        self.assertGreater(ss['i_ss'], 0)
        self.assertGreater(ss['efficiency'], 0)
        self.assertLess(ss['efficiency'], 1.0)

    def test_settling_time(self):
        """Settling time should be found correctly for a simple signal."""
        times = [i * 0.01 for i in range(100)]
        # Signal that reaches 1.0 at t=0.5 and stays
        values = [min(1.0, t * 2.0) for t in times]
        st = find_settling_time(times, values, 1.0, tolerance=0.02)
        self.assertLess(st, 0.55)  # Should settle around t=0.5

    def test_rise_time(self):
        """Rise time for a linear ramp should be predictable."""
        times = [i * 0.001 for i in range(1000)]
        target = 100.0
        values = [target * t for t in times]  # Linear ramp, reaches target at t=1
        rt = find_rise_time(times, values, target)
        # 10% at t=0.1, 90% at t=0.9, rise time = 0.8
        self.assertAlmostEqual(rt, 0.8, delta=0.01)

    def test_overshoot(self):
        """Overshoot should be computed correctly."""
        values = [0, 50, 100, 120, 110, 100, 100]  # 20% overshoot
        os = find_overshoot(values, 100.0)
        self.assertAlmostEqual(os, 20.0, places=1)


if __name__ == "__main__":
    unittest.main()
