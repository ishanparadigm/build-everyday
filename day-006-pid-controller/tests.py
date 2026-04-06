"""
Day 006: PID Controller — Test Suite

Tests import from my_solution. Run with: python3 tests.py
"""

import sys
import numpy as np

from my_solution import PIDController, ThermalPlant, simulate, analyze_step_response


def test_p_only_proportional_response():
    """P-only controller output should be proportional to error."""
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0, setpoint=100.0, dt=0.1)
    output = pid.update(80.0)  # error = 20
    assert abs(output - 40.0) < 1e-6, f"Expected 40.0, got {output}"
    print("  PASS: P-only proportional response")


def test_p_only_steady_state_error():
    """P-only controller should have nonzero steady-state error."""
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0, setpoint=50.0, dt=0.05)
    plant = ThermalPlant(initial_temp=20.0, ambient_temp=20.0, tau=10.0, gain=1.0, dt=0.05)
    times, temps, _ = simulate(pid, plant, duration=60.0, dt=0.05)
    metrics = analyze_step_response(times, temps, 50.0)
    assert metrics["steady_state_error"] > 0.5, \
        f"P-only should have significant SS error, got {metrics['steady_state_error']:.4f}"
    print("  PASS: P-only has steady-state error")


def test_integral_eliminates_steady_state_error():
    """PI controller should drive steady-state error near zero."""
    pid = PIDController(kp=2.0, ki=0.2, kd=0.0, setpoint=50.0, dt=0.05)
    plant = ThermalPlant(initial_temp=20.0, ambient_temp=20.0, tau=10.0, gain=1.0, dt=0.05)
    times, temps, _ = simulate(pid, plant, duration=80.0, dt=0.05)
    metrics = analyze_step_response(times, temps, 50.0)
    assert metrics["steady_state_error"] < 0.5, \
        f"PI should eliminate SS error, got {metrics['steady_state_error']:.4f}"
    print("  PASS: Integral eliminates steady-state error")


def test_derivative_reduces_overshoot():
    """Adding D term should reduce overshoot compared to PI-only."""
    # PI-only
    pid_pi = PIDController(kp=3.0, ki=0.2, kd=0.0, setpoint=50.0, dt=0.05)
    plant_pi = ThermalPlant(initial_temp=20.0, ambient_temp=20.0, tau=10.0, gain=1.0, dt=0.05)
    times_pi, temps_pi, _ = simulate(pid_pi, plant_pi, duration=80.0, dt=0.05)
    metrics_pi = analyze_step_response(times_pi, temps_pi, 50.0)

    # PID
    pid_pid = PIDController(kp=3.0, ki=0.2, kd=2.0, setpoint=50.0, dt=0.05)
    plant_pid = ThermalPlant(initial_temp=20.0, ambient_temp=20.0, tau=10.0, gain=1.0, dt=0.05)
    times_pid, temps_pid, _ = simulate(pid_pid, plant_pid, duration=80.0, dt=0.05)
    metrics_pid = analyze_step_response(times_pid, temps_pid, 50.0)

    assert metrics_pid["overshoot_pct"] <= metrics_pi["overshoot_pct"], \
        f"PID overshoot ({metrics_pid['overshoot_pct']:.2f}%) should be <= PI ({metrics_pi['overshoot_pct']:.2f}%)"
    print("  PASS: Derivative reduces overshoot")


def test_pid_converges_to_setpoint():
    """Full PID should converge close to the setpoint."""
    pid = PIDController(kp=2.0, ki=0.1, kd=1.0, setpoint=50.0, dt=0.05)
    plant = ThermalPlant(initial_temp=20.0, ambient_temp=20.0, tau=10.0, gain=1.0, dt=0.05)
    times, temps, _ = simulate(pid, plant, duration=80.0, dt=0.05)

    final_temp = np.mean(temps[-100:])
    assert abs(final_temp - 50.0) < 1.0, \
        f"PID should converge to ~50.0, got {final_temp:.2f}"
    print("  PASS: PID converges to setpoint")


def test_reset_clears_state():
    """After reset, integral and prev_error should be cleared."""
    pid = PIDController(kp=1.0, ki=1.0, kd=1.0, setpoint=50.0, dt=0.1)

    # Run a few updates to build up state
    pid.update(20.0)
    pid.update(25.0)
    pid.update(30.0)

    # Verify state exists
    assert pid._integral != 0.0, "Integral should be nonzero before reset"
    assert pid._prev_error is not None, "prev_error should be set before reset"

    # Reset
    pid.reset()
    assert pid._integral == 0.0, f"Integral should be 0 after reset, got {pid._integral}"
    assert pid._prev_error is None, f"prev_error should be None after reset, got {pid._prev_error}"
    print("  PASS: Reset clears internal state")


def test_zero_error_zero_output():
    """When measurement equals setpoint, initial output should be zero."""
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0, setpoint=50.0, dt=0.1)
    output = pid.update(50.0)
    assert abs(output) < 1e-6, f"Expected 0 output at setpoint, got {output}"
    print("  PASS: Zero error produces zero output")


def test_integral_accumulates():
    """Integral term should grow over repeated calls with constant error."""
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, setpoint=10.0, dt=1.0)

    out1 = pid.update(0.0)  # error=10, integral=10*1=10, output=10
    out2 = pid.update(0.0)  # error=10, integral=10+10=20, output=20
    out3 = pid.update(0.0)  # error=10, integral=20+10=30, output=30

    assert abs(out1 - 10.0) < 1e-6, f"Expected 10.0, got {out1}"
    assert abs(out2 - 20.0) < 1e-6, f"Expected 20.0, got {out2}"
    assert abs(out3 - 30.0) < 1e-6, f"Expected 30.0, got {out3}"
    print("  PASS: Integral accumulates correctly")


def test_derivative_responds_to_change():
    """Derivative term should respond to error rate of change."""
    pid = PIDController(kp=0.0, ki=0.0, kd=1.0, setpoint=10.0, dt=1.0)

    pid.update(5.0)   # error=5, no prev -> d=0
    out2 = pid.update(3.0)  # error=7, prev_error=5, d=(7-5)/1=2, output=2

    assert abs(out2 - 2.0) < 1e-6, f"Expected 2.0, got {out2}"
    print("  PASS: Derivative responds to error change rate")


def test_plant_heats_up():
    """Plant should increase temperature when given positive control input."""
    plant = ThermalPlant(initial_temp=20.0, ambient_temp=20.0, tau=10.0, gain=1.0, dt=0.1)
    initial = plant.temperature
    for _ in range(100):
        plant.step(50.0)  # apply heat
    assert plant.temperature > initial + 5.0, \
        f"Plant should heat up significantly, got {plant.temperature:.2f}"
    print("  PASS: Plant heats up with positive input")


def test_analyze_step_response_returns_keys():
    """analyze_step_response should return all expected metric keys."""
    times = np.linspace(0, 10, 200)
    values = 50.0 - 30.0 * np.exp(-times)  # exponential approach to 50
    metrics = analyze_step_response(times, values, 50.0)
    expected_keys = {"overshoot_pct", "settling_time", "steady_state_error", "rise_time"}
    assert set(metrics.keys()) == expected_keys, f"Missing keys: {expected_keys - set(metrics.keys())}"
    print("  PASS: analyze_step_response returns all metric keys")


if __name__ == "__main__":
    tests = [
        test_p_only_proportional_response,
        test_p_only_steady_state_error,
        test_integral_eliminates_steady_state_error,
        test_derivative_reduces_overshoot,
        test_pid_converges_to_setpoint,
        test_reset_clears_state,
        test_zero_error_zero_output,
        test_integral_accumulates,
        test_derivative_responds_to_change,
        test_plant_heats_up,
        test_analyze_step_response_returns_keys,
    ]

    print(f"\nRunning {len(tests)} tests for Day 006: PID Controller\n")

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} -- {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)
