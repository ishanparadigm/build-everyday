"""
Day 022: Servo Control Patterns — Test Suite

Run: python3 -m pytest tests.py -v
  or: python3 tests.py
"""

import math
import unittest
from my_solution import (
    angle_to_pulse_width, pulse_width_to_duty_cycle,
    LinearProfile, TrapezoidalProfile, EasingProfile,
    Servo, ActiveMovement, Keyframe, ServoController,
    MIN_PULSE_MS, MAX_PULSE_MS, PWM_PERIOD_MS,
)


class TestPWMBasics(unittest.TestCase):
    """Test PWM signal calculations."""

    def test_angle_0_gives_min_pulse(self):
        self.assertAlmostEqual(angle_to_pulse_width(0.0), MIN_PULSE_MS, places=4)

    def test_angle_180_gives_max_pulse(self):
        self.assertAlmostEqual(angle_to_pulse_width(180.0), MAX_PULSE_MS, places=4)

    def test_angle_90_gives_center_pulse(self):
        center = (MIN_PULSE_MS + MAX_PULSE_MS) / 2  # 1.5ms
        self.assertAlmostEqual(angle_to_pulse_width(90.0), center, places=4)

    def test_pulse_width_clamping(self):
        """Out-of-range angles should be clamped, not extrapolated."""
        self.assertAlmostEqual(angle_to_pulse_width(-10.0), MIN_PULSE_MS, places=4)
        self.assertAlmostEqual(angle_to_pulse_width(200.0), MAX_PULSE_MS, places=4)

    def test_duty_cycle_range(self):
        """Duty cycle should be between 5% and 10% for standard servos."""
        dc_min = pulse_width_to_duty_cycle(MIN_PULSE_MS)
        dc_max = pulse_width_to_duty_cycle(MAX_PULSE_MS)
        self.assertAlmostEqual(dc_min, 0.05, places=4)
        self.assertAlmostEqual(dc_max, 0.10, places=4)


class TestLinearProfile(unittest.TestCase):
    """Test linear (constant velocity) motion profile."""

    def test_starts_at_zero(self):
        p = LinearProfile(1.0)
        self.assertAlmostEqual(p.interpolate(0.0), 0.0, places=4)

    def test_ends_at_one(self):
        p = LinearProfile(1.0)
        self.assertAlmostEqual(p.interpolate(1.0), 1.0, places=4)

    def test_midpoint(self):
        p = LinearProfile(2.0)
        self.assertAlmostEqual(p.interpolate(1.0), 0.5, places=4)

    def test_clamping(self):
        """Time before 0 or after duration should clamp."""
        p = LinearProfile(1.0)
        self.assertAlmostEqual(p.interpolate(-0.5), 0.0, places=4)
        self.assertAlmostEqual(p.interpolate(1.5), 1.0, places=4)


class TestTrapezoidalProfile(unittest.TestCase):
    """Test trapezoidal velocity motion profile."""

    def test_starts_at_zero(self):
        p = TrapezoidalProfile(1.0)
        self.assertAlmostEqual(p.interpolate(0.0), 0.0, places=4)

    def test_ends_at_one(self):
        p = TrapezoidalProfile(1.0)
        self.assertAlmostEqual(p.interpolate(1.0), 1.0, places=3)

    def test_monotonically_increasing(self):
        """Position should always increase (no going backwards)."""
        p = TrapezoidalProfile(1.0, accel_fraction=0.3)
        prev = 0.0
        for i in range(1, 21):
            t = i * 0.05
            val = p.interpolate(t)
            self.assertGreaterEqual(val, prev - 1e-10,
                                    f"Decreased at t={t}: {val} < {prev}")
            prev = val

    def test_smooth_acceleration(self):
        """Velocity should start at 0 (not jump like linear)."""
        p = TrapezoidalProfile(1.0, accel_fraction=0.25)
        # Velocity near t=0 should be very small
        dt = 0.001
        v_start = (p.interpolate(dt) - p.interpolate(0.0)) / dt
        v_mid = (p.interpolate(0.51) - p.interpolate(0.49)) / 0.02
        self.assertLess(v_start, v_mid * 0.1,
                        "Velocity at start should be much less than at midpoint")


class TestEasingProfile(unittest.TestCase):
    """Test easing motion profiles."""

    def test_all_types_start_at_zero(self):
        for etype in ["ease_in", "ease_out", "ease_in_out", "sine"]:
            p = EasingProfile(1.0, etype)
            self.assertAlmostEqual(p.interpolate(0.0), 0.0, places=4,
                                   msg=f"{etype} didn't start at 0")

    def test_all_types_end_at_one(self):
        for etype in ["ease_in", "ease_out", "ease_in_out", "sine"]:
            p = EasingProfile(1.0, etype)
            self.assertAlmostEqual(p.interpolate(1.0), 1.0, places=4,
                                   msg=f"{etype} didn't end at 1")

    def test_ease_in_out_midpoint(self):
        """Smoothstep should be 0.5 at midpoint (symmetric)."""
        p = EasingProfile(1.0, "ease_in_out")
        self.assertAlmostEqual(p.interpolate(0.5), 0.5, places=4)

    def test_sine_midpoint(self):
        """Sine easing should be 0.5 at midpoint."""
        p = EasingProfile(1.0, "sine")
        self.assertAlmostEqual(p.interpolate(0.5), 0.5, places=4)


class TestServo(unittest.TestCase):
    """Test servo model."""

    def test_initial_angle_clamped(self):
        s = Servo(name="s", min_angle=10.0, max_angle=170.0, current_angle=200.0)
        self.assertEqual(s.current_angle, 170.0)

    def test_set_angle_clamps(self):
        s = Servo(name="s", current_angle=90.0)
        s.set_angle(200.0)
        self.assertEqual(s.current_angle, 180.0)
        s.set_angle(-10.0)
        self.assertEqual(s.current_angle, 0.0)

    def test_pulse_width_at_center(self):
        s = Servo(name="s", current_angle=90.0)
        self.assertAlmostEqual(s.get_pulse_width(), 1.5, places=3)

    def test_time_to_reach(self):
        s = Servo(name="s", current_angle=0.0, speed_dps=180.0)
        self.assertAlmostEqual(s.time_to_reach(180.0), 1.0, places=3)


class TestServoController(unittest.TestCase):
    """Test the controller orchestrator."""

    def test_single_servo_reaches_target(self):
        ctrl = ServoController()
        ctrl.add_servo(Servo(name="arm", current_angle=0.0))
        ctrl.move_servo("arm", 90.0, duration=1.0, profile_type="linear")

        # Step through
        for _ in range(100):
            ctrl.update(0.01)

        self.assertAlmostEqual(ctrl.get_servo("arm").current_angle, 90.0, places=1)

    def test_synchronized_finish_together(self):
        """All servos should finish at approximately the same time."""
        ctrl = ServoController()
        ctrl.add_servo(Servo(name="a", current_angle=0.0, speed_dps=100.0))
        ctrl.add_servo(Servo(name="b", current_angle=0.0, speed_dps=300.0))

        ctrl.move_synchronized({"a": 50.0, "b": 150.0}, profile_type="linear")

        # Step through
        for _ in range(200):
            if not ctrl.is_moving():
                break
            ctrl.update(0.01)

        self.assertAlmostEqual(ctrl.get_servo("a").current_angle, 50.0, places=0)
        self.assertAlmostEqual(ctrl.get_servo("b").current_angle, 150.0, places=0)

    def test_keyframe_sequence(self):
        """Sequence should visit each keyframe's target."""
        ctrl = ServoController()
        ctrl.add_servo(Servo(name="s", current_angle=0.0))

        keyframes = [
            Keyframe(target_angles={"s": 90.0}, duration=0.5, profile_type="linear"),
            Keyframe(target_angles={"s": 45.0}, duration=0.5, profile_type="linear"),
        ]

        trajectory = ctrl.execute_sequence(keyframes, time_step=0.02)

        # Final position should be ~45 degrees
        self.assertAlmostEqual(ctrl.get_servo("s").current_angle, 45.0, places=0)
        # Trajectory should have entries
        self.assertGreater(len(trajectory), 10)


if __name__ == "__main__":
    unittest.main()
