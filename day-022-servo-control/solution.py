"""
Day 022: Servo Control Patterns

A complete servo control system implementing PWM signal generation,
motion profiles (linear, trapezoidal, easing), multi-servo synchronization,
and keyframe-based sequence execution.

Run: python3 solution.py
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


# =============================================================================
# PWM and Servo Fundamentals
# =============================================================================

# Standard hobby servo PWM parameters
# These values are industry-standard for SG90, MG996R, and similar servos.
# Period is 20ms (50Hz). Pulse width 1-2ms maps to 0-180 degrees.
PWM_PERIOD_MS = 20.0        # 50Hz standard
MIN_PULSE_MS = 1.0          # Pulse width for 0 degrees
MAX_PULSE_MS = 2.0          # Pulse width for 180 degrees
DEFAULT_MIN_ANGLE = 0.0
DEFAULT_MAX_ANGLE = 180.0


def angle_to_pulse_width(angle: float, min_angle: float = DEFAULT_MIN_ANGLE,
                         max_angle: float = DEFAULT_MAX_ANGLE) -> float:
    """
    Convert a servo angle to PWM pulse width in milliseconds.

    The mapping is linear: min_angle -> MIN_PULSE_MS, max_angle -> MAX_PULSE_MS.
    This is the fundamental equation that drives all servo positioning.

    Why linear? Servo internals use a potentiometer for feedback — the pot's
    voltage varies linearly with angle, and the control circuit compares this
    against the PWM signal width. Linear in, linear out.
    """
    # Clamp to valid range — sending out-of-range pulses can damage servos
    # by driving them into their mechanical stops
    angle = max(min_angle, min(max_angle, angle))

    # Normalize to [0, 1], then scale to pulse width range
    normalized = (angle - min_angle) / (max_angle - min_angle)
    return MIN_PULSE_MS + normalized * (MAX_PULSE_MS - MIN_PULSE_MS)


def pulse_width_to_duty_cycle(pulse_width_ms: float) -> float:
    """
    Convert pulse width to duty cycle (0.0 to 1.0).

    Duty cycle = pulse_width / period. For servos, this ranges from
    5% (1ms/20ms) to 10% (2ms/20ms). This is what you'd actually set
    on a hardware PWM peripheral or timer.
    """
    return pulse_width_ms / PWM_PERIOD_MS


# =============================================================================
# Motion Profiles
# =============================================================================
# Motion profiles define HOW a servo moves from A to B over time.
# They all implement the same interface: given a time t, return the
# interpolation factor in [0, 1]. This factor is then used to compute
# the actual angle: angle = start + factor * (end - start).
#
# Separating the profile from the servo follows the Strategy pattern —
# you can swap profiles without touching servo code.

class MotionProfile:
    """Base class for motion profiles."""

    def __init__(self, duration: float):
        """
        Args:
            duration: Total movement time in seconds.
        """
        self.duration = duration

    def interpolate(self, t: float) -> float:
        """
        Return interpolation factor in [0, 1] for time t.

        Args:
            t: Current time in seconds (0 to duration).

        Returns:
            Factor in [0, 1] representing progress through the movement.
        """
        raise NotImplementedError


class LinearProfile(MotionProfile):
    """
    Constant-velocity movement. Simplest possible profile.

    Problem: velocity discontinuity at t=0 and t=duration. The servo
    instantly jumps from stationary to full speed and back. This causes
    mechanical shock and audible "clunk" at start/end of movement.

    Use case: only when smoothness doesn't matter and you need simplicity.
    """

    def interpolate(self, t: float) -> float:
        t = max(0.0, min(self.duration, t))
        if self.duration <= 0:
            return 1.0
        return t / self.duration


class TrapezoidalProfile(MotionProfile):
    """
    Trapezoidal velocity profile — the workhorse of industrial servo control.

    Three phases: accelerate, cruise at constant velocity, decelerate.
    Eliminates velocity discontinuities (smooth start/stop) while
    maintaining efficient constant-speed cruising in the middle.

    The velocity curve looks like a trapezoid:
        /‾‾‾‾‾‾‾‾\
       /            \
      /              \
    accel  cruise  decel

    If the distance is too short to reach max velocity, it becomes
    a triangle profile (accel directly into decel, no cruise phase).
    """

    def __init__(self, duration: float, accel_fraction: float = 0.25):
        """
        Args:
            duration: Total movement time in seconds.
            accel_fraction: Fraction of total time spent accelerating (and
                           decelerating). 0.25 means 25% accel + 50% cruise + 25% decel.
                           Must be in (0, 0.5]. At 0.5, there's no cruise phase (triangle).
        """
        super().__init__(duration)
        # Clamp accel_fraction to valid range
        self.accel_fraction = max(0.01, min(0.5, accel_fraction))

        # Precompute phase boundaries
        # t_a = accel time, t_c = cruise end time, duration = decel end
        self.t_accel = self.accel_fraction * duration
        self.t_decel_start = duration - self.t_accel

    def interpolate(self, t: float) -> float:
        t = max(0.0, min(self.duration, t))
        if self.duration <= 0:
            return 1.0

        ta = self.t_accel
        td = self.t_decel_start
        T = self.duration

        # The key insight: we normalize everything so total displacement = 1.0.
        # During accel, position follows a quadratic (constant acceleration).
        # During cruise, position increases linearly.
        # During decel, position follows another quadratic (constant deceleration).

        # Peak velocity (normalized so total distance = 1):
        # v_peak = 1 / (T - ta)  [since area under trapezoid = 1]
        if T - ta <= 1e-10:
            return 1.0
        v_peak = 1.0 / (T - ta)

        if t <= ta:
            # Acceleration phase: position = 0.5 * a * t^2
            # With a = v_peak / ta, position = 0.5 * (v_peak/ta) * t^2
            if ta <= 1e-10:
                return 0.0
            return 0.5 * (v_peak / ta) * t * t
        elif t <= td:
            # Cruise phase: position = accel_distance + v_peak * (t - ta)
            accel_distance = 0.5 * v_peak * ta
            return accel_distance + v_peak * (t - ta)
        else:
            # Deceleration phase: mirror of acceleration
            # Time remaining in decel
            t_remaining = T - t
            decel_ta = T - td  # decel duration (same as ta)
            if decel_ta <= 1e-10:
                return 1.0
            # Distance remaining = 0.5 * (v_peak/decel_ta) * t_remaining^2
            return 1.0 - 0.5 * (v_peak / decel_ta) * t_remaining * t_remaining


class EasingProfile(MotionProfile):
    """
    Animation-style easing functions for natural-looking servo motion.

    These come from the animation/game industry and produce visually
    pleasing motion. The math is simpler than trapezoidal profiles
    but they don't guarantee specific velocity/acceleration limits.

    Supported easing types:
    - ease_in: slow start, fast end (quadratic)
    - ease_out: fast start, slow end (quadratic)
    - ease_in_out: slow start and end (cubic Hermite / smoothstep)
    - sine: sinusoidal easing (very smooth, no sharp transitions)
    """

    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    SINE = "sine"

    def __init__(self, duration: float, easing_type: str = "ease_in_out"):
        super().__init__(duration)
        self.easing_type = easing_type

    def interpolate(self, t: float) -> float:
        t = max(0.0, min(self.duration, t))
        if self.duration <= 0:
            return 1.0

        # Normalize time to [0, 1]
        s = t / self.duration

        if self.easing_type == self.EASE_IN:
            # f(s) = s^2: starts slow (derivative = 0 at s=0), ends fast
            return s * s
        elif self.easing_type == self.EASE_OUT:
            # f(s) = 1 - (1-s)^2: starts fast, ends slow (derivative = 0 at s=1)
            return 1.0 - (1.0 - s) ** 2
        elif self.easing_type == self.EASE_IN_OUT:
            # Smoothstep (cubic Hermite): f(s) = 3s^2 - 2s^3
            # Both endpoints have zero derivative — smooth start AND end.
            # This is the most commonly used easing for servo motion.
            return 3.0 * s * s - 2.0 * s * s * s
        elif self.easing_type == self.SINE:
            # Sine easing: f(s) = (1 - cos(pi*s)) / 2
            # Smooth everywhere, derivative is sinusoidal. Very gentle.
            return (1.0 - math.cos(math.pi * s)) / 2.0
        else:
            # Fallback to linear
            return s


# =============================================================================
# Servo Model
# =============================================================================

@dataclass
class Servo:
    """
    Model of a single servo motor.

    Tracks current angle, enforces mechanical limits, and computes
    PWM parameters. In real hardware, this would wrap a GPIO PWM output.

    Attributes:
        name: Human-readable identifier (e.g., "shoulder", "gripper")
        min_angle: Mechanical minimum in degrees
        max_angle: Mechanical maximum in degrees
        current_angle: Where the servo is right now
        speed_dps: Maximum speed in degrees per second (for profile calculations)
    """
    name: str
    min_angle: float = 0.0
    max_angle: float = 180.0
    current_angle: float = 90.0
    speed_dps: float = 300.0  # Typical hobby servo: ~60deg in 0.2s = 300 dps

    def __post_init__(self):
        self.current_angle = self._clamp(self.current_angle)

    def _clamp(self, angle: float) -> float:
        """Enforce mechanical limits. Going beyond these damages the servo."""
        return max(self.min_angle, min(self.max_angle, angle))

    def set_angle(self, angle: float) -> float:
        """Set servo to angle, returns actual angle after clamping."""
        self.current_angle = self._clamp(angle)
        return self.current_angle

    def get_pulse_width(self) -> float:
        """Get current PWM pulse width in ms."""
        return angle_to_pulse_width(self.current_angle, self.min_angle, self.max_angle)

    def get_duty_cycle(self) -> float:
        """Get current duty cycle (0-1)."""
        return pulse_width_to_duty_cycle(self.get_pulse_width())

    def time_to_reach(self, target_angle: float) -> float:
        """
        Minimum time to reach target angle at max speed.
        Used for synchronization — the slowest servo sets the pace.
        """
        distance = abs(target_angle - self.current_angle)
        if self.speed_dps <= 0:
            return float('inf')
        return distance / self.speed_dps


# =============================================================================
# Active Movement — ties a servo to a motion profile
# =============================================================================

@dataclass
class ActiveMovement:
    """
    An in-progress servo movement: servo + profile + start/end angles.

    This is the runtime binding between a servo and its motion plan.
    The controller creates these when you command a move, then steps
    through them each tick.
    """
    servo: Servo
    profile: MotionProfile
    start_angle: float
    end_angle: float
    elapsed: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.elapsed >= self.profile.duration

    def update(self, dt: float) -> float:
        """
        Advance by dt seconds, return new angle.

        The profile gives us an interpolation factor [0,1], which we
        use to lerp between start and end angle.
        """
        self.elapsed += dt
        factor = self.profile.interpolate(self.elapsed)
        angle = self.start_angle + factor * (self.end_angle - self.start_angle)
        self.servo.set_angle(angle)
        return self.servo.current_angle


# =============================================================================
# Keyframe Sequence
# =============================================================================

@dataclass
class Keyframe:
    """
    A target pose (set of servo angles) with a duration and motion profile.

    A sequence of keyframes defines a complete motion — like a robot arm
    reaching out, grasping, and retracting. Each keyframe specifies:
    - Where each servo should be (target_angles)
    - How long the transition should take (duration)
    - What motion profile to use (profile_type)
    """
    target_angles: dict  # servo_name -> target_angle
    duration: float      # seconds to reach this pose
    profile_type: str = "trapezoidal"  # "linear", "trapezoidal", "ease_in_out", "sine"


# =============================================================================
# Servo Controller
# =============================================================================

class ServoController:
    """
    Manages multiple servos, coordinates movements, and executes sequences.

    This is the main orchestrator. In a real robot, this would run in a
    control loop at 50-100Hz, updating PWM outputs each tick.

    Key responsibilities:
    1. Track all servos by name
    2. Create and manage motion profiles for movements
    3. Synchronize multi-servo movements
    4. Execute keyframe sequences
    """

    def __init__(self):
        self.servos: dict[str, Servo] = {}
        self.active_movements: list[ActiveMovement] = []

    def add_servo(self, servo: Servo) -> None:
        """Register a servo with the controller."""
        self.servos[servo.name] = servo

    def get_servo(self, name: str) -> Servo:
        """Get a servo by name."""
        return self.servos[name]

    def _make_profile(self, profile_type: str, duration: float) -> MotionProfile:
        """Factory method for motion profiles."""
        if profile_type == "linear":
            return LinearProfile(duration)
        elif profile_type == "trapezoidal":
            return TrapezoidalProfile(duration, accel_fraction=0.25)
        elif profile_type == "ease_in":
            return EasingProfile(duration, EasingProfile.EASE_IN)
        elif profile_type == "ease_out":
            return EasingProfile(duration, EasingProfile.EASE_OUT)
        elif profile_type == "ease_in_out":
            return EasingProfile(duration, EasingProfile.EASE_IN_OUT)
        elif profile_type == "sine":
            return EasingProfile(duration, EasingProfile.SINE)
        else:
            return TrapezoidalProfile(duration)

    def move_servo(self, name: str, target_angle: float, duration: float,
                   profile_type: str = "trapezoidal") -> ActiveMovement:
        """
        Command a single servo to move to target_angle over duration seconds.

        Creates an ActiveMovement that will be stepped by update().
        """
        servo = self.servos[name]
        profile = self._make_profile(profile_type, duration)
        movement = ActiveMovement(
            servo=servo,
            profile=profile,
            start_angle=servo.current_angle,
            end_angle=target_angle,
            elapsed=0.0
        )
        self.active_movements.append(movement)
        return movement

    def move_synchronized(self, targets: dict[str, float],
                          profile_type: str = "trapezoidal",
                          duration: Optional[float] = None) -> list[ActiveMovement]:
        """
        Move multiple servos simultaneously, synchronized to finish together.

        If duration is not specified, it's calculated from the slowest servo
        (the one that needs the most time at its max speed). All servos then
        use this same duration so they start and finish together.

        This is critical for coordinated motion — e.g., a robotic arm where
        shoulder and elbow must move in sync for a straight-line path.
        """
        if duration is None:
            # Find the slowest servo (the one that needs the most time)
            max_time = 0.0
            for name, target in targets.items():
                servo = self.servos[name]
                t = servo.time_to_reach(target)
                max_time = max(max_time, t)
            # Add 10% margin for smooth acceleration/deceleration
            duration = max_time * 1.1
            # Minimum duration to prevent instant jumps
            duration = max(duration, 0.1)

        movements = []
        for name, target in targets.items():
            movement = self.move_servo(name, target, duration, profile_type)
            movements.append(movement)

        return movements

    def update(self, dt: float) -> list[Tuple[str, float]]:
        """
        Step all active movements forward by dt seconds.

        Returns list of (servo_name, current_angle) for all moving servos.
        Removes completed movements.

        This is the core control loop tick — in real hardware, you'd call
        this at 50Hz (every 20ms, matching the PWM period) and write the
        resulting angles to PWM outputs.
        """
        results = []
        still_active = []

        for movement in self.active_movements:
            angle = movement.update(dt)
            results.append((movement.servo.name, angle))
            if not movement.is_complete:
                still_active.append(movement)

        self.active_movements = still_active
        return results

    def is_moving(self) -> bool:
        """Check if any servos are currently moving."""
        return len(self.active_movements) > 0

    def execute_sequence(self, keyframes: list[Keyframe],
                         time_step: float = 0.02) -> list[dict]:
        """
        Execute a sequence of keyframes, returning the full trajectory log.

        Each keyframe moves all specified servos to their targets over
        the keyframe's duration. The next keyframe starts only when the
        current one completes.

        Returns a list of snapshots: [{time, servo_name: angle, ...}, ...]
        This trajectory log is useful for visualization and debugging.

        Args:
            keyframes: Ordered list of target poses with durations.
            time_step: Simulation step in seconds (0.02 = 50Hz).
        """
        trajectory = []
        global_time = 0.0

        for kf_idx, kf in enumerate(keyframes):
            # Start synchronized movement for this keyframe
            self.move_synchronized(
                kf.target_angles,
                profile_type=kf.profile_type,
                duration=kf.duration
            )

            # Step through until all movements complete
            elapsed = 0.0
            while self.is_moving() and elapsed <= kf.duration + time_step:
                self.update(time_step)
                elapsed += time_step
                global_time += time_step

                # Log snapshot
                snapshot = {"time": round(global_time, 4)}
                for name, servo in self.servos.items():
                    snapshot[name] = round(servo.current_angle, 2)
                trajectory.append(snapshot)

        return trajectory


# =============================================================================
# Demonstration
# =============================================================================

def demo_pwm_basics():
    """Show PWM calculations for a range of angles."""
    print("=" * 60)
    print("PWM BASICS: Angle -> Pulse Width -> Duty Cycle")
    print("=" * 60)
    print(f"{'Angle (deg)':>12} {'Pulse (ms)':>12} {'Duty Cycle':>12}")
    print("-" * 40)

    for angle in [0, 30, 45, 60, 90, 120, 135, 150, 180]:
        pw = angle_to_pulse_width(angle)
        dc = pulse_width_to_duty_cycle(pw)
        print(f"{angle:>12.1f} {pw:>12.3f} {dc:>11.2%}")

    print()


def demo_motion_profiles():
    """Compare different motion profiles side by side."""
    print("=" * 60)
    print("MOTION PROFILES: Position over time (0=start, 1=end)")
    print("=" * 60)

    duration = 1.0
    profiles = {
        "Linear": LinearProfile(duration),
        "Trapezoid": TrapezoidalProfile(duration, accel_fraction=0.25),
        "EaseInOut": EasingProfile(duration, EasingProfile.EASE_IN_OUT),
        "Sine": EasingProfile(duration, EasingProfile.SINE),
    }

    steps = 10
    header = f"{'Time':>6}"
    for name in profiles:
        header += f" {name:>10}"
    print(header)
    print("-" * (6 + 11 * len(profiles)))

    for i in range(steps + 1):
        t = i / steps * duration
        row = f"{t:>6.2f}"
        for name, profile in profiles.items():
            val = profile.interpolate(t)
            row += f" {val:>10.4f}"
        print(row)

    # Show velocity approximation (finite differences)
    print(f"\nApproximate velocity (position change per step):")
    dt = duration / steps
    header = f"{'Time':>6}"
    for name in profiles:
        header += f" {name:>10}"
    print(header)
    print("-" * (6 + 11 * len(profiles)))

    for i in range(steps):
        t = i / steps * duration
        row = f"{t:>6.2f}"
        for name, profile in profiles.items():
            v0 = profile.interpolate(t)
            v1 = profile.interpolate(t + dt)
            velocity = (v1 - v0) / dt
            row += f" {velocity:>10.4f}"
        print(row)

    print("\nKey observation: Linear has constant velocity but jumps at endpoints.")
    print("Trapezoidal and easing profiles have smooth transitions.\n")


def demo_single_servo():
    """Demonstrate single servo movement with different profiles."""
    print("=" * 60)
    print("SINGLE SERVO: Moving from 30 to 150 degrees")
    print("=" * 60)

    for profile_type in ["linear", "trapezoidal", "ease_in_out"]:
        servo = Servo(name="test", current_angle=30.0)
        controller = ServoController()
        controller.add_servo(servo)

        controller.move_servo("test", 150.0, duration=1.0, profile_type=profile_type)

        print(f"\nProfile: {profile_type}")
        print(f"  {'Time':>6} {'Angle':>8} {'Pulse(ms)':>10} {'Duty':>8}")

        t = 0.0
        dt = 0.1
        while controller.is_moving():
            controller.update(dt)
            t += dt
            pw = servo.get_pulse_width()
            dc = servo.get_duty_cycle()
            print(f"  {t:>6.2f} {servo.current_angle:>8.2f} {pw:>10.3f} {dc:>7.2%}")

    print()


def demo_synchronized_movement():
    """Show multi-servo synchronization."""
    print("=" * 60)
    print("SYNCHRONIZED MOVEMENT: 3-DOF Robot Arm")
    print("=" * 60)

    controller = ServoController()
    controller.add_servo(Servo(name="base", current_angle=90.0, speed_dps=200.0))
    controller.add_servo(Servo(name="shoulder", current_angle=45.0, speed_dps=150.0))
    controller.add_servo(Servo(name="elbow", current_angle=90.0, speed_dps=300.0))

    targets = {"base": 45.0, "shoulder": 120.0, "elbow": 30.0}

    print(f"\nStarting positions:")
    for name, servo in controller.servos.items():
        print(f"  {name}: {servo.current_angle:.1f} deg")

    print(f"\nTarget positions:")
    for name, angle in targets.items():
        print(f"  {name}: {angle:.1f} deg")

    # Without sync, show how long each would take alone
    print(f"\nIndividual minimum times:")
    for name, target in targets.items():
        t = controller.servos[name].time_to_reach(target)
        print(f"  {name}: {t:.3f}s (distance: {abs(target - controller.servos[name].current_angle):.1f} deg)")

    # Execute synchronized
    movements = controller.move_synchronized(targets, profile_type="trapezoidal")
    sync_duration = movements[0].profile.duration
    print(f"\nSynchronized duration: {sync_duration:.3f}s (all finish together)")

    print(f"\n  {'Time':>6} {'Base':>8} {'Shoulder':>10} {'Elbow':>8}")
    print("  " + "-" * 36)

    t = 0.0
    dt = 0.05
    while controller.is_moving():
        controller.update(dt)
        t += dt
        if int(t * 100) % 10 == 0:  # Print every 0.1s
            b = controller.servos["base"].current_angle
            s = controller.servos["shoulder"].current_angle
            e = controller.servos["elbow"].current_angle
            print(f"  {t:>6.2f} {b:>8.2f} {s:>10.2f} {e:>8.2f}")

    print(f"\nAll servos reached targets simultaneously!\n")


def demo_keyframe_sequence():
    """Execute a pick-and-place sequence using keyframes."""
    print("=" * 60)
    print("KEYFRAME SEQUENCE: Pick and Place Operation")
    print("=" * 60)

    controller = ServoController()
    controller.add_servo(Servo(name="base", current_angle=90.0))
    controller.add_servo(Servo(name="shoulder", current_angle=90.0))
    controller.add_servo(Servo(name="gripper", current_angle=10.0, min_angle=10.0, max_angle=70.0))

    # Define a pick-and-place sequence
    # Each keyframe is a pose the robot transitions to
    keyframes = [
        Keyframe(  # Move to hover above object
            target_angles={"base": 45.0, "shoulder": 60.0, "gripper": 10.0},
            duration=1.0,
            profile_type="trapezoidal"
        ),
        Keyframe(  # Lower to object
            target_angles={"base": 45.0, "shoulder": 30.0, "gripper": 10.0},
            duration=0.5,
            profile_type="ease_in_out"
        ),
        Keyframe(  # Close gripper (grasp)
            target_angles={"base": 45.0, "shoulder": 30.0, "gripper": 60.0},
            duration=0.3,
            profile_type="ease_in"
        ),
        Keyframe(  # Lift object
            target_angles={"base": 45.0, "shoulder": 70.0, "gripper": 60.0},
            duration=0.5,
            profile_type="trapezoidal"
        ),
        Keyframe(  # Rotate to drop zone
            target_angles={"base": 135.0, "shoulder": 70.0, "gripper": 60.0},
            duration=0.8,
            profile_type="sine"
        ),
        Keyframe(  # Lower to drop zone
            target_angles={"base": 135.0, "shoulder": 40.0, "gripper": 60.0},
            duration=0.5,
            profile_type="ease_in_out"
        ),
        Keyframe(  # Release object
            target_angles={"base": 135.0, "shoulder": 40.0, "gripper": 10.0},
            duration=0.3,
            profile_type="ease_out"
        ),
        Keyframe(  # Return to home
            target_angles={"base": 90.0, "shoulder": 90.0, "gripper": 10.0},
            duration=1.0,
            profile_type="trapezoidal"
        ),
    ]

    print("\nSequence steps:")
    descriptions = [
        "Hover above object", "Lower to object", "Close gripper",
        "Lift object", "Rotate to drop zone", "Lower to place",
        "Release object", "Return home"
    ]
    for i, (kf, desc) in enumerate(zip(keyframes, descriptions)):
        angles = ", ".join(f"{k}={v:.0f}" for k, v in kf.target_angles.items())
        print(f"  {i+1}. {desc} ({kf.duration}s, {kf.profile_type}): {angles}")

    # Execute and get trajectory
    trajectory = controller.execute_sequence(keyframes, time_step=0.02)

    # Print sampled trajectory (every ~0.2s)
    print(f"\nTrajectory ({len(trajectory)} steps at 50Hz):")
    print(f"  {'Time':>6} {'Base':>8} {'Shoulder':>10} {'Gripper':>9}")
    print("  " + "-" * 37)

    last_printed = -1.0
    for snap in trajectory:
        if snap["time"] - last_printed >= 0.19:
            print(f"  {snap['time']:>6.2f} {snap['base']:>8.2f} "
                  f"{snap['shoulder']:>10.2f} {snap['gripper']:>9.2f}")
            last_printed = snap["time"]

    print(f"\nTotal trajectory time: {trajectory[-1]['time']:.2f}s")
    print(f"Final pose: base={controller.servos['base'].current_angle:.1f}, "
          f"shoulder={controller.servos['shoulder'].current_angle:.1f}, "
          f"gripper={controller.servos['gripper'].current_angle:.1f}")
    print()


if __name__ == "__main__":
    demo_pwm_basics()
    demo_motion_profiles()
    demo_single_servo()
    demo_synchronized_movement()
    demo_keyframe_sequence()

    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
