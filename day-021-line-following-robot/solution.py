"""
Day 021: Line-Following Robot Logic

Complete simulation of a differential-drive robot following a line using
reflectance sensor arrays and multiple control strategies (bang-bang, P, PID).

Builds on Day 006 (PID fundamentals), Day 007 (state machines), Day 014 (motor control),
and Day 016 (sensor simulation) to tackle a real-world navigation task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


# ---------------------------------------------------------------------------
# Track definition — the line the robot follows
# ---------------------------------------------------------------------------

@dataclass
class TrackSegment:
    """A straight or arc segment of the track."""
    start: tuple[float, float]
    end: tuple[float, float]
    # For arcs: center and radius. If radius == 0, it's a straight segment.
    center: tuple[float, float] | None = None
    radius: float = 0.0


def build_track(segments: list[dict]) -> list[tuple[float, float]]:
    """
    Build a track as a list of densely-sampled (x, y) waypoints.

    Each segment dict has:
      - type: "straight" or "arc"
      - For straight: "start", "end"
      - For arc: "center", "radius", "start_angle", "end_angle" (radians)
      - "resolution": points per unit length (default 10)

    Returns a list of (x, y) points defining the track centerline.
    """
    points: list[tuple[float, float]] = []

    for seg in segments:
        res = seg.get("resolution", 100)

        if seg["type"] == "straight":
            sx, sy = seg["start"]
            ex, ey = seg["end"]
            length = math.hypot(ex - sx, ey - sy)
            n_points = max(int(length * res), 2)
            for i in range(n_points):
                t = i / (n_points - 1)
                points.append((sx + t * (ex - sx), sy + t * (ey - sy)))

        elif seg["type"] == "arc":
            cx, cy = seg["center"]
            r = seg["radius"]
            a0 = seg["start_angle"]
            a1 = seg["end_angle"]
            arc_len = abs(a1 - a0) * r
            n_points = max(int(arc_len * res), 2)
            for i in range(n_points):
                t = i / (n_points - 1)
                angle = a0 + t * (a1 - a0)
                points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    return points


def closest_point_on_track(
    track: list[tuple[float, float]], px: float, py: float
) -> tuple[float, float, float]:
    """
    Find the closest point on the track to a query point (px, py).

    Uses brute-force over track segments (pairs of consecutive waypoints).
    Returns (closest_x, closest_y, distance).

    Why brute-force? For our simulation with ~1000 waypoints, this is fast enough
    and avoids the complexity of spatial indexing. In production you'd use a KD-tree.
    """
    best_dist = float("inf")
    best_point = track[0]

    for i in range(len(track) - 1):
        ax, ay = track[i]
        bx, by = track[i + 1]

        # Project (px, py) onto segment [a, b]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-12:
            # Degenerate segment
            cx, cy = ax, ay
        else:
            # t is the projection parameter, clamped to [0, 1]
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
            cx = ax + t * dx
            cy = ay + t * dy

        dist = math.hypot(px - cx, py - cy)
        if dist < best_dist:
            best_dist = dist
            best_point = (cx, cy)

    return best_point[0], best_point[1], best_dist


def signed_distance_to_track(
    track: list[tuple[float, float]], px: float, py: float, heading: float
) -> float:
    """
    Compute the signed lateral distance from point (px, py) to the track.

    Positive = line is to the LEFT of the robot's heading.
    Negative = line is to the RIGHT.

    This sign convention matters for the controller: a positive error means
    "turn left to get back to the line."
    """
    cx, cy, dist = closest_point_on_track(track, px, py)

    # Vector from robot to closest track point
    dx = cx - px
    dy = cy - py

    # Cross product with heading vector gives sign
    # heading_vec = (cos(heading), sin(heading))
    # cross = heading_vec x (dx, dy) = cos(h)*dy - sin(h)*dx
    cross = math.cos(heading) * dy - math.sin(heading) * dx

    # If cross > 0, track point is to the left; if < 0, to the right
    return math.copysign(dist, cross)


# ---------------------------------------------------------------------------
# Sensor array simulation
# ---------------------------------------------------------------------------

@dataclass
class SensorArray:
    """
    Simulates an array of infrared reflectance sensors.

    Attributes:
        n_sensors: Number of sensors in the array (typically 3-8)
        array_width: Total width of the sensor array in meters
        line_sigma: Controls how wide the "line" appears to sensors.
                    Larger sigma = wider line = easier to track.
    """
    n_sensors: int = 5
    array_width: float = 0.06  # 6 cm total width
    line_sigma: float = 0.012  # ~1.2 cm effective line half-width

    @property
    def sensor_positions(self) -> list[float]:
        """
        Return the lateral offset of each sensor from the robot center.
        Negative = left, positive = right.
        """
        if self.n_sensors == 1:
            return [0.0]
        half = self.array_width / 2
        step = self.array_width / (self.n_sensors - 1)
        return [-half + i * step for i in range(self.n_sensors)]

    def read(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        track: list[tuple[float, float]],
    ) -> list[float]:
        """
        Compute raw sensor readings for each sensor in the array.

        For each sensor:
        1. Compute its world position based on robot pose
        2. Find the distance to the nearest track point
        3. Apply the Gaussian reflectance model

        Returns a list of readings in [0, 1] where 0 = on line, 1 = off line.
        """
        readings = []
        cos_h = math.cos(robot_heading)
        sin_h = math.sin(robot_heading)

        for offset in self.sensor_positions:
            # Sensor is offset laterally from robot center
            # "Left" in robot frame = perpendicular-left of heading
            sx = robot_x + offset * (-sin_h)  # perpendicular to heading
            sy = robot_y + offset * cos_h

            _, _, dist = closest_point_on_track(track, sx, sy)

            # Gaussian reflectance: closer to line = lower reading (darker)
            reading = 1.0 - math.exp(-(dist ** 2) / (2 * self.line_sigma ** 2))
            readings.append(reading)

        return readings

    def estimate_line_position(self, readings: list[float]) -> float | None:
        """
        Estimate the line position from sensor readings using weighted average.

        Returns the estimated lateral offset of the line from robot center:
          - Negative = line is to the left
          - Positive = line is to the right
          - None = line not detected (all sensors read high)

        The weighted average formula:
          position = sum((1 - r_i) * x_i) / sum(1 - r_i)

        where r_i is the reading and x_i is the sensor position.
        """
        positions = self.sensor_positions
        weights = [1.0 - r for r in readings]
        total_weight = sum(weights)

        # If total weight is very small, the line isn't under any sensor
        if total_weight < 0.05:
            return None

        return sum(w * p for w, p in zip(weights, positions)) / total_weight


# ---------------------------------------------------------------------------
# Controllers — three strategies with increasing sophistication
# ---------------------------------------------------------------------------

class Controller(Protocol):
    """Interface for line-following controllers."""

    def compute(self, error: float, dt: float) -> float:
        """Given lateral error, return steering correction."""
        ...

    def reset(self) -> None:
        """Reset internal state."""
        ...

    @property
    def name(self) -> str:
        ...


@dataclass
class BangBangController:
    """
    Simplest possible controller: full correction in one direction or the other.

    If the line is left (error > 0), steer left with magnitude `strength`.
    If the line is right (error < 0), steer right.

    Fast to react but causes constant oscillation — the robot zigzags
    aggressively across the line.
    """
    strength: float = 0.08
    name: str = "Bang-Bang"

    def compute(self, error: float, dt: float) -> float:
        if abs(error) < 1e-6:
            return 0.0
        return self.strength if error > 0 else -self.strength

    def reset(self) -> None:
        pass


@dataclass
class PController:
    """
    Proportional controller: correction proportional to error.

    correction = Kp * error

    Smoother than bang-bang, but cannot eliminate steady-state error on curves
    because it only reacts to current error — it has no "memory" of past errors.
    """
    kp: float = 3.0
    name: str = "P-Only"

    def compute(self, error: float, dt: float) -> float:
        return self.kp * error

    def reset(self) -> None:
        pass


@dataclass
class PIDController:
    """
    Full PID controller for line following.

    correction = Kp * error + Ki * integral(error) + Kd * d(error)/dt

    - Kp: Proportional gain. Higher = more aggressive tracking.
    - Ki: Integral gain. Eliminates steady-state offset on curves.
           Too high = integral windup and overshoot.
    - Kd: Derivative gain. Damps oscillation by anticipating error changes.
           Too high = amplifies sensor noise.

    Integral windup protection: clamp the integral term to prevent
    it from growing unbounded when the robot is far from the line.
    """
    kp: float = 3.0
    ki: float = 0.5
    kd: float = 0.8
    integral_limit: float = 0.05  # Anti-windup clamp
    name: str = "PID"

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: float | None = field(default=None, init=False, repr=False)

    def compute(self, error: float, dt: float) -> float:
        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup clamping
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral

        # Derivative term (0 on first call)
        if self._prev_error is not None and dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0.0
        self._prev_error = error

        return p_term + i_term + d_term

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None


# ---------------------------------------------------------------------------
# Differential drive robot model
# ---------------------------------------------------------------------------

@dataclass
class DiffDriveRobot:
    """
    Simulates a differential-drive robot.

    State: (x, y, theta) where theta is heading in radians (0 = +x direction).
    Control: left and right wheel velocities.

    The robot moves according to standard differential drive kinematics:
      v = (v_right + v_left) / 2
      omega = (v_right - v_left) / wheel_base
      x += v * cos(theta) * dt
      y += v * sin(theta) * dt
      theta += omega * dt
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    wheel_base: float = 0.08  # 8 cm between wheels
    max_wheel_speed: float = 0.3  # m/s per wheel
    base_speed: float = 0.15  # m/s forward speed

    def set_wheel_speeds(
        self, correction: float
    ) -> tuple[float, float]:
        """
        Convert a steering correction to left/right wheel speeds.

        correction > 0 means "turn left" (slow left wheel, speed up right).
        correction < 0 means "turn right" (slow right wheel, speed up left).

        Returns (v_left, v_right) clamped to physical limits.
        """
        v_left = self.base_speed - correction
        v_right = self.base_speed + correction

        # Clamp to physical wheel speed limits
        v_left = max(-self.max_wheel_speed, min(self.max_wheel_speed, v_left))
        v_right = max(-self.max_wheel_speed, min(self.max_wheel_speed, v_right))

        return v_left, v_right

    def update(self, v_left: float, v_right: float, dt: float) -> None:
        """
        Update robot pose using differential drive kinematics.

        Uses Euler integration — adequate for small dt (1-10 ms).
        For higher accuracy you'd use Runge-Kutta, but for line-following
        the control loop frequency matters more than integration precision.
        """
        v = (v_right + v_left) / 2.0
        omega = (v_right - v_left) / self.wheel_base

        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt

        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Stores the full result of a simulation run for analysis."""
    controller_name: str
    times: list[float]
    positions: list[tuple[float, float]]
    headings: list[float]
    errors: list[float]
    corrections: list[float]
    wheel_speeds: list[tuple[float, float]]
    sensor_readings: list[list[float]]
    line_lost_count: int


def run_simulation(
    track: list[tuple[float, float]],
    controller: Controller,
    sensor_array: SensorArray | None = None,
    robot: DiffDriveRobot | None = None,
    dt: float = 0.005,
    max_time: float = 15.0,
    start_offset: float = 0.0,
) -> SimResult:
    """
    Run a line-following simulation.

    Args:
        track: List of (x, y) waypoints defining the line.
        controller: The control strategy to use.
        sensor_array: Sensor configuration (default: 5-sensor array).
        robot: Robot model (default: standard diff-drive).
        dt: Simulation timestep in seconds.
        max_time: Maximum simulation duration.
        start_offset: Initial lateral offset from track start (for testing).

    Returns:
        SimResult with full trajectory and performance data.
    """
    if sensor_array is None:
        sensor_array = SensorArray()
    if robot is None:
        robot = DiffDriveRobot()

    # Initialize robot at track start
    # Heading is computed from first two track points
    tx0, ty0 = track[0]
    tx1, ty1 = track[min(5, len(track) - 1)]
    start_heading = math.atan2(ty1 - ty0, tx1 - tx0)

    robot.x = tx0 + start_offset * (-math.sin(start_heading))
    robot.y = ty0 + start_offset * math.cos(start_heading)
    robot.theta = start_heading

    controller.reset()

    # Data logging
    times: list[float] = []
    positions: list[tuple[float, float]] = []
    headings: list[float] = []
    errors: list[float] = []
    corrections: list[float] = []
    wheel_speeds: list[tuple[float, float]] = []
    sensor_log: list[list[float]] = []
    line_lost_count = 0

    last_error = 0.0  # Fallback when line is lost
    t = 0.0

    while t < max_time:
        # 1. Read sensors
        readings = sensor_array.read(robot.x, robot.y, robot.theta, track)

        # 2. Estimate line position
        line_pos = sensor_array.estimate_line_position(readings)

        if line_pos is None:
            # Line lost — use last known error with decay
            # This is a common real-world strategy: keep turning in the last
            # known direction, but don't accelerate the turn
            error = last_error * 0.95
            line_lost_count += 1
        else:
            # line_pos < 0 means line is to robot's right → need negative correction (turn right)
            # line_pos > 0 means line is to robot's left → need positive correction (turn left)
            # So error = line_pos directly drives the correction in the correct direction
            error = line_pos
            last_error = error

        # 3. Compute control output
        correction = controller.compute(error, dt)

        # 4. Set wheel speeds
        v_left, v_right = robot.set_wheel_speeds(correction)

        # 5. Update robot pose
        robot.update(v_left, v_right, dt)

        # 6. Log data
        times.append(t)
        positions.append((robot.x, robot.y))
        headings.append(robot.theta)
        errors.append(error)
        corrections.append(correction)
        wheel_speeds.append((v_left, v_right))
        sensor_log.append(readings)

        t += dt

        # Stop if robot is close to track end
        end_x, end_y = track[-1]
        dist_to_end = math.hypot(robot.x - end_x, robot.y - end_y)
        if dist_to_end < 0.03 and t > 1.0:
            break

    return SimResult(
        controller_name=controller.name,
        times=times,
        positions=positions,
        headings=headings,
        errors=errors,
        corrections=corrections,
        wheel_speeds=wheel_speeds,
        sensor_readings=sensor_log,
        line_lost_count=line_lost_count,
    )


# ---------------------------------------------------------------------------
# Performance analysis
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Quantitative evaluation of a controller's performance."""
    controller_name: str
    mae: float               # Mean Absolute Error
    max_error: float          # Worst-case deviation
    error_std: float          # Oscillation measure
    line_lost_count: int      # Critical failures
    correction_smoothness: float  # Lower = smoother actuator commands
    total_time: float         # Time to traverse track


def analyze_performance(result: SimResult) -> PerformanceMetrics:
    """
    Compute performance metrics from a simulation result.

    These metrics tell you everything about the controller's quality:
    - MAE: Average tracking accuracy
    - Max error: Safety margin (how far does it ever stray?)
    - Error std: Oscillation (is the ride smooth or jerky?)
    - Line lost count: Reliability (did we ever completely lose the line?)
    - Correction smoothness: Actuator wear (how aggressively are motors commanded?)
    """
    abs_errors = [abs(e) for e in result.errors]
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    max_error = max(abs_errors) if abs_errors else 0.0

    mean_error = sum(result.errors) / len(result.errors) if result.errors else 0.0
    error_std = (
        sum((e - mean_error) ** 2 for e in result.errors) / len(result.errors)
    ) ** 0.5 if result.errors else 0.0

    # Correction smoothness: average absolute change between consecutive corrections
    if len(result.corrections) > 1:
        diffs = [
            abs(result.corrections[i + 1] - result.corrections[i])
            for i in range(len(result.corrections) - 1)
        ]
        smoothness = sum(diffs) / len(diffs)
    else:
        smoothness = 0.0

    return PerformanceMetrics(
        controller_name=result.controller_name,
        mae=mae,
        max_error=max_error,
        error_std=error_std,
        line_lost_count=result.line_lost_count,
        correction_smoothness=smoothness,
        total_time=result.times[-1] if result.times else 0.0,
    )


# ---------------------------------------------------------------------------
# Track builder — create test tracks of increasing difficulty
# ---------------------------------------------------------------------------

def make_test_track() -> list[tuple[float, float]]:
    """
    Build a test track with straights and curves of varying difficulty.

    Layout (each segment connects exactly to the next):
      1. Straight (1m along +x) — warm-up
      2. Gentle right curve (r=0.4m, 90°) — robot turns to face -y
      3. Straight (0.6m downward)
      4. Sharp left curve (r=0.2m, 90°) — tight turn, faces -x
      5. Straight (0.5m to the left)
      6. Gentle left curve (r=0.4m, 90°) — turns to face +y
      7. Final straight (0.4m upward)

    Total length: ~3.5 meters — typical for a tabletop line-following course.
    """
    segments = [
        # 1. Straight along +x axis
        {"type": "straight", "start": (0.0, 0.0), "end": (1.0, 0.0)},
        # 2. Gentle right 90° — center at (1.0, -0.4), r=0.4
        #    Start angle π/2 → (1.0, 0.0), end angle 0 → (1.4, -0.4)
        {"type": "arc", "center": (1.0, -0.4), "radius": 0.4,
         "start_angle": math.pi / 2, "end_angle": 0.0},
        # 3. Straight downward
        {"type": "straight", "start": (1.4, -0.4), "end": (1.4, -1.0)},
        # 4. Sharp left 90° — center at (1.2, -1.0), r=0.2
        #    Start angle 0 → (1.4, -1.0), end angle -π/2 → (1.2, -1.2)
        {"type": "arc", "center": (1.2, -1.0), "radius": 0.2,
         "start_angle": 0.0, "end_angle": -math.pi / 2},
        # 5. Straight to the left
        {"type": "straight", "start": (1.2, -1.2), "end": (0.7, -1.2)},
        # 6. Gentle left 90° — center at (0.7, -0.8), r=0.4
        #    Start angle -π/2 → (0.7, -1.2), end angle -π → (0.3, -0.8)
        {"type": "arc", "center": (0.7, -0.8), "radius": 0.4,
         "start_angle": -math.pi / 2, "end_angle": -math.pi},
        # 7. Final straight upward
        {"type": "straight", "start": (0.3, -0.8), "end": (0.3, -0.4)},
    ]
    return build_track(segments)


# ---------------------------------------------------------------------------
# Main — demonstrate and compare all three controllers
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Day 021: Line-Following Robot Simulation")
    print("=" * 70)

    # Build the test track
    track = make_test_track()
    print(f"\nTrack: {len(track)} waypoints")
    print(f"  Start: ({track[0][0]:.2f}, {track[0][1]:.2f})")
    print(f"  End:   ({track[-1][0]:.2f}, {track[-1][1]:.2f})")

    # Sensor configuration
    sensor = SensorArray(n_sensors=5, array_width=0.06, line_sigma=0.012)
    print(f"\nSensor array: {sensor.n_sensors} sensors across {sensor.array_width*100:.0f} cm")
    print(f"  Positions (mm): {[f'{p*1000:.1f}' for p in sensor.sensor_positions]}")

    # --- Test the sensor model ---
    print("\n" + "-" * 50)
    print("Sensor Model Demo")
    print("-" * 50)
    print("  Readings at various distances from line center:")
    for dist_mm in [0, 3, 6, 10, 15, 25]:
        dist = dist_mm / 1000.0
        reading = 1.0 - math.exp(-(dist ** 2) / (2 * sensor.line_sigma ** 2))
        print(f"    {dist_mm:3d} mm: {reading:.4f}  ({'ON LINE' if reading < 0.3 else 'transitioning' if reading < 0.7 else 'OFF LINE'})")

    # --- Run all three controllers ---
    controllers: list[Controller] = [
        BangBangController(strength=0.08),
        PController(kp=3.0),
        PIDController(kp=3.0, ki=0.5, kd=0.8),
    ]

    results: list[SimResult] = []
    metrics: list[PerformanceMetrics] = []

    print("\n" + "=" * 70)
    print("Running simulations...")
    print("=" * 70)

    for ctrl in controllers:
        result = run_simulation(
            track=track,
            controller=ctrl,
            sensor_array=sensor,
            dt=0.005,
            max_time=30.0,
            start_offset=0.005,  # Start 5mm off-center
        )
        perf = analyze_performance(result)
        results.append(result)
        metrics.append(perf)

        print(f"\n  {ctrl.name}:")
        print(f"    Sim time: {result.times[-1]:.2f}s ({len(result.times)} steps)")
        print(f"    Final pos: ({result.positions[-1][0]:.3f}, {result.positions[-1][1]:.3f})")

    # --- Performance comparison ---
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)

    # Header
    print(f"\n  {'Metric':<25} ", end="")
    for m in metrics:
        print(f"{m.controller_name:>12}", end="")
    print()
    print("  " + "-" * (25 + 12 * len(metrics)))

    # MAE (lower is better)
    print(f"  {'MAE (mm)':<25} ", end="")
    for m in metrics:
        print(f"{m.mae * 1000:>12.2f}", end="")
    print("  (lower = better)")

    # Max error
    print(f"  {'Max Error (mm)':<25} ", end="")
    for m in metrics:
        print(f"{m.max_error * 1000:>12.2f}", end="")
    print("  (lower = safer)")

    # Oscillation (error std)
    print(f"  {'Oscillation (mm)':<25} ", end="")
    for m in metrics:
        print(f"{m.error_std * 1000:>12.2f}", end="")
    print("  (lower = smoother)")

    # Correction smoothness
    print(f"  {'Cmd Smoothness':<25} ", end="")
    for m in metrics:
        print(f"{m.correction_smoothness:>12.4f}", end="")
    print("  (lower = less motor jerk)")

    # Line lost events
    print(f"  {'Line Lost Events':<25} ", end="")
    for m in metrics:
        print(f"{m.line_lost_count:>12d}", end="")
    print("  (0 = ideal)")

    # --- Detailed PID analysis ---
    pid_result = results[2]
    pid_metrics = metrics[2]

    print("\n" + "=" * 70)
    print("Detailed PID Controller Analysis")
    print("=" * 70)

    # Show error evolution over time
    n_samples = len(pid_result.errors)
    segment_size = n_samples // 5
    print("\n  Error evolution across track segments:")
    segment_names = ["Start/Straight", "Gentle Curve", "Transition", "Sharp Curve", "S-Curve/End"]

    for i, name in enumerate(segment_names):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, n_samples)
        seg_errors = [abs(e) for e in pid_result.errors[start_idx:end_idx]]
        if seg_errors:
            seg_mae = sum(seg_errors) / len(seg_errors)
            seg_max = max(seg_errors)
            print(f"    {name:<20}: MAE={seg_mae*1000:.2f}mm, Max={seg_max*1000:.2f}mm")

    # Show a snapshot of sensor readings mid-curve
    mid_idx = n_samples // 3
    print(f"\n  Sensor snapshot at t={pid_result.times[mid_idx]:.2f}s (mid-curve):")
    readings = pid_result.sensor_readings[mid_idx]
    positions = sensor.sensor_positions
    for pos, reading in zip(positions, readings):
        bar = "#" * int((1 - reading) * 30)
        print(f"    Sensor at {pos*1000:+6.1f}mm: {reading:.3f} |{bar}")
    line_est = sensor.estimate_line_position(readings)
    if line_est is not None:
        print(f"    -> Estimated line position: {line_est*1000:+.2f}mm from center")

    # --- Key insights ---
    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)

    bb_mae = metrics[0].mae
    p_mae = metrics[1].mae
    pid_mae = metrics[2].mae

    print(f"""
  1. Bang-Bang oscillates wildly (MAE={bb_mae*1000:.1f}mm) because it applies
     full correction regardless of error magnitude. Every correction overshoots.

  2. P-Only improves dramatically (MAE={p_mae*1000:.1f}mm) by scaling correction
     to error. But on sustained curves it develops steady-state offset because
     it has no memory of accumulated error.

  3. PID achieves best tracking (MAE={pid_mae*1000:.1f}mm) — the integral term
     eliminates steady-state offset and the derivative term damps oscillation.
     The improvement is most visible on the sharp curve sections.

  4. In real hardware, sensor noise limits how high you can set Kd (derivative
     gain). A Kalman filter on sensor readings (Day 016) would help.

  5. The 5-sensor array provides ~6cm of lateral coverage. If the robot deviates
     more than ~3cm, it loses the line entirely. Wider arrays or edge sensors
     provide more margin for aggressive maneuvers.
""")


if __name__ == "__main__":
    main()
