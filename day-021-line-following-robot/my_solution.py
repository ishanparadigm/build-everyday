"""
Day 021: Line-Following Robot Logic — Your Implementation

Build a line-following robot simulation with reflectance sensors and
multiple control strategies. Fill in each function below.

Run tests with: python3 -m pytest tests.py -v
Run this file:  python3 my_solution.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Track builder
# ---------------------------------------------------------------------------

def build_track(segments: list[dict]) -> list[tuple[float, float]]:
    """
    Build a track as densely-sampled (x, y) waypoints.

    Each segment dict has:
      - type: "straight" or "arc"
      - For straight: "start" (x,y), "end" (x,y)
      - For arc: "center" (x,y), "radius", "start_angle", "end_angle" (radians)
      - "resolution": points per unit length (default 10)

    Returns a list of (x, y) points.

    Hint: For straights, linearly interpolate between start and end.
    For arcs, sweep the angle from start_angle to end_angle and compute
    x = cx + r*cos(angle), y = cy + r*sin(angle).
    Use resolution default of 100 points per unit length.
    """
    raise NotImplementedError("TODO: implement this")


def closest_point_on_track(
    track: list[tuple[float, float]], px: float, py: float
) -> tuple[float, float, float]:
    """
    Find the closest point on the track to query point (px, py).

    Returns (closest_x, closest_y, distance).

    Hint: For each consecutive pair of waypoints, project (px, py) onto
    the line segment and clamp the projection parameter to [0, 1].
    Keep the closest result across all segments.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Sensor array
# ---------------------------------------------------------------------------

@dataclass
class SensorArray:
    """
    Simulates an array of infrared reflectance sensors.

    Attributes:
        n_sensors: Number of sensors (typically 3-8)
        array_width: Total width in meters
        line_sigma: Gaussian width parameter for line detection
    """
    n_sensors: int = 5
    array_width: float = 0.06
    line_sigma: float = 0.012

    @property
    def sensor_positions(self) -> list[float]:
        """
        Return lateral offset of each sensor from robot center.
        Negative = left, positive = right.
        Evenly spaced across array_width.
        """
        raise NotImplementedError("TODO: implement this")

    def read(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        track: list[tuple[float, float]],
    ) -> list[float]:
        """
        Compute sensor readings. Each reading in [0, 1]:
          0 = directly on line, 1 = far from line.

        For each sensor:
        1. Compute world position: offset perpendicular to robot heading
        2. Find distance to closest track point
        3. Apply Gaussian model: reading = 1 - exp(-d^2 / (2*sigma^2))

        Hint: Perpendicular-left of heading (cos(h), sin(h)) is (-sin(h), cos(h)).
        """
        raise NotImplementedError("TODO: implement this")

    def estimate_line_position(self, readings: list[float]) -> float | None:
        """
        Weighted average of sensor positions using (1 - reading) as weights.

        Returns estimated lateral offset of line from robot center, or
        None if line is not detected (total weight < 0.05).

        Formula: position = sum((1 - r_i) * x_i) / sum(1 - r_i)
        """
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

@dataclass
class BangBangController:
    """
    Full correction in one direction or the other.
    If error > 0, return +strength. If error < 0, return -strength.
    """
    strength: float = 0.08
    name: str = "Bang-Bang"

    def compute(self, error: float, dt: float) -> float:
        """Hint: Just check the sign of the error."""
        raise NotImplementedError("TODO: implement this")

    def reset(self) -> None:
        pass


@dataclass
class PController:
    """
    Proportional controller: correction = Kp * error.
    """
    kp: float = 3.0
    name: str = "P-Only"

    def compute(self, error: float, dt: float) -> float:
        """Hint: One line of code."""
        raise NotImplementedError("TODO: implement this")

    def reset(self) -> None:
        pass


@dataclass
class PIDController:
    """
    PID controller with anti-windup on the integral term.

    Hint: Track _integral and _prev_error as internal state.
    P = Kp * error
    I = Ki * integral (clamped to [-integral_limit, integral_limit])
    D = Kd * (error - prev_error) / dt
    """
    kp: float = 3.0
    ki: float = 0.5
    kd: float = 0.8
    integral_limit: float = 0.05
    name: str = "PID"

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: float | None = field(default=None, init=False, repr=False)

    def compute(self, error: float, dt: float) -> float:
        raise NotImplementedError("TODO: implement this")

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None


# ---------------------------------------------------------------------------
# Robot model
# ---------------------------------------------------------------------------

@dataclass
class DiffDriveRobot:
    """
    Differential-drive robot with (x, y, theta) state.

    Kinematics:
      v = (v_right + v_left) / 2
      omega = (v_right - v_left) / wheel_base
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    wheel_base: float = 0.08
    max_wheel_speed: float = 0.3
    base_speed: float = 0.15

    def set_wheel_speeds(self, correction: float) -> tuple[float, float]:
        """
        Convert steering correction to (v_left, v_right).
        correction > 0 = turn left (slow left, speed up right).
        Clamp to [-max_wheel_speed, max_wheel_speed].
        """
        raise NotImplementedError("TODO: implement this")

    def update(self, v_left: float, v_right: float, dt: float) -> None:
        """
        Update pose using differential drive kinematics.
        Don't forget to normalize theta to [-pi, pi].

        Hint: Use math.atan2(sin(theta), cos(theta)) for normalization.
        """
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
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
    controller,
    sensor_array: SensorArray | None = None,
    robot: DiffDriveRobot | None = None,
    dt: float = 0.005,
    max_time: float = 15.0,
    start_offset: float = 0.0,
) -> SimResult:
    """
    Run a line-following simulation.

    Loop:
    1. Read sensors
    2. Estimate line position -> error
    3. Controller computes correction
    4. Set wheel speeds from correction
    5. Update robot pose
    6. Log everything

    Hint: When line is lost (estimate returns None), use last known error * 0.95.
    Error convention: error = line_pos (line_pos < 0 means line is right, turn right).
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Performance analysis
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    controller_name: str
    mae: float
    max_error: float
    error_std: float
    line_lost_count: int
    correction_smoothness: float
    total_time: float


def analyze_performance(result: SimResult) -> PerformanceMetrics:
    """
    Compute MAE, max error, error std, line lost count, and correction smoothness.

    Smoothness = average |correction[i+1] - correction[i]|.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Main — test your implementation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a simple track to test with
    segments = [
        {"type": "straight", "start": (0.0, 0.0), "end": (0.5, 0.0)},
        {"type": "arc", "center": (0.5, -0.3), "radius": 0.3,
         "start_angle": math.pi / 2, "end_angle": 0.0},
    ]
    track = build_track(segments)
    print(f"Track built: {len(track)} waypoints")

    # Test sensor array
    sensor = SensorArray(n_sensors=5)
    readings = sensor.read(0.0, 0.0, 0.0, track)
    print(f"Sensor readings at origin: {[f'{r:.3f}' for r in readings]}")
    line_pos = sensor.estimate_line_position(readings)
    print(f"Estimated line position: {line_pos}")

    # Test each controller
    for ctrl in [BangBangController(), PController(), PIDController()]:
        result = run_simulation(track, ctrl, sensor, dt=0.005, max_time=10.0)
        perf = analyze_performance(result)
        print(f"\n{ctrl.name}: MAE={perf.mae*1000:.2f}mm, "
              f"Max={perf.max_error*1000:.2f}mm, Lost={perf.line_lost_count}")
