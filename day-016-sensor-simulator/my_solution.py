"""
Day 16: Sensor Reading Simulator — Your Implementation

Build a sensor simulation framework that models how robots perceive the world.
Fill in each class/function below. Run tests.py to check your work.

Key concepts to remember:
- Measurement = true_value + bias + gaussian_noise
- Bias drifts over time (random walk)
- Sensors have sample rates — they don't fire every tick
- Sensor fusion combines multiple measurements optimally
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Data Structures (provided — no changes needed)
# =============================================================================

@dataclass
class Vec2:
    x: float
    y: float

    def distance_to(self, other: "Vec2") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f})"


@dataclass
class RobotState:
    position: Vec2
    heading: float          # radians
    linear_vel: float       # m/s
    angular_vel: float      # rad/s
    timestamp: float        # seconds


@dataclass
class LineSegment:
    p1: Vec2
    p2: Vec2


# =============================================================================
# Base Sensor
# =============================================================================

class Sensor(ABC):
    """
    Abstract base class for all sensors.

    Hint: Think about what every sensor has in common:
    - noise_sigma, bias, bias_drift_sigma, rate_hz, min_val, max_val
    - Timing: only produce a reading when enough time has passed (1/rate_hz)
    - Noise model: measurement = true_value + bias + gaussian_noise
    - Bias drift: each sample, bias += gaussian(0, bias_drift_sigma)
    - Clamping: output must stay within [min_val, max_val]
    """

    def __init__(
        self,
        noise_sigma: float = 0.0,
        bias: float = 0.0,
        bias_drift_sigma: float = 0.0,
        rate_hz: float = 10.0,
        min_val: float = float("-inf"),
        max_val: float = float("inf"),
        name: str = "sensor",
    ):
        self.noise_sigma = noise_sigma
        self.bias = bias
        self.bias_drift_sigma = bias_drift_sigma
        self.rate_hz = rate_hz
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self._last_reading_time: float = -1.0
        self._initial_bias = bias

    def update(self, state: RobotState, **kwargs) -> Optional[float]:
        """
        Called every simulation tick. Returns a noisy reading if enough time
        has passed since last reading, otherwise None.

        Hint: Check timing, drift the bias, get true measurement from _measure(),
        add noise, clamp to range.
        """
        raise NotImplementedError("TODO: implement sensor update logic")

    @abstractmethod
    def _measure(self, state: RobotState, **kwargs) -> float:
        """Return the ideal (noise-free) measurement given the true state."""
        ...

    def reset(self) -> None:
        self.bias = self._initial_bias
        self._last_reading_time = -1.0


# =============================================================================
# LIDAR Range Sensor
# =============================================================================

class LidarSensor(Sensor):
    """
    Simulates a single-beam LIDAR that casts a ray into a 2D world.

    Hint: For ray-segment intersection, use parametric lines:
      Ray:     P = origin + t * direction,    t >= 0
      Segment: Q = p1 + s * (p2 - p1),       0 <= s <= 1
    Solve the 2x2 system. If both t >= 0 and 0 <= s <= 1, there's a hit at distance t.
    Return the nearest hit, or max_range if nothing is hit.
    """

    def __init__(
        self,
        angle_offset: float = 0.0,
        max_range: float = 10.0,
        noise_sigma: float = 0.02,
        rate_hz: float = 20.0,
        name: str = "lidar",
    ):
        super().__init__(
            noise_sigma=noise_sigma,
            rate_hz=rate_hz,
            min_val=0.05,
            max_val=max_range,
            name=name,
        )
        self.angle_offset = angle_offset
        self.max_range = max_range

    def _measure(self, state: RobotState, **kwargs) -> float:
        """Cast a ray from robot position and find nearest obstacle."""
        raise NotImplementedError("TODO: implement ray-segment intersection")

    def scan(self, state: RobotState, num_beams: int, fov: float, obstacles: list[LineSegment]) -> list[tuple[float, Optional[float]]]:
        """
        Perform a full scan with num_beams spread across fov radians.
        Returns list of (angle, distance) pairs.

        Hint: Temporarily change self.angle_offset for each beam, call _measure(),
        add noise manually, then restore the original offset.
        """
        raise NotImplementedError("TODO: implement LIDAR scan")


# =============================================================================
# IMU Sensor
# =============================================================================

class IMUSensor:
    """
    Simulates accelerometer + gyroscope with bias drift.

    Hint: Model gyro and accel independently. Each has its own bias that
    drifts via random walk. The gyro measures angular_vel, the accel measures
    linear acceleration (use 0 for constant velocity, simplified).

    Reading = true_value + bias + gaussian_noise
    Bias update: bias += gaussian(0, bias_drift_sigma) each sample
    """

    def __init__(
        self,
        accel_noise: float = 0.1,
        accel_bias_drift: float = 0.001,
        gyro_noise: float = 0.01,
        gyro_bias_drift: float = 0.0005,
        rate_hz: float = 100.0,
    ):
        self._gyro_noise = gyro_noise
        self._gyro_bias_drift = gyro_bias_drift
        self._accel_noise = accel_noise
        self._accel_bias_drift = accel_bias_drift
        self._rate_hz = rate_hz
        self._last_time = -1.0
        self._gyro_bias = 0.0
        self._accel_bias = 0.0

    def update(self, state: RobotState) -> Optional[dict]:
        """
        Returns {'angular_vel': ..., 'linear_accel': ...} or None if not time.
        """
        raise NotImplementedError("TODO: implement IMU update")

    def reset(self) -> None:
        self._last_time = -1.0
        self._gyro_bias = 0.0
        self._accel_bias = 0.0


# =============================================================================
# Wheel Encoder / Odometry Sensor
# =============================================================================

class OdometrySensor:
    """
    Simulates wheel encoders with dead reckoning integration.

    Hint: Odometry noise is PROPORTIONAL TO DISTANCE TRAVELED, not time.
    - Compute true displacement: dist = linear_vel * dt, dtheta = angular_vel * dt
    - Add noise proportional to |dist|
    - Integrate: use midpoint heading for better accuracy
      mid_heading = est_heading + noisy_dtheta / 2
      est_x += noisy_dist * cos(mid_heading)
      est_y += noisy_dist * sin(mid_heading)
    """

    def __init__(
        self,
        ticks_per_meter: float = 1000.0,
        noise_per_meter: float = 0.005,
        rate_hz: float = 50.0,
        wheel_base: float = 0.3,
    ):
        self.ticks_per_meter = ticks_per_meter
        self.noise_per_meter = noise_per_meter
        self.rate_hz = rate_hz
        self.wheel_base = wheel_base
        self._last_time = -1.0
        self.est_x = 0.0
        self.est_y = 0.0
        self.est_heading = 0.0

    def update(self, state: RobotState, dt: float) -> Optional[dict]:
        """
        Returns {'est_position': Vec2, 'est_heading': float} or None.
        """
        raise NotImplementedError("TODO: implement odometry update")

    def reset(self) -> None:
        self._last_time = -1.0
        self.est_x = 0.0
        self.est_y = 0.0
        self.est_heading = 0.0


# =============================================================================
# Sensor Fusion
# =============================================================================

def fuse_gaussian_measurements(
    measurements: list[tuple[float, float]]
) -> tuple[float, float]:
    """
    Fuse multiple independent Gaussian measurements of the same quantity.
    Each measurement is (value, sigma).

    Returns (fused_value, fused_sigma).

    Hint: The optimal estimate is the precision-weighted average:
      precision = 1 / sigma^2
      fused_value = sum(z_i * precision_i) / sum(precision_i)
      fused_variance = 1 / sum(precision_i)
    """
    raise NotImplementedError("TODO: implement Gaussian sensor fusion")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # Test 1: Sensor fusion
    print("Test 1: Sensor Fusion")
    print("-" * 40)
    fused_val, fused_sigma = fuse_gaussian_measurements([(5.1, 0.1), (4.8, 0.3)])
    print(f"Sensor A: 5.1 (sigma=0.1)")
    print(f"Sensor B: 4.8 (sigma=0.3)")
    print(f"Fused:    {fused_val:.4f} (sigma={fused_sigma:.4f})")
    print(f"Expected: heavily weighted toward A (more precise)")
    print()

    # Test 2: LIDAR
    print("Test 2: LIDAR Range Reading")
    print("-" * 40)
    lidar = LidarSensor(noise_sigma=0.02, max_range=10.0)
    state = RobotState(Vec2(5, 5), heading=0.0, linear_vel=0, angular_vel=0, timestamp=0.0)
    obstacles = [LineSegment(Vec2(8, 0), Vec2(8, 10))]  # Wall at x=8
    reading = lidar.update(state, obstacles=obstacles)
    print(f"Robot at (5,5) facing right, wall at x=8")
    print(f"LIDAR reading: {reading:.3f}m (true distance: 3.000m)")
    print()

    # Test 3: Odometry drift
    print("Test 3: Odometry Drift Over Distance")
    print("-" * 40)
    odom = OdometrySensor(noise_per_meter=0.01)
    state = RobotState(Vec2(0, 0), heading=0.0, linear_vel=1.0, angular_vel=0.0, timestamp=0.0)
    dt = 0.02
    for i in range(500):  # 10 seconds at 1 m/s = 10m traveled
        state.timestamp = i * dt
        state.position.x += state.linear_vel * dt
        result = odom.update(state, dt=dt)

    print(f"True position:  ({state.position.x:.3f}, 0.000)")
    print(f"Odom estimate:  ({odom.est_x:.3f}, {odom.est_y:.3f})")
    err = math.sqrt((state.position.x - odom.est_x)**2 + odom.est_y**2)
    print(f"Position error: {err:.4f}m after 10m traveled")
    print()

    # Test 4: IMU
    print("Test 4: IMU Reading")
    print("-" * 40)
    imu = IMUSensor(gyro_noise=0.01, gyro_bias_drift=0.001)
    state = RobotState(Vec2(0, 0), heading=0.0, linear_vel=0.0, angular_vel=0.5, timestamp=0.0)
    reading = imu.update(state)
    print(f"True angular vel: 0.5 rad/s")
    print(f"IMU gyro reading: {reading['angular_vel']:.4f} rad/s")
