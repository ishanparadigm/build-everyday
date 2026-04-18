"""
Day 16: Sensor Reading Simulator

A realistic sensor simulation framework for robotics. Models noise, bias drift,
range limits, and sensor fusion — the building blocks of robot perception.

Key insight: Every sensor measurement is a random variable. Algorithms that treat
sensor data as ground truth will fail on real hardware. This simulator teaches you
to think probabilistically about perception.
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Vec2:
    """2D vector / point. Used for positions and directions."""
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
    """Ground truth state of a differential-drive robot."""
    position: Vec2          # (x, y) in meters
    heading: float          # radians, 0 = +x axis, CCW positive
    linear_vel: float       # m/s
    angular_vel: float      # rad/s
    timestamp: float        # seconds


@dataclass
class LineSegment:
    """An obstacle wall defined by two endpoints."""
    p1: Vec2
    p2: Vec2


# =============================================================================
# Base Sensor
# =============================================================================

class Sensor(ABC):
    """
    Abstract base class for all sensors.

    Every sensor has:
    - noise_sigma: standard deviation of zero-mean Gaussian noise
    - bias: systematic offset (can drift over time)
    - rate_hz: how often the sensor produces readings
    - min_val / max_val: physical limits of the sensor's output range

    The update() method is called at the simulation rate. It only produces a
    reading when enough time has passed (based on rate_hz). This models the
    real behavior where sensors have fixed sample rates.
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
        Called every simulation tick. Returns a reading if it's time to sample,
        otherwise None. This enforces the sensor's actual sample rate.
        """
        period = 1.0 / self.rate_hz
        if state.timestamp - self._last_reading_time < period:
            return None

        self._last_reading_time = state.timestamp

        # Bias random walk: the bias drifts slightly each sample.
        # Over N samples, bias wanders by ~sqrt(N) * bias_drift_sigma.
        # This models gyroscope drift, temperature-dependent offset shifts, etc.
        if self.bias_drift_sigma > 0:
            self.bias += random.gauss(0, self.bias_drift_sigma)

        # Get the ideal (noiseless) measurement
        true_value = self._measure(state, **kwargs)

        # Add bias + Gaussian noise, then clamp to sensor range.
        # Clamping is critical: a LIDAR can't measure negative distances,
        # and an encoder can't count negative ticks.
        noisy = true_value + self.bias + random.gauss(0, self.noise_sigma)
        clamped = max(self.min_val, min(self.max_val, noisy))
        return clamped

    @abstractmethod
    def _measure(self, state: RobotState, **kwargs) -> float:
        """Return the ideal (noise-free) measurement given the true state."""
        ...

    def reset(self) -> None:
        """Reset sensor state (bias, timing) for a fresh simulation run."""
        self.bias = self._initial_bias
        self._last_reading_time = -1.0


# =============================================================================
# LIDAR Range Sensor
# =============================================================================

class LidarSensor(Sensor):
    """
    Simulates a single-beam LIDAR (or a scanning LIDAR at one angle).

    Casts a ray from the robot's position at a given angle offset from heading.
    Returns the distance to the nearest obstacle, or max_range if nothing is hit.

    The ray-segment intersection uses the parametric line intersection formula:
    Ray: P = origin + t * direction,  t >= 0
    Segment: Q = p1 + s * (p2 - p1),  0 <= s <= 1

    Solving the 2x2 system gives t and s. If both are in valid range, we have
    an intersection at distance t.
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
            min_val=0.05,       # Minimum range: 5cm (can't measure closer)
            max_val=max_range,
            name=name,
        )
        self.angle_offset = angle_offset
        self.max_range = max_range

    def _measure(self, state: RobotState, **kwargs) -> float:
        """Cast a ray and find the nearest obstacle intersection."""
        obstacles: list[LineSegment] = kwargs.get("obstacles", [])

        # Ray origin and direction
        angle = state.heading + self.angle_offset
        dx = math.cos(angle)
        dy = math.sin(angle)
        ox, oy = state.position.x, state.position.y

        nearest = self.max_range

        for seg in obstacles:
            # Parametric intersection of ray with line segment.
            # This is the standard 2D ray-segment test used in raycasting engines.
            seg_dx = seg.p2.x - seg.p1.x
            seg_dy = seg.p2.y - seg.p1.y

            # Denominator of the parametric solution (cross product of directions)
            denom = dx * seg_dy - dy * seg_dx
            if abs(denom) < 1e-10:
                # Ray and segment are parallel — no intersection
                continue

            # t = distance along ray to intersection point
            # s = parameter along segment (0 to 1 means between p1 and p2)
            t_num = (seg.p1.x - ox) * seg_dy - (seg.p1.y - oy) * seg_dx
            s_num = (seg.p1.x - ox) * dy - (seg.p1.y - oy) * dx

            t = t_num / denom
            s = s_num / denom

            # Valid hit: t >= 0 (in front of sensor) and 0 <= s <= 1 (on segment)
            if t >= 0 and 0 <= s <= 1:
                nearest = min(nearest, t)

        return nearest

    def scan(self, state: RobotState, num_beams: int, fov: float, obstacles: list[LineSegment]) -> list[tuple[float, Optional[float]]]:
        """
        Perform a full LIDAR scan: multiple beams spread across the FOV.
        Returns list of (angle, distance) pairs.

        This simulates a rotating LIDAR that takes num_beams measurements
        across the field of view in a single scan cycle.
        """
        readings = []
        original_offset = self.angle_offset
        for i in range(num_beams):
            # Spread beams evenly across FOV, centered on forward
            beam_angle = -fov / 2 + (fov / (num_beams - 1)) * i if num_beams > 1 else 0
            self.angle_offset = beam_angle

            # Force a reading regardless of timing for scan mode
            true_val = self._measure(state, obstacles=obstacles)
            noisy_val = true_val + self.bias + random.gauss(0, self.noise_sigma)
            clamped = max(self.min_val, min(self.max_val, noisy_val))
            readings.append((beam_angle, clamped))

        self.angle_offset = original_offset
        return readings


# =============================================================================
# IMU Sensor
# =============================================================================

class IMUSensor:
    """
    Simulates an Inertial Measurement Unit with accelerometer and gyroscope.

    The IMU is the most complex sensor to model because:
    1. Both accel and gyro have independent noise AND bias drift
    2. The gyro bias drift causes orientation error to grow as sqrt(t)
    3. Integrating noisy acceleration twice to get position causes error
       to grow as t^(3/2) — this is why pure IMU navigation fails fast

    We model each axis independently with its own Sensor instance.
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
        Returns IMU reading dict with 'angular_vel' and 'linear_accel',
        or None if not time to sample yet.
        """
        period = 1.0 / self._rate_hz
        if state.timestamp - self._last_time < period:
            return None

        self._last_time = state.timestamp

        # Gyro bias random walk
        self._gyro_bias += random.gauss(0, self._gyro_bias_drift)
        # Accel bias random walk
        self._accel_bias += random.gauss(0, self._accel_bias_drift)

        # Gyro reading: true angular velocity + bias + noise
        gyro_reading = (
            state.angular_vel
            + self._gyro_bias
            + random.gauss(0, self._gyro_noise)
        )

        # Accelerometer reading: we simplify to 1D (forward acceleration)
        # In reality you'd model 3 axes + gravity, but the key concepts are the same
        accel_reading = (
            0.0  # constant velocity → zero acceleration (simplified)
            + self._accel_bias
            + random.gauss(0, self._accel_noise)
        )

        return {"angular_vel": gyro_reading, "linear_accel": accel_reading}

    def reset(self) -> None:
        self._last_time = -1.0
        self._gyro_bias = 0.0
        self._accel_bias = 0.0


# =============================================================================
# Wheel Encoder / Odometry Sensor
# =============================================================================

class OdometrySensor:
    """
    Simulates wheel encoders on a differential-drive robot.

    Odometry estimates position by integrating wheel rotations:
      delta_distance = (left_ticks + right_ticks) / 2 * meters_per_tick
      delta_heading = (right_ticks - left_ticks) / wheel_base * meters_per_tick

    The critical insight: odometry error is PROPORTIONAL TO DISTANCE TRAVELED,
    not time elapsed. A stationary robot has zero odometry drift. A moving one
    accumulates error continuously. This is different from IMU drift (time-based).

    Furthermore, position error from odometry grows quadratically with distance
    because heading errors cause all subsequent distance estimates to be in the
    wrong direction. A 1-degree heading error at step 1 causes the robot to be
    off by sin(1 deg) * total_distance at the end.
    """

    def __init__(
        self,
        ticks_per_meter: float = 1000.0,
        noise_per_meter: float = 0.005,  # 0.5% of distance traveled
        rate_hz: float = 50.0,
        wheel_base: float = 0.3,  # distance between wheels in meters
    ):
        self.ticks_per_meter = ticks_per_meter
        self.noise_per_meter = noise_per_meter
        self.rate_hz = rate_hz
        self.wheel_base = wheel_base
        self._last_time = -1.0

        # Accumulated odometry estimate
        self.est_x = 0.0
        self.est_y = 0.0
        self.est_heading = 0.0

    def update(self, state: RobotState, dt: float) -> Optional[dict]:
        """
        Returns odometry estimate or None if not time to sample.
        The estimate is the integrated position — what the robot THINKS
        its position is based on wheel data alone.
        """
        period = 1.0 / self.rate_hz
        if state.timestamp - self._last_time < period:
            return None

        self._last_time = state.timestamp

        # True displacement this timestep
        true_dist = state.linear_vel * dt
        true_dtheta = state.angular_vel * dt

        # Add noise proportional to distance moved.
        # This models wheel slip, uneven terrain, tire wear, etc.
        dist_noise = random.gauss(0, abs(true_dist) * self.noise_per_meter + 1e-6)
        theta_noise = random.gauss(0, abs(true_dist) * self.noise_per_meter / self.wheel_base + 1e-6)

        noisy_dist = true_dist + dist_noise
        noisy_dtheta = true_dtheta + theta_noise

        # Dead reckoning integration: update estimated pose
        # We use the midpoint heading for better accuracy (trapezoidal integration)
        mid_heading = self.est_heading + noisy_dtheta / 2
        self.est_x += noisy_dist * math.cos(mid_heading)
        self.est_y += noisy_dist * math.sin(mid_heading)
        self.est_heading += noisy_dtheta

        return {
            "est_position": Vec2(self.est_x, self.est_y),
            "est_heading": self.est_heading,
        }

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
    Optimally fuse multiple independent Gaussian measurements of the same quantity.

    Each measurement is (value, sigma). Returns (fused_value, fused_sigma).

    The math: For independent Gaussians, the optimal (minimum variance) estimate is
    the precision-weighted average:

        fused_value = sum(z_i / sigma_i^2) / sum(1 / sigma_i^2)
        fused_variance = 1 / sum(1 / sigma_i^2)

    This is the core of Kalman filtering. The fused variance is ALWAYS smaller
    than any individual variance — combining sensors always helps (assuming
    correct noise models and independence).

    Args:
        measurements: list of (measured_value, noise_sigma) tuples

    Returns:
        (fused_value, fused_sigma) — the optimal combined estimate
    """
    if not measurements:
        raise ValueError("Need at least one measurement")
    if len(measurements) == 1:
        return measurements[0]

    # Sum of precisions (inverse variances)
    precision_sum = sum(1.0 / (sigma ** 2) for _, sigma in measurements)
    # Precision-weighted sum of values
    weighted_sum = sum(z / (sigma ** 2) for z, sigma in measurements)

    fused_value = weighted_sum / precision_sum
    fused_sigma = math.sqrt(1.0 / precision_sum)

    return fused_value, fused_sigma


# =============================================================================
# 2D World Simulation
# =============================================================================

@dataclass
class World:
    """
    A simple 2D world with line-segment obstacles (walls).
    The robot moves through this world collecting sensor data.
    """
    obstacles: list[LineSegment] = field(default_factory=list)

    @staticmethod
    def create_box_world(width: float = 10.0, height: float = 10.0) -> "World":
        """Create a rectangular room with some internal walls."""
        walls = [
            # Room boundaries
            LineSegment(Vec2(0, 0), Vec2(width, 0)),
            LineSegment(Vec2(width, 0), Vec2(width, height)),
            LineSegment(Vec2(width, height), Vec2(0, height)),
            LineSegment(Vec2(0, height), Vec2(0, 0)),
            # Internal obstacles
            LineSegment(Vec2(3, 0), Vec2(3, 4)),       # vertical wall
            LineSegment(Vec2(6, 6), Vec2(6, 10)),      # vertical wall
            LineSegment(Vec2(4, 5), Vec2(8, 5)),       # horizontal wall
        ]
        return World(obstacles=walls)


def simulate_robot_path(
    world: World,
    duration: float = 10.0,
    dt: float = 0.01,
) -> None:
    """
    Run a full simulation: robot moves through the world, sensors collect data,
    and we compare true state vs sensor estimates.

    The robot follows a simple curved path (constant linear + angular velocity)
    to demonstrate how sensor errors accumulate over time.
    """
    # --- Setup sensors ---
    lidar = LidarSensor(noise_sigma=0.02, max_range=8.0, rate_hz=20.0, name="lidar_front")
    imu = IMUSensor(gyro_noise=0.01, gyro_bias_drift=0.0005, rate_hz=100.0)
    odom = OdometrySensor(noise_per_meter=0.005, rate_hz=50.0)

    # Second range sensor for fusion demo (noisier, like an ultrasonic)
    lidar_noisy = LidarSensor(noise_sigma=0.08, max_range=5.0, rate_hz=20.0, name="ultrasonic")

    # --- Robot trajectory: gentle curve ---
    linear_vel = 0.5    # m/s
    angular_vel = 0.2   # rad/s — gentle left turn

    # Start in center of room, facing right
    state = RobotState(
        position=Vec2(5.0, 3.0),
        heading=0.0,
        linear_vel=linear_vel,
        angular_vel=angular_vel,
        timestamp=0.0,
    )

    # --- Logging ---
    log_interval = 1.0  # Print status every second
    next_log_time = 0.0

    true_positions: list[Vec2] = []
    odom_positions: list[Vec2] = []
    lidar_readings: list[float] = []
    fused_readings: list[tuple[float, float]] = []

    print("=" * 70)
    print("SENSOR SIMULATION — Robot moving through 2D world")
    print("=" * 70)
    print(f"Duration: {duration}s | dt: {dt}s | Path: constant curve")
    print(f"Sensors: LIDAR (sigma=0.02m), Ultrasonic (sigma=0.08m), IMU, Odometry")
    print()

    steps = int(duration / dt)
    for step in range(steps):
        t = step * dt
        state.timestamp = t

        # --- Update true state (ground truth) ---
        state.heading += angular_vel * dt
        state.position.x += linear_vel * math.cos(state.heading) * dt
        state.position.y += linear_vel * math.sin(state.heading) * dt

        true_positions.append(Vec2(state.position.x, state.position.y))

        # --- Collect sensor readings ---
        lidar_val = lidar.update(state, obstacles=world.obstacles)
        lidar_noisy_val = lidar_noisy.update(state, obstacles=world.obstacles)
        imu_val = imu.update(state)
        odom_val = odom.update(state, dt=dt)

        if lidar_val is not None:
            lidar_readings.append(lidar_val)

        # Sensor fusion demo: when both range sensors fire, fuse them
        if lidar_val is not None and lidar_noisy_val is not None:
            fused_val, fused_sigma = fuse_gaussian_measurements([
                (lidar_val, 0.02),
                (lidar_noisy_val, 0.08),
            ])
            fused_readings.append((fused_val, fused_sigma))

        if odom_val is not None:
            odom_positions.append(odom_val["est_position"])

        # --- Periodic logging ---
        if t >= next_log_time:
            print(f"t={t:5.1f}s | True pos: {state.position} | Heading: {math.degrees(state.heading):6.1f} deg")
            if odom_val is not None:
                odom_pos = odom_val["est_position"]
                err = state.position.distance_to(odom_pos)
                print(f"         | Odom pos: {odom_pos} | Odom error: {err:.4f}m")
            if lidar_val is not None:
                print(f"         | LIDAR range: {lidar_val:.3f}m", end="")
                if lidar_noisy_val is not None and fused_readings:
                    fv, fs = fused_readings[-1]
                    print(f" | Ultrasonic: {lidar_noisy_val:.3f}m | Fused: {fv:.3f}m (sigma={fs:.4f})")
                else:
                    print()
            if imu_val is not None:
                print(f"         | IMU gyro: {imu_val['angular_vel']:.4f} rad/s (true: {angular_vel:.4f})")
            print()
            next_log_time += log_interval

    # --- Summary statistics ---
    print("=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)

    # Odometry drift analysis
    if odom_positions:
        final_odom = odom_positions[-1]
        final_true = true_positions[-1]
        odom_error = final_true.distance_to(final_odom)
        total_dist = linear_vel * duration
        print(f"\nOdometry:")
        print(f"  Final true position:  {final_true}")
        print(f"  Final odom estimate:  {final_odom}")
        print(f"  Position error:       {odom_error:.4f}m")
        print(f"  Error as % of travel: {odom_error / total_dist * 100:.2f}%")

    # LIDAR noise analysis
    if lidar_readings:
        mean_range = sum(lidar_readings) / len(lidar_readings)
        variance = sum((r - mean_range) ** 2 for r in lidar_readings) / len(lidar_readings)
        print(f"\nLIDAR ({len(lidar_readings)} readings):")
        print(f"  Mean range:     {mean_range:.3f}m")
        print(f"  Std deviation:  {math.sqrt(variance):.4f}m")

    # Sensor fusion improvement
    if fused_readings:
        # The fused sigma should be ~0.0194m (vs 0.02 for LIDAR alone, 0.08 for ultrasonic)
        avg_fused_sigma = sum(fs for _, fs in fused_readings) / len(fused_readings)
        print(f"\nSensor Fusion:")
        print(f"  LIDAR sigma:       0.0200m")
        print(f"  Ultrasonic sigma:  0.0800m")
        print(f"  Fused sigma:       {avg_fused_sigma:.4f}m")
        improvement = (1 - avg_fused_sigma / 0.02) * 100
        print(f"  Improvement over LIDAR alone: {improvement:.1f}%")

    # IMU bias drift
    print(f"\nIMU gyro bias drift: {imu._gyro_bias:.6f} rad/s (started at 0)")
    print(f"  After {duration}s, this bias would cause {math.degrees(imu._gyro_bias * duration):.2f} deg heading error")

    # LIDAR scan demo
    print("\n" + "=" * 70)
    print("LIDAR FULL SCAN DEMO (36 beams, 360 deg FOV)")
    print("=" * 70)
    scan = lidar.scan(state, num_beams=36, fov=2 * math.pi, obstacles=world.obstacles)
    for angle, dist in scan:
        bar_len = int(dist / 8.0 * 40)  # Scale to 40 chars max
        bar = "#" * bar_len
        print(f"  {math.degrees(angle):7.1f} deg | {dist:5.2f}m | {bar}")


if __name__ == "__main__":
    random.seed(42)  # Reproducible results

    # Create world and run simulation
    world = World.create_box_world(10.0, 10.0)
    simulate_robot_path(world, duration=10.0, dt=0.01)

    # --- Fusion demo with controlled experiment ---
    print("\n" + "=" * 70)
    print("SENSOR FUSION CONTROLLED EXPERIMENT")
    print("=" * 70)
    print("\nMeasuring the same true distance 100 times with two sensors:")
    print("  Sensor A: sigma = 0.10m (precise)")
    print("  Sensor B: sigma = 0.30m (noisy)")
    print()

    true_distance = 5.0
    errors_a = []
    errors_b = []
    errors_fused = []

    for _ in range(100):
        # Simulate two independent measurements
        z_a = true_distance + random.gauss(0, 0.10)
        z_b = true_distance + random.gauss(0, 0.30)

        # Fuse them
        fused, fused_sigma = fuse_gaussian_measurements([(z_a, 0.10), (z_b, 0.30)])

        errors_a.append(abs(z_a - true_distance))
        errors_b.append(abs(z_b - true_distance))
        errors_fused.append(abs(fused - true_distance))

    mae_a = sum(errors_a) / len(errors_a)
    mae_b = sum(errors_b) / len(errors_b)
    mae_fused = sum(errors_fused) / len(errors_fused)

    print(f"  Mean Absolute Error:")
    print(f"    Sensor A alone:  {mae_a:.4f}m")
    print(f"    Sensor B alone:  {mae_b:.4f}m")
    print(f"    Fused estimate:  {mae_fused:.4f}m")
    print(f"\n  Fused is {(1 - mae_fused/mae_a)*100:.1f}% better than the BEST individual sensor!")
    print(f"  Theoretical fused sigma: {fused_sigma:.4f}m (vs A: 0.10m, B: 0.30m)")
