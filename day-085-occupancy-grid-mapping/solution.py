"""
Day 85: Occupancy Grid Mapping

Build a probabilistic 2D map from noisy range sensor data using log-odds
representation and Bresenham ray casting.

This implementation simulates a lidar-equipped robot exploring a 2D environment,
fusing hundreds of noisy range measurements into a coherent occupancy grid.
"""

import math
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Bresenham's Line Algorithm
# ---------------------------------------------------------------------------
# Why Bresenham? We need to know EXACTLY which grid cells a laser beam passes
# through. Floating-point line interpolation can miss cells or double-count
# them. Bresenham uses only integer arithmetic and guarantees we visit every
# cell the line crosses — no more, no fewer.

def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Return all grid cells along the line from (x0,y0) to (x1,y1).

    Uses Bresenham's line algorithm — integer arithmetic only.
    The returned list includes both endpoints and is ordered from start to end.
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    # Step direction: +1 or -1 in each axis
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    # The "error" tracks accumulated deviation from the true line.
    # We start with the difference and adjust as we step.
    err = dx - dy

    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        # e2 = 2*err avoids a division by 2 — classic Bresenham trick
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return cells


# ---------------------------------------------------------------------------
# 2. OccupancyGrid class
# ---------------------------------------------------------------------------

class OccupancyGrid:
    """
    A 2D occupancy grid using log-odds representation.

    Why log-odds instead of raw probabilities?
    - Bayes updates become addition instead of multiplication → no underflow
    - Unknown = 0, so initialization is free (np.zeros)
    - Symmetric: positive = occupied, negative = free
    - Clamping log-odds bounds confidence, allowing map adaptation

    Parameters:
        width, height: World dimensions in meters
        resolution: Meters per grid cell (smaller = more detail, more memory)
        origin: World coordinates of the grid's bottom-left corner
        l_occ: Log-odds increment for occupied cells (positive)
        l_free: Log-odds increment for free cells (negative)
        l_min, l_max: Clamping bounds to prevent over-confidence
    """

    def __init__(
        self,
        width: float = 20.0,
        height: float = 20.0,
        resolution: float = 0.1,
        origin: Tuple[float, float] = (0.0, 0.0),
        l_occ: float = 0.85,
        l_free: float = -0.4,
        l_min: float = -5.0,
        l_max: float = 5.0,
    ):
        self.resolution = resolution
        self.origin = origin
        self.l_occ = l_occ
        self.l_free = l_free
        self.l_min = l_min
        self.l_max = l_max

        # Grid dimensions in cells
        self.grid_w = int(math.ceil(width / resolution))
        self.grid_h = int(math.ceil(height / resolution))

        # Log-odds grid — initialized to 0 (P=0.5, unknown)
        self.log_odds = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)

    # --- Coordinate transforms ---
    # These convert between continuous world coordinates and discrete grid indices.
    # The origin offset ensures the grid can represent any region of the world,
    # not just areas starting at (0,0).

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices."""
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell center)."""
        wx = gx * self.resolution + self.origin[0] + self.resolution / 2
        wy = gy * self.resolution + self.origin[1] + self.resolution / 2
        return wx, wy

    def in_bounds(self, gx: int, gy: int) -> bool:
        """Check if grid indices are within the grid."""
        return 0 <= gx < self.grid_w and 0 <= gy < self.grid_h

    # --- Core update ---

    def update_cell(self, gx: int, gy: int, log_odds_update: float) -> None:
        """
        Update a single cell's log-odds and clamp to bounds.

        Why clamp? Without it, a cell observed as occupied 10,000 times would
        need 10,000 free observations to become uncertain. Clamping at ±5
        (P ≈ 0.007 to 0.993) keeps the map adaptable to environmental changes.
        """
        if self.in_bounds(gx, gy):
            self.log_odds[gy, gx] += log_odds_update
            self.log_odds[gy, gx] = np.clip(
                self.log_odds[gy, gx], self.l_min, self.l_max
            )

    def inverse_sensor_model(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        beam_angle: float,
        measured_range: float,
        max_range: float,
    ) -> None:
        """
        Update the grid for a single range measurement.

        The inverse sensor model answers: "Given that the sensor at pose
        (robot_x, robot_y, robot_theta) measured range z at angle beam_angle,
        what can we infer about each cell?"

        - Cells along the ray (before endpoint): FREE — the beam passed through
        - Cell at the endpoint: OCCUPIED — the beam hit something
        - Cells beyond endpoint: UNKNOWN — no information, leave unchanged

        If measured_range >= max_range, the beam didn't hit anything — all cells
        along the ray are marked free, but no cell is marked occupied.
        """
        # Absolute angle of this beam in world frame
        angle = robot_theta + beam_angle

        # Endpoint of the beam in world coordinates
        # This is where the sensor says the obstacle is
        hit_x = robot_x + measured_range * math.cos(angle)
        hit_y = robot_y + measured_range * math.sin(angle)

        # Convert robot position and hit point to grid coordinates
        gx0, gy0 = self.world_to_grid(robot_x, robot_y)
        gx1, gy1 = self.world_to_grid(hit_x, hit_y)

        # Trace the ray using Bresenham
        ray_cells = bresenham(gx0, gy0, gx1, gy1)

        # All cells EXCEPT the last one are free (beam passed through)
        for cell in ray_cells[:-1]:
            self.update_cell(cell[0], cell[1], self.l_free)

        # The last cell is occupied — but only if the beam actually hit something
        # (i.e., the range is less than max_range)
        if measured_range < max_range and ray_cells:
            last = ray_cells[-1]
            self.update_cell(last[0], last[1], self.l_occ)

    def update_scan(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        ranges: np.ndarray,
        angles: np.ndarray,
        max_range: float,
    ) -> None:
        """
        Process a full lidar scan (multiple beams) from a given pose.

        Each beam is independent — this is what makes occupancy grid mapping
        embarrassingly parallelizable on GPUs. Here we process sequentially
        for clarity.
        """
        for r, a in zip(ranges, angles):
            self.inverse_sensor_model(robot_x, robot_y, robot_theta, a, r, max_range)

    # --- Probability conversion ---

    def get_probability_map(self) -> np.ndarray:
        """
        Convert log-odds grid to probability grid.

        P = 1 - 1/(1 + exp(l))

        This is only needed for visualization or export — all internal
        computation stays in log-odds for numerical stability.
        """
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def get_map_stats(self) -> dict:
        """Return statistics about the current map state."""
        prob = self.get_probability_map()
        total_cells = self.grid_w * self.grid_h
        # Thresholds for classification
        free_cells = int(np.sum(prob < 0.3))
        occupied_cells = int(np.sum(prob > 0.7))
        unknown_cells = total_cells - free_cells - occupied_cells
        return {
            "total_cells": total_cells,
            "free_cells": free_cells,
            "occupied_cells": occupied_cells,
            "unknown_cells": unknown_cells,
            "free_pct": free_cells / total_cells * 100,
            "occupied_pct": occupied_cells / total_cells * 100,
            "unknown_pct": unknown_cells / total_cells * 100,
        }


# ---------------------------------------------------------------------------
# 3. Environment Simulation
# ---------------------------------------------------------------------------
# We need a ground-truth environment to simulate lidar readings against.
# This is a simple 2D world with line-segment walls.

class Environment:
    """
    A 2D environment defined by line-segment walls.

    Provides ray casting against the ground truth geometry to simulate
    what a lidar sensor would measure. In the real world, this is replaced
    by actual sensor hardware.
    """

    def __init__(self, walls: List[Tuple[float, float, float, float]]):
        """
        walls: List of (x1, y1, x2, y2) line segments representing obstacles.
        """
        self.walls = walls

    def cast_ray(
        self, ox: float, oy: float, angle: float, max_range: float
    ) -> float:
        """
        Cast a single ray from (ox, oy) at the given angle.
        Returns the distance to the nearest wall hit, or max_range if no hit.

        Uses ray-segment intersection: for each wall segment, compute the
        intersection point (if any) and return the closest one.

        The math: parameterize the ray as R(t) = O + t*D and the segment as
        S(u) = A + u*(B-A). Solve for t and u simultaneously. Valid intersection
        requires t > 0 (forward along ray) and 0 <= u <= 1 (on the segment).
        """
        dx = math.cos(angle)
        dy = math.sin(angle)
        closest = max_range

        for x1, y1, x2, y2 in self.walls:
            # Wall direction vector
            wx = x2 - x1
            wy = y2 - y1

            # Denominator of the parametric intersection
            denom = dx * wy - dy * wx
            if abs(denom) < 1e-10:
                # Ray is parallel to wall — no intersection
                continue

            # Parameters for intersection point
            t = ((x1 - ox) * wy - (y1 - oy) * wx) / denom
            u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom

            # t > 0: intersection is in front of the sensor
            # 0 <= u <= 1: intersection is on the wall segment
            if t > 0 and 0 <= u <= 1:
                closest = min(closest, t)

        return closest

    def simulate_lidar(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        num_beams: int = 360,
        max_range: float = 10.0,
        noise_std: float = 0.05,
        fov: float = 2 * math.pi,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a full lidar scan from the given robot pose.

        Returns (ranges, angles) where:
        - ranges: measured distance for each beam (with Gaussian noise)
        - angles: beam angles relative to robot heading

        noise_std models real sensor noise — typically 1-5cm for a good lidar.
        Without noise, the map would converge instantly; with noise, we see how
        probabilistic fusion handles measurement uncertainty.
        """
        angles = np.linspace(-fov / 2, fov / 2, num_beams, endpoint=False)
        ranges = np.zeros(num_beams)

        for i, beam_angle in enumerate(angles):
            true_range = self.cast_ray(
                robot_x, robot_y, robot_theta + beam_angle, max_range
            )
            # Add Gaussian noise, but clamp to [0, max_range]
            noisy_range = true_range + np.random.normal(0, noise_std)
            ranges[i] = np.clip(noisy_range, 0.0, max_range)

        return ranges, angles


# ---------------------------------------------------------------------------
# 4. Helper: Create a test environment
# ---------------------------------------------------------------------------

def create_room_environment() -> Environment:
    """
    Create a room with interior walls and obstacles.

    Layout (20m x 20m):
    - Outer walls forming the room boundary
    - An L-shaped interior wall
    - Two rectangular obstacles (like furniture)

    This gives a realistic test case with corners, corridors, and occluded areas.
    """
    walls = [
        # Outer walls (20m x 20m room)
        (1.0, 1.0, 19.0, 1.0),    # bottom
        (19.0, 1.0, 19.0, 19.0),  # right
        (19.0, 19.0, 1.0, 19.0),  # top
        (1.0, 19.0, 1.0, 1.0),    # left
        # L-shaped interior wall
        (7.0, 1.0, 7.0, 10.0),    # vertical part
        (7.0, 10.0, 12.0, 10.0),  # horizontal part
        # Rectangular obstacle 1 (like a table)
        (14.0, 4.0, 17.0, 4.0),
        (17.0, 4.0, 17.0, 6.0),
        (17.0, 6.0, 14.0, 6.0),
        (14.0, 6.0, 14.0, 4.0),
        # Rectangular obstacle 2
        (10.0, 14.0, 13.0, 14.0),
        (13.0, 14.0, 13.0, 17.0),
        (13.0, 17.0, 10.0, 17.0),
        (10.0, 17.0, 10.0, 14.0),
    ]
    return Environment(walls)


def generate_exploration_path(
    num_poses: int = 50,
) -> List[Tuple[float, float, float]]:
    """
    Generate a robot trajectory that explores the room.

    The path visits multiple areas to ensure good coverage. A real exploration
    system would use frontier-based exploration (seeking unknown cells), but
    for this demo we hardcode a reasonable path.

    Returns list of (x, y, theta) poses.
    """
    path = []
    # Segment 1: Move along the bottom-left area
    for i in range(12):
        t = i / 11
        x = 3.0 + t * 3.0
        y = 3.0
        theta = 0.0  # facing right
        path.append((x, y, theta))

    # Segment 2: Move up along the left side of the L-wall
    for i in range(10):
        t = i / 9
        x = 4.0
        y = 4.0 + t * 10.0
        theta = math.pi / 2  # facing up
        path.append((x, y, theta))

    # Segment 3: Move across the top
    for i in range(12):
        t = i / 11
        x = 4.0 + t * 12.0
        y = 16.0
        theta = 0.0  # facing right
        path.append((x, y, theta))

    # Segment 4: Move down the right side
    for i in range(10):
        t = i / 9
        x = 16.0
        y = 15.0 - t * 10.0
        theta = -math.pi / 2  # facing down
        path.append((x, y, theta))

    # Segment 5: Move across the middle-right area
    for i in range(6):
        t = i / 5
        x = 15.0 - t * 5.0
        y = 7.0
        theta = math.pi  # facing left
        path.append((x, y, theta))

    return path


# ---------------------------------------------------------------------------
# 5. Text-based map visualization
# ---------------------------------------------------------------------------

def render_map_ascii(grid: OccupancyGrid, downsample: int = 4) -> str:
    """
    Render the occupancy grid as ASCII art.

    Downsamples the grid for terminal display. Each character represents
    a block of cells — we take the mean probability over the block.

    Characters:
      '##' = occupied (P > 0.7)
      '..' = free (P < 0.3)
      '  ' = unknown (0.3 <= P <= 0.7)
    """
    prob = grid.get_probability_map()
    h, w = prob.shape
    lines = []

    # Iterate top-to-bottom for natural display orientation
    for gy in range(h - 1, -1, -downsample):
        row = ""
        for gx in range(0, w, downsample):
            # Average probability in this block
            block = prob[
                max(0, gy - downsample + 1) : gy + 1,
                gx : min(w, gx + downsample),
            ]
            avg_p = np.mean(block)
            if avg_p > 0.7:
                row += "##"
            elif avg_p < 0.3:
                row += ".."
            else:
                row += "  "
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main: Run the full mapping pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)  # Reproducibility

    print("=" * 70)
    print("OCCUPANCY GRID MAPPING")
    print("=" * 70)

    # --- Setup ---
    print("\n[1] Creating environment and grid...")
    env = create_room_environment()
    grid = OccupancyGrid(
        width=20.0,
        height=20.0,
        resolution=0.1,  # 10cm cells → 200x200 grid = 40,000 cells
        origin=(0.0, 0.0),
        l_occ=0.85,    # Strong occupied update
        l_free=-0.4,   # Weaker free update (asymmetry is intentional)
        l_min=-5.0,    # Clamp bounds
        l_max=5.0,
    )
    print(f"  Grid size: {grid.grid_w} x {grid.grid_h} = {grid.grid_w * grid.grid_h:,} cells")
    print(f"  Resolution: {grid.resolution}m per cell")
    print(f"  Log-odds params: l_occ={grid.l_occ}, l_free={grid.l_free}")

    # --- Generate trajectory ---
    print("\n[2] Generating exploration trajectory...")
    path = generate_exploration_path()
    print(f"  {len(path)} poses planned")

    # --- Run mapping ---
    print("\n[3] Running occupancy grid mapping...")
    lidar_params = {
        "num_beams": 180,       # 2° angular resolution
        "max_range": 8.0,       # 8m max range
        "noise_std": 0.03,      # 3cm noise (typical for a good lidar)
        "fov": 2 * math.pi,     # 360° field of view
    }

    for i, (rx, ry, rtheta) in enumerate(path):
        ranges, angles = env.simulate_lidar(rx, ry, rtheta, **lidar_params)
        grid.update_scan(rx, ry, rtheta, ranges, angles, lidar_params["max_range"])

        # Print progress every 10 poses
        if (i + 1) % 10 == 0 or i == 0:
            stats = grid.get_map_stats()
            print(
                f"  Pose {i+1:3d}/{len(path)}: "
                f"free={stats['free_pct']:.1f}% "
                f"occupied={stats['occupied_pct']:.1f}% "
                f"unknown={stats['unknown_pct']:.1f}%"
            )

    # --- Final statistics ---
    print("\n[4] Final map statistics:")
    stats = grid.get_map_stats()
    for key, val in stats.items():
        if key.endswith("_pct"):
            print(f"  {key}: {val:.1f}%")
        else:
            print(f"  {key}: {val:,}")

    # --- Log-odds analysis ---
    print("\n[5] Log-odds distribution analysis:")
    lo = grid.log_odds
    non_zero = lo[lo != 0]
    if len(non_zero) > 0:
        print(f"  Cells updated (non-zero log-odds): {len(non_zero):,}")
        print(f"  Min log-odds: {np.min(non_zero):.2f} (P={1 - 1/(1 + math.exp(np.min(non_zero))):.4f})")
        print(f"  Max log-odds: {np.max(non_zero):.2f} (P={1 - 1/(1 + math.exp(np.max(non_zero))):.4f})")
        print(f"  Mean log-odds (non-zero): {np.mean(non_zero):.2f}")

    # --- Verify specific cells ---
    print("\n[6] Spot-checking known locations:")
    # Check a cell that should be free (middle of the room, left side)
    test_free = grid.world_to_grid(4.0, 5.0)
    lo_free = grid.log_odds[test_free[1], test_free[0]]
    p_free = 1 - 1 / (1 + math.exp(lo_free))
    print(f"  Center-left (4.0, 5.0) → grid {test_free}: log-odds={lo_free:.2f}, P(occ)={p_free:.3f} {'[FREE ✓]' if p_free < 0.3 else '[?]'}")

    # Check a cell on a wall
    test_wall = grid.world_to_grid(7.0, 5.0)
    lo_wall = grid.log_odds[test_wall[1], test_wall[0]]
    p_wall = 1 - 1 / (1 + math.exp(lo_wall))
    print(f"  L-wall (7.0, 5.0) → grid {test_wall}: log-odds={lo_wall:.2f}, P(occ)={p_wall:.3f} {'[OCCUPIED ✓]' if p_wall > 0.7 else '[?]'}")

    # Check a cell that was never observed
    test_unk = grid.world_to_grid(0.5, 0.5)
    lo_unk = grid.log_odds[test_unk[1], test_unk[0]]
    p_unk = 1 - 1 / (1 + math.exp(lo_unk))
    print(f"  Outside room (0.5, 0.5) → grid {test_unk}: log-odds={lo_unk:.2f}, P(occ)={p_unk:.3f} {'[UNKNOWN ✓]' if 0.3 <= p_unk <= 0.7 else '[?]'}")

    # --- ASCII map ---
    print("\n[7] ASCII map (## = occupied, .. = free, spaces = unknown):")
    print("-" * 70)
    ascii_map = render_map_ascii(grid, downsample=5)
    print(ascii_map)
    print("-" * 70)

    print("\n[8] Demonstrating log-odds update arithmetic:")
    print("  Starting from P=0.5 (log-odds=0.0):")
    l = 0.0
    for step in range(1, 6):
        l += grid.l_occ
        p = 1 - 1 / (1 + math.exp(l))
        print(f"    After {step} occupied observation(s): log-odds={l:.2f}, P(occ)={p:.4f}")
    print("  Now adding free observations to 'undo' it:")
    for step in range(1, 8):
        l += grid.l_free
        p = 1 - 1 / (1 + math.exp(l))
        print(f"    After {step} free observation(s): log-odds={l:.2f}, P(occ)={p:.4f}")

    print("\nDone! The occupancy grid correctly maps the simulated environment.")
