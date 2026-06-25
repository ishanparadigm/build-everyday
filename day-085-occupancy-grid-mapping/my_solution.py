"""
Day 85: Occupancy Grid Mapping — Your Implementation

Build a probabilistic 2D map from noisy range sensor data using log-odds
representation and Bresenham ray casting.

Key concepts to implement:
1. Bresenham's line algorithm for ray tracing through grid cells
2. Log-odds representation for numerically stable probability updates
3. Inverse sensor model: which cells are free vs occupied for a given reading
4. Full scan processing to fuse multiple beams into the map

Run tests: python3 -m pytest tests.py -v
"""

import math
import numpy as np
from typing import List, Tuple


def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Return all grid cells along the line from (x0,y0) to (x1,y1).

    Uses Bresenham's line algorithm — integer arithmetic only.
    The returned list includes both endpoints and is ordered from start to end.

    Hint: Track an error term that accumulates as you step along the major axis.
    When the error exceeds a threshold, step in the minor axis too.
    """
    raise NotImplementedError("TODO: implement Bresenham's line algorithm")


class OccupancyGrid:
    """
    A 2D occupancy grid using log-odds representation.

    Hint: Log-odds l maps to probability P via: P = 1 - 1/(1 + exp(l))
    The inverse: l = log(P / (1 - P))
    Unknown state: l = 0 → P = 0.5
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

        self.grid_w = int(math.ceil(width / resolution))
        self.grid_h = int(math.ceil(height / resolution))

        # Hint: Initialize with zeros — in log-odds, 0 means P=0.5 (unknown)
        self.log_odds = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid cell indices.

        Hint: Subtract the origin, divide by resolution, and truncate to int.
        """
        raise NotImplementedError("TODO: implement world-to-grid transform")

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """
        Convert grid cell indices to world coordinates (cell center).

        Hint: Multiply by resolution, add origin, and offset by half a cell
        to get the center rather than the corner.
        """
        raise NotImplementedError("TODO: implement grid-to-world transform")

    def in_bounds(self, gx: int, gy: int) -> bool:
        """Check if grid indices are within the grid."""
        raise NotImplementedError("TODO: implement bounds checking")

    def update_cell(self, gx: int, gy: int, log_odds_update: float) -> None:
        """
        Update a single cell's log-odds and clamp to [l_min, l_max].

        Hint: Just add the update value and clamp. Note the array indexing
        is [gy, gx] because rows = y, columns = x.
        """
        raise NotImplementedError("TODO: implement cell update with clamping")

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

        Steps:
        1. Compute beam endpoint in world coordinates
        2. Convert robot pos and endpoint to grid coordinates
        3. Use Bresenham to get cells along the ray
        4. Mark traversed cells (all but last) as free
        5. Mark endpoint cell as occupied (only if range < max_range)

        Hint: The absolute beam angle is robot_theta + beam_angle.
        Endpoint: (robot_x + range*cos(angle), robot_y + range*sin(angle))
        """
        raise NotImplementedError("TODO: implement the inverse sensor model")

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

        Hint: Just loop over all (range, angle) pairs and call
        inverse_sensor_model for each.
        """
        raise NotImplementedError("TODO: implement full scan processing")

    def get_probability_map(self) -> np.ndarray:
        """
        Convert log-odds grid to probability grid.

        Hint: P = 1 - 1/(1 + exp(l))
        Use numpy operations for the whole array at once.
        """
        raise NotImplementedError("TODO: implement log-odds to probability conversion")

    def get_map_stats(self) -> dict:
        """Return statistics about the current map state."""
        prob = self.get_probability_map()
        total_cells = self.grid_w * self.grid_h
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


class Environment:
    """
    A 2D environment defined by line-segment walls.
    Provides simulated lidar readings via ray casting.
    """

    def __init__(self, walls: List[Tuple[float, float, float, float]]):
        self.walls = walls

    def cast_ray(
        self, ox: float, oy: float, angle: float, max_range: float
    ) -> float:
        """
        Cast a single ray from (ox, oy) at the given angle.
        Returns distance to nearest wall hit, or max_range if no hit.

        Hint: For each wall segment, compute ray-segment intersection using
        parametric form. The ray is R(t) = O + t*D, the segment is
        S(u) = A + u*(B-A). Solve for t and u.
        Valid hit: t > 0 and 0 <= u <= 1.
        """
        raise NotImplementedError("TODO: implement ray casting")

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

        Returns (ranges, angles).

        Hint: Generate evenly-spaced angles across the FOV, cast a ray for
        each, then add Gaussian noise to the measured ranges.
        """
        raise NotImplementedError("TODO: implement lidar simulation")


if __name__ == "__main__":
    # Test your implementation
    np.random.seed(42)

    # Test 1: Bresenham
    print("Testing Bresenham...")
    cells = bresenham(0, 0, 5, 3)
    print(f"  Line (0,0)→(5,3): {cells}")

    # Test 2: Grid basics
    print("\nTesting OccupancyGrid...")
    grid = OccupancyGrid(width=10.0, height=10.0, resolution=0.5)
    print(f"  Grid size: {grid.grid_w}x{grid.grid_h}")
    gx, gy = grid.world_to_grid(5.0, 5.0)
    print(f"  World (5,5) → grid ({gx},{gy})")
    wx, wy = grid.grid_to_world(gx, gy)
    print(f"  Grid ({gx},{gy}) → world ({wx},{wy})")

    # Test 3: Environment and mapping
    print("\nTesting Environment + Mapping...")
    walls = [
        (0, 0, 10, 0), (10, 0, 10, 10),
        (10, 10, 0, 10), (0, 10, 0, 0),
    ]
    env = Environment(walls)
    grid = OccupancyGrid(width=10.0, height=10.0, resolution=0.2)

    ranges, angles = env.simulate_lidar(5.0, 5.0, 0.0, num_beams=36, max_range=8.0)
    grid.update_scan(5.0, 5.0, 0.0, ranges, angles, 8.0)

    stats = grid.get_map_stats()
    print(f"  After 1 scan: free={stats['free_pct']:.1f}%, occupied={stats['occupied_pct']:.1f}%")

    print("\nAll basic tests passed!")
