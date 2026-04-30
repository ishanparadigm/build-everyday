"""
Day 028: A* Pathfinding — Your Implementation

Implement the A* search algorithm on a 2D grid with obstacles.

Key concepts to implement:
- Priority queue (open set) ordered by f = g + h
- Closed set to track expanded nodes
- Path reconstruction via parent pointers
- Multiple heuristic functions

Run this file to test your implementation:
    python3 my_solution.py
"""

import heapq
import math
from typing import Optional


class Grid:
    """A 2D grid world with obstacles.

    Stores obstacles in a set for O(1) lookup.
    Provides neighbor functions for 4-dir and 8-dir movement.
    """

    def __init__(self, width: int, height: int, obstacles: Optional[set[tuple[int, int]]] = None):
        self.width = width
        self.height = height
        self.obstacles: set[tuple[int, int]] = obstacles or set()

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        """Check if a position is within grid boundaries."""
        raise NotImplementedError("TODO: implement bounds checking")

    def is_passable(self, pos: tuple[int, int]) -> bool:
        """Check if a position is not an obstacle."""
        raise NotImplementedError("TODO: implement obstacle checking")

    def neighbors_4dir(self, pos: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        """Return valid 4-directional neighbors with movement cost.

        Cardinal moves cost 1.0.
        Returns list of (neighbor_pos, cost) tuples.

        Hint: Check all 4 cardinal directions (up, down, left, right).
        Filter out positions that are out of bounds or obstacles.
        """
        raise NotImplementedError("TODO: implement 4-directional neighbors")

    def neighbors_8dir(self, pos: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        """Return valid 8-directional neighbors with movement cost.

        Cardinal moves cost 1.0, diagonal moves cost sqrt(2).
        Block diagonal movement through corners (both adjacent
        cardinal cells are obstacles).

        Hint: Handle cardinal and diagonal separately.
        For diagonals, check that the two adjacent cardinal cells
        aren't both blocked (corner-cutting prevention).
        """
        raise NotImplementedError("TODO: implement 8-directional neighbors")


# ---------------------------------------------------------------------------
# Heuristic functions
# ---------------------------------------------------------------------------
# Hint: Each heuristic estimates the remaining distance to the goal.
# Admissibility means never overestimating the true cost.

def manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Manhattan distance: sum of absolute differences.

    Admissible for 4-directional movement.
    """
    raise NotImplementedError("TODO: implement Manhattan distance")


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean (straight-line) distance.

    Always admissible — straight line is always <= actual path.
    """
    raise NotImplementedError("TODO: implement Euclidean distance")


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Chebyshev distance: max of absolute differences.

    Admissible for 8-directional with uniform cost.
    """
    raise NotImplementedError("TODO: implement Chebyshev distance")


def octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Octile distance: optimal for 8-dir grids with diagonal cost sqrt(2).

    Hint: Make min(|dr|,|dc|) diagonal moves, then the remainder as straight moves.
    """
    raise NotImplementedError("TODO: implement Octile distance")


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
}


# ---------------------------------------------------------------------------
# A* algorithm
# ---------------------------------------------------------------------------

def astar(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic_name: str = "manhattan",
    eight_directional: bool = False,
) -> Optional[tuple[list[tuple[int, int]], float, set[tuple[int, int]]]]:
    """A* pathfinding algorithm.

    Args:
        grid: The Grid world to search.
        start: Starting position (row, col).
        goal: Goal position (row, col).
        heuristic_name: Which heuristic to use.
        eight_directional: If True, allow diagonal movement.

    Returns:
        Tuple of (path, cost, explored_set) if path found, None otherwise.

    Hints:
    - Use heapq for the open set: entries are (f_score, counter, node)
    - Keep a dict for g_scores (default infinity for unknown nodes)
    - Keep a dict for parent pointers (came_from)
    - Keep a set for closed (expanded) nodes
    - Check goal when you POP from the queue, not when you discover
    - For path reconstruction, follow came_from pointers back to start
    """
    raise NotImplementedError("TODO: implement A* algorithm")


# ---------------------------------------------------------------------------
# Visualization (provided — no need to implement)
# ---------------------------------------------------------------------------

def visualize_grid(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: Optional[list[tuple[int, int]]] = None,
    explored: Optional[set[tuple[int, int]]] = None,
) -> str:
    """Render the grid as a string. S=start, G=goal, #=obstacle, *=path, .=explored."""
    path_set = set(path) if path else set()
    explored = explored or set()
    lines = []
    header = "    " + "".join(f"{c % 10}" for c in range(grid.width))
    lines.append(header)
    lines.append("    " + "-" * grid.width)
    for r in range(grid.height):
        row_str = f"{r:2d} |"
        for c in range(grid.width):
            pos = (r, c)
            if pos == start:
                row_str += "S"
            elif pos == goal:
                row_str += "G"
            elif pos in grid.obstacles:
                row_str += "#"
            elif pos in path_set:
                row_str += "*"
            elif pos in explored:
                row_str += "."
            else:
                row_str += " "
        row_str += "|"
        lines.append(row_str)
    lines.append("    " + "-" * grid.width)
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with a simple grid first
    print("=== Simple Grid Test ===")
    grid = Grid(5, 5)
    result = astar(grid, (0, 0), (4, 4), heuristic_name="manhattan")
    if result:
        path, cost, explored = result
        print(f"Path: {path}")
        print(f"Cost: {cost}, Explored: {len(explored)}")
        print(visualize_grid(grid, (0, 0), (4, 4), path, explored))

    # Test with obstacles
    print("\n=== Grid with Obstacles ===")
    obstacles = {(r, 3) for r in range(0, 8)}  # Vertical wall
    grid = Grid(10, 10, obstacles)
    result = astar(grid, (5, 1), (5, 8), heuristic_name="manhattan")
    if result:
        path, cost, explored = result
        print(f"Path found! Cost: {cost}, Steps: {len(path)}")
        print(visualize_grid(grid, (5, 1), (5, 8), path, explored))
    else:
        print("No path found!")

    # Test no-path scenario
    print("\n=== No Path Test ===")
    wall = {(r, 5) for r in range(10)}  # Complete wall
    grid = Grid(10, 10, wall)
    result = astar(grid, (5, 2), (5, 8))
    print(f"No path result: {result}")  # Should be None
