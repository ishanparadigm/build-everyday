"""
Day 41: Maze Solver with BFS/DFS — Your Implementation

Implement BFS and DFS to solve grid-based mazes.

Hint: The ONLY difference between BFS and DFS is the data structure:
- BFS uses a FIFO queue (collections.deque with popleft)
- DFS uses a LIFO stack (list with pop)
Everything else — visited set, parent tracking, path reconstruction — is identical.

Key gotcha: Mark cells as visited when you ADD them to the queue/stack,
NOT when you remove them. This prevents duplicate entries.

Run tests: python3 -m pytest tests.py -v
"""

from collections import deque
from typing import Optional


Cell = tuple[int, int]
Path = list[Cell]
Maze = list[list[int]]


def parse_maze(maze_str: str) -> tuple[Maze, Cell, Cell]:
    """Parse a string maze representation into a grid with start and goal positions.

    The maze string uses:
        0 = open cell
        1 = wall
        S = start (treated as open, value 0 in grid)
        G = goal (treated as open, value 0 in grid)

    Args:
        maze_str: Multi-line string with space-separated values

    Returns:
        (grid, start, goal) — grid is list of lists of 0/1 ints

    Raises:
        ValueError: If maze has no S or G
    """
    raise NotImplementedError("TODO: implement this")


def get_neighbors(grid: Maze, cell: Cell) -> list[Cell]:
    """Return valid neighboring cells (4-connected: up, right, down, left).

    A neighbor is valid if it's within grid bounds and not a wall (value 0).

    Args:
        grid: The maze grid (0 = open, 1 = wall)
        cell: Current position as (row, col)

    Returns:
        List of valid neighbor positions

    Hint: Check bounds FIRST (short-circuit), then check wall status.
    Use directions [(-1,0), (0,1), (1,0), (0,-1)] for up/right/down/left.
    """
    raise NotImplementedError("TODO: implement this")


def bfs(grid: Maze, start: Cell, goal: Cell) -> tuple[Optional[Path], set[Cell], int]:
    """Breadth-First Search — finds the SHORTEST path.

    Uses a FIFO queue (deque). Explores all cells at distance d before
    any cell at distance d+1.

    Args:
        grid: The maze grid
        start: Starting cell (row, col)
        goal: Goal cell (row, col)

    Returns:
        (path, visited_cells, nodes_explored_count)
        path is None if no path exists

    Hint: Use collections.deque for O(1) popleft.
    Track a parent dict to reconstruct the path.
    Mark cells visited when ENQUEUING, not when dequeuing.
    """
    raise NotImplementedError("TODO: implement this")


def dfs(grid: Maze, start: Cell, goal: Cell) -> tuple[Optional[Path], set[Cell], int]:
    """Depth-First Search — finds A path (not necessarily shortest).

    Uses a LIFO stack. Explores as deep as possible before backtracking.

    Args:
        grid: The maze grid
        start: Starting cell (row, col)
        goal: Goal cell (row, col)

    Returns:
        (path, visited_cells, nodes_explored_count)
        path is None if no path exists

    Hint: Almost identical to BFS — just change the data structure
    from deque (popleft) to list (pop). That's literally it.
    """
    raise NotImplementedError("TODO: implement this")


if __name__ == '__main__':
    # Test your implementation with a simple maze
    test_maze = """\
S 0 0 1 0
0 1 0 1 0
0 0 0 0 0
1 1 0 1 1
0 0 0 0 G"""

    print("Parsing maze...")
    grid, start, goal = parse_maze(test_maze)
    print(f"Grid size: {len(grid)}x{len(grid[0])}")
    print(f"Start: {start}, Goal: {goal}")

    print("\nRunning BFS...")
    bfs_path, bfs_visited, bfs_explored = bfs(grid, start, goal)
    if bfs_path:
        print(f"BFS path ({len(bfs_path)} steps): {bfs_path}")
        print(f"Nodes explored: {bfs_explored}")
    else:
        print("BFS: No path found")

    print("\nRunning DFS...")
    dfs_path, dfs_visited, dfs_explored = dfs(grid, start, goal)
    if dfs_path:
        print(f"DFS path ({len(dfs_path)} steps): {dfs_path}")
        print(f"Nodes explored: {dfs_explored}")
    else:
        print("DFS: No path found")

    if bfs_path and dfs_path:
        print(f"\nBFS path length: {len(bfs_path)} (guaranteed shortest)")
        print(f"DFS path length: {len(dfs_path)} (may be longer)")
