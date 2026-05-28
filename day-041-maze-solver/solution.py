"""
Day 41: Maze Solver with BFS/DFS

Complete implementation of BFS and DFS maze solvers with visualization
and comparative analysis.

Key insight: BFS and DFS differ by exactly ONE data structure choice
(queue vs stack), yet this produces fundamentally different behavior —
optimal shortest paths vs deep exploration. Understanding this deeply
is the foundation for all graph-based path planning in robotics.
"""

from collections import deque
from typing import Optional
import time


# Type aliases for clarity
Cell = tuple[int, int]
Path = list[Cell]
Maze = list[list[int]]


def parse_maze(maze_str: str) -> tuple[Maze, Cell, Cell]:
    """Parse a string maze representation into a grid with start and goal positions.

    The maze string uses:
        0 = open cell
        1 = wall
        S = start (treated as open)
        G = goal (treated as open)

    Returns:
        (grid, start, goal) where grid uses 0/1 integers

    Why parse from string? In robotics, maze/map data comes from various sources
    (occupancy grids from SLAM, floor plans, sensor data). Having a clean parser
    that validates input prevents silent failures downstream.
    """
    grid: Maze = []
    start: Optional[Cell] = None
    goal: Optional[Cell] = None

    for row_idx, line in enumerate(maze_str.strip().split('\n')):
        row = []
        for col_idx, char in enumerate(line.strip().split()):
            if char == 'S':
                start = (row_idx, col_idx)
                row.append(0)  # Start is passable
            elif char == 'G':
                goal = (row_idx, col_idx)
                row.append(0)  # Goal is passable
            else:
                row.append(int(char))
        grid.append(row)

    if start is None or goal is None:
        raise ValueError("Maze must contain both 'S' (start) and 'G' (goal)")

    return grid, start, goal


def get_neighbors(grid: Maze, cell: Cell) -> list[Cell]:
    """Return valid neighboring cells (4-connected, within bounds, not walls).

    We use 4-connectivity (up, down, left, right) rather than 8-connectivity
    because in grid-based robotics, diagonal movement through a gap between
    two walls would require the robot to fit through a diagonal — which is
    sqrt(2) * cell_size wide but the robot's body sweeps through both
    adjacent cells. Most occupancy grid planners use 4-connected for safety.

    The order of directions affects DFS behavior (which branch it explores first)
    but not BFS correctness. We use URDL order consistently.
    """
    rows, cols = len(grid), len(grid[0])
    row, col = cell
    neighbors = []

    # Up, Right, Down, Left — consistent ordering matters for reproducibility
    for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        nr, nc = row + dr, col + dc
        # Bounds check first (short-circuit), then wall check
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbors.append((nr, nc))

    return neighbors


def bfs(grid: Maze, start: Cell, goal: Cell) -> tuple[Optional[Path], set[Cell], int]:
    """Breadth-First Search — finds the SHORTEST path in an unweighted graph.

    BFS explores nodes layer by layer: all nodes at distance d from start
    are processed before any node at distance d+1. This guarantees the first
    time we reach the goal, we've found the minimum-step path.

    Critical implementation detail: we mark cells as visited when ENQUEUING,
    not when DEQUEUING. If we waited until dequeue, the same cell could be
    added to the queue multiple times by different neighbors, wasting O(V)
    extra memory and time. This is the #1 BFS implementation bug.

    Returns:
        (path, visited_cells, nodes_explored_count)
        path is None if no path exists
    """
    # deque gives O(1) popleft — using a list would make popleft O(n),
    # turning our O(V+E) algorithm into O(V^2)
    queue: deque[Cell] = deque([start])
    visited: set[Cell] = {start}
    parent: dict[Cell, Optional[Cell]] = {start: None}
    nodes_explored = 0

    while queue:
        current = queue.popleft()  # FIFO: oldest node first = breadth-first
        nodes_explored += 1

        if current == goal:
            # Reconstruct path by following parent pointers back to start
            path = _reconstruct_path(parent, goal)
            return path, visited, nodes_explored

        for neighbor in get_neighbors(grid, current):
            if neighbor not in visited:
                visited.add(neighbor)        # Mark visited NOW, not on dequeue
                parent[neighbor] = current
                queue.append(neighbor)

    return None, visited, nodes_explored  # No path exists


def dfs(grid: Maze, start: Cell, goal: Cell) -> tuple[Optional[Path], set[Cell], int]:
    """Depth-First Search — finds A path (not necessarily shortest).

    DFS explores as deep as possible before backtracking. It uses a stack
    (LIFO) instead of a queue (FIFO) — this single change produces
    fundamentally different exploration behavior.

    We use an explicit stack rather than recursion because:
    1. Python's default recursion limit is 1000 — a 50×50 maze can exceed this
    2. Explicit stack gives us control over the visited-on-push optimization
    3. It's easier to instrument (count nodes, track timing)

    Returns:
        (path, visited_cells, nodes_explored_count)
        path is None if no path exists
    """
    stack: list[Cell] = [start]
    visited: set[Cell] = {start}
    parent: dict[Cell, Optional[Cell]] = {start: None}
    nodes_explored = 0

    while stack:
        current = stack.pop()  # LIFO: newest node first = depth-first
        nodes_explored += 1

        if current == goal:
            path = _reconstruct_path(parent, goal)
            return path, visited, nodes_explored

        for neighbor in get_neighbors(grid, current):
            if neighbor not in visited:
                visited.add(neighbor)        # Same optimization as BFS
                parent[neighbor] = current
                stack.append(neighbor)

    return None, visited, nodes_explored


def _reconstruct_path(parent: dict[Cell, Optional[Cell]], goal: Cell) -> Path:
    """Trace parent pointers from goal back to start, then reverse.

    This is the universal path reconstruction pattern used in BFS, DFS,
    Dijkstra's, and A*. The parent dict forms a tree rooted at the start
    node — we just walk up the tree from the goal.

    Time: O(path_length). We reverse at the end rather than prepending
    (which would be O(n^2) with a list) or using a deque (unnecessary overhead).
    """
    path: Path = []
    current: Optional[Cell] = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def visualize_maze(grid: Maze, start: Cell, goal: Cell,
                   path: Optional[Path] = None,
                   visited: Optional[set[Cell]] = None,
                   title: str = "Maze") -> str:
    """Render the maze as a string with path and visited cells highlighted.

    Legend:
        S = start, G = goal
        # = wall
        * = path cell
        . = visited but not on path
        (space) = unvisited open cell

    This visualization is critical for building intuition about how BFS
    and DFS explore differently. BFS visited cells form concentric rings;
    DFS visited cells form long tendrils.
    """
    path_set = set(path) if path else set()
    visited_set = visited or set()

    lines = [f"\n  {title}"]
    lines.append("  " + "+" + "---" * len(grid[0]) + "+")

    for r, row in enumerate(grid):
        line = "  |"
        for c, cell in enumerate(row):
            pos = (r, c)
            if pos == start:
                line += " S "
            elif pos == goal:
                line += " G "
            elif cell == 1:
                line += " # "
            elif pos in path_set:
                line += " * "
            elif pos in visited_set:
                line += " . "
            else:
                line += "   "
        line += "|"
        lines.append(line)

    lines.append("  " + "+" + "---" * len(grid[0]) + "+")
    return '\n'.join(lines)


def compare_algorithms(maze_str: str) -> None:
    """Run BFS and DFS on the same maze and compare results.

    This is where theory meets practice. The numbers tell the story:
    - BFS always finds the shortest path but may explore more cells
    - DFS may find a longer path but can get lucky and explore fewer cells
    - In maze-like graphs (narrow corridors), the difference is dramatic
    """
    grid, start, goal = parse_maze(maze_str)

    print("=" * 60)
    print("MAZE SOLVER: BFS vs DFS Comparison")
    print("=" * 60)

    # Show the original maze
    print(visualize_maze(grid, start, goal, title="Original Maze"))
    print(f"\n  Start: {start}  |  Goal: {goal}")
    print(f"  Grid size: {len(grid)} x {len(grid[0])}")

    # --- BFS ---
    print("\n" + "-" * 60)
    print("BFS (Breadth-First Search)")
    print("-" * 60)

    t0 = time.perf_counter()
    bfs_path, bfs_visited, bfs_explored = bfs(grid, start, goal)
    bfs_time = time.perf_counter() - t0

    if bfs_path:
        print(f"  Path found! Length: {len(bfs_path)} steps")
        print(f"  Nodes explored: {bfs_explored}")
        print(f"  Cells visited: {len(bfs_visited)}")
        print(f"  Time: {bfs_time*1000:.3f} ms")
        print(f"  Path: {bfs_path}")
        print(visualize_maze(grid, start, goal, bfs_path, bfs_visited,
                            title="BFS Result (. = explored, * = path)"))
    else:
        print("  No path found!")
        print(f"  Nodes explored before giving up: {bfs_explored}")

    # --- DFS ---
    print("\n" + "-" * 60)
    print("DFS (Depth-First Search)")
    print("-" * 60)

    t0 = time.perf_counter()
    dfs_path, dfs_visited, dfs_explored = dfs(grid, start, goal)
    dfs_time = time.perf_counter() - t0

    if dfs_path:
        print(f"  Path found! Length: {len(dfs_path)} steps")
        print(f"  Nodes explored: {dfs_explored}")
        print(f"  Cells visited: {len(dfs_visited)}")
        print(f"  Time: {dfs_time*1000:.3f} ms")
        print(f"  Path: {dfs_path}")
        print(visualize_maze(grid, start, goal, dfs_path, dfs_visited,
                            title="DFS Result (. = explored, * = path)"))
    else:
        print("  No path found!")
        print(f"  Nodes explored before giving up: {dfs_explored}")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    if bfs_path and dfs_path:
        print(f"  {'Metric':<25} {'BFS':>10} {'DFS':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10}")
        print(f"  {'Path length':<25} {len(bfs_path):>10} {len(dfs_path):>10}")
        print(f"  {'Nodes explored':<25} {bfs_explored:>10} {dfs_explored:>10}")
        print(f"  {'Cells visited':<25} {len(bfs_visited):>10} {len(dfs_visited):>10}")
        print(f"  {'Time (ms)':<25} {bfs_time*1000:>10.3f} {dfs_time*1000:>10.3f}")

        if len(dfs_path) > len(bfs_path):
            overhead = len(dfs_path) - len(bfs_path)
            print(f"\n  DFS path is {overhead} steps longer than optimal (BFS).")
            print(f"  That's {overhead/len(bfs_path)*100:.1f}% overhead — in a warehouse,")
            print(f"  that's {overhead} extra seconds of robot travel time per trip.")
        elif len(dfs_path) == len(bfs_path):
            print(f"\n  Both found the same length path! DFS got lucky on this maze.")
    print()


def generate_maze(rows: int, cols: int, wall_density: float = 0.3,
                  seed: int = 42) -> str:
    """Generate a random maze string for testing.

    Uses random wall placement with guaranteed start/goal positions.
    wall_density controls how many cells are walls (0.0 = empty, 1.0 = all walls).

    Note: This doesn't guarantee a path exists — which is actually useful for
    testing that our algorithms correctly report "no path found."
    For guaranteed-solvable mazes, use randomized DFS generation instead.
    """
    import random
    rng = random.Random(seed)

    lines = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if (r, c) == (0, 0):
                row.append('S')
            elif (r, c) == (rows - 1, cols - 1):
                row.append('G')
            elif rng.random() < wall_density:
                row.append('1')
            else:
                row.append('0')
        lines.append(' '.join(row))

    return '\n'.join(lines)


if __name__ == '__main__':
    # =====================================================
    # Example 1: A simple, hand-crafted maze
    # =====================================================
    simple_maze = """\
S 0 0 1 0
0 1 0 1 0
0 0 0 0 0
1 1 0 1 1
0 0 0 0 G"""

    print("\n>>> EXAMPLE 1: Simple 5x5 maze")
    print("    This maze has a clear shortest path. Watch how BFS")
    print("    finds it directly while DFS may wander.\n")
    compare_algorithms(simple_maze)

    # =====================================================
    # Example 2: Maze with a long detour
    # =====================================================
    # This maze is designed so the shortest path goes right then down,
    # but DFS (exploring up first) will go the long way around
    detour_maze = """\
S 0 0 0 0 0 0
1 1 1 1 1 1 0
0 0 0 0 0 0 0
0 1 1 1 1 1 1
0 0 0 0 0 0 G"""

    print("\n>>> EXAMPLE 2: Serpentine maze (forces DFS detour)")
    print("    The walls create a serpentine path. BFS finds the")
    print("    shortest route; DFS follows the winding corridor.\n")
    compare_algorithms(detour_maze)

    # =====================================================
    # Example 3: No path exists
    # =====================================================
    blocked_maze = """\
S 0 1 0 0
0 0 1 0 0
1 1 1 0 0
0 0 0 1 0
0 0 0 1 G"""

    print("\n>>> EXAMPLE 3: Blocked maze (no path exists)")
    print("    The wall completely separates start from goal.")
    print("    Both algorithms should report no path found.\n")
    compare_algorithms(blocked_maze)

    # =====================================================
    # Example 4: Larger random maze
    # =====================================================
    print("\n>>> EXAMPLE 4: Random 10x10 maze")
    print("    Larger maze shows the exploration pattern differences")
    print("    more clearly. BFS expands in rings; DFS goes deep.\n")
    random_maze = generate_maze(10, 10, wall_density=0.25, seed=123)
    compare_algorithms(random_maze)

    # =====================================================
    # Key Takeaways
    # =====================================================
    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
  1. BFS guarantees the shortest path in unweighted graphs.
     Use it when path optimality matters (warehouse robots, game AI).

  2. DFS uses less memory on average but may find longer paths.
     Use it when you just need ANY path, or for maze generation.

  3. The ONLY code difference is queue (FIFO) vs stack (LIFO).
     This tiny change produces fundamentally different behavior.

  4. Both are O(V+E) time — neither is inherently "faster."
     The difference is in what they explore, not how fast.

  5. In robotics: BFS gives waypoints, but real robots need
     trajectory smoothing on top (see Day 28: A*, Day 34: RRT).
""")
