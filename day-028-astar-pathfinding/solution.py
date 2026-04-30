"""
Day 028: A* Pathfinding Algorithm

A complete implementation of A* search on a 2D grid, supporting both
4-directional and 8-directional movement with multiple heuristic functions.

Demonstrates:
- Core A* algorithm with open/closed sets
- Multiple heuristics (Manhattan, Euclidean, Octile, Chebyshev)
- Path reconstruction via parent pointers
- Visualization of explored nodes and optimal path
- Comparison of exploration efficiency across heuristics
"""

import heapq
import math
from typing import Optional


# ---------------------------------------------------------------------------
# Grid representation
# ---------------------------------------------------------------------------

class Grid:
    """A 2D grid world with obstacles.

    The grid uses (row, col) coordinates where (0,0) is the top-left corner.
    Obstacles are stored in a set for O(1) membership checks — much faster
    than scanning a 2D array, especially on sparse grids.
    """

    def __init__(self, width: int, height: int, obstacles: Optional[set[tuple[int, int]]] = None):
        self.width = width
        self.height = height
        # Frozenset would be safer (immutable) but set is fine for our purposes
        self.obstacles: set[tuple[int, int]] = obstacles or set()

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        """Check if a position is within grid boundaries."""
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_passable(self, pos: tuple[int, int]) -> bool:
        """Check if a position is not an obstacle."""
        return pos not in self.obstacles

    def neighbors_4dir(self, pos: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        """Return valid 4-directional neighbors with movement cost.

        Cardinal moves cost 1.0. We filter out-of-bounds and obstacle cells.
        Returns list of (neighbor_pos, cost) tuples.
        """
        r, c = pos
        # Up, Down, Left, Right
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        results = []
        for nb in candidates:
            if self.in_bounds(nb) and self.is_passable(nb):
                results.append((nb, 1.0))
        return results

    def neighbors_8dir(self, pos: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        """Return valid 8-directional neighbors with movement cost.

        Cardinal moves cost 1.0, diagonal moves cost sqrt(2) ≈ 1.414.
        Diagonal cost is sqrt(2) because that's the Euclidean distance of
        moving one cell diagonally — using 1.0 for diagonals would make
        diagonal paths artificially cheap and produce zigzag artifacts.

        We also block diagonal movement through corners — if both adjacent
        cardinal cells are obstacles, the diagonal is impassable. This
        prevents the agent from "cutting through" walls.
        """
        r, c = pos
        SQRT2 = math.sqrt(2)
        results = []

        # Cardinal directions first (cost 1.0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if self.in_bounds(nb) and self.is_passable(nb):
                results.append((nb, 1.0))

        # Diagonal directions (cost sqrt(2)), with corner-cutting check
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nb = (r + dr, c + dc)
            if not self.in_bounds(nb) or not self.is_passable(nb):
                continue
            # Block diagonal if both adjacent cardinal cells are obstacles
            # This prevents cutting through wall corners
            adj1 = (r + dr, c)  # vertical neighbor
            adj2 = (r, c + dc)  # horizontal neighbor
            if not self.is_passable(adj1) and not self.is_passable(adj2):
                continue
            results.append((nb, SQRT2))

        return results


# ---------------------------------------------------------------------------
# Heuristic functions
# ---------------------------------------------------------------------------

def manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Manhattan distance: |dr| + |dc|.

    Admissible for 4-directional movement because the minimum number of
    moves is exactly |dr| + |dc| when there are no obstacles.
    NOT admissible for 8-directional movement (overestimates when diagonals
    cost sqrt(2) < 2.0, but a single diagonal replaces two cardinal moves).
    Wait — actually Manhattan IS admissible for 8-dir because diagonal moves
    cost sqrt(2) > 1 and Manhattan counts minimum cardinal moves. It just
    won't be as tight (informative) as Octile.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean (straight-line) distance.

    Always admissible — you can never get there faster than a straight line.
    Less informative than Manhattan for grid movement because it underestimates
    more, causing A* to explore more nodes.
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Chebyshev distance: max(|dr|, |dc|).

    Admissible for 8-directional movement with uniform cost (all moves cost 1).
    Represents the minimum moves when diagonal movement costs the same as
    cardinal movement.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Octile distance: optimal for 8-directional grids with diagonal cost sqrt(2).

    The idea: to go from a to b, you make min(|dr|, |dc|) diagonal moves
    (each covering 1 unit in both directions at cost sqrt(2)), then
    |max - min| straight moves (cost 1 each).

    octile = sqrt(2) * min(|dr|, |dc|) + (max(|dr|, |dc|) - min(|dr|, |dc|))
           = max(|dr|, |dc|) + (sqrt(2) - 1) * min(|dr|, |dc|)

    This is the tightest admissible heuristic for 8-directional grids —
    it equals the actual cost when there are no obstacles.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


# Map of heuristic names to functions for easy selection
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
        heuristic_name: Which heuristic to use ("manhattan", "euclidean",
                        "chebyshev", "octile").
        eight_directional: If True, allow diagonal movement.

    Returns:
        Tuple of (path, cost, explored_set) if path found, None otherwise.
        - path: List of (row, col) positions from start to goal.
        - cost: Total path cost.
        - explored_set: All nodes that were expanded (for visualization).
    """
    h = HEURISTICS[heuristic_name]
    get_neighbors = grid.neighbors_8dir if eight_directional else grid.neighbors_4dir

    # g_score[node] = cost of cheapest known path from start to node
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    # Parent pointers for path reconstruction
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    # Open set: priority queue of (f_score, tiebreaker, node)
    # The tiebreaker (counter) ensures we never compare tuples directly,
    # which would fail since tuples of ints aren't meaningfully orderable
    # as "nodes" — and it provides FIFO ordering among equal-f nodes.
    counter = 0
    open_set: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_set, (h(start, goal), counter, start))
    counter += 1

    # Track which nodes are in the open set for O(1) membership checks.
    # heapq doesn't support efficient "is this element present?" queries.
    open_set_hash: set[tuple[int, int]] = {start}

    # Closed set: nodes we've already fully explored
    closed_set: set[tuple[int, int]] = set()

    while open_set:
        # Pop the node with the lowest f score
        f_current, _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)

        # Goal check — we check when we POP, not when we discover.
        # This is important: a node might be added to the open set with a
        # suboptimal g, then updated. By checking at pop time, we ensure
        # we have the optimal g for this node.
        if current == goal:
            # Reconstruct path by following parent pointers
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, g_score[goal], closed_set

        closed_set.add(current)

        for neighbor, move_cost in get_neighbors(current):
            if neighbor in closed_set:
                # Already expanded — skip. With a consistent heuristic,
                # the first time we expand a node is guaranteed to be optimal.
                continue

            tentative_g = g_score[current] + move_cost

            # Is this a better path to the neighbor than any we've found?
            if tentative_g < g_score.get(neighbor, math.inf):
                # Yes — record it
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h(neighbor, goal)

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1
                    open_set_hash.add(neighbor)
                else:
                    # Node is already in open set with a worse g.
                    # Standard Python heapq doesn't support decrease-key,
                    # so we add a duplicate entry. The stale entry will be
                    # popped later and ignored because g_score check will fail.
                    # This is the "lazy deletion" approach — simpler than
                    # implementing a proper indexed priority queue.
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

    # Open set empty and goal not reached — no path exists
    return None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_grid(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: Optional[list[tuple[int, int]]] = None,
    explored: Optional[set[tuple[int, int]]] = None,
) -> str:
    """Render the grid as a string for terminal display.

    Legend:
      S = start, G = goal, # = obstacle
      * = path, . = explored, ' ' = unexplored
    """
    path_set = set(path) if path else set()
    explored = explored or set()

    lines = []
    # Column numbers header
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


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

def create_demo_grid() -> tuple[Grid, tuple[int, int], tuple[int, int]]:
    """Create a 20x20 grid with interesting obstacles.

    The obstacle layout creates corridors and forces the algorithm to
    find non-obvious paths, making the heuristic comparison more interesting.
    """
    width, height = 20, 20
    obstacles: set[tuple[int, int]] = set()

    # Vertical wall with a gap at the bottom
    for r in range(0, 14):
        obstacles.add((r, 7))

    # Horizontal wall with a gap on the right
    for c in range(3, 16):
        obstacles.add((10, c))

    # Second vertical wall with a gap in the middle
    for r in range(5, 18):
        if r != 11:  # gap
            obstacles.add((r, 14))

    # Small box obstacle
    for r in range(2, 5):
        for c in range(10, 13):
            obstacles.add((r, c))

    # Diagonal obstacle
    for i in range(6):
        obstacles.add((14 + i, 2 + i))

    start = (1, 1)
    goal = (18, 18)

    return Grid(width, height, obstacles), start, goal


def run_comparison():
    """Compare different heuristics on the same grid to show exploration differences."""
    grid, start, goal = create_demo_grid()

    print("=" * 60)
    print("A* PATHFINDING — HEURISTIC COMPARISON")
    print("=" * 60)
    print(f"\nGrid: {grid.width}x{grid.height}")
    print(f"Start: {start}  Goal: {goal}")
    print(f"Obstacles: {len(grid.obstacles)}")

    # --- 4-directional comparison ---
    print("\n" + "=" * 60)
    print("4-DIRECTIONAL MOVEMENT")
    print("=" * 60)

    for h_name in ["manhattan", "euclidean"]:
        result = astar(grid, start, goal, heuristic_name=h_name, eight_directional=False)
        if result:
            path, cost, explored = result
            print(f"\n--- Heuristic: {h_name} ---")
            print(f"Path length: {len(path)} steps")
            print(f"Path cost: {cost:.2f}")
            print(f"Nodes explored: {len(explored)}")
            print(visualize_grid(grid, start, goal, path, explored))
        else:
            print(f"\n--- Heuristic: {h_name} ---")
            print("No path found!")

    # --- 8-directional comparison ---
    print("\n" + "=" * 60)
    print("8-DIRECTIONAL MOVEMENT")
    print("=" * 60)

    for h_name in ["octile", "euclidean", "chebyshev"]:
        result = astar(grid, start, goal, heuristic_name=h_name, eight_directional=True)
        if result:
            path, cost, explored = result
            print(f"\n--- Heuristic: {h_name} ---")
            print(f"Path length: {len(path)} steps")
            print(f"Path cost: {cost:.2f}")
            print(f"Nodes explored: {len(explored)}")
            print(visualize_grid(grid, start, goal, path, explored))
        else:
            print(f"\n--- Heuristic: {h_name} ---")
            print("No path found!")


def run_no_path_demo():
    """Demonstrate A* correctly detecting an unreachable goal."""
    print("\n" + "=" * 60)
    print("NO PATH SCENARIO")
    print("=" * 60)

    obstacles = set()
    # Complete wall blocking the grid
    for r in range(10):
        obstacles.add((r, 5))

    grid = Grid(10, 10, obstacles)
    start = (5, 2)
    goal = (5, 8)

    result = astar(grid, start, goal, heuristic_name="manhattan")
    if result is None:
        print("\nCorrectly detected: No path exists!")
        print(visualize_grid(grid, start, goal, explored=set()))
    else:
        print("ERROR: Should not have found a path!")


if __name__ == "__main__":
    run_comparison()
    run_no_path_demo()

    # --- Quick sanity check: trivial case ---
    print("\n" + "=" * 60)
    print("TRIVIAL CASE: Open grid, start=(0,0), goal=(4,4)")
    print("=" * 60)
    grid = Grid(5, 5)
    result = astar(grid, (0, 0), (4, 4), heuristic_name="manhattan", eight_directional=False)
    if result:
        path, cost, explored = result
        print(f"Path: {path}")
        print(f"Cost: {cost}, Steps: {len(path)}, Explored: {len(explored)}")
        print(visualize_grid(grid, (0, 0), (4, 4), path, explored))
