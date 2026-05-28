# Day 41: Maze Solver with BFS/DFS

## Overview

Build a maze solver that implements both Breadth-First Search (BFS) and Depth-First Search (DFS) to find paths through a grid-based maze. This is a foundational robotics skill — every mobile robot that navigates indoor environments (warehouses, hospitals, homes) needs to solve some variant of this problem. The difference between BFS and DFS isn't academic trivia; it determines whether your robot finds the shortest path or wanders down dead ends.

This challenge builds directly on Day 28 (A* pathfinding) and Day 34 (RRT path planning). Where A* uses a heuristic to guide search and RRT samples randomly, BFS and DFS are the *uninformed* baselines. Understanding them deeply is essential because: (1) they're the building blocks A* is built on, (2) in many real scenarios a simple BFS outperforms fancier algorithms, and (3) knowing when each fails tells you when you actually need something more sophisticated.

## Core Concepts

### Graph Search as Maze Navigation

A grid maze is just a graph where each cell is a node and each passable neighbor is an edge. The maze-solving problem reduces to: **find a path from node S to node G in an unweighted graph**.

The key insight: in an unweighted graph (all edges cost 1), BFS is *optimal* — it always finds the shortest path. DFS is *not* optimal but uses less memory. This tradeoff is fundamental to all of robotics path planning.

### Breadth-First Search (BFS)

BFS explores nodes in order of their distance from the start. It uses a **FIFO queue**:

```
Algorithm BFS(start, goal):
    queue = [start]
    visited = {start}
    parent = {start: None}
    
    while queue is not empty:
        current = queue.popleft()      # FIFO — oldest node first
        if current == goal:
            return reconstruct_path(parent, goal)
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    return None  # no path exists
```

**Why BFS finds the shortest path:** It processes all nodes at distance d before any node at distance d+1. So the first time it reaches the goal, that path has the minimum number of steps. This is a *proof by induction* on the distance layers.

**Complexity:**
- Time: O(V + E) where V = cells, E = passable edges. For an N×M grid: O(N·M)
- Space: O(V) for the queue and visited set. Worst case the queue holds an entire "frontier ring"

**When BFS fails in practice:** Memory. In a 1000×1000 grid, BFS might hold 4000 nodes in its queue simultaneously. For a 3D voxel grid (common in drone navigation), this explodes to millions.

### Depth-First Search (DFS)

DFS explores as deep as possible before backtracking. It uses a **LIFO stack** (or recursion):

```
Algorithm DFS(start, goal):
    stack = [start]
    visited = {start}
    parent = {start: None}
    
    while stack is not empty:
        current = stack.pop()          # LIFO — newest node first
        if current == goal:
            return reconstruct_path(parent, goal)
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)
    return None
```

**Why DFS does NOT find the shortest path:** It dives deep into one branch. If the goal is 3 steps away but DFS went left first, it might explore a 100-step detour before finding the goal.

**Complexity:**
- Time: O(V + E) — same as BFS
- Space: O(V) worst case, but *typical* case is O(longest_path_length), which for a maze with narrow corridors is much less than BFS

**The real DFS advantage:** In recursive form, DFS naturally backtracks. This makes it the basis for algorithms like maze *generation* (randomized DFS creates perfect mazes) and topological sort.

### BFS vs DFS — The Core Tradeoff

| Property | BFS | DFS |
|----------|-----|-----|
| Data structure | Queue (FIFO) | Stack (LIFO) |
| Optimal path? | Yes (unweighted) | No |
| Memory usage | O(branching_factor^d) | O(d) typical |
| Complete? | Yes | Yes (with visited set) |
| Best for | Shortest path, small search space | Existence check, large search space |

In robotics, BFS is used when you need the optimal path (warehouse robots). DFS is used when you just need *any* path quickly (exploration), or as a subroutine in more complex algorithms.

### Maze Representation

We represent the maze as a 2D grid:
- `0` = open cell (passable)
- `1` = wall (blocked)
- `S` = start
- `G` = goal

Neighbors are the 4-connected cells (up, down, left, right). We don't use diagonals because in grid-based robotics, diagonal movement requires checking two cells for collision (the robot's body sweeps through both adjacent cells).

## Step-by-Step Breakdown

### Step 1: Maze Representation and Validation
Parse the maze grid, locate start and goal positions, validate that both exist and are not walled off. Without validation, you'll get silent failures — the algorithm returns "no path" when really the input was malformed.

### Step 2: Neighbor Generation
For a given cell (row, col), generate valid neighbors: within bounds, not a wall, respecting 4-connectivity. This is the "graph edge" function. Getting this wrong (e.g., off-by-one on bounds) causes the most bugs in grid search.

### Step 3: BFS Implementation
Implement BFS with a deque for O(1) popleft. Track visited nodes *when enqueuing* (not when dequeuing) — this is a subtle but critical optimization. If you mark visited on dequeue, you'll add duplicate nodes to the queue, wasting memory and time.

### Step 4: DFS Implementation
Implement DFS with an explicit stack (not recursion, to avoid stack overflow on large mazes). Same visited-on-push optimization as BFS.

### Step 5: Path Reconstruction
Use the parent dictionary to trace back from goal to start. Reverse the result to get start-to-goal order. This is the same technique used in A* and Dijkstra's.

### Step 6: Visualization
Render the maze with the found path, visited cells, and statistics. Seeing which cells each algorithm explores makes the BFS/DFS difference viscerally clear.

### Step 7: Comparison Analysis
Run both algorithms on the same maze. Compare: path length, cells explored, execution time. This quantitative comparison is what transforms textbook knowledge into engineering intuition.

## Learning Objectives

- Implement BFS and DFS from scratch with correct visited-set management
- Understand why BFS guarantees shortest paths in unweighted graphs (and when that guarantee breaks)
- Understand the memory/optimality tradeoff between BFS and DFS
- Build grid-based maze representations used in real robotics navigation
- Reconstruct paths from parent pointers — a pattern used across all graph search algorithms
- Develop intuition for when to use which algorithm through empirical comparison

## Going Deeper

- **Bidirectional BFS**: Start BFS from both start and goal simultaneously. Meets in the middle, reducing search space from O(b^d) to O(b^(d/2)). Used in social network shortest-path queries.
- **Iterative Deepening DFS (IDDFS)**: Combines BFS optimality with DFS memory efficiency. Runs DFS with depth limit 1, then 2, then 3... Sounds wasteful but the overhead is only O(b) factor.
- **Connection to A***: BFS is A* with h(n) = 0. A* is BFS with a priority queue sorted by f(n) = g(n) + h(n). Understanding BFS deeply makes A* intuitive.
- **Maze generation**: Randomized DFS (recursive backtracker) generates perfect mazes. Randomized Kruskal's generates mazes with a different character. Try generating mazes and solving them.
- **Real robot constraints**: Physical robots can't teleport between cells. The path must be *smooth* — BFS gives waypoints, but you need trajectory planning (Day 34's RRT) to connect them.
- **Multi-agent pathfinding**: When multiple robots share a maze, BFS per-agent isn't enough — you need conflict-based search (CBS) or prioritized planning.
