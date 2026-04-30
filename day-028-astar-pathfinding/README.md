# Day 028: A* Pathfinding

## Overview

Build the A* (A-star) search algorithm from scratch — the gold standard for finding shortest paths in weighted graphs and grids. A* powers navigation in robotics, video games, logistics, and autonomous vehicles. Unlike Dijkstra's algorithm (which explores in all directions equally), A* uses a **heuristic** to intelligently focus its search toward the goal, dramatically reducing the number of nodes explored.

You'll implement A* on a 2D grid with obstacles, visualize the search process, and understand exactly why it finds optimal paths — and when it doesn't.

## Core Concepts

### The Shortest Path Problem

Given a graph (or grid) with weighted edges, find the path from start to goal with the minimum total cost. On a grid, each cell is a node and movement to adjacent cells has a cost (typically 1 for cardinal, ~1.414 for diagonal).

### What Makes A* Special: The f = g + h Formula

A* maintains a priority queue of nodes to explore, ordered by:

```
f(n) = g(n) + h(n)
```

- **g(n)**: The exact cost from the start node to node n. This is known — we've been tracking it as we explore.
- **h(n)**: The **heuristic estimate** of the cost from n to the goal. This is a guess — but a carefully chosen one.
- **f(n)**: The estimated total cost of the cheapest path through n.

The key insight: by combining actual cost (g) with estimated remaining cost (h), A* focuses exploration toward the goal without sacrificing optimality.

### Heuristics: The Heart of A*

The heuristic function h(n) estimates distance from n to the goal. Its properties determine A*'s behavior:

**Admissibility**: A heuristic is *admissible* if it never overestimates the true cost: h(n) <= actual cost from n to goal. An admissible heuristic guarantees A* finds the optimal path.

**Consistency** (monotonicity): A heuristic is *consistent* if for every node n and neighbor n': h(n) <= cost(n, n') + h(n'). This is the triangle inequality. Consistency implies admissibility and ensures each node is expanded at most once.

Common heuristics for grids:

| Heuristic | Formula | Best For |
|-----------|---------|----------|
| Manhattan | \|dx\| + \|dy\| | 4-directional movement |
| Euclidean | sqrt(dx^2 + dy^2) | Any-angle movement |
| Chebyshev | max(\|dx\|, \|dy\|) | 8-directional, uniform cost |
| Octile | max(\|dx\|,\|dy\|) + (sqrt(2)-1) * min(\|dx\|,\|dy\|) | 8-directional, diagonal costs sqrt(2) |

**Why admissibility matters**: If h overestimates, A* might skip the optimal path because it looks expensive. Manhattan distance is admissible for 4-directional grids because you can't possibly get there in fewer moves. Euclidean is admissible for any movement because straight-line distance is always <= actual path distance.

### The Algorithm Step-by-Step

1. **Initialize**: Add start node to the open set (priority queue) with f = h(start).
2. **Loop**: While open set is not empty:
   a. Pop the node with lowest f value — call it `current`.
   b. If `current` is the goal, reconstruct the path and return.
   c. Add `current` to the closed set (already explored).
   d. For each neighbor of `current`:
      - If neighbor is in the closed set or is an obstacle, skip it.
      - Calculate tentative g = g(current) + cost(current, neighbor).
      - If neighbor is not in the open set, add it.
      - If tentative g >= neighbor's known g, skip (we already found a better path).
      - Otherwise, update neighbor's parent, g, and f values.
3. **If open set empties**: No path exists.

### Why A* is Optimal (Proof Sketch)

Assume h is admissible and A* returns path P with cost C. Suppose optimal path P* has cost C* < C. When A* terminated, some node n on P* must have been in the open set. Its f(n) = g(n) + h(n) <= g(n) + actual_remaining = C*. But A* chose to expand the goal instead, meaning f(goal) = C <= f(n) <= C*. Contradiction: C <= C* but we assumed C* < C.

### Comparison with Other Algorithms

| Algorithm | Explores | Optimal? | Complete? | Time |
|-----------|----------|----------|-----------|------|
| BFS | All directions equally | Yes (unweighted) | Yes | O(V+E) |
| Dijkstra | All directions, by cost | Yes | Yes | O((V+E) log V) |
| Greedy Best-First | Toward goal only | No | No | O((V+E) log V) |
| A* | Toward goal, with guarantees | Yes (admissible h) | Yes | O((V+E) log V) |

A* with h=0 degenerates to Dijkstra. A* with g=0 degenerates to Greedy Best-First Search.

## Step-by-Step Breakdown

### Step 1: Grid Representation
Represent the world as a 2D grid where each cell is either passable (cost 1) or an obstacle. Store obstacles in a set for O(1) lookup.

### Step 2: Heuristic Functions
Implement multiple heuristics. For 4-directional movement, Manhattan distance is the natural choice. For 8-directional, use Octile distance to properly account for diagonal costs.

### Step 3: Priority Queue (Open Set)
Use Python's `heapq` module. Each entry is (f_score, counter, node) — the counter breaks ties consistently and avoids comparing nodes directly.

### Step 4: The Main Loop
Track g_scores in a dict (default infinity), parent pointers in a dict for path reconstruction, and a closed set for already-expanded nodes.

### Step 5: Path Reconstruction
Once the goal is reached, follow parent pointers back from goal to start, then reverse.

### Step 6: Visualization
Print the grid showing the path, explored nodes, and obstacles to understand how A* focuses its search.

## Learning Objectives

- Implement A* search with correct open/closed set management
- Understand how heuristic choice affects search behavior and optimality
- Compare A* exploration patterns against BFS and Dijkstra
- Handle 4-directional and 8-directional movement with appropriate heuristics
- Reconstruct optimal paths using parent pointers
- Analyze time and space complexity of informed search

## Going Deeper

- **Weighted A***: Use f = g + w*h where w > 1. Finds suboptimal paths faster — bounded suboptimality guarantee of w.
- **IDA***: Iterative deepening A* uses linear memory instead of exponential. Critical for large state spaces.
- **Jump Point Search**: Exploits grid symmetry to skip redundant nodes — 10-100x faster on uniform grids.
- **Theta***: Any-angle pathfinding that produces smoother paths by allowing line-of-sight shortcuts.
- **D* Lite**: Incremental replanning for dynamic environments where obstacles appear/disappear — used in Mars rovers.
- **Bidirectional A***: Search from both start and goal simultaneously; meet in the middle.
- **In production**: Navigation systems (Google Maps uses A* variants), game AI (every RTS and RPG), warehouse robots (Amazon Kiva), surgical robot path planning.
