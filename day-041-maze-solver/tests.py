"""
Day 41: Maze Solver Tests

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import unittest
from my_solution import parse_maze, get_neighbors, bfs, dfs


class TestParseMaze(unittest.TestCase):
    """Tests for maze parsing."""

    def test_basic_parse(self):
        maze_str = "S 0 1\n0 0 G"
        grid, start, goal = parse_maze(maze_str)
        self.assertEqual(grid, [[0, 0, 1], [0, 0, 0]])
        self.assertEqual(start, (0, 0))
        self.assertEqual(goal, (1, 2))

    def test_missing_start_raises(self):
        with self.assertRaises(ValueError):
            parse_maze("0 0 G\n0 0 0")

    def test_missing_goal_raises(self):
        with self.assertRaises(ValueError):
            parse_maze("S 0 0\n0 0 0")


class TestGetNeighbors(unittest.TestCase):
    """Tests for neighbor generation."""

    def test_center_cell_open(self):
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        neighbors = get_neighbors(grid, (1, 1))
        self.assertEqual(len(neighbors), 4)
        self.assertIn((0, 1), neighbors)
        self.assertIn((2, 1), neighbors)
        self.assertIn((1, 0), neighbors)
        self.assertIn((1, 2), neighbors)

    def test_corner_cell(self):
        grid = [[0, 0], [0, 0]]
        neighbors = get_neighbors(grid, (0, 0))
        self.assertEqual(len(neighbors), 2)
        self.assertIn((0, 1), neighbors)
        self.assertIn((1, 0), neighbors)

    def test_walls_block_neighbors(self):
        grid = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        neighbors = get_neighbors(grid, (1, 1))
        self.assertEqual(neighbors, [])


class TestBFS(unittest.TestCase):
    """Tests for BFS solver."""

    def test_simple_path(self):
        maze_str = "S 0 G"
        grid, start, goal = parse_maze(maze_str)
        path, visited, explored = bfs(grid, start, goal)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertEqual(len(path), 3)  # S -> (0,1) -> G

    def test_shortest_path(self):
        """BFS must find the shortest path."""
        maze_str = "S 0 0 0 0\n0 1 1 1 0\n0 0 0 0 G"
        grid, start, goal = parse_maze(maze_str)
        path, _, _ = bfs(grid, start, goal)
        self.assertIsNotNone(path)
        # Shortest path goes right along top, then down
        self.assertEqual(len(path), 9)

    def test_no_path(self):
        maze_str = "S 1 G"
        grid, start, goal = parse_maze(maze_str)
        path, visited, explored = bfs(grid, start, goal)
        self.assertIsNone(path)

    def test_start_equals_goal(self):
        """Edge case: start and goal are the same cell."""
        grid = [[0]]
        path, visited, explored = bfs(grid, (0, 0), (0, 0))
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], (0, 0))

    def test_visited_tracking(self):
        """BFS should track all visited cells."""
        maze_str = "S 0\n0 G"
        grid, start, goal = parse_maze(maze_str)
        path, visited, explored = bfs(grid, start, goal)
        self.assertIn(start, visited)
        self.assertIn(goal, visited)


class TestDFS(unittest.TestCase):
    """Tests for DFS solver."""

    def test_finds_a_path(self):
        maze_str = "S 0 G"
        grid, start, goal = parse_maze(maze_str)
        path, visited, explored = dfs(grid, start, goal)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)

    def test_no_path(self):
        maze_str = "S 1 G"
        grid, start, goal = parse_maze(maze_str)
        path, visited, explored = dfs(grid, start, goal)
        self.assertIsNone(path)

    def test_path_is_valid(self):
        """Every consecutive pair in the path must be adjacent."""
        maze_str = "S 0 0 1 0\n0 1 0 1 0\n0 0 0 0 G"
        grid, start, goal = parse_maze(maze_str)
        path, _, _ = dfs(grid, start, goal)
        self.assertIsNotNone(path)
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            self.assertEqual(abs(r1 - r2) + abs(c1 - c2), 1,
                             f"Non-adjacent cells in path: {path[i]} -> {path[i+1]}")

    def test_large_open_grid(self):
        """DFS should handle a large open grid without issues."""
        size = 20
        rows = []
        for r in range(size):
            row = []
            for c in range(size):
                if (r, c) == (0, 0):
                    row.append('S')
                elif (r, c) == (size - 1, size - 1):
                    row.append('G')
                else:
                    row.append('0')
            rows.append(' '.join(row))
        maze_str = '\n'.join(rows)
        grid, start, goal = parse_maze(maze_str)
        path, _, _ = dfs(grid, start, goal)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)


if __name__ == '__main__':
    unittest.main()
