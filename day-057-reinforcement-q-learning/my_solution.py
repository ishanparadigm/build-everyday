"""
Day 057: Reinforcement Learning — Q-Learning from Scratch

YOUR TASK: Implement a Q-learning agent that learns to navigate a grid world.

Key concepts to implement:
  - GridWorld environment with step rewards and terminal states
  - Q-table for storing state-action values
  - ε-greedy exploration strategy
  - Bellman update rule for learning Q-values
  - Policy extraction from learned Q-values

Start with the GridWorld, then the agent, then connect them in train().
Run this file to test as you go — the main block exercises everything.
"""

import random
import numpy as np
from typing import Tuple, List, Optional, Dict

# Cell types
EMPTY = 0
WALL = 1
TRAP = 2
GOAL = 3
START = 4

# Actions: index → (row_delta, col_delta)
ACTIONS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
NUM_ACTIONS = len(ACTIONS)


class GridWorld:
    """
    A grid-based MDP environment.

    The grid is a 2D array where each cell is EMPTY, WALL, TRAP, GOAL, or START.

    Rewards:
      - Step on EMPTY/START: -1 (encourages short paths)
      - Hit WALL or go out of bounds: -1 (agent stays in place)
      - Step on TRAP: -10 (episode ends)
      - Reach GOAL: +100 (episode ends)
    """

    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        # Hint: store the grid as a numpy array, save start/goal, set initial state
        raise NotImplementedError("TODO: implement __init__")

    @property
    def num_states(self) -> int:
        """Total number of grid cells (rows * cols)."""
        raise NotImplementedError("TODO: implement num_states")

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to a flat index. Hint: row * cols + col."""
        raise NotImplementedError("TODO: implement state_to_index")

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat index back to (row, col)."""
        raise NotImplementedError("TODO: implement index_to_state")

    def reset(self) -> Tuple[int, int]:
        """Reset agent to start position. Returns start state."""
        raise NotImplementedError("TODO: implement reset")

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute an action.

        Returns: (next_state, reward, done)

        Hint: Compute the proposed new position. Check bounds and walls
        (stay in place if invalid). Then check cell type for reward/done.
        """
        raise NotImplementedError("TODO: implement step")


class QLearningAgent:
    """
    Tabular Q-learning agent.

    Maintains a Q-table: Q[state_index, action_index] → expected future reward.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        # Hint: initialize Q-table as zeros with shape (num_states, num_actions)
        # Store all hyperparameters and create lists for tracking metrics
        raise NotImplementedError("TODO: implement __init__")

    def choose_action(self, state_index: int) -> int:
        """
        ε-greedy action selection.

        With probability ε → random action.
        Otherwise → action with highest Q-value (break ties randomly!).

        Hint: Use random.random() < self.epsilon for the exploration check.
        For tie-breaking, find ALL actions with the max Q-value and pick randomly.
        """
        raise NotImplementedError("TODO: implement choose_action")

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> float:
        """
        Q-learning (Bellman) update.

        Q(s,a) ← Q(s,a) + α · [target - Q(s,a)]

        Where target = r                           if done (terminal state)
              target = r + γ · max_a'(Q(s',a'))    otherwise

        Returns the TD error (target - current Q).

        Hint: The key insight is using MAX over next-state Q-values,
        which makes this OFF-policy (learning optimal Q regardless of
        what action was actually taken next).
        """
        raise NotImplementedError("TODO: implement update")

    def decay_epsilon(self):
        """
        Reduce ε after each episode: ε = max(ε_min, ε * decay_rate).

        Hint: One line. Exponential decay with a floor.
        """
        raise NotImplementedError("TODO: implement decay_epsilon")

    def get_policy(self, env: GridWorld) -> Dict[Tuple[int, int], int]:
        """
        Extract greedy policy: policy[state] = argmax_a Q(state, a).

        Iterate over all non-wall cells and pick the best action.
        """
        raise NotImplementedError("TODO: implement get_policy")


def train(
    env: GridWorld,
    agent: QLearningAgent,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 200,
    verbose_every: int = 100,
) -> QLearningAgent:
    """
    Train the agent through repeated episodes.

    Each episode:
    1. Reset environment
    2. Loop: choose action → step → update Q → transition
    3. End on done or max_steps
    4. Decay ε

    Hint: track total reward and steps per episode for monitoring.
    """
    raise NotImplementedError("TODO: implement train")


def print_grid(env: GridWorld, policy: Optional[Dict] = None):
    """Print the grid with optional policy arrows."""
    symbols = {EMPTY: ".", WALL: "#", TRAP: "X", GOAL: "G", START: "S"}
    print("\n" + "=" * (env.cols * 4 + 1))
    for r in range(env.rows):
        row_str = "| "
        for c in range(env.cols):
            cell = env.grid[r, c]
            if cell in (WALL, GOAL, TRAP, START):
                row_str += symbols[cell] + " | "
            elif policy and (r, c) in policy:
                row_str += ACTION_ARROWS[policy[(r, c)]] + " | "
            else:
                row_str += symbols[cell] + " | "
        print(row_str)
    print("=" * (env.cols * 4 + 1))


# ---------------------------------------------------------------------------
# Test your implementation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    grid = [
        [START, EMPTY, EMPTY, WALL,  EMPTY, EMPTY],
        [EMPTY, WALL,  EMPTY, EMPTY, TRAP,  EMPTY],
        [EMPTY, WALL,  EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, WALL,  TRAP,  EMPTY],
        [TRAP,  EMPTY, EMPTY, WALL,  EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, GOAL],
    ]

    env = GridWorld(grid, start=(0, 0), goal=(5, 5))
    print(f"Grid: {env.rows}x{env.cols}, States: {env.num_states}")

    print("\n--- Initial Grid ---")
    print_grid(env)

    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=NUM_ACTIONS,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    print("\n--- Training ---\n")
    agent = train(env, agent, num_episodes=1000, verbose_every=200)

    policy = agent.get_policy(env)
    print("\n--- Learned Policy ---")
    print_grid(env, policy)

    # Test a greedy episode
    print("\n--- Greedy Episode ---\n")
    state = env.reset()
    for step in range(50):
        state_idx = env.state_to_index(state)
        action = int(np.argmax(agent.q_table[state_idx]))
        next_state, reward, done = env.step(action)
        print(f"  {state} --{ACTION_NAMES[action]}--> {next_state} (r={reward:+.0f})")
        state = next_state
        if done:
            break

    if env.grid[state] == GOAL:
        print("\n  SUCCESS: Goal reached!")
    else:
        print("\n  FAILED: Did not reach goal.")
