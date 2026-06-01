"""
Day 057: Reinforcement Learning — Q-Learning from Scratch

A complete Q-learning implementation with a grid world environment.
The agent learns to navigate from start to goal while avoiding traps,
using only reward signals — no labeled data, no supervision.

Key ideas implemented:
  - Markov Decision Process (MDP) environment
  - Tabular Q-learning with Bellman updates
  - ε-greedy exploration with decay
  - Policy extraction and visualization
"""

import random
import numpy as np
from typing import Tuple, List, Optional, Dict

# ---------------------------------------------------------------------------
# Grid World Environment
# ---------------------------------------------------------------------------

# Cell types
EMPTY = 0
WALL = 1
TRAP = 2
GOAL = 3
START = 4

# Actions: indices into this list give (row_delta, col_delta)
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

    The grid is a 2D array where each cell is one of:
      EMPTY (0) — passable, small step penalty
      WALL  (1) — impassable, agent stays in place
      TRAP  (2) — passable but delivers a large negative reward and ends episode
      GOAL  (3) — the target; large positive reward, ends episode
      START (4) — the agent's starting position (behaves like EMPTY)

    Rewards encode what we want the agent to learn:
      - Step penalty (-1): encourages finding SHORT paths, not just any path
      - Wall bump penalty (-1): same as a wasted step
      - Trap penalty (-10): teaches avoidance of dangerous states
      - Goal reward (+100): the objective

    The transition model is deterministic: action → predictable next state.
    (Stochastic transitions are a natural extension but add complexity we
    don't need to understand the core algorithm.)
    """

    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.goal = goal
        self.state = start  # Current agent position

        # Validate the grid
        assert self.grid[start] == START, f"Start cell must be START type, got {self.grid[start]}"
        assert self.grid[goal] == GOAL, f"Goal cell must be GOAL type, got {self.grid[goal]}"

    @property
    def num_states(self) -> int:
        """Total number of grid cells (including walls — simplifies indexing)."""
        return self.rows * self.cols

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to a flat index for Q-table lookup."""
        return state[0] * self.cols + state[1]

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat index back to (row, col)."""
        return (index // self.cols, index % self.cols)

    def reset(self) -> Tuple[int, int]:
        """Reset agent to start. Returns initial state."""
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute an action in the environment.

        Returns:
            next_state: (row, col) after the action
            reward: immediate reward signal
            done: whether the episode has ended (goal reached or trap hit)

        The agent proposes a move. If the move would go out of bounds or into
        a wall, the agent stays in place (but still pays the step cost — you
        don't get free "thinking time" in the real world).
        """
        row, col = self.state
        dr, dc = ACTIONS[action]
        new_row, new_col = row + dr, col + dc

        # Boundary check — stay in place if out of bounds
        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            return self.state, -1.0, False

        # Wall check — stay in place if hitting a wall
        if self.grid[new_row, new_col] == WALL:
            return self.state, -1.0, False

        # Valid move — update state
        self.state = (new_row, new_col)
        cell = self.grid[new_row, new_col]

        if cell == GOAL:
            return self.state, 100.0, True  # Big reward, episode over
        elif cell == TRAP:
            return self.state, -10.0, True  # Penalty, episode over
        else:
            return self.state, -1.0, False  # Small step cost, keep going


# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-learning agent.

    Maintains a Q-table mapping (state, action) → expected cumulative reward.
    Learns through the Bellman update rule applied to each experience tuple.

    The Q-table is a 2D numpy array: Q[state_index, action_index].
    We use flat state indices for efficient array access.
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
        # Q-table initialized to zeros.
        # Alternative: optimistic initialization (e.g., all 10s) encourages
        # exploration because every real experience will be lower than the
        # initial estimate, making the agent try everything at least once.
        # We use zero init for clarity — ε-greedy handles exploration.
        self.q_table = np.zeros((num_states, num_actions))

        self.lr = learning_rate       # α — step size for updates
        self.gamma = discount_factor  # γ — how much to value future rewards
        self.epsilon = epsilon        # ε — exploration probability
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions

        # Track training metrics for analysis
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def choose_action(self, state_index: int) -> int:
        """
        ε-greedy action selection.

        With probability ε: pick a random action (explore).
        With probability 1-ε: pick the action with highest Q-value (exploit).

        When multiple actions tie for the best Q-value, we break ties randomly.
        Without random tie-breaking, the agent would always prefer action 0
        (the first max), creating a systematic bias toward UP moves early
        in training when all Q-values are still 0.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Exploit: find all actions tied for the maximum Q-value
        q_values = self.q_table[state_index]
        max_q = np.max(q_values)
        # np.where returns indices where condition is True
        best_actions = np.where(q_values == max_q)[0]
        return int(random.choice(best_actions))

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> float:
        """
        Apply the Q-learning (Bellman) update rule.

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a'(Q(s',a')) - Q(s,a)]

        For terminal states (done=True), there is no future value,
        so the target is just the immediate reward r.

        Returns the TD error for monitoring convergence.

        WHY off-policy matters: We use max_a'(Q(s',a')) regardless of what
        action the agent actually takes next. This means we're always updating
        toward the OPTIMAL policy, even while following an exploratory ε-greedy
        behavior policy. SARSA would instead use Q(s', a_next) where a_next is
        the action actually chosen — making it on-policy and more conservative.
        """
        current_q = self.q_table[state_idx, action]

        if done:
            # Terminal state: no future rewards to consider
            target = reward
        else:
            # Non-terminal: immediate reward + discounted best future value
            best_next_q = np.max(self.q_table[next_state_idx])
            target = reward + self.gamma * best_next_q

        # TD error: how wrong was our current estimate?
        td_error = target - current_q

        # Update Q-value: nudge it toward the target by α * td_error
        self.q_table[state_idx, action] = current_q + self.lr * td_error

        return td_error

    def decay_epsilon(self):
        """
        Reduce exploration rate after each episode.

        Exponential decay: ε *= decay_rate, with a floor at ε_min.
        Early episodes are mostly random (high ε → exploration).
        Later episodes are mostly greedy (low ε → exploitation of learned values).

        The decay rate controls how quickly the agent shifts from exploring to
        exploiting. Too fast → premature convergence to a suboptimal policy.
        Too slow → wastes time on random actions after Q-values have stabilized.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self, env: GridWorld) -> Dict[Tuple[int, int], int]:
        """
        Extract the learned policy: for each state, which action is best?

        policy[state] = argmax_a Q(state, a)

        This is the "greedy" policy — no exploration, pure exploitation.
        After good training, this should show a clear path from start to goal.
        """
        policy = {}
        for r in range(env.rows):
            for c in range(env.cols):
                if env.grid[r, c] != WALL:
                    state_idx = env.state_to_index((r, c))
                    best_action = int(np.argmax(self.q_table[state_idx]))
                    policy[(r, c)] = best_action
        return policy


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    env: GridWorld,
    agent: QLearningAgent,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 200,
    verbose_every: int = 100,
) -> QLearningAgent:
    """
    Train the Q-learning agent through repeated episodes.

    Each episode:
    1. Reset environment to start state
    2. Loop: choose action → take step → update Q-table → repeat
    3. Episode ends when goal/trap reached or max steps exceeded
    4. Decay ε to gradually shift from exploration to exploitation

    max_steps_per_episode prevents infinite loops during early training
    when the agent hasn't learned anything useful and wanders randomly.
    Without this cap, a single episode could run forever in a grid with
    no traps.
    """
    for episode in range(num_episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        total_reward = 0.0
        steps = 0

        for step in range(max_steps_per_episode):
            # 1. Choose action using ε-greedy
            action = agent.choose_action(state_idx)

            # 2. Take the action, observe result
            next_state, reward, done = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            # 3. Update Q-table with this experience
            agent.update(state_idx, action, reward, next_state_idx, done)

            # 4. Transition to next state
            state = next_state
            state_idx = next_state_idx
            total_reward += reward
            steps += 1

            if done:
                break

        # Track metrics
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)

        # Decay exploration rate
        agent.decay_epsilon()

        if verbose_every and (episode + 1) % verbose_every == 0:
            avg_reward = np.mean(agent.episode_rewards[-verbose_every:])
            avg_length = np.mean(agent.episode_lengths[-verbose_every:])
            print(
                f"Episode {episode + 1:>5d} | "
                f"Avg Reward: {avg_reward:>8.1f} | "
                f"Avg Length: {avg_length:>5.1f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    return agent


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def print_grid(env: GridWorld, policy: Optional[Dict] = None):
    """
    Print the grid world with optional policy arrows.

    Legend: S=start, G=goal, X=trap, #=wall, .=empty
    If a policy is provided, empty cells show the best action as an arrow.
    """
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


def print_q_values(env: GridWorld, agent: QLearningAgent):
    """Print Q-values for each non-wall cell to show what the agent learned."""
    print("\nQ-Values (UP / DOWN / LEFT / RIGHT):")
    print("-" * 60)
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r, c] != WALL:
                idx = env.state_to_index((r, c))
                q = agent.q_table[idx]
                label = f"({r},{c})"
                print(f"  {label:>6s}: [{q[0]:>7.1f} {q[1]:>7.1f} {q[2]:>7.1f} {q[3]:>7.1f}]")
    print()


def evaluate_policy(env: GridWorld, agent: QLearningAgent, num_eval: int = 100) -> Tuple[float, float, float]:
    """
    Evaluate the learned policy (greedy, no exploration) over multiple episodes.

    Returns:
        success_rate: fraction of episodes that reached the goal
        avg_reward: average cumulative reward
        avg_steps: average steps per episode (only for successful episodes)
    """
    successes = 0
    total_reward = 0.0
    success_steps = []

    for _ in range(num_eval):
        state = env.reset()
        ep_reward = 0.0
        steps = 0

        for step in range(200):
            state_idx = env.state_to_index(state)
            # Greedy action — no exploration during evaluation
            action = int(np.argmax(agent.q_table[state_idx]))
            state, reward, done = env.step(action)
            ep_reward += reward
            steps += 1
            if done:
                break

        total_reward += ep_reward
        if env.grid[state] == GOAL:
            successes += 1
            success_steps.append(steps)

    success_rate = successes / num_eval
    avg_reward = total_reward / num_eval
    avg_steps = np.mean(success_steps) if success_steps else float('inf')
    return success_rate, avg_reward, avg_steps


# ---------------------------------------------------------------------------
# Main: demonstrate Q-learning on a grid world
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # --- Define the grid world ---
    # 6x6 grid with walls and traps creating an interesting navigation problem.
    #
    #   S . . # . .       S = Start (0,0)
    #   . # . . X .       G = Goal  (5,5)
    #   . # . . . .       X = Trap
    #   . . . # X .       # = Wall
    #   X . . # . .
    #   . . . . . G
    #
    grid = [
        [START, EMPTY, EMPTY, WALL,  EMPTY, EMPTY],
        [EMPTY, WALL,  EMPTY, EMPTY, TRAP,  EMPTY],
        [EMPTY, WALL,  EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, WALL,  TRAP,  EMPTY],
        [TRAP,  EMPTY, EMPTY, WALL,  EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, GOAL],
    ]

    env = GridWorld(grid, start=(0, 0), goal=(5, 5))

    print("=" * 60)
    print("REINFORCEMENT LEARNING: Q-LEARNING FROM SCRATCH")
    print("=" * 60)
    print(f"\nGrid size: {env.rows}x{env.cols}")
    print(f"States: {env.num_states} | Actions: {NUM_ACTIONS}")
    print(f"Start: {env.start} | Goal: {env.goal}")

    print("\n--- Initial Grid ---")
    print_grid(env)

    # --- Create and train the agent ---
    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=NUM_ACTIONS,
        learning_rate=0.1,       # α: moderate step size
        discount_factor=0.95,    # γ: value future rewards but prefer sooner ones
        epsilon=1.0,             # Start fully exploratory
        epsilon_min=0.01,        # Always keep 1% exploration
        epsilon_decay=0.995,     # Gradual shift to exploitation
    )

    print("\n--- Training (1000 episodes) ---\n")
    agent = train(env, agent, num_episodes=1000, verbose_every=200)

    # --- Show results ---
    policy = agent.get_policy(env)

    print("\n--- Learned Policy ---")
    print("(Arrows show the best action in each cell)")
    print_grid(env, policy)

    # Show Q-values for key cells so the reader can verify the agent's reasoning
    print_q_values(env, agent)

    # --- Evaluate the learned policy ---
    success_rate, avg_reward, avg_steps = evaluate_policy(env, agent)
    print("--- Policy Evaluation (100 greedy episodes) ---")
    print(f"  Success rate: {success_rate * 100:.0f}%")
    print(f"  Avg reward:   {avg_reward:.1f}")
    print(f"  Avg steps:    {avg_steps:.1f}")

    # --- Show a single greedy episode step-by-step ---
    print("\n--- Example Greedy Episode ---\n")
    state = env.reset()
    path = [state]
    total_r = 0.0

    for step in range(50):
        state_idx = env.state_to_index(state)
        action = int(np.argmax(agent.q_table[state_idx]))
        next_state, reward, done = env.step(action)
        total_r += reward
        print(f"  Step {step + 1}: {state} --{ACTION_NAMES[action]}--> {next_state}  (reward: {reward:+.0f})")
        state = next_state
        path.append(state)
        if done:
            break

    result = "GOAL REACHED!" if env.grid[state] == GOAL else "FAILED"
    print(f"\n  Result: {result}")
    print(f"  Total reward: {total_r:.0f}")
    print(f"  Path length: {len(path) - 1} steps")

    # --- Hyperparameter sensitivity demonstration ---
    print("\n--- Hyperparameter Comparison ---\n")
    configs = [
        ("High α=0.5, γ=0.95", 0.5, 0.95),
        ("Low α=0.01, γ=0.95", 0.01, 0.95),
        ("α=0.1, Low γ=0.5 (myopic)", 0.1, 0.5),
        ("α=0.1, High γ=0.99 (far-sighted)", 0.1, 0.99),
    ]

    for name, lr, gamma in configs:
        random.seed(42)
        np.random.seed(42)
        test_env = GridWorld(grid, start=(0, 0), goal=(5, 5))
        test_agent = QLearningAgent(
            num_states=test_env.num_states,
            num_actions=NUM_ACTIONS,
            learning_rate=lr,
            discount_factor=gamma,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
        )
        train(test_env, test_agent, num_episodes=1000, verbose_every=0)
        sr, ar, ast = evaluate_policy(test_env, test_agent)
        print(f"  {name:>40s} | Success: {sr * 100:>5.0f}% | Avg reward: {ar:>7.1f} | Steps: {ast:>5.1f}")

    print("\nDone! The Q-learning agent has learned to navigate the grid world.")
