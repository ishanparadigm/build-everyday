"""
Tests for Day 057: Q-Learning Implementation

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import random
import numpy as np
from my_solution import (
    GridWorld, QLearningAgent, train,
    EMPTY, WALL, TRAP, GOAL, START, NUM_ACTIONS,
)


class TestGridWorld(unittest.TestCase):
    """Tests for the GridWorld environment."""

    def setUp(self):
        """Create a simple 3x3 grid for testing."""
        self.grid = [
            [START, EMPTY, EMPTY],
            [EMPTY, WALL,  EMPTY],
            [EMPTY, EMPTY, GOAL],
        ]
        self.env = GridWorld(self.grid, start=(0, 0), goal=(2, 2))

    def test_grid_dimensions(self):
        """Environment should correctly report grid dimensions."""
        self.assertEqual(self.env.rows, 3)
        self.assertEqual(self.env.cols, 3)
        self.assertEqual(self.env.num_states, 9)

    def test_reset_returns_start(self):
        """Reset should return the start position."""
        state = self.env.reset()
        self.assertEqual(state, (0, 0))

    def test_state_index_roundtrip(self):
        """state_to_index and index_to_state should be inverses."""
        for r in range(3):
            for c in range(3):
                idx = self.env.state_to_index((r, c))
                self.assertEqual(self.env.index_to_state(idx), (r, c))

    def test_step_into_wall(self):
        """Stepping into a wall should keep agent in place with -1 reward."""
        self.env.reset()
        # Move down to (1,0), then try to move right into wall at (1,1)
        self.env.step(1)  # DOWN to (1,0)
        state, reward, done = self.env.step(3)  # RIGHT into wall
        self.assertEqual(state, (1, 0))  # Should stay at (1,0)
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)

    def test_step_out_of_bounds(self):
        """Stepping out of bounds should keep agent in place with -1 reward."""
        self.env.reset()  # At (0,0)
        state, reward, done = self.env.step(0)  # UP — out of bounds
        self.assertEqual(state, (0, 0))
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)

    def test_reach_goal(self):
        """Reaching the goal should give +100 reward and end the episode."""
        grid = [
            [START, GOAL],
        ]
        env = GridWorld(grid, start=(0, 0), goal=(0, 1))
        env.reset()
        state, reward, done = env.step(3)  # RIGHT to goal
        self.assertEqual(state, (0, 1))
        self.assertEqual(reward, 100.0)
        self.assertTrue(done)

    def test_trap_ends_episode(self):
        """Stepping on a trap should give -10 and end the episode."""
        grid = [
            [START, TRAP],
        ]
        env = GridWorld(grid, start=(0, 0), goal=(0, 0))  # goal doesn't matter here
        # Manually set grid so it's valid (goal cell must be GOAL type)
        # Instead, use a grid where we can test trap:
        grid2 = [
            [START, TRAP, GOAL],
        ]
        env2 = GridWorld(grid2, start=(0, 0), goal=(0, 2))
        env2.reset()
        state, reward, done = env2.step(3)  # RIGHT into trap
        self.assertEqual(state, (0, 1))
        self.assertEqual(reward, -10.0)
        self.assertTrue(done)


class TestQLearningAgent(unittest.TestCase):
    """Tests for the Q-learning agent."""

    def setUp(self):
        """Create a basic agent."""
        self.agent = QLearningAgent(
            num_states=9,
            num_actions=4,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.99,
        )

    def test_q_table_shape(self):
        """Q-table should have shape (num_states, num_actions)."""
        self.assertEqual(self.agent.q_table.shape, (9, 4))

    def test_q_table_initialized_to_zero(self):
        """Q-table should start with all zeros."""
        np.testing.assert_array_equal(self.agent.q_table, np.zeros((9, 4)))

    def test_update_changes_q_value(self):
        """A single update should modify the Q-value for the given (state, action)."""
        old_q = self.agent.q_table[0, 0]
        self.agent.update(state_idx=0, action=0, reward=10.0, next_state_idx=1, done=False)
        new_q = self.agent.q_table[0, 0]
        self.assertNotEqual(old_q, new_q)
        # With Q initialized to 0, target = 10 + 0.95*0 = 10
        # New Q = 0 + 0.1 * (10 - 0) = 1.0
        self.assertAlmostEqual(new_q, 1.0)

    def test_terminal_update_no_future(self):
        """For terminal states (done=True), the target should just be the reward."""
        self.agent.update(state_idx=0, action=0, reward=100.0, next_state_idx=1, done=True)
        # target = 100 (no future), new Q = 0 + 0.1 * 100 = 10.0
        self.assertAlmostEqual(self.agent.q_table[0, 0], 10.0)

    def test_epsilon_decay(self):
        """Epsilon should decrease after decay, bounded by epsilon_min."""
        initial = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial)
        # Decay many times — should not go below min
        for _ in range(10000):
            self.agent.decay_epsilon()
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)

    def test_choose_action_explores(self):
        """With ε=1.0, the agent should take random actions (all actions sampled)."""
        random.seed(42)
        actions_seen = set()
        for _ in range(100):
            actions_seen.add(self.agent.choose_action(0))
        # With 100 samples and 4 actions at ε=1.0, we should see all 4
        self.assertEqual(len(actions_seen), 4)

    def test_choose_action_exploits(self):
        """With ε=0, the agent should always pick the best action."""
        self.agent.epsilon = 0.0
        self.agent.q_table[0] = [1.0, 5.0, 2.0, 3.0]  # Action 1 is best
        for _ in range(20):
            self.assertEqual(self.agent.choose_action(0), 1)


class TestTraining(unittest.TestCase):
    """Tests for the full training pipeline."""

    def test_agent_learns_simple_grid(self):
        """Agent should learn to reach the goal in a simple grid."""
        random.seed(42)
        np.random.seed(42)

        # Simple 3x4 grid: start on left, goal on right, one trap
        grid = [
            [START, EMPTY, EMPTY, GOAL],
        ]
        env = GridWorld(grid, start=(0, 0), goal=(0, 3))
        agent = QLearningAgent(
            num_states=env.num_states,
            num_actions=NUM_ACTIONS,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.99,
        )

        agent = train(env, agent, num_episodes=500, max_steps_per_episode=100, verbose_every=0)

        # After training, the greedy policy from start should go RIGHT repeatedly
        state = env.reset()
        for _ in range(10):
            idx = env.state_to_index(state)
            action = int(np.argmax(agent.q_table[idx]))
            state, _, done = env.step(action)
            if done:
                break

        self.assertEqual(state, (0, 3), "Agent should reach the goal")

    def test_training_improves_rewards(self):
        """Average rewards should improve over training."""
        random.seed(42)
        np.random.seed(42)

        grid = [
            [START, EMPTY, EMPTY],
            [EMPTY, WALL,  EMPTY],
            [EMPTY, EMPTY, GOAL],
        ]
        env = GridWorld(grid, start=(0, 0), goal=(2, 2))
        agent = QLearningAgent(
            num_states=env.num_states,
            num_actions=NUM_ACTIONS,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
        )

        agent = train(env, agent, num_episodes=500, max_steps_per_episode=100, verbose_every=0)

        # Later episodes should have better rewards than early ones
        early_avg = np.mean(agent.episode_rewards[:50])
        late_avg = np.mean(agent.episode_rewards[-50:])
        self.assertGreater(late_avg, early_avg, "Agent should improve over training")


if __name__ == "__main__":
    unittest.main()
