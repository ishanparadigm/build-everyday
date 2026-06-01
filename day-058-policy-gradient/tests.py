"""
Day 058: Policy Gradient Methods — Test Suite

Tests for the REINFORCE policy gradient implementation.
Covers: activation functions, softmax, policy network, return computation,
normalization, CartPole environment, and end-to-end training.

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import unittest
import numpy as np

from my_solution import (
    relu,
    relu_derivative,
    softmax,
    PolicyNetwork,
    REINFORCEAgent,
    CartPoleEnv,
    train_reinforce,
    evaluate_agent,
)


class TestActivations(unittest.TestCase):
    """Test ReLU and its derivative."""

    def test_relu_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        result = relu(x)
        np.testing.assert_array_equal(result, x)

    def test_relu_negative(self):
        x = np.array([-1.0, -2.0, -3.0])
        result = relu(x)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_relu_mixed(self):
        x = np.array([-2.0, 0.0, 3.0, -1.0, 5.0])
        expected = np.array([0.0, 0.0, 3.0, 0.0, 5.0])
        np.testing.assert_array_equal(relu(x), expected)

    def test_relu_derivative_values(self):
        x = np.array([-2.0, 0.0, 3.0, -1.0, 5.0])
        expected = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(relu_derivative(x), expected)


class TestSoftmax(unittest.TestCase):
    """Test softmax function properties."""

    def test_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = softmax(logits)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

    def test_all_positive(self):
        logits = np.array([-5.0, 0.0, 5.0])
        probs = softmax(logits)
        self.assertTrue(np.all(probs > 0))

    def test_uniform_for_equal_logits(self):
        logits = np.array([3.0, 3.0, 3.0])
        probs = softmax(logits)
        np.testing.assert_array_almost_equal(probs, [1/3, 1/3, 1/3], decimal=6)

    def test_numerical_stability(self):
        """Large logits should not cause overflow."""
        logits = np.array([1000.0, 1000.0])
        probs = softmax(logits)
        self.assertFalse(np.any(np.isnan(probs)))
        self.assertFalse(np.any(np.isinf(probs)))
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

    def test_ordering_preserved(self):
        """Higher logit → higher probability."""
        logits = np.array([1.0, 3.0])
        probs = softmax(logits)
        self.assertGreater(probs[1], probs[0])


class TestPolicyNetwork(unittest.TestCase):
    """Test the neural network policy."""

    def setUp(self):
        self.policy = PolicyNetwork(state_dim=4, hidden_dim=16, action_dim=2, seed=42)

    def test_output_is_probability(self):
        state = np.array([0.1, -0.2, 0.05, 0.0])
        probs = self.policy.forward(state)
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
        self.assertTrue(np.all(probs > 0))

    def test_deterministic_with_seed(self):
        """Same seed → same output."""
        p1 = PolicyNetwork(state_dim=4, hidden_dim=16, action_dim=2, seed=99)
        p2 = PolicyNetwork(state_dim=4, hidden_dim=16, action_dim=2, seed=99)
        state = np.array([0.1, -0.2, 0.05, 0.0])
        np.testing.assert_array_almost_equal(p1.forward(state), p2.forward(state))

    def test_backward_returns_correct_keys(self):
        state = np.array([0.1, -0.2, 0.05, 0.0])
        self.policy.forward(state)
        grads = self.policy.backward(action=0, advantage=1.0)
        self.assertIn('W1', grads)
        self.assertIn('b1', grads)
        self.assertIn('W2', grads)
        self.assertIn('b2', grads)

    def test_backward_gradient_shapes(self):
        state = np.array([0.1, -0.2, 0.05, 0.0])
        self.policy.forward(state)
        grads = self.policy.backward(action=1, advantage=0.5)
        self.assertEqual(grads['W1'].shape, self.policy.W1.shape)
        self.assertEqual(grads['b1'].shape, self.policy.b1.shape)
        self.assertEqual(grads['W2'].shape, self.policy.W2.shape)
        self.assertEqual(grads['b2'].shape, self.policy.b2.shape)


class TestREINFORCEAgent(unittest.TestCase):
    """Test the REINFORCE agent components."""

    def setUp(self):
        self.agent = REINFORCEAgent(
            state_dim=4, action_dim=2, hidden_dim=16,
            learning_rate=0.01, gamma=0.99, seed=42
        )

    def test_compute_returns_simple(self):
        """Test return computation with known values."""
        for _ in range(3):
            self.agent.store_transition(np.zeros(4), 0, 1.0)
        returns = self.agent.compute_returns()
        # G₀ = 1 + 0.99 + 0.99² = 2.9701
        # G₁ = 1 + 0.99 = 1.99
        # G₂ = 1
        self.assertAlmostEqual(returns[0], 1 + 0.99 + 0.99**2, places=4)
        self.assertAlmostEqual(returns[1], 1 + 0.99, places=4)
        self.assertAlmostEqual(returns[2], 1.0, places=4)

    def test_compute_returns_discounting(self):
        """Later timesteps should have lower returns (for constant rewards)."""
        for _ in range(5):
            self.agent.store_transition(np.zeros(4), 0, 1.0)
        returns = self.agent.compute_returns()
        # Returns should be decreasing
        for i in range(len(returns) - 1):
            self.assertGreater(returns[i], returns[i + 1])

    def test_normalize_returns(self):
        """Normalized returns should have ~zero mean and ~unit std."""
        returns = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normed = self.agent.normalize_returns(returns)
        self.assertAlmostEqual(np.mean(normed), 0.0, places=5)
        self.assertAlmostEqual(np.std(normed), 1.0, places=1)

    def test_normalize_constant_returns(self):
        """If all returns are identical, should return zeros (no signal)."""
        returns = np.array([5.0, 5.0, 5.0])
        normed = self.agent.normalize_returns(returns)
        np.testing.assert_array_almost_equal(normed, np.zeros(3), decimal=5)

    def test_select_action_valid(self):
        """Action should be 0 or 1 for CartPole."""
        state = np.array([0.01, -0.02, 0.03, -0.01])
        action, probs = self.agent.select_action(state)
        self.assertIn(action, [0, 1])
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)


class TestCartPoleEnv(unittest.TestCase):
    """Test the CartPole environment."""

    def setUp(self):
        self.env = CartPoleEnv(seed=42)

    def test_reset_returns_state(self):
        state = self.env.reset()
        self.assertEqual(len(state), 4)
        # State should be near zero (close to equilibrium)
        self.assertTrue(np.all(np.abs(state) < 0.1))

    def test_step_returns_tuple(self):
        self.env.reset()
        next_state, reward, done = self.env.step(1)
        self.assertEqual(len(next_state), 4)
        self.assertEqual(reward, 1.0)
        self.assertIsInstance(done, (bool, np.bool_))

    def test_episode_terminates(self):
        """An episode with random actions should eventually terminate."""
        self.env.reset()
        rng = np.random.RandomState(42)
        for _ in range(1000):
            _, _, done = self.env.step(rng.choice(2))
            if done:
                break
        self.assertTrue(done, "Episode should terminate within 1000 steps")

    def test_max_steps_limit(self):
        """Episode should terminate at max_steps even if pole is balanced."""
        self.env.reset()
        # Even with perfect balancing, should stop at 500
        steps = 0
        done = False
        while not done and steps < 600:
            _, _, done = self.env.step(1 if self.env.state[2] > 0 else 0)
            steps += 1
        self.assertLessEqual(steps, 500)


class TestEndToEnd(unittest.TestCase):
    """Test that training actually improves the agent."""

    def test_training_improves_performance(self):
        """After training, the agent should score significantly above random."""
        agent, rewards = train_reinforce(
            num_episodes=300,
            hidden_dim=32,
            learning_rate=0.01,
            gamma=0.99,
            seed=42,
            print_every=1000  # suppress printing
        )
        # Random policy scores ~20-30 on CartPole
        # After 300 episodes, agent should do meaningfully better
        late_avg = np.mean(rewards[-50:])
        early_avg = np.mean(rewards[:50])
        self.assertGreater(late_avg, early_avg,
                           f"Agent should improve: early={early_avg:.1f}, late={late_avg:.1f}")


if __name__ == '__main__':
    unittest.main()
