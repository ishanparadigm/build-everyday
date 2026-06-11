"""
Day 71: Tests for RL Robot Control

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
import math
import random

from my_solution import (
    RobotArmEnv,
    ReplayBuffer,
    Transition,
    NeuralNetwork,
    DQNAgent,
)


class TestRobotArmEnv(unittest.TestCase):
    """Tests for the 2-link robot arm environment."""

    def setUp(self):
        self.env = RobotArmEnv()

    def test_action_space_size(self):
        """Should have 9 discrete actions ({-1,0,1} x {-1,0,1})."""
        self.assertEqual(self.env.n_actions, 9)
        self.assertEqual(len(self.env.action_map), 9)

    def test_state_dimension(self):
        """State should be 6D: θ₁, θ₂, θ̇₁, θ̇₂, x_target, y_target."""
        state = self.env.reset()
        self.assertEqual(len(state), 6)

    def test_forward_kinematics_zero_angles(self):
        """At θ₁=0, θ₂=0, end-effector should be at (l1+l2, 0)."""
        theta = np.array([0.0, 0.0])
        ee = self.env.forward_kinematics(theta)
        expected_x = self.env.l1 + self.env.l2
        self.assertAlmostEqual(ee[0], expected_x, places=5)
        self.assertAlmostEqual(ee[1], 0.0, places=5)

    def test_forward_kinematics_bent(self):
        """At θ₁=0, θ₂=π/2, check FK geometry."""
        theta = np.array([0.0, math.pi / 2])
        ee = self.env.forward_kinematics(theta)
        # link1 points right, link2 points up
        self.assertAlmostEqual(ee[0], self.env.l1, places=5)
        self.assertAlmostEqual(ee[1], self.env.l2, places=5)

    def test_target_is_reachable(self):
        """Sampled target should be within the workspace."""
        for _ in range(50):
            self.env.reset()
            dist = np.linalg.norm(self.env.target)
            max_reach = self.env.l1 + self.env.l2
            self.assertLessEqual(dist, max_reach)

    def test_step_returns_correct_shape(self):
        """Step should return (state, reward, done, info)."""
        self.env.reset()
        state, reward, done, info = self.env.step(0)
        self.assertEqual(len(state), 6)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("distance", info)
        self.assertIn("reached", info)

    def test_episode_terminates(self):
        """Episode should end within max_steps."""
        self.env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = self.env.step(4)  # no-op action (0, 0)
            steps += 1
        self.assertLessEqual(steps, self.env.max_steps)


class TestReplayBuffer(unittest.TestCase):
    """Tests for the experience replay buffer."""

    def test_push_and_length(self):
        """Buffer should track its size correctly."""
        buf = ReplayBuffer(capacity=100)
        self.assertEqual(len(buf), 0)
        buf.push(np.zeros(6), 0, 1.0, np.zeros(6), False)
        self.assertEqual(len(buf), 1)

    def test_capacity_limit(self):
        """Buffer should not exceed capacity."""
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(np.ones(6) * i, 0, 0.0, np.ones(6) * i, False)
        self.assertEqual(len(buf), 10)

    def test_sample_returns_correct_count(self):
        """Sample should return exactly batch_size transitions."""
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(np.ones(6) * i, i % 9, float(i), np.ones(6) * i, False)
        batch = buf.sample(16)
        self.assertEqual(len(batch), 16)

    def test_sample_returns_transitions(self):
        """Each sample should be a Transition with correct types."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(20):
            buf.push(np.random.randn(6), 3, -1.0, np.random.randn(6), True)
        batch = buf.sample(5)
        for t in batch:
            self.assertEqual(len(t.state), 6)
            self.assertIsInstance(t.action, int)
            self.assertIsInstance(t.reward, float)
            self.assertIsInstance(t.done, bool)


class TestNeuralNetwork(unittest.TestCase):
    """Tests for the from-scratch neural network."""

    def test_forward_output_shape(self):
        """Output shape should be (batch_size, n_actions)."""
        net = NeuralNetwork(6, 128, 9)
        x = np.random.randn(16, 6)
        out = net.forward(x)
        self.assertEqual(out.shape, (16, 9))

    def test_forward_single_input(self):
        """Should handle single-sample input."""
        net = NeuralNetwork(6, 128, 9)
        x = np.random.randn(1, 6)
        out = net.forward(x)
        self.assertEqual(out.shape, (1, 9))

    def test_backward_updates_weights(self):
        """Backward pass should change the weights."""
        net = NeuralNetwork(6, 64, 9, lr=0.01)
        x = np.random.randn(8, 6)
        w1_before = net.W1.copy()
        net.forward(x)
        grad = np.random.randn(8, 9)
        net.backward(grad)
        self.assertFalse(np.allclose(net.W1, w1_before))

    def test_copy_weights(self):
        """copy_weights_from should produce identical outputs."""
        net1 = NeuralNetwork(6, 64, 9)
        net2 = NeuralNetwork(6, 64, 9)
        net2.copy_weights_from(net1)
        x = np.random.randn(4, 6)
        out1 = net1.forward(x)
        out2 = net2.forward(x)
        np.testing.assert_array_almost_equal(out1, out2)


class TestDQNAgent(unittest.TestCase):
    """Tests for the DQN agent."""

    def setUp(self):
        np.random.seed(42)
        random.seed(42)
        self.agent = DQNAgent(state_dim=6, n_actions=9)

    def test_select_action_range(self):
        """Actions should be in [0, n_actions)."""
        state = np.random.randn(6)
        for _ in range(100):
            action = self.agent.select_action(state, training=True)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 9)

    def test_select_action_greedy(self):
        """With training=False, action should be deterministic."""
        state = np.random.randn(6)
        actions = [self.agent.select_action(state, training=False) for _ in range(10)]
        self.assertTrue(all(a == actions[0] for a in actions))

    def test_epsilon_decay(self):
        """Epsilon should decrease after decay."""
        eps_before = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, eps_before)

    def test_train_returns_none_when_empty(self):
        """Training should return None when buffer is too small."""
        result = self.agent.train()
        self.assertIsNone(result)

    def test_train_returns_loss(self):
        """After enough transitions, training should return a loss value."""
        for _ in range(100):
            s = np.random.randn(6)
            a = random.randint(0, 8)
            r = random.random()
            ns = np.random.randn(6)
            self.agent.store_transition(s, a, r, ns, False)
        loss = self.agent.train()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)


if __name__ == "__main__":
    unittest.main()
