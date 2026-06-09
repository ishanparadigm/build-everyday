"""
Tests for Day 069: Mixture of Experts (MoE)

Run with: python3 -m pytest tests.py -v
Or: python3 tests.py
"""

import unittest
import numpy as np


class TestActivations(unittest.TestCase):
    """Test basic activation functions."""

    def test_relu_positive(self):
        from my_solution import relu
        x = np.array([1.0, 2.0, 3.0])
        result = relu(x)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_relu_negative(self):
        from my_solution import relu
        x = np.array([-1.0, -2.0, 0.0, 1.0])
        result = relu(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 1.0])

    def test_softmax_sums_to_one(self):
        from my_solution import softmax
        x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        result = softmax(x)
        np.testing.assert_allclose(np.sum(result, axis=-1), [1.0, 1.0], atol=1e-6)

    def test_softmax_numerical_stability(self):
        """Softmax should handle large values without overflow."""
        from my_solution import softmax
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = softmax(x)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(np.sum(result), 1.0, atol=1e-6)

    def test_softplus_positive(self):
        from my_solution import softplus
        x = np.array([0.0, 1.0, 5.0])
        result = softplus(x)
        self.assertTrue(np.all(result > 0))


class TestExpert(unittest.TestCase):
    """Test the Expert MLP."""

    def test_expert_output_shape(self):
        from my_solution import Expert
        expert = Expert(input_dim=4, hidden_dim=8, output_dim=3, expert_id=0)
        x = np.random.randn(10, 4)
        out = expert.forward(x)
        self.assertEqual(out.shape, (10, 3))

    def test_expert_backward_shape(self):
        from my_solution import Expert
        expert = Expert(input_dim=4, hidden_dim=8, output_dim=3, expert_id=0)
        x = np.random.randn(10, 4)
        out = expert.forward(x)
        grad = np.random.randn(10, 3)
        grad_x = expert.backward(grad)
        self.assertEqual(grad_x.shape, (10, 4))

    def test_expert_different_weights(self):
        """Two experts should have different weights (independent init)."""
        from my_solution import Expert
        np.random.seed(42)
        e1 = Expert(input_dim=4, hidden_dim=8, output_dim=3, expert_id=0)
        e2 = Expert(input_dim=4, hidden_dim=8, output_dim=3, expert_id=1)
        self.assertFalse(np.allclose(e1.W1, e2.W1))


class TestGatingNetwork(unittest.TestCase):
    """Test the gating network (router)."""

    def test_gate_output_shape(self):
        from my_solution import GatingNetwork
        gate = GatingNetwork(input_dim=4, num_experts=8, top_k=2)
        x = np.random.randn(10, 4)
        weights, probs = gate.forward(x)
        self.assertEqual(weights.shape, (10, 8))
        self.assertEqual(probs.shape, (10, 8))

    def test_gate_top_k_sparsity(self):
        """Gate weights should have exactly top_k non-zero entries per row."""
        from my_solution import GatingNetwork
        gate = GatingNetwork(input_dim=4, num_experts=8, top_k=2)
        x = np.random.randn(20, 4)
        weights, _ = gate.forward(x)
        nonzero_per_row = np.sum(weights > 0, axis=1)
        np.testing.assert_array_equal(nonzero_per_row, np.full(20, 2))

    def test_gate_weights_sum_to_one(self):
        """After renormalization, gate weights should sum to 1."""
        from my_solution import GatingNetwork
        gate = GatingNetwork(input_dim=4, num_experts=8, top_k=2)
        x = np.random.randn(10, 4)
        weights, _ = gate.forward(x)
        row_sums = np.sum(weights, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(10), atol=1e-5)

    def test_full_probs_are_valid_distribution(self):
        """Full probs should be a valid probability distribution."""
        from my_solution import GatingNetwork
        gate = GatingNetwork(input_dim=4, num_experts=8, top_k=2)
        x = np.random.randn(10, 4)
        _, probs = gate.forward(x)
        self.assertTrue(np.all(probs >= 0))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(10), atol=1e-5)


class TestMoELayer(unittest.TestCase):
    """Test the MoE layer."""

    def test_moe_output_shape(self):
        from my_solution import MoELayer
        moe = MoELayer(input_dim=4, hidden_dim=8, output_dim=3,
                       num_experts=4, top_k=2)
        x = np.random.randn(10, 4)
        out, bl = moe.forward(x)
        self.assertEqual(out.shape, (10, 3))

    def test_moe_balance_loss_positive(self):
        """Balance loss should be a positive scalar."""
        from my_solution import MoELayer
        moe = MoELayer(input_dim=4, hidden_dim=8, output_dim=3,
                       num_experts=4, top_k=2, balance_coeff=0.01)
        x = np.random.randn(50, 4)
        _, bl = moe.forward(x)
        self.assertGreater(bl, 0)

    def test_moe_utilization_tracking(self):
        """Expert utilization should be tracked correctly."""
        from my_solution import MoELayer
        moe = MoELayer(input_dim=4, hidden_dim=8, output_dim=3,
                       num_experts=4, top_k=2)
        moe.reset_utilization()
        x = np.random.randn(100, 4)
        moe.forward(x)
        util = moe.get_utilization()
        # With top-2, each input uses 2 experts, so average utilization = 2/4 = 0.5
        self.assertAlmostEqual(np.sum(util), 2.0, places=1)


class TestMoEClassifier(unittest.TestCase):
    """Test the full MoE classifier."""

    def test_classifier_trains(self):
        """The classifier should be able to reduce loss on a simple problem."""
        from my_solution import MoEClassifier, cross_entropy_loss, generate_multi_region_data

        np.random.seed(42)
        X, y = generate_multi_region_data(n_samples=300, n_classes=3, seed=42)

        model = MoEClassifier(
            input_dim=2, num_classes=3,
            moe_hidden=16, moe_output=8,
            num_experts=4, top_k=2,
            balance_coeff=0.01
        )

        # Compute initial loss
        model.set_training(True)
        logits_before, _ = model.forward(X)
        loss_before, _ = cross_entropy_loss(logits_before, y)

        # Train for a few steps
        for _ in range(30):
            perm = np.random.permutation(len(X))
            for start in range(0, len(X), 64):
                end = min(start + 64, len(X))
                xb, yb = X[perm[start:end]], y[perm[start:end]]
                logits, bl = model.forward(xb)
                loss, grad = cross_entropy_loss(logits, yb)
                model.backward(grad)
                model.update(0.05)

        logits_after, _ = model.forward(X)
        loss_after, _ = cross_entropy_loss(logits_after, y)

        self.assertLess(loss_after, loss_before,
                       f"Loss should decrease: {loss_before:.4f} -> {loss_after:.4f}")


class TestCrossEntropyLoss(unittest.TestCase):
    """Test the loss function."""

    def test_loss_shape(self):
        from my_solution import cross_entropy_loss
        logits = np.random.randn(10, 5)
        labels = np.random.randint(0, 5, size=10)
        loss, grad = cross_entropy_loss(logits, labels)
        self.assertIsInstance(loss, float)
        self.assertEqual(grad.shape, (10, 5))

    def test_loss_perfect_predictions(self):
        """Loss should be low when logits strongly favor correct class."""
        from my_solution import cross_entropy_loss
        logits = np.array([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        labels = np.array([0, 1])
        loss, _ = cross_entropy_loss(logits, labels)
        self.assertLess(loss, 0.01)

    def test_loss_gradient_direction(self):
        """Gradient should push logits toward correct class."""
        from my_solution import cross_entropy_loss
        logits = np.array([[0.0, 0.0, 0.0]])
        labels = np.array([1])
        _, grad = cross_entropy_loss(logits, labels)
        # Gradient for correct class (1) should be negative (push logit up)
        self.assertLess(grad[0, 1], 0)


if __name__ == '__main__':
    unittest.main()
