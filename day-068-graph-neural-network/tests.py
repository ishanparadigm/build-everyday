"""
Tests for Day 068: Graph Neural Network (GCN)

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    Graph, GCNLayer, GCN, cross_entropy_loss, Adam,
    generate_citation_network
)


class TestGraphNormalization(unittest.TestCase):
    """Test adjacency matrix normalization."""

    def test_self_loops_added(self):
        """Normalized adjacency should have nonzero diagonal (self-loops)."""
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=float)
        features = np.eye(3)
        g = Graph(adj, features)

        # Diagonal should be nonzero after adding self-loops
        for i in range(3):
            self.assertGreater(g.adj_norm[i, i], 0,
                               "Self-loop missing in normalized adjacency")

    def test_symmetric_normalization(self):
        """Normalized adjacency should be symmetric."""
        adj = np.array([[0, 1, 1, 0],
                        [1, 0, 1, 1],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0]], dtype=float)
        features = np.eye(4)
        g = Graph(adj, features)

        np.testing.assert_array_almost_equal(
            g.adj_norm, g.adj_norm.T,
            err_msg="Normalized adjacency is not symmetric"
        )

    def test_row_sums_bounded(self):
        """Row sums of normalized adjacency should be approximately 1."""
        rng = np.random.RandomState(42)
        n = 20
        adj = (rng.rand(n, n) > 0.7).astype(float)
        adj = np.triu(adj, 1)
        adj = adj + adj.T  # Make symmetric
        features = rng.randn(n, 5)
        g = Graph(adj, features)

        row_sums = g.adj_norm.sum(axis=1)
        # Row sums should be close to 1 (not exactly, due to normalization)
        for s in row_sums:
            self.assertGreater(s, 0.5, "Row sum too small")
            self.assertLess(s, 1.5, "Row sum too large")

    def test_isolated_node(self):
        """Isolated nodes (no edges) should still work via self-loop."""
        adj = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]], dtype=float)
        features = np.eye(3)
        g = Graph(adj, features)

        # Node 0 is isolated — should have self-loop only
        self.assertGreater(g.adj_norm[0, 0], 0)
        self.assertAlmostEqual(g.adj_norm[0, 1], 0)
        self.assertAlmostEqual(g.adj_norm[0, 2], 0)


class TestGCNLayer(unittest.TestCase):
    """Test individual GCN layer forward and backward."""

    def test_forward_shape(self):
        """Output shape should be (num_nodes, out_features)."""
        layer = GCNLayer(4, 8, activation='relu', seed=42)
        A_hat = np.eye(3)  # Simple identity (each node only sees itself)
        H = np.random.randn(3, 4)
        out = layer.forward(A_hat, H)
        self.assertEqual(out.shape, (3, 8))

    def test_relu_activation(self):
        """ReLU should zero out negative values."""
        layer = GCNLayer(2, 2, activation='relu', seed=42)
        A_hat = np.eye(2)
        H = np.array([[1.0, -1.0], [-1.0, 1.0]])
        out = layer.forward(A_hat, H)
        # All outputs should be >= 0 after ReLU
        self.assertTrue(np.all(out >= 0), "ReLU should zero out negatives")

    def test_no_activation(self):
        """With 'none' activation, output should be the raw linear transform."""
        layer = GCNLayer(2, 2, activation='none', seed=42)
        A_hat = np.eye(2)
        H = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = layer.forward(A_hat, H)
        expected = H @ layer.W + layer.b
        np.testing.assert_array_almost_equal(out, expected)

    def test_backward_gradient_shape(self):
        """Backward should return gradient with same shape as input."""
        layer = GCNLayer(4, 3, activation='relu', seed=42)
        A_hat = np.eye(5)
        H = np.random.randn(5, 4)
        out = layer.forward(A_hat, H)
        d_out = np.ones_like(out)
        d_H = layer.backward(d_out)
        self.assertEqual(d_H.shape, H.shape)
        self.assertEqual(layer.dW.shape, layer.W.shape)
        self.assertEqual(layer.db.shape, layer.b.shape)


class TestCrossEntropyLoss(unittest.TestCase):
    """Test the masked cross-entropy loss function."""

    def test_perfect_predictions(self):
        """Loss should be near zero for confident correct predictions."""
        logits = np.array([[10.0, -10.0, -10.0],
                           [-10.0, 10.0, -10.0]])
        labels = np.array([0, 1])
        mask = np.array([1.0, 1.0])
        loss, _ = cross_entropy_loss(logits, labels, mask)
        self.assertLess(loss, 0.01, "Loss should be near zero for perfect predictions")

    def test_masked_loss(self):
        """Only masked nodes should contribute to loss."""
        logits = np.array([[10.0, -10.0],    # Correct and confident
                           [-10.0, -10.0]])   # Wrong (but masked out)
        labels = np.array([0, 0])
        mask = np.array([1.0, 0.0])  # Only first node
        loss, d_logits = cross_entropy_loss(logits, labels, mask)
        self.assertLess(loss, 0.01)
        # Gradient for masked-out node should be zero
        np.testing.assert_array_almost_equal(d_logits[1], [0.0, 0.0])

    def test_gradient_shape(self):
        """Gradient should have same shape as logits."""
        logits = np.random.randn(10, 4)
        labels = np.random.randint(0, 4, size=10)
        mask = np.ones(10)
        _, d_logits = cross_entropy_loss(logits, labels, mask)
        self.assertEqual(d_logits.shape, logits.shape)


class TestGCNModel(unittest.TestCase):
    """Test the full GCN model."""

    def test_forward_shape(self):
        """Model output should be (num_nodes, num_classes)."""
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        features = np.random.randn(2, 4)
        g = Graph(adj, features, labels=np.array([0, 1]))
        model = GCN(4, 8, 2, dropout=0.0, seed=42)
        model.eval()
        logits = model.forward(g)
        self.assertEqual(logits.shape, (2, 2))

    def test_training_reduces_loss(self):
        """Training for several epochs should reduce the loss."""
        graph, train_mask, _, _ = generate_citation_network(
            num_nodes=50, num_classes=3, feature_dim=8, seed=123
        )
        model = GCN(8, 16, 3, dropout=0.0, seed=42)
        optimizer = Adam(model.parameters(), lr=0.01)

        # Record initial loss
        model.train()
        logits = model.forward(graph)
        initial_loss, _ = cross_entropy_loss(
            logits, graph.labels, train_mask.astype(np.float64))

        # Train for 50 epochs
        for _ in range(50):
            model.train()
            logits = model.forward(graph)
            loss, d_logits = cross_entropy_loss(
                logits, graph.labels, train_mask.astype(np.float64))
            optimizer.zero_grad()
            model.backward(d_logits)
            optimizer.step()

        model.train()
        logits = model.forward(graph)
        final_loss, _ = cross_entropy_loss(
            logits, graph.labels, train_mask.astype(np.float64))

        self.assertLess(final_loss, initial_loss,
                        "Training should reduce the loss")


class TestEndToEnd(unittest.TestCase):
    """Test the complete pipeline."""

    def test_generate_citation_network(self):
        """Generated graph should have correct structure."""
        graph, train_mask, val_mask, test_mask = generate_citation_network(
            num_nodes=100, num_classes=4, feature_dim=8, seed=42
        )
        self.assertEqual(graph.num_nodes, 100)
        self.assertEqual(graph.features.shape, (100, 8))
        self.assertEqual(len(graph.labels), 100)
        self.assertTrue(train_mask.sum() > 0)
        self.assertTrue(val_mask.sum() > 0)
        self.assertTrue(test_mask.sum() > 0)
        # Masks should not overlap
        self.assertEqual((train_mask & val_mask).sum(), 0)
        self.assertEqual((train_mask & test_mask).sum(), 0)

    def test_gcn_beats_random(self):
        """Trained GCN should beat random guessing accuracy."""
        graph, train_mask, val_mask, test_mask = generate_citation_network(
            num_nodes=100, num_classes=4, feature_dim=16, seed=42
        )
        model = GCN(16, 32, 4, dropout=0.3, seed=42)
        optimizer = Adam(model.parameters(), lr=0.01)

        for _ in range(100):
            model.train()
            logits = model.forward(graph)
            loss, d_logits = cross_entropy_loss(
                logits, graph.labels, train_mask.astype(np.float64))
            optimizer.zero_grad()
            model.backward(d_logits)
            optimizer.step()

        model.eval()
        logits = model.forward(graph)
        preds = logits.argmax(axis=1)
        test_acc = (preds[test_mask] == graph.labels[test_mask]).mean()

        # Random guessing for 4 classes = 25% accuracy
        self.assertGreater(test_acc, 0.4,
                           f"GCN accuracy {test_acc:.3f} should beat random (0.25)")


if __name__ == '__main__':
    unittest.main()
