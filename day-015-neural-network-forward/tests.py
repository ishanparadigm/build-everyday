"""
Tests for Day 015: Neural Network Forward Pass

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    relu, sigmoid, softmax, initialize_weights,
    NeuralNetwork, one_hot_encode, generate_spiral_data
)


class TestActivationFunctions(unittest.TestCase):
    """Test activation functions for correctness and numerical stability."""

    def test_relu_basic(self):
        """ReLU should pass positives, zero negatives."""
        z = np.array([[-3.0, -1.0, 0.0, 1.0, 3.0]])
        result = relu(z)
        expected = np.array([[0.0, 0.0, 0.0, 1.0, 3.0]])
        np.testing.assert_array_equal(result, expected)

    def test_relu_preserves_shape(self):
        """ReLU should not change input shape."""
        z = np.random.randn(10, 5)
        self.assertEqual(relu(z).shape, z.shape)

    def test_sigmoid_range(self):
        """Sigmoid output must be in (0, 1)."""
        z = np.random.randn(100, 50) * 10
        result = sigmoid(z)
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(result < 1))

    def test_sigmoid_known_values(self):
        """Sigmoid(0) = 0.5, sigmoid is monotonically increasing."""
        z = np.array([[0.0]])
        self.assertAlmostEqual(sigmoid(z)[0, 0], 0.5, places=5)
        # Monotonicity
        z2 = np.array([[-5.0, 0.0, 5.0]])
        result = sigmoid(z2)[0]
        self.assertTrue(result[0] < result[1] < result[2])

    def test_sigmoid_numerical_stability(self):
        """Sigmoid should handle extreme values without overflow."""
        z = np.array([[-1000.0, 1000.0]])
        result = sigmoid(z)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_softmax_sums_to_one(self):
        """Softmax outputs must sum to 1 per row."""
        z = np.random.randn(10, 5)
        result = softmax(z)
        row_sums = np.sum(result, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-7)

    def test_softmax_positive(self):
        """Softmax outputs must all be positive."""
        z = np.random.randn(10, 5) * 100
        result = softmax(z)
        self.assertTrue(np.all(result > 0))

    def test_softmax_numerical_stability(self):
        """Softmax should handle large values without overflow."""
        z = np.array([[1000.0, 1001.0, 999.0]])
        result = softmax(z)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        np.testing.assert_allclose(np.sum(result), 1.0, atol=1e-7)


class TestWeightInitialization(unittest.TestCase):
    """Test that He initialization produces correct shapes and statistics."""

    def test_correct_shapes(self):
        """Weight matrices and biases must have correct dimensions."""
        params = initialize_weights([4, 8, 3])
        self.assertEqual(params[0]['W'].shape, (4, 8))
        self.assertEqual(params[0]['b'].shape, (1, 8))
        self.assertEqual(params[1]['W'].shape, (8, 3))
        self.assertEqual(params[1]['b'].shape, (1, 3))

    def test_he_initialization_scale(self):
        """He init std should be approximately sqrt(2/n_in)."""
        params = initialize_weights([1000, 500])
        expected_std = np.sqrt(2.0 / 1000)
        actual_std = np.std(params[0]['W'])
        self.assertAlmostEqual(actual_std, expected_std, places=2)

    def test_biases_zero(self):
        """Biases should be initialized to zero."""
        params = initialize_weights([10, 20, 5])
        for p in params:
            np.testing.assert_array_equal(p['b'], np.zeros_like(p['b']))


class TestNeuralNetwork(unittest.TestCase):
    """Test the full neural network forward pass."""

    def test_output_shape(self):
        """Output shape must be (batch_size, n_classes)."""
        nn = NeuralNetwork([4, 8, 3])
        X = np.random.randn(10, 4)
        output, _ = nn.forward(X)
        self.assertEqual(output.shape, (10, 3))

    def test_output_is_probability_distribution(self):
        """Output must be valid probabilities (positive, sum to 1)."""
        nn = NeuralNetwork([2, 16, 4])
        X = np.random.randn(20, 2)
        output, _ = nn.forward(X)
        self.assertTrue(np.all(output > 0))
        np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-7)

    def test_cache_length(self):
        """Cache should have one entry per layer."""
        nn = NeuralNetwork([2, 8, 4, 3])
        X = np.random.randn(5, 2)
        _, cache = nn.forward(X)
        self.assertEqual(len(cache), 3)  # 3 weight layers

    def test_loss_positive(self):
        """Cross-entropy loss should be positive."""
        nn = NeuralNetwork([2, 8, 3])
        X = np.random.randn(10, 2)
        y = one_hot_encode(np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]), 3)
        output, _ = nn.forward(X)
        loss = nn.compute_loss(output, y)
        self.assertGreater(loss, 0)

    def test_predict_valid_classes(self):
        """Predictions must be valid class indices."""
        nn = NeuralNetwork([2, 8, 5])
        X = np.random.randn(20, 2)
        preds = nn.predict(X)
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds < 5))

    def test_accuracy_range(self):
        """Accuracy must be between 0 and 1."""
        nn = NeuralNetwork([2, 8, 3])
        X = np.random.randn(30, 2)
        y = np.random.randint(0, 3, 30)
        acc = nn.accuracy(X, y)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


class TestDataUtilities(unittest.TestCase):
    """Test data generation and encoding."""

    def test_one_hot_encode(self):
        """One-hot encoding should produce correct vectors."""
        labels = np.array([0, 2, 1])
        result = one_hot_encode(labels, 3)
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_spiral_data_shape(self):
        """Spiral data should have correct shape and class balance."""
        X, y = generate_spiral_data(n_samples_per_class=50, n_classes=3)
        self.assertEqual(X.shape, (150, 2))
        self.assertEqual(y.shape, (150,))
        self.assertTrue(np.all(np.bincount(y) == 50))


if __name__ == '__main__':
    unittest.main()
