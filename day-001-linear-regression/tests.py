"""
Day 001: Linear Regression -- Test Suite

Run with:
    python3 -m pytest tests.py -v
    python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    generate_data,
    add_bias_column,
    normal_equation,
    gradient_descent,
    standardize,
    mse,
    r_squared,
)


class TestAddBiasColumn(unittest.TestCase):
    """Tests for the add_bias_column function."""

    def test_adds_ones_column(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = add_bias_column(X)
        np.testing.assert_array_equal(result[:, 0], np.ones(3))

    def test_preserves_original_features(self):
        X = np.array([[1, 2], [3, 4]])
        result = add_bias_column(X)
        np.testing.assert_array_equal(result[:, 1:], X)

    def test_output_shape(self):
        X = np.zeros((10, 5))
        result = add_bias_column(X)
        self.assertEqual(result.shape, (10, 6))


class TestNormalEquation(unittest.TestCase):
    """Tests for the normal_equation closed-form solver."""

    def test_recovers_known_weights(self):
        """Normal equation should recover the true weights on noise-free data."""
        np.random.seed(0)
        X = np.random.randn(100, 2)
        true_w = np.array([5.0, 3.0, -2.0])  # bias, w1, w2
        X_aug = np.hstack([np.ones((100, 1)), X])
        y = X_aug @ true_w
        w_hat = normal_equation(X_aug, y)
        np.testing.assert_allclose(w_hat, true_w, atol=1e-10)

    def test_single_feature(self):
        """Should work with a single feature."""
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
        y = np.array([3, 5, 7, 9], dtype=float)  # y = 1 + 2*x
        w = normal_equation(X, y)
        np.testing.assert_allclose(w, [1.0, 2.0], atol=1e-10)


class TestGradientDescent(unittest.TestCase):
    """Tests for gradient_descent iterative solver."""

    def test_converges_to_normal_equation(self):
        """GD should converge to approximately the same weights as normal equation."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        true_w = np.array([1.0, 3.0, -1.5])
        X_aug = np.hstack([np.ones((200, 1)), X])
        y = X_aug @ true_w

        w_ne = normal_equation(X_aug, y)
        w_gd, _ = gradient_descent(X_aug, y, learning_rate=0.01, n_iterations=5000)
        np.testing.assert_allclose(w_gd, w_ne, atol=0.1)

    def test_loss_decreases(self):
        """Loss should decrease monotonically (for appropriate learning rate)."""
        np.random.seed(0)
        X = np.random.randn(50, 2)
        X_aug = np.hstack([np.ones((50, 1)), X])
        y = X_aug @ np.array([1.0, 2.0, 3.0]) + np.random.randn(50) * 0.1

        _, loss_history = gradient_descent(X_aug, y, learning_rate=0.01, n_iterations=200)
        # Check that loss generally decreases (allow minor numerical noise)
        for i in range(1, len(loss_history)):
            self.assertLessEqual(loss_history[i], loss_history[i - 1] + 1e-10)

    def test_returns_loss_history(self):
        """Should return a loss value for each iteration."""
        X = np.random.randn(20, 2)
        X_aug = np.hstack([np.ones((20, 1)), X])
        y = np.random.randn(20)
        _, loss_history = gradient_descent(X_aug, y, n_iterations=100)
        self.assertEqual(len(loss_history), 100)


class TestStandardize(unittest.TestCase):
    """Tests for feature standardization."""

    def test_zero_mean_unit_variance(self):
        """Standardized features should have ~zero mean and ~unit variance."""
        np.random.seed(0)
        X = np.random.randn(1000, 3) * 5 + 10
        X_scaled, _, _ = standardize(X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)

    def test_uses_provided_mean_std(self):
        """Should use provided mean/std instead of computing from data."""
        X = np.array([[2, 4], [4, 6]], dtype=float)
        mean = np.array([3.0, 5.0])
        std = np.array([1.0, 1.0])
        X_scaled, m, s = standardize(X, mean=mean, std=std)
        expected = np.array([[-1, -1], [1, 1]], dtype=float)
        np.testing.assert_array_almost_equal(X_scaled, expected)


class TestMSE(unittest.TestCase):
    """Tests for mean squared error."""

    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mse(y_true, y_pred), 0.0)

    def test_nonzero_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # Each error is 1, so MSE = mean([1, 1, 1]) = 1.0
        self.assertAlmostEqual(mse(y_true, y_pred), 1.0)


class TestRSquared(unittest.TestCase):
    """Tests for R-squared metric."""

    def test_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        self.assertAlmostEqual(r_squared(y_true, y_pred), 1.0)

    def test_mean_prediction(self):
        """Predicting the mean gives R^2 = 0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, y_true.mean())
        self.assertAlmostEqual(r_squared(y_true, y_pred), 0.0)

    def test_worse_than_mean(self):
        """Predictions worse than mean give R^2 < 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])
        self.assertLess(r_squared(y_true, y_pred), 0.0)


if __name__ == "__main__":
    unittest.main()
