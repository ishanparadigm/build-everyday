"""
Day 003: Logistic Regression -- Test Suite

Run with:
    python3 -m pytest tests.py -v
    python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    LogisticRegression,
    compute_metrics,
    generate_binary_dataset,
)


class TestSigmoid(unittest.TestCase):
    """Tests for the sigmoid activation function."""

    def test_sigmoid_zero(self):
        """sigmoid(0) should be exactly 0.5."""
        result = LogisticRegression._sigmoid(np.array([0.0]))
        self.assertAlmostEqual(result[0], 0.5)

    def test_sigmoid_large_positive(self):
        """sigmoid of large positive number should be close to 1.0."""
        result = LogisticRegression._sigmoid(np.array([100.0]))
        self.assertAlmostEqual(result[0], 1.0, places=5)

    def test_sigmoid_large_negative(self):
        """sigmoid of large negative number should be close to 0.0."""
        result = LogisticRegression._sigmoid(np.array([-100.0]))
        self.assertAlmostEqual(result[0], 0.0, places=5)

    def test_sigmoid_symmetry(self):
        """sigmoid(-z) should equal 1 - sigmoid(z)."""
        z = np.array([0.5, 1.0, 2.0, -3.0])
        s_pos = LogisticRegression._sigmoid(z)
        s_neg = LogisticRegression._sigmoid(-z)
        np.testing.assert_allclose(s_neg, 1.0 - s_pos, atol=1e-10)

    def test_sigmoid_no_overflow(self):
        """Should handle extreme values without overflow warnings."""
        z = np.array([-1000.0, 1000.0])
        result = LogisticRegression._sigmoid(z)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], 1.0, places=5)


class TestPredictProba(unittest.TestCase):
    """Tests for probability predictions."""

    def test_values_between_0_and_1(self):
        """All predicted probabilities must be in [0, 1]."""
        X_train, X_test, y_train, y_test = generate_binary_dataset(seed=42)
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        self.assertTrue(np.all(proba >= 0.0))
        self.assertTrue(np.all(proba <= 1.0))

    def test_output_shape(self):
        """predict_proba should return one value per sample."""
        X_train, X_test, y_train, _ = generate_binary_dataset(seed=42)
        model = LogisticRegression(n_iterations=100)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        self.assertEqual(proba.shape, (X_test.shape[0],))


class TestTraining(unittest.TestCase):
    """Tests for the training loop."""

    def test_loss_decreases(self):
        """Loss should decrease monotonically during training."""
        X_train, _, y_train, _ = generate_binary_dataset(seed=42)
        model = LogisticRegression(learning_rate=0.1, n_iterations=300)
        model.fit(X_train, y_train)

        # Allow tiny numerical noise
        for i in range(1, len(model.loss_history)):
            self.assertLessEqual(
                model.loss_history[i],
                model.loss_history[i - 1] + 1e-10,
                f"Loss increased at iteration {i}: "
                f"{model.loss_history[i]} > {model.loss_history[i - 1]}"
            )

    def test_accuracy_on_separable_data(self):
        """Model should achieve >80% accuracy on well-separated data."""
        X_train, X_test, y_train, y_test = generate_binary_dataset(
            n_samples=400, separation=2.0, seed=42
        )
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        self.assertGreater(accuracy, 0.80,
                           f"Accuracy {accuracy:.2%} is below 80% threshold")

    def test_loss_history_length(self):
        """loss_history should have one entry per iteration."""
        X_train, _, y_train, _ = generate_binary_dataset(seed=42)
        model = LogisticRegression(n_iterations=150)
        model.fit(X_train, y_train)
        self.assertEqual(len(model.loss_history), 150)


class TestComputeMetrics(unittest.TestCase):
    """Tests for the compute_metrics function."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1_score"], 1.0)

    def test_known_confusion_matrix(self):
        """Verify precision and recall for a known confusion matrix."""
        # 2 TP, 1 FP, 1 FN, 1 TN
        y_true = np.array([1, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        # precision = 2/(2+1) = 2/3
        self.assertAlmostEqual(m["precision"], 2.0 / 3.0, places=5)
        # recall = 2/(2+1) = 2/3
        self.assertAlmostEqual(m["recall"], 2.0 / 3.0, places=5)

    def test_no_positive_predictions(self):
        """Should handle the case where no positive predictions are made."""
        y_true = np.array([1, 1, 0])
        y_pred = np.array([0, 0, 0])
        m = compute_metrics(y_true, y_pred)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)


class TestGenerateBinaryDataset(unittest.TestCase):
    """Tests for the data generation utility."""

    def test_output_shapes(self):
        X_train, X_test, y_train, y_test = generate_binary_dataset(
            n_samples=100, n_features=3, seed=0
        )
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(X_train.shape[1], 3)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

    def test_binary_labels(self):
        """Labels should only contain 0 and 1."""
        _, _, y_train, y_test = generate_binary_dataset(seed=0)
        all_labels = np.concatenate([y_train, y_test])
        unique = set(all_labels)
        self.assertTrue(unique.issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
