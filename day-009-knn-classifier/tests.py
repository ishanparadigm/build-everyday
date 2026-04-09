"""
Day 009: KNN Classifier Tests

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
    StandardScaler,
    KNNClassifier,
    cross_validate_k,
)


class TestDistanceMetrics(unittest.TestCase):
    """Test distance metric implementations."""

    def test_euclidean_basic(self):
        """Euclidean distance of a 3-4-5 triangle."""
        d = euclidean_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        self.assertAlmostEqual(d, 5.0)

    def test_euclidean_same_point(self):
        """Distance from a point to itself is zero."""
        p = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(euclidean_distance(p, p), 0.0)

    def test_manhattan_basic(self):
        """Manhattan distance is sum of absolute differences."""
        d = manhattan_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        self.assertAlmostEqual(d, 7.0)

    def test_manhattan_negative(self):
        """Manhattan distance works with negative coordinates."""
        d = manhattan_distance(np.array([-1.0, -2.0]), np.array([1.0, 2.0]))
        self.assertAlmostEqual(d, 6.0)

    def test_minkowski_p1_is_manhattan(self):
        """Minkowski with p=1 should equal Manhattan distance."""
        a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
        self.assertAlmostEqual(
            minkowski_distance(a, b, p=1),
            manhattan_distance(a, b),
            places=5,
        )

    def test_minkowski_p2_is_euclidean(self):
        """Minkowski with p=2 should equal Euclidean distance."""
        a, b = np.array([1.0, 2.0]), np.array([4.0, 6.0])
        self.assertAlmostEqual(
            minkowski_distance(a, b, p=2),
            euclidean_distance(a, b),
            places=5,
        )


class TestStandardScaler(unittest.TestCase):
    """Test feature scaling."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.X = self.rng.randn(100, 3) * np.array([10, 0.1, 5]) + np.array([100, -5, 0])

    def test_zero_mean_unit_variance(self):
        """After fit_transform, mean should be ~0 and std ~1."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), [0, 0, 0], decimal=10)
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), [1, 1, 1], decimal=10)

    def test_transform_uses_train_params(self):
        """transform() on new data should use the stored mean/std, not recompute."""
        scaler = StandardScaler()
        scaler.fit(self.X)

        X_new = self.rng.randn(10, 3) * np.array([10, 0.1, 5]) + np.array([100, -5, 0])
        X_new_scaled = scaler.transform(X_new)

        # New data mean won't be exactly 0 — that's correct, because we're
        # using the training set's parameters
        self.assertFalse(
            np.allclose(X_new_scaled.mean(axis=0), [0, 0, 0], atol=0.01),
            "New data should NOT have exact zero mean (that would mean you refit)"
        )

    def test_constant_feature_handled(self):
        """A constant feature (zero std) shouldn't cause division by zero."""
        X_const = np.column_stack([self.X, np.ones(100) * 5.0])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_const)
        self.assertTrue(np.all(np.isfinite(X_scaled)))


class TestKNNClassifier(unittest.TestCase):
    """Test the KNN classifier."""

    def setUp(self):
        """Create a simple 2-class dataset with clear separation."""
        rng = np.random.RandomState(42)
        n = 50
        self.X_train = np.vstack([
            rng.randn(n, 2) + np.array([3, 3]),
            rng.randn(n, 2) + np.array([-3, -3]),
        ])
        self.y_train = np.array([0] * n + [1] * n)
        self.X_test = np.array([[2.5, 2.5], [-2.5, -2.5], [0, 0]])
        self.y_test = np.array([0, 1, 0])  # Boundary point — either class is OK

    def test_k1_memorizes_training_data(self):
        """With k=1, training accuracy should be 100%."""
        knn = KNNClassifier(k=1)
        knn.fit(self.X_train, self.y_train)
        self.assertEqual(knn.score(self.X_train, self.y_train), 1.0)

    def test_well_separated_clusters(self):
        """Points clearly in one cluster should be classified correctly."""
        knn = KNNClassifier(k=5)
        knn.fit(self.X_train, self.y_train)
        preds = knn.predict(self.X_test[:2])
        np.testing.assert_array_equal(preds, [0, 1])

    def test_distance_weighted_voting(self):
        """Distance-weighted KNN should run without errors and return valid classes."""
        knn = KNNClassifier(k=5, weights="distance")
        knn.fit(self.X_train, self.y_train)
        preds = knn.predict(self.X_test)
        # All predictions should be valid class labels
        for p in preds:
            self.assertIn(p, [0, 1])

    def test_manhattan_metric(self):
        """KNN should work with Manhattan distance."""
        knn = KNNClassifier(k=5, metric="manhattan")
        knn.fit(self.X_train, self.y_train)
        acc = knn.score(self.X_test[:2], self.y_test[:2])
        self.assertEqual(acc, 1.0)

    def test_invalid_k_raises(self):
        """k < 1 should raise ValueError."""
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)

    def test_predict_shape(self):
        """predict() should return array with same length as input."""
        knn = KNNClassifier(k=3)
        knn.fit(self.X_train, self.y_train)
        preds = knn.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))


class TestCrossValidation(unittest.TestCase):
    """Test cross-validation for k selection."""

    def test_returns_all_k_values(self):
        """cross_validate_k should return a result for each k tested."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        k_values = [1, 3, 5, 7]
        results = cross_validate_k(X, y, k_values, n_folds=3)

        self.assertEqual(set(results.keys()), set(k_values))
        for acc in results.values():
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_reasonable_accuracy(self):
        """On a well-separated dataset, CV accuracy should be high."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(50, 2) + [5, 5], rng.randn(50, 2) + [-5, -5]])
        y = np.array([0] * 50 + [1] * 50)

        results = cross_validate_k(X, y, [3, 5], n_folds=5)
        # Well-separated clusters => accuracy should be > 0.9
        self.assertGreater(max(results.values()), 0.9)


if __name__ == "__main__":
    unittest.main()
