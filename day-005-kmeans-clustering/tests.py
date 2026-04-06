"""
Day 005: K-Means Clustering — Test Suite

Run with:
    python3 -m pytest tests.py -v
    python3 tests.py
"""

import unittest
import numpy as np
from itertools import permutations

from my_solution import KMeans, elbow_analysis, find_elbow


def make_blobs(seed=42):
    """Generate 3 well-separated Gaussian blobs for testing."""
    rng = np.random.default_rng(seed)
    true_centers = np.array([[2.0, 2.0], [8.0, 3.0], [5.0, 8.0]])
    cluster_sizes = [100, 150, 120]
    X = np.vstack([
        rng.normal(loc=center, scale=0.8, size=(size, 2))
        for center, size in zip(true_centers, cluster_sizes)
    ])
    true_labels = np.concatenate([np.full(s, i) for i, s in enumerate(cluster_sizes)])
    return X, true_labels, true_centers


class TestKMeansPlusPlus(unittest.TestCase):
    """Tests for K-Means++ initialization."""

    def test_returns_k_centroids(self):
        """_kmeans_plus_plus should return exactly k centroids."""
        X, _, _ = make_blobs()
        km = KMeans(k=3, random_state=0)
        # Need to set rng before calling
        centroids = km._kmeans_plus_plus(X)
        self.assertEqual(centroids.shape, (3, 2))

    def test_centroids_are_unique(self):
        """All k centroids should be distinct points."""
        X, _, _ = make_blobs()
        km = KMeans(k=3, random_state=0)
        centroids = km._kmeans_plus_plus(X)
        # Check no two centroids are identical
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                self.assertFalse(
                    np.allclose(centroids[i], centroids[j]),
                    f"Centroids {i} and {j} are identical"
                )

    def test_centroids_are_from_data(self):
        """Each centroid should be an actual data point."""
        X, _, _ = make_blobs()
        km = KMeans(k=3, random_state=0)
        centroids = km._kmeans_plus_plus(X)
        for c in centroids:
            dists = np.sum((X - c) ** 2, axis=1)
            self.assertAlmostEqual(dists.min(), 0.0, places=10)


class TestAssignClusters(unittest.TestCase):
    """Tests for cluster assignment."""

    def test_assigns_to_nearest_centroid(self):
        """Each point should be assigned to its nearest centroid."""
        km = KMeans(k=2, random_state=0)
        X = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.1], [9.9, 9.9]])
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = km._assign_clusters(X, centroids)
        np.testing.assert_array_equal(labels, [0, 1, 0, 1])


class TestComputeInertia(unittest.TestCase):
    """Tests for inertia (WCSS) computation."""

    def test_zero_inertia_when_centroids_equal_data(self):
        """Inertia should be 0 when centroids exactly match data points."""
        km = KMeans(k=3, random_state=0)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([0, 1, 2])
        centroids = X.copy()
        inertia = km._compute_inertia(X, labels, centroids)
        self.assertAlmostEqual(inertia, 0.0)

    def test_positive_inertia_otherwise(self):
        """Inertia should be positive when points differ from centroids."""
        km = KMeans(k=2, random_state=0)
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        centroids = np.array([[0.5, 0.0], [10.5, 0.0]])
        inertia = km._compute_inertia(X, labels, centroids)
        self.assertGreater(inertia, 0.0)


class TestKMeansFit(unittest.TestCase):
    """Tests for the full K-Means fit pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.true_labels, cls.true_centers = make_blobs(seed=42)

    def test_perfect_clustering_on_well_separated_blobs(self):
        """K-Means with k=3 on 3 well-separated blobs should achieve perfect clustering."""
        km = KMeans(k=3, n_restarts=5, random_state=0)
        km.fit(self.X)

        # Find best label permutation
        best_accuracy = 0
        for perm in permutations(range(3)):
            remapped = np.array([perm[l] for l in km.labels])
            acc = np.mean(remapped == self.true_labels)
            best_accuracy = max(best_accuracy, acc)

        self.assertGreater(best_accuracy, 0.95,
                           f"Expected >95% clustering accuracy, got {best_accuracy:.2%}")

    def test_predict_matches_fit_labels(self):
        """predict() on training data should match the labels from fit()."""
        km = KMeans(k=3, n_restarts=5, random_state=0)
        km.fit(self.X)
        preds = km.predict(self.X)
        np.testing.assert_array_equal(preds, km.labels)

    def test_centroids_have_correct_dimensionality(self):
        """Centroids should have the same dimensionality as input data."""
        km = KMeans(k=3, n_restarts=3, random_state=0)
        km.fit(self.X)
        self.assertEqual(km.centroids.shape, (3, self.X.shape[1]))

    def test_fit_reduces_inertia(self):
        """Fitted inertia should be finite and positive."""
        km = KMeans(k=3, n_restarts=5, random_state=0)
        km.fit(self.X)
        self.assertGreater(km.inertia, 0.0)
        self.assertTrue(np.isfinite(km.inertia))


class TestElbowAnalysis(unittest.TestCase):
    """Tests for elbow analysis functions."""

    @classmethod
    def setUpClass(cls):
        cls.X, _, _ = make_blobs(seed=42)

    def test_inertias_decrease_with_k(self):
        """WCSS (inertia) should generally decrease as k increases."""
        k_range = range(1, 7)
        inertias = elbow_analysis(self.X, k_range, n_restarts=3, seed=42)
        self.assertEqual(len(inertias), len(k_range))
        # Each inertia should be <= the previous (more clusters = lower WCSS)
        for i in range(1, len(inertias)):
            self.assertLessEqual(
                inertias[i], inertias[i - 1] + 1e-6,
                f"Inertia at k={i+1} ({inertias[i]:.1f}) should be <= k={i} ({inertias[i-1]:.1f})"
            )

    def test_find_elbow_returns_valid_k(self):
        """find_elbow should return a k value within the given range."""
        k_range = range(1, 9)
        inertias = elbow_analysis(self.X, k_range, n_restarts=3, seed=42)
        elbow_k = find_elbow(k_range, inertias)
        self.assertIn(elbow_k, list(k_range))

    def test_find_elbow_detects_three_clusters(self):
        """On data with 3 true clusters, elbow should be at k=2, 3, or 4."""
        k_range = range(1, 9)
        inertias = elbow_analysis(self.X, k_range, n_restarts=3, seed=42)
        elbow_k = find_elbow(k_range, inertias)
        self.assertIn(elbow_k, [2, 3, 4],
                      f"Expected elbow near 3, got {elbow_k}")


if __name__ == "__main__":
    unittest.main()
