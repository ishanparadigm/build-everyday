"""
Day 009: K-Nearest Neighbors (KNN) Classifier — Your Implementation

Implement KNN from scratch. The key components:
1. Distance metrics (Euclidean, Manhattan)
2. Feature scaling (StandardScaler)
3. KNN classifier with uniform and distance-weighted voting
4. Cross-validation for k selection

Hint: Start with euclidean_distance and KNNClassifier with uniform voting.
Once that works, add distance-weighted voting and the scaler.
"""

import numpy as np
from collections import Counter
from typing import Literal


# =============================================================================
# Distance Metrics
# =============================================================================

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute L2 (Euclidean) distance between two points.

    Hint: np.sqrt and np.sum are your friends. Remember to square the
    differences element-wise before summing.
    """
    raise NotImplementedError("TODO: implement this")


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute L1 (Manhattan) distance between two points.

    Hint: Sum of absolute differences. No squaring needed.
    """
    raise NotImplementedError("TODO: implement this")


def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float = 2) -> float:
    """
    Compute Minkowski distance with parameter p.
    p=1 gives Manhattan, p=2 gives Euclidean.

    Hint: Generalize the pattern: raise |diff| to power p, sum, then
    take the p-th root.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Feature Scaling
# =============================================================================

class StandardScaler:
    """
    Z-score normalization: zero mean, unit variance.

    Hint: fit() computes mean and std from training data.
    transform() applies (X - mean) / std using the stored parameters.
    Be careful with zero-std features (constant columns).
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and std from training data."""
        raise NotImplementedError("TODO: implement this")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned scaling parameters."""
        raise NotImplementedError("TODO: implement this")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit and transform in one call."""
        return self.fit(X).transform(X)


# =============================================================================
# KNN Classifier
# =============================================================================

class KNNClassifier:
    """
    K-Nearest Neighbors classifier.

    Hint: The core algorithm is simple:
    1. Store training data in fit()
    2. For each test point in predict():
       a. Compute distance to ALL training points
       b. Find the k closest
       c. Take a majority vote of their labels
    """

    def __init__(
        self,
        k: int = 5,
        metric: Literal["euclidean", "manhattan", "minkowski"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
        p: float = 2,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")

        self.k = k
        self.weights = weights
        self.p = p

        # Hint: map metric names to your distance functions
        self._metric_fn = None  # TODO: set this based on metric parameter

        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        Store training data. KNN has no training phase —
        it's a 'lazy learner' that memorizes the dataset.
        """
        raise NotImplementedError("TODO: implement this")

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Compute distance from query point x to all training points.
        Returns array of shape (n_train,).

        Hint: Loop over self.X_train and apply self._metric_fn.
        """
        raise NotImplementedError("TODO: implement this")

    def _get_k_nearest(self, distances: np.ndarray) -> np.ndarray:
        """
        Return indices of k nearest neighbors, sorted by distance.

        Hint: np.argpartition is O(n) for finding k smallest elements.
        Much faster than full sort when k << n.
        """
        raise NotImplementedError("TODO: implement this")

    def _predict_single(self, x: np.ndarray) -> int:
        """
        Classify a single point.

        Hint: For uniform weights, use Counter for majority vote.
        For distance weights, weight each vote by 1/(distance + epsilon).
        Handle ties by picking the nearest neighbor's class.
        """
        raise NotImplementedError("TODO: implement this")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify multiple query points."""
        raise NotImplementedError("TODO: implement this")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy: fraction of correct predictions."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Cross-Validation
# =============================================================================

def cross_validate_k(
    X: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    n_folds: int = 5,
    weights: str = "uniform",
) -> dict[int, float]:
    """
    K-fold cross-validation to find optimal k.

    Hint: Split data into n_folds parts. For each fold, train on
    (n_folds - 1) parts and test on the remaining part. Average
    accuracy across folds. Remember to scale features inside each
    fold (fit scaler on train portion only).
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Main: Test your implementation
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Generate simple test data: 3 Gaussian clusters
    def make_data(n=200):
        rng = np.random.RandomState(42)
        c0 = rng.randn(n // 3, 2) * 0.8 + np.array([-3, -3])
        c1 = rng.randn(n // 3, 2) * 0.8 + np.array([3, 3])
        c2 = rng.randn(n // 3, 2) * 0.8 + np.array([-3, 3])
        X = np.vstack([c0, c1, c2])
        y = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n // 3))
        idx = rng.permutation(len(X))
        return X[idx], y[idx]

    X, y = make_data(300)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Testing distance metrics...")
    a, b = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    print(f"  Euclidean({a}, {b}) = {euclidean_distance(a, b)}")  # Should be 5.0
    print(f"  Manhattan({a}, {b}) = {manhattan_distance(a, b)}")  # Should be 7.0

    print("\nTesting StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  Train mean: {X_train_s.mean(axis=0).round(6)}")  # Should be ~[0, 0]
    print(f"  Train std:  {X_train_s.std(axis=0).round(6)}")    # Should be ~[1, 1]

    print("\nTesting KNN classifier...")
    knn = KNNClassifier(k=5)
    knn.fit(X_train_s, y_train)
    acc = knn.score(X_test_s, y_test)
    print(f"  Accuracy (k=5, uniform): {acc:.3f}")

    print("\nTesting distance-weighted KNN...")
    knn_w = KNNClassifier(k=5, weights="distance")
    knn_w.fit(X_train_s, y_train)
    acc_w = knn_w.score(X_test_s, y_test)
    print(f"  Accuracy (k=5, distance): {acc_w:.3f}")

    print("\nCross-validating k...")
    cv = cross_validate_k(X_train, y_train, [1, 3, 5, 7, 9])
    for k, a in cv.items():
        print(f"  k={k}: {a:.3f}")

    print("\nAll tests passed!" if acc > 0.8 else "\nAccuracy seems low — check your implementation.")
