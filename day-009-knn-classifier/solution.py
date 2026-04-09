"""
Day 009: K-Nearest Neighbors (KNN) Classifier from Scratch

A complete implementation of KNN with multiple distance metrics,
weighted voting, feature scaling, and cross-validation for k selection.

Key insight: KNN is a "lazy learner" — it does no work at training time
and all the work at prediction time. This is the opposite of most ML
algorithms. Understanding when this tradeoff makes sense (small datasets,
complex decision boundaries, few predictions needed) is essential.
"""

import numpy as np
from collections import Counter
from typing import Literal


# =============================================================================
# Distance Metrics
# =============================================================================

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    L2 norm: straight-line distance in Euclidean space.

    We use np.sqrt(np.sum(...)) rather than np.linalg.norm for clarity,
    though in production you'd use the optimized library function.

    Complexity: O(d) where d = number of features.
    """
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    L1 norm: sum of absolute differences along each axis.

    More robust to outliers than Euclidean because it doesn't square
    the differences — a single large deviation in one feature doesn't
    dominate the total distance.
    """
    return np.sum(np.abs(x - y))


def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float = 2) -> float:
    """
    Generalized distance: p=1 gives Manhattan, p=2 gives Euclidean.

    As p -> infinity, this approaches the Chebyshev distance
    (max absolute difference along any single dimension).
    """
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


# =============================================================================
# Feature Scaling
# =============================================================================

class StandardScaler:
    """
    Z-score normalization: transform features to zero mean, unit variance.

    Critical for KNN because distance metrics are sensitive to feature scale.
    A feature ranging 0-1000 would completely dominate one ranging 0-1.

    We store mean_ and std_ from fit() and apply them in transform().
    This separation is essential: in production, you fit on training data
    and transform both training and test data with the SAME parameters.
    Fitting on test data would be information leakage.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and std from training data."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Replace zero std with 1 to avoid division by zero
        # (constant features get scaled to 0, which is correct)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned scaling parameters."""
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit and transform in one call."""
        return self.fit(X).transform(X)


# =============================================================================
# KNN Classifier
# =============================================================================

class KNNClassifier:
    """
    K-Nearest Neighbors classifier.

    Architecture decision: we store the entire training set and compute
    distances at prediction time. This is the defining characteristic of
    a "lazy learner" — no model is built during training.

    Tradeoff:
    - Training: O(1) — just store the data
    - Prediction: O(n * d) per query — compute distance to every training point
    - Memory: O(n * d) — must keep entire training set in memory

    For large n, this becomes impractical. Production systems use KD-trees
    (O(log n) average case) or approximate methods (FAISS, Annoy).
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
        self.p = p  # Only used for Minkowski

        # Map metric name to function
        # Using a dict lookup instead of if/elif for cleaner extension
        self._metric_fn = {
            "euclidean": euclidean_distance,
            "manhattan": manhattan_distance,
            "minkowski": lambda x, y: minkowski_distance(x, y, p=self.p),
        }[metric]

        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        'Training' for KNN = storing the data. That's it.

        We do validate shapes here — catching shape mismatches early
        saves debugging time later when distances look wrong.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]} labels"
            )
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Compute distance from a single query point to all training points.

        Returns an array of shape (n_train,) with distances.

        Performance note: for Euclidean distance specifically, you could
        vectorize this as np.sqrt(np.sum((X_train - x)**2, axis=1)),
        which is much faster. We use the loop for clarity and to support
        arbitrary distance functions.
        """
        distances = np.array([
            self._metric_fn(x, x_train) for x_train in self.X_train
        ])
        return distances

    def _get_k_nearest(self, distances: np.ndarray) -> np.ndarray:
        """
        Find indices of k nearest neighbors.

        np.argpartition is O(n) average case — much better than full sort
        O(n log n) when k << n. It places the k smallest elements in the
        first k positions (unordered), then we sort just those k elements.

        Why not just np.argsort? For n=1M and k=5, argpartition + small sort
        is ~20x faster than full argsort.
        """
        # Handle case where k >= n_train
        k = min(self.k, len(distances))

        # argpartition: O(n) to get k smallest in arbitrary order
        partitioned_indices = np.argpartition(distances, k)[:k]

        # Sort just the k neighbors by distance (for tie-breaking and display)
        sorted_within_k = np.argsort(distances[partitioned_indices])

        return partitioned_indices[sorted_within_k]

    def _predict_single(self, x: np.ndarray) -> int:
        """
        Classify a single query point.

        Process:
        1. Compute distances to all training points
        2. Find k nearest neighbors
        3. Vote (uniform or distance-weighted)
        4. Return majority class
        """
        distances = self._compute_distances(x)
        neighbor_indices = self._get_k_nearest(distances)
        neighbor_labels = self.y_train[neighbor_indices]
        neighbor_distances = distances[neighbor_indices]

        if self.weights == "uniform":
            # Simple majority vote
            # Counter.most_common(1) returns [(label, count)]
            vote_counts = Counter(neighbor_labels)
            max_count = max(vote_counts.values())
            tied_classes = [cls for cls, cnt in vote_counts.items() if cnt == max_count]

            if len(tied_classes) == 1:
                return tied_classes[0]

            # Tie-breaking: pick the class of the single nearest neighbor
            # This is more principled than random because closer = more similar
            return neighbor_labels[0]

        else:  # distance-weighted
            # Weight = 1 / (distance + epsilon)
            # epsilon prevents division by zero for exact matches
            epsilon = 1e-10
            weights = 1.0 / (neighbor_distances + epsilon)

            # Sum weights per class
            classes = np.unique(neighbor_labels)
            weighted_votes = {}
            for cls in classes:
                mask = neighbor_labels == cls
                weighted_votes[cls] = np.sum(weights[mask])

            return max(weighted_votes, key=weighted_votes.get)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify multiple query points."""
        return np.array([self._predict_single(x) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy: fraction of correct predictions."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# =============================================================================
# Cross-Validation for k Selection
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

    For each candidate k value, we:
    1. Split data into n_folds equal parts
    2. Train on (n_folds - 1) parts, test on the held-out part
    3. Rotate which part is held out
    4. Average accuracy across all folds

    This gives an unbiased estimate of how well each k value generalizes,
    unlike training accuracy which always favors k=1.

    Why not just use a train/test split? With small datasets, a single
    split is high-variance — you might get lucky or unlucky with which
    points end up in test. Cross-validation averages over multiple splits.
    """
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)

    # Create fold assignments
    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[:n % n_folds] += 1  # Distribute remainder

    results = {}

    for k in k_values:
        fold_accuracies = []
        current = 0

        for fold_size in fold_sizes:
            # Split indices into test (current fold) and train (everything else)
            test_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
            current += fold_size

            # Scale features (fit on train only!)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X[train_idx])
            X_test_scaled = scaler.transform(X[test_idx])

            # Train and evaluate
            knn = KNNClassifier(k=k, weights=weights)
            knn.fit(X_train_scaled, y[train_idx])
            acc = knn.score(X_test_scaled, y[test_idx])
            fold_accuracies.append(acc)

        results[k] = np.mean(fold_accuracies)

    return results


# =============================================================================
# Dataset Generation
# =============================================================================

def make_classification_data(
    n_samples: int = 200,
    n_features: int = 2,
    n_classes: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic classification dataset with Gaussian clusters.

    Each class is a blob centered at a random point with some spread.
    This gives us a dataset where KNN should work well (classes are
    spatially coherent) but with overlapping boundaries that test
    the algorithm's nuance.
    """
    rng = np.random.RandomState(random_state)

    samples_per_class = n_samples // n_classes
    X_parts = []
    y_parts = []

    # Generate cluster centers spread out in feature space
    centers = rng.uniform(-5, 5, size=(n_classes, n_features))

    for i in range(n_classes):
        X_class = rng.randn(samples_per_class, n_features) * 1.2 + centers[i]
        X_parts.append(X_class)
        y_parts.append(np.full(samples_per_class, i))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle the data
    shuffle_idx = rng.permutation(len(X))
    return X[shuffle_idx], y[shuffle_idx]


def demonstrate_curse_of_dimensionality(
    dims: list[int] = [2, 5, 10, 20, 50, 100],
    n_samples: int = 200,
    random_state: int = 42,
) -> dict[int, float]:
    """
    Empirically show how KNN degrades in high dimensions.

    We measure the ratio of nearest to farthest neighbor distance.
    In low dimensions, this ratio is small (nearest is much closer than farthest).
    In high dimensions, this ratio approaches 1 (all points are equidistant).

    When this ratio is ~1, the concept of "nearest neighbor" becomes meaningless —
    the nearest neighbor is barely closer than the farthest point.
    """
    rng = np.random.RandomState(random_state)
    ratios = {}

    for d in dims:
        # Generate random points in d dimensions
        X = rng.randn(n_samples, d)

        # Pick a query point
        query = rng.randn(d)

        # Compute all distances
        distances = np.sqrt(np.sum((X - query) ** 2, axis=1))

        # Ratio of nearest to farthest distance
        ratio = np.min(distances) / np.max(distances)
        ratios[d] = ratio

    return ratios


# =============================================================================
# Main: Demonstration
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("K-NEAREST NEIGHBORS FROM SCRATCH")
    print("=" * 70)

    # --- 1. Generate dataset ---
    print("\n1. GENERATING DATASET")
    print("-" * 40)
    X, y = make_classification_data(n_samples=300, n_features=2, n_classes=3)
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Classes: {np.unique(y)} (counts: {np.bincount(y)})")
    print(f"   Feature ranges: [{X.min():.2f}, {X.max():.2f}]")

    # --- 2. Train/test split ---
    print("\n2. TRAIN/TEST SPLIT")
    print("-" * 40)
    split = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # --- 3. Feature scaling ---
    print("\n3. FEATURE SCALING")
    print("-" * 40)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   Train mean after scaling: {X_train_scaled.mean(axis=0).round(6)}")
    print(f"   Train std after scaling:  {X_train_scaled.std(axis=0).round(6)}")
    print(f"   Test mean (using train params): {X_test_scaled.mean(axis=0).round(2)}")

    # --- 4. Basic KNN with different k values ---
    print("\n4. KNN WITH DIFFERENT k VALUES")
    print("-" * 40)
    for k in [1, 3, 5, 9, 15, 25]:
        knn = KNNClassifier(k=k, weights="uniform")
        knn.fit(X_train_scaled, y_train)
        train_acc = knn.score(X_train_scaled, y_train)
        test_acc = knn.score(X_test_scaled, y_test)
        print(f"   k={k:2d}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

    # --- 5. Uniform vs distance-weighted ---
    print("\n5. UNIFORM vs DISTANCE-WEIGHTED VOTING")
    print("-" * 40)
    for weights in ["uniform", "distance"]:
        knn = KNNClassifier(k=5, weights=weights)
        knn.fit(X_train_scaled, y_train)
        test_acc = knn.score(X_test_scaled, y_test)
        print(f"   weights={weights:10s}: test_acc={test_acc:.3f}")

    # --- 6. Different distance metrics ---
    print("\n6. DISTANCE METRICS COMPARISON")
    print("-" * 40)
    for metric in ["euclidean", "manhattan"]:
        knn = KNNClassifier(k=5, metric=metric)
        knn.fit(X_train_scaled, y_train)
        test_acc = knn.score(X_test_scaled, y_test)
        print(f"   metric={metric:12s}: test_acc={test_acc:.3f}")

    # Minkowski with different p values
    for p in [1, 2, 3]:
        knn = KNNClassifier(k=5, metric="minkowski", p=p)
        knn.fit(X_train_scaled, y_train)
        test_acc = knn.score(X_test_scaled, y_test)
        print(f"   minkowski(p={p}):      test_acc={test_acc:.3f}")

    # --- 7. Cross-validation for k selection ---
    print("\n7. CROSS-VALIDATION FOR OPTIMAL k")
    print("-" * 40)
    k_values = list(range(1, 22, 2))  # Odd values only to avoid ties
    cv_results = cross_validate_k(X_train, y_train, k_values, n_folds=5)

    best_k = max(cv_results, key=cv_results.get)
    print(f"   {'k':>4s} | {'CV Accuracy':>12s}")
    print(f"   {'----':>4s} | {'------------':>12s}")
    for k, acc in sorted(cv_results.items()):
        marker = " <-- best" if k == best_k else ""
        print(f"   {k:4d} | {acc:12.3f}{marker}")

    print(f"\n   Best k = {best_k} (CV accuracy = {cv_results[best_k]:.3f})")

    # Evaluate best k on test set
    knn_best = KNNClassifier(k=best_k, weights="distance")
    knn_best.fit(X_train_scaled, y_train)
    final_acc = knn_best.score(X_test_scaled, y_test)
    print(f"   Test accuracy with best k: {final_acc:.3f}")

    # --- 8. Curse of dimensionality ---
    print("\n8. CURSE OF DIMENSIONALITY")
    print("-" * 40)
    print("   As dimensions increase, nearest/farthest distance ratio -> 1")
    print("   (meaning 'nearest neighbor' becomes meaningless)\n")

    ratios = demonstrate_curse_of_dimensionality()
    print(f"   {'Dims':>6s} | {'Min/Max Distance Ratio':>22s} | {'Visualization':>20s}")
    print(f"   {'------':>6s} | {'----------------------':>22s} | {'--------------------':>20s}")
    for d, ratio in ratios.items():
        bar = "#" * int(ratio * 30)
        print(f"   {d:6d} | {ratio:22.4f} | {bar}")

    # --- 9. Detailed prediction example ---
    print("\n9. DETAILED PREDICTION WALKTHROUGH")
    print("-" * 40)
    query = X_test_scaled[0]
    true_label = y_test[0]

    knn_demo = KNNClassifier(k=5, weights="distance")
    knn_demo.fit(X_train_scaled, y_train)

    # Manually show the process
    distances = knn_demo._compute_distances(query)
    neighbor_idx = knn_demo._get_k_nearest(distances)

    print(f"   Query point: [{query[0]:.3f}, {query[1]:.3f}]")
    print(f"   True label: {true_label}")
    print(f"\n   5 nearest neighbors:")
    for i, idx in enumerate(neighbor_idx):
        print(f"     #{i+1}: point=[{X_train_scaled[idx][0]:.3f}, {X_train_scaled[idx][1]:.3f}], "
              f"label={y_train[idx]}, distance={distances[idx]:.4f}")

    prediction = knn_demo._predict_single(query)
    print(f"\n   Prediction: {prediction} ({'CORRECT' if prediction == true_label else 'WRONG'})")

    print("\n" + "=" * 70)
    print("DONE. KNN is simple but powerful — the key is understanding when")
    print("it works (low-d, sufficient data) and when it doesn't (high-d, sparse).")
    print("=" * 70)
