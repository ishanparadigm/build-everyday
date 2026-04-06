"""
Day 005: K-Means Clustering from Scratch

Building on our supervised learning foundations (Days 001-004), we now tackle
unsupervised learning. K-Means discovers structure in unlabeled data by
iteratively partitioning points into K groups that minimize within-cluster
variance.

Key insight: K-Means is coordinate descent on the WCSS objective —
the assignment step optimizes over assignments (holding centroids fixed),
the update step optimizes over centroids (holding assignments fixed).
"""

import warnings
import numpy as np
from typing import Optional


class KMeans:
    """K-Means clustering with K-Means++ initialization.

    Parameters
    ----------
    k : int
        Number of clusters.
    max_iters : int
        Maximum iterations per run.
    n_restarts : int
        Number of independent runs with different initializations.
        We keep the result with lowest WCSS (inertia).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 3,
        max_iters: int = 300,
        n_restarts: int = 5,
        random_state: Optional[int] = None,
    ):
        # Hint: store parameters and initialize fitted attributes
        # (centroids, labels, inertia, n_iters).
        # Use np.random.default_rng(random_state) for reproducible randomness.
        raise NotImplementedError("TODO: implement this")

    def _kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ initialization: spread initial centroids across the data.

        1. Choose the first centroid uniformly at random from the data points.
        2. For each subsequent centroid, choose a data point with probability
           proportional to D(x)^2 — squared distance to nearest existing centroid.

        Returns an array of shape (k, d) with the initial centroids.
        """
        # Hint: use self.rng.integers(n) and self.rng.choice(n, p=probs)
        # For distance computation, broadcast X against existing centroids
        # and take the min squared distance per point.
        raise NotImplementedError("TODO: implement this")

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid (Voronoi partition).

        Returns an array of shape (n,) with cluster indices.
        """
        # Hint: use the expansion ||x - c||^2 = ||x||^2 - 2*x.c + ||c||^2
        # to avoid creating an (n, K, d) intermediate tensor.
        # Then np.argmin along the centroid axis.
        raise NotImplementedError("TODO: implement this")

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, old_centroids: np.ndarray
    ) -> np.ndarray:
        """Recompute centroids as the mean of assigned points.

        If a cluster has zero assigned points, keep the old centroid.
        Returns an array of shape (k, d).
        """
        # Hint: for each cluster i, compute X[labels == i].mean(axis=0).
        # Handle the empty-cluster edge case.
        raise NotImplementedError("TODO: implement this")

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute WCSS (within-cluster sum of squares), aka inertia.

        J = Σ_i ||x_i - μ_{label_i}||^2

        Lower is better, but only meaningful for comparing runs with the same K.
        """
        # Hint: centroids[labels] gives each point's centroid; then sum squared diffs.
        raise NotImplementedError("TODO: implement this")

    def _fit_single(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Run one complete K-Means from initialization to convergence.

        Returns (centroids, labels, inertia, n_iterations).
        """
        # Hint: initialize with _kmeans_plus_plus, then loop:
        #   1. _assign_clusters
        #   2. _update_centroids
        #   3. Check convergence (assignments didn't change)
        raise NotImplementedError("TODO: implement this")

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-Means with multiple restarts, keeping the best result.

        Multiple restarts are critical because K-Means converges to local minima.
        """
        # Hint: convert X to float64, run _fit_single n_restarts times,
        # keep the result with lowest inertia.
        raise NotImplementedError("TODO: implement this")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to nearest centroid.

        Returns an array of shape (n,) with cluster indices.
        """
        # Hint: convert X to float64, then call _assign_clusters with self.centroids
        raise NotImplementedError("TODO: implement this")


def elbow_analysis(X: np.ndarray, k_range: range, n_restarts: int = 5, seed: int = 42) -> list[float]:
    """Run K-Means for multiple K values and return WCSS (inertia) for each.

    The 'elbow' in the WCSS vs K plot suggests the natural number of clusters.
    """
    # Hint: for each k in k_range, fit a KMeans and collect km.inertia
    raise NotImplementedError("TODO: implement this")


def find_elbow(k_range: range, inertias: list[float]) -> int:
    """Find the elbow point using the maximum second derivative.

    Second derivative: f''(k) ≈ f(k+1) - 2f(k) + f(k-1)
    The elbow is at the k with maximum second derivative.
    """
    # Hint: compute the discrete second derivative for each interior point,
    # then return the k with the largest value. Use np.argmax.
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Main: test your implementation as you build it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # Generate synthetic data: 3 well-separated Gaussian clusters
    rng = np.random.default_rng(42)
    true_centers = np.array([[2.0, 2.0], [8.0, 3.0], [5.0, 8.0]])
    cluster_sizes = [100, 150, 120]

    X = np.vstack([
        rng.normal(loc=center, scale=0.8, size=(size, 2))
        for center, size in zip(true_centers, cluster_sizes)
    ])
    true_labels = np.concatenate([np.full(s, i) for i, s in enumerate(cluster_sizes)])

    print("=" * 65)
    print("K-MEANS CLUSTERING — YOUR IMPLEMENTATION")
    print("=" * 65)
    print(f"\nDataset: {X.shape[0]} points in {X.shape[1]}D, 3 true clusters")

    # Fit KMeans
    print("\n--- Fitting K-Means (K=3) ---")
    km = KMeans(k=3, n_restarts=5, random_state=0)
    km.fit(X)
    print(f"Converged in {km.n_iters} iterations")
    print(f"Inertia (WCSS): {km.inertia:.2f}")
    print(f"Centroids:\n{km.centroids}")

    # Check clustering accuracy (with label alignment)
    from itertools import permutations
    best_accuracy = 0
    for perm in permutations(range(3)):
        remapped = np.array([perm[l] for l in km.labels])
        acc = np.mean(remapped == true_labels)
        best_accuracy = max(best_accuracy, acc)
    print(f"\nClustering accuracy (best label mapping): {best_accuracy:.1%}")

    # Predict new points
    print("\n--- Predicting new points ---")
    new_points = np.array([[2.5, 2.5], [7.5, 3.5], [5.0, 7.0]])
    predictions = km.predict(new_points)
    for point, label in zip(new_points, predictions):
        print(f"  {point} -> Cluster {label}")

    # Elbow analysis
    print("\n--- Elbow analysis ---")
    k_range = range(1, 9)
    inertias = elbow_analysis(X, k_range, n_restarts=3, seed=42)
    for k, inertia in zip(k_range, inertias):
        print(f"  K={k}: WCSS={inertia:.1f}")

    suggested_k = find_elbow(k_range, inertias)
    print(f"\nElbow detected at K = {suggested_k}")
