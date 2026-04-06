"""
Day 005: K-Means Clustering from Scratch

Building on our supervised learning foundations (Days 001-004), we now tackle
unsupervised learning. K-Means discovers structure in unlabeled data by
iteratively partitioning points into K groups that minimize within-cluster
variance.

Key insight: K-Means is coordinate descent on the WCSS objective —
the assignment step optimizes over assignments (holding centroids fixed),
the update step optimizes over centroids (holding assignments fixed).
Each step is globally optimal for its subproblem, but the joint problem
is non-convex, hence we only find local minima.
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
        Maximum iterations per run. Acts as a safety valve — well-initialized
        K-Means on clean data typically converges in 10-30 iterations.
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
        self.k = k
        self.max_iters = max_iters
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(random_state)

        # Fitted attributes
        self.centroids: Optional[np.ndarray] = None  # (k, d)
        self.labels: Optional[np.ndarray] = None  # (n,)
        self.inertia: float = float("inf")  # best WCSS
        self.n_iters: int = 0  # iterations for best run

    def _kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ initialization: spread initial centroids across the data.

        The key idea is distance-weighted sampling. After choosing the first
        centroid randomly, each subsequent centroid is chosen with probability
        proportional to D(x)^2 — the squared distance to the nearest existing
        centroid. This biases selection toward distant points, ensuring good
        coverage of the data space.

        Theoretical guarantee: expected WCSS is O(log K) times optimal.
        This is remarkable — a simple O(n*K*d) initialization step gives a
        logarithmic approximation ratio.
        """
        n, d = X.shape
        centroids = np.empty((self.k, d))

        # Step 1: First centroid chosen uniformly at random
        idx = self.rng.integers(n)
        centroids[0] = X[idx]

        for i in range(1, self.k):
            # Compute squared distance from each point to nearest existing centroid.
            # We use squared distance because:
            # (a) it avoids a sqrt we'd just square again for the probability
            # (b) D(x)^2 weighting is what gives the O(log K) guarantee
            dists = np.min(
                np.sum((X[:, np.newaxis, :] - centroids[:i][np.newaxis, :, :]) ** 2, axis=2),
                axis=1,
            )

            # Convert distances to probability distribution.
            # Points far from all existing centroids get high probability.
            probs = dists / dists.sum()

            # Step 2: Sample next centroid according to D(x)^2 weighting
            idx = self.rng.choice(n, p=probs)
            centroids[i] = X[idx]

        return centroids

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid (Voronoi partition).

        This is the 'E-step' if you think of K-Means as a special case of EM.
        For each point, we compute distance to all K centroids and pick the
        minimum. This creates a Voronoi tessellation of the feature space —
        each cluster occupies a convex polytope (which is why K-Means struggles
        with non-convex clusters like spirals or rings).

        Complexity: O(n * K * d) — dominates the per-iteration cost.
        """
        # Vectorized distance computation using the expansion:
        # ||x - c||^2 = ||x||^2 - 2*x·c + ||c||^2
        # This avoids creating an (n, K, d) intermediate tensor.
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
        C_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, K)
        dists = X_sq - 2 * X @ centroids.T + C_sq  # (n, K)

        return np.argmin(dists, axis=1)

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, old_centroids: np.ndarray
    ) -> np.ndarray:
        """Recompute centroids as the mean of assigned points.

        This is the 'M-step' in the EM interpretation. The mean minimizes
        sum of squared distances — this is why it's the right update for
        Euclidean K-Means. (For K-Medians with L1 distance, you'd use the
        component-wise median instead.)

        Edge case: if a cluster has zero assigned points, we keep the old
        centroid. An alternative is to reinitialize it at the farthest point
        from all centroids, but in practice with K-Means++ initialization,
        empty clusters are rare.
        """
        centroids = np.empty_like(old_centroids)

        for i in range(self.k):
            mask = labels == i
            if mask.sum() > 0:
                centroids[i] = X[mask].mean(axis=0)
            else:
                # Empty cluster — keep previous position to avoid NaN.
                # With K-Means++ init, this almost never triggers.
                centroids[i] = old_centroids[i]

        return centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute WCSS (within-cluster sum of squares), aka inertia.

        J = Σ_i ||x_i - μ_{label_i}||^2

        This is THE objective function K-Means minimizes. Lower is better,
        but it always decreases with more clusters (trivially 0 when K=n),
        so it's only meaningful for comparing runs with the same K.
        """
        return float(np.sum((X - centroids[labels]) ** 2))

    def _fit_single(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Run one complete K-Means from initialization to convergence.

        Returns (centroids, labels, inertia, n_iterations).
        """
        centroids = self._kmeans_plus_plus(X)

        for iteration in range(1, self.max_iters + 1):
            # Assignment step: each point → nearest centroid
            labels = self._assign_clusters(X, centroids)

            # Update step: each centroid → mean of its points
            new_centroids = self._update_centroids(X, labels, centroids)

            # Convergence check: if centroids didn't move, we're done.
            # Using exact equality on assignments is more robust than
            # threshold on centroid movement (which requires tuning epsilon).
            if np.array_equal(self._assign_clusters(X, new_centroids), labels):
                centroids = new_centroids
                break

            centroids = new_centroids

        labels = self._assign_clusters(X, centroids)
        inertia = self._compute_inertia(X, labels, centroids)

        return centroids, labels, inertia, iteration

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-Means with multiple restarts, keeping the best result.

        Multiple restarts are critical because K-Means converges to LOCAL
        minima. Different initializations explore different regions of the
        (extremely non-convex) objective landscape. With K-Means++ init,
        3-5 restarts usually suffice; with random init, you might need 20+.
        """
        X = np.asarray(X, dtype=np.float64)
        best_inertia = float("inf")

        for restart in range(self.n_restarts):
            centroids, labels, inertia, n_iters = self._fit_single(X)

            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids = centroids
                self.labels = labels
                self.inertia = inertia
                self.n_iters = n_iters

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to nearest centroid.

        After fitting, the centroids define a Voronoi partition of the entire
        feature space. Any new point simply gets assigned to the nearest
        centroid — no retraining needed.
        """
        X = np.asarray(X, dtype=np.float64)
        return self._assign_clusters(X, self.centroids)


def elbow_analysis(X: np.ndarray, k_range: range, n_restarts: int = 5, seed: int = 42) -> list[float]:
    """Run K-Means for multiple K values and return WCSS for each.

    The 'elbow' in the WCSS vs K plot suggests the natural number of clusters.
    The intuition: going from K to K+1, WCSS always decreases, but the rate
    of decrease levels off once you have 'enough' clusters. The elbow is
    where marginal benefit of adding a cluster drops sharply.

    More formally, we're looking for the K that maximizes the second
    derivative (discrete curvature) of the WCSS curve.
    """
    inertias = []
    for k in k_range:
        km = KMeans(k=k, n_restarts=n_restarts, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            km.fit(X)
        inertias.append(km.inertia)
    return inertias


def find_elbow(k_range: range, inertias: list[float]) -> int:
    """Find the elbow point using the maximum second derivative.

    The second derivative of the WCSS curve measures how quickly the rate
    of improvement is decelerating. The maximum point is where adding
    another cluster transitions from 'significant improvement' to
    'diminishing returns'.
    """
    ks = list(k_range)
    if len(ks) < 3:
        return ks[0]

    # Second derivative: f''(k) ≈ f(k+1) - 2f(k) + f(k-1)
    second_deriv = []
    for i in range(1, len(inertias) - 1):
        d2 = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
        second_deriv.append(d2)

    # The elbow is at the maximum second derivative
    best_idx = np.argmax(second_deriv) + 1  # +1 because we started at index 1
    return ks[best_idx]


# ─────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # ── Generate synthetic data: 3 well-separated Gaussian clusters ──
    rng = np.random.default_rng(42)

    # True cluster centers and sizes
    true_centers = np.array([[2.0, 2.0], [8.0, 3.0], [5.0, 8.0]])
    cluster_sizes = [100, 150, 120]
    true_labels = np.concatenate([np.full(s, i) for i, s in enumerate(cluster_sizes)])

    # Generate points around each center
    X = np.vstack([
        rng.normal(loc=center, scale=0.8, size=(size, 2))
        for center, size in zip(true_centers, cluster_sizes)
    ])

    print("=" * 65)
    print("K-MEANS CLUSTERING FROM SCRATCH")
    print("=" * 65)
    print(f"\nDataset: {X.shape[0]} points in {X.shape[1]}D, 3 true clusters")
    print(f"True centers:\n{true_centers}")

    # ── Fit K-Means ──────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("FITTING K-MEANS (K=3, 5 restarts)")
    print("─" * 65)

    km = KMeans(k=3, n_restarts=5, random_state=0)
    km.fit(X)

    print(f"\nConverged in {km.n_iters} iterations")
    print(f"Final WCSS (inertia): {km.inertia:.2f}")
    print(f"\nLearned centroids:")
    for i, c in enumerate(km.centroids):
        count = np.sum(km.labels == i)
        print(f"  Cluster {i}: center = {c}, size = {count}")

    # ── Compare to true clusters ─────────────────────────────────────
    # K-Means labels are arbitrary (cluster 0 might correspond to true cluster 2)
    # so we find the best permutation mapping
    print(f"\nTrue centers (for comparison):")
    for i, c in enumerate(true_centers):
        print(f"  True cluster {i}: center = {c}, size = {cluster_sizes[i]}")

    # Compute centroid-to-true-center distances to find mapping
    from itertools import permutations

    best_accuracy = 0
    for perm in permutations(range(3)):
        remapped = np.array([perm[l] for l in km.labels])
        acc = np.mean(remapped == true_labels)
        best_accuracy = max(best_accuracy, acc)

    print(f"\nClustering accuracy (best label mapping): {best_accuracy:.1%}")

    # ── Elbow Analysis ───────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("ELBOW ANALYSIS: finding optimal K")
    print("─" * 65)

    k_range = range(1, 9)
    inertias = elbow_analysis(X, k_range, n_restarts=3, seed=42)

    print(f"\n{'K':>3} | {'WCSS':>10} | {'ΔWCSS':>10} | Plot")
    print("─" * 55)
    for i, (k, inertia) in enumerate(zip(k_range, inertias)):
        delta = f"{inertias[i-1] - inertia:>10.1f}" if i > 0 else f"{'—':>10}"
        bar = "█" * int(inertia / max(inertias) * 30)
        print(f"{k:>3} | {inertia:>10.1f} | {delta} | {bar}")

    suggested_k = find_elbow(k_range, inertias)
    print(f"\nElbow detected at K = {suggested_k}")

    # ── Predict new points ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("PREDICTING NEW POINTS")
    print("─" * 65)

    new_points = np.array([[2.5, 2.5], [7.5, 3.5], [5.0, 7.0], [0.0, 0.0]])
    predictions = km.predict(new_points)

    print(f"\nUsing fitted centroids to classify new points:")
    for point, label in zip(new_points, predictions):
        nearest_centroid = km.centroids[label]
        dist = np.sqrt(np.sum((point - nearest_centroid) ** 2))
        print(f"  {point} → Cluster {label} (distance to centroid: {dist:.2f})")

    # ── Convergence demonstration ────────────────────────────────────
    print("\n" + "─" * 65)
    print("CONVERGENCE BEHAVIOR (single run, verbose)")
    print("─" * 65)

    # Manual step-through to show iteration progress
    X_small = X[:50]  # smaller subset for clarity
    centroids = KMeans(k=3, random_state=7)._kmeans_plus_plus(X_small)
    print(f"\nInitial centroids (K-Means++):")
    for i, c in enumerate(centroids):
        print(f"  μ_{i} = {c}")

    prev_labels = None
    for step in range(1, 20):
        # Assignment
        X_sq = np.sum(X_small ** 2, axis=1, keepdims=True)
        C_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
        dists = X_sq - 2 * X_small @ centroids.T + C_sq
        labels = np.argmin(dists, axis=1)

        # Inertia
        inertia = float(np.sum((X_small - centroids[labels]) ** 2))

        # Update
        new_centroids = np.array([
            X_small[labels == i].mean(axis=0) if (labels == i).sum() > 0 else centroids[i]
            for i in range(3)
        ])

        changed = 0 if prev_labels is None else np.sum(labels != prev_labels)
        print(f"  Iter {step:>2}: WCSS = {inertia:>8.2f}, points reassigned = {changed}")

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            print(f"  ✓ Converged after {step} iterations!")
            break

        centroids = new_centroids
        prev_labels = labels.copy()

    print("\n" + "=" * 65)
    print("K-Means finds structure in unlabeled data by iterating two steps:")
    print("  1. Assign points to nearest centroid (Voronoi partition)")
    print("  2. Move centroids to the mean of their points")
    print("Each step reduces WCSS — convergence is guaranteed.")
    print("K-Means++ initialization + restarts → robust results.")
    print("=" * 65)
