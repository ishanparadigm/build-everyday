# Day 005: K-Means Clustering

## Overview

K-Means is the foundational unsupervised learning algorithm — given a set of unlabeled data points, it discovers natural groupings (clusters) by iteratively assigning points to the nearest centroid and updating centroids to the mean of their assigned points. Unlike the supervised algorithms we've built in previous days (linear/logistic regression, decision trees), K-Means receives *no labels*. It must find structure purely from the geometry of the data.

**Real-world applications:** Customer segmentation, image compression (color quantization), anomaly detection, document clustering, feature engineering for downstream supervised models, and vector quantization in codebooks for retrieval systems.

## Core Concepts

### The Objective: Minimizing Within-Cluster Sum of Squares (WCSS)

K-Means minimizes the total squared distance from each point to its assigned centroid:

```
J = Σ_{i=1}^{K} Σ_{x ∈ C_i} ||x - μ_i||²
```

Where:
- `K` is the number of clusters
- `C_i` is the set of points assigned to cluster `i`
- `μ_i` is the centroid (mean) of cluster `i`
- `||x - μ_i||²` is the squared Euclidean distance

This objective is **NP-hard** to minimize globally, but the iterative algorithm finds a local minimum efficiently.

### Why Squared Distance?

Squared Euclidean distance has a special property: the point that minimizes the sum of squared distances to a set of points is exactly the **arithmetic mean**. This is why the "update centroids to the mean" step is mathematically justified — it's the optimal centroid given fixed assignments. If we used absolute distance (L1 norm), the optimal center would be the **median**, leading to K-Medians. If we required centroids to be actual data points, we'd get K-Medoids.

### Lloyd's Algorithm (The Standard K-Means)

1. **Initialize** K centroids (randomly or via K-Means++)
2. **Assignment step:** Assign each point to the nearest centroid
3. **Update step:** Recompute each centroid as the mean of its assigned points
4. **Repeat** steps 2-3 until convergence (assignments don't change) or max iterations reached

**Convergence guarantee:** Each step monotonically decreases (or maintains) J. Since there are finitely many possible assignments, the algorithm must converge. However, it may converge to a local minimum — the result depends heavily on initialization.

### The Initialization Problem and K-Means++

Random initialization can lead to terrible results. Consider placing two initial centroids in the same dense cluster — one cluster gets split, another goes unrepresented.

**K-Means++** (Arthur & Vassilvitskii, 2007) fixes this with a principled initialization:

1. Choose the first centroid uniformly at random from the data
2. For each remaining centroid, choose a data point with probability proportional to D(x)², where D(x) is the distance to the nearest existing centroid
3. Repeat until all K centroids are chosen

**Intuition:** Points far from existing centroids are more likely to be chosen, spreading centroids across the data. This gives an O(log K) approximation guarantee on the optimal WCSS — a remarkable theoretical result for such a simple modification.

### The Elbow Method: Choosing K

K-Means requires you to specify K upfront. The **elbow method** runs K-Means for K = 1, 2, ..., K_max and plots WCSS vs K. The "elbow" — where adding another cluster gives diminishing returns — suggests the natural number of clusters.

Mathematically, we're looking for the K where the second derivative of the WCSS curve is maximized (the point of maximum curvature).

## Step-by-Step Breakdown

### Step 1: K-Means++ Initialization
Choose initial centroids that are well-spread across the data. Without this, we might need many random restarts to find a good solution. The distance-weighted sampling ensures centroids "cover" the data space.

### Step 2: Assignment Step
For each data point, compute its distance to all K centroids and assign it to the nearest one. This is the most expensive step: O(n·K·d) where n = points, K = clusters, d = dimensions. The assignment creates a Voronoi partition of the space.

### Step 3: Update Step
Recompute each centroid as the mean of its assigned points. If a cluster becomes empty (possible with bad initialization), we either reassign it to the farthest point or leave it — our implementation handles this. Cost: O(n·d).

### Step 4: Convergence Check
Compare current assignments to previous. If identical, we've converged. We also set a max iteration limit as a safety valve — pathological cases can oscillate for many iterations near the boundary between clusters.

### Step 5: Multiple Restarts
Since K-Means finds *local* minima, running it multiple times with different initializations and keeping the best result (lowest WCSS) significantly improves solution quality. K-Means++ usually makes 3-5 restarts sufficient.

### Step 6: Elbow Analysis
Run the full algorithm for multiple values of K and analyze the WCSS curve to recommend the optimal number of clusters.

## Learning Objectives

- Implement K-Means with K-Means++ initialization from scratch
- Understand unsupervised learning vs. supervised (Days 001-004)
- Learn why initialization matters and how K-Means++ provides theoretical guarantees
- Implement the elbow method for model selection
- Analyze convergence behavior and computational complexity
- Handle edge cases: empty clusters, single-point clusters, convergence criteria

## Going Deeper

- **Mini-batch K-Means:** For large datasets, update centroids using random subsets per iteration — trades accuracy for 10-100x speed
- **K-Medoids (PAM):** Restricts centroids to actual data points — more robust to outliers but O(n²)
- **Soft K-Means (EM for Gaussians):** Instead of hard assignments, compute probability of membership — this generalizes to Gaussian Mixture Models
- **Spectral Clustering:** For non-convex clusters, project data into eigenspace of the similarity graph's Laplacian, then run K-Means — handles rings, spirals, etc.
- **Connection to PCA:** K-Means can be viewed as a matrix factorization problem; the cluster assignments approximate a low-rank decomposition of the data matrix
- **Vector quantization in ML:** K-Means is used in product quantization for approximate nearest neighbor search (e.g., FAISS), directly connecting to the embeddings/RAG work in later weeks
