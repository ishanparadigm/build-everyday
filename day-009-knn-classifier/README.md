# Day 009: K-Nearest Neighbors from Scratch

## Overview

Build a complete K-Nearest Neighbors (KNN) classifier from scratch — no scikit-learn, no shortcuts. KNN is the simplest "non-parametric" classifier: instead of learning a model, it memorizes the training data and classifies new points by majority vote among their closest neighbors. Despite its simplicity, KNN is used in production for recommendation systems, anomaly detection, and image recognition. Understanding it deeply reveals fundamental tradeoffs in machine learning: bias vs. variance, computational cost vs. accuracy, and the curse of dimensionality.

## Core Concepts

### Distance Metrics — How Do We Define "Close"?

The entire KNN algorithm rests on one question: how do you measure distance between two data points?

**Euclidean distance** (L2 norm) is the straight-line distance:

```
d(x, y) = sqrt(sum((x_i - y_i)^2))
```

This is the default choice, but it has a subtle problem: it treats all features equally. If one feature ranges from 0-1000 and another from 0-1, the first feature dominates the distance calculation. This is why **feature scaling** is critical for KNN (unlike tree-based methods which are scale-invariant).

**Manhattan distance** (L1 norm) sums absolute differences:

```
d(x, y) = sum(|x_i - y_i|)
```

Manhattan distance is more robust to outliers because it doesn't square the differences. In high dimensions, L1 and L2 distances converge (part of the curse of dimensionality), so the choice matters less.

**Minkowski distance** generalizes both:

```
d(x, y) = (sum(|x_i - y_i|^p))^(1/p)
```

Where p=1 gives Manhattan, p=2 gives Euclidean. As p approaches infinity, you get the Chebyshev distance (max absolute difference along any dimension).

### The k Hyperparameter — Bias-Variance Tradeoff in Action

Choosing k is a perfect illustration of the bias-variance tradeoff:

- **k=1**: Zero training error (each point is its own nearest neighbor), but extremely sensitive to noise. A single mislabeled point creates a wrong prediction region. High variance, low bias.
- **k=N** (all points): Every prediction is the majority class of the entire dataset. Zero variance, maximum bias. Useless as a classifier.
- **Sweet spot**: Typically k = sqrt(N) is a reasonable starting point, but cross-validation is the real answer.

**Why use odd k for binary classification?** With even k, you can get ties (3 votes class A, 3 votes class B). Odd k guarantees a majority winner in binary problems.

### The Curse of Dimensionality — Why KNN Breaks in High Dimensions

This is the most important concept to internalize. In high dimensions:

1. **All points become equidistant.** The ratio of the nearest to farthest neighbor distance approaches 1 as dimensions grow. When everything is "equally far," nearest neighbor is meaningless.

2. **Volume concentrates at the surface.** In a d-dimensional unit hypercube, the volume within distance epsilon of the surface is 1 - (1-2*epsilon)^d. For d=100 and epsilon=0.01, that's 87% of the volume. Data points cluster near the edges.

3. **You need exponentially more data.** To maintain the same density of training points, you need N^d samples as dimensionality grows. This is why dimensionality reduction (PCA, feature selection) is essential preprocessing for KNN.

### Weighted Voting — Distance Should Matter

Basic KNN treats all k neighbors equally. But a neighbor at distance 0.1 should matter more than one at distance 10. **Distance-weighted KNN** assigns each neighbor a weight inversely proportional to its distance:

```
weight_i = 1 / (distance_i + epsilon)
```

The epsilon prevents division by zero when a test point exactly matches a training point. This simple change often improves accuracy significantly, especially when the decision boundary is complex.

## Step-by-Step Breakdown

### Step 1: Distance Computation

Compute distances from the query point to all training points. This is O(n*d) for n training points in d dimensions — the main bottleneck of KNN.

**Why this matters:** Unlike parametric models (logistic regression, neural nets) where prediction is fast and training is slow, KNN has zero training time but O(n*d) prediction time. This makes it impractical for large datasets without acceleration structures (KD-trees, ball trees).

### Step 2: Neighbor Selection

Find the k smallest distances. A naive sort is O(n log n), but we only need the k smallest — a partial sort or max-heap gives us O(n log k), which matters when k << n.

**What would go wrong without this optimization?** For k=5 and n=1,000,000, sorting takes ~20x longer than a heap-based approach.

### Step 3: Majority Vote (or Weighted Vote)

Count class labels among the k neighbors. For weighted voting, sum the weights per class instead of counting.

**Tie-breaking strategy:** When two classes have equal votes, we use the class of the single nearest neighbor. This is more principled than random tie-breaking because the closest point carries the most information.

### Step 4: Feature Scaling

Standardize features to zero mean and unit variance (z-score normalization) or scale to [0,1] (min-max scaling). This must be fit on training data only — applying test data statistics would leak information.

**Why fit on training data only?** In production, you don't have access to test data statistics. If you normalize using the test set, your offline metrics will be optimistically biased compared to real-world performance.

### Step 5: Cross-Validation for k Selection

Test k values from 1 to sqrt(n) using k-fold cross-validation. Plot accuracy vs. k to find the sweet spot between underfitting (large k) and overfitting (small k).

## Learning Objectives

- Implement distance metrics (Euclidean, Manhattan, Minkowski) from numpy operations
- Build a non-parametric classifier with both uniform and distance-weighted voting
- Understand the curse of dimensionality through empirical demonstration
- Implement cross-validation for hyperparameter selection
- Apply feature scaling correctly (fit on train, transform on test)
- Analyze decision boundaries and how they change with k

## Going Deeper

- **KD-Trees**: Partition space to achieve O(log n) average-case neighbor lookup instead of O(n). But they degrade to O(n) in high dimensions (>20), which ties back to the curse of dimensionality.
- **Ball Trees**: Alternative spatial index that works better than KD-trees in moderate dimensions (20-50).
- **Approximate Nearest Neighbors (ANN)**: Libraries like FAISS and Annoy trade exact correctness for massive speed gains. Used in production recommendation systems serving millions of queries per second.
- **KNN for regression**: Instead of majority vote, average the target values of k neighbors. Weighted averaging by distance works even better.
- **Connection to Day 005 (K-Means)**: K-Means uses the same distance computation but for clustering. The "K" in both refers to different things — in KNN it's the neighbor count, in K-Means it's the cluster count. Both suffer from the curse of dimensionality.
- **Connection to Day 001 (Linear Regression)**: Linear regression learns a global model; KNN makes purely local predictions. This locality gives KNN flexibility to model any decision boundary, but requires more data and is sensitive to noise.
