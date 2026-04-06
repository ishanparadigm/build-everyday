"""
Day 004: Decision Tree Classifier from Scratch

A complete CART-style decision tree that handles continuous features,
uses information gain (entropy) or Gini impurity for splitting,
and includes regularization via max_depth and min_samples_split.

Building on Days 001 and 003: unlike linear regression and logistic regression,
decision trees can learn non-linear decision boundaries without feature engineering.
"""

import math
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Node: the recursive building block of the tree
# ---------------------------------------------------------------------------
# Hint: A decision tree is a binary tree. Internal nodes store a split rule
# (feature index + threshold). Leaves store a class label.

class Node:
    """A node in the decision tree."""

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        label: Optional[int] = None,
    ):
        raise NotImplementedError("TODO: implement this")

    def is_leaf(self) -> bool:
        """Return True if this node is a leaf (has a class label)."""
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Impurity measures: entropy and Gini
# ---------------------------------------------------------------------------
# Hint: Both measure how "mixed" a set of labels is.
# Pure set -> 0, maximally mixed -> highest value.
# Use collections.Counter to count label frequencies.

def entropy(labels: list[int]) -> float:
    """
    Shannon entropy: H(S) = -Σ p_i * log2(p_i)

    Measures the average number of bits needed to encode a randomly drawn label.
    Convention: 0 * log2(0) = 0 (a class with zero probability adds no uncertainty).

    Returns 0.0 for empty input.
    """
    # Hint: compute p_i = count_i / n for each class, then sum -p * log2(p)
    raise NotImplementedError("TODO: implement this")


def gini_impurity(labels: list[int]) -> float:
    """
    Gini impurity: Gini(S) = 1 - Σ p_i²

    Probability that two randomly drawn samples would have different labels.
    Cheaper than entropy (no logarithm), produces nearly identical trees.

    Returns 0.0 for empty input.
    """
    # Hint: compute p_i = count_i / n for each class, then 1 - sum(p_i^2)
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Information gain: the criterion for choosing splits
# ---------------------------------------------------------------------------

def information_gain(
    parent_labels: list[int],
    left_labels: list[int],
    right_labels: list[int],
    criterion: str = "entropy",
) -> float:
    """
    Information gain = parent impurity - weighted average of children impurity.

    Higher information gain means the split better separates the classes.
    The weighting by |child|/|parent| accounts for how many samples go each way.
    """
    # Hint: pick the impurity function based on criterion ("entropy" or "gini"),
    # compute parent impurity, then subtract the weighted child impurities.
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# DecisionTreeClassifier: the main class
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """
    CART-style decision tree for classification.

    Parameters:
        max_depth: Maximum depth of the tree. None = unlimited.
        min_samples_split: Minimum samples required to attempt a split.
        criterion: "entropy" (information gain) or "gini" (Gini impurity).
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        criterion: str = "entropy",
    ):
        raise NotImplementedError("TODO: implement this")

    def fit(self, X: list[list[float]], y: list[int]) -> "DecisionTreeClassifier":
        """
        Build the decision tree from training data.

        X: list of feature vectors (each sample is a list of floats)
        y: list of class labels (integers)

        Returns self for method chaining.
        """
        # Hint: store n_features, then call _build_tree to create self.root
        raise NotImplementedError("TODO: implement this")

    def _build_tree(self, X: list[list[float]], y: list[int], depth: int) -> Node:
        """
        Recursively build the tree.

        At each call, either:
        1. Create a leaf (if stopping criteria are met), or
        2. Find the best split, partition the data, and recurse on each half.
        """
        # Hint: stopping criteria — pure node, max_depth reached, too few samples.
        # For leaf, use majority vote (Counter.most_common).
        #
        # To find best split: iterate over features, find sorted unique values,
        # try midpoints between consecutive values as thresholds.
        # Pick the split with highest information_gain.
        #
        # Then partition X, y by the best split and recurse.
        raise NotImplementedError("TODO: implement this")

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict class labels for a list of samples."""
        # Hint: call _predict_one for each sample
        raise NotImplementedError("TODO: implement this")

    def _predict_one(self, x: list[float], node: Node) -> int:
        """
        Traverse the tree for a single sample.

        At each internal node, go left if x[feature_idx] <= threshold, else right.
        Return the label when a leaf is reached.
        """
        # Hint: recursive traversal — check is_leaf(), then branch left or right
        raise NotImplementedError("TODO: implement this")

    def print_tree(self, node: Optional[Node] = None, indent: str = "", feature_names: Optional[list[str]] = None) -> None:
        """
        Print the tree structure for interpretability.

        One of decision trees' biggest advantages: you can read the learned rules.
        """
        # Hint: recursively print each node. Leaves print the class label.
        # Internal nodes print "feature <= threshold?" and recurse left/right.
        raise NotImplementedError("TODO: implement this")

    def feature_importances(self, X: list[list[float]], y: list[int]) -> list[float]:
        """
        Compute feature importance based on total information gain.

        For each feature, sum up the information gain from every split that uses it,
        weighted by the number of samples reaching that node.
        Normalize so importances sum to 1.
        """
        # Hint: create a list of zeros (one per feature), then recursively
        # walk the tree accumulating gain * n_samples at each split node.
        # Finally normalize by dividing by the total.
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Fraction of correct predictions."""
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Dataset generation (provided — no need to implement)
# ---------------------------------------------------------------------------

def make_classification_data(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[list[list[float]], list[int], list[str]]:
    """
    Generate a synthetic 2D dataset with a non-linear circular boundary.
    Points inside radius 1.2 from origin -> class 1, outside -> class 0.
    Includes a noise feature and ~5% label noise.
    """
    import random
    random.seed(seed)

    X = []
    y = []

    for _ in range(n_samples):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        noise_feature = random.uniform(-1, 1)

        distance = math.sqrt(x1 ** 2 + x2 ** 2)
        label = 1 if distance < 1.2 else 0

        if random.random() < 0.05:
            label = 1 - label

        X.append([x1, x2, noise_feature])
        y.append(label)

    feature_names = ["x1", "x2", "noise"]
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Main: test your implementation as you build it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("DECISION TREE CLASSIFIER — YOUR IMPLEMENTATION")
    print("=" * 65)

    # Test impurity measures
    print("\n--- Testing impurity measures ---")
    pure = [1, 1, 1, 1, 1]
    mixed = [0, 0, 1, 1]
    print(f"Entropy of pure set:  {entropy(pure):.4f}  (expected: 0.0)")
    print(f"Entropy of 50/50:     {entropy(mixed):.4f}  (expected: 1.0)")
    print(f"Gini of pure set:     {gini_impurity(pure):.4f}  (expected: 0.0)")
    print(f"Gini of 50/50:        {gini_impurity(mixed):.4f}  (expected: 0.5)")

    # Test information gain
    print("\n--- Testing information gain ---")
    parent = [0, 0, 0, 1, 1, 1]
    left = [0, 0, 0]
    right = [1, 1, 1]
    gain = information_gain(parent, left, right)
    print(f"Perfect split gain: {gain:.4f}  (expected: 1.0)")

    # Generate data and train
    print("\n--- Training decision tree ---")
    X, y, feature_names = make_classification_data(n_samples=200, seed=42)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    tree = DecisionTreeClassifier(max_depth=5, criterion="entropy")
    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)

    print(f"Train accuracy: {accuracy(y_train, train_pred):.4f}")
    print(f"Test accuracy:  {accuracy(y_test, test_pred):.4f}")

    # Print tree
    print("\n--- Tree structure ---")
    tree.print_tree(feature_names=feature_names)

    # Feature importances
    print("\n--- Feature importances ---")
    importances = tree.feature_importances(X_train, y_train)
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.4f}")
    print(f"  Sum: {sum(importances):.4f}  (expected: ~1.0)")
