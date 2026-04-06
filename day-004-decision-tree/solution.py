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
# A decision tree is literally a tree data structure. Each internal node stores
# a splitting rule (feature index + threshold). Each leaf stores a class label.
# We use a single class for both — leaves have left=right=None.

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
        self.feature_idx = feature_idx    # which feature to split on (None for leaves)
        self.threshold = threshold        # split threshold: <= goes left, > goes right
        self.left = left                  # left child subtree
        self.right = right                # right child subtree
        self.label = label                # predicted class (only set for leaves)

    def is_leaf(self) -> bool:
        return self.label is not None


# ---------------------------------------------------------------------------
# Impurity measures: entropy and Gini
# ---------------------------------------------------------------------------
# These functions measure how "mixed" a set of labels is. A pure set (all one
# class) has impurity 0. Maximum impurity occurs when classes are equally
# distributed. The split that reduces impurity the most is the best split.

def entropy(labels: list[int]) -> float:
    """
    Shannon entropy: H(S) = -Σ p_i * log2(p_i)

    Why entropy? It comes from information theory — it measures the average
    number of bits needed to encode a randomly drawn label. When we split
    data, we want to reduce this uncertainty as much as possible.

    Edge case: p_i = 0 contributes 0 (by convention, 0*log(0) = 0),
    because a class with zero probability adds no uncertainty.
    """
    n = len(labels)
    if n == 0:
        return 0.0

    counts = Counter(labels)
    h = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def gini_impurity(labels: list[int]) -> float:
    """
    Gini impurity: Gini(S) = 1 - Σ p_i²

    Interpretation: probability that two randomly drawn samples from this
    set would have different labels. Cheaper to compute than entropy
    (no logarithm), and produces nearly identical trees in practice.
    """
    n = len(labels)
    if n == 0:
        return 0.0

    counts = Counter(labels)
    return 1.0 - sum((count / n) ** 2 for count in counts.values())


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

    This is the core metric that drives tree construction. A higher information
    gain means the split does a better job of separating the classes.

    The weighting by |child|/|parent| is crucial — without it, we'd favor
    splits that create one tiny pure child and one large impure child.
    The weighted average ensures we account for how many samples go each way.
    """
    impurity_fn = entropy if criterion == "entropy" else gini_impurity

    parent_impurity = impurity_fn(parent_labels)

    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    # Weighted average of child impurities
    child_impurity = (n_left / n) * impurity_fn(left_labels) + \
                     (n_right / n) * impurity_fn(right_labels)

    return parent_impurity - child_impurity


# ---------------------------------------------------------------------------
# DecisionTreeClassifier: the main class
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """
    CART-style decision tree for classification.

    Parameters:
        max_depth: Maximum depth of the tree. None = unlimited.
                   This is the primary regularization knob. Shallow trees
                   underfit (high bias), deep trees overfit (high variance).
        min_samples_split: Minimum samples required to attempt a split.
                           Prevents creating nodes from tiny subsets.
        criterion: "entropy" (information gain) or "gini" (Gini impurity).
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        criterion: str = "entropy",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root: Optional[Node] = None
        self.n_features: int = 0

    def fit(self, X: list[list[float]], y: list[int]) -> "DecisionTreeClassifier":
        """
        Build the decision tree from training data.

        X: list of feature vectors (each sample is a list of floats)
        y: list of class labels (integers)

        The tree is built top-down by recursively finding the best split.
        This is a greedy algorithm — we pick the locally optimal split at
        each node, which is an O(n * m * n) operation per level
        (n samples, m features, n thresholds to try per feature).
        """
        self.n_features = len(X[0])
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: list[list[float]], y: list[int], depth: int) -> Node:
        """
        Recursively build the tree. This is where the magic happens.

        At each call, we either:
        1. Create a leaf (if stopping criteria are met), or
        2. Find the best split, partition the data, and recurse on each half.
        """
        n_samples = len(y)
        n_classes = len(set(y))

        # --- Stopping criteria ---
        # These prevent overfitting by limiting tree growth.

        # Pure node: all samples have the same label. No split can improve this.
        # Max depth reached: regularization to prevent the tree from memorizing noise.
        # Too few samples: not enough data to make a meaningful split.
        if (n_classes == 1 or
            (self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split):
            # Majority vote: the most common label becomes the prediction.
            # This is the tree's equivalent of "best guess given the data here."
            leaf_label = Counter(y).most_common(1)[0][0]
            return Node(label=leaf_label)

        # --- Find the best split ---
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(self.n_features):
            # Extract this feature's values and find candidate thresholds.
            # We only need to try thresholds at midpoints between consecutive
            # distinct values — splits between identical values can't help.
            values = [X[i][feature_idx] for i in range(n_samples)]
            unique_vals = sorted(set(values))

            # Midpoints between consecutive unique values
            # Why midpoints? A threshold at the exact value of a data point
            # creates ambiguity. Midpoints ensure clean separation.
            for i in range(len(unique_vals) - 1):
                threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0

                # Partition labels based on threshold
                left_labels = [y[j] for j in range(n_samples) if X[j][feature_idx] <= threshold]
                right_labels = [y[j] for j in range(n_samples) if X[j][feature_idx] > threshold]

                # Skip if either side is empty (no actual split)
                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue

                gain = information_gain(y, left_labels, right_labels, self.criterion)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # If no split improves things (gain <= 0 or no valid split found),
        # create a leaf. This can happen when all features are identical.
        if best_gain <= 0 or best_feature is None:
            leaf_label = Counter(y).most_common(1)[0][0]
            return Node(label=leaf_label)

        # --- Partition data and recurse ---
        left_indices = [i for i in range(n_samples) if X[i][best_feature] <= best_threshold]
        right_indices = [i for i in range(n_samples) if X[i][best_feature] > best_threshold]

        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]

        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        return Node(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict class labels for a list of samples."""
        return [self._predict_one(x, self.root) for x in X]

    def _predict_one(self, x: list[float], node: Node) -> int:
        """
        Traverse the tree for a single sample.

        At each internal node, compare the sample's feature value to the
        threshold. Go left if <=, right if >. When we hit a leaf, return
        its label. This is O(depth) per sample.
        """
        if node.is_leaf():
            return node.label

        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def print_tree(self, node: Optional[Node] = None, indent: str = "", feature_names: Optional[list[str]] = None) -> None:
        """
        Print the tree structure for interpretability.

        One of decision trees' biggest advantages: you can read the learned
        rules directly. Try doing that with a neural network.
        """
        if node is None:
            node = self.root

        if node.is_leaf():
            print(f"{indent}→ class {node.label}")
            return

        feature_name = feature_names[node.feature_idx] if feature_names else f"feature[{node.feature_idx}]"

        print(f"{indent}{feature_name} <= {node.threshold:.4f}?")
        print(f"{indent}├─ Yes:")
        self.print_tree(node.left, indent + "│  ", feature_names)
        print(f"{indent}└─ No:")
        self.print_tree(node.right, indent + "   ", feature_names)

    def feature_importances(self, X: list[list[float]], y: list[int]) -> list[float]:
        """
        Compute feature importance based on total information gain.

        For each feature, sum up the information gain from every split
        that uses it. Features that appear higher in the tree and split
        more samples will have higher importance.

        This is a simplified version — scikit-learn weights by the number
        of samples reaching each node, which we also do here.
        """
        importances = [0.0] * self.n_features
        self._compute_importances(self.root, X, y, importances)

        # Normalize to sum to 1
        total = sum(importances)
        if total > 0:
            importances = [imp / total for imp in importances]

        return importances

    def _compute_importances(
        self,
        node: Node,
        X: list[list[float]],
        y: list[int],
        importances: list[float],
    ) -> None:
        """Recursively accumulate feature importance from each split."""
        if node.is_leaf() or len(y) == 0:
            return

        # Compute gain at this node
        left_indices = [i for i in range(len(y)) if X[i][node.feature_idx] <= node.threshold]
        right_indices = [i for i in range(len(y)) if X[i][node.feature_idx] > node.threshold]

        left_y = [y[i] for i in left_indices]
        right_y = [y[i] for i in right_indices]

        gain = information_gain(y, left_y, right_y, self.criterion)

        # Weight by number of samples (splits higher in the tree affect more samples)
        importances[node.feature_idx] += gain * len(y)

        left_X = [X[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]

        self._compute_importances(node.left, left_X, left_y, importances)
        self._compute_importances(node.right, right_X, right_y, importances)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_classification_data(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[list[list[float]], list[int], list[str]]:
    """
    Generate a synthetic 2D dataset with a non-linear decision boundary.

    We create data where the true boundary is circular: points inside a
    radius belong to class 1, outside to class 0. This is impossible for
    linear models (Day 001, 003) but trivial for decision trees.

    Also adds a third noise feature to test whether the tree correctly
    ignores irrelevant features.
    """
    import random
    random.seed(seed)

    X = []
    y = []

    for _ in range(n_samples):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        noise_feature = random.uniform(-1, 1)  # irrelevant feature

        # True boundary: circle of radius 1.2
        # Points inside the circle = class 1, outside = class 0
        distance = math.sqrt(x1 ** 2 + x2 ** 2)
        label = 1 if distance < 1.2 else 0

        # Add some label noise (~5%) to make it realistic
        if random.random() < 0.05:
            label = 1 - label

        X.append([x1, x2, noise_feature])
        y.append(label)

    feature_names = ["x1", "x2", "noise"]
    return X, y, feature_names


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Fraction of correct predictions."""
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


# ---------------------------------------------------------------------------
# Main: demonstrate the full decision tree pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("DECISION TREE CLASSIFIER FROM SCRATCH")
    print("=" * 65)

    # --- Step 1: Generate data ---
    print("\n--- Step 1: Generate synthetic data ---")
    print("True boundary: circular (x1² + x2² < 1.44 → class 1)")
    print("This is NON-LINEAR — logistic regression from Day 003 would fail here.\n")

    X, y, feature_names = make_classification_data(n_samples=200, seed=42)

    # Train/test split (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Features:         {feature_names}")
    print(f"Class distribution (train): {Counter(y_train)}")

    # --- Step 2: Demonstrate entropy and Gini ---
    print("\n--- Step 2: Understanding impurity measures ---")

    pure = [1, 1, 1, 1, 1]
    mixed = [0, 0, 1, 1]
    imbalanced = [0, 0, 0, 0, 1]

    for name, labels in [("Pure [1,1,1,1,1]", pure),
                          ("Mixed [0,0,1,1]", mixed),
                          ("Imbalanced [0,0,0,0,1]", imbalanced)]:
        print(f"  {name}:")
        print(f"    Entropy: {entropy(labels):.4f}    Gini: {gini_impurity(labels):.4f}")

    print("\n  Key insight: both measures are 0 for pure sets and maximize")
    print("  for uniform distributions. Entropy uses bits, Gini uses probability.")

    # --- Step 3: Train with different depths ---
    print("\n--- Step 3: Effect of max_depth (bias-variance tradeoff) ---")
    print(f"  {'Depth':<8} {'Train Acc':<12} {'Test Acc':<12} {'Overfit?'}")
    print(f"  {'-'*44}")

    for depth in [1, 2, 3, 5, 10, None]:
        tree = DecisionTreeClassifier(max_depth=depth, criterion="entropy")
        tree.fit(X_train, y_train)

        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)

        train_acc = accuracy(y_train, train_pred)
        test_acc = accuracy(y_test, test_pred)

        depth_str = str(depth) if depth else "None"
        gap = train_acc - test_acc
        overfit = "← likely" if gap > 0.08 else ""

        print(f"  {depth_str:<8} {train_acc:<12.4f} {test_acc:<12.4f} {overfit}")

    print("\n  Notice: very shallow trees underfit (low train AND test accuracy).")
    print("  Very deep trees overfit (high train accuracy, lower test accuracy).")
    print("  The sweet spot is usually a moderate depth.")

    # --- Step 4: Train the best model and inspect it ---
    print("\n--- Step 4: Inspecting the learned tree (max_depth=4) ---\n")

    best_tree = DecisionTreeClassifier(max_depth=4, criterion="entropy")
    best_tree.fit(X_train, y_train)

    best_tree.print_tree(feature_names=feature_names)

    test_pred = best_tree.predict(X_test)
    test_acc = accuracy(y_test, test_pred)
    print(f"\n  Test accuracy: {test_acc:.4f}")

    # --- Step 5: Feature importances ---
    print("\n--- Step 5: Feature importance ---")
    importances = best_tree.feature_importances(X_train, y_train)
    for name, imp in zip(feature_names, importances):
        bar = "█" * int(imp * 40)
        print(f"  {name:<8} {imp:.4f}  {bar}")

    print("\n  The tree correctly identifies x1 and x2 as important features")
    print("  and assigns low importance to the noise feature.")
    print("  This automatic feature selection is a major advantage of trees.")

    # --- Step 6: Compare entropy vs gini ---
    print("\n--- Step 6: Entropy vs Gini comparison ---")
    for criterion in ["entropy", "gini"]:
        tree = DecisionTreeClassifier(max_depth=4, criterion=criterion)
        tree.fit(X_train, y_train)
        test_pred = tree.predict(X_test)
        test_acc = accuracy(y_test, test_pred)
        print(f"  {criterion.capitalize():<10} Test accuracy: {test_acc:.4f}")

    print("  (They usually produce very similar results — Gini is just faster to compute)")

    # --- Step 7: Trace a single prediction ---
    print("\n--- Step 7: Tracing a single prediction ---")
    sample = X_test[0]
    true_label = y_test[0]
    predicted = best_tree.predict([sample])[0]

    print(f"  Sample: {feature_names[0]}={sample[0]:.3f}, "
          f"{feature_names[1]}={sample[1]:.3f}, "
          f"{feature_names[2]}={sample[2]:.3f}")
    print(f"  True label: {true_label}    Predicted: {predicted}")
    dist = math.sqrt(sample[0]**2 + sample[1]**2)
    print(f"  Distance from origin: {dist:.3f} (boundary at 1.2)")
    print(f"  The tree approximates the circular boundary with axis-aligned splits!")

    print("\n" + "=" * 65)
    print("KEY TAKEAWAYS")
    print("=" * 65)
    print("1. Decision trees learn non-linear boundaries via recursive splitting")
    print("2. Information gain (entropy) guides which feature to split on")
    print("3. max_depth controls the bias-variance tradeoff")
    print("4. Trees are interpretable — you can read the learned rules directly")
    print("5. Trees automatically rank features by importance")
    print("6. Next up: ensemble methods (Random Forest, Boosting) fix overfitting")
    print("   by combining many trees — the insight behind XGBoost and LightGBM")
