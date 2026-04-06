"""
Day 004: Decision Tree — Test Suite

Run with:
    python3 -m pytest tests.py -v
    python3 tests.py
"""

import math
import unittest
from collections import Counter

from my_solution import (
    Node,
    entropy,
    gini_impurity,
    information_gain,
    DecisionTreeClassifier,
    accuracy,
    make_classification_data,
)


class TestNode(unittest.TestCase):
    """Tests for the Node class."""

    def test_leaf_node(self):
        node = Node(label=1)
        self.assertTrue(node.is_leaf())

    def test_internal_node(self):
        left = Node(label=0)
        right = Node(label=1)
        node = Node(feature_idx=0, threshold=0.5, left=left, right=right)
        self.assertFalse(node.is_leaf())


class TestEntropy(unittest.TestCase):
    """Tests for the entropy function."""

    def test_pure_set_returns_zero(self):
        """Entropy of a pure set (all same class) should be 0."""
        self.assertAlmostEqual(entropy([1, 1, 1, 1, 1]), 0.0)
        self.assertAlmostEqual(entropy([0, 0, 0]), 0.0)

    def test_balanced_binary_returns_one(self):
        """Entropy of a 50/50 binary split should be 1.0 bit."""
        self.assertAlmostEqual(entropy([0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(entropy([0, 1]), 1.0)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(entropy([]), 0.0)

    def test_imbalanced(self):
        """Entropy of an imbalanced set should be between 0 and 1."""
        h = entropy([0, 0, 0, 0, 1])
        self.assertGreater(h, 0.0)
        self.assertLess(h, 1.0)


class TestGiniImpurity(unittest.TestCase):
    """Tests for the gini_impurity function."""

    def test_pure_set_returns_zero(self):
        """Gini impurity of a pure set should be 0."""
        self.assertAlmostEqual(gini_impurity([1, 1, 1]), 0.0)
        self.assertAlmostEqual(gini_impurity([0, 0, 0, 0]), 0.0)

    def test_balanced_binary(self):
        """Gini of 50/50 split should be 0.5."""
        self.assertAlmostEqual(gini_impurity([0, 1]), 0.5)
        self.assertAlmostEqual(gini_impurity([0, 0, 1, 1]), 0.5)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(gini_impurity([]), 0.0)


class TestInformationGain(unittest.TestCase):
    """Tests for the information_gain function."""

    def test_perfect_split_has_max_gain(self):
        """A perfect split of a balanced set should yield gain = 1.0."""
        parent = [0, 0, 0, 1, 1, 1]
        left = [0, 0, 0]
        right = [1, 1, 1]
        gain = information_gain(parent, left, right, "entropy")
        self.assertAlmostEqual(gain, 1.0)

    def test_no_split_has_zero_gain(self):
        """A split that doesn't change the distribution yields 0 gain."""
        parent = [0, 0, 1, 1]
        left = [0, 1]
        right = [0, 1]
        gain = information_gain(parent, left, right, "entropy")
        self.assertAlmostEqual(gain, 0.0)

    def test_positive_gain_for_good_split(self):
        """A reasonable split should produce positive information gain."""
        parent = [0, 0, 0, 1, 1, 1, 0, 1]
        left = [0, 0, 0, 1]
        right = [1, 1, 1, 0]
        gain = information_gain(parent, left, right, "entropy")
        self.assertGreater(gain, 0.0)

    def test_gini_criterion(self):
        """Information gain should also work with 'gini' criterion."""
        parent = [0, 0, 1, 1]
        left = [0, 0]
        right = [1, 1]
        gain = information_gain(parent, left, right, "gini")
        self.assertAlmostEqual(gain, 0.5)


class TestDecisionTreeClassifier(unittest.TestCase):
    """Tests for the DecisionTreeClassifier."""

    @classmethod
    def setUpClass(cls):
        """Generate data once for all tree tests."""
        cls.X, cls.y, cls.feature_names = make_classification_data(
            n_samples=200, seed=42
        )
        split = int(0.8 * len(cls.X))
        cls.X_train = cls.X[:split]
        cls.y_train = cls.y[:split]
        cls.X_test = cls.X[split:]
        cls.y_test = cls.y[split:]

    def test_tree_accuracy_above_85(self):
        """Tree should achieve >85% accuracy on the circular boundary dataset."""
        tree = DecisionTreeClassifier(max_depth=5, criterion="entropy")
        tree.fit(self.X_train, self.y_train)
        preds = tree.predict(self.X_test)
        acc = accuracy(self.y_test, preds)
        self.assertGreater(acc, 0.85, f"Expected >85% accuracy, got {acc:.2%}")

    def test_stump_lower_accuracy_than_deep_tree(self):
        """A depth-1 stump should have lower accuracy than a deeper tree."""
        stump = DecisionTreeClassifier(max_depth=1, criterion="entropy")
        stump.fit(self.X_train, self.y_train)
        stump_acc = accuracy(self.y_test, stump.predict(self.X_test))

        deep = DecisionTreeClassifier(max_depth=5, criterion="entropy")
        deep.fit(self.X_train, self.y_train)
        deep_acc = accuracy(self.y_test, deep.predict(self.X_test))

        self.assertLess(
            stump_acc, deep_acc,
            f"Stump accuracy ({stump_acc:.2%}) should be less than deep tree ({deep_acc:.2%})"
        )

    def test_predict_output_length(self):
        """predict() output length should match input length."""
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(self.X_train, self.y_train)
        preds = tree.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_feature_importances_sum_to_one(self):
        """Feature importances should sum to approximately 1.0."""
        tree = DecisionTreeClassifier(max_depth=5, criterion="entropy")
        tree.fit(self.X_train, self.y_train)
        importances = tree.feature_importances(self.X_train, self.y_train)
        self.assertAlmostEqual(sum(importances), 1.0, places=5)

    def test_pure_data_creates_single_leaf(self):
        """Training on data with one class should produce a single leaf."""
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [0, 0, 0]
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        self.assertTrue(tree.root.is_leaf())
        self.assertEqual(tree.predict([[0.0, 0.0]]), [0])

    def test_gini_criterion_works(self):
        """Tree with gini criterion should also achieve reasonable accuracy."""
        tree = DecisionTreeClassifier(max_depth=5, criterion="gini")
        tree.fit(self.X_train, self.y_train)
        preds = tree.predict(self.X_test)
        acc = accuracy(self.y_test, preds)
        self.assertGreater(acc, 0.80)


class TestAccuracy(unittest.TestCase):
    """Tests for the accuracy function."""

    def test_perfect(self):
        self.assertAlmostEqual(accuracy([0, 1, 0, 1], [0, 1, 0, 1]), 1.0)

    def test_half(self):
        self.assertAlmostEqual(accuracy([0, 0, 1, 1], [0, 0, 0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
