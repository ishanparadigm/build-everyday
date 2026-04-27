"""
Day 025: Tests for AI Anomaly Detection with Blockchain Audit Trail

Run with: python3 -m pytest tests.py -v
     or: python3 tests.py
"""

import math
import random
import unittest

from my_solution import (
    IsolationForest,
    IsolationTreeNode,
    AuditLedger,
    AuditRecord,
    _build_isolation_tree,
    _c,
    _path_length,
    generate_normal_data,
    generate_stream_with_anomalies,
    compute_metrics,
)


class TestBSTPathLength(unittest.TestCase):
    """Test the c(n) normalization function."""

    def test_c_base_cases(self):
        self.assertAlmostEqual(_c(1), 0.0)
        self.assertAlmostEqual(_c(2), 1.0)

    def test_c_monotonically_increasing(self):
        """c(n) should increase with n — more points = deeper average tree."""
        values = [_c(n) for n in range(2, 100)]
        for i in range(len(values) - 1):
            self.assertLess(values[i], values[i + 1])

    def test_c_known_value(self):
        """c(256) ≈ 9.898 (from the original paper)."""
        self.assertAlmostEqual(_c(256), 2 * (math.log(255) + 0.5772156649) - 2 * 255 / 256, places=3)


class TestIsolationTree(unittest.TestCase):
    """Test individual isolation tree construction and path length."""

    def test_single_point_is_leaf(self):
        tree = _build_isolation_tree([[1.0, 2.0]], current_depth=0, max_depth=10)
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.size, 1)

    def test_identical_points_become_leaf(self):
        data = [[1.0, 2.0]] * 10
        tree = _build_isolation_tree(data, current_depth=0, max_depth=10)
        self.assertTrue(tree.is_leaf)
        self.assertEqual(tree.size, 10)

    def test_max_depth_respected(self):
        """Tree should not exceed max_depth."""
        random.seed(42)
        data = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(100)]
        tree = _build_isolation_tree(data, current_depth=0, max_depth=3)

        def _max_depth(node, d=0):
            if node.is_leaf:
                return d
            return max(_max_depth(node.left, d + 1), _max_depth(node.right, d + 1))

        self.assertLessEqual(_max_depth(tree), 3)

    def test_path_length_positive(self):
        random.seed(42)
        data = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(50)]
        tree = _build_isolation_tree(data, current_depth=0, max_depth=8)
        pl = _path_length([0.0, 0.0], tree, depth=0)
        self.assertGreater(pl, 0)


class TestIsolationForest(unittest.TestCase):
    """Test the full Isolation Forest."""

    def setUp(self):
        random.seed(42)
        self.normal_data = generate_normal_data(300, 2, seed=42)
        self.forest = IsolationForest(n_trees=50, sample_size=128, contamination=0.1)
        self.forest.fit(self.normal_data)

    def test_scores_in_range(self):
        """All anomaly scores should be in [0, 1]."""
        for point in self.normal_data[:50]:
            score = self.forest.anomaly_score(point)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_outlier_scores_higher(self):
        """Points far from the training distribution should score higher."""
        normal_scores = [self.forest.anomaly_score(p) for p in self.normal_data[:30]]
        outliers = [[10.0, 10.0], [-10.0, -10.0], [15.0, -15.0]]
        outlier_scores = [self.forest.anomaly_score(p) for p in outliers]

        avg_normal = sum(normal_scores) / len(normal_scores)
        avg_outlier = sum(outlier_scores) / len(outlier_scores)
        self.assertGreater(avg_outlier, avg_normal,
                           "Outliers should have higher average anomaly scores than normal points")

    def test_predict_returns_tuple(self):
        score, is_anomaly = self.forest.predict([0.0, 0.0])
        self.assertIsInstance(score, float)
        self.assertIsInstance(is_anomaly, bool)

    def test_extreme_outlier_detected(self):
        """A point at [100, 100] should definitely be flagged."""
        _, is_anomaly = self.forest.predict([100.0, 100.0])
        self.assertTrue(is_anomaly, "Extreme outlier should be detected as anomaly")

    def test_threshold_set_after_fit(self):
        """Threshold should be set to a reasonable value after fitting."""
        self.assertGreater(self.forest.threshold, 0.0)
        self.assertLess(self.forest.threshold, 1.0)


class TestAuditLedger(unittest.TestCase):
    """Test the hash-chained audit ledger."""

    def test_empty_chain_is_valid(self):
        ledger = AuditLedger()
        is_valid, broken = ledger.verify_integrity()
        self.assertTrue(is_valid)
        self.assertIsNone(broken)

    def test_single_record_valid(self):
        ledger = AuditLedger()
        ledger.append([1.0, 2.0], 0.7, True, 0.6)
        is_valid, broken = ledger.verify_integrity()
        self.assertTrue(is_valid)

    def test_chain_links_correctly(self):
        """Each record's previous_hash should equal the prior record's hash."""
        ledger = AuditLedger()
        for i in range(5):
            ledger.append([float(i)], 0.5, False, 0.6)

        for i in range(1, len(ledger.chain)):
            self.assertEqual(
                ledger.chain[i].previous_hash,
                ledger.chain[i - 1].record_hash,
            )

    def test_tamper_detection(self):
        """Modifying a record should break the chain."""
        ledger = AuditLedger()
        for i in range(10):
            ledger.append([float(i)], 0.5, False, 0.6)

        # Tamper with record #3
        ledger.chain[3].anomaly_score = 0.999

        is_valid, broken = ledger.verify_integrity()
        self.assertFalse(is_valid, "Tampered chain should fail verification")
        self.assertEqual(broken, 3, "Should detect tampering at index 3")

    def test_genesis_previous_hash(self):
        """First record should have zeroed previous hash."""
        ledger = AuditLedger()
        ledger.append([1.0], 0.5, False, 0.6)
        self.assertEqual(ledger.chain[0].previous_hash, "0" * 64)

    def test_hash_deterministic(self):
        """Same inputs should produce the same hash."""
        record = AuditRecord(
            index=0, timestamp=1000.0, data_point=[1.0, 2.0],
            anomaly_score=0.75, is_anomaly=True, threshold=0.6,
            previous_hash="0" * 64,
        )
        h1 = record.compute_hash()
        h2 = record.compute_hash()
        self.assertEqual(h1, h2)


class TestMetrics(unittest.TestCase):
    """Test evaluation metric computation."""

    def test_perfect_predictions(self):
        preds = [True, False, True, False]
        truth = [True, False, True, False]
        m = compute_metrics(preds, truth)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)
        self.assertAlmostEqual(m["accuracy"], 1.0)

    def test_all_false_negatives(self):
        preds = [False, False, False]
        truth = [True, True, True]
        m = compute_metrics(preds, truth)
        self.assertAlmostEqual(m["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
