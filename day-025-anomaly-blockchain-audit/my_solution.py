"""
Day 025: AI Anomaly Detection with Blockchain Audit Trail — Your Implementation

Build an Isolation Forest for anomaly detection and a hash-chained audit ledger
that records every detection decision with cryptographic tamper evidence.

Hints are provided as comments. The __main__ block exercises your code.
"""

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Tree & Forest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IsolationTreeNode:
    """A node in an Isolation Tree.

    Internal nodes: split_feature + split_value, with left/right children
    Leaf nodes: is_leaf=True, size = number of data points that reached here
    """
    split_feature: Optional[int] = None
    split_value: Optional[float] = None
    left: Optional["IsolationTreeNode"] = None
    right: Optional["IsolationTreeNode"] = None
    size: int = 0
    is_leaf: bool = False


def _build_isolation_tree(
    data: list[list[float]],
    current_depth: int,
    max_depth: int,
) -> IsolationTreeNode:
    """Recursively build an isolation tree via random splits.

    Hint: Pick a random feature, find min/max of that feature in data,
    pick a random split value in [min, max], partition data, recurse.

    Stop when: n_samples <= 1, all points identical, or depth >= max_depth.
    """
    raise NotImplementedError("TODO: implement this")


def _path_length(point: list[float], node: IsolationTreeNode, depth: int) -> float:
    """Compute the path length for a point traversing the tree.

    Hint: Traverse left/right based on the split. At a leaf, return
    depth + _c(node.size) to account for the unbuilt subtree.
    """
    raise NotImplementedError("TODO: implement this")


def _c(n: int) -> float:
    """Average path length of unsuccessful search in BST.

    c(n) = 2*H(n-1) - 2*(n-1)/n
    H(i) = ln(i) + 0.5772156649 (Euler-Mascheroni constant)

    Special cases: c(1) = 0, c(2) = 1
    """
    raise NotImplementedError("TODO: implement this")


class IsolationForest:
    """Isolation Forest for unsupervised anomaly detection.

    Hint: fit() builds n_trees trees, each on a random subsample.
    anomaly_score() averages path lengths and converts via 2^(-E(h)/c(n)).
    """

    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 256,
        contamination: float = 0.1,
    ) -> None:
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        self.trees: list[IsolationTreeNode] = []
        self.threshold: float = 0.5

    def fit(self, data: list[list[float]]) -> "IsolationForest":
        """Build the forest and set the anomaly threshold.

        Hint: max_depth = ceil(log2(sample_size)).
        Threshold = score at the (1 - contamination) quantile of training scores.
        """
        raise NotImplementedError("TODO: implement this")

    def anomaly_score(self, point: list[float]) -> float:
        """Compute anomaly score: s(x, n) = 2^(-E(h(x)) / c(n)).

        Hint: Average path length across all trees, then normalize.
        """
        raise NotImplementedError("TODO: implement this")

    def predict(self, point: list[float]) -> tuple[float, bool]:
        """Score a point and classify as anomaly (score >= threshold) or normal."""
        raise NotImplementedError("TODO: implement this")


# ─────────────────────────────────────────────────────────────────────────────
# Blockchain Audit Ledger
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """A single entry in the audit chain.

    Hint: compute_hash() should SHA-256 hash ALL fields except record_hash itself.
    Use json.dumps with sort_keys=True for deterministic serialization.
    """
    index: int
    timestamp: float
    data_point: list[float]
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    previous_hash: str
    record_hash: str = ""

    def compute_hash(self) -> str:
        """SHA-256 hash of (previous_hash + record contents)."""
        raise NotImplementedError("TODO: implement this")


class AuditLedger:
    """Immutable, hash-chained audit log.

    Hint: Each record's previous_hash = prior record's record_hash.
    Genesis block uses "0" * 64 as previous_hash.
    """

    def __init__(self) -> None:
        self.chain: list[AuditRecord] = []

    def append(
        self,
        data_point: list[float],
        anomaly_score: float,
        is_anomaly: bool,
        threshold: float,
    ) -> AuditRecord:
        """Add a new detection record to the chain."""
        raise NotImplementedError("TODO: implement this")

    def verify_integrity(self) -> tuple[bool, Optional[int]]:
        """Verify the entire chain. Returns (is_valid, first_broken_index)."""
        raise NotImplementedError("TODO: implement this")


# ─────────────────────────────────────────────────────────────────────────────
# Data Generation (provided — no need to implement)
# ─────────────────────────────────────────────────────────────────────────────

def generate_normal_data(
    n_points: int,
    n_features: int,
    seed: int = 42,
) -> list[list[float]]:
    """Generate normally-distributed multimodal data for training."""
    rng = random.Random(seed)
    data = []
    centers = [[rng.gauss(0, 0.5) for _ in range(n_features)] for _ in range(3)]
    for _ in range(n_points):
        center = rng.choice(centers)
        point = [center[j] + rng.gauss(0, 0.3) for j in range(n_features)]
        data.append(point)
    return data


def generate_stream_with_anomalies(
    n_points: int,
    n_features: int,
    anomaly_fraction: float = 0.1,
    seed: int = 123,
) -> list[tuple[list[float], bool]]:
    """Generate a stream of (point, is_anomaly) tuples."""
    rng = random.Random(seed)
    stream = []
    centers = [[rng.gauss(0, 0.5) for _ in range(n_features)] for _ in range(3)]
    for _ in range(n_points):
        if rng.random() < anomaly_fraction:
            point = [rng.gauss(0, 3.0) + rng.choice([-3, 3]) for _ in range(n_features)]
            stream.append((point, True))
        else:
            center = rng.choice(centers)
            point = [center[j] + rng.gauss(0, 0.3) for j in range(n_features)]
            stream.append((point, False))
    return stream


def compute_metrics(
    predictions: list[bool],
    ground_truth: list[bool],
) -> dict[str, float]:
    """Compute precision, recall, F1, accuracy."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


# ─────────────────────────────────────────────────────────────────────────────
# Test your implementation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    # Train
    train_data = generate_normal_data(500, 3, seed=42)
    print("Training Isolation Forest...")
    forest = IsolationForest(n_trees=100, sample_size=256, contamination=0.1)
    forest.fit(train_data)
    print(f"Threshold: {forest.threshold:.6f}")

    # Stream + audit
    ledger = AuditLedger()
    stream = generate_stream_with_anomalies(50, 3, anomaly_fraction=0.15, seed=123)
    predictions, truth = [], []

    for point, is_true_anomaly in stream:
        score, is_predicted = forest.predict(point)
        ledger.append(point, score, is_predicted, forest.threshold)
        predictions.append(is_predicted)
        truth.append(is_true_anomaly)

    # Metrics
    metrics = compute_metrics(predictions, truth)
    print(f"F1: {metrics['f1']:.3f}, Recall: {metrics['recall']:.3f}")

    # Verify chain
    valid, broken = ledger.verify_integrity()
    print(f"Chain integrity: {'VALID' if valid else f'BROKEN at {broken}'}")
    print("Done!")
