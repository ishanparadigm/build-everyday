"""
Day 025: AI Anomaly Detection with Blockchain Audit Trail

Combines Isolation Forest (unsupervised anomaly detection) with a hash-chained
audit ledger to create a tamper-evident detection pipeline.

Key ideas:
  - Isolation Forest isolates anomalies via random recursive partitioning
  - Anomalous points have shorter average path lengths across the ensemble
  - Each detection decision is recorded on a SHA-256 hash chain
  - Modifying any historical record breaks the chain — cryptographic tamper evidence
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

    Internal nodes store a split (feature index + split value).
    Leaf nodes store the count of data points that reached them — this is
    used to estimate path lengths for points that hit max depth.
    """
    split_feature: Optional[int] = None
    split_value: Optional[float] = None
    left: Optional["IsolationTreeNode"] = None
    right: Optional["IsolationTreeNode"] = None
    size: int = 0  # number of data points at this leaf
    is_leaf: bool = False


def _build_isolation_tree(
    data: list[list[float]],
    current_depth: int,
    max_depth: int,
) -> IsolationTreeNode:
    """Recursively build an isolation tree via random splits.

    We stop splitting when:
      1. Only one point remains (perfectly isolated)
      2. All points are identical (can't split further)
      3. Max depth reached (controls memory; deeper ≠ better for anomaly detection)

    The randomness is the core mechanism: in dense regions, random splits rarely
    isolate a single point. In sparse regions (anomalies), a single random split
    often does the job.
    """
    n_samples = len(data)
    n_features = len(data[0]) if data else 0

    # Base cases: create a leaf
    if n_samples <= 1 or current_depth >= max_depth:
        node = IsolationTreeNode(size=n_samples, is_leaf=True)
        return node

    # Check if all points are identical (no split possible)
    if all(data[i] == data[0] for i in range(n_samples)):
        node = IsolationTreeNode(size=n_samples, is_leaf=True)
        return node

    # Pick a random feature and a random split value within the data's range
    # for that feature. Using the actual data range (not arbitrary bounds)
    # ensures we don't waste splits on empty regions.
    feature_idx = random.randint(0, n_features - 1)
    feature_values = [row[feature_idx] for row in data]
    min_val, max_val = min(feature_values), max(feature_values)

    # If this feature is constant, try another one
    # (rare with continuous data, common with categorical)
    attempts = 0
    while min_val == max_val and attempts < n_features:
        feature_idx = (feature_idx + 1) % n_features
        feature_values = [row[feature_idx] for row in data]
        min_val, max_val = min(feature_values), max(feature_values)
        attempts += 1

    if min_val == max_val:
        return IsolationTreeNode(size=n_samples, is_leaf=True)

    split_value = random.uniform(min_val, max_val)

    # Partition data
    left_data = [row for row in data if row[feature_idx] < split_value]
    right_data = [row for row in data if row[feature_idx] >= split_value]

    # Edge case: if one side is empty, just make a leaf
    # (shouldn't happen often with uniform split in [min, max])
    if not left_data or not right_data:
        return IsolationTreeNode(size=n_samples, is_leaf=True)

    node = IsolationTreeNode(
        split_feature=feature_idx,
        split_value=split_value,
    )
    node.left = _build_isolation_tree(left_data, current_depth + 1, max_depth)
    node.right = _build_isolation_tree(right_data, current_depth + 1, max_depth)
    return node


def _path_length(point: list[float], node: IsolationTreeNode, depth: int) -> float:
    """Compute the path length for a point traversing the tree.

    If we reach a leaf with size > 1, we add the expected additional path length
    c(size) — this accounts for the fact that we stopped early due to max depth
    and the remaining points would have taken c(size) more splits on average.
    """
    if node.is_leaf:
        # Add the expected path length for the remaining points at this leaf
        return depth + _c(node.size)

    # Traverse based on the split
    if point[node.split_feature] < node.split_value:
        return _path_length(point, node.left, depth + 1)
    else:
        return _path_length(point, node.right, depth + 1)


def _c(n: int) -> float:
    """Average path length of unsuccessful search in BST.

    This is the normalization constant from the Isolation Forest paper (Liu et al. 2008).
    It converts raw path lengths into a score that's comparable across different
    dataset sizes.

    c(n) = 2*H(n-1) - 2*(n-1)/n
    where H(i) = ln(i) + 0.5772156649 (Euler-Mascheroni constant)

    Special cases: c(1) = 0, c(2) = 1
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    h = math.log(n - 1) + 0.5772156649  # Euler-Mascheroni constant
    return 2.0 * h - 2.0 * (n - 1) / n


class IsolationForest:
    """Isolation Forest for unsupervised anomaly detection.

    Parameters:
        n_trees: Number of isolation trees in the ensemble. More trees = more
                 stable scores, but diminishing returns past ~100.
        sample_size: Subsample size for each tree. The original paper recommends
                     256 — large enough to capture structure, small enough to
                     ensure diversity between trees.
        contamination: Expected fraction of anomalies. Used to set the threshold
                       automatically after fitting. If None, user must set threshold.
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
        self.threshold: float = 0.5  # will be set during fit

    def fit(self, data: list[list[float]]) -> "IsolationForest":
        """Build the forest on training data (assumed mostly normal).

        Each tree gets a random subsample of size min(sample_size, len(data)).
        Max depth is ceil(log2(sample_size)) because that's the average tree
        height for sample_size points — going deeper adds no useful signal.
        """
        n = len(data)
        actual_sample_size = min(self.sample_size, n)
        max_depth = int(math.ceil(math.log2(max(actual_sample_size, 2))))

        self.trees = []
        for _ in range(self.n_trees):
            # Subsample without replacement
            sample = random.sample(data, actual_sample_size)
            tree = _build_isolation_tree(sample, current_depth=0, max_depth=max_depth)
            self.trees.append(tree)

        # Set threshold based on contamination rate:
        # Score all training points and pick the (1 - contamination) quantile
        scores = [self.anomaly_score(point) for point in data]
        scores.sort()
        idx = int(len(scores) * (1 - self.contamination))
        self.threshold = scores[min(idx, len(scores) - 1)]

        return self

    def anomaly_score(self, point: list[float]) -> float:
        """Compute the anomaly score for a single point.

        s(x, n) = 2^(-E(h(x)) / c(n))

        Returns a value in [0, 1]:
          - Close to 1.0 → strong anomaly (short average path)
          - Close to 0.5 → normal (average path length)
          - Close to 0.0 → definite inlier (unusually long path)
        """
        if not self.trees:
            raise RuntimeError("Must call fit() before scoring")

        # Average path length across all trees
        avg_path = sum(
            _path_length(point, tree, depth=0) for tree in self.trees
        ) / len(self.trees)

        # Normalize by c(sample_size) and convert to score
        cn = _c(self.sample_size)
        if cn == 0:
            return 0.5  # degenerate case with tiny sample
        score = 2.0 ** (-avg_path / cn)
        return score

    def predict(self, point: list[float]) -> tuple[float, bool]:
        """Score a point and classify it as anomaly or normal.

        Returns:
            (anomaly_score, is_anomaly)
        """
        score = self.anomaly_score(point)
        return score, score >= self.threshold


# ─────────────────────────────────────────────────────────────────────────────
# Blockchain Audit Ledger
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """A single entry in the audit chain.

    Contains the detection decision, the evidence (data point and score),
    and the cryptographic link to the previous record.
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
        """SHA-256 hash of (previous_hash + record contents).

        The hash covers ALL fields except record_hash itself. This means
        changing any field — timestamp, score, decision, or even the link
        to the previous record — will produce a different hash.
        """
        payload = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data_point": self.data_point,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "threshold": self.threshold,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data_point": [round(v, 4) for v in self.data_point],
            "anomaly_score": round(self.anomaly_score, 6),
            "is_anomaly": self.is_anomaly,
            "threshold": round(self.threshold, 6),
            "previous_hash": self.previous_hash[:16] + "...",
            "record_hash": self.record_hash[:16] + "...",
        }


class AuditLedger:
    """Immutable, hash-chained audit log for detection decisions.

    Each record's hash depends on the previous record's hash, forming a chain.
    Modifying any record in the middle invalidates all subsequent hashes.

    This is the same fundamental structure as a blockchain, but without
    consensus mechanisms (single writer, not distributed).
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
        previous_hash = self.chain[-1].record_hash if self.chain else "0" * 64
        record = AuditRecord(
            index=len(self.chain),
            timestamp=time.time(),
            data_point=data_point,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            threshold=threshold,
            previous_hash=previous_hash,
        )
        record.record_hash = record.compute_hash()
        self.chain.append(record)
        return record

    def verify_integrity(self) -> tuple[bool, Optional[int]]:
        """Verify the entire chain's integrity.

        Recomputes every hash from scratch and checks:
          1. Each record's hash matches its contents
          2. Each record's previous_hash matches the prior record's hash
          3. The genesis record has a zeroed previous_hash

        Returns:
            (is_valid, first_broken_index) — first_broken_index is None if valid
        """
        for i, record in enumerate(self.chain):
            # Check previous hash link
            expected_prev = self.chain[i - 1].record_hash if i > 0 else "0" * 64
            if record.previous_hash != expected_prev:
                return False, i

            # Check record's own hash
            if record.compute_hash() != record.record_hash:
                return False, i

        return True, None

    def tamper_with(self, index: int, new_score: float) -> None:
        """Intentionally tamper with a record (for demonstration only).

        Modifies the anomaly_score of a record WITHOUT updating hashes.
        This simulates an attacker trying to alter historical records.
        """
        if 0 <= index < len(self.chain):
            self.chain[index].anomaly_score = new_score
            # Note: we deliberately do NOT recompute hashes — that's the point


# ─────────────────────────────────────────────────────────────────────────────
# Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_normal_data(
    n_points: int,
    n_features: int,
    seed: int = 42,
) -> list[list[float]]:
    """Generate normally-distributed data for training.

    Uses multiple clusters to create a realistic multimodal distribution
    rather than a single Gaussian blob. This tests whether the isolation
    forest can handle non-trivial density shapes.
    """
    rng = random.Random(seed)
    data = []

    # Create 3 clusters with different centers
    centers = [
        [rng.gauss(0, 0.5) for _ in range(n_features)] for _ in range(3)
    ]

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
    """Generate a stream of data points with injected anomalies.

    Normal points come from the same distribution as training data.
    Anomalies are generated far from the cluster centers — they should
    have high isolation forest scores.

    Returns list of (point, is_truly_anomaly) for evaluation.
    """
    rng = random.Random(seed)
    stream = []

    centers = [
        [rng.gauss(0, 0.5) for _ in range(n_features)] for _ in range(3)
    ]

    for _ in range(n_points):
        if rng.random() < anomaly_fraction:
            # Anomaly: far from any cluster center
            # Scale is 3-5x the normal spread
            point = [rng.gauss(0, 3.0) + rng.choice([-3, 3]) for _ in range(n_features)]
            stream.append((point, True))
        else:
            center = rng.choice(centers)
            point = [center[j] + rng.gauss(0, 0.3) for j in range(n_features)]
            stream.append((point, False))

    return stream


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: list[bool],
    ground_truth: list[bool],
) -> dict[str, float]:
    """Compute precision, recall, F1, and accuracy for anomaly detection.

    In anomaly detection, recall (sensitivity) is usually more important
    than precision — missing a real anomaly is worse than a false alarm.
    """
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    print("=" * 70)
    print("AI ANOMALY DETECTION WITH BLOCKCHAIN AUDIT TRAIL")
    print("=" * 70)

    # ── Step 1: Generate training data ────────────────────────────────────
    N_FEATURES = 3
    TRAIN_SIZE = 500
    STREAM_SIZE = 100

    print(f"\n[1] Generating {TRAIN_SIZE} normal training points ({N_FEATURES}D)...")
    train_data = generate_normal_data(TRAIN_SIZE, N_FEATURES, seed=42)
    print(f"    Sample point: {[round(v, 3) for v in train_data[0]]}")

    # ── Step 2: Train Isolation Forest ────────────────────────────────────
    print(f"\n[2] Training Isolation Forest (100 trees, subsample=256)...")
    forest = IsolationForest(n_trees=100, sample_size=256, contamination=0.1)
    forest.fit(train_data)
    print(f"    Anomaly threshold: {forest.threshold:.6f}")

    # Show score distribution on training data
    train_scores = [forest.anomaly_score(p) for p in train_data[:20]]
    print(f"    Training scores (first 20): min={min(train_scores):.4f}, "
          f"max={max(train_scores):.4f}, mean={sum(train_scores)/len(train_scores):.4f}")

    # ── Step 3: Initialize audit ledger ───────────────────────────────────
    print(f"\n[3] Initializing hash-chained audit ledger...")
    ledger = AuditLedger()

    # ── Step 4: Stream data through the pipeline ──────────────────────────
    print(f"\n[4] Processing {STREAM_SIZE} streaming data points...")
    stream = generate_stream_with_anomalies(STREAM_SIZE, N_FEATURES, anomaly_fraction=0.15, seed=123)

    predictions = []
    ground_truth = []
    detected_anomalies = []

    for i, (point, is_truly_anomaly) in enumerate(stream):
        # Score and classify
        score, is_predicted_anomaly = forest.predict(point)

        # Record on audit ledger
        record = ledger.append(
            data_point=point,
            anomaly_score=score,
            is_anomaly=is_predicted_anomaly,
            threshold=forest.threshold,
        )

        predictions.append(is_predicted_anomaly)
        ground_truth.append(is_truly_anomaly)

        if is_predicted_anomaly:
            detected_anomalies.append((i, score, is_truly_anomaly))

        # Print first few and any anomalies
        if i < 5 or is_predicted_anomaly:
            label = "ANOMALY" if is_predicted_anomaly else "normal"
            truth = "  (true anomaly)" if is_truly_anomaly else ""
            if i < 5 or len(detected_anomalies) <= 10:
                print(f"    [{i:3d}] score={score:.4f} → {label}{truth}")

    # ── Step 5: Evaluate detection quality ────────────────────────────────
    print(f"\n[5] Detection metrics:")
    metrics = compute_metrics(predictions, ground_truth)
    true_anomaly_count = sum(1 for _, g in stream if g)
    print(f"    True anomalies in stream: {true_anomaly_count}")
    print(f"    Detected anomalies:       {len(detected_anomalies)}")
    print(f"    Precision: {metrics['precision']:.3f}  (of flagged, how many were real)")
    print(f"    Recall:    {metrics['recall']:.3f}  (of real, how many were caught)")
    print(f"    F1 Score:  {metrics['f1']:.3f}")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}")

    # ── Step 6: Verify audit ledger integrity ─────────────────────────────
    print(f"\n[6] Verifying audit ledger integrity...")
    print(f"    Chain length: {len(ledger.chain)} records")
    is_valid, broken_idx = ledger.verify_integrity()
    print(f"    Integrity check: {'PASSED' if is_valid else 'FAILED'}")

    # Show a few records
    print(f"\n    Sample audit records:")
    for record in ledger.chain[:3]:
        print(f"      {json.dumps(record.to_dict(), indent=6)}")

    # ── Step 7: Demonstrate tamper detection ──────────────────────────────
    print(f"\n[7] Demonstrating tamper detection...")
    target_idx = 5
    original_score = ledger.chain[target_idx].anomaly_score
    print(f"    Original record #{target_idx} score: {original_score:.6f}")
    print(f"    Tampering: changing score to 0.000001...")
    ledger.tamper_with(target_idx, 0.000001)

    is_valid, broken_idx = ledger.verify_integrity()
    print(f"    Integrity check after tampering: {'PASSED' if is_valid else 'FAILED'}")
    if broken_idx is not None:
        print(f"    First broken record: index {broken_idx}")
        print(f"    >> Tamper detected! The hash chain proves the record was modified.")

    # Restore for clean state
    ledger.chain[target_idx].anomaly_score = original_score
    ledger.chain[target_idx].record_hash = ledger.chain[target_idx].compute_hash()

    print(f"\n{'=' * 70}")
    print("Pipeline complete. Every detection decision is cryptographically")
    print("linked in an immutable audit chain.")
    print(f"{'=' * 70}")
