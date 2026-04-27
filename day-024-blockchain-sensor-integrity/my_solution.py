"""
Day 024: Blockchain-Verified Sensor Data Pipeline — Your Implementation

Build a system that:
1. Simulates robot sensor readings (temperature, pressure, acceleration)
2. Chains each reading into a tamper-evident blockchain ledger
3. Detects anomalous readings using rolling z-score
4. Proves the chain catches tampering

Hint: Think about WHY each block needs the previous block's hash.
Hint: The z-score tells you "how many standard deviations from normal is this?"
Hint: Keep separate rolling windows per sensor — their scales are different.
"""

import hashlib
import json
import time
import random
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# ============================================================================
# Part 1: Sensor Simulator
# ============================================================================

@dataclass
class SensorReading:
    """A single sensor measurement with metadata."""
    timestamp: float
    sensor_id: str
    value: float
    unit: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class SensorSimulator:
    """
    Generates realistic sensor data streams with configurable noise and anomalies.

    Uses Gaussian noise around a base value with occasional spikes and drift.

    Args:
        sensor_id: Unique identifier for this sensor
        base_value: The "normal" center value for readings
        noise_std: Standard deviation of Gaussian noise
        unit: Unit of measurement
        anomaly_probability: Chance of generating an anomalous reading (0-1)
        spike_magnitude: How many std devs an anomaly spike should be
        drift_rate: How fast the baseline drifts
    """

    def __init__(
        self,
        sensor_id: str,
        base_value: float,
        noise_std: float,
        unit: str,
        anomaly_probability: float = 0.05,
        spike_magnitude: float = 5.0,
        drift_rate: float = 0.01,
    ):
        self.sensor_id = sensor_id
        self.base_value = base_value
        self.noise_std = noise_std
        self.unit = unit
        self.anomaly_probability = anomaly_probability
        self.spike_magnitude = spike_magnitude
        self.drift_rate = drift_rate
        self._drift_offset = 0.0
        self._reading_count = 0

    def generate_reading(self) -> Tuple[SensorReading, bool]:
        """
        Generate a single sensor reading.

        Returns:
            Tuple of (reading, is_anomaly) — is_anomaly is ground truth.

        Hint: Add Gaussian noise to base_value + drift. Then with some probability,
        add a large spike. Use random.gauss() for noise and math.sin() for drift.
        """
        raise NotImplementedError("TODO: implement this")


# ============================================================================
# Part 2: Blockchain-Style Hash Chaining
# ============================================================================

@dataclass
class Block:
    """
    A single block in the sensor data chain.

    Hint: The hash must include ALL fields (index, timestamp, data, previous_hash,
    nonce). Use json.dumps with sort_keys=True for deterministic serialization.
    """
    index: int
    timestamp: float
    sensor_data: dict
    is_anomaly: bool
    previous_hash: str
    nonce: int = 0
    hash: str = ""

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of block contents (everything except self.hash).

        Hint: Serialize the block fields to a JSON string, then hash with SHA-256.
        Don't include the hash field itself — that's what you're computing!
        """
        raise NotImplementedError("TODO: implement this")


class SensorBlockchain:
    """
    A blockchain-style ledger for sensor readings.

    Hint: The genesis block starts with previous_hash = "0" * 64.
    Each new block's previous_hash should be the last block's hash.
    Verification recomputes hashes and checks the chain links.
    """

    def __init__(self):
        self.chain: List[Block] = []
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """Create the first block with no real data."""
        raise NotImplementedError("TODO: implement this")

    def add_reading(self, reading: SensorReading, is_anomaly: bool) -> Block:
        """
        Add a sensor reading to the chain.

        Hint: Create a new block, set its previous_hash to the last block's hash,
        compute and store its own hash, then append to the chain.
        """
        raise NotImplementedError("TODO: implement this")

    def verify_integrity(self) -> Tuple[bool, Optional[int]]:
        """
        Verify the entire chain's integrity.

        Returns:
            (is_valid, broken_at) — broken_at is the index where the chain breaks.

        Hint: For each block (starting at 1), check two things:
        1. Does recomputing the hash match the stored hash?
        2. Does previous_hash match the prior block's actual hash?
        """
        raise NotImplementedError("TODO: implement this")

    def get_anomalies(self) -> List[Block]:
        """Return all blocks flagged as anomalies."""
        raise NotImplementedError("TODO: implement this")


# ============================================================================
# Part 3: Anomaly Detection
# ============================================================================

class RollingZScoreDetector:
    """
    Streaming anomaly detector using rolling z-score.

    z = (x - mean) / std

    Hint: Keep a sliding window per sensor. When the window is full, remove
    the oldest value before adding the new one. Need at least ~5 values
    before the statistics are meaningful.
    """

    def __init__(self, window_size: int = 50, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.windows: dict[str, list[float]] = {}

    def score(self, reading: SensorReading) -> Tuple[float, bool]:
        """
        Score a reading and determine if it's anomalous.

        Returns:
            (z_score, is_anomaly) — z_score is 0.0 during warmup.

        Hint: Compute mean and std of the window, then z = (value - mean) / std.
        Flag as anomaly if |z| > threshold.
        """
        raise NotImplementedError("TODO: implement this")


# ============================================================================
# Part 4: Pipeline Orchestrator
# ============================================================================

class SensorDataPipeline:
    """
    Integrates sensor simulation, anomaly detection, and blockchain integrity.

    Hint: Process readings in order: generate -> score -> record on chain.
    Track true positives, false positives, false negatives for evaluation.
    """

    def __init__(self, sensors: List[SensorSimulator], detector: RollingZScoreDetector):
        self.sensors = sensors
        self.detector = detector
        self.blockchain = SensorBlockchain()
        self.total_readings = 0
        self.true_anomalies = 0
        self.detected_anomalies = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def process_readings(self, num_readings_per_sensor: int) -> None:
        """
        Run the pipeline for num_readings_per_sensor readings per sensor.

        Hint: Loop through readings, for each sensor: generate, score, record.
        Update confusion matrix statistics using ground truth vs detected.
        """
        raise NotImplementedError("TODO: implement this")

    def get_detection_stats(self) -> dict:
        """
        Compute precision, recall, and F1 score.

        Hint:
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - F1 = 2 * P * R / (P + R)
        """
        raise NotImplementedError("TODO: implement this")


# ============================================================================
# Part 5: Test your implementation
# ============================================================================

if __name__ == "__main__":
    random.seed(42)

    print("Setting up sensors...")
    sensors = [
        SensorSimulator("TEMP-001", 22.0, 0.5, "celsius", 0.05, 5.0),
        SensorSimulator("PRES-001", 101.3, 0.3, "kPa", 0.05, 4.0),
        SensorSimulator("ACCEL-001", 0.0, 0.1, "m/s^2", 0.08, 6.0),
    ]

    detector = RollingZScoreDetector(window_size=50, threshold=3.0)
    pipeline = SensorDataPipeline(sensors, detector)

    print("Processing 100 readings per sensor...")
    pipeline.process_readings(100)

    print(f"Chain length: {len(pipeline.blockchain.chain)} blocks")

    stats = pipeline.get_detection_stats()
    print("\nDetection stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nVerifying chain integrity...")
    valid, broken = pipeline.blockchain.verify_integrity()
    print(f"Chain valid: {valid}")

    # Tamper test
    print("\nTampering with block #50...")
    pipeline.blockchain.chain[50].sensor_data["value"] += 100
    valid, broken = pipeline.blockchain.verify_integrity()
    print(f"Chain valid after tampering: {valid}")
    print(f"Broken at block: {broken}")
