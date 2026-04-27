"""
Day 024: Blockchain-Verified Sensor Data Pipeline

A complete integration of robot sensor simulation, blockchain-style hash chaining
for data integrity, and statistical anomaly detection. This demonstrates how
autonomous systems can produce tamper-evident, quality-checked data streams.

Architecture:
    SensorSimulator -> AnomalyDetector -> SensorBlockchain -> Verification
    (produces data)   (scores data)      (chains data)       (proves integrity)
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

    The simulator produces Gaussian noise around a base value, with two types
    of injected anomalies:
    - Spikes: sudden large deviations (e.g., sensor glitch or impact)
    - Drift: gradual baseline shift (e.g., sensor degradation or environmental change)

    Why Gaussian noise? Most physical measurements follow a normal distribution
    due to the Central Limit Theorem — the sum of many small independent error
    sources converges to Gaussian. This makes our z-score detector well-matched
    to the noise model.
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

        # Internal state for drift simulation
        self._drift_offset = 0.0
        self._reading_count = 0

    def generate_reading(self) -> Tuple[SensorReading, bool]:
        """
        Generate a single sensor reading.

        Returns:
            Tuple of (reading, is_anomaly) where is_anomaly indicates if this
            reading was intentionally generated as anomalous.

        The anomaly flag is ground truth for evaluating our detector — in production,
        you wouldn't have this label, which is exactly why you need the detector.
        """
        self._reading_count += 1

        # Gradual drift: simulates sensor degradation or slow environmental change
        # We use a sine wave to make drift periodic (like daily temperature cycles)
        self._drift_offset = self.drift_rate * math.sin(self._reading_count * 0.05)

        # Base value + drift + Gaussian noise
        value = self.base_value + self._drift_offset + random.gauss(0, self.noise_std)

        is_anomaly = False

        # Inject anomalies with configured probability
        if random.random() < self.anomaly_probability:
            is_anomaly = True
            # Randomly choose spike direction
            spike = self.spike_magnitude * self.noise_std * random.choice([-1, 1])
            value += spike

        reading = SensorReading(
            timestamp=time.time(),
            sensor_id=self.sensor_id,
            value=round(value, 4),
            unit=self.unit,
        )

        return reading, is_anomaly


# ============================================================================
# Part 2: Blockchain-Style Hash Chaining
# ============================================================================

@dataclass
class Block:
    """
    A single block in the sensor data chain.

    Each block contains:
    - The sensor data (reading + anomaly flag)
    - A link to the previous block (previous_hash)
    - Its own hash computed from all contents

    The hash function is SHA-256, chosen for its collision resistance.
    We don't use proof-of-work (nonce is always 0) because our chain has a
    single trusted producer. In multi-robot scenarios, you'd add consensus.
    """
    index: int
    timestamp: float
    sensor_data: dict  # The sensor reading as a dictionary
    is_anomaly: bool
    previous_hash: str
    nonce: int = 0
    hash: str = ""

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of block contents.

        We serialize to a sorted JSON string for deterministic hashing.
        Without sorting keys, the same dictionary could produce different
        JSON strings (and thus different hashes) depending on insertion order.
        """
        block_content = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "sensor_data": self.sensor_data,
                "is_anomaly": self.is_anomaly,
                "previous_hash": self.previous_hash,
                "nonce": self.nonce,
            },
            sort_keys=True,
        )
        return hashlib.sha256(block_content.encode()).hexdigest()


class SensorBlockchain:
    """
    A blockchain-style ledger for sensor readings.

    The chain starts with a genesis block (index 0, no data) and each
    subsequent block contains one sensor reading plus its anomaly flag.

    Key properties:
    - Append-only: new blocks are added to the end
    - Tamper-evident: modifying any block breaks the hash chain
    - Verifiable: anyone can check integrity by recomputing hashes

    This is NOT a distributed blockchain — it's a local integrity chain.
    The difference matters: a distributed blockchain resists a malicious
    writer (via consensus), while our chain only detects post-hoc tampering.
    """

    def __init__(self):
        self.chain: List[Block] = []
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """
        Create the first block in the chain.

        The genesis block has no meaningful data — it exists solely to give
        the first real block a previous_hash to reference. Its hash is the
        anchor for the entire chain's integrity.
        """
        genesis = Block(
            index=0,
            timestamp=time.time(),
            sensor_data={"genesis": True},
            is_anomaly=False,
            previous_hash="0" * 64,  # Convention: genesis points to all zeros
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)

    def add_reading(self, reading: SensorReading, is_anomaly: bool) -> Block:
        """
        Add a sensor reading to the chain.

        Creates a new block containing the reading data and the anomaly flag,
        links it to the previous block's hash, computes the new block's hash,
        and appends it to the chain.

        Returns the newly created block.
        """
        previous_block = self.chain[-1]

        new_block = Block(
            index=len(self.chain),
            timestamp=reading.timestamp,
            sensor_data=reading.to_dict(),
            is_anomaly=is_anomaly,
            previous_hash=previous_block.hash,
        )
        new_block.hash = new_block.compute_hash()
        self.chain.append(new_block)
        return new_block

    def verify_integrity(self) -> Tuple[bool, Optional[int]]:
        """
        Verify the entire chain's integrity.

        Checks two things for each block:
        1. The block's stored hash matches its recomputed hash (data wasn't changed)
        2. The block's previous_hash matches the prior block's hash (chain is intact)

        Returns:
            (is_valid, broken_at) — if invalid, broken_at is the index where
            the chain first breaks. This tells you exactly which reading was
            tampered with.

        Time complexity: O(n) where n is chain length.
        Space complexity: O(1) — we only need the current and previous block.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check 1: Has this block's data been modified?
            if current.hash != current.compute_hash():
                return False, i

            # Check 2: Is this block properly linked to the previous one?
            if current.previous_hash != previous.hash:
                return False, i

        return True, None

    def get_anomalies(self) -> List[Block]:
        """Return all blocks flagged as anomalies."""
        return [b for b in self.chain[1:] if b.is_anomaly]


# ============================================================================
# Part 3: Anomaly Detection
# ============================================================================

class RollingZScoreDetector:
    """
    Streaming anomaly detector using rolling z-score.

    Maintains a sliding window of recent values and flags readings whose
    z-score exceeds a threshold.

    z = (x - mean) / std

    For normally distributed data:
    - |z| > 2: ~4.6% of readings (too many false positives)
    - |z| > 3: ~0.3% of readings (good default)
    - |z| > 4: ~0.006% of readings (very conservative)

    Why rolling instead of global? Sensor readings drift over time.
    A global mean would flag normal daily temperature variation as anomalous.
    The rolling window adapts to slow changes while catching sudden spikes.

    Why not exponential moving average (EMA)? EMA is O(1) space but gives
    more weight to recent values, which can mask anomalies in rapidly changing
    environments. The sliding window treats all recent values equally, which
    is more robust for our use case.
    """

    def __init__(self, window_size: int = 50, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.windows: dict[str, list[float]] = {}  # Per-sensor windows

    def score(self, reading: SensorReading) -> Tuple[float, bool]:
        """
        Score a reading and determine if it's anomalous.

        Returns:
            (z_score, is_anomaly) — z_score is 0.0 during warmup period
            (not enough data to compute meaningful statistics).

        We maintain separate windows per sensor_id because different sensors
        have different value distributions. Mixing them would make the
        statistics meaningless.
        """
        sensor_id = reading.sensor_id
        value = reading.value

        # Initialize window for new sensors
        if sensor_id not in self.windows:
            self.windows[sensor_id] = []

        window = self.windows[sensor_id]
        window.append(value)

        # Keep window at configured size
        if len(window) > self.window_size:
            window.pop(0)

        # Need at least 5 readings to compute meaningful statistics
        # With fewer, the standard deviation is unreliable
        if len(window) < 5:
            return 0.0, False

        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 1e-10  # Avoid division by zero

        z_score = (value - mean) / std
        is_anomaly = abs(z_score) > self.threshold

        return round(z_score, 4), is_anomaly


# ============================================================================
# Part 4: Pipeline Orchestrator
# ============================================================================

class SensorDataPipeline:
    """
    Integrates sensor simulation, anomaly detection, and blockchain integrity.

    The pipeline processes readings in this order:
    1. Sensor generates a reading
    2. Anomaly detector scores it
    3. Reading + score is recorded in the blockchain
    4. Summary statistics are updated

    This ordering matters: we score BEFORE recording so the anomaly flag
    is part of the immutable record. If we scored after, an attacker could
    modify a reading and its anomaly flag independently.
    """

    def __init__(self, sensors: List[SensorSimulator], detector: RollingZScoreDetector):
        self.sensors = sensors
        self.detector = detector
        self.blockchain = SensorBlockchain()

        # Statistics
        self.total_readings = 0
        self.true_anomalies = 0  # Ground truth injected anomalies
        self.detected_anomalies = 0  # What our detector flagged
        self.true_positives = 0  # Correctly detected anomalies
        self.false_positives = 0  # Normal readings flagged as anomalous
        self.false_negatives = 0  # Anomalies missed by detector

    def process_readings(self, num_readings_per_sensor: int) -> None:
        """
        Run the pipeline for a specified number of readings per sensor.

        Processes readings round-robin across sensors to simulate
        concurrent multi-sensor operation.
        """
        for _ in range(num_readings_per_sensor):
            for sensor in self.sensors:
                reading, ground_truth_anomaly = sensor.generate_reading()
                z_score, detected_anomaly = self.detector.score(reading)

                # Record in blockchain — the detected flag (not ground truth)
                # is what goes on-chain, because in production we don't have
                # ground truth labels
                self.blockchain.add_reading(reading, detected_anomaly)

                # Update statistics using ground truth for evaluation
                self.total_readings += 1
                if ground_truth_anomaly:
                    self.true_anomalies += 1
                if detected_anomaly:
                    self.detected_anomalies += 1

                if ground_truth_anomaly and detected_anomaly:
                    self.true_positives += 1
                elif not ground_truth_anomaly and detected_anomaly:
                    self.false_positives += 1
                elif ground_truth_anomaly and not detected_anomaly:
                    self.false_negatives += 1

    def get_detection_stats(self) -> dict:
        """
        Compute precision, recall, and F1 score for the anomaly detector.

        Precision = TP / (TP + FP) — "of the readings we flagged, how many were real?"
        Recall = TP / (TP + FN) — "of the real anomalies, how many did we catch?"
        F1 = harmonic mean of precision and recall

        High precision, low recall = conservative (misses anomalies but rarely cries wolf)
        Low precision, high recall = aggressive (catches everything but many false alarms)
        """
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0
            else 0.0
        )
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "total_readings": self.total_readings,
            "true_anomalies": self.true_anomalies,
            "detected_anomalies": self.detected_anomalies,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }


# ============================================================================
# Part 5: Main — Demonstration
# ============================================================================

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)

    print("=" * 70)
    print("  Blockchain-Verified Sensor Data Pipeline")
    print("=" * 70)

    # --- Step 1: Set up sensors ---
    print("\n[1] Setting up sensors...")
    sensors = [
        SensorSimulator(
            sensor_id="TEMP-001",
            base_value=22.0,   # Room temperature in Celsius
            noise_std=0.5,
            unit="celsius",
            anomaly_probability=0.05,
            spike_magnitude=5.0,
        ),
        SensorSimulator(
            sensor_id="PRES-001",
            base_value=101.3,  # Standard atmospheric pressure in kPa
            noise_std=0.3,
            unit="kPa",
            anomaly_probability=0.05,
            spike_magnitude=4.0,
        ),
        SensorSimulator(
            sensor_id="ACCEL-001",
            base_value=0.0,    # At rest, acceleration ~0
            noise_std=0.1,
            unit="m/s^2",
            anomaly_probability=0.08,  # More frequent — accelerometer is noisier
            spike_magnitude=6.0,
        ),
    ]
    print(f"   Created {len(sensors)} sensors: {[s.sensor_id for s in sensors]}")

    # --- Step 2: Set up anomaly detector ---
    print("\n[2] Setting up anomaly detector...")
    detector = RollingZScoreDetector(window_size=50, threshold=3.0)
    print(f"   Window size: {detector.window_size}, Z-score threshold: {detector.threshold}")

    # --- Step 3: Run pipeline ---
    print("\n[3] Running pipeline (100 readings per sensor)...")
    pipeline = SensorDataPipeline(sensors, detector)
    pipeline.process_readings(num_readings_per_sensor=100)

    chain_length = len(pipeline.blockchain.chain)
    print(f"   Blockchain now has {chain_length} blocks (including genesis)")

    # --- Step 4: Show sample blocks ---
    print("\n[4] Sample blocks from the chain:")
    for block in pipeline.blockchain.chain[1:4]:  # First 3 data blocks
        print(f"   Block #{block.index}:")
        print(f"     Sensor: {block.sensor_data['sensor_id']}")
        print(f"     Value:  {block.sensor_data['value']} {block.sensor_data['unit']}")
        print(f"     Anomaly: {block.is_anomaly}")
        print(f"     Hash:   {block.hash[:32]}...")
        print(f"     Prev:   {block.previous_hash[:32]}...")

    # --- Step 5: Detection statistics ---
    print("\n[5] Anomaly detection performance:")
    stats = pipeline.get_detection_stats()
    for key, value in stats.items():
        print(f"   {key:>20}: {value}")

    # --- Step 6: Verify chain integrity ---
    print("\n[6] Verifying blockchain integrity...")
    is_valid, broken_at = pipeline.blockchain.verify_integrity()
    print(f"   Chain valid: {is_valid}")

    # --- Step 7: Tamper detection demo ---
    print("\n[7] Tamper detection demo:")
    print("   Modifying block #50's sensor value...")

    # Save original value for display
    original_value = pipeline.blockchain.chain[50].sensor_data["value"]
    tampered_value = original_value + 100.0

    # Tamper with the data (but don't recompute the hash — an attacker might
    # try this, hoping no one checks)
    pipeline.blockchain.chain[50].sensor_data["value"] = tampered_value
    print(f"   Original value: {original_value}")
    print(f"   Tampered value: {tampered_value}")

    # Now verify — should catch the tampering
    is_valid, broken_at = pipeline.blockchain.verify_integrity()
    print(f"   Chain valid after tampering: {is_valid}")
    print(f"   Chain broken at block: {broken_at}")

    # Even if the attacker recomputes block 50's hash...
    print("\n   Attacker recomputes block 50's hash...")
    pipeline.blockchain.chain[50].hash = pipeline.blockchain.chain[50].compute_hash()

    # ...block 51's previous_hash still points to the OLD hash
    is_valid, broken_at = pipeline.blockchain.verify_integrity()
    print(f"   Chain valid after hash fix: {is_valid}")
    print(f"   Chain broken at block: {broken_at}")
    print("   -> The chain catches it! Block 51's previous_hash doesn't match.")

    # --- Step 8: List detected anomalies ---
    print("\n[8] Anomalous readings on-chain:")
    anomalies = pipeline.blockchain.get_anomalies()
    for block in anomalies[:5]:  # Show first 5
        print(
            f"   Block #{block.index}: {block.sensor_data['sensor_id']} "
            f"= {block.sensor_data['value']} {block.sensor_data['unit']}"
        )
    if len(anomalies) > 5:
        print(f"   ... and {len(anomalies) - 5} more")

    print(f"\n   Total anomalies recorded on-chain: {len(anomalies)}")
    print("\n" + "=" * 70)
    print("  Pipeline complete. Sensor data is chained, scored, and verifiable.")
    print("=" * 70)
