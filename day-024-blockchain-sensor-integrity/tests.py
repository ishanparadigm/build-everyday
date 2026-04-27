"""
Day 024: Tests for Blockchain-Verified Sensor Data Pipeline

Run with: python3 -m pytest tests.py -v
     or: python3 tests.py
"""

import unittest
import random
import time
from my_solution import (
    SensorReading,
    SensorSimulator,
    Block,
    SensorBlockchain,
    RollingZScoreDetector,
    SensorDataPipeline,
)


class TestSensorReading(unittest.TestCase):
    """Test sensor reading data class."""

    def test_to_dict(self):
        reading = SensorReading(1234.0, "TEMP-001", 22.5, "celsius")
        d = reading.to_dict()
        self.assertEqual(d["sensor_id"], "TEMP-001")
        self.assertEqual(d["value"], 22.5)
        self.assertEqual(d["unit"], "celsius")

    def test_to_json_deterministic(self):
        reading = SensorReading(1234.0, "TEMP-001", 22.5, "celsius")
        j1 = reading.to_json()
        j2 = reading.to_json()
        self.assertEqual(j1, j2, "JSON serialization should be deterministic")


class TestSensorSimulator(unittest.TestCase):
    """Test sensor data generation."""

    def test_generates_readings(self):
        sim = SensorSimulator("TEMP", 22.0, 0.5, "celsius")
        reading, is_anomaly = sim.generate_reading()
        self.assertIsInstance(reading, SensorReading)
        self.assertEqual(reading.sensor_id, "TEMP")
        self.assertEqual(reading.unit, "celsius")

    def test_values_near_base(self):
        """Most readings should be near the base value (within 5 std devs)."""
        random.seed(42)
        sim = SensorSimulator("TEMP", 22.0, 0.5, "celsius", anomaly_probability=0.0)
        values = [sim.generate_reading()[0].value for _ in range(100)]
        for v in values:
            self.assertAlmostEqual(v, 22.0, delta=5.0)

    def test_anomalies_are_generated(self):
        """With high anomaly probability, anomalies should appear."""
        random.seed(42)
        sim = SensorSimulator("TEMP", 22.0, 0.5, "celsius", anomaly_probability=0.5)
        anomaly_count = sum(sim.generate_reading()[1] for _ in range(100))
        self.assertGreater(anomaly_count, 10, "Should generate anomalies")


class TestBlock(unittest.TestCase):
    """Test block hash computation."""

    def test_compute_hash_deterministic(self):
        block = Block(1, 1234.0, {"value": 22.5}, False, "0" * 64)
        h1 = block.compute_hash()
        h2 = block.compute_hash()
        self.assertEqual(h1, h2)

    def test_different_data_different_hash(self):
        b1 = Block(1, 1234.0, {"value": 22.5}, False, "0" * 64)
        b2 = Block(1, 1234.0, {"value": 22.6}, False, "0" * 64)
        self.assertNotEqual(b1.compute_hash(), b2.compute_hash())

    def test_hash_is_sha256_hex(self):
        block = Block(1, 1234.0, {"value": 22.5}, False, "0" * 64)
        h = block.compute_hash()
        self.assertEqual(len(h), 64, "SHA-256 hex digest should be 64 chars")


class TestSensorBlockchain(unittest.TestCase):
    """Test blockchain integrity."""

    def test_genesis_block_created(self):
        bc = SensorBlockchain()
        self.assertEqual(len(bc.chain), 1)
        self.assertEqual(bc.chain[0].index, 0)

    def test_add_reading(self):
        bc = SensorBlockchain()
        reading = SensorReading(time.time(), "TEMP", 22.5, "celsius")
        block = bc.add_reading(reading, False)
        self.assertEqual(len(bc.chain), 2)
        self.assertEqual(block.index, 1)
        self.assertEqual(block.previous_hash, bc.chain[0].hash)

    def test_verify_valid_chain(self):
        bc = SensorBlockchain()
        for i in range(10):
            reading = SensorReading(time.time(), "TEMP", 22.0 + i * 0.1, "celsius")
            bc.add_reading(reading, False)
        valid, broken = bc.verify_integrity()
        self.assertTrue(valid)
        self.assertIsNone(broken)

    def test_detect_tampered_data(self):
        bc = SensorBlockchain()
        for i in range(10):
            reading = SensorReading(time.time(), "TEMP", 22.0 + i * 0.1, "celsius")
            bc.add_reading(reading, False)

        # Tamper with block 5
        bc.chain[5].sensor_data["value"] = 999.0
        valid, broken = bc.verify_integrity()
        self.assertFalse(valid)
        self.assertEqual(broken, 5)

    def test_detect_hash_recomputation_attack(self):
        """Even if attacker recomputes the tampered block's hash, next block breaks."""
        bc = SensorBlockchain()
        for i in range(10):
            reading = SensorReading(time.time(), "TEMP", 22.0 + i * 0.1, "celsius")
            bc.add_reading(reading, False)

        # Tamper and recompute hash
        bc.chain[5].sensor_data["value"] = 999.0
        bc.chain[5].hash = bc.chain[5].compute_hash()

        valid, broken = bc.verify_integrity()
        self.assertFalse(valid)
        self.assertEqual(broken, 6, "Should break at block AFTER the tampered one")

    def test_get_anomalies(self):
        bc = SensorBlockchain()
        reading1 = SensorReading(time.time(), "TEMP", 22.5, "celsius")
        reading2 = SensorReading(time.time(), "TEMP", 99.0, "celsius")
        bc.add_reading(reading1, False)
        bc.add_reading(reading2, True)
        anomalies = bc.get_anomalies()
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].sensor_data["value"], 99.0)


class TestRollingZScoreDetector(unittest.TestCase):
    """Test anomaly detection."""

    def test_warmup_returns_zero(self):
        detector = RollingZScoreDetector(window_size=50, threshold=3.0)
        reading = SensorReading(time.time(), "TEMP", 22.0, "celsius")
        z, anomaly = detector.score(reading)
        self.assertEqual(z, 0.0)
        self.assertFalse(anomaly)

    def test_detects_spike(self):
        random.seed(42)
        detector = RollingZScoreDetector(window_size=50, threshold=3.0)
        # Feed normal readings to fill window
        for _ in range(60):
            reading = SensorReading(time.time(), "TEMP", 22.0 + random.gauss(0, 0.5), "celsius")
            detector.score(reading)
        # Now feed a massive spike
        spike_reading = SensorReading(time.time(), "TEMP", 50.0, "celsius")
        z, anomaly = detector.score(spike_reading)
        self.assertTrue(anomaly, "Should detect a massive spike")
        self.assertGreater(abs(z), 3.0)

    def test_normal_readings_not_flagged(self):
        random.seed(42)
        detector = RollingZScoreDetector(window_size=50, threshold=3.0)
        flagged = 0
        for _ in range(200):
            reading = SensorReading(time.time(), "TEMP", 22.0 + random.gauss(0, 0.5), "celsius")
            _, anomaly = detector.score(reading)
            if anomaly:
                flagged += 1
        # With threshold=3, expect <2% false positives on normal data
        self.assertLess(flagged / 200, 0.05, "False positive rate should be low")


class TestSensorDataPipeline(unittest.TestCase):
    """Test the full pipeline integration."""

    def test_pipeline_processes_readings(self):
        random.seed(42)
        sensors = [SensorSimulator("TEMP", 22.0, 0.5, "celsius", 0.05, 5.0)]
        detector = RollingZScoreDetector(window_size=50, threshold=3.0)
        pipeline = SensorDataPipeline(sensors, detector)
        pipeline.process_readings(50)
        self.assertEqual(pipeline.total_readings, 50)
        # Chain has genesis + 50 readings
        self.assertEqual(len(pipeline.blockchain.chain), 51)

    def test_detection_stats_valid(self):
        random.seed(42)
        sensors = [SensorSimulator("TEMP", 22.0, 0.5, "celsius", 0.1, 5.0)]
        detector = RollingZScoreDetector(window_size=50, threshold=3.0)
        pipeline = SensorDataPipeline(sensors, detector)
        pipeline.process_readings(200)
        stats = pipeline.get_detection_stats()
        self.assertIn("precision", stats)
        self.assertIn("recall", stats)
        self.assertIn("f1_score", stats)
        self.assertGreaterEqual(stats["precision"], 0.0)
        self.assertLessEqual(stats["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
