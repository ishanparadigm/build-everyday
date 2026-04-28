"""
Day 027: Proof of Work — Test Suite

Run with: python3 -m pytest tests.py -v
    or:  python3 tests.py
"""

import time
import unittest
from my_solution import (
    Block,
    ProofOfWork,
    analyze_mining_stats,
    simulate_attack,
    theoretical_attack_probability,
)


class TestBlock(unittest.TestCase):
    """Tests for the Block data structure."""

    def test_compute_hash_deterministic(self):
        """Same block contents should always produce the same hash."""
        b = Block(index=1, timestamp=1000.0, data="test", previous_hash="abc")
        h1 = b.compute_hash()
        h2 = b.compute_hash()
        self.assertEqual(h1, h2)

    def test_compute_hash_changes_with_nonce(self):
        """Different nonces must produce different hashes (avalanche effect)."""
        b = Block(index=1, timestamp=1000.0, data="test", previous_hash="abc")
        b.nonce = 0
        h1 = b.compute_hash()
        b.nonce = 1
        h2 = b.compute_hash()
        self.assertNotEqual(h1, h2)

    def test_compute_hash_is_sha256_hex(self):
        """Hash should be a 64-character hex string (256 bits)."""
        b = Block(index=0, timestamp=0.0, data="genesis", previous_hash="0" * 64)
        h = b.compute_hash()
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_hash_excludes_metadata(self):
        """mining_time and attempts should NOT affect the hash."""
        b1 = Block(index=1, timestamp=1000.0, data="test", previous_hash="abc",
                    mining_time=0.0, attempts=0)
        b2 = Block(index=1, timestamp=1000.0, data="test", previous_hash="abc",
                    mining_time=5.0, attempts=99999)
        self.assertEqual(b1.compute_hash(), b2.compute_hash())


class TestProofOfWork(unittest.TestCase):
    """Tests for the PoW mining engine."""

    def test_genesis_block_created(self):
        """Engine should create a genesis block on initialization."""
        pow_engine = ProofOfWork(initial_difficulty=1)
        self.assertEqual(len(pow_engine.chain), 1)
        self.assertEqual(pow_engine.chain[0].index, 0)
        self.assertEqual(pow_engine.chain[0].data, "Genesis Block")

    def test_mine_block_meets_difficulty(self):
        """Mined block's hash must start with the required number of zeros."""
        pow_engine = ProofOfWork(initial_difficulty=2, adjustment_interval=100)
        block = pow_engine.mine_block("test data")
        self.assertTrue(block.hash.startswith("00"))

    def test_mine_block_chain_linkage(self):
        """Each block's previous_hash must match the prior block's hash."""
        pow_engine = ProofOfWork(initial_difficulty=1, adjustment_interval=100)
        pow_engine.mine_block("block 1")
        pow_engine.mine_block("block 2")
        self.assertEqual(pow_engine.chain[2].previous_hash, pow_engine.chain[1].hash)

    def test_mine_block_sequential_indices(self):
        """Block indices should be sequential starting from 0."""
        pow_engine = ProofOfWork(initial_difficulty=1, adjustment_interval=100)
        for i in range(5):
            pow_engine.mine_block(f"block {i+1}")
        indices = [b.index for b in pow_engine.chain]
        self.assertEqual(indices, [0, 1, 2, 3, 4, 5])

    def test_validate_chain_valid(self):
        """A freshly mined chain should pass validation."""
        pow_engine = ProofOfWork(initial_difficulty=2, adjustment_interval=100)
        for i in range(3):
            pow_engine.mine_block(f"data {i}")
        is_valid, msg = pow_engine.validate_chain()
        self.assertTrue(is_valid)

    def test_validate_chain_detects_tamper(self):
        """Changing a block's data should be detected by validation."""
        pow_engine = ProofOfWork(initial_difficulty=1, adjustment_interval=100)
        pow_engine.mine_block("original")
        pow_engine.mine_block("another")
        # Tamper with block 1
        pow_engine.chain[1].data = "TAMPERED"
        is_valid, msg = pow_engine.validate_chain()
        self.assertFalse(is_valid)
        self.assertIn("Block 1", msg)

    def test_get_target(self):
        """Target should be a string of zeros matching difficulty."""
        pow_engine = ProofOfWork(initial_difficulty=3)
        self.assertEqual(pow_engine.get_target(), "000")


class TestAttackSimulation(unittest.TestCase):
    """Tests for the attack probability functions."""

    def test_theoretical_majority_attacker(self):
        """Attacker with >= 50% hash power always catches up."""
        self.assertEqual(theoretical_attack_probability(0.5, 6), 1.0)
        self.assertEqual(theoretical_attack_probability(0.7, 6), 1.0)

    def test_theoretical_zero_attacker(self):
        """Attacker with ~0% hash power never catches up."""
        p = theoretical_attack_probability(0.001, 6)
        self.assertAlmostEqual(p, 0.0, places=5)

    def test_simulate_attack_majority(self):
        """Simulated majority attacker should succeed most of the time."""
        p = simulate_attack(0.5, 1, simulations=1000)
        self.assertGreater(p, 0.8)  # Should be close to 1.0

    def test_simulate_attack_minority(self):
        """Simulated 10% attacker from 6 blocks behind should rarely succeed."""
        p = simulate_attack(0.1, 6, simulations=1000)
        self.assertLess(p, 0.05)


class TestMiningStats(unittest.TestCase):
    """Tests for mining statistics analysis."""

    def test_analyze_excludes_genesis(self):
        """Stats should only count mined blocks (not genesis)."""
        pow_engine = ProofOfWork(initial_difficulty=1, adjustment_interval=100)
        pow_engine.mine_block("test")
        stats = analyze_mining_stats(pow_engine.chain)
        self.assertEqual(stats["blocks_mined"], 1)

    def test_analyze_empty_chain(self):
        """Chain with only genesis should return empty dict or zero blocks."""
        pow_engine = ProofOfWork(initial_difficulty=1)
        stats = analyze_mining_stats(pow_engine.chain)
        # Either empty dict or blocks_mined=0 is acceptable
        if stats:
            self.assertEqual(stats.get("blocks_mined", 0), 0)


if __name__ == "__main__":
    unittest.main()
