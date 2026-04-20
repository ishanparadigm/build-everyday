"""
Tests for Day 018: On-Chain ML Model Registry

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import time
from my_solution import (
    sha256,
    generate_keypair,
    sign,
    verify_signature,
    ModelEntry,
    merkle_root,
    merkle_proof,
    verify_merkle_proof,
    Block,
    Blockchain,
    ModelRegistry,
)


class TestCryptoPrimitives(unittest.TestCase):
    """Test the foundational cryptographic functions."""

    def test_sha256_deterministic(self):
        """Same input should always produce the same hash."""
        h1 = sha256("hello world")
        h2 = sha256("hello world")
        self.assertEqual(h1, h2)

    def test_sha256_length(self):
        """SHA-256 produces a 64-character hex string (256 bits)."""
        h = sha256("test")
        self.assertEqual(len(h), 64)
        # Should be valid hex
        int(h, 16)

    def test_sha256_avalanche(self):
        """Changing one character should produce a completely different hash."""
        h1 = sha256("hello")
        h2 = sha256("hellp")  # One character different
        self.assertNotEqual(h1, h2)

    def test_keypair_uniqueness(self):
        """Each keypair should be unique."""
        k1 = generate_keypair()
        k2 = generate_keypair()
        self.assertNotEqual(k1[0], k2[0])  # Different private keys
        self.assertNotEqual(k1[1], k2[1])  # Different public keys

    def test_signature_valid(self):
        """A valid signature should verify correctly."""
        priv, pub = generate_keypair()
        msg = "register model XYZ"
        sig = sign(priv, msg)
        self.assertTrue(verify_signature(priv, msg, sig))

    def test_signature_wrong_message(self):
        """Signature should fail for a different message."""
        priv, pub = generate_keypair()
        sig = sign(priv, "original message")
        self.assertFalse(verify_signature(priv, "tampered message", sig))

    def test_signature_wrong_key(self):
        """Signature should fail with a different key."""
        priv1, _ = generate_keypair()
        priv2, _ = generate_keypair()
        sig = sign(priv1, "test")
        self.assertFalse(verify_signature(priv2, "test", sig))


class TestModelEntry(unittest.TestCase):
    """Test model entry creation and hashing."""

    def setUp(self):
        self.priv, self.pub = generate_keypair()

    def _make_entry(self, **overrides):
        defaults = dict(
            model_hash=sha256("weights"),
            name="test-model",
            version="1.0.0",
            architecture="MLP",
            hyperparameters={"lr": 0.01},
            training_data_hash=sha256("data"),
            metrics={"accuracy": 0.95},
            owner=self.pub,
            timestamp=1000.0,
        )
        defaults.update(overrides)
        return ModelEntry(**defaults)

    def test_canonical_json_deterministic(self):
        """Canonical JSON should be deterministic regardless of dict ordering."""
        e1 = self._make_entry(hyperparameters={"a": 1, "b": 2})
        e2 = self._make_entry(hyperparameters={"b": 2, "a": 1})
        self.assertEqual(e1.canonical_json(), e2.canonical_json())

    def test_entry_hash_changes_with_metrics(self):
        """Changing metrics should change the entry hash."""
        e1 = self._make_entry(metrics={"accuracy": 0.90})
        e2 = self._make_entry(metrics={"accuracy": 0.95})
        e1.finalize(self.priv)
        e2.finalize(self.priv)
        self.assertNotEqual(e1.entry_hash, e2.entry_hash)

    def test_finalize_sets_hash_and_signature(self):
        """Finalize should set both entry_hash and signature."""
        entry = self._make_entry()
        self.assertEqual(entry.entry_hash, "")
        self.assertEqual(entry.signature, "")
        entry.finalize(self.priv)
        self.assertNotEqual(entry.entry_hash, "")
        self.assertNotEqual(entry.signature, "")
        self.assertEqual(len(entry.entry_hash), 64)


class TestMerkleTree(unittest.TestCase):
    """Test Merkle tree construction and proof verification."""

    def test_single_hash(self):
        """Merkle root of a single hash should still be a valid hash."""
        h = sha256("leaf")
        root = merkle_root([h])
        self.assertEqual(len(root), 64)

    def test_empty_hashes(self):
        """Empty list should return a deterministic hash."""
        root = merkle_root([])
        self.assertEqual(root, sha256("empty"))

    def test_root_changes_with_content(self):
        """Different leaves should produce different roots."""
        hashes1 = [sha256("a"), sha256("b")]
        hashes2 = [sha256("a"), sha256("c")]
        self.assertNotEqual(merkle_root(hashes1), merkle_root(hashes2))

    def test_proof_verification(self):
        """Merkle proof should verify for each leaf in a 4-leaf tree."""
        hashes = [sha256(str(i)) for i in range(4)]
        root = merkle_root(hashes)
        for i in range(4):
            proof = merkle_proof(hashes, i)
            self.assertTrue(
                verify_merkle_proof(hashes[i], proof, root),
                f"Proof failed for index {i}",
            )

    def test_proof_verification_odd_count(self):
        """Merkle proof should work with an odd number of leaves."""
        hashes = [sha256(str(i)) for i in range(5)]
        root = merkle_root(hashes)
        for i in range(5):
            proof = merkle_proof(hashes, i)
            self.assertTrue(
                verify_merkle_proof(hashes[i], proof, root),
                f"Proof failed for index {i} (odd count)",
            )

    def test_proof_fails_wrong_hash(self):
        """Proof should fail if the entry hash is wrong."""
        hashes = [sha256(str(i)) for i in range(4)]
        root = merkle_root(hashes)
        proof = merkle_proof(hashes, 0)
        wrong_hash = sha256("wrong")
        self.assertFalse(verify_merkle_proof(wrong_hash, proof, root))


class TestBlockchain(unittest.TestCase):
    """Test blockchain construction and integrity verification."""

    def test_genesis_block(self):
        """Blockchain should start with a genesis block."""
        bc = Blockchain(difficulty=1)
        self.assertEqual(len(bc.chain), 1)
        self.assertEqual(bc.chain[0].index, 0)
        self.assertEqual(bc.chain[0].previous_hash, "0" * 64)

    def test_add_block(self):
        """Adding a block should extend the chain."""
        bc = Blockchain(difficulty=1)
        priv, pub = generate_keypair()
        entry = ModelEntry(
            model_hash=sha256("w"), name="m", version="1.0",
            architecture="A", hyperparameters={},
            training_data_hash=sha256("d"), metrics={},
            owner=pub, timestamp=time.time(),
        )
        entry.finalize(priv)
        bc.add_block([entry])
        self.assertEqual(len(bc.chain), 2)

    def test_chain_integrity_valid(self):
        """A properly constructed chain should pass verification."""
        bc = Blockchain(difficulty=1)
        priv, pub = generate_keypair()
        for i in range(3):
            entry = ModelEntry(
                model_hash=sha256(f"w{i}"), name=f"m{i}", version="1.0",
                architecture="A", hyperparameters={},
                training_data_hash=sha256("d"), metrics={},
                owner=pub, timestamp=time.time(),
            )
            entry.finalize(priv)
            bc.add_block([entry])
        valid, msg = bc.verify_integrity()
        self.assertTrue(valid, msg)

    def test_chain_detects_tampering(self):
        """Modifying a block's previous_hash should be detected."""
        bc = Blockchain(difficulty=1)
        priv, pub = generate_keypair()
        for i in range(2):
            entry = ModelEntry(
                model_hash=sha256(f"w{i}"), name=f"m{i}", version="1.0",
                architecture="A", hyperparameters={},
                training_data_hash=sha256("d"), metrics={},
                owner=pub, timestamp=time.time(),
            )
            entry.finalize(priv)
            bc.add_block([entry])
        # Tamper with chain link
        bc.chain[2].previous_hash = sha256("fake")
        valid, msg = bc.verify_integrity()
        self.assertFalse(valid)


class TestModelRegistry(unittest.TestCase):
    """Test the high-level registry API."""

    def setUp(self):
        self.registry = ModelRegistry(difficulty=1)
        self.priv, self.pub = generate_keypair()

    def _register(self, name="model", version="1.0", weights="w", **kw):
        defaults = dict(
            model_weights=weights,
            name=name,
            version=version,
            architecture="MLP",
            hyperparameters={"lr": 0.01},
            training_data="data",
            metrics={"accuracy": 0.9},
            private_key=self.priv,
            public_key=self.pub,
        )
        defaults.update(kw)
        return self.registry.register_model(**defaults)

    def test_register_and_lookup(self):
        """Registered model should be retrievable by hash."""
        entry = self._register()
        found = self.registry.get_model(entry.model_hash)
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "model")

    def test_register_and_lookup_by_name(self):
        """Should find all versions of a named model."""
        self._register(name="foo", version="1.0", weights="w1")
        self._register(name="foo", version="2.0", weights="w2")
        self._register(name="bar", version="1.0", weights="w3")
        results = self.registry.get_model_by_name("foo")
        self.assertEqual(len(results), 2)

    def test_verify_model_passes(self):
        """A properly registered model should pass verification."""
        entry = self._register()
        valid, msg = self.registry.verify_model(entry.model_hash)
        self.assertTrue(valid, msg)

    def test_verify_detects_metric_tampering(self):
        """Tampering with metrics should fail verification."""
        entry = self._register()
        entry.metrics["accuracy"] = 0.999
        valid, msg = self.registry.verify_model(entry.model_hash)
        self.assertFalse(valid)
        self.assertIn("tampered", msg.lower())

    def test_verify_chain_passes(self):
        """Chain with multiple models should verify."""
        self._register(weights="w1")
        self._register(weights="w2")
        self._register(weights="w3")
        valid, msg = self.registry.verify_chain()
        self.assertTrue(valid, msg)

    def test_list_models(self):
        """list_models should return all registered models."""
        self._register(weights="w1")
        self._register(weights="w2")
        models = self.registry.list_models()
        self.assertEqual(len(models), 2)

    def test_inclusion_proof(self):
        """Inclusion proof should verify independently."""
        entry = self._register()
        proof_data = self.registry.get_inclusion_proof(entry.model_hash)
        self.assertIsNotNone(proof_data)
        verified = verify_merkle_proof(
            proof_data["entry_hash"],
            proof_data["proof"],
            proof_data["merkle_root"],
        )
        self.assertTrue(verified)

    def test_nonexistent_model(self):
        """Looking up a nonexistent model should return None/fail."""
        self.assertIsNone(self.registry.get_model(sha256("nonexistent")))
        valid, msg = self.registry.verify_model(sha256("nonexistent"))
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
