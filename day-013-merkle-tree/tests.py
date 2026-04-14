"""
Day 013: Merkle Tree — Test Suite

Run with:
    python3 -m pytest tests.py -v
    python3 tests.py
"""

import unittest
from my_solution import (
    sha256, hash_leaf, hash_pair,
    MerkleTree, verify_proof
)


class TestHashFunctions(unittest.TestCase):
    """Test the building-block hash functions."""

    def test_sha256_deterministic(self):
        """Same input always produces same output."""
        h1 = sha256(b"hello")
        h2 = sha256(b"hello")
        self.assertEqual(h1, h2)

    def test_hash_leaf_domain_separation(self):
        """Leaf hash should differ from raw SHA-256 of the same data."""
        raw = sha256(b"test")
        leaf = hash_leaf("test")
        self.assertNotEqual(raw, leaf)

    def test_hash_pair_order_matters(self):
        """H(a || b) != H(b || a) — concatenation order is significant."""
        h1 = hash_leaf("a")
        h2 = hash_leaf("b")
        self.assertNotEqual(hash_pair(h1, h2), hash_pair(h2, h1))

    def test_hash_leaf_different_inputs(self):
        """Different inputs produce different hashes."""
        self.assertNotEqual(hash_leaf("foo"), hash_leaf("bar"))


class TestMerkleTreeConstruction(unittest.TestCase):
    """Test tree building with various sizes."""

    def test_single_leaf(self):
        """A tree with one leaf: root = leaf hash."""
        tree = MerkleTree(["only"])
        self.assertEqual(tree.root, hash_leaf("only"))
        self.assertEqual(tree.height, 1)

    def test_two_leaves(self):
        """Two leaves: root = H(leaf0, leaf1)."""
        tree = MerkleTree(["a", "b"])
        expected_root = hash_pair(hash_leaf("a"), hash_leaf("b"))
        self.assertEqual(tree.root, expected_root)
        self.assertEqual(tree.height, 2)

    def test_four_leaves_structure(self):
        """Four leaves should produce a balanced tree with 3 levels."""
        tree = MerkleTree(["a", "b", "c", "d"])
        self.assertEqual(tree.height, 3)
        self.assertEqual(len(tree.levels[0]), 4)  # 4 leaves
        self.assertEqual(len(tree.levels[1]), 2)  # 2 internal
        self.assertEqual(len(tree.levels[2]), 1)  # 1 root

    def test_empty_raises(self):
        """Empty data should raise ValueError."""
        with self.assertRaises(ValueError):
            MerkleTree([])

    def test_odd_leaves_three(self):
        """Three leaves: one gets promoted, all proofs still work."""
        tree = MerkleTree(["a", "b", "c"])
        # Should have 3 levels: [3 leaves] -> [H(a,b), c] -> [root]
        self.assertEqual(len(tree.levels[0]), 3)
        self.assertEqual(len(tree.levels[1]), 2)
        self.assertEqual(len(tree.levels[2]), 1)


class TestMerkleProofs(unittest.TestCase):
    """Test proof generation and verification."""

    def test_all_proofs_valid_power_of_two(self):
        """Every leaf's proof should verify for a 4-leaf tree."""
        data = ["tx1", "tx2", "tx3", "tx4"]
        tree = MerkleTree(data)
        for i, d in enumerate(data):
            proof = tree.get_proof(i)
            self.assertTrue(verify_proof(d, proof, tree.root),
                            f"Proof failed for index {i}")

    def test_all_proofs_valid_odd_count(self):
        """Every leaf's proof should verify for a 5-leaf tree."""
        data = ["a", "b", "c", "d", "e"]
        tree = MerkleTree(data)
        for i, d in enumerate(data):
            proof = tree.get_proof(i)
            self.assertTrue(verify_proof(d, proof, tree.root),
                            f"Proof failed for index {i}")

    def test_tampered_data_fails(self):
        """Proof should fail if the leaf data is modified."""
        data = ["a", "b", "c", "d"]
        tree = MerkleTree(data)
        proof = tree.get_proof(1)
        # Original data should verify
        self.assertTrue(verify_proof("b", proof, tree.root))
        # Tampered data should NOT verify
        self.assertFalse(verify_proof("b_tampered", proof, tree.root))

    def test_wrong_root_fails(self):
        """Proof should fail if verified against a different root."""
        data = ["a", "b", "c", "d"]
        tree = MerkleTree(data)
        proof = tree.get_proof(0)
        fake_root = "0" * 64
        self.assertFalse(verify_proof("a", proof, fake_root))

    def test_proof_index_out_of_range(self):
        """get_proof should raise IndexError for bad indices."""
        tree = MerkleTree(["a", "b"])
        with self.assertRaises(IndexError):
            tree.get_proof(5)
        with self.assertRaises(IndexError):
            tree.get_proof(-1)

    def test_proof_size_logarithmic(self):
        """Proof size should be O(log n)."""
        # 8 leaves -> proof should have 3 elements (log2(8) = 3)
        data = [f"item_{i}" for i in range(8)]
        tree = MerkleTree(data)
        proof = tree.get_proof(0)
        self.assertEqual(len(proof), 3)

    def test_different_trees_different_roots(self):
        """Changing any leaf should change the root."""
        tree1 = MerkleTree(["a", "b", "c", "d"])
        tree2 = MerkleTree(["a", "b", "c", "x"])
        self.assertNotEqual(tree1.root, tree2.root)


if __name__ == '__main__':
    unittest.main()
