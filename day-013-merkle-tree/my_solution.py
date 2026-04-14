"""
Day 013: Merkle Tree from Scratch — Your Implementation

Build a Merkle tree that can:
1. Construct a tree from a list of data blocks
2. Compute a root hash that summarizes all data
3. Generate O(log n) inclusion proofs
4. Verify proofs without needing the full tree

Hints:
- Start with hash_leaf() and hash_pair() — these are your building blocks
- Remember domain separation: prefix leaves with 0x00 and internal nodes with 0x01
- For odd-level handling, promote the last node (don't duplicate like Bitcoin)
- A proof is just the list of siblings you need to reconstruct the path to root
- Verification walks from leaf to root, combining with proof elements at each step
"""

import hashlib
from typing import Optional


def sha256(data: bytes) -> str:
    """Compute SHA-256 hash and return as hex string."""
    return hashlib.sha256(data).hexdigest()


def hash_leaf(data: str) -> str:
    """Hash a leaf node's data with domain separation prefix 0x00.

    Args:
        data: The string data to hash.

    Returns:
        Hex-encoded SHA-256 hash with leaf prefix.

    Hint: Prepend b'\\x00' before the UTF-8 encoded data, then SHA-256 hash it.
    This prevents leaf/internal node confusion (second-preimage attacks).
    """
    raise NotImplementedError("TODO: implement this")


def hash_pair(left: str, right: str) -> str:
    """Hash two child node hashes to produce a parent hash.

    Args:
        left: Hex-encoded hash of the left child.
        right: Hex-encoded hash of the right child.

    Returns:
        Hex-encoded SHA-256 hash with internal node prefix.

    Hint: Prepend b'\\x01', then concatenate the raw bytes of both hashes.
    Use bytes.fromhex() to convert hex strings back to bytes.
    """
    raise NotImplementedError("TODO: implement this")


class MerkleTree:
    """A binary Merkle tree built from a list of data blocks.

    Attributes:
        data_blocks: Original string data
        leaves: List of leaf hashes
        levels: List of lists — levels[0] = leaves, levels[-1] = [root]
    """

    def __init__(self, data_blocks: list[str]) -> None:
        """Build a Merkle tree from data blocks.

        Args:
            data_blocks: List of strings (must be non-empty).

        Raises:
            ValueError: If data_blocks is empty.

        Hint: Hash all blocks to create leaves, then call _build_tree().
        """
        raise NotImplementedError("TODO: implement this")

    def _build_tree(self) -> list[list[str]]:
        """Build tree levels from bottom (leaves) to top (root).

        Returns:
            List of levels where levels[0] = leaves, levels[-1] = [root].

        Hint:
        - Start with a copy of self.leaves as level 0
        - While the current level has more than 1 node:
          - Pair adjacent nodes: hash_pair(current[i], current[i+1])
          - If odd number of nodes, promote the last one to next level
        - Think about what "promote" means: it goes up WITHOUT being hashed
        """
        raise NotImplementedError("TODO: implement this")

    @property
    def root(self) -> str:
        """The root hash of the tree."""
        raise NotImplementedError("TODO: implement this")

    @property
    def height(self) -> int:
        """Number of levels in the tree (including leaves and root)."""
        raise NotImplementedError("TODO: implement this")

    def get_proof(self, index: int) -> list[tuple[str, str]]:
        """Generate a Merkle proof for the leaf at the given index.

        Args:
            index: Zero-based index of the leaf.

        Returns:
            List of (sibling_hash, direction) tuples.
            direction is 'left' or 'right' — the side the SIBLING is on.

        Raises:
            IndexError: If index is out of range.

        Hint:
        - Walk from the leaf level to the root level
        - At each level, determine if you're a left child (even index)
          or right child (odd index)
        - Collect the sibling's hash and which side it's on
        - If you're a left child with no right sibling (promoted), skip
        - Move to parent: idx = idx // 2
        """
        raise NotImplementedError("TODO: implement this")


def verify_proof(
    leaf_data: str,
    proof: list[tuple[str, str]],
    expected_root: str
) -> bool:
    """Verify a Merkle proof for a given leaf.

    Args:
        leaf_data: The original string data for the leaf.
        proof: List of (hash, direction) pairs from get_proof().
        expected_root: The root hash to verify against.

    Returns:
        True if the proof is valid, False otherwise.

    Hint:
    - Start with hash_leaf(leaf_data)
    - For each (sibling_hash, direction) in the proof:
      - If direction == 'left': hash_pair(sibling, current)
      - If direction == 'right': hash_pair(current, sibling)
    - Compare final hash to expected_root
    """
    raise NotImplementedError("TODO: implement this")


if __name__ == '__main__':
    # Test your implementation step by step

    # Step 1: Test hashing
    print("Step 1: Testing hash functions")
    h = hash_leaf("hello")
    print(f"  hash_leaf('hello') = {h[:16]}...")

    h1 = hash_leaf("a")
    h2 = hash_leaf("b")
    parent = hash_pair(h1, h2)
    print(f"  hash_pair(H('a'), H('b')) = {parent[:16]}...")

    # Step 2: Build a small tree
    print("\nStep 2: Building a 4-leaf tree")
    data = ["tx_A", "tx_B", "tx_C", "tx_D"]
    tree = MerkleTree(data)
    print(f"  Root: {tree.root[:32]}...")
    print(f"  Height: {tree.height}")
    print(f"  Level sizes: {[len(level) for level in tree.levels]}")

    # Step 3: Generate and verify proofs
    print("\nStep 3: Proofs")
    for i in range(len(data)):
        proof = tree.get_proof(i)
        valid = verify_proof(data[i], proof, tree.root)
        print(f"  [{i}] '{data[i]}': proof_size={len(proof)}, valid={valid}")

    # Step 4: Tamper detection
    print("\nStep 4: Tamper detection")
    proof = tree.get_proof(1)
    valid_original = verify_proof("tx_B", proof, tree.root)
    valid_tampered = verify_proof("tx_B_FAKE", proof, tree.root)
    print(f"  Original data verifies: {valid_original}")
    print(f"  Tampered data verifies: {valid_tampered}")

    # Step 5: Odd number of leaves
    print("\nStep 5: Odd leaf counts")
    for n in [3, 5, 7]:
        blocks = [f"item_{i}" for i in range(n)]
        t = MerkleTree(blocks)
        all_ok = all(verify_proof(blocks[j], t.get_proof(j), t.root) for j in range(n))
        print(f"  {n} leaves -> height {t.height}, all proofs valid: {all_ok}")

    print("\nAll tests passed!" if True else "")
