"""
Day 013: Merkle Tree from Scratch

A complete implementation of a Merkle tree with:
- Tree construction from arbitrary data blocks
- Root hash computation
- Merkle proof generation (audit paths)
- Proof verification
- Tamper detection demonstration

This builds on Day 002 (SHA-256) — we use Python's hashlib SHA-256 here
since the focus is on the tree structure, not the hash function internals.
"""

import hashlib
from typing import Optional


def sha256(data: bytes) -> str:
    """Compute SHA-256 hash and return as hex string.

    We use hex strings throughout for readability. In production you'd
    keep raw bytes to avoid the 2x size overhead of hex encoding.
    """
    return hashlib.sha256(data).hexdigest()


def hash_leaf(data: str) -> str:
    """Hash a leaf node's data.

    We prefix leaf hashes with 0x00 to prevent a second-preimage attack.
    Without this prefix, an attacker could create an internal node that
    looks like a leaf (or vice versa), potentially forging proofs.

    This is a real concern — Certificate Transparency (RFC 6962) mandates
    this domain separation.
    """
    return sha256(b'\x00' + data.encode('utf-8'))


def hash_pair(left: str, right: str) -> str:
    """Hash two child nodes to produce a parent hash.

    We prefix internal node hashes with 0x01 for domain separation
    (distinguishing internal nodes from leaves).

    The concatenation order matters: H(left || right) != H(right || left).
    This is why proofs must track which side each sibling is on.
    """
    return sha256(b'\x01' + bytes.fromhex(left) + bytes.fromhex(right))


class MerkleTree:
    """A binary Merkle tree built from a list of data blocks.

    Architecture:
    - self.leaves: list of leaf hashes (bottom level)
    - self.levels: list of lists, where levels[0] = leaves, levels[-1] = [root]
    - Each level is built by pairing adjacent nodes and hashing them

    Odd-level handling: if a level has an odd number of nodes, the last
    node is promoted to the next level without pairing. This avoids the
    Bitcoin duplication vulnerability where H(A,A) could create ambiguity.
    """

    def __init__(self, data_blocks: list[str]) -> None:
        """Build a Merkle tree from a list of string data blocks.

        Args:
            data_blocks: List of strings to include as leaves.
                         Must contain at least one element.

        Raises:
            ValueError: If data_blocks is empty.
        """
        if not data_blocks:
            raise ValueError("Cannot build Merkle tree from empty data")

        self.data_blocks = data_blocks
        # Step 1: Hash all leaves
        self.leaves = [hash_leaf(block) for block in data_blocks]
        # Step 2: Build the tree bottom-up
        self.levels = self._build_tree()

    def _build_tree(self) -> list[list[str]]:
        """Build tree levels from leaves to root.

        Returns a list of levels where:
        - levels[0] = leaf hashes
        - levels[-1] = [root_hash]

        At each level, we pair adjacent nodes and hash them.
        If a level has an odd number of nodes, the last one is
        promoted (carried up) to the next level unpaired.

        Example with 5 leaves [A, B, C, D, E]:
          Level 0: [A, B, C, D, E]           (5 nodes)
          Level 1: [H(A,B), H(C,D), E]       (3 nodes — E promoted)
          Level 2: [H(H(A,B), H(C,D)), E]    (2 nodes — E promoted again)
          Level 3: [H(..., E)]                (1 node — root)
        """
        levels = [self.leaves[:]]  # Copy so we don't mutate

        current = levels[0]
        while len(current) > 1:
            next_level = []
            i = 0
            while i < len(current) - 1:
                # Pair adjacent nodes and hash them together
                parent = hash_pair(current[i], current[i + 1])
                next_level.append(parent)
                i += 2

            if len(current) % 2 == 1:
                # Odd node out — promote it to the next level
                # This is safer than Bitcoin's duplication approach
                next_level.append(current[-1])

            levels.append(next_level)
            current = next_level

        return levels

    @property
    def root(self) -> str:
        """The root hash — a single 256-bit digest summarizing all data.

        This is what gets stored in a block header (in blockchain context)
        or used as a commit hash (in Git context).
        """
        return self.levels[-1][0]

    @property
    def height(self) -> int:
        """Number of levels in the tree (including leaves and root)."""
        return len(self.levels)

    def get_proof(self, index: int) -> list[tuple[str, str]]:
        """Generate a Merkle proof for the leaf at the given index.

        A proof is a list of (sibling_hash, direction) pairs, where
        direction is 'left' or 'right' indicating which side the
        sibling sits on when concatenating for the parent hash.

        Args:
            index: Zero-based index of the leaf to prove.

        Returns:
            List of (hash, direction) tuples from leaf to root.

        Raises:
            IndexError: If index is out of range.

        Complexity: O(log n) — we collect one sibling per level.
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of range [0, {len(self.leaves)})")

        proof = []
        idx = index

        for level in range(len(self.levels) - 1):
            current_level = self.levels[level]

            # Determine sibling index and direction
            if idx % 2 == 0:
                # We're a left child — sibling is to the right
                sibling_idx = idx + 1
                if sibling_idx < len(current_level):
                    proof.append((current_level[sibling_idx], 'right'))
                # If no sibling (promoted node), no proof element needed at this level
            else:
                # We're a right child — sibling is to the left
                sibling_idx = idx - 1
                proof.append((current_level[sibling_idx], 'left'))

            # Move to parent index in the next level
            idx = idx // 2

        return proof

    def print_tree(self) -> None:
        """Pretty-print the tree structure for debugging."""
        print(f"\nMerkle Tree ({len(self.leaves)} leaves, {self.height} levels)")
        print("=" * 60)
        for i, level in enumerate(reversed(self.levels)):
            depth = len(self.levels) - 1 - i
            indent = "  " * i
            label = "Root" if depth == len(self.levels) - 1 else f"Level {depth}"
            for j, node_hash in enumerate(level):
                short = node_hash[:16] + "..."
                print(f"{indent}{label}[{j}]: {short}")


def verify_proof(
    leaf_data: str,
    proof: list[tuple[str, str]],
    expected_root: str
) -> bool:
    """Verify a Merkle proof for a given leaf.

    This is the function a lightweight client would use. They have:
    1. The data they want to verify (leaf_data)
    2. The proof (list of sibling hashes + directions)
    3. The trusted root hash (e.g., from a block header)

    They do NOT need the full tree — that's the whole point.

    Args:
        leaf_data: The original string data for the leaf.
        proof: List of (hash, direction) pairs from get_proof().
        expected_root: The root hash to verify against.

    Returns:
        True if the proof is valid, False otherwise.

    Complexity: O(log n) — one hash computation per proof element.
    """
    # Start with the leaf hash
    current_hash = hash_leaf(leaf_data)

    # Walk up the tree, combining with proof elements
    for sibling_hash, direction in proof:
        if direction == 'left':
            # Sibling is on the left: H(sibling || current)
            current_hash = hash_pair(sibling_hash, current_hash)
        else:
            # Sibling is on the right: H(current || sibling)
            current_hash = hash_pair(current_hash, sibling_hash)

    # If we end up at the expected root, the proof is valid
    return current_hash == expected_root


def demonstrate_basic_tree() -> None:
    """Build a simple tree and explore its structure."""
    print("=" * 60)
    print("PART 1: Building a Merkle Tree")
    print("=" * 60)

    # These could be transactions, file chunks, or any data
    transactions = [
        "Alice pays Bob 5 BTC",
        "Bob pays Charlie 2 BTC",
        "Charlie pays Dave 1 BTC",
        "Dave pays Eve 3 BTC",
    ]

    print(f"\nData blocks ({len(transactions)} transactions):")
    for i, tx in enumerate(transactions):
        print(f"  [{i}] {tx}")

    tree = MerkleTree(transactions)
    tree.print_tree()

    print(f"\nRoot hash: {tree.root}")
    print(f"Tree height: {tree.height} levels")
    print(f"  -> With 4 leaves, we need ceil(log2(4)) + 1 = 3 levels")

    return tree, transactions


def demonstrate_proofs(tree: MerkleTree, transactions: list[str]) -> None:
    """Generate and verify Merkle proofs."""
    print("\n" + "=" * 60)
    print("PART 2: Generating and Verifying Proofs")
    print("=" * 60)

    # Generate proof for transaction #2
    target_idx = 2
    proof = tree.get_proof(target_idx)

    print(f"\nProving: '{transactions[target_idx]}' (index {target_idx})")
    print(f"Proof contains {len(proof)} sibling hashes:")
    for i, (h, direction) in enumerate(proof):
        print(f"  Level {i}: {h[:16]}... ({direction})")

    # Verify the proof
    is_valid = verify_proof(transactions[target_idx], proof, tree.root)
    print(f"\nVerification result: {'VALID' if is_valid else 'INVALID'}")

    # Verify all leaves
    print("\nVerifying ALL leaves:")
    for i, tx in enumerate(transactions):
        p = tree.get_proof(i)
        valid = verify_proof(tx, p, tree.root)
        print(f"  [{i}] {tx}: {'VALID' if valid else 'INVALID'} (proof size: {len(p)})")


def demonstrate_tamper_detection() -> None:
    """Show that modifying any data invalidates proofs."""
    print("\n" + "=" * 60)
    print("PART 3: Tamper Detection")
    print("=" * 60)

    original_data = ["tx_A", "tx_B", "tx_C", "tx_D"]
    tree = MerkleTree(original_data)

    # Get a proof for tx_B (index 1)
    proof = tree.get_proof(1)
    original_root = tree.root

    print(f"\nOriginal root:  {original_root[:32]}...")
    print(f"Proof for 'tx_B' generated ({len(proof)} elements)")

    # Tamper with tx_B
    tampered_data = ["tx_A", "tx_B_TAMPERED", "tx_C", "tx_D"]
    tampered_tree = MerkleTree(tampered_data)

    print(f"Tampered root:  {tampered_tree.root[:32]}...")
    print(f"Roots match: {original_root == tampered_tree.root}")

    # Try to verify original proof with tampered data
    valid = verify_proof("tx_B_TAMPERED", proof, original_root)
    print(f"\nUsing original proof with tampered data: {'VALID' if valid else 'INVALID'}")
    print("  -> The proof fails because the leaf hash changed,")
    print("     cascading different hashes up to the root.")

    # The legitimate data still verifies
    valid = verify_proof("tx_B", proof, original_root)
    print(f"\nUsing original proof with original data: {'VALID' if valid else 'INVALID'}")


def demonstrate_odd_leaves() -> None:
    """Show how the tree handles non-power-of-2 leaf counts."""
    print("\n" + "=" * 60)
    print("PART 4: Handling Odd Numbers of Leaves")
    print("=" * 60)

    for n in [3, 5, 7, 10]:
        data = [f"block_{i}" for i in range(n)]
        tree = MerkleTree(data)

        # Verify all proofs work
        all_valid = all(
            verify_proof(data[i], tree.get_proof(i), tree.root)
            for i in range(n)
        )

        print(f"\n  {n} leaves -> {tree.height} levels, root={tree.root[:16]}...")
        print(f"    All {n} proofs valid: {all_valid}")
        print(f"    Level sizes: {[len(level) for level in tree.levels]}")


def demonstrate_scaling() -> None:
    """Show the O(log n) proof size scaling."""
    print("\n" + "=" * 60)
    print("PART 5: Logarithmic Scaling")
    print("=" * 60)

    print(f"\n  {'Leaves':>10} | {'Tree Height':>12} | {'Proof Size':>11} | {'Ratio':>8}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*11}-+-{'-'*8}")

    for n in [4, 8, 16, 64, 256, 1024]:
        data = [f"item_{i}" for i in range(n)]
        tree = MerkleTree(data)
        proof = tree.get_proof(0)
        proof_bytes = len(proof) * 32  # Each hash is 32 bytes

        print(f"  {n:>10} | {tree.height:>12} | {len(proof):>8} hashes | {proof_bytes:>5} bytes")

    print("\n  Notice: doubling the leaves adds only 1 hash to the proof!")
    print("  1024 leaves need only 10 hashes (320 bytes) — not 1024 hashes.")


if __name__ == '__main__':
    tree, txs = demonstrate_basic_tree()
    demonstrate_proofs(tree, txs)
    demonstrate_tamper_detection()
    demonstrate_odd_leaves()
    demonstrate_scaling()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Merkle trees give us:
  1. A single hash (root) that commits to ALL data
  2. O(log n) proofs that any piece of data is in the tree
  3. Tamper detection — any modification invalidates the root
  4. Domain separation — leaf/internal node prefix prevents forgery

This is the foundation of:
  - Bitcoin/Ethereum transaction verification
  - Git's content-addressable storage
  - Certificate Transparency logs
  - IPFS file verification
""")
