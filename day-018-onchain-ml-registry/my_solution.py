"""
Day 018: On-Chain ML Model Registry — Your Implementation

Build a blockchain-backed registry that stores ML model metadata immutably.
This integrates AI (model metadata), Crypto (blockchain, Merkle trees, signatures),
and Robotics (auditable model versioning for autonomous systems).

Hints:
- Start with the cryptographic primitives (sha256, sign/verify)
- Then build ModelEntry with canonical JSON serialization
- Then Merkle tree (reuse concepts from Day 013)
- Then Block and Blockchain
- Finally, the ModelRegistry API that ties it all together

Run tests as you go: python3 -m pytest tests.py -v
"""

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# 1. CRYPTOGRAPHIC PRIMITIVES
# =============================================================================

def sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string.

    Hint: Use hashlib.sha256 with UTF-8 encoding.
    """
    raise NotImplementedError("TODO: implement this")


def generate_keypair() -> tuple[str, str]:
    """Generate a simulated key pair (private_key, public_key).

    Hint: Use os.urandom(32).hex() for the private key.
    Derive public key as sha256("pubkey:" + private_key) to simulate
    the one-way property of real key derivation.

    Returns:
        (private_key, public_key) tuple of hex strings
    """
    raise NotImplementedError("TODO: implement this")


def sign(private_key: str, message: str) -> str:
    """Sign a message with a private key using HMAC-SHA256.

    Hint: Use hmac.new() with the private key as the HMAC key
    and SHA-256 as the hash function.

    Returns:
        Hex digest of the HMAC signature
    """
    raise NotImplementedError("TODO: implement this")


def verify_signature(private_key: str, message: str, signature: str) -> bool:
    """Verify an HMAC signature.

    Hint: Recompute the signature and use hmac.compare_digest()
    for constant-time comparison (prevents timing attacks).
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# 2. MODEL ENTRY
# =============================================================================

@dataclass
class ModelEntry:
    """A single model registration record.

    Hint: The key design principle is CONTENT-ADDRESSED IDENTITY.
    The model is identified by its hash (what it IS), not its name.
    """
    model_hash: str          # SHA-256 of serialized model weights
    name: str                # Human-readable name
    version: str             # Semantic version
    architecture: str        # Model type description
    hyperparameters: dict    # Training configuration
    training_data_hash: str  # SHA-256 of training dataset
    metrics: dict            # Evaluation results
    owner: str               # Public key of registrant
    signature: str = ""      # Digital signature
    timestamp: float = 0.0   # Registration time
    entry_hash: str = ""     # Hash of this entry

    def canonical_json(self) -> str:
        """Produce a deterministic JSON representation for hashing.

        Hint: Use json.dumps with sort_keys=True and separators=(",", ":")
        to ensure the same entry always produces the same hash.
        Exclude signature and entry_hash from the dict (they create
        circular dependencies).
        """
        raise NotImplementedError("TODO: implement this")

    def compute_hash(self) -> str:
        """Compute the SHA-256 hash of this entry's canonical form."""
        raise NotImplementedError("TODO: implement this")

    def finalize(self, private_key: str) -> None:
        """Compute entry hash and sign it.

        Hint: Two steps:
        1. self.entry_hash = self.compute_hash()
        2. self.signature = sign(private_key, self.entry_hash)
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# 3. MERKLE TREE
# =============================================================================

def merkle_root(hashes: list[str]) -> str:
    """Compute the Merkle root of a list of hashes.

    Hint: Build the tree bottom-up:
    1. Start with leaf hashes
    2. Pair them and hash: sha256("node:" + left + right)
    3. If odd count, duplicate the last hash
    4. Repeat until one hash remains
    Return sha256("empty") for empty lists.
    """
    raise NotImplementedError("TODO: implement this")


def merkle_proof(hashes: list[str], index: int) -> list[tuple[str, str]]:
    """Generate a Merkle inclusion proof for the entry at `index`.

    Hint: As you build each level, track which sibling pairs with
    your target index. Record (sibling_hash, "left"/"right") indicating
    which side the sibling is on. Update index = index // 2 each level.

    Returns:
        List of (sibling_hash, direction) tuples
    """
    raise NotImplementedError("TODO: implement this")


def verify_merkle_proof(
    entry_hash: str,
    proof: list[tuple[str, str]],
    expected_root: str,
) -> bool:
    """Verify a Merkle inclusion proof.

    Hint: Start with entry_hash, for each (sibling, direction) in proof:
    - If direction == "right": hash = sha256("node:" + current + sibling)
    - If direction == "left":  hash = sha256("node:" + sibling + current)
    Final hash should equal expected_root.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# 4. BLOCK AND BLOCKCHAIN
# =============================================================================

@dataclass
class Block:
    """A single block in the model registry blockchain.

    Hint: The block header = index + timestamp + previous_hash + merkle_root + nonce.
    The block hash is SHA-256 of the header string.
    """
    index: int
    timestamp: float
    entries: list[ModelEntry]
    previous_hash: str
    merkle_root_hash: str = ""
    nonce: int = 0
    block_hash: str = ""

    def compute_merkle_root(self) -> str:
        """Compute Merkle root from all entry hashes in this block."""
        raise NotImplementedError("TODO: implement this")

    def compute_hash(self) -> str:
        """Compute the block hash from header fields.

        Hint: Concatenate header fields as:
        f"{index}:{timestamp}:{previous_hash}:{merkle_root_hash}:{nonce}"
        Then SHA-256 hash the result.
        """
        raise NotImplementedError("TODO: implement this")

    def mine(self, difficulty: int = 2) -> None:
        """Find a nonce that produces a block hash with `difficulty` leading zeros.

        Hint: Set merkle_root_hash first, then increment nonce in a loop
        until block_hash starts with "0" * difficulty.
        """
        raise NotImplementedError("TODO: implement this")


@dataclass
class Blockchain:
    """The blockchain backing the model registry.

    Hint: Initialize with a genesis block (index=0, previous_hash="0"*64).
    """
    chain: list[Block] = field(default_factory=list)
    difficulty: int = 2

    def __post_init__(self) -> None:
        """Create the genesis block if the chain is empty."""
        raise NotImplementedError("TODO: implement this")

    def latest_block(self) -> Block:
        raise NotImplementedError("TODO: implement this")

    def add_block(self, entries: list[ModelEntry]) -> Block:
        """Create and mine a new block containing the given model entries."""
        raise NotImplementedError("TODO: implement this")

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify the entire chain's integrity.

        Hint: For each block (skipping genesis), check:
        1. block_hash matches recomputed hash
        2. previous_hash matches prior block's hash
        3. merkle_root matches recomputed root from entries
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# 5. MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """On-chain ML model registry — the high-level API.

    Hint: This wraps the Blockchain and provides user-friendly methods.
    Maintain an index (model_hash -> block/entry location) for fast lookup.
    """

    def __init__(self, difficulty: int = 2) -> None:
        self.blockchain = Blockchain(difficulty=difficulty)
        self._pending: list[ModelEntry] = []
        self._key_map: dict[str, str] = {}  # public_key -> private_key
        self._index: dict[str, tuple[int, int]] = {}  # model_hash -> (block_idx, entry_idx)

    def register_keypair(self, private_key: str, public_key: str) -> None:
        """Register a key pair for signature verification."""
        raise NotImplementedError("TODO: implement this")

    def register_model(
        self,
        model_weights: str,
        name: str,
        version: str,
        architecture: str,
        hyperparameters: dict,
        training_data: str,
        metrics: dict,
        private_key: str,
        public_key: str,
        auto_mine: bool = True,
    ) -> ModelEntry:
        """Register a new ML model on the blockchain.

        Hint:
        1. Hash model_weights and training_data with sha256
        2. Create ModelEntry with all fields + timestamp
        3. Call entry.finalize(private_key)
        4. Register the keypair
        5. If auto_mine, mine a block; otherwise add to pending
        6. Update the index
        """
        raise NotImplementedError("TODO: implement this")

    def mine_pending(self) -> Optional[Block]:
        """Mine all pending entries into a new block."""
        raise NotImplementedError("TODO: implement this")

    def get_model(self, model_hash: str) -> Optional[ModelEntry]:
        """Look up a model by its content hash."""
        raise NotImplementedError("TODO: implement this")

    def get_model_by_name(self, name: str) -> list[ModelEntry]:
        """Find all model versions registered under a given name."""
        raise NotImplementedError("TODO: implement this")

    def list_models(self) -> list[ModelEntry]:
        """List all registered models across all blocks."""
        raise NotImplementedError("TODO: implement this")

    def verify_chain(self) -> tuple[bool, str]:
        """Verify blockchain integrity."""
        raise NotImplementedError("TODO: implement this")

    def verify_model(self, model_hash: str) -> tuple[bool, str]:
        """Verify a specific model's integrity and ownership.

        Hint: Three checks:
        1. Recompute entry hash, compare to stored
        2. Verify signature with owner's key
        3. Verify Merkle inclusion proof
        """
        raise NotImplementedError("TODO: implement this")

    def get_inclusion_proof(self, model_hash: str) -> Optional[dict]:
        """Get a Merkle inclusion proof for a model."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

if __name__ == "__main__":
    print("Testing your On-Chain ML Model Registry...")
    print("=" * 50)

    # Test 1: Crypto primitives
    print("\n1. Testing crypto primitives...")
    h = sha256("hello")
    assert len(h) == 64, "SHA-256 should produce 64-char hex string"
    priv, pub = generate_keypair()
    sig = sign(priv, "test message")
    assert verify_signature(priv, "test message", sig), "Signature should verify"
    assert not verify_signature(priv, "wrong message", sig), "Wrong message should fail"
    print("   PASSED")

    # Test 2: Model entry
    print("\n2. Testing model entry...")
    entry = ModelEntry(
        model_hash=sha256("weights"),
        name="test-model",
        version="1.0.0",
        architecture="TestNet",
        hyperparameters={"lr": 0.01},
        training_data_hash=sha256("data"),
        metrics={"accuracy": 0.95},
        owner=pub,
        timestamp=1000.0,
    )
    entry.finalize(priv)
    assert entry.entry_hash != "", "Entry hash should be set"
    assert entry.signature != "", "Signature should be set"
    print("   PASSED")

    # Test 3: Merkle tree
    print("\n3. Testing Merkle tree...")
    hashes = [sha256(str(i)) for i in range(4)]
    root = merkle_root(hashes)
    assert len(root) == 64, "Root should be a hash"
    proof = merkle_proof(hashes, 1)
    assert verify_merkle_proof(hashes[1], proof, root), "Proof should verify"
    print("   PASSED")

    # Test 4: Full registry
    print("\n4. Testing full registry...")
    registry = ModelRegistry(difficulty=1)
    priv, pub = generate_keypair()
    entry = registry.register_model(
        model_weights="my_model_weights",
        name="test-model",
        version="1.0.0",
        architecture="MLP",
        hyperparameters={"lr": 0.01},
        training_data="training_data",
        metrics={"accuracy": 0.95},
        private_key=priv,
        public_key=pub,
    )
    valid, msg = registry.verify_chain()
    assert valid, f"Chain should be valid: {msg}"
    valid, msg = registry.verify_model(entry.model_hash)
    assert valid, f"Model should verify: {msg}"
    print("   PASSED")

    print("\n" + "=" * 50)
    print("All basic tests passed! Run 'python3 -m pytest tests.py' for full test suite.")
