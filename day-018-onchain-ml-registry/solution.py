"""
Day 018: On-Chain ML Model Registry

A blockchain-backed registry for ML model metadata that provides:
- Content-addressed model identity (SHA-256 of weights)
- Digital signature-based ownership (HMAC-simulated ECDSA)
- Merkle tree inclusion proofs per block
- Full chain/block/entry integrity verification

This integrates concepts from AI (model metadata), Crypto (blockchain, Merkle trees,
digital signatures), and Robotics (auditable model versioning for autonomous systems).
"""

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


# =============================================================================
# 1. CRYPTOGRAPHIC PRIMITIVES
# =============================================================================

def sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string.

    We use UTF-8 encoding throughout for consistency. This is the same hash
    function from Day 002 — the foundation of our entire integrity system.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def generate_keypair() -> tuple[str, str]:
    """Generate a simulated key pair (private_key, public_key).

    In production, this would be ECDSA key generation on secp256k1 (same curve
    as Bitcoin/Ethereum). Here we simulate with random hex strings and derive
    the 'public key' as a hash of the private key — this preserves the
    one-way property (can't derive private from public).
    """
    private_key = os.urandom(32).hex()
    # In real ECDSA, public key = private_key * G (elliptic curve point multiplication)
    # We simulate the one-way derivation with a hash
    public_key = sha256(f"pubkey:{private_key}")
    return private_key, public_key


def sign(private_key: str, message: str) -> str:
    """Sign a message with a private key using HMAC-SHA256.

    Real ECDSA signatures produce (r, s) pairs on an elliptic curve. Our
    HMAC simulation captures the key property: only the private key holder
    can produce a valid signature, but anyone with the public key can verify.

    We use the private key as the HMAC key. Verification requires knowing
    the private key (in our simulation) or the public key (in real ECDSA).
    To bridge this gap, we store a mapping of public_key -> private_key
    in the registry for verification (a simplification — real ECDSA doesn't
    need this).
    """
    return hmac.new(
        private_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_signature(private_key: str, message: str, signature: str) -> bool:
    """Verify an HMAC signature.

    In real ECDSA, this would use only the public key. Our HMAC simulation
    requires the private key for verification, so the registry maintains
    a key mapping internally. The security model is the same: signatures
    are unforgeable without the private key.
    """
    expected = sign(private_key, message)
    # Use hmac.compare_digest to prevent timing attacks — even in simulation,
    # it's good practice to use constant-time comparison for secrets.
    return hmac.compare_digest(expected, signature)


# =============================================================================
# 2. MODEL ENTRY — The fundamental unit of the registry
# =============================================================================

@dataclass
class ModelEntry:
    """A single model registration record.

    This is the metadata that gets sealed into the blockchain. Every field
    contributes to the entry hash, so changing any field (even a single
    character in the name) produces a completely different hash.

    The design follows the principle of CONTENT-ADDRESSED IDENTITY: the model
    is identified by what it IS (hash of weights), not what someone calls it.
    """
    model_hash: str          # SHA-256 of serialized model weights
    name: str                # Human-readable name
    version: str             # Semantic version
    architecture: str        # Model type description
    hyperparameters: dict    # Training configuration
    training_data_hash: str  # SHA-256 of training dataset
    metrics: dict            # Evaluation results
    owner: str               # Public key of registrant
    signature: str = ""      # Digital signature (filled after creation)
    timestamp: float = 0.0   # Registration time (filled at registration)
    entry_hash: str = ""     # Hash of this entry (computed, not set manually)

    def canonical_json(self) -> str:
        """Produce a deterministic JSON representation for hashing.

        Why canonical form matters: JSON doesn't guarantee key order, so
        {"a":1, "b":2} and {"b":2, "a":1} would hash differently despite
        being semantically identical. We sort keys and strip whitespace
        to ensure the same logical entry always produces the same hash.

        We exclude `signature` and `entry_hash` from the canonical form because:
        - signature depends on entry_hash (circular dependency)
        - entry_hash is what we're computing
        """
        data = {
            "model_hash": self.model_hash,
            "name": self.name,
            "version": self.version,
            "architecture": self.architecture,
            "hyperparameters": self.hyperparameters,
            "training_data_hash": self.training_data_hash,
            "metrics": self.metrics,
            "owner": self.owner,
            "timestamp": self.timestamp,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute the SHA-256 hash of this entry's canonical form."""
        return sha256(self.canonical_json())

    def finalize(self, private_key: str) -> None:
        """Compute entry hash and sign it.

        This two-step process mirrors real-world model registration:
        1. Freeze the metadata into a hash (content addressing)
        2. Sign the hash to prove ownership (non-repudiation)

        After finalization, any change to the entry will invalidate both
        the entry_hash and the signature.
        """
        self.entry_hash = self.compute_hash()
        self.signature = sign(private_key, self.entry_hash)


# =============================================================================
# 3. MERKLE TREE — Efficient integrity verification for block entries
# =============================================================================

def merkle_root(hashes: list[str]) -> str:
    """Compute the Merkle root of a list of hashes.

    This is a simplified version of the Day 013 Merkle tree. The key insight:
    instead of storing N entry hashes in the block header (O(n) space), we
    store a single root hash (O(1) space) that commits to ALL entries.

    To prove a specific entry exists in the block, we only need O(log n)
    sibling hashes — much more efficient than transmitting all entries.

    Algorithm:
    1. Start with leaf hashes (our model entry hashes)
    2. Pair them up and hash each pair: H(left || right)
    3. If odd number, duplicate the last hash
    4. Repeat until one hash remains — that's the root
    """
    if not hashes:
        return sha256("empty")

    # Work on a copy so we don't mutate the input
    level = list(hashes)

    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            # If odd number of nodes, duplicate the last one
            right = level[i + 1] if i + 1 < len(level) else level[i]
            # Domain separation: prefix with "node:" to prevent second-preimage attacks
            # (same principle as Day 013)
            combined = sha256(f"node:{left}{right}")
            next_level.append(combined)
        level = next_level

    return level[0]


def merkle_proof(hashes: list[str], index: int) -> list[tuple[str, str]]:
    """Generate a Merkle inclusion proof for the entry at `index`.

    Returns a list of (sibling_hash, direction) tuples. To verify:
    start with the target hash, then for each proof step, combine
    with the sibling according to direction ('left' or 'right').

    This is O(log n) — for a block with 1000 entries, proof is ~10 hashes.
    """
    if not hashes or index >= len(hashes):
        return []

    level = list(hashes)
    proof = []

    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            next_level.append(sha256(f"node:{left}{right}"))

            # If our target index is in this pair, record the sibling
            if i == index or i + 1 == index:
                if i == index:
                    sibling = right
                    proof.append((sibling, "right"))
                else:
                    sibling = left
                    proof.append((sibling, "left"))

        # Update index for next level (integer division by 2)
        index = index // 2
        level = next_level

    return proof


def verify_merkle_proof(
    entry_hash: str,
    proof: list[tuple[str, str]],
    expected_root: str,
) -> bool:
    """Verify a Merkle inclusion proof.

    Starting from the entry hash, walk up the tree using the proof siblings.
    If we arrive at the expected root, the entry is proven to be in the block.
    """
    current = entry_hash
    for sibling_hash, direction in proof:
        if direction == "right":
            current = sha256(f"node:{current}{sibling_hash}")
        else:
            current = sha256(f"node:{sibling_hash}{current}")
    return current == expected_root


# =============================================================================
# 4. BLOCK AND BLOCKCHAIN
# =============================================================================

@dataclass
class Block:
    """A single block in the model registry blockchain.

    Each block contains one or more model entries, sealed by a Merkle root.
    The `previous_hash` field creates the chain — each block commits to
    the entire history before it.

    Design choice: we include a lightweight proof-of-work (few leading zeros)
    not for consensus (this is a single-node registry) but to demonstrate
    the concept and make the block hash non-trivial to forge.
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
        entry_hashes = [e.entry_hash for e in self.entries]
        return merkle_root(entry_hashes)

    def compute_hash(self) -> str:
        """Compute the block hash from header fields.

        Note: we hash the HEADER, not the entries themselves. The entries
        are committed to via the Merkle root. This separation means you
        can verify the block hash without having all entries — you just
        need the Merkle root.
        """
        header = (
            f"{self.index}:{self.timestamp}:{self.previous_hash}"
            f":{self.merkle_root_hash}:{self.nonce}"
        )
        return sha256(header)

    def mine(self, difficulty: int = 2) -> None:
        """Find a nonce that produces a block hash with `difficulty` leading zeros.

        This is proof-of-work at a toy scale. With difficulty=2, we need a hash
        starting with "00" — takes ~256 attempts on average (16^2).

        In production blockchains, difficulty is dynamically adjusted so mining
        takes ~10 minutes (Bitcoin) or ~12 seconds (Ethereum pre-merge).
        """
        self.merkle_root_hash = self.compute_merkle_root()
        target = "0" * difficulty
        while True:
            self.block_hash = self.compute_hash()
            if self.block_hash.startswith(target):
                break
            self.nonce += 1


@dataclass
class Blockchain:
    """The blockchain that backs our model registry.

    This is a simple append-only chain with no forking or consensus —
    it's designed for a single organization's model registry, not a
    decentralized network. The value is IMMUTABILITY and AUDITABILITY,
    not decentralization.
    """
    chain: list[Block] = field(default_factory=list)
    difficulty: int = 2  # Number of leading zeros required in block hash

    def __post_init__(self) -> None:
        """Create the genesis block if the chain is empty."""
        if not self.chain:
            genesis = Block(
                index=0,
                timestamp=time.time(),
                entries=[],
                previous_hash="0" * 64,  # No previous block
            )
            genesis.merkle_root_hash = genesis.compute_merkle_root()
            genesis.block_hash = genesis.compute_hash()
            self.chain.append(genesis)

    def latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, entries: list[ModelEntry]) -> Block:
        """Create and mine a new block containing the given model entries.

        Each block links to the previous via `previous_hash`, forming the chain.
        Mining ensures the block hash meets the difficulty target.
        """
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            entries=entries,
            previous_hash=self.latest_block().block_hash,
        )
        new_block.mine(self.difficulty)
        self.chain.append(new_block)
        return new_block

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify the entire chain's integrity.

        Three checks per block:
        1. Block hash is correctly computed from header
        2. Previous hash matches actual hash of prior block
        3. Merkle root matches recomputed root from entries

        Returns (is_valid, message). On failure, the message identifies
        exactly which block and which check failed — crucial for forensics.
        """
        for i in range(1, len(self.chain)):
            block = self.chain[i]
            prev_block = self.chain[i - 1]

            # Check 1: block hash integrity
            if block.block_hash != block.compute_hash():
                return False, f"Block {i}: hash mismatch (block was modified)"

            # Check 2: chain linkage
            if block.previous_hash != prev_block.block_hash:
                return False, f"Block {i}: broken chain link to block {i-1}"

            # Check 3: Merkle root integrity
            expected_merkle = block.compute_merkle_root()
            if block.merkle_root_hash != expected_merkle:
                return False, f"Block {i}: Merkle root mismatch (entries modified)"

        return True, "Chain integrity verified"


# =============================================================================
# 5. MODEL REGISTRY — The high-level API
# =============================================================================

class ModelRegistry:
    """On-chain ML model registry.

    This is the user-facing API that wraps the blockchain. It provides:
    - Model registration with automatic signing
    - Lookup by model hash or name
    - Full verification (chain + signatures)
    - Inclusion proofs for specific models

    The registry also maintains a key mapping (public -> private) for
    signature verification. In a real system, only public keys would be
    stored, and verification would use ECDSA math directly.
    """

    def __init__(self, difficulty: int = 2) -> None:
        self.blockchain = Blockchain(difficulty=difficulty)
        # Pending entries waiting to be mined into a block
        self._pending: list[ModelEntry] = []
        # Key mapping for signature verification (simulation only)
        self._key_map: dict[str, str] = {}
        # Index: model_hash -> (block_index, entry_index) for fast lookup
        self._index: dict[str, tuple[int, int]] = {}

    def register_keypair(self, private_key: str, public_key: str) -> None:
        """Register a key pair for signature verification."""
        self._key_map[public_key] = private_key

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

        This is the primary API. It:
        1. Hashes the model weights for content-addressed identity
        2. Hashes the training data for provenance tracking
        3. Creates a ModelEntry with all metadata
        4. Signs the entry with the owner's private key
        5. Optionally mines it into a new block immediately

        Args:
            model_weights: Serialized model weights (or simulated string)
            name: Human-readable model name
            version: Semantic version string
            architecture: Model architecture description
            hyperparameters: Training configuration dict
            training_data: Training dataset (or simulated string)
            metrics: Evaluation metrics dict
            private_key: Owner's private key for signing
            public_key: Owner's public key for identity
            auto_mine: If True, immediately mine a block for this entry

        Returns:
            The finalized ModelEntry with hash and signature
        """
        # Content-addressed identity: hash the weights, not the name
        model_hash = sha256(model_weights)
        training_data_hash = sha256(training_data)

        entry = ModelEntry(
            model_hash=model_hash,
            name=name,
            version=version,
            architecture=architecture,
            hyperparameters=hyperparameters,
            training_data_hash=training_data_hash,
            metrics=metrics,
            owner=public_key,
            timestamp=time.time(),
        )
        entry.finalize(private_key)

        # Register key mapping for later verification
        self.register_keypair(private_key, public_key)

        if auto_mine:
            block = self.blockchain.add_block([entry])
            self._index[model_hash] = (block.index, 0)
        else:
            self._pending.append(entry)

        return entry

    def mine_pending(self) -> Optional[Block]:
        """Mine all pending entries into a new block."""
        if not self._pending:
            return None
        block = self.blockchain.add_block(self._pending)
        for i, entry in enumerate(self._pending):
            self._index[entry.model_hash] = (block.index, i)
        self._pending = []
        return block

    def get_model(self, model_hash: str) -> Optional[ModelEntry]:
        """Look up a model by its content hash."""
        location = self._index.get(model_hash)
        if location is None:
            return None
        block_idx, entry_idx = location
        return self.blockchain.chain[block_idx].entries[entry_idx]

    def get_model_by_name(self, name: str) -> list[ModelEntry]:
        """Find all model versions registered under a given name."""
        results = []
        for block in self.blockchain.chain:
            for entry in block.entries:
                if entry.name == name:
                    results.append(entry)
        return results

    def list_models(self) -> list[ModelEntry]:
        """List all registered models across all blocks."""
        models = []
        for block in self.blockchain.chain:
            models.extend(block.entries)
        return models

    def verify_chain(self) -> tuple[bool, str]:
        """Verify blockchain integrity."""
        return self.blockchain.verify_integrity()

    def verify_model(self, model_hash: str) -> tuple[bool, str]:
        """Verify a specific model's integrity and ownership.

        Three-level verification:
        1. Entry hash: recompute from metadata, compare to stored hash
        2. Signature: verify the owner's signature over the entry hash
        3. Merkle proof: verify the entry is included in its block

        This is the full audit trail — if all three pass, the model metadata
        is exactly what the owner registered, and it hasn't been tampered with.
        """
        location = self._index.get(model_hash)
        if location is None:
            return False, "Model not found in registry"

        block_idx, entry_idx = location
        block = self.blockchain.chain[block_idx]
        entry = block.entries[entry_idx]

        # Check 1: entry hash integrity
        recomputed_hash = entry.compute_hash()
        if recomputed_hash != entry.entry_hash:
            return False, "Entry hash mismatch — metadata was tampered with"

        # Check 2: signature verification
        private_key = self._key_map.get(entry.owner)
        if private_key is None:
            return False, "Owner's key not found — cannot verify signature"
        if not verify_signature(private_key, entry.entry_hash, entry.signature):
            return False, "Signature invalid — entry was not signed by claimed owner"

        # Check 3: Merkle inclusion proof
        entry_hashes = [e.entry_hash for e in block.entries]
        proof = merkle_proof(entry_hashes, entry_idx)
        if not verify_merkle_proof(entry.entry_hash, proof, block.merkle_root_hash):
            return False, "Merkle proof failed — entry not properly included in block"

        return True, f"Model verified: hash={model_hash[:16]}... in block {block_idx}"

    def get_inclusion_proof(self, model_hash: str) -> Optional[dict]:
        """Get a Merkle inclusion proof for a model.

        This proof can be shared with a third party who has the block header
        (containing the Merkle root) to verify the model exists in the registry
        without downloading all entries in the block.
        """
        location = self._index.get(model_hash)
        if location is None:
            return None

        block_idx, entry_idx = location
        block = self.blockchain.chain[block_idx]
        entry_hashes = [e.entry_hash for e in block.entries]
        proof = merkle_proof(entry_hashes, entry_idx)

        return {
            "model_hash": model_hash,
            "entry_hash": block.entries[entry_idx].entry_hash,
            "block_index": block_idx,
            "merkle_root": block.merkle_root_hash,
            "proof": proof,
        }


# =============================================================================
# 6. DEMONSTRATION
# =============================================================================

def simulate_model_weights(arch: str, version: str) -> str:
    """Simulate serialized model weights as a deterministic string.

    In production, this would be the actual bytes of saved model weights
    (e.g., PyTorch state_dict serialized with torch.save).
    """
    return f"WEIGHTS:{arch}:v{version}:{os.urandom(16).hex()}"


def simulate_training_data(name: str) -> str:
    """Simulate training data as a string for hashing."""
    return f"DATASET:{name}:{os.urandom(8).hex()}"


if __name__ == "__main__":
    print("=" * 70)
    print("ON-CHAIN ML MODEL REGISTRY")
    print("=" * 70)

    # --- Setup: Create the registry ---
    registry = ModelRegistry(difficulty=2)
    print(f"\nRegistry initialized with difficulty={registry.blockchain.difficulty}")
    print(f"Genesis block hash: {registry.blockchain.chain[0].block_hash[:32]}...")

    # --- Generate key pairs for two different "owners" ---
    alice_priv, alice_pub = generate_keypair()
    bob_priv, bob_pub = generate_keypair()
    print(f"\nAlice's public key: {alice_pub[:32]}...")
    print(f"Bob's public key:   {bob_pub[:32]}...")

    # --- Register models ---
    print("\n" + "-" * 70)
    print("REGISTERING MODELS")
    print("-" * 70)

    # Model 1: Alice registers a sentiment classifier
    weights1 = simulate_model_weights("LSTM", "1.0.0")
    data1 = simulate_training_data("imdb_reviews_50k")
    entry1 = registry.register_model(
        model_weights=weights1,
        name="sentiment-classifier",
        version="1.0.0",
        architecture="LSTM (128 hidden, 2 layers)",
        hyperparameters={"lr": 0.001, "epochs": 20, "batch_size": 64},
        training_data=data1,
        metrics={"accuracy": 0.923, "f1": 0.918, "auc_roc": 0.961},
        private_key=alice_priv,
        public_key=alice_pub,
    )
    print(f"\n[1] Registered: {entry1.name} v{entry1.version}")
    print(f"    Model hash:  {entry1.model_hash[:32]}...")
    print(f"    Entry hash:  {entry1.entry_hash[:32]}...")
    print(f"    Signature:   {entry1.signature[:32]}...")
    print(f"    Owner:       Alice ({entry1.owner[:16]}...)")

    # Model 2: Alice registers v2 of the same model (improved)
    weights2 = simulate_model_weights("Transformer", "2.0.0")
    entry2 = registry.register_model(
        model_weights=weights2,
        name="sentiment-classifier",
        version="2.0.0",
        architecture="DistilBERT fine-tuned",
        hyperparameters={"lr": 2e-5, "epochs": 3, "warmup_steps": 500},
        training_data=data1,  # Same training data
        metrics={"accuracy": 0.957, "f1": 0.954, "auc_roc": 0.983},
        private_key=alice_priv,
        public_key=alice_pub,
    )
    print(f"\n[2] Registered: {entry2.name} v{entry2.version}")
    print(f"    Model hash:  {entry2.model_hash[:32]}...")
    print(f"    Metrics improved: accuracy {0.923} -> {0.957}")

    # Model 3: Bob registers a different model
    weights3 = simulate_model_weights("RandomForest", "1.0.0")
    data3 = simulate_training_data("credit_scoring_100k")
    entry3 = registry.register_model(
        model_weights=weights3,
        name="credit-risk-scorer",
        version="1.0.0",
        architecture="RandomForest (500 trees, max_depth=15)",
        hyperparameters={"n_estimators": 500, "max_depth": 15, "min_samples_leaf": 5},
        training_data=data3,
        metrics={"accuracy": 0.891, "f1": 0.884, "ks_statistic": 0.723},
        private_key=bob_priv,
        public_key=bob_pub,
    )
    print(f"\n[3] Registered: {entry3.name} v{entry3.version}")
    print(f"    Model hash:  {entry3.model_hash[:32]}...")
    print(f"    Owner:       Bob ({entry3.owner[:16]}...)")

    # --- Inspect the blockchain ---
    print("\n" + "-" * 70)
    print("BLOCKCHAIN STATE")
    print("-" * 70)

    for block in registry.blockchain.chain:
        print(f"\n  Block {block.index}:")
        print(f"    Hash:          {block.block_hash[:32]}...")
        print(f"    Previous:      {block.previous_hash[:32]}...")
        print(f"    Merkle root:   {block.merkle_root_hash[:32]}...")
        print(f"    Nonce:         {block.nonce}")
        print(f"    Entries:       {len(block.entries)}")
        for entry in block.entries:
            print(f"      - {entry.name} v{entry.version} ({entry.model_hash[:16]}...)")

    # --- Verify everything ---
    print("\n" + "-" * 70)
    print("VERIFICATION")
    print("-" * 70)

    # Full chain verification
    valid, msg = registry.verify_chain()
    print(f"\nChain integrity: {'PASS' if valid else 'FAIL'} — {msg}")

    # Individual model verification
    for entry in [entry1, entry2, entry3]:
        valid, msg = registry.verify_model(entry.model_hash)
        status = "PASS" if valid else "FAIL"
        print(f"Model {entry.name} v{entry.version}: {status} — {msg}")

    # --- Merkle inclusion proof ---
    print("\n" + "-" * 70)
    print("MERKLE INCLUSION PROOF")
    print("-" * 70)

    proof_data = registry.get_inclusion_proof(entry1.model_hash)
    if proof_data:
        print(f"\nProof for: {entry1.name} v{entry1.version}")
        print(f"  Block:       {proof_data['block_index']}")
        print(f"  Merkle root: {proof_data['merkle_root'][:32]}...")
        print(f"  Proof steps: {len(proof_data['proof'])}")
        for i, (sibling, direction) in enumerate(proof_data["proof"]):
            print(f"    Step {i+1}: combine with {sibling[:16]}... on {direction}")

        # Verify the proof independently
        verified = verify_merkle_proof(
            proof_data["entry_hash"],
            proof_data["proof"],
            proof_data["merkle_root"],
        )
        print(f"  Proof valid: {verified}")

    # --- Tamper detection demo ---
    print("\n" + "-" * 70)
    print("TAMPER DETECTION DEMO")
    print("-" * 70)

    print("\nScenario: Attacker modifies a model's claimed accuracy...")

    # Save original values
    original_metrics = entry1.metrics.copy()
    original_entry_hash = entry1.entry_hash

    # Tamper with the metrics
    entry1.metrics["accuracy"] = 0.999  # Inflate accuracy
    print(f"  Original accuracy: {original_metrics['accuracy']}")
    print(f"  Tampered accuracy: {entry1.metrics['accuracy']}")

    # Try to verify — should FAIL
    valid, msg = registry.verify_model(entry1.model_hash)
    print(f"\n  Verification result: {'PASS' if valid else 'FAIL'}")
    print(f"  Message: {msg}")

    # The entry hash no longer matches because metrics changed
    recomputed = entry1.compute_hash()
    print(f"\n  Stored entry hash:     {original_entry_hash[:32]}...")
    print(f"  Recomputed entry hash: {recomputed[:32]}...")
    print(f"  Match: {original_entry_hash == recomputed}")

    # Restore original for chain verification
    entry1.metrics = original_metrics

    # Now try tampering with a block's previous_hash
    print("\nScenario: Attacker tries to break the chain link...")
    if len(registry.blockchain.chain) > 2:
        original_prev_hash = registry.blockchain.chain[2].previous_hash
        registry.blockchain.chain[2].previous_hash = sha256("fake_previous_block")

        valid, msg = registry.verify_chain()
        print(f"  Verification result: {'PASS' if valid else 'FAIL'}")
        print(f"  Message: {msg}")

        # Restore
        registry.blockchain.chain[2].previous_hash = original_prev_hash

    # --- Lookup demo ---
    print("\n" + "-" * 70)
    print("MODEL LOOKUP")
    print("-" * 70)

    # Find all versions of sentiment-classifier
    versions = registry.get_model_by_name("sentiment-classifier")
    print(f"\nAll versions of 'sentiment-classifier': {len(versions)} found")
    for v in versions:
        print(f"  v{v.version}: accuracy={v.metrics['accuracy']}, hash={v.model_hash[:16]}...")

    # List all models
    all_models = registry.list_models()
    print(f"\nTotal models in registry: {len(all_models)}")

    # Final chain verification (should pass after we restored tampering)
    valid, msg = registry.verify_chain()
    print(f"\nFinal chain integrity check: {'PASS' if valid else 'FAIL'} — {msg}")

    print("\n" + "=" * 70)
    print("REGISTRY DEMO COMPLETE")
    print("=" * 70)
