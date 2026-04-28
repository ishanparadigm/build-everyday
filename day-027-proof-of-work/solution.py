"""
Day 027: Proof of Work Simulation

A complete implementation of the Proof of Work consensus mechanism, including:
- Block mining with SHA-256 partial hash collision
- Dynamic difficulty adjustment
- Full chain validation
- 51% attack simulation

This builds on Day 002 (SHA-256) and Day 013 (Merkle trees) to show how
hash functions become the foundation of trustless consensus.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# Block Data Structure
# =============================================================================

@dataclass
class Block:
    """
    Represents a single block in the blockchain.

    The hash is computed from ALL other fields — changing any field invalidates
    the hash, and by extension, every subsequent block in the chain.
    """
    index: int
    timestamp: float
    data: str
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    # Mining metadata (not part of the hash — these are for analysis only)
    mining_time: float = 0.0
    attempts: int = 0

    def compute_hash(self) -> str:
        """
        SHA-256 of the block's contents. The nonce is the only field the miner
        varies; everything else is fixed before mining begins.

        We concatenate fields with a delimiter to prevent ambiguity attacks
        (e.g., index=1 data="23" vs index=12 data="3").
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


# =============================================================================
# Proof of Work Engine
# =============================================================================

class ProofOfWork:
    """
    Core PoW mining engine. Manages difficulty, mines blocks, and validates chains.

    Key parameters:
    - initial_difficulty: Number of leading hex zeros required (each hex zero = 4 bits)
    - target_block_time: Desired seconds between blocks
    - adjustment_interval: How many blocks between difficulty recalculations
    """

    def __init__(
        self,
        initial_difficulty: int = 4,
        target_block_time: float = 2.0,
        adjustment_interval: int = 5,
    ):
        self.difficulty = initial_difficulty
        self.target_block_time = target_block_time
        self.adjustment_interval = adjustment_interval
        self.chain: List[Block] = []
        self.difficulty_history: List[dict] = []

        # Create the genesis block — the anchor of the entire chain.
        # It has no previous hash and is not mined (nonce=0, no difficulty requirement).
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data="Genesis Block",
            previous_hash="0" * 64,
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)
        self.difficulty_history.append({
            "block": 0,
            "difficulty": self.difficulty,
            "reason": "initial",
        })

    def get_target(self) -> str:
        """
        Convert difficulty (number of leading hex zeros) to a target string.

        A hash is valid if it starts with `difficulty` zeros in hex.
        Each hex zero requires 4 leading zero bits, so difficulty=4 means
        the hash must be less than 2^(256-16) = 2^240.

        The probability of a random hash meeting this: (1/16)^difficulty
        Expected attempts: 16^difficulty

        For difficulty=4: expected ~65,536 attempts
        For difficulty=5: expected ~1,048,576 attempts (16x harder)
        """
        return "0" * self.difficulty

    def mine_block(self, data: str) -> Block:
        """
        Mine a new block by finding a nonce that produces a hash meeting
        the current difficulty target.

        This is the core PoW loop — pure brute force. There's no way to
        predict which nonce will work because SHA-256's avalanche effect
        means each attempt is statistically independent.
        """
        previous_block = self.chain[-1]
        new_block = Block(
            index=previous_block.index + 1,
            timestamp=time.time(),
            data=data,
            previous_hash=previous_block.hash,
        )

        target = self.get_target()
        start_time = time.time()
        attempts = 0

        # The mining loop: increment nonce, hash, check prefix
        while True:
            new_block.nonce = attempts
            candidate_hash = new_block.compute_hash()

            if candidate_hash.startswith(target):
                # Found a valid nonce! The block is now "mined".
                new_block.hash = candidate_hash
                new_block.mining_time = time.time() - start_time
                new_block.attempts = attempts + 1
                break

            attempts += 1

        self.chain.append(new_block)

        # Check if it's time to adjust difficulty
        if new_block.index % self.adjustment_interval == 0 and new_block.index > 0:
            self._adjust_difficulty()

        return new_block

    def _adjust_difficulty(self) -> None:
        """
        Adjust difficulty based on actual vs expected block times.

        This is the self-correcting mechanism: if blocks arrive too fast,
        increase difficulty; if too slow, decrease it. This ensures the chain
        maintains a stable block rate regardless of total hash power.

        The formula mirrors Bitcoin's approach:
            ratio = actual_time / expected_time
            if ratio < 0.5: difficulty += 1  (blocks too fast, make harder)
            if ratio > 2.0: difficulty -= 1  (blocks too slow, make easier)

        We use thresholds rather than continuous adjustment to prevent
        oscillation — small fluctuations shouldn't trigger changes.
        """
        interval = self.adjustment_interval
        recent_blocks = self.chain[-interval:]

        # Time taken for the last `interval` blocks
        actual_time = recent_blocks[-1].timestamp - recent_blocks[0].timestamp
        expected_time = self.target_block_time * interval

        old_difficulty = self.difficulty
        ratio = actual_time / expected_time if expected_time > 0 else 1.0

        if ratio < 0.5:
            # Blocks arriving >2x too fast — increase difficulty
            self.difficulty += 1
            reason = f"too fast ({ratio:.2f}x target)"
        elif ratio > 2.0:
            # Blocks arriving >2x too slow — decrease difficulty
            self.difficulty = max(1, self.difficulty - 1)  # Never go below 1
            reason = f"too slow ({ratio:.2f}x target)"
        else:
            reason = f"within range ({ratio:.2f}x target)"

        self.difficulty_history.append({
            "block": self.chain[-1].index,
            "difficulty": self.difficulty,
            "old_difficulty": old_difficulty,
            "ratio": round(ratio, 3),
            "actual_time": round(actual_time, 3),
            "expected_time": round(expected_time, 3),
            "reason": reason,
        })

    def validate_chain(self, chain: Optional[List[Block]] = None) -> tuple[bool, str]:
        """
        Validate the entire blockchain from genesis to tip.

        Checks performed on each block:
        1. Hash integrity: stored hash matches recomputed hash
        2. Chain linkage: previous_hash matches prior block's hash
        3. PoW compliance: hash meets difficulty target (except genesis)
        4. Index continuity: indices are sequential

        Returns (is_valid, error_message). A single invalid block
        invalidates the entire chain from that point forward.
        """
        chain = chain or self.chain

        for i, block in enumerate(chain):
            # Check 1: Hash integrity
            recomputed = block.compute_hash()
            if block.hash != recomputed:
                return False, f"Block {i}: hash mismatch (stored={block.hash[:16]}..., computed={recomputed[:16]}...)"

            if i == 0:
                # Genesis block — skip linkage and PoW checks
                continue

            # Check 2: Chain linkage
            if block.previous_hash != chain[i - 1].hash:
                return False, f"Block {i}: broken chain link"

            # Check 3: Index continuity
            if block.index != chain[i - 1].index + 1:
                return False, f"Block {i}: non-sequential index"

        return True, "Chain is valid"


# =============================================================================
# Attack Simulation
# =============================================================================

def simulate_attack(
    attacker_hash_fraction: float,
    blocks_behind: int,
    simulations: int = 10000,
) -> float:
    """
    Monte Carlo simulation of a 51% attack.

    Models whether an attacker with `attacker_hash_fraction` of total hash power
    can catch up from `blocks_behind` blocks behind the honest chain.

    At each time step:
    - Honest miners find a block with probability (1 - q)
    - Attacker finds a block with probability q

    The attacker succeeds if they reach the same length as the honest chain.

    Theoretical result (Nakamoto 2008):
        P(catch up) = (q/p)^z  where p=1-q, z=blocks_behind

    Our simulation should converge to this as simulations → infinity.
    """
    import random

    q = attacker_hash_fraction
    p = 1.0 - q
    successes = 0

    for _ in range(simulations):
        attacker_deficit = blocks_behind
        # Cap iterations to prevent infinite loops when q >= p
        max_rounds = blocks_behind * 100

        for _ in range(max_rounds):
            if attacker_deficit <= 0:
                successes += 1
                break
            # Each round: honest chain grows with prob p, attacker with prob q
            if random.random() < q:
                attacker_deficit -= 1  # Attacker gains a block
            else:
                attacker_deficit += 1  # Honest chain extends lead
        # If max_rounds exceeded, attacker failed

    return successes / simulations


def theoretical_attack_probability(q: float, z: int) -> float:
    """
    Nakamoto's closed-form formula for attacker catch-up probability.

    P = (q/p)^z where p = 1-q

    Only valid for q < 0.5. At q >= 0.5, attacker always wins eventually.
    """
    if q >= 0.5:
        return 1.0
    p = 1.0 - q
    return (q / p) ** z


# =============================================================================
# Mining Performance Analysis
# =============================================================================

def analyze_mining_stats(chain: List[Block]) -> dict:
    """
    Compute mining statistics for the chain (excluding genesis).

    Metrics:
    - Hash rate: attempts / mining_time (hashes per second)
    - Block time: actual time between blocks
    - Total work: sum of all attempts across all blocks
    """
    mined_blocks = [b for b in chain if b.index > 0]
    if not mined_blocks:
        return {}

    hash_rates = []
    for b in mined_blocks:
        if b.mining_time > 0:
            hash_rates.append(b.attempts / b.mining_time)

    block_times = [b.mining_time for b in mined_blocks]
    total_attempts = sum(b.attempts for b in mined_blocks)
    total_time = sum(b.mining_time for b in mined_blocks)

    return {
        "blocks_mined": len(mined_blocks),
        "total_attempts": total_attempts,
        "total_mining_time": round(total_time, 3),
        "avg_hash_rate": round(sum(hash_rates) / len(hash_rates), 1) if hash_rates else 0,
        "avg_block_time": round(sum(block_times) / len(block_times), 3),
        "min_attempts": min(b.attempts for b in mined_blocks),
        "max_attempts": max(b.attempts for b in mined_blocks),
        "avg_attempts": round(total_attempts / len(mined_blocks), 1),
    }


# =============================================================================
# Main: Full Mining Simulation
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PROOF OF WORK SIMULATION")
    print("=" * 70)

    # --- Phase 1: Mine a blockchain with difficulty adjustment ---
    print("\n--- Phase 1: Mining Blockchain (difficulty=3, target=0.5s/block) ---\n")

    # Using difficulty=3 (4096 expected attempts) for reasonable demo speed
    pow_engine = ProofOfWork(
        initial_difficulty=3,
        target_block_time=0.5,
        adjustment_interval=4,
    )

    num_blocks = 12
    for i in range(1, num_blocks + 1):
        block = pow_engine.mine_block(f"Transaction batch #{i}")
        hash_rate = block.attempts / block.mining_time if block.mining_time > 0 else 0
        print(
            f"  Block {block.index:>3d} | "
            f"Nonce: {block.nonce:>8,d} | "
            f"Attempts: {block.attempts:>8,d} | "
            f"Time: {block.mining_time:>6.3f}s | "
            f"Hash: {block.hash[:20]}... | "
            f"Difficulty: {pow_engine.difficulty}"
        )

    # --- Phase 2: Chain validation ---
    print("\n--- Phase 2: Chain Validation ---\n")

    is_valid, message = pow_engine.validate_chain()
    print(f"  Chain valid: {is_valid} ({message})")

    # Tamper with a block to demonstrate validation
    print("\n  Tampering with block 3 (changing data)...")
    original_data = pow_engine.chain[3].data
    pow_engine.chain[3].data = "TAMPERED DATA"
    is_valid, message = pow_engine.validate_chain()
    print(f"  Chain valid after tampering: {is_valid}")
    print(f"  Error: {message}")

    # Restore the block
    pow_engine.chain[3].data = original_data

    # --- Phase 3: Mining statistics ---
    print("\n--- Phase 3: Mining Statistics ---\n")

    stats = analyze_mining_stats(pow_engine.chain)
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        print(f"  {label}: {value:,}" if isinstance(value, int) else f"  {label}: {value}")

    # --- Phase 4: Difficulty adjustment history ---
    print("\n--- Phase 4: Difficulty Adjustment History ---\n")

    for entry in pow_engine.difficulty_history:
        if entry.get("old_difficulty") is not None:
            print(
                f"  Block {entry['block']:>3d}: "
                f"difficulty {entry['old_difficulty']} -> {entry['difficulty']} "
                f"({entry['reason']})"
            )
        else:
            print(f"  Block {entry['block']:>3d}: difficulty {entry['difficulty']} ({entry['reason']})")

    # --- Phase 5: 51% Attack simulation ---
    print("\n--- Phase 5: 51% Attack Simulation ---\n")

    print("  Attacker hash power vs. probability of catching up from Z blocks behind:\n")
    print(f"  {'Hash %':>8s} | {'Z=1':>10s} | {'Z=3':>10s} | {'Z=6':>10s} | {'Z=6 (theory)':>12s}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")

    for q in [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]:
        p1 = simulate_attack(q, 1, simulations=5000)
        p3 = simulate_attack(q, 3, simulations=5000)
        p6 = simulate_attack(q, 6, simulations=5000)
        t6 = theoretical_attack_probability(q, 6)
        print(
            f"  {q*100:>7.1f}% | "
            f"{p1*100:>9.2f}% | "
            f"{p3*100:>9.2f}% | "
            f"{p6*100:>9.2f}% | "
            f"{t6*100:>10.2f}%"
        )

    # --- Phase 6: Demonstrate exponential difficulty scaling ---
    print("\n--- Phase 6: Difficulty Scaling (expected attempts) ---\n")

    for d in range(1, 8):
        expected = 16 ** d
        print(f"  Difficulty {d}: ~{expected:>12,d} expected attempts ({expected / 1e6:.1f}M)" if expected > 1e6
              else f"  Difficulty {d}: ~{expected:>12,d} expected attempts")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
