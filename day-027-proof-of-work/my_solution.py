"""
Day 027: Proof of Work Simulation — Your Implementation

Build a complete Proof of Work consensus mechanism from scratch.

Hints:
- Review Day 002 (SHA-256) — PoW uses hash functions as computational puzzles
- The mining loop is brute force: try nonces until the hash meets the target
- Difficulty = number of leading hex zeros required in the hash
- Difficulty adjustment keeps block time stable regardless of hash power
- Chain validation must check: hash integrity, linkage, PoW compliance, index order
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Block:
    """
    Represents a single block in the blockchain.

    Fields: index, timestamp, data, previous_hash, nonce, hash,
            mining_time (metadata), attempts (metadata)
    """
    index: int
    timestamp: float
    data: str
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    mining_time: float = 0.0
    attempts: int = 0

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of the block's contents (index, timestamp, data,
        previous_hash, nonce). Use json.dumps with sort_keys for deterministic output.

        Hint: mining_time and attempts are metadata — do NOT include them in the hash.
        """
        raise NotImplementedError("TODO: implement this")


class ProofOfWork:
    """
    Core PoW mining engine with difficulty adjustment.

    Hint: Think about what state the engine needs to track:
    - The chain of blocks
    - Current difficulty
    - Configuration (target block time, adjustment interval)
    - Difficulty history for analysis
    """

    def __init__(
        self,
        initial_difficulty: int = 4,
        target_block_time: float = 2.0,
        adjustment_interval: int = 5,
    ):
        """
        Initialize the PoW engine and create the genesis block.

        Hint: The genesis block is special — it has no previous block,
        uses "0"*64 as previous_hash, and is NOT mined (no PoW required).
        """
        raise NotImplementedError("TODO: implement this")

    def get_target(self) -> str:
        """
        Convert difficulty to a target prefix string.

        Hint: If difficulty=3, the target is "000" — the hash must start
        with 3 zeros in hexadecimal.
        """
        raise NotImplementedError("TODO: implement this")

    def mine_block(self, data: str) -> Block:
        """
        Mine a new block by brute-force searching for a valid nonce.

        Steps:
        1. Create a new Block with the next index and current timestamp
        2. Set previous_hash to the last block's hash
        3. Try nonces starting from 0 until hash starts with target prefix
        4. Record mining_time and attempts for analysis
        5. Append to chain
        6. Check if difficulty adjustment is needed

        Hint: The nonce search is O(16^difficulty) expected attempts.
        """
        raise NotImplementedError("TODO: implement this")

    def _adjust_difficulty(self) -> None:
        """
        Adjust difficulty based on actual vs expected block times.

        Hint: Compare the time taken for the last `adjustment_interval` blocks
        against the expected time. If blocks are >2x too fast, increase difficulty.
        If >2x too slow, decrease (but never below 1).
        """
        raise NotImplementedError("TODO: implement this")

    def validate_chain(self, chain: Optional[List[Block]] = None) -> tuple[bool, str]:
        """
        Validate the entire blockchain.

        Checks for each block (except genesis):
        1. Stored hash matches recomputed hash
        2. previous_hash matches prior block's hash
        3. Indices are sequential

        Returns (is_valid, error_message).

        Hint: Don't forget to handle the genesis block differently.
        """
        raise NotImplementedError("TODO: implement this")


def simulate_attack(
    attacker_hash_fraction: float,
    blocks_behind: int,
    simulations: int = 10000,
) -> float:
    """
    Monte Carlo simulation of a 51% attack.

    Each round: honest chain extends with prob (1-q), attacker with prob q.
    Attacker succeeds if deficit reaches 0.

    Hint: Use random.random() < q to decide who mines each round.
    Cap iterations to prevent infinite loops when q >= 0.5.
    """
    raise NotImplementedError("TODO: implement this")


def theoretical_attack_probability(q: float, z: int) -> float:
    """
    Nakamoto's formula: P = (q/p)^z where p = 1-q.

    Hint: This is only valid for q < 0.5. At q >= 0.5, return 1.0.
    """
    raise NotImplementedError("TODO: implement this")


def analyze_mining_stats(chain: List[Block]) -> dict:
    """
    Compute mining statistics from the chain (excluding genesis).

    Metrics: blocks_mined, total_attempts, total_mining_time,
    avg_hash_rate, avg_block_time, min/max/avg attempts.

    Hint: Hash rate = attempts / mining_time for each block.
    """
    raise NotImplementedError("TODO: implement this")


if __name__ == "__main__":
    print("=" * 70)
    print("PROOF OF WORK SIMULATION — Your Implementation")
    print("=" * 70)

    # Phase 1: Mine a blockchain
    print("\n--- Phase 1: Mining Blockchain ---\n")
    pow_engine = ProofOfWork(
        initial_difficulty=3,
        target_block_time=0.5,
        adjustment_interval=4,
    )

    for i in range(1, 13):
        block = pow_engine.mine_block(f"Transaction batch #{i}")
        hash_rate = block.attempts / block.mining_time if block.mining_time > 0 else 0
        print(
            f"  Block {block.index:>3d} | "
            f"Attempts: {block.attempts:>8,d} | "
            f"Time: {block.mining_time:>6.3f}s | "
            f"Hash: {block.hash[:20]}..."
        )

    # Phase 2: Validate chain
    print("\n--- Phase 2: Chain Validation ---\n")
    is_valid, message = pow_engine.validate_chain()
    print(f"  Chain valid: {is_valid} ({message})")

    # Phase 3: Mining stats
    print("\n--- Phase 3: Mining Statistics ---\n")
    stats = analyze_mining_stats(pow_engine.chain)
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        print(f"  {label}: {value}")

    # Phase 4: Attack simulation
    print("\n--- Phase 4: Attack Simulation ---\n")
    for q in [0.1, 0.3, 0.5]:
        p = simulate_attack(q, 6, simulations=5000)
        t = theoretical_attack_probability(q, 6)
        print(f"  q={q:.1f}: simulated={p*100:.2f}%, theory={t*100:.2f}%")

    print("\nDone!")
