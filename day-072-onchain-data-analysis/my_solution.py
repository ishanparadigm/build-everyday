"""
Day 072: On-Chain Data Analysis with Python — Your Implementation

Build an on-chain analytics engine that processes blockchain transaction data
to produce wallet profiles, token flow graphs, time-series anomaly detection,
and wealth distribution metrics.

Implement each function below. Run `python3 tests.py` to check your work.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1. DATA MODEL (provided — these are your data structures)
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """A single blockchain transaction."""
    tx_hash: str
    block_number: int
    timestamp: int
    from_addr: str
    to_addr: str
    value: float
    gas_used: int
    gas_price: float
    is_token_transfer: bool = False
    token_amount: float = 0.0

    @property
    def gas_cost_eth(self) -> float:
        return self.gas_used * self.gas_price * 1e-9


@dataclass
class AddressProfile:
    """Statistical profile of an on-chain address."""
    address: str
    tx_count: int = 0
    sent_count: int = 0
    received_count: int = 0
    total_sent: float = 0.0
    total_received: float = 0.0
    unique_counterparties: int = 0
    avg_gas_price: float = 0.0
    max_gas_price: float = 0.0
    first_seen: int = 0
    last_seen: int = 0
    label: str = "unknown"


@dataclass
class FlowEdge:
    """An edge in the token flow graph."""
    from_addr: str
    to_addr: str
    total_value: float = 0.0
    tx_count: int = 0


@dataclass
class TimeSeriesPoint:
    """A single data point in a time-series aggregation."""
    timestamp: int
    block_number: int
    tx_count: int = 0
    total_volume: float = 0.0
    unique_addresses: int = 0
    avg_gas_price: float = 0.0
    max_value: float = 0.0


@dataclass
class Anomaly:
    """A detected anomaly in the time series."""
    block_number: int
    timestamp: int
    metric_name: str
    observed_value: float
    expected_mean: float
    z_score: float
    severity: str


# ---------------------------------------------------------------------------
# 2. BLOCKCHAIN DATA SIMULATOR
# ---------------------------------------------------------------------------

def generate_address(prefix: str, index: int) -> str:
    """Generate a deterministic hex address from a prefix and index.

    Hint: Use SHA-256 hash of the prefix + index string, take first 40 hex chars.
    """
    raise NotImplementedError("TODO: implement this")


def simulate_blockchain_data(
    n_blocks: int = 200,
    block_time: int = 12,
    start_timestamp: int = 1700000000,
    seed: int = 42,
) -> list[Transaction]:
    """Generate realistic blockchain transaction data.

    Hint: Create pools of addresses (whales, bots, users, contracts, exchanges)
    with different behavioral patterns:
    - Bots: high frequency, high gas, interact with contracts
    - Whales: low frequency, high value
    - Users: moderate everything
    - Inject anomaly spike around block 148-152 (3x normal tx count)
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# 3. TRANSACTION INDEXER
# ---------------------------------------------------------------------------

class TransactionIndex:
    """Index transactions for efficient querying.

    Hint: Build dictionaries keyed by address and block number.
    Each address entry should include all txs where the address is sender OR receiver.
    """

    def __init__(self, transactions: list[Transaction]) -> None:
        raise NotImplementedError("TODO: implement this")

    def get_address_txs(self, address: str) -> list[Transaction]:
        """All transactions involving an address."""
        raise NotImplementedError("TODO: implement this")

    def get_block_txs(self, block: int) -> list[Transaction]:
        """All transactions in a specific block."""
        raise NotImplementedError("TODO: implement this")

    def get_time_range(self, start_ts: int, end_ts: int) -> list[Transaction]:
        """All transactions within a timestamp range."""
        raise NotImplementedError("TODO: implement this")

    @property
    def unique_addresses(self) -> set[str]:
        raise NotImplementedError("TODO: implement this")

    @property
    def block_range(self) -> tuple[int, int]:
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# 4. ADDRESS PROFILER
# ---------------------------------------------------------------------------

def profile_address(address: str, index: TransactionIndex) -> AddressProfile:
    """Compute a statistical profile for a single address.

    Hint: Separate sent vs received txs, count unique counterparties,
    compute gas price stats, track first/last seen timestamps.
    """
    raise NotImplementedError("TODO: implement this")


def classify_address(profile: AddressProfile) -> str:
    """Classify an address based on behavioral features.

    Hint: Use these features in priority order:
    1. Exchange: unique_counterparties > 40
    2. Bot/MEV: high tx_per_hour (>5) AND avg_gas_price > 45
    3. Contract: receives much more than sends, many counterparties
    4. Whale: total volume > 500
    5. Default: regular_user
    """
    raise NotImplementedError("TODO: implement this")


def profile_all_addresses(index: TransactionIndex) -> dict[str, AddressProfile]:
    """Profile and classify every address in the dataset."""
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# 5. TOKEN FLOW GRAPH
# ---------------------------------------------------------------------------

class TokenFlowGraph:
    """Directed weighted graph of value transfers.

    Hint: Use a nested dict: edges[from][to] = FlowEdge
    """

    def __init__(self) -> None:
        raise NotImplementedError("TODO: implement this")

    @classmethod
    def from_transactions(cls, transactions: list[Transaction]) -> TokenFlowGraph:
        """Build flow graph from raw transactions."""
        raise NotImplementedError("TODO: implement this")

    def add_transfer(self, from_addr: str, to_addr: str, value: float) -> None:
        """Add or update an edge."""
        raise NotImplementedError("TODO: implement this")

    def out_degree(self, addr: str) -> int:
        """Number of unique addresses this address has sent to."""
        raise NotImplementedError("TODO: implement this")

    def in_degree(self, addr: str) -> int:
        """Number of unique addresses that have sent to this address."""
        raise NotImplementedError("TODO: implement this")

    def pagerank(self, damping: float = 0.85, iterations: int = 50) -> dict[str, float]:
        """Compute PageRank over the flow graph.

        Hint: The formula is PR(v) = (1-d)/N + d * Σ_{u->v} PR(u) / out_degree(u)
        Start with uniform 1/N, iterate until stable.
        """
        raise NotImplementedError("TODO: implement this")

    def connected_components(self) -> list[set[str]]:
        """Find connected components (undirected).

        Hint: BFS/DFS over the undirected version of the graph.
        """
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# 6. TIME-SERIES ANALYTICS
# ---------------------------------------------------------------------------

def aggregate_by_block(transactions: list[Transaction]) -> list[TimeSeriesPoint]:
    """Aggregate transaction metrics per block.

    Hint: Group txs by block number, compute count, volume, unique addresses,
    avg gas price per block.
    """
    raise NotImplementedError("TODO: implement this")


def rolling_mean_std(values: list[float], window: int) -> list[tuple[float, float]]:
    """Compute rolling mean and standard deviation.

    Hint: For position i, use values[max(0, i-window+1):i+1].
    Use sample std (divide by n-1).
    """
    raise NotImplementedError("TODO: implement this")


def detect_anomalies(
    time_series: list[TimeSeriesPoint],
    window: int = 20,
    z_threshold: float = 2.5,
) -> list[Anomaly]:
    """Detect anomalies using rolling z-score.

    Hint: For each metric (tx_count, total_volume, avg_gas_price):
    1. Compute rolling mean/std
    2. Skip warmup period (first `window` points)
    3. Flag points where |z-score| > threshold
    4. Severity: >5 = high, >3.5 = medium, else low
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# 7. WEALTH DISTRIBUTION
# ---------------------------------------------------------------------------

def compute_balances(transactions: list[Transaction]) -> dict[str, float]:
    """Compute net balance for each address.

    Hint: Start with base balance of 100 for each address, subtract sends,
    add receives. Clamp negatives to 0.
    """
    raise NotImplementedError("TODO: implement this")


def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient.

    Hint: Sort values ascending, then:
    G = (2 * Σᵢ (i+1) * xᵢ) / (n * Σ xᵢ) - (n + 1) / n
    """
    raise NotImplementedError("TODO: implement this")


def lorenz_curve(values: list[float]) -> list[tuple[float, float]]:
    """Compute Lorenz curve data points.

    Hint: Sort ascending, compute cumulative wealth / total wealth
    for each cumulative population fraction.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# MAIN — Test your implementation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating blockchain data...")
    transactions = simulate_blockchain_data(n_blocks=200, seed=42)
    print(f"Generated {len(transactions)} transactions")

    print("\nBuilding index...")
    index = TransactionIndex(transactions)
    print(f"Unique addresses: {len(index.unique_addresses)}")

    print("\nProfiling addresses...")
    profiles = profile_all_addresses(index)
    labels = defaultdict(int)
    for p in profiles.values():
        labels[p.label] += 1
    print(f"Classifications: {dict(labels)}")

    print("\nBuilding flow graph...")
    graph = TokenFlowGraph.from_transactions(transactions)
    print(f"Nodes: {len(graph.nodes)}, Components: {len(graph.connected_components())}")

    print("\nDetecting anomalies...")
    ts = aggregate_by_block(transactions)
    anomalies = detect_anomalies(ts)
    print(f"Found {len(anomalies)} anomalies")

    print("\nWealth distribution...")
    balances = compute_balances(transactions)
    gini = gini_coefficient(list(balances.values()))
    print(f"Gini coefficient: {gini:.4f}")

    print("\nDone! Run tests.py to verify your implementation.")
