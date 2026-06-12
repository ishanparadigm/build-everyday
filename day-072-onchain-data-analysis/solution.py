"""
Day 072: On-Chain Data Analysis with Python

A complete on-chain analytics engine that processes blockchain transaction data
to produce wallet profiles, token flow graphs, time-series anomaly detection,
and wealth distribution metrics.

We simulate realistic blockchain data (multiple address types, varying patterns)
then analyze it using the same techniques real analytics platforms use:
statistical profiling, graph analysis, rolling z-score anomaly detection, and
Gini coefficient computation.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1. DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """A single blockchain transaction."""
    tx_hash: str
    block_number: int
    timestamp: int          # Unix seconds
    from_addr: str
    to_addr: str
    value: float            # In ETH (not wei, for readability)
    gas_used: int
    gas_price: float        # In Gwei
    is_token_transfer: bool = False
    token_amount: float = 0.0

    @property
    def gas_cost_eth(self) -> float:
        """Total gas cost in ETH."""
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
    first_seen: int = 0       # timestamp
    last_seen: int = 0        # timestamp
    label: str = "unknown"    # classified type


@dataclass
class FlowEdge:
    """An edge in the token flow graph (aggregated transfers between two addresses)."""
    from_addr: str
    to_addr: str
    total_value: float = 0.0
    tx_count: int = 0


# ---------------------------------------------------------------------------
# 2. BLOCKCHAIN DATA SIMULATOR
# ---------------------------------------------------------------------------

def generate_address(prefix: str, index: int) -> str:
    """Generate a deterministic hex address from a prefix and index.

    Real addresses are 20-byte hex strings. We create shorter readable ones
    for demonstration, but the analysis works identically on real addresses.
    """
    raw = f"{prefix}_{index}"
    return "0x" + hashlib.sha256(raw.encode()).hexdigest()[:40]


def simulate_blockchain_data(
    n_blocks: int = 200,
    block_time: int = 12,      # seconds between blocks (Ethereum-like)
    start_timestamp: int = 1700000000,
    seed: int = 42,
) -> list[Transaction]:
    """Generate realistic blockchain transaction data.

    We create several address types with distinct behavioral patterns:
    - Whales: few addresses, large infrequent transfers
    - Bots/MEV: high frequency, aggressive gas prices
    - Regular users: moderate activity, standard gas
    - Contracts (DEX, lending): receive many calls
    - Exchange hot wallets: very high counterparty count

    This mirrors real blockchain activity distributions where most volume
    comes from a small number of addresses (power law).
    """
    rng = random.Random(seed)

    # Create address pools with distinct behaviors
    whales = [generate_address("whale", i) for i in range(5)]
    bots = [generate_address("bot", i) for i in range(10)]
    users = [generate_address("user", i) for i in range(50)]
    contracts = [generate_address("contract", i) for i in range(8)]
    exchanges = [generate_address("exchange", i) for i in range(3)]

    all_addresses = whales + bots + users + contracts + exchanges

    transactions: list[Transaction] = []
    tx_counter = 0

    for block in range(n_blocks):
        block_number = 18_000_000 + block
        timestamp = start_timestamp + block * block_time

        # Number of transactions per block varies (Poisson-like)
        n_txs = max(1, int(rng.gauss(15, 5)))

        # Inject anomaly: a sudden volume spike around block 150
        # This is the kind of event our anomaly detector should catch
        if 148 <= block <= 152:
            n_txs = int(n_txs * 3)

        for _ in range(n_txs):
            tx_counter += 1
            tx_hash = "0x" + hashlib.sha256(f"tx_{tx_counter}".encode()).hexdigest()[:64]

            # Choose sender/receiver based on realistic probability distribution
            # Bots transact frequently, whales rarely but with large values
            roll = rng.random()
            if roll < 0.30:
                # Bot transaction: high frequency, moderate value, high gas
                from_addr = rng.choice(bots)
                to_addr = rng.choice(contracts)  # Bots mostly interact with contracts
                value = rng.uniform(0.1, 5.0)
                gas_price = rng.uniform(50, 200)  # Aggressive gas bidding
                is_token = rng.random() < 0.7
            elif roll < 0.45:
                # Whale transaction: infrequent, high value
                from_addr = rng.choice(whales)
                to_addr = rng.choice(contracts + exchanges)
                value = rng.uniform(50, 500)
                gas_price = rng.uniform(20, 40)
                is_token = rng.random() < 0.5
            elif roll < 0.80:
                # Regular user: moderate everything
                from_addr = rng.choice(users)
                to_addr = rng.choice(contracts + users + exchanges)
                value = rng.uniform(0.01, 2.0)
                gas_price = rng.uniform(15, 35)
                is_token = rng.random() < 0.4
            elif roll < 0.90:
                # Exchange withdrawal to user
                from_addr = rng.choice(exchanges)
                to_addr = rng.choice(users + whales)
                value = rng.uniform(0.5, 20.0)
                gas_price = rng.uniform(20, 30)
                is_token = rng.random() < 0.3
            else:
                # Contract-to-contract (internal-like)
                from_addr = rng.choice(contracts)
                to_addr = rng.choice(contracts)
                value = rng.uniform(1.0, 50.0)
                gas_price = rng.uniform(20, 35)
                is_token = rng.random() < 0.8

            gas_used = rng.randint(21000, 300000)
            token_amount = value * rng.uniform(100, 5000) if is_token else 0.0

            transactions.append(Transaction(
                tx_hash=tx_hash,
                block_number=block_number,
                timestamp=timestamp,
                from_addr=from_addr,
                to_addr=to_addr,
                value=round(value, 6),
                gas_used=gas_used,
                gas_price=round(gas_price, 2),
                is_token_transfer=is_token,
                token_amount=round(token_amount, 2),
            ))

    return transactions


# ---------------------------------------------------------------------------
# 3. TRANSACTION INDEXER
# ---------------------------------------------------------------------------

class TransactionIndex:
    """Index transactions for efficient querying by address, block, and time.

    In production, this would be a database (PostgreSQL, ClickHouse) with
    indexes on each column. Here we use in-memory dictionaries for the same
    O(1) lookup semantics.
    """

    def __init__(self, transactions: list[Transaction]) -> None:
        self.all_txs = transactions
        # Index by address (both sender and receiver)
        self.by_address: dict[str, list[Transaction]] = defaultdict(list)
        # Index by block number
        self.by_block: dict[int, list[Transaction]] = defaultdict(list)

        for tx in transactions:
            self.by_address[tx.from_addr].append(tx)
            self.by_address[tx.to_addr].append(tx)
            self.by_block[tx.block_number].append(tx)

    def get_address_txs(self, address: str) -> list[Transaction]:
        """All transactions involving an address (as sender or receiver)."""
        return self.by_address.get(address, [])

    def get_block_txs(self, block: int) -> list[Transaction]:
        """All transactions in a specific block."""
        return self.by_block.get(block, [])

    def get_time_range(self, start_ts: int, end_ts: int) -> list[Transaction]:
        """All transactions within a timestamp range.

        Linear scan — in production you'd use a B-tree index on timestamp.
        For our data sizes this is perfectly fine.
        """
        return [tx for tx in self.all_txs if start_ts <= tx.timestamp <= end_ts]

    @property
    def unique_addresses(self) -> set[str]:
        return set(self.by_address.keys())

    @property
    def block_range(self) -> tuple[int, int]:
        blocks = sorted(self.by_block.keys())
        return (blocks[0], blocks[-1])


# ---------------------------------------------------------------------------
# 4. ADDRESS PROFILER
# ---------------------------------------------------------------------------

def profile_address(address: str, index: TransactionIndex) -> AddressProfile:
    """Compute a statistical profile for a single address.

    We aggregate transaction data into features that distinguish address types.
    These same features are what analytics platforms like Nansen use for
    entity labeling (though they also incorporate known address databases).
    """
    txs = index.get_address_txs(address)
    if not txs:
        return AddressProfile(address=address)

    sent_txs = [tx for tx in txs if tx.from_addr == address]
    recv_txs = [tx for tx in txs if tx.to_addr == address]

    # Counterparty analysis: who does this address interact with?
    counterparties: set[str] = set()
    for tx in sent_txs:
        counterparties.add(tx.to_addr)
    for tx in recv_txs:
        counterparties.add(tx.from_addr)

    gas_prices = [tx.gas_price for tx in sent_txs] if sent_txs else [0.0]
    timestamps = [tx.timestamp for tx in txs]

    return AddressProfile(
        address=address,
        tx_count=len(txs),
        sent_count=len(sent_txs),
        received_count=len(recv_txs),
        total_sent=sum(tx.value for tx in sent_txs),
        total_received=sum(tx.value for tx in recv_txs),
        unique_counterparties=len(counterparties),
        avg_gas_price=sum(gas_prices) / len(gas_prices),
        max_gas_price=max(gas_prices),
        first_seen=min(timestamps),
        last_seen=max(timestamps),
    )


def classify_address(profile: AddressProfile) -> str:
    """Classify an address based on its behavioral profile.

    This is a heuristic classifier — real systems use ML models trained on
    labeled data. But the features are the same: frequency, volume, gas
    patterns, and counterparty counts.

    Classification rules (applied in priority order):
    1. Exchange: very high counterparty count (they interact with everyone)
    2. Bot/MEV: high tx frequency AND aggressive gas prices
    3. Whale: high total volume but lower frequency
    4. Contract: receives many more txs than it sends (called, not calling)
    5. Regular user: everything else
    """
    if profile.tx_count == 0:
        return "inactive"

    # Feature: transactions per active period (crude frequency)
    active_period = max(1, profile.last_seen - profile.first_seen)
    tx_per_hour = profile.tx_count / (active_period / 3600)

    total_volume = profile.total_sent + profile.total_received
    recv_ratio = profile.received_count / max(1, profile.tx_count)

    # Exchange: many unique counterparties
    if profile.unique_counterparties > 40:
        return "exchange"

    # Bot/MEV: high frequency + high gas prices
    if tx_per_hour > 5 and profile.avg_gas_price > 45:
        return "bot_mev"

    # Contract: mostly receives transactions
    if recv_ratio > 0.65 and profile.unique_counterparties > 15:
        return "contract"

    # Whale: high volume, moderate frequency
    if total_volume > 500:
        return "whale"

    return "regular_user"


def profile_all_addresses(index: TransactionIndex) -> dict[str, AddressProfile]:
    """Profile and classify every address in the dataset."""
    profiles: dict[str, AddressProfile] = {}
    for addr in index.unique_addresses:
        profile = profile_address(addr, index)
        profile.label = classify_address(profile)
        profiles[addr] = profile
    return profiles


# ---------------------------------------------------------------------------
# 5. TOKEN FLOW GRAPH
# ---------------------------------------------------------------------------

class TokenFlowGraph:
    """Directed weighted graph of token/value transfers between addresses.

    This is the core data structure for tracing fund flows. Each edge
    aggregates all transfers between a pair of addresses. Graph metrics
    like degree distribution and PageRank reveal the structure of economic
    activity on-chain.
    """

    def __init__(self) -> None:
        # Adjacency: from_addr -> {to_addr -> FlowEdge}
        self.edges: dict[str, dict[str, FlowEdge]] = defaultdict(dict)
        self.nodes: set[str] = set()

    @classmethod
    def from_transactions(cls, transactions: list[Transaction]) -> TokenFlowGraph:
        """Build flow graph from raw transactions."""
        graph = cls()
        for tx in transactions:
            graph.add_transfer(tx.from_addr, tx.to_addr, tx.value)
        return graph

    def add_transfer(self, from_addr: str, to_addr: str, value: float) -> None:
        """Add or update an edge in the flow graph."""
        self.nodes.add(from_addr)
        self.nodes.add(to_addr)
        if to_addr not in self.edges[from_addr]:
            self.edges[from_addr][to_addr] = FlowEdge(from_addr, to_addr)
        edge = self.edges[from_addr][to_addr]
        edge.total_value += value
        edge.tx_count += 1

    def out_degree(self, addr: str) -> int:
        """Number of unique addresses this address has sent to."""
        return len(self.edges.get(addr, {}))

    def in_degree(self, addr: str) -> int:
        """Number of unique addresses that have sent to this address."""
        count = 0
        for from_addr in self.edges:
            if addr in self.edges[from_addr]:
                count += 1
        return count

    def total_out_flow(self, addr: str) -> float:
        """Total value sent from this address."""
        return sum(e.total_value for e in self.edges.get(addr, {}).values())

    def total_in_flow(self, addr: str) -> float:
        """Total value received by this address."""
        total = 0.0
        for from_addr in self.edges:
            if addr in self.edges[from_addr]:
                total += self.edges[from_addr][addr].total_value
        return total

    def pagerank(self, damping: float = 0.85, iterations: int = 50) -> dict[str, float]:
        """Compute PageRank over the flow graph.

        PageRank identifies "important" nodes — addresses that receive
        value from many other important addresses. Originally designed
        for web page ranking, it works beautifully for on-chain flow
        analysis because it captures transitive importance.

        The algorithm:
        PR(v) = (1-d)/N + d * Σ_{u->v} PR(u) / out_degree(u)

        Where d is the damping factor (probability of following a link
        vs jumping to a random node). We iterate until convergence.
        """
        nodes = list(self.nodes)
        n = len(nodes)
        if n == 0:
            return {}

        # Initialize: uniform probability
        pr = {node: 1.0 / n for node in nodes}

        for _ in range(iterations):
            new_pr: dict[str, float] = {}
            for node in nodes:
                # Sum contributions from all incoming edges
                incoming_sum = 0.0
                for from_addr in self.edges:
                    if node in self.edges[from_addr]:
                        out_deg = len(self.edges[from_addr])
                        if out_deg > 0:
                            incoming_sum += pr[from_addr] / out_deg

                new_pr[node] = (1.0 - damping) / n + damping * incoming_sum
            pr = new_pr

        return pr

    def connected_components(self) -> list[set[str]]:
        """Find connected components (treating graph as undirected).

        Components reveal clusters of addresses that interact with each
        other. Isolated components might represent distinct protocols,
        communities, or obfuscation layers.
        """
        # Build undirected adjacency
        adj: dict[str, set[str]] = defaultdict(set)
        for from_addr, targets in self.edges.items():
            for to_addr in targets:
                adj[from_addr].add(to_addr)
                adj[to_addr].add(from_addr)

        visited: set[str] = set()
        components: list[set[str]] = []

        for node in self.nodes:
            if node not in visited:
                # BFS to find connected component
                component: set[str] = set()
                queue = [node]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)
                    for neighbor in adj.get(current, set()):
                        if neighbor not in visited:
                            queue.append(neighbor)
                components.append(component)

        return components

    def degree_distribution(self) -> dict[str, tuple[int, int]]:
        """Return (in_degree, out_degree) for each node."""
        dist: dict[str, tuple[int, int]] = {}
        for node in self.nodes:
            dist[node] = (self.in_degree(node), self.out_degree(node))
        return dist


# ---------------------------------------------------------------------------
# 6. TIME-SERIES ANALYTICS
# ---------------------------------------------------------------------------

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


def aggregate_by_block(transactions: list[Transaction]) -> list[TimeSeriesPoint]:
    """Aggregate transaction metrics per block.

    Block-level aggregation is the finest granularity for on-chain data
    (every ~12 seconds on Ethereum). For longer-term analysis you'd
    aggregate by hour or day, but block-level lets us detect single-block
    anomalies like exploit transactions.
    """
    by_block: dict[int, list[Transaction]] = defaultdict(list)
    for tx in transactions:
        by_block[tx.block_number].append(tx)

    points: list[TimeSeriesPoint] = []
    for block_num in sorted(by_block.keys()):
        block_txs = by_block[block_num]
        addrs: set[str] = set()
        for tx in block_txs:
            addrs.add(tx.from_addr)
            addrs.add(tx.to_addr)

        gas_prices = [tx.gas_price for tx in block_txs]

        points.append(TimeSeriesPoint(
            timestamp=block_txs[0].timestamp,
            block_number=block_num,
            tx_count=len(block_txs),
            total_volume=sum(tx.value for tx in block_txs),
            unique_addresses=len(addrs),
            avg_gas_price=sum(gas_prices) / len(gas_prices),
            max_value=max(tx.value for tx in block_txs),
        ))

    return points


def rolling_mean_std(values: list[float], window: int) -> list[tuple[float, float]]:
    """Compute rolling mean and standard deviation.

    Uses an online algorithm for efficiency: maintain a running sum and
    sum-of-squares, adding the new value and removing the oldest.

    Returns (mean, std) for each position. For the first `window-1` points,
    uses all available data (expanding window).
    """
    results: list[tuple[float, float]] = []

    for i in range(len(values)):
        # Window: values[max(0, i-window+1) : i+1]
        start = max(0, i - window + 1)
        w = values[start:i + 1]
        n = len(w)
        mean = sum(w) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in w) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        results.append((mean, std))

    return results


@dataclass
class Anomaly:
    """A detected anomaly in the time series."""
    block_number: int
    timestamp: int
    metric_name: str
    observed_value: float
    expected_mean: float
    z_score: float
    severity: str  # "low", "medium", "high"


def detect_anomalies(
    time_series: list[TimeSeriesPoint],
    window: int = 20,
    z_threshold: float = 2.5,
) -> list[Anomaly]:
    """Detect anomalies using rolling z-score method.

    For each metric, we compute the rolling mean and std over a window,
    then flag points where the z-score exceeds the threshold. The z-score
    tells us how many standard deviations a value is from the rolling mean.

    z = (observed - mean) / std

    Thresholds:
    - |z| > 2.5: low severity (unusual but could be noise)
    - |z| > 3.5: medium severity (likely significant)
    - |z| > 5.0: high severity (almost certainly anomalous)

    This is the same approach used by Chainlink price feed circuit breakers
    and exchange monitoring systems.
    """
    anomalies: list[Anomaly] = []

    # Analyze multiple metrics
    metrics = {
        "tx_count": [p.tx_count for p in time_series],
        "total_volume": [p.total_volume for p in time_series],
        "avg_gas_price": [p.avg_gas_price for p in time_series],
    }

    for metric_name, values in metrics.items():
        float_values = [float(v) for v in values]
        stats = rolling_mean_std(float_values, window)

        # Skip the warmup period (first `window` points)
        for i in range(window, len(values)):
            mean, std = stats[i]
            if std < 1e-10:
                continue  # No variance = can't compute z-score

            z = (float_values[i] - mean) / std

            if abs(z) > z_threshold:
                if abs(z) > 5.0:
                    severity = "high"
                elif abs(z) > 3.5:
                    severity = "medium"
                else:
                    severity = "low"

                anomalies.append(Anomaly(
                    block_number=time_series[i].block_number,
                    timestamp=time_series[i].timestamp,
                    metric_name=metric_name,
                    observed_value=float_values[i],
                    expected_mean=round(mean, 2),
                    z_score=round(z, 2),
                    severity=severity,
                ))

    return anomalies


# ---------------------------------------------------------------------------
# 7. WEALTH DISTRIBUTION (GINI COEFFICIENT)
# ---------------------------------------------------------------------------

def compute_balances(transactions: list[Transaction]) -> dict[str, float]:
    """Compute net balance for each address from transaction history.

    In reality, you'd query the blockchain state directly. Here we
    reconstruct balances from the transaction log. We start everyone
    at a positive base (simulating prior history) so balances stay
    meaningful.
    """
    balances: dict[str, float] = defaultdict(lambda: 100.0)  # Base balance
    for tx in transactions:
        balances[tx.from_addr] -= tx.value
        balances[tx.to_addr] += tx.value
    # Clamp negatives (in reality, transactions wouldn't be valid if insufficient balance)
    return {addr: max(0.0, bal) for addr, bal in balances.items()}


def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient of a distribution.

    The Gini coefficient measures inequality. For N values sorted ascending:

    G = (2 * Σᵢ i * xᵢ) / (n * Σ xᵢ) - (n + 1) / n

    Where i is 1-indexed. Range: 0 (perfect equality) to 1 (maximum inequality).

    For crypto token distributions:
    - G < 0.4: relatively decentralized (rare for crypto)
    - G 0.4-0.7: moderate concentration
    - G > 0.7: highly concentrated (common in DeFi governance tokens)
    - G > 0.9: extreme whale dominance
    """
    if not values or sum(values) == 0:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)

    # Weighted sum: Σ (1-indexed position) * value
    weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_values))

    gini = (2.0 * weighted_sum) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))  # Clamp to [0, 1]


def lorenz_curve(values: list[float]) -> list[tuple[float, float]]:
    """Compute Lorenz curve data points.

    The Lorenz curve plots the cumulative share of wealth (y-axis)
    against the cumulative share of population (x-axis). A diagonal
    line represents perfect equality. The area between the diagonal
    and the Lorenz curve is half the Gini coefficient.

    Returns list of (population_fraction, wealth_fraction) points.
    """
    if not values:
        return [(0.0, 0.0), (1.0, 1.0)]

    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)

    if total == 0:
        return [(0.0, 0.0), (1.0, 1.0)]

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    cumulative = 0.0
    for i, v in enumerate(sorted_values):
        cumulative += v
        pop_frac = (i + 1) / n
        wealth_frac = cumulative / total
        points.append((round(pop_frac, 4), round(wealth_frac, 4)))

    return points


# ---------------------------------------------------------------------------
# 8. ANALYTICS REPORT
# ---------------------------------------------------------------------------

def generate_report(
    transactions: list[Transaction],
    index: TransactionIndex,
    profiles: dict[str, AddressProfile],
    graph: TokenFlowGraph,
    time_series: list[TimeSeriesPoint],
    anomalies: list[Anomaly],
    balances: dict[str, float],
) -> dict:
    """Generate a comprehensive analytics report.

    This is the "dashboard" — a structured summary of all analysis results.
    In production, this would be rendered as charts and tables in a web UI
    (like Dune Analytics dashboards).
    """
    # Address classification summary
    label_counts: dict[str, int] = defaultdict(int)
    for p in profiles.values():
        label_counts[p.label] += 1

    # Top addresses by volume
    addr_volumes = [
        (addr, p.total_sent + p.total_received)
        for addr, p in profiles.items()
    ]
    addr_volumes.sort(key=lambda x: -x[1])

    # Graph metrics
    pagerank = graph.pagerank()
    top_pagerank = sorted(pagerank.items(), key=lambda x: -x[1])[:5]
    components = graph.connected_components()

    # Wealth distribution
    balance_values = list(balances.values())
    gini = gini_coefficient(balance_values)
    lorenz = lorenz_curve(balance_values)

    # Time-series summary
    total_volume = sum(p.total_volume for p in time_series)
    avg_block_txs = sum(p.tx_count for p in time_series) / max(1, len(time_series))

    return {
        "overview": {
            "total_transactions": len(transactions),
            "unique_addresses": len(profiles),
            "total_volume_eth": round(total_volume, 2),
            "avg_txs_per_block": round(avg_block_txs, 1),
            "block_range": index.block_range,
        },
        "address_classification": dict(label_counts),
        "top_addresses_by_volume": [
            {"address": addr[:12] + "...", "volume": round(vol, 2), "label": profiles[addr].label}
            for addr, vol in addr_volumes[:10]
        ],
        "graph_metrics": {
            "nodes": len(graph.nodes),
            "total_edges": sum(len(targets) for targets in graph.edges.values()),
            "connected_components": len(components),
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "top_pagerank": [
                {"address": addr[:12] + "...", "score": round(score, 6)}
                for addr, score in top_pagerank
            ],
        },
        "anomalies": {
            "total_detected": len(anomalies),
            "by_severity": {
                "high": sum(1 for a in anomalies if a.severity == "high"),
                "medium": sum(1 for a in anomalies if a.severity == "medium"),
                "low": sum(1 for a in anomalies if a.severity == "low"),
            },
            "details": [
                {
                    "block": a.block_number,
                    "metric": a.metric_name,
                    "observed": round(a.observed_value, 2),
                    "expected": a.expected_mean,
                    "z_score": a.z_score,
                    "severity": a.severity,
                }
                for a in anomalies[:10]  # Show top 10
            ],
        },
        "wealth_distribution": {
            "gini_coefficient": round(gini, 4),
            "interpretation": (
                "highly concentrated" if gini > 0.7
                else "moderately concentrated" if gini > 0.4
                else "relatively decentralized"
            ),
            "top_1pct_share": _top_pct_share(balance_values, 0.01),
            "top_10pct_share": _top_pct_share(balance_values, 0.10),
            "lorenz_curve_sample": lorenz[::max(1, len(lorenz) // 10)],  # Subsample for display
        },
    }


def _top_pct_share(values: list[float], pct: float) -> float:
    """What fraction of total wealth is held by the top pct% of addresses."""
    if not values:
        return 0.0
    sorted_desc = sorted(values, reverse=True)
    n_top = max(1, int(len(sorted_desc) * pct))
    total = sum(sorted_desc)
    if total == 0:
        return 0.0
    return round(sum(sorted_desc[:n_top]) / total, 4)


# ---------------------------------------------------------------------------
# 9. MAIN — END-TO-END DEMO
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # ── Step 1: Generate simulated blockchain data ──────────────────────
    print_section("STEP 1: Simulating Blockchain Data")
    transactions = simulate_blockchain_data(n_blocks=200, seed=42)
    print(f"Generated {len(transactions)} transactions across 200 blocks")
    print(f"Sample transaction:")
    tx = transactions[0]
    print(f"  Hash:      {tx.tx_hash[:20]}...")
    print(f"  Block:     {tx.block_number}")
    print(f"  From:      {tx.from_addr[:16]}...")
    print(f"  To:        {tx.to_addr[:16]}...")
    print(f"  Value:     {tx.value} ETH")
    print(f"  Gas:       {tx.gas_used} @ {tx.gas_price} Gwei")

    # ── Step 2: Build transaction index ─────────────────────────────────
    print_section("STEP 2: Building Transaction Index")
    index = TransactionIndex(transactions)
    print(f"Indexed {len(transactions)} transactions")
    print(f"Unique addresses: {len(index.unique_addresses)}")
    print(f"Block range: {index.block_range[0]} - {index.block_range[1]}")

    # ── Step 3: Profile all addresses ───────────────────────────────────
    print_section("STEP 3: Address Profiling & Classification")
    profiles = profile_all_addresses(index)

    label_counts: dict[str, int] = defaultdict(int)
    for p in profiles.values():
        label_counts[p.label] += 1

    print("Address classification:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:15s}: {count:3d} addresses")

    # Show a few interesting profiles
    print("\nTop 5 addresses by transaction count:")
    top_by_tx = sorted(profiles.values(), key=lambda p: -p.tx_count)[:5]
    for p in top_by_tx:
        net_flow = p.total_received - p.total_sent
        print(f"  {p.address[:16]}... | {p.label:13s} | "
              f"txs={p.tx_count:4d} | vol={p.total_sent + p.total_received:8.1f} ETH | "
              f"net={'+'if net_flow >= 0 else ''}{net_flow:.1f} ETH | "
              f"peers={p.unique_counterparties}")

    # ── Step 4: Token flow graph analysis ───────────────────────────────
    print_section("STEP 4: Token Flow Graph Analysis")
    graph = TokenFlowGraph.from_transactions(transactions)

    print(f"Graph: {len(graph.nodes)} nodes, "
          f"{sum(len(t) for t in graph.edges.values())} edges")

    # Connected components
    components = graph.connected_components()
    print(f"Connected components: {len(components)}")
    print(f"Largest component: {max(len(c) for c in components)} nodes")

    # PageRank — identifies structurally important addresses
    pagerank = graph.pagerank()
    top_pr = sorted(pagerank.items(), key=lambda x: -x[1])[:5]
    print("\nTop 5 by PageRank (structural importance):")
    for addr, score in top_pr:
        label = profiles[addr].label if addr in profiles else "?"
        print(f"  {addr[:16]}... | PR={score:.6f} | {label}")

    # Degree distribution summary
    degrees = graph.degree_distribution()
    in_degrees = [d[0] for d in degrees.values()]
    out_degrees = [d[1] for d in degrees.values()]
    print(f"\nDegree distribution:")
    print(f"  In-degree:  min={min(in_degrees)}, max={max(in_degrees)}, "
          f"avg={sum(in_degrees)/len(in_degrees):.1f}")
    print(f"  Out-degree: min={min(out_degrees)}, max={max(out_degrees)}, "
          f"avg={sum(out_degrees)/len(out_degrees):.1f}")

    # ── Step 5: Time-series anomaly detection ───────────────────────────
    print_section("STEP 5: Time-Series Anomaly Detection")
    time_series = aggregate_by_block(transactions)
    print(f"Aggregated {len(time_series)} block-level data points")

    # Show a few data points
    print("\nSample block metrics (first 3 blocks):")
    for pt in time_series[:3]:
        print(f"  Block {pt.block_number}: txs={pt.tx_count:3d}, "
              f"vol={pt.total_volume:7.1f} ETH, "
              f"addrs={pt.unique_addresses:2d}, "
              f"gas={pt.avg_gas_price:.1f} Gwei")

    anomalies = detect_anomalies(time_series, window=20, z_threshold=2.5)
    print(f"\nDetected {len(anomalies)} anomalies:")

    severity_counts = defaultdict(int)
    for a in anomalies:
        severity_counts[a.severity] += 1
    for sev in ["high", "medium", "low"]:
        print(f"  {sev:6s}: {severity_counts.get(sev, 0)}")

    if anomalies:
        print("\nTop anomalies (highest z-score):")
        top_anomalies = sorted(anomalies, key=lambda a: -abs(a.z_score))[:5]
        for a in top_anomalies:
            print(f"  Block {a.block_number} | {a.metric_name:14s} | "
                  f"observed={a.observed_value:8.1f} vs expected={a.expected_mean:8.1f} | "
                  f"z={a.z_score:+6.2f} | {a.severity}")

    # ── Step 6: Wealth distribution ─────────────────────────────────────
    print_section("STEP 6: Wealth Distribution Analysis")
    balances = compute_balances(transactions)
    balance_values = list(balances.values())

    gini = gini_coefficient(balance_values)
    print(f"Gini coefficient: {gini:.4f}")
    if gini > 0.7:
        print("  → Highly concentrated (typical of most crypto tokens)")
    elif gini > 0.4:
        print("  → Moderately concentrated")
    else:
        print("  → Relatively decentralized")

    # Top holders
    sorted_balances = sorted(balances.items(), key=lambda x: -x[1])
    total_supply = sum(balance_values)

    print(f"\nTotal supply: {total_supply:.1f} ETH across {len(balances)} addresses")
    print("\nTop 5 holders:")
    cumulative = 0.0
    for addr, bal in sorted_balances[:5]:
        cumulative += bal
        pct = bal / total_supply * 100
        cum_pct = cumulative / total_supply * 100
        label = profiles[addr].label if addr in profiles else "?"
        print(f"  {addr[:16]}... | {bal:8.1f} ETH ({pct:5.1f}%) | "
              f"cumul: {cum_pct:5.1f}% | {label}")

    # Lorenz curve sample
    lorenz = lorenz_curve(balance_values)
    print("\nLorenz curve (population% → wealth%):")
    for pop, wealth in lorenz[::max(1, len(lorenz) // 8)]:
        bar_len = int(wealth * 40)
        print(f"  {pop*100:5.1f}% pop → {wealth*100:5.1f}% wealth  {'█' * bar_len}")

    # Top 1% / 10% concentration
    top_1 = _top_pct_share(balance_values, 0.01)
    top_10 = _top_pct_share(balance_values, 0.10)
    print(f"\nConcentration: top 1% holds {top_1*100:.1f}%, top 10% holds {top_10*100:.1f}%")

    # ── Step 7: Full report ─────────────────────────────────────────────
    print_section("STEP 7: Analytics Dashboard Summary")
    report = generate_report(
        transactions, index, profiles, graph, time_series, anomalies, balances
    )

    print(f"Overview:")
    for k, v in report["overview"].items():
        print(f"  {k}: {v}")

    print(f"\nAddress types: {report['address_classification']}")

    print(f"\nGraph: {report['graph_metrics']['nodes']} nodes, "
          f"{report['graph_metrics']['total_edges']} edges, "
          f"{report['graph_metrics']['connected_components']} components")

    print(f"\nAnomalies: {report['anomalies']['total_detected']} total "
          f"({report['anomalies']['by_severity']})")

    print(f"\nWealth: Gini={report['wealth_distribution']['gini_coefficient']} "
          f"({report['wealth_distribution']['interpretation']})")

    print(f"\n{'='*70}")
    print("  Analysis complete. All metrics computed from raw transaction data.")
    print(f"{'='*70}")
