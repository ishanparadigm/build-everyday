"""
Day 072: On-Chain Data Analysis — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import math
from collections import defaultdict

from my_solution import (
    Transaction, AddressProfile, FlowEdge, TimeSeriesPoint, Anomaly,
    generate_address, simulate_blockchain_data,
    TransactionIndex,
    profile_address, classify_address, profile_all_addresses,
    TokenFlowGraph,
    aggregate_by_block, rolling_mean_std, detect_anomalies,
    compute_balances, gini_coefficient, lorenz_curve,
)


class TestAddressGeneration(unittest.TestCase):
    """Test deterministic address generation."""

    def test_format(self):
        addr = generate_address("test", 0)
        self.assertTrue(addr.startswith("0x"))
        self.assertEqual(len(addr), 42)  # 0x + 40 hex chars

    def test_deterministic(self):
        a1 = generate_address("whale", 3)
        a2 = generate_address("whale", 3)
        self.assertEqual(a1, a2)

    def test_unique(self):
        addrs = [generate_address("user", i) for i in range(20)]
        self.assertEqual(len(set(addrs)), 20)


class TestBlockchainSimulator(unittest.TestCase):
    """Test that simulated data has expected properties."""

    def setUp(self):
        self.txs = simulate_blockchain_data(n_blocks=50, seed=123)

    def test_produces_transactions(self):
        self.assertGreater(len(self.txs), 100)

    def test_transaction_fields(self):
        tx = self.txs[0]
        self.assertTrue(tx.tx_hash.startswith("0x"))
        self.assertGreater(tx.block_number, 0)
        self.assertGreater(tx.timestamp, 0)
        self.assertTrue(tx.from_addr.startswith("0x"))
        self.assertTrue(tx.to_addr.startswith("0x"))
        self.assertGreater(tx.value, 0)

    def test_deterministic_seed(self):
        txs2 = simulate_blockchain_data(n_blocks=50, seed=123)
        self.assertEqual(len(self.txs), len(txs2))
        self.assertEqual(self.txs[0].tx_hash, txs2[0].tx_hash)


class TestTransactionIndex(unittest.TestCase):
    """Test transaction indexing and querying."""

    def setUp(self):
        self.txs = simulate_blockchain_data(n_blocks=30, seed=42)
        self.index = TransactionIndex(self.txs)

    def test_unique_addresses(self):
        addrs = self.index.unique_addresses
        self.assertGreater(len(addrs), 10)

    def test_block_range(self):
        lo, hi = self.index.block_range
        self.assertLess(lo, hi)

    def test_address_lookup(self):
        addr = self.txs[0].from_addr
        results = self.index.get_address_txs(addr)
        self.assertGreater(len(results), 0)
        for tx in results:
            self.assertTrue(tx.from_addr == addr or tx.to_addr == addr)

    def test_block_lookup(self):
        block = self.txs[0].block_number
        results = self.index.get_block_txs(block)
        self.assertGreater(len(results), 0)
        for tx in results:
            self.assertEqual(tx.block_number, block)

    def test_time_range(self):
        t0 = self.txs[0].timestamp
        t1 = t0 + 60  # 60 seconds
        results = self.index.get_time_range(t0, t1)
        self.assertGreater(len(results), 0)
        for tx in results:
            self.assertGreaterEqual(tx.timestamp, t0)
            self.assertLessEqual(tx.timestamp, t1)


class TestAddressProfiler(unittest.TestCase):
    """Test address profiling and classification."""

    def setUp(self):
        self.txs = simulate_blockchain_data(n_blocks=100, seed=42)
        self.index = TransactionIndex(self.txs)

    def test_profile_has_counts(self):
        addr = self.txs[0].from_addr
        profile = profile_address(addr, self.index)
        self.assertGreater(profile.tx_count, 0)
        self.assertGreater(profile.sent_count, 0)

    def test_profile_timestamps(self):
        addr = self.txs[0].from_addr
        profile = profile_address(addr, self.index)
        self.assertGreater(profile.first_seen, 0)
        self.assertGreaterEqual(profile.last_seen, profile.first_seen)

    def test_classify_returns_valid_label(self):
        profiles = profile_all_addresses(self.index)
        valid_labels = {"exchange", "bot_mev", "contract", "whale", "regular_user", "inactive"}
        for p in profiles.values():
            self.assertIn(p.label, valid_labels, f"Unknown label: {p.label}")

    def test_multiple_classifications_present(self):
        """With enough data, we should see at least 2 different labels."""
        profiles = profile_all_addresses(self.index)
        labels = set(p.label for p in profiles.values())
        self.assertGreaterEqual(len(labels), 2)


class TestTokenFlowGraph(unittest.TestCase):
    """Test flow graph construction and analysis."""

    def test_basic_graph(self):
        graph = TokenFlowGraph()
        graph.add_transfer("A", "B", 10.0)
        graph.add_transfer("A", "C", 5.0)
        graph.add_transfer("B", "C", 3.0)

        self.assertEqual(graph.out_degree("A"), 2)
        self.assertEqual(graph.in_degree("C"), 2)
        self.assertEqual(graph.out_degree("C"), 0)
        self.assertIn("A", graph.nodes)
        self.assertIn("B", graph.nodes)
        self.assertIn("C", graph.nodes)

    def test_edge_aggregation(self):
        graph = TokenFlowGraph()
        graph.add_transfer("A", "B", 10.0)
        graph.add_transfer("A", "B", 5.0)
        edge = graph.edges["A"]["B"]
        self.assertAlmostEqual(edge.total_value, 15.0)
        self.assertEqual(edge.tx_count, 2)

    def test_pagerank(self):
        graph = TokenFlowGraph()
        # Star graph: everyone sends to C
        for name in ["A", "B", "D", "E"]:
            graph.add_transfer(name, "C", 1.0)
        pr = graph.pagerank()
        # C should have highest PageRank since everyone points to it
        self.assertEqual(max(pr, key=pr.get), "C")

    def test_connected_components(self):
        graph = TokenFlowGraph()
        graph.add_transfer("A", "B", 1.0)
        graph.add_transfer("C", "D", 1.0)
        comps = graph.connected_components()
        self.assertEqual(len(comps), 2)

    def test_from_transactions(self):
        txs = simulate_blockchain_data(n_blocks=20, seed=42)
        graph = TokenFlowGraph.from_transactions(txs)
        self.assertGreater(len(graph.nodes), 0)


class TestTimeSeries(unittest.TestCase):
    """Test time-series aggregation and anomaly detection."""

    def test_aggregate_by_block(self):
        txs = simulate_blockchain_data(n_blocks=20, seed=42)
        ts = aggregate_by_block(txs)
        self.assertEqual(len(ts), 20)
        for pt in ts:
            self.assertGreater(pt.tx_count, 0)
            self.assertGreater(pt.total_volume, 0)

    def test_rolling_mean_std(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = rolling_mean_std(values, window=3)
        # At index 4, window is [3.0, 4.0, 5.0], mean = 4.0
        self.assertAlmostEqual(stats[4][0], 4.0)
        self.assertEqual(len(stats), 5)

    def test_anomaly_detection_catches_spike(self):
        """The simulated data has a spike at blocks 148-152. Detector should flag it."""
        txs = simulate_blockchain_data(n_blocks=200, seed=42)
        ts = aggregate_by_block(txs)
        anomalies = detect_anomalies(ts, window=20, z_threshold=2.5)
        # There should be at least some anomalies near the spike
        self.assertGreater(len(anomalies), 0)
        # Check that at least one anomaly is in the spike region
        spike_blocks = set(range(18_000_148, 18_000_153))
        anomaly_blocks = set(a.block_number for a in anomalies)
        self.assertTrue(
            spike_blocks & anomaly_blocks,
            "Expected anomalies in spike region (blocks 148-152)"
        )


class TestWealthDistribution(unittest.TestCase):
    """Test balance computation and Gini coefficient."""

    def test_compute_balances(self):
        txs = simulate_blockchain_data(n_blocks=30, seed=42)
        balances = compute_balances(txs)
        self.assertGreater(len(balances), 0)
        # All balances should be non-negative
        for bal in balances.values():
            self.assertGreaterEqual(bal, 0.0)

    def test_gini_perfect_equality(self):
        values = [100.0] * 10
        g = gini_coefficient(values)
        self.assertAlmostEqual(g, 0.0, places=2)

    def test_gini_high_inequality(self):
        values = [0.0] * 99 + [1000.0]
        g = gini_coefficient(values)
        self.assertGreater(g, 0.9)

    def test_gini_range(self):
        txs = simulate_blockchain_data(n_blocks=100, seed=42)
        balances = compute_balances(txs)
        g = gini_coefficient(list(balances.values()))
        self.assertGreaterEqual(g, 0.0)
        self.assertLessEqual(g, 1.0)

    def test_lorenz_curve_endpoints(self):
        values = [10.0, 20.0, 30.0]
        curve = lorenz_curve(values)
        # Should start at (0,0) and end at (1,1)
        self.assertAlmostEqual(curve[0][0], 0.0)
        self.assertAlmostEqual(curve[0][1], 0.0)
        self.assertAlmostEqual(curve[-1][0], 1.0)
        self.assertAlmostEqual(curve[-1][1], 1.0)

    def test_lorenz_monotonic(self):
        values = [5.0, 15.0, 25.0, 55.0]
        curve = lorenz_curve(values)
        for i in range(1, len(curve)):
            self.assertGreaterEqual(curve[i][1], curve[i-1][1])


if __name__ == "__main__":
    unittest.main()
