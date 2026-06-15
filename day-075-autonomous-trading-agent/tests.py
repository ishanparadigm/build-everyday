"""
Day 075: Full Autonomous Trading Agent — Test Suite

Run: python3 -m pytest tests.py -v
  or: python3 tests.py
"""

import math
import unittest

from my_solution import (
    Action,
    ACTION_POSITION_MAP,
    ExecutionEngine,
    Experience,
    FeatureEngine,
    Fill,
    MarketRegime,
    MarketSimulator,
    MarketTick,
    Portfolio,
    QNetwork,
    ReplayBuffer,
    RiskConfig,
    RiskManager,
    compute_reward,
)


class TestMarketSimulator(unittest.TestCase):
    """Tests for the market data generator."""

    def test_generates_correct_number_of_ticks(self):
        sim = MarketSimulator(seed=42, num_ticks=500)
        ticks = sim.generate()
        self.assertEqual(len(ticks), 500)

    def test_prices_are_positive(self):
        sim = MarketSimulator(seed=42, num_ticks=1000)
        ticks = sim.generate()
        for tick in ticks:
            self.assertGreater(tick.price, 0.0, "Prices must be positive")

    def test_deterministic_with_same_seed(self):
        ticks1 = MarketSimulator(seed=123, num_ticks=100).generate()
        ticks2 = MarketSimulator(seed=123, num_ticks=100).generate()
        for t1, t2 in zip(ticks1, ticks2):
            self.assertAlmostEqual(t1.price, t2.price, places=10)

    def test_different_seeds_produce_different_data(self):
        ticks1 = MarketSimulator(seed=1, num_ticks=100).generate()
        ticks2 = MarketSimulator(seed=2, num_ticks=100).generate()
        prices1 = [t.price for t in ticks1]
        prices2 = [t.price for t in ticks2]
        self.assertNotEqual(prices1, prices2)

    def test_on_chain_signals_bounded(self):
        sim = MarketSimulator(seed=42, num_ticks=500)
        ticks = sim.generate()
        for tick in ticks:
            self.assertGreaterEqual(tick.active_addresses, 0.0)
            self.assertLessEqual(tick.active_addresses, 1.0)
            self.assertGreaterEqual(tick.dex_imbalance, -1.0)
            self.assertLessEqual(tick.dex_imbalance, 1.0)


class TestFeatureEngine(unittest.TestCase):
    """Tests for feature extraction and normalization."""

    def test_state_dimension(self):
        fe = FeatureEngine(lookback=10)
        ticks = MarketSimulator(seed=42, num_ticks=50).generate()
        for tick in ticks:
            state = fe.update(tick, 0.0, 0.0)
        self.assertEqual(len(state), FeatureEngine.STATE_DIM)

    def test_warmup_returns_zeros(self):
        fe = FeatureEngine(lookback=20)
        tick = MarketTick(0, 100.0, 1000.0, 0.1, 0.5, 0.0, MarketRegime.TRENDING)
        state = fe.update(tick, 0.0, 0.0)
        self.assertEqual(state, [0.0] * FeatureEngine.STATE_DIM)

    def test_features_are_bounded(self):
        fe = FeatureEngine(lookback=10)
        ticks = MarketSimulator(seed=42, num_ticks=100).generate()
        for tick in ticks:
            state = fe.update(tick, 0.5, 0.02)
        for val in state:
            self.assertGreaterEqual(val, -1.5, "Features should be roughly bounded")
            self.assertLessEqual(val, 1.5, "Features should be roughly bounded")


class TestQNetwork(unittest.TestCase):
    """Tests for the neural network Q-function approximator."""

    def test_forward_output_shape(self):
        net = QNetwork(state_dim=8, num_actions=5, hidden_dim=16)
        q_values = net.forward([0.1] * 8)
        self.assertEqual(len(q_values), 5)

    def test_update_reduces_loss(self):
        net = QNetwork(state_dim=4, num_actions=3, hidden_dim=8, learning_rate=0.01)
        state = [0.5, -0.3, 0.1, 0.8]
        target = 1.0
        action = 1

        loss1 = net.update(state, action, target)
        # Run several more updates
        for _ in range(50):
            loss = net.update(state, action, target)
        self.assertLess(loss, loss1, "Loss should decrease with training")

    def test_copy_from(self):
        net1 = QNetwork(state_dim=4, num_actions=3, hidden_dim=8, seed=1)
        net2 = QNetwork(state_dim=4, num_actions=3, hidden_dim=8, seed=2)
        state = [0.1, 0.2, 0.3, 0.4]

        # Before copy, outputs differ
        q1 = net1.forward(state)
        q2 = net2.forward(state)
        self.assertNotEqual(q1, q2)

        # After copy, outputs match
        net2.copy_from(net1)
        q2_after = net2.forward(state)
        for a, b in zip(q1, q2_after):
            self.assertAlmostEqual(a, b, places=10)


class TestPortfolio(unittest.TestCase):
    """Tests for portfolio tracking and PnL calculation."""

    def test_initial_equity(self):
        port = Portfolio(initial_cash=10000)
        self.assertAlmostEqual(port.equity(100.0), 10000.0)

    def test_buy_and_price_increase(self):
        port = Portfolio(initial_cash=10000, cash=10000)
        fill = Fill(0, "BUY", 10.0, 100.0, 1.0, 0.01)
        port.apply_fill(fill)
        # Cash = 10000 - 10*100 - 1 = 8999
        self.assertAlmostEqual(port.cash, 8999.0)
        self.assertAlmostEqual(port.position, 10.0)
        # Equity at price 110: 8999 + 10*110 = 10099
        self.assertAlmostEqual(port.equity(110.0), 10099.0)

    def test_realized_pnl_on_close(self):
        port = Portfolio(initial_cash=10000, cash=10000)
        # Buy 10 @ 100
        buy_fill = Fill(0, "BUY", 10.0, 100.0, 0.0, 0.0)
        port.apply_fill(buy_fill)
        # Sell 10 @ 120 — should realize profit
        sell_fill = Fill(1, "SELL", 10.0, 120.0, 0.0, 0.0)
        pnl = port.apply_fill(sell_fill)
        self.assertAlmostEqual(pnl, 200.0)  # 10 * (120 - 100)
        self.assertAlmostEqual(port.position, 0.0)


class TestRiskManager(unittest.TestCase):
    """Tests for risk management constraints."""

    def test_kelly_fraction_conservative_initially(self):
        rm = RiskManager(RiskConfig(fractional_kelly=0.25))
        kelly = rm.kelly_fraction()
        self.assertGreater(kelly, 0.0)
        self.assertLess(kelly, 0.25)

    def test_cooldown_after_consecutive_losses(self):
        rm = RiskManager(RiskConfig(cooldown_after_losses=3, cooldown_ticks=5))
        rm.daily_start_equity = 10000
        rm.peak_equity = 10000
        rm.record_trade(-100)
        rm.record_trade(-50)
        rm.record_trade(-75)
        # After 3 consecutive losses, should be in cooldown
        action, size = rm.filter_action(Action.BUY, 9000.0, 100.0)
        self.assertEqual(action, Action.HOLD)
        self.assertEqual(size, 0.0)

    def test_drawdown_scaling(self):
        rm = RiskManager(RiskConfig(max_drawdown_pct=0.15))
        rm.peak_equity = 10000
        # At peak: full scale
        self.assertAlmostEqual(rm.drawdown_scale(10000), 1.0)
        # At 7.5% drawdown: half scale
        self.assertAlmostEqual(rm.drawdown_scale(9250), 0.5)
        # At or beyond 15% drawdown: zero
        self.assertAlmostEqual(rm.drawdown_scale(8500), 0.0)

    def test_daily_loss_limit(self):
        rm = RiskManager(RiskConfig(daily_loss_limit_pct=0.03))
        rm.daily_start_equity = 10000
        rm.peak_equity = 10000
        # 4% daily loss — should force HOLD
        action, size = rm.filter_action(Action.STRONG_BUY, 9600.0, 100.0)
        self.assertEqual(action, Action.HOLD)


class TestExecutionEngine(unittest.TestCase):
    """Tests for order execution simulation."""

    def test_buy_slippage_increases_price(self):
        ee = ExecutionEngine(slippage_factor=0.001)
        fill = ee.execute(0, 100.0, 10.0, 1000.0)
        self.assertIsNotNone(fill)
        self.assertGreater(fill.price, 100.0, "Buy should execute above mid price")

    def test_sell_slippage_decreases_price(self):
        ee = ExecutionEngine(slippage_factor=0.001)
        fill = ee.execute(0, 100.0, -10.0, 1000.0)
        self.assertIsNotNone(fill)
        self.assertLess(fill.price, 100.0, "Sell should execute below mid price")

    def test_zero_size_returns_none(self):
        ee = ExecutionEngine()
        fill = ee.execute(0, 100.0, 0.0, 1000.0)
        self.assertIsNone(fill)

    def test_fees_are_positive(self):
        ee = ExecutionEngine()
        fill = ee.execute(0, 100.0, 5.0, 1000.0)
        self.assertGreater(fill.fee, 0.0)


class TestReplayBuffer(unittest.TestCase):
    """Tests for experience replay."""

    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(Experience([float(i)], 0, 1.0, [float(i + 1)], False))
        self.assertEqual(len(buf), 10)
        batch = buf.sample(5)
        self.assertEqual(len(batch), 5)

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(Experience([float(i)], 0, 0.0, [0.0], False))
        self.assertEqual(len(buf), 5)


class TestRewardFunction(unittest.TestCase):
    """Tests for the risk-adjusted reward computation."""

    def test_positive_return_gives_positive_reward(self):
        r = compute_reward(10000, 10100, 0.01, 0.0)
        self.assertGreater(r, 0.0)

    def test_negative_return_gives_negative_reward(self):
        r = compute_reward(10000, 9900, 0.01, 0.0)
        self.assertLess(r, 0.0)

    def test_high_drawdown_penalizes(self):
        r_low_dd = compute_reward(10000, 10050, 0.01, 0.02)
        r_high_dd = compute_reward(10000, 10050, 0.01, 0.10)
        self.assertGreater(r_low_dd, r_high_dd,
                           "Higher drawdown should produce lower reward")


if __name__ == "__main__":
    unittest.main()
