"""
Day 019: Autonomous Trading Bot Skeleton — Tests

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import math
import random
from my_solution import (
    EventBus, EventType, Event, Side,
    MarketConfig, MarketSimulator,
    SignalConfig, SignalGenerator,
    RiskConfig, RiskEngine,
    ExecutionEngine,
    PerformanceMetrics, compute_performance,
)


class TestEventBus(unittest.TestCase):
    """Test the pub/sub event bus."""

    def test_subscribe_and_publish(self):
        """Events should be delivered to subscribers."""
        bus = EventBus()
        received = []
        bus.subscribe(EventType.MARKET_DATA, lambda e: received.append(e))
        event = Event(EventType.MARKET_DATA, {"price": 100.0}, timestamp=1)
        bus.publish(event)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data["price"], 100.0)

    def test_event_log(self):
        """All published events should be recorded in the log."""
        bus = EventBus()
        bus.publish(Event(EventType.MARKET_DATA, {"price": 1}, timestamp=1))
        bus.publish(Event(EventType.SIGNAL, {"side": "BUY"}, timestamp=2))
        self.assertEqual(len(bus.event_log), 2)

    def test_no_cross_delivery(self):
        """Events should only go to subscribers of that type."""
        bus = EventBus()
        received_market = []
        received_signal = []
        bus.subscribe(EventType.MARKET_DATA, lambda e: received_market.append(e))
        bus.subscribe(EventType.SIGNAL, lambda e: received_signal.append(e))
        bus.publish(Event(EventType.MARKET_DATA, {}, timestamp=1))
        self.assertEqual(len(received_market), 1)
        self.assertEqual(len(received_signal), 0)


class TestMarketSimulator(unittest.TestCase):
    """Test market data generation."""

    def test_price_stays_positive(self):
        """GBM should keep prices positive even with high volatility."""
        random.seed(123)
        bus = EventBus()
        config = MarketConfig(initial_price=100.0, volatility=0.05, num_ticks=200)
        market = MarketSimulator(config, bus)
        for _ in range(200):
            market.step()
        self.assertTrue(all(p > 0 for p in market.price_history))

    def test_bid_ask_spread(self):
        """Bid should be below mid, ask should be above mid."""
        random.seed(42)
        bus = EventBus()
        prices = []
        bus.subscribe(EventType.MARKET_DATA, lambda e: prices.append(e.data))
        config = MarketConfig(spread_bps=20.0, num_ticks=10)
        market = MarketSimulator(config, bus)
        for _ in range(10):
            market.step()
        for p in prices:
            self.assertLess(p["bid"], p["mid_price"])
            self.assertGreater(p["ask"], p["mid_price"])

    def test_price_history_length(self):
        """Price history should have initial + num_ticks entries."""
        random.seed(42)
        bus = EventBus()
        config = MarketConfig(num_ticks=50)
        market = MarketSimulator(config, bus)
        for _ in range(50):
            market.step()
        self.assertEqual(len(market.price_history), 51)  # initial + 50 ticks


class TestSignalGenerator(unittest.TestCase):
    """Test signal generation logic."""

    def test_no_signal_insufficient_data(self):
        """No signals should be generated before slow_window prices are seen."""
        bus = EventBus()
        config = SignalConfig(fast_window=5, slow_window=10, rsi_window=7)
        gen = SignalGenerator(config, bus)
        signals = []
        bus.subscribe(EventType.SIGNAL, lambda e: signals.append(e))
        # Send only 8 prices (less than slow_window=10)
        for i in range(8):
            bus.publish(Event(EventType.MARKET_DATA, {"mid_price": 100.0 + i * 0.1}, timestamp=i))
        self.assertEqual(len(signals), 0)

    def test_crossover_generates_signal(self):
        """A clear price trend reversal should generate a signal."""
        bus = EventBus()
        config = SignalConfig(fast_window=3, slow_window=7, rsi_window=5)
        gen = SignalGenerator(config, bus)
        signals = []
        bus.subscribe(EventType.SIGNAL, lambda e: signals.append(e))
        # Declining then rising prices to create a bullish crossover
        prices = [100 - i * 0.5 for i in range(15)] + [85 + i * 1.0 for i in range(20)]
        for i, p in enumerate(prices):
            bus.publish(Event(EventType.MARKET_DATA, {"mid_price": p}, timestamp=i))
        # Should have generated at least one signal
        self.assertGreater(len(signals), 0)


class TestRiskEngine(unittest.TestCase):
    """Test risk management constraints."""

    def test_position_limit(self):
        """Orders that exceed position limit should be clipped or rejected."""
        bus = EventBus()
        config = RiskConfig(max_position_size=20.0, order_size=10.0)
        risk = RiskEngine(config, bus, initial_capital=10000.0)
        orders = []
        bus.subscribe(EventType.ORDER, lambda e: orders.append(e))
        # Send 3 buy signals — third should be clipped or rejected
        for i in range(3):
            bus.publish(Event(EventType.SIGNAL, {
                "side": Side.BUY, "strength": 0.01, "price": 100.0,
                "fast_ma": 101, "slow_ma": 100, "rsi": 50,
            }, timestamp=i))
            # Simulate fill to update position
            if orders:
                last = orders[-1]
                bus.publish(Event(EventType.FILL, {
                    "side": Side.BUY,
                    "size": last.data["size"],
                    "realized_pnl": 0,
                    "portfolio_equity": 10000.0,
                }, timestamp=i))
        # Should have at most 2 full orders (20 units max)
        total_size = sum(o.data["size"] for o in orders)
        self.assertLessEqual(total_size, 20.0)

    def test_drawdown_halts_trading(self):
        """Exceeding max drawdown should halt the engine."""
        bus = EventBus()
        config = RiskConfig(max_portfolio_drawdown=0.05)
        risk = RiskEngine(config, bus, initial_capital=10000.0)
        # Simulate equity drop > 5%
        risk.current_equity = 9400.0  # 6% drawdown
        orders = []
        bus.subscribe(EventType.ORDER, lambda e: orders.append(e))
        bus.publish(Event(EventType.SIGNAL, {
            "side": Side.BUY, "strength": 0.01, "price": 100.0,
            "fast_ma": 101, "slow_ma": 100, "rsi": 50,
        }, timestamp=1))
        self.assertEqual(len(orders), 0)
        self.assertTrue(risk.halted)


class TestExecutionEngine(unittest.TestCase):
    """Test order execution."""

    def test_buy_costs_more_than_mid(self):
        """Buy fill price should be higher than mid price (spread + slippage)."""
        random.seed(42)
        bus = EventBus()
        config = MarketConfig(spread_bps=20.0, slippage_bps=5.0, num_ticks=1)
        market = MarketSimulator(config, bus)
        market.step()
        exec_engine = ExecutionEngine(bus, market, initial_capital=10000.0)
        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))
        bus.publish(Event(EventType.ORDER, {
            "side": Side.BUY, "size": 10.0, "signal_strength": 0.01, "price_at_signal": 100.0,
        }, timestamp=1))
        self.assertEqual(len(fills), 1)
        self.assertGreater(fills[0].data["fill_price"], market.price)

    def test_sell_costs_less_than_mid(self):
        """Sell fill price should be lower than mid price."""
        random.seed(42)
        bus = EventBus()
        config = MarketConfig(spread_bps=20.0, slippage_bps=5.0, num_ticks=1)
        market = MarketSimulator(config, bus)
        market.step()
        exec_engine = ExecutionEngine(bus, market, initial_capital=10000.0)
        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))
        bus.publish(Event(EventType.ORDER, {
            "side": Side.SELL, "size": 10.0, "signal_strength": 0.01, "price_at_signal": 100.0,
        }, timestamp=1))
        self.assertEqual(len(fills), 1)
        self.assertLess(fills[0].data["fill_price"], market.price)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance calculation."""

    def test_no_trades_returns_defaults(self):
        """Empty trade list should return zero metrics."""
        metrics = compute_performance([], 10000.0, [100.0])
        self.assertEqual(metrics.num_trades, 0)
        self.assertEqual(metrics.total_return_pct, 0.0)

    def test_profitable_trades(self):
        """Trades with positive P&L should show positive return."""
        trades = [
            {"realized_pnl": 50.0, "fee": 1.0, "portfolio_equity": 10049.0, "tick": 1,
             "side": Side.BUY, "size": 10, "fill_price": 100, "position_after": 10, "cash_after": 9000},
            {"realized_pnl": 100.0, "fee": 1.0, "portfolio_equity": 10148.0, "tick": 2,
             "side": Side.SELL, "size": 10, "fill_price": 110, "position_after": 0, "cash_after": 10148},
        ]
        metrics = compute_performance(trades, 10000.0, [100.0, 105.0, 110.0])
        self.assertGreater(metrics.total_return_pct, 0)
        self.assertGreater(metrics.profit_factor, 1.0)


if __name__ == "__main__":
    unittest.main()
