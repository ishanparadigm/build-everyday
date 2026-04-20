"""
Day 019: Autonomous Trading Bot Skeleton

An event-driven trading bot framework integrating:
- AI: Moving average crossover + RSI signal generation
- Crypto: Simulated order book execution with spread and slippage
- Robotics: Control-system risk management (portfolio state as a physical system)

Architecture:
    EventBus -> MarketSimulator -> SignalGenerator -> RiskEngine -> ExecutionEngine -> PortfolioTracker

Each component is decoupled via the EventBus — they communicate through typed events,
not direct method calls. This mirrors production trading systems where components may
run in separate processes or even on separate machines.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


# =============================================================================
# Event System
# =============================================================================
# The event bus is the backbone of the architecture. It implements the observer
# pattern: components register callbacks for event types they care about, and
# the bus routes events to the right subscribers. This gives us loose coupling —
# the signal generator doesn't need a reference to the risk engine, it just
# emits Signal events and trusts the bus to deliver them.

class EventType(Enum):
    """All event types in the system. Using an enum prevents typo-based bugs."""
    MARKET_DATA = auto()   # New price tick from market
    SIGNAL = auto()        # Trading signal from strategy
    ORDER = auto()         # Order submitted for execution
    FILL = auto()          # Order filled (executed)
    RISK_ALERT = auto()    # Risk limit breached


@dataclass
class Event:
    """Base event with type and arbitrary payload."""
    event_type: EventType
    data: dict[str, Any]
    timestamp: int = 0  # Simulation tick number


class EventBus:
    """
    Publish-subscribe event bus.

    Why not just call methods directly? Because:
    1. Adding a new component doesn't require modifying existing ones
    2. Components can be tested in isolation by injecting fake events
    3. You can add logging/monitoring by subscribing to all events
    4. In production, this can be replaced with a message queue (Kafka, Redis Streams)
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = defaultdict(list)
        self._event_log: list[Event] = []  # Full audit trail — critical for debugging strategies

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Register a callback for a specific event type."""
        self._subscribers[event_type].append(callback)

    def publish(self, event: Event) -> None:
        """
        Dispatch an event to all subscribers.

        Events are processed synchronously in subscription order. In production,
        you'd want async dispatch with priority queues, but synchronous processing
        makes debugging much easier — events happen in a deterministic order.
        """
        self._event_log.append(event)
        for callback in self._subscribers[event.event_type]:
            callback(event)

    @property
    def event_log(self) -> list[Event]:
        return self._event_log


# =============================================================================
# Market Simulation
# =============================================================================
# We simulate a crypto market using a geometric Brownian motion (GBM) model:
#   dP/P = mu*dt + sigma*dW
# where mu is drift (trend), sigma is volatility, and dW is a Wiener process.
#
# GBM is the standard model in finance (it underlies Black-Scholes). It ensures
# prices stay positive and returns are log-normally distributed. It's wrong in
# important ways (fat tails, volatility clustering) but good enough for testing
# a bot's architecture.

@dataclass
class MarketConfig:
    """Configuration for the market simulator."""
    initial_price: float = 100.0
    drift: float = 0.0001          # mu: slight upward bias per tick
    volatility: float = 0.02       # sigma: 2% per-tick volatility (high for crypto)
    spread_bps: float = 10.0       # Bid-ask spread in basis points (1 bp = 0.01%)
    slippage_bps: float = 5.0      # Additional cost per unit of size
    num_ticks: int = 500


class MarketSimulator:
    """
    Generates synthetic market data and publishes it to the event bus.

    The simulator produces mid-prices via GBM, then adds a spread to create
    bid/ask prices. This models the cost of trading — you always buy at a
    slightly higher price (ask) and sell at a slightly lower price (bid).
    """

    def __init__(self, config: MarketConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.price = config.initial_price
        self.tick = 0
        self.price_history: list[float] = [self.price]

    def step(self) -> None:
        """
        Advance the market by one tick.

        Uses the discrete approximation to GBM:
            P(t+1) = P(t) * exp(mu - sigma^2/2 + sigma * Z)
        where Z ~ N(0, 1). The -sigma^2/2 term is the Ito correction that
        ensures E[P(t+1)] = P(t) * exp(mu), not P(t) * exp(mu + sigma^2/2).
        Without it, volatility would create an artificial upward drift.
        """
        self.tick += 1
        z = random.gauss(0, 1)

        # GBM step with Ito correction
        log_return = self.config.drift - 0.5 * self.config.volatility ** 2 + self.config.volatility * z
        self.price *= math.exp(log_return)
        self.price_history.append(self.price)

        # Compute bid/ask from mid price
        # Spread widens with volatility — this is realistic. Market makers widen
        # spreads when uncertainty is high to compensate for adverse selection risk.
        half_spread = self.price * (self.config.spread_bps / 10000) / 2

        self.bus.publish(Event(
            event_type=EventType.MARKET_DATA,
            data={
                "mid_price": self.price,
                "bid": self.price - half_spread,
                "ask": self.price + half_spread,
                "tick": self.tick,
            },
            timestamp=self.tick,
        ))


# =============================================================================
# Signal Generation
# =============================================================================
# The signal generator is the "brain" of the bot. It consumes price data and
# emits trading signals. We implement two classic technical indicators:
#
# 1. Moving Average Crossover: Detects momentum shifts
# 2. RSI (Relative Strength Index): Filters overbought/oversold conditions
#
# These are simple but illustrate the core pattern: transform raw data into
# actionable signals with confidence scores.

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    fast_window: int = 10    # Short-term MA period
    slow_window: int = 30    # Long-term MA period
    rsi_window: int = 14     # RSI lookback period
    rsi_overbought: float = 70.0   # Don't buy above this RSI
    rsi_oversold: float = 30.0     # Don't sell below this RSI


class SignalGenerator:
    """
    Generates trading signals from market data using MA crossover + RSI filter.

    The signal generation pipeline:
    1. Receive new price -> update rolling windows
    2. Compute fast and slow moving averages
    3. Detect crossover events (fast crosses above/below slow)
    4. Compute RSI as a confirmation filter
    5. Emit signal only if crossover + RSI agree

    Why two indicators? Each indicator has failure modes. MA crossover whipsaws
    in ranging markets; RSI gives false signals in trending markets. Combining
    them reduces false positives at the cost of missing some true signals.
    This is the classic precision-recall tradeoff from ML applied to trading.
    """

    def __init__(self, config: SignalConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        # deque with maxlen automatically evicts old data — O(1) append
        self.prices: deque[float] = deque(maxlen=config.slow_window)
        self.gains: deque[float] = deque(maxlen=config.rsi_window)
        self.losses: deque[float] = deque(maxlen=config.rsi_window)
        self.prev_fast_above: bool | None = None  # Track crossover state
        self.signal_count = 0

        bus.subscribe(EventType.MARKET_DATA, self.on_market_data)

    def _compute_sma(self, window: int) -> float | None:
        """
        Simple Moving Average over the last `window` prices.

        SMA = (1/n) * sum(P(t-i) for i in 0..n-1)

        Returns None if we don't have enough data yet. This is important —
        generating signals on insufficient data leads to garbage signals
        that happen to look meaningful (a form of lookahead bias).
        """
        if len(self.prices) < window:
            return None
        # Slice the deque from the right (most recent) — deques support slicing via list()
        recent = list(self.prices)[-window:]
        return sum(recent) / window

    def _compute_rsi(self) -> float | None:
        """
        Relative Strength Index.

        RSI = 100 - 100 / (1 + RS)
        RS = average_gain / average_loss over N periods

        The RSI oscillates between 0 and 100:
        - RSI > 70: overbought (price has risen too fast, likely to pull back)
        - RSI < 30: oversold (price has fallen too fast, likely to bounce)

        Note: we use simple averages here. Wilder's original uses exponential
        smoothing, which gives more weight to recent data. For our purposes,
        simple averaging is clearer and the difference is marginal.
        """
        if len(self.gains) < self.config.rsi_window:
            return None
        avg_gain = sum(self.gains) / len(self.gains)
        avg_loss = sum(self.losses) / len(self.losses)
        if avg_loss == 0:
            return 100.0  # All gains, no losses = maximally overbought
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_market_data(self, event: Event) -> None:
        """
        Process a new price tick.

        The logic flow:
        1. Update price history and gain/loss series
        2. Compute MAs — need at least slow_window data points
        3. Check for crossover: did fast MA just cross above/below slow MA?
        4. Filter with RSI: is the market confirming the signal?
        5. Emit signal with strength = |fast - slow| / slow (divergence magnitude)
        """
        price = event.data["mid_price"]

        # Track gains and losses for RSI
        if len(self.prices) > 0:
            change = price - self.prices[-1]
            self.gains.append(max(0, change))
            self.losses.append(max(0, -change))

        self.prices.append(price)

        # Need enough data for the slow MA
        fast_ma = self._compute_sma(self.config.fast_window)
        slow_ma = self._compute_sma(self.config.slow_window)
        rsi = self._compute_rsi()

        if fast_ma is None or slow_ma is None or rsi is None:
            return  # Not enough data yet — stay flat

        fast_above = fast_ma > slow_ma

        # Detect crossover — we need a state CHANGE, not just fast > slow
        # Without tracking the previous state, we'd generate a signal on every
        # tick where fast > slow, which is useless noise
        if self.prev_fast_above is not None and fast_above != self.prev_fast_above:
            # Signal strength: how far apart the MAs are, as a percentage
            # Larger divergence = stronger signal = higher conviction
            strength = abs(fast_ma - slow_ma) / slow_ma

            if fast_above and rsi < self.config.rsi_overbought:
                # Bullish crossover confirmed by RSI (not overbought)
                self.signal_count += 1
                self.bus.publish(Event(
                    event_type=EventType.SIGNAL,
                    data={
                        "side": Side.BUY,
                        "strength": strength,
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                        "rsi": rsi,
                        "price": price,
                    },
                    timestamp=event.timestamp,
                ))
            elif not fast_above and rsi > self.config.rsi_oversold:
                # Bearish crossover confirmed by RSI (not oversold)
                self.signal_count += 1
                self.bus.publish(Event(
                    event_type=EventType.SIGNAL,
                    data={
                        "side": Side.SELL,
                        "strength": strength,
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                        "rsi": rsi,
                        "price": price,
                    },
                    timestamp=event.timestamp,
                ))

        self.prev_fast_above = fast_above


# =============================================================================
# Risk Engine
# =============================================================================
# The risk engine is the "safety controller" — analogous to a PID controller
# that prevents a physical system from exceeding safe operating limits.
#
# In trading, violating risk limits can be catastrophic:
# - Knight Capital lost $440M in 45 minutes due to a missing risk check
# - LTCM's unchecked leverage nearly crashed the global financial system
#
# Our risk engine enforces:
# 1. Max position size (prevent concentration)
# 2. Max drawdown (circuit breaker)
# 3. Daily loss limit (stop-loss for the day)

@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_position_size: float = 50.0     # Max units in any direction
    max_portfolio_drawdown: float = 0.10  # 10% max drawdown from peak
    daily_loss_limit: float = 500.0     # Max daily loss in currency
    order_size: float = 10.0            # Fixed order size per signal


class RiskEngine:
    """
    Evaluates signals against risk constraints and emits vetted orders.

    The risk engine sits between signal generation and execution. Every signal
    must pass through risk checks before becoming an order. This is a critical
    safety layer — in production, it's the last line of defense before real
    money moves.

    Design principle: the risk engine can REDUCE or REJECT orders, never
    amplify them. It's strictly conservative. This is like the safety
    constraints on a robot arm — they can prevent movement, never add it.
    """

    def __init__(self, config: RiskConfig, bus: EventBus, initial_capital: float) -> None:
        self.config = config
        self.bus = bus
        self.initial_capital = initial_capital
        self.position: float = 0.0      # Current position (positive = long, negative = short)
        self.daily_pnl: float = 0.0     # Running daily P&L
        self.peak_equity: float = initial_capital
        self.current_equity: float = initial_capital
        self.halted: bool = False        # Circuit breaker state
        self.orders_approved: int = 0
        self.orders_rejected: int = 0

        bus.subscribe(EventType.SIGNAL, self.on_signal)
        bus.subscribe(EventType.FILL, self.on_fill)

    def _check_drawdown(self) -> bool:
        """
        Check if portfolio drawdown exceeds limit.

        Drawdown = (peak_equity - current_equity) / peak_equity

        This is the most important risk check. A 10% drawdown means you need
        an 11.1% gain to recover. A 50% drawdown needs a 100% gain. The
        relationship is non-linear and punishing, which is why professional
        risk management focuses heavily on drawdown control.
        """
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        return drawdown < self.config.max_portfolio_drawdown

    def _check_position_limit(self, side: Side, size: float) -> float:
        """
        Enforce maximum position size. Returns the allowed order size.

        If we're long 45 units and max is 50, a buy order of 10 gets clipped
        to 5. This prevents gradual accumulation past the limit from multiple
        signals.
        """
        if side == Side.BUY:
            new_position = self.position + size
            if new_position > self.config.max_position_size:
                return max(0.0, self.config.max_position_size - self.position)
        else:
            new_position = self.position - size
            if new_position < -self.config.max_position_size:
                return max(0.0, self.config.max_position_size + self.position)
        return size

    def _check_daily_loss(self) -> bool:
        """Check if daily loss limit has been reached."""
        return self.daily_pnl > -self.config.daily_loss_limit

    def on_signal(self, event: Event) -> None:
        """
        Process a signal through the risk gauntlet.

        Order of checks matters — we check the cheapest (computation-wise)
        and most critical checks first:
        1. Circuit breaker (instant reject if halted)
        2. Drawdown check (portfolio-level)
        3. Daily loss limit (time-based)
        4. Position limit (may clip size)
        """
        if self.halted:
            self.orders_rejected += 1
            return

        if not self._check_drawdown():
            self.halted = True
            self.orders_rejected += 1
            self.bus.publish(Event(
                event_type=EventType.RISK_ALERT,
                data={"reason": "MAX_DRAWDOWN", "equity": self.current_equity, "peak": self.peak_equity},
                timestamp=event.timestamp,
            ))
            return

        if not self._check_daily_loss():
            self.orders_rejected += 1
            self.bus.publish(Event(
                event_type=EventType.RISK_ALERT,
                data={"reason": "DAILY_LOSS_LIMIT", "daily_pnl": self.daily_pnl},
                timestamp=event.timestamp,
            ))
            return

        side = event.data["side"]
        size = self._check_position_limit(side, self.config.order_size)

        if size <= 0:
            self.orders_rejected += 1
            return

        self.orders_approved += 1
        self.bus.publish(Event(
            event_type=EventType.ORDER,
            data={
                "side": side,
                "size": size,
                "signal_strength": event.data["strength"],
                "price_at_signal": event.data["price"],
            },
            timestamp=event.timestamp,
        ))

    def on_fill(self, event: Event) -> None:
        """Update position and P&L tracking after a fill."""
        fill = event.data
        if fill["side"] == Side.BUY:
            self.position += fill["size"]
        else:
            self.position -= fill["size"]
        self.daily_pnl += fill.get("realized_pnl", 0.0)
        self.current_equity = fill["portfolio_equity"]


# =============================================================================
# Execution Engine
# =============================================================================
# The execution engine simulates order fills with realistic market mechanics.
# In real trading, the gap between "I want to buy" and "I bought" is where
# most money is lost. This gap has several components:
#
# 1. Spread: you buy at the ask (higher) and sell at the bid (lower)
# 2. Slippage: your order moves the market against you
# 3. Latency: by the time your order reaches the exchange, the price has moved
#
# Together, these are called "execution costs" or "transaction costs."

class ExecutionEngine:
    """
    Simulates order execution with realistic market friction.

    Key concept: the execution price is always WORSE than the mid price.
    - Buying: you pay ask + slippage (higher than mid)
    - Selling: you receive bid - slippage (lower than mid)

    This is the cost of immediacy — if you want to trade NOW, you pay the spread.
    If you're willing to wait (limit orders), you can potentially get a better
    price, but you risk not getting filled at all.
    """

    def __init__(self, bus: EventBus, market: MarketSimulator, initial_capital: float) -> None:
        self.bus = bus
        self.market = market
        self.cash: float = initial_capital
        self.position: float = 0.0
        self.avg_entry_price: float = 0.0  # Volume-weighted average entry price
        self.trades: list[dict[str, Any]] = []
        self.total_fees: float = 0.0

        bus.subscribe(EventType.ORDER, self.on_order)

    def _compute_fill_price(self, side: Side, size: float) -> float:
        """
        Compute the execution price including spread and slippage.

        Fill price = mid +/- half_spread +/- slippage * size

        Slippage is proportional to order size — this models market impact.
        A large order consumes multiple levels of the order book, getting
        progressively worse prices. This is why large funds use execution
        algorithms (TWAP, VWAP) to split orders into smaller pieces.
        """
        mid = self.market.price
        half_spread = mid * (self.market.config.spread_bps / 10000) / 2
        slippage = mid * (self.market.config.slippage_bps / 10000) * size / 100

        if side == Side.BUY:
            return mid + half_spread + slippage  # Pay more when buying
        else:
            return mid - half_spread - slippage  # Receive less when selling

    def on_order(self, event: Event) -> None:
        """
        Execute an order and publish a fill event.

        This method also handles P&L calculation. When we close or reduce a
        position, we realize a profit or loss equal to:
            realized_pnl = size * (fill_price - avg_entry_price) for longs
            realized_pnl = size * (avg_entry_price - fill_price) for shorts
        """
        side = event.data["side"]
        size = event.data["size"]
        fill_price = self._compute_fill_price(side, size)

        # Calculate realized P&L if reducing position
        realized_pnl = 0.0
        if side == Side.BUY:
            if self.position < 0:  # Covering a short
                cover_size = min(size, abs(self.position))
                realized_pnl = cover_size * (self.avg_entry_price - fill_price)
            # Update average entry price for the new/increased position
            if self.position >= 0:
                # Adding to long: weighted average
                total_cost = self.avg_entry_price * self.position + fill_price * size
                new_position = self.position + size
                self.avg_entry_price = total_cost / new_position if new_position > 0 else 0
            elif self.position + size > 0:
                # Flipping from short to long
                self.avg_entry_price = fill_price
            self.position += size
            self.cash -= fill_price * size
        else:  # SELL
            if self.position > 0:  # Closing a long
                close_size = min(size, self.position)
                realized_pnl = close_size * (fill_price - self.avg_entry_price)
            if self.position <= 0:
                # Adding to short: weighted average
                total_value = self.avg_entry_price * abs(self.position) + fill_price * size
                new_position = abs(self.position) + size
                self.avg_entry_price = total_value / new_position if new_position > 0 else 0
            elif self.position - size < 0:
                # Flipping from long to short
                self.avg_entry_price = fill_price
            self.position -= size
            self.cash += fill_price * size

        fee = fill_price * size * 0.001  # 10 bps fee (typical for crypto exchanges)
        self.cash -= fee
        self.total_fees += fee

        # Portfolio equity = cash + mark-to-market position value
        unrealized_pnl = self.position * (self.market.price - self.avg_entry_price) if self.position != 0 else 0
        portfolio_equity = self.cash + abs(self.position) * self.market.price * (1 if self.position >= 0 else -1)
        # Simpler: equity = cash + position * current_price
        portfolio_equity = self.cash + self.position * self.market.price

        trade = {
            "tick": event.timestamp,
            "side": side,
            "size": size,
            "fill_price": fill_price,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "position_after": self.position,
            "cash_after": self.cash,
            "portfolio_equity": portfolio_equity,
        }
        self.trades.append(trade)

        self.bus.publish(Event(
            event_type=EventType.FILL,
            data=trade,
            timestamp=event.timestamp,
        ))


# =============================================================================
# Performance Analytics
# =============================================================================
# You can't improve what you don't measure. These metrics tell you whether
# your strategy has edge (positive expected value) and how much risk it takes
# to capture that edge.

@dataclass
class PerformanceMetrics:
    """Computed performance statistics for a trading strategy."""
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_fees: float = 0.0
    avg_trade_pnl: float = 0.0


def compute_performance(
    trades: list[dict[str, Any]],
    initial_capital: float,
    price_history: list[float],
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics from trade history.

    These metrics are standard in quantitative finance. Every hedge fund
    reports Sharpe ratio and max drawdown to their investors. Understanding
    what they measure (and what they don't) is essential for evaluating
    any systematic strategy.
    """
    metrics = PerformanceMetrics()

    if not trades:
        return metrics

    metrics.num_trades = len(trades)
    metrics.total_fees = sum(t["fee"] for t in trades)

    # --- Return calculation ---
    final_equity = trades[-1]["portfolio_equity"]
    metrics.total_return_pct = (final_equity - initial_capital) / initial_capital * 100

    # --- Win rate and profit factor ---
    # A trade is "winning" if its realized P&L is positive. Note: unrealized
    # P&L doesn't count — you haven't locked in the gain/loss yet.
    pnls = [t["realized_pnl"] - t["fee"] for t in trades]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    wins = sum(1 for p in pnls if p > 0)

    metrics.win_rate = wins / len(pnls) * 100 if pnls else 0.0
    metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    metrics.avg_trade_pnl = sum(pnls) / len(pnls) if pnls else 0.0

    # --- Equity curve and drawdown ---
    # Build equity curve from trade snapshots
    equity_curve = [initial_capital]
    for t in trades:
        equity_curve.append(t["portfolio_equity"])

    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    metrics.max_drawdown_pct = max_dd * 100

    # --- Sharpe ratio ---
    # Sharpe = mean(returns) / std(returns) * sqrt(annualization_factor)
    # For daily returns with 252 trading days: sqrt(252)
    # For our tick-based simulation, we use the raw ratio without annualization
    # since "ticks" don't map to real time periods
    if len(equity_curve) >= 2:
        returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))
            if equity_curve[i - 1] != 0
        ]
        if returns:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            std_ret = math.sqrt(variance) if variance > 0 else 1e-10
            # Annualize assuming 252 trading periods
            metrics.sharpe_ratio = (mean_ret / std_ret) * math.sqrt(252)

    return metrics


# =============================================================================
# Main: Wire it all together and run
# =============================================================================

def run_simulation(seed: int = 42) -> tuple[PerformanceMetrics, list[dict], EventBus]:
    """
    Run a complete trading bot simulation.

    This function demonstrates the full architecture:
    1. Create the event bus (communication backbone)
    2. Instantiate all components with their configs
    3. Run the market simulation tick by tick
    4. Compute and return performance metrics

    The seed parameter ensures reproducibility — same seed, same results.
    This is critical for debugging: you can replay exact scenarios.
    """
    random.seed(seed)

    initial_capital = 10000.0

    # Create the event bus — the central nervous system
    bus = EventBus()

    # Market simulator — the environment our bot operates in
    market_config = MarketConfig(
        initial_price=100.0,
        drift=0.0002,        # Slight bullish trend
        volatility=0.015,    # Moderate volatility
        spread_bps=10.0,
        slippage_bps=3.0,
        num_ticks=500,
    )
    market = MarketSimulator(market_config, bus)

    # Signal generator — the brain
    signal_config = SignalConfig(
        fast_window=10,
        slow_window=30,
        rsi_window=14,
        rsi_overbought=70.0,
        rsi_oversold=30.0,
    )
    signal_gen = SignalGenerator(signal_config, bus)

    # Risk engine — the safety controller
    risk_config = RiskConfig(
        max_position_size=50.0,
        max_portfolio_drawdown=0.10,
        daily_loss_limit=500.0,
        order_size=10.0,
    )
    risk_engine = RiskEngine(risk_config, bus, initial_capital)

    # Execution engine — the hands
    exec_engine = ExecutionEngine(bus, market, initial_capital)

    # Risk alert logger — subscribe to see when risk limits are hit
    risk_alerts: list[Event] = []
    bus.subscribe(EventType.RISK_ALERT, lambda e: risk_alerts.append(e))

    # ---- Run the simulation ----
    for _ in range(market_config.num_ticks):
        market.step()

    # ---- Compute results ----
    metrics = compute_performance(exec_engine.trades, initial_capital, market.price_history)

    return metrics, exec_engine.trades, bus, risk_engine, signal_gen, risk_alerts, market


def print_results(
    metrics: PerformanceMetrics,
    trades: list[dict],
    risk_engine: RiskEngine,
    signal_gen: SignalGenerator,
    risk_alerts: list[Event],
    market: MarketSimulator,
) -> None:
    """Print a comprehensive summary of the simulation results."""

    print("=" * 70)
    print("   AUTONOMOUS TRADING BOT — SIMULATION RESULTS")
    print("=" * 70)

    # Market summary
    print(f"\n{'MARKET SUMMARY':=^70}")
    print(f"  Start price:    ${market.price_history[0]:.2f}")
    print(f"  End price:      ${market.price_history[-1]:.2f}")
    price_change = (market.price_history[-1] - market.price_history[0]) / market.price_history[0] * 100
    print(f"  Price change:   {price_change:+.2f}%")
    print(f"  Ticks:          {len(market.price_history) - 1}")

    # Signal summary
    print(f"\n{'SIGNAL GENERATION':=^70}")
    print(f"  Total signals:   {signal_gen.signal_count}")
    print(f"  Orders approved: {risk_engine.orders_approved}")
    print(f"  Orders rejected: {risk_engine.orders_rejected}")
    if risk_alerts:
        print(f"  Risk alerts:     {len(risk_alerts)}")
        for alert in risk_alerts[:3]:
            print(f"    - {alert.data['reason']} at tick {alert.timestamp}")

    # Trade log (first 5 and last 5)
    print(f"\n{'TRADE LOG (sample)':=^70}")
    print(f"  {'Tick':>5} {'Side':<5} {'Size':>6} {'Price':>9} {'PnL':>9} {'Position':>9}")
    print(f"  {'-'*5} {'-'*5} {'-'*6} {'-'*9} {'-'*9} {'-'*9}")
    show_trades = trades[:5] + ([{"sep": True}] if len(trades) > 10 else []) + trades[-5:] if len(trades) > 10 else trades
    for t in show_trades:
        if "sep" in t:
            print(f"  {'...':>5} {'...':^5} {'...':>6} {'...':>9} {'...':>9} {'...':>9}")
            continue
        side_str = t["side"].value if isinstance(t["side"], Side) else t["side"]
        print(f"  {t['tick']:>5} {side_str:<5} {t['size']:>6.1f} {t['fill_price']:>9.2f} {t['realized_pnl']:>+9.2f} {t['position_after']:>+9.1f}")

    # Performance metrics
    print(f"\n{'PERFORMANCE METRICS':=^70}")
    print(f"  Total return:    {metrics.total_return_pct:+.2f}%")
    print(f"  Sharpe ratio:    {metrics.sharpe_ratio:.2f}")
    print(f"  Max drawdown:    {metrics.max_drawdown_pct:.2f}%")
    print(f"  Win rate:        {metrics.win_rate:.1f}%")
    print(f"  Profit factor:   {metrics.profit_factor:.2f}")
    print(f"  Avg trade P&L:   ${metrics.avg_trade_pnl:.2f}")
    print(f"  Total fees:      ${metrics.total_fees:.2f}")
    print(f"  Number of trades: {metrics.num_trades}")

    # Risk engine state
    print(f"\n{'RISK ENGINE STATE':=^70}")
    print(f"  Final position:  {risk_engine.position:+.1f} units")
    print(f"  Peak equity:     ${risk_engine.peak_equity:.2f}")
    print(f"  Current equity:  ${risk_engine.current_equity:.2f}")
    print(f"  Halted:          {risk_engine.halted}")

    # Architecture diagram
    print(f"\n{'ARCHITECTURE':=^70}")
    print("""
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
    │ MarketSimulator  │────>│ SignalGenerator   │────>│ RiskEngine  │
    │ (GBM prices)     │     │ (MA cross + RSI)  │     │ (limits)    │
    └─────────────────┘     └──────────────────┘     └──────┬──────┘
                                                            │
                                                            v
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
    │ PortfolioTracker │<────│ Fill events       │<────│ Execution   │
    │ (metrics)        │     │ (P&L, position)   │     │ (spread+slip│
    └─────────────────┘     └──────────────────┘     └─────────────┘

    All components communicate via EventBus (pub/sub)
    """)

    print("=" * 70)


if __name__ == "__main__":
    # Run with a fixed seed for reproducibility
    result = run_simulation(seed=42)
    metrics, trades, bus, risk_engine, signal_gen, risk_alerts, market = result

    print_results(metrics, trades, risk_engine, signal_gen, risk_alerts, market)

    # Show event bus statistics
    print("\nEvent bus statistics:")
    from collections import Counter
    event_counts = Counter(e.event_type.name for e in bus.event_log)
    for etype, count in event_counts.most_common():
        print(f"  {etype}: {count} events")
