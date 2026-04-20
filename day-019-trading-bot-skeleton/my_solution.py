"""
Day 019: Autonomous Trading Bot Skeleton — Your Implementation

Build an event-driven trading bot with:
- Event bus for decoupled communication
- Market simulation with GBM price model
- Signal generation using MA crossover + RSI
- Risk engine with position limits and drawdown circuit breaker
- Execution engine with spread and slippage simulation

Hints:
- Start with the EventBus — everything depends on it
- The signal generator needs enough price history before emitting signals
- The risk engine should be strictly conservative — it can reduce orders, never amplify
- Track average entry price for correct P&L calculation
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

class EventType(Enum):
    """All event types in the system."""
    MARKET_DATA = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    RISK_ALERT = auto()


@dataclass
class Event:
    """Base event with type and arbitrary payload."""
    event_type: EventType
    data: dict[str, Any]
    timestamp: int = 0


class EventBus:
    """
    Publish-subscribe event bus.

    Hint: Use a defaultdict(list) to map EventType -> list of callbacks.
    Keep an event log for debugging and analytics.
    """

    def __init__(self) -> None:
        raise NotImplementedError("TODO: initialize subscriber dict and event log")

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Register a callback for a specific event type."""
        raise NotImplementedError("TODO: append callback to subscribers for this event type")

    def publish(self, event: Event) -> None:
        """Dispatch an event to all subscribers. Log every event."""
        raise NotImplementedError("TODO: log the event, then call each subscriber's callback")

    @property
    def event_log(self) -> list[Event]:
        raise NotImplementedError("TODO: return the event log")


# =============================================================================
# Market Simulation
# =============================================================================

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class MarketConfig:
    """Configuration for the market simulator."""
    initial_price: float = 100.0
    drift: float = 0.0001
    volatility: float = 0.02
    spread_bps: float = 10.0
    slippage_bps: float = 5.0
    num_ticks: int = 500


class MarketSimulator:
    """
    Generates synthetic market data using geometric Brownian motion.

    Hint: GBM discrete step with Ito correction:
        log_return = drift - 0.5 * volatility^2 + volatility * Z
        price *= exp(log_return)
    where Z ~ N(0,1). The -0.5*sigma^2 prevents volatility from creating fake drift.
    """

    def __init__(self, config: MarketConfig, bus: EventBus) -> None:
        raise NotImplementedError("TODO: store config, bus, initial price, and price history")

    def step(self) -> None:
        """Advance market by one tick. Publish MarketData event with mid, bid, ask."""
        raise NotImplementedError(
            "TODO: compute new price via GBM, calculate bid/ask from spread, publish event"
        )


# =============================================================================
# Signal Generation
# =============================================================================

@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    fast_window: int = 10
    slow_window: int = 30
    rsi_window: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


class SignalGenerator:
    """
    Generates trading signals from MA crossover with RSI confirmation.

    Hint: You need to track:
    - Price history (deque with maxlen=slow_window)
    - Gains and losses (deques with maxlen=rsi_window) for RSI
    - Previous crossover state to detect CHANGES, not just levels

    Signal logic:
    - Fast MA crosses ABOVE slow MA + RSI < overbought -> BUY
    - Fast MA crosses BELOW slow MA + RSI > oversold -> SELL
    """

    def __init__(self, config: SignalConfig, bus: EventBus) -> None:
        raise NotImplementedError("TODO: initialize config, price/gain/loss deques, subscribe to MARKET_DATA")

    def _compute_sma(self, window: int) -> float | None:
        """Compute Simple Moving Average. Return None if not enough data."""
        raise NotImplementedError("TODO: average the last `window` prices, or None if insufficient data")

    def _compute_rsi(self) -> float | None:
        """
        Compute RSI = 100 - 100/(1 + RS), where RS = avg_gain / avg_loss.
        Return None if not enough data. Handle avg_loss == 0 (return 100.0).
        """
        raise NotImplementedError("TODO: implement RSI calculation")

    def on_market_data(self, event: Event) -> None:
        """Process price tick: update windows, detect crossover, apply RSI filter, emit signal."""
        raise NotImplementedError(
            "TODO: update prices/gains/losses, compute MAs and RSI, detect crossover, emit SIGNAL event"
        )


# =============================================================================
# Risk Engine
# =============================================================================

@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_position_size: float = 50.0
    max_portfolio_drawdown: float = 0.10
    daily_loss_limit: float = 500.0
    order_size: float = 10.0


class RiskEngine:
    """
    Evaluates signals against risk constraints before allowing execution.

    Hint: Check order of risk checks (cheapest first):
    1. Is the engine halted? (circuit breaker)
    2. Drawdown check: (peak - current) / peak < max_drawdown
    3. Daily loss limit: daily_pnl > -limit
    4. Position limit: clip order size if it would exceed max_position

    The engine can REDUCE or REJECT orders, never amplify them.
    """

    def __init__(self, config: RiskConfig, bus: EventBus, initial_capital: float) -> None:
        raise NotImplementedError(
            "TODO: store config, initialize position/pnl/equity tracking, subscribe to SIGNAL and FILL"
        )

    def _check_drawdown(self) -> bool:
        """Return True if drawdown is within acceptable limits."""
        raise NotImplementedError("TODO: compare (peak - current) / peak against max_drawdown")

    def _check_position_limit(self, side: Side, size: float) -> float:
        """Return the allowed order size (may be clipped or zero)."""
        raise NotImplementedError("TODO: clip size if position + size would exceed max_position_size")

    def _check_daily_loss(self) -> bool:
        """Return True if daily loss is within limits."""
        raise NotImplementedError("TODO: check if daily_pnl > -daily_loss_limit")

    def on_signal(self, event: Event) -> None:
        """Run signal through risk checks. Emit ORDER if approved."""
        raise NotImplementedError("TODO: check halted, drawdown, daily loss, position limit, then emit ORDER")

    def on_fill(self, event: Event) -> None:
        """Update position and P&L tracking after a fill."""
        raise NotImplementedError("TODO: update position, daily_pnl, current_equity from fill data")


# =============================================================================
# Execution Engine
# =============================================================================

class ExecutionEngine:
    """
    Simulates order fills with spread and slippage.

    Hint: Fill price for buys = mid + half_spread + slippage*size/100
          Fill price for sells = mid - half_spread - slippage*size/100
    Track average entry price (weighted) for P&L calculation.
    Portfolio equity = cash + position * current_price
    """

    def __init__(self, bus: EventBus, market: MarketSimulator, initial_capital: float) -> None:
        raise NotImplementedError("TODO: store bus/market, init cash/position/avg_entry, subscribe to ORDER")

    def _compute_fill_price(self, side: Side, size: float) -> float:
        """Compute execution price including spread and slippage."""
        raise NotImplementedError("TODO: mid +/- half_spread +/- slippage proportional to size")

    def on_order(self, event: Event) -> None:
        """Execute order: compute fill price, update position/cash, publish FILL event."""
        raise NotImplementedError(
            "TODO: fill the order, calculate realized P&L, update position, emit FILL event"
        )


# =============================================================================
# Performance Analytics
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Computed performance statistics."""
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
    Compute performance metrics from trade history.

    Hint:
    - Total return: (final_equity - initial) / initial * 100
    - Win rate: count trades with realized_pnl - fee > 0
    - Profit factor: gross_profit / gross_loss
    - Sharpe: mean(returns) / std(returns) * sqrt(252)
    - Max drawdown: largest peak-to-trough decline in equity curve
    """
    raise NotImplementedError("TODO: implement all performance metric calculations")


# =============================================================================
# Main
# =============================================================================

def run_simulation(seed: int = 42):
    """
    Wire all components together and run the simulation.

    Steps:
    1. Create EventBus
    2. Create MarketSimulator, SignalGenerator, RiskEngine, ExecutionEngine
    3. Run market.step() for num_ticks iterations
    4. Compute and return performance metrics
    """
    raise NotImplementedError("TODO: create all components, run simulation, return results")


if __name__ == "__main__":
    result = run_simulation(seed=42)
    if result:
        metrics, trades, bus, risk_engine, signal_gen, risk_alerts, market = result
        print(f"Total return: {metrics.total_return_pct:+.2f}%")
        print(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Max drawdown: {metrics.max_drawdown_pct:.2f}%")
        print(f"Win rate:     {metrics.win_rate:.1f}%")
        print(f"Trades:       {metrics.num_trades}")
