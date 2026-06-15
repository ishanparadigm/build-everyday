"""
Day 075: Full Autonomous Trading Agent — Your Implementation

Build a complete autonomous trading agent combining:
  - Q-learning with neural network function approximation
  - On-chain signal generation
  - Kelly criterion position sizing
  - Hard-constraint risk management
  - Event-driven execution

Run tests: python3 -m pytest tests.py -v
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Market Simulator
# Hint: Generate price series using geometric returns with regime changes.
#       Each regime (trending, mean-reverting, volatile) has different drift
#       and volatility parameters.
# ---------------------------------------------------------------------------

class MarketRegime(IntEnum):
    TRENDING = 0
    MEAN_REVERTING = 1
    VOLATILE = 2


@dataclass
class MarketTick:
    """A single market data point with price and on-chain metrics."""
    timestamp: int
    price: float
    volume: float
    whale_flow: float
    active_addresses: float
    dex_imbalance: float
    regime: MarketRegime


class MarketSimulator:
    """Generates synthetic market data with regime changes and correlated
    on-chain signals."""

    def __init__(self, seed: int = 42, initial_price: float = 100.0, num_ticks: int = 2000):
        self.rng = random.Random(seed)
        self.initial_price = initial_price
        self.num_ticks = num_ticks

    def generate(self) -> list[MarketTick]:
        """Generate a list of MarketTick objects with realistic price dynamics
        and correlated on-chain signals.

        Hint: Use a Markov chain for regime transitions. Each regime has:
          - TRENDING: positive drift, moderate vol
          - MEAN_REVERTING: drift toward initial_price, low vol
          - VOLATILE: no drift, high vol

        On-chain signals should slightly lead price movements.
        """
        raise NotImplementedError("TODO: implement market simulator")


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# Hint: Compute normalized features from raw market data. Think about what
#       information the agent needs: momentum, volatility, on-chain signals,
#       and its own position state.
# ---------------------------------------------------------------------------

class FeatureEngine:
    """Converts raw market ticks into a fixed-size state vector."""

    STATE_DIM = 8

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.price_history: list[float] = []
        self.volume_history: list[float] = []

    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()

    def update(self, tick: MarketTick, position_ratio: float, unrealized_pnl: float) -> list[float]:
        """Add new tick and return the current state vector.

        Features should be:
          [0] price_momentum    — log return over lookback, scaled to [-1, 1]
          [1] price_volatility  — rolling std of returns, normalized to [0, 1]
          [2] volume_zscore     — (volume - mean) / std, clipped to [-1, 1]
          [3] whale_flow        — clipped to [-1, 1]
          [4] network_activity  — already [0, 1]
          [5] dex_imbalance     — already [-1, 1]
          [6] position_ratio    — current position / max position
          [7] unrealized_pnl    — normalized PnL

        Hint: Return zeros until you have enough history (lookback + 1 ticks).
        """
        raise NotImplementedError("TODO: implement feature engineering")


# ---------------------------------------------------------------------------
# 3. Q-Network
# Hint: 2-layer MLP with ReLU activation. Forward pass computes Q-values,
#       backward pass uses chain rule through ReLU (gradient is 0 where
#       pre-activation < 0, else passthrough).
# ---------------------------------------------------------------------------

class QNetwork:
    """Simple 2-layer MLP for Q-value estimation."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 32,
                 learning_rate: float = 0.001, seed: int = 42):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        # TODO: Initialize weights with Xavier initialization
        # Hint: scale = sqrt(2 / fan_in)
        raise NotImplementedError("TODO: implement Q-network initialization")

    def forward(self, state: list[float]) -> list[float]:
        """Forward pass: state -> hidden (ReLU) -> Q-values (linear).

        Hint: Cache intermediate values for backward pass.
        """
        raise NotImplementedError("TODO: implement forward pass")

    def update(self, state: list[float], action: int, target: float) -> float:
        """One step of SGD on loss = (Q(s,a) - target)^2.

        Hint: Backprop through output layer first, then through ReLU
        (zero gradient where pre-activation was negative), then through
        input layer. Return the loss for monitoring.
        """
        raise NotImplementedError("TODO: implement backward pass and weight update")

    def copy_from(self, other: "QNetwork") -> None:
        """Copy weights from another network (for target network)."""
        raise NotImplementedError("TODO: implement weight copying")


# ---------------------------------------------------------------------------
# 4. Actions
# ---------------------------------------------------------------------------

class Action(IntEnum):
    STRONG_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    STRONG_BUY = 4


ACTION_POSITION_MAP = {
    Action.STRONG_SELL: -1.0,
    Action.SELL: -0.5,
    Action.HOLD: 0.0,
    Action.BUY: 0.5,
    Action.STRONG_BUY: 1.0,
}


# ---------------------------------------------------------------------------
# 5. Risk Manager
# Hint: The risk manager wraps the agent's decisions with hard constraints.
#       It uses Kelly criterion for sizing and drawdown limits for scaling.
#       Key insight: these are NON-NEGOTIABLE — the agent proposes, the risk
#       manager disposes.
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    max_position_pct: float = 0.20
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    cooldown_after_losses: int = 3
    cooldown_ticks: int = 10
    fractional_kelly: float = 0.25


class RiskManager:
    """Enforces hard risk constraints on the agent's proposed actions."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.daily_start_equity: Optional[float] = None
        self.peak_equity = 0.0
        self.trade_history: list[float] = []

    def reset_daily(self, equity: float) -> None:
        self.daily_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def kelly_fraction(self) -> float:
        """Compute fractional Kelly criterion from recent trade history.

        f* = (p * b - q) / b
        where p = win probability, b = win/loss ratio, q = 1 - p

        Hint: Need at least 10 trades for meaningful statistics.
        Multiply full Kelly by fractional_kelly for safety.
        """
        raise NotImplementedError("TODO: implement Kelly criterion")

    def drawdown_scale(self, equity: float) -> float:
        """Linear scaling: 1.0 at 0% drawdown, 0.0 at max_drawdown.

        Hint: Track peak_equity and compute current drawdown percentage.
        """
        raise NotImplementedError("TODO: implement drawdown scaling")

    def filter_action(self, action: Action, equity: float, current_price: float) -> tuple[Action, float]:
        """Apply risk constraints. Returns (action, position_size_in_units).

        Checks in order:
        1. Cooldown — force HOLD if cooling down
        2. Daily loss limit — force HOLD if daily loss exceeded
        3. If HOLD, return 0 size
        4. Compute size = min(max_position, kelly * intensity * dd_scale) * equity / price
        5. Apply direction (negative for sells)
        """
        raise NotImplementedError("TODO: implement risk filtering")

    def record_trade(self, pnl: float) -> None:
        """Record trade result for Kelly and cooldown tracking.

        Hint: Track consecutive losses. After cooldown_after_losses
        consecutive losses, set cooldown_remaining = cooldown_ticks.
        """
        raise NotImplementedError("TODO: implement trade recording")


# ---------------------------------------------------------------------------
# 6. Execution Engine
# Hint: Slippage is proportional to order_size * price / volume.
#       Fees are in basis points (1 bps = 0.01%).
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    timestamp: int
    side: str
    size: float
    price: float
    fee: float
    slippage: float


class ExecutionEngine:
    """Simulates order execution with slippage and fees."""

    FEE_BPS = 10

    def __init__(self, slippage_factor: float = 0.001):
        self.slippage_factor = slippage_factor

    def execute(self, timestamp: int, price: float, size: float, volume: float) -> Optional[Fill]:
        """Execute a trade. size > 0 = buy, size < 0 = sell.

        Slippage: impact = slippage_factor * |size| * price / volume
        Buy pushes price up, sell pushes price down.
        Fee = |size| * exec_price * FEE_BPS / 10000
        """
        raise NotImplementedError("TODO: implement execution engine")


# ---------------------------------------------------------------------------
# 7. Portfolio Tracker
# Hint: Track cash, position, avg entry price. When closing a position,
#       compute realized PnL. When adding to a position, update the
#       weighted average entry price.
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    initial_cash: float = 10000.0
    cash: float = 10000.0
    position: float = 0.0
    avg_entry_price: float = 0.0
    total_fees: float = 0.0
    total_trades: int = 0
    equity_history: list[float] = field(default_factory=list)

    def equity(self, current_price: float) -> float:
        """Total portfolio value = cash + position * price."""
        raise NotImplementedError("TODO: implement equity calculation")

    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized PnL = position * (current_price - avg_entry_price)."""
        raise NotImplementedError("TODO: implement unrealized PnL")

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Unrealized PnL as fraction of equity."""
        raise NotImplementedError("TODO: implement unrealized PnL percentage")

    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill to the portfolio. Returns realized PnL.

        Hint: When reducing a position, compute realized PnL from the
        difference between fill price and avg entry. When adding to a
        position, compute weighted average entry price.
        """
        raise NotImplementedError("TODO: implement fill application")


# ---------------------------------------------------------------------------
# 8. Experience Replay
# Hint: Fixed-size buffer with random sampling. Breaks temporal correlation.
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 5000, seed: int = 42):
        self.capacity = capacity
        self.buffer: list[Experience] = []
        self.rng = random.Random(seed)

    def push(self, exp: Experience) -> None:
        """Add experience, evicting oldest if at capacity."""
        raise NotImplementedError("TODO: implement replay push")

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a random batch of experiences."""
        raise NotImplementedError("TODO: implement replay sample")

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 9. Trading Agent
# Hint: DQN with epsilon-greedy, target network, and experience replay.
#       The target network stabilizes learning by providing consistent
#       TD targets (updated periodically, not every step).
# ---------------------------------------------------------------------------

class TradingAgent:
    def __init__(self, state_dim: int = FeatureEngine.STATE_DIM,
                 num_actions: int = len(Action),
                 hidden_dim: int = 32,
                 lr: float = 0.001,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 50,
                 batch_size: int = 32,
                 seed: int = 42):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        raise NotImplementedError("TODO: initialize Q-networks, target network, and replay buffer")

    def select_action(self, state: list[float]) -> Action:
        """Epsilon-greedy: with probability epsilon, random action;
        otherwise, argmax of Q-values.
        """
        raise NotImplementedError("TODO: implement action selection")

    def learn_step(self) -> float:
        """Sample batch, compute TD targets, update Q-network. Return avg loss.

        TD target: r + gamma * max_a' Q_target(s', a') if not done, else r
        """
        raise NotImplementedError("TODO: implement learning step")

    def store_experience(self, exp: Experience) -> None:
        raise NotImplementedError("TODO: implement experience storage")


# ---------------------------------------------------------------------------
# 10. Reward Function
# ---------------------------------------------------------------------------

def compute_reward(prev_equity: float, curr_equity: float, volatility: float,
                   drawdown_pct: float) -> float:
    """Risk-adjusted reward = return / vol - drawdown_penalty.

    Hint: Penalize drawdowns above 5% to discourage reckless strategies.
    """
    raise NotImplementedError("TODO: implement reward function")


# ---------------------------------------------------------------------------
# Main — test your implementation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing your implementation...\n")

    # Test 1: Market simulator
    print("1. Market Simulator")
    sim = MarketSimulator(seed=42, num_ticks=100)
    ticks = sim.generate()
    print(f"   Generated {len(ticks)} ticks")
    print(f"   Price range: {min(t.price for t in ticks):.2f} - {max(t.price for t in ticks):.2f}")

    # Test 2: Feature engine
    print("\n2. Feature Engine")
    fe = FeatureEngine(lookback=20)
    state = fe.update(ticks[0], 0.0, 0.0)
    print(f"   State dim: {len(state)}")

    # Test 3: Q-Network
    print("\n3. Q-Network")
    qn = QNetwork(state_dim=8, num_actions=5, hidden_dim=16)
    q_vals = qn.forward([0.1] * 8)
    print(f"   Q-values: {[f'{v:.4f}' for v in q_vals]}")

    # Test 4: Portfolio
    print("\n4. Portfolio")
    port = Portfolio(initial_cash=10000)
    print(f"   Equity: ${port.equity(100.0):.2f}")

    # Test 5: Risk Manager
    print("\n5. Risk Manager")
    rm = RiskManager(RiskConfig())
    rm.daily_start_equity = 10000
    rm.peak_equity = 10000
    action, size = rm.filter_action(Action.BUY, 10000.0, 100.0)
    print(f"   Action: {action.name}, Size: {size:.4f}")

    # Test 6: Execution
    print("\n6. Execution Engine")
    ee = ExecutionEngine()
    fill = ee.execute(0, 100.0, 1.0, 1000.0)
    if fill:
        print(f"   Fill: {fill.side} {fill.size:.4f} @ ${fill.price:.4f}")

    # Test 7: Reward
    print("\n7. Reward Function")
    r = compute_reward(10000, 10050, 0.01, 0.02)
    print(f"   Reward: {r:.4f}")

    print("\nAll components working! Ready for full training.")
