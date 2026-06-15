"""
Day 075: Full Autonomous Trading Agent

A complete trading system integrating:
  - Q-learning with neural network function approximation for trade decisions
  - On-chain signal generation (volume, whale activity, network metrics)
  - Kelly criterion position sizing with fractional scaling
  - Hard-constraint risk management (drawdown limits, daily loss, cooldown)
  - Event-driven architecture with realistic execution simulation

Run: python3 solution.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Market Simulator — Generates realistic price series with regime changes
# ---------------------------------------------------------------------------

class MarketRegime(IntEnum):
    """Market operates in distinct regimes that change the statistical properties
    of returns. Trending markets have positive drift, mean-reverting markets
    oscillate around a level, and volatile markets have wide swings."""
    TRENDING = 0
    MEAN_REVERTING = 1
    VOLATILE = 2


@dataclass
class MarketTick:
    """A single market data point with price and on-chain metrics."""
    timestamp: int
    price: float
    volume: float
    whale_flow: float        # net large-holder flow: positive = accumulation
    active_addresses: float  # normalized network activity
    dex_imbalance: float     # (buy_vol - sell_vol) / total_vol in [-1, 1]
    regime: MarketRegime


class MarketSimulator:
    """Generates synthetic market data with regime changes and correlated on-chain
    signals. The key insight: on-chain signals slightly *lead* price movements,
    giving the agent an informational edge if it can learn to read them."""

    def __init__(self, seed: int = 42, initial_price: float = 100.0, num_ticks: int = 2000):
        self.rng = random.Random(seed)
        self.initial_price = initial_price
        self.num_ticks = num_ticks

    def generate(self) -> list[MarketTick]:
        ticks: list[MarketTick] = []
        price = self.initial_price
        regime = MarketRegime.TRENDING
        regime_duration = 0

        for t in range(self.num_ticks):
            # --- Regime transitions (Markov chain) ---
            # Average regime lasts ~100-300 ticks. This creates non-stationarity
            # that the agent must adapt to.
            regime_duration += 1
            if regime_duration > self.rng.randint(100, 300):
                regime = MarketRegime(self.rng.randint(0, 2))
                regime_duration = 0

            # --- Price dynamics per regime ---
            if regime == MarketRegime.TRENDING:
                # Positive drift with moderate noise
                drift = 0.0003
                vol = 0.008
            elif regime == MarketRegime.MEAN_REVERTING:
                # Mean reversion toward initial price
                drift = -0.001 * (price - self.initial_price) / self.initial_price
                vol = 0.006
            else:  # VOLATILE
                # No drift, high volatility
                drift = 0.0
                vol = 0.02

            # Geometric return: price * exp(drift + vol * noise)
            noise = self.rng.gauss(0, 1)
            ret = drift + vol * noise
            price = price * math.exp(ret)

            # --- On-chain signals (correlated with FUTURE price, creating alpha) ---
            # Volume leads price: high volume predicts trend continuation
            base_volume = 1000.0
            volume = base_volume * (1.0 + 0.5 * abs(noise) + 0.3 * self.rng.gauss(0, 1))

            # Whale flow: slightly predictive of next-period returns
            # In trending regimes, whales accumulate; in volatile, they distribute
            if regime == MarketRegime.TRENDING:
                whale_signal = 0.3 + 0.2 * self.rng.gauss(0, 1)
            elif regime == MarketRegime.VOLATILE:
                whale_signal = -0.2 + 0.3 * self.rng.gauss(0, 1)
            else:
                whale_signal = 0.1 * self.rng.gauss(0, 1)

            # Active addresses: higher during trending, lower during mean-reversion
            active = 0.5 + (0.2 if regime == MarketRegime.TRENDING else -0.1) + 0.15 * self.rng.gauss(0, 1)
            active = max(0.0, min(1.0, active))

            # DEX imbalance: partially predictive, with noise
            dex_imb = 0.3 * math.copysign(1, drift) + 0.4 * self.rng.gauss(0, 1)
            dex_imb = max(-1.0, min(1.0, dex_imb))

            ticks.append(MarketTick(
                timestamp=t,
                price=price,
                volume=volume,
                whale_flow=whale_signal,
                active_addresses=active,
                dex_imbalance=dex_imb,
                regime=regime,
            ))

        return ticks


# ---------------------------------------------------------------------------
# 2. Feature Engineering — Transform raw data into normalized state vector
# ---------------------------------------------------------------------------

class FeatureEngine:
    """Converts raw market ticks into a fixed-size state vector for the RL agent.

    Features:
      [0] price_momentum    — rate of change over lookback window, normalized
      [1] price_volatility  — rolling std of returns, normalized
      [2] volume_zscore     — how unusual current volume is
      [3] whale_flow        — directional large-holder signal
      [4] network_activity  — active addresses (already 0-1)
      [5] dex_imbalance     — buy/sell pressure
      [6] position_ratio    — current position as fraction of max
      [7] unrealized_pnl    — current unrealized PnL, normalized
    """

    STATE_DIM = 8

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.price_history: list[float] = []
        self.volume_history: list[float] = []

    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()

    def update(self, tick: MarketTick, position_ratio: float, unrealized_pnl: float) -> list[float]:
        """Add new tick and return the current state vector."""
        self.price_history.append(tick.price)
        self.volume_history.append(tick.volume)

        # Need enough history to compute features
        if len(self.price_history) < self.lookback + 1:
            return [0.0] * self.STATE_DIM

        recent_prices = self.price_history[-(self.lookback + 1):]
        recent_volumes = self.volume_history[-self.lookback:]

        # Price momentum: log return over lookback, clamped to [-1, 1]
        # We use log returns because they're additive and approximately normal
        momentum = math.log(recent_prices[-1] / recent_prices[0])
        momentum = max(-1.0, min(1.0, momentum * 10))  # scale up for sensitivity

        # Volatility: rolling std of log returns, normalized to [0, 1]
        log_returns = [math.log(recent_prices[i + 1] / recent_prices[i])
                       for i in range(len(recent_prices) - 1)]
        mean_ret = sum(log_returns) / len(log_returns)
        var = sum((r - mean_ret) ** 2 for r in log_returns) / len(log_returns)
        volatility = math.sqrt(var)
        # Typical daily vol ~ 0.01-0.03, map to [0, 1]
        volatility_norm = min(1.0, volatility / 0.03)

        # Volume z-score: how many std devs above/below mean
        mean_vol = sum(recent_volumes) / len(recent_volumes)
        vol_std = math.sqrt(sum((v - mean_vol) ** 2 for v in recent_volumes) / len(recent_volumes))
        vol_zscore = (tick.volume - mean_vol) / max(vol_std, 1e-8)
        vol_zscore = max(-3.0, min(3.0, vol_zscore)) / 3.0  # normalize to [-1, 1]

        # On-chain features are already roughly normalized
        whale = max(-1.0, min(1.0, tick.whale_flow))
        network = tick.active_addresses  # already [0, 1]
        dex_imb = tick.dex_imbalance     # already [-1, 1]

        # Portfolio state
        pos_ratio = max(-1.0, min(1.0, position_ratio))
        pnl_norm = max(-1.0, min(1.0, unrealized_pnl))

        return [momentum, volatility_norm, vol_zscore, whale, network, dex_imb, pos_ratio, pnl_norm]


# ---------------------------------------------------------------------------
# 3. Neural Network for Q-Value Approximation
# ---------------------------------------------------------------------------

class QNetwork:
    """Simple 2-layer MLP for Q-value estimation. Maps state -> Q-values for
    each action. Uses ReLU activations and Xavier initialization.

    We implement manual forward/backward passes (no framework) for transparency.
    In production you'd use PyTorch, but seeing the raw math is the point here.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 32,
                 learning_rate: float = 0.001, seed: int = 42):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        rng = random.Random(seed)

        # Xavier initialization: scale by sqrt(2 / fan_in) for each layer
        # This prevents vanishing/exploding gradients at initialization
        scale1 = math.sqrt(2.0 / state_dim)
        self.W1 = [[rng.gauss(0, scale1) for _ in range(state_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim

        scale2 = math.sqrt(2.0 / hidden_dim)
        self.W2 = [[rng.gauss(0, scale2) for _ in range(hidden_dim)] for _ in range(num_actions)]
        self.b2 = [0.0] * num_actions

        # Cache for backward pass
        self._input: list[float] = []
        self._hidden: list[float] = []
        self._hidden_pre: list[float] = []

    def forward(self, state: list[float]) -> list[float]:
        """Forward pass: state -> hidden (ReLU) -> Q-values (linear output)."""
        self._input = state

        # Hidden layer: z = W1 @ state + b1, then ReLU
        self._hidden_pre = [
            sum(self.W1[h][i] * state[i] for i in range(self.state_dim)) + self.b1[h]
            for h in range(self.hidden_dim)
        ]
        self._hidden = [max(0.0, z) for z in self._hidden_pre]

        # Output layer: Q = W2 @ hidden + b2 (linear — no activation on Q-values)
        q_values = [
            sum(self.W2[a][h] * self._hidden[h] for h in range(self.hidden_dim)) + self.b2[a]
            for a in range(self.num_actions)
        ]
        return q_values

    def update(self, state: list[float], action: int, target: float) -> float:
        """One step of SGD. Computes loss = (Q(s,a) - target)^2 and updates weights.

        Returns the loss for monitoring convergence."""
        q_values = self.forward(state)
        prediction = q_values[action]
        error = prediction - target
        loss = error ** 2

        # Gradient of loss w.r.t. Q(s,a) output: d_loss/d_Q = 2 * error
        # We absorb the 2 into the learning rate for simplicity
        d_output = [0.0] * self.num_actions
        d_output[action] = error

        # Backprop through output layer: d_loss/d_W2[a][h] = d_output[a] * hidden[h]
        d_hidden = [0.0] * self.hidden_dim
        for a in range(self.num_actions):
            if d_output[a] == 0.0:
                continue
            for h in range(self.hidden_dim):
                # Weight gradient
                self.W2[a][h] -= self.lr * d_output[a] * self._hidden[h]
                # Accumulate gradient for hidden layer
                d_hidden[h] += self.W2[a][h] * d_output[a]
            self.b2[a] -= self.lr * d_output[a]

        # Backprop through ReLU: zero gradient where pre-activation was negative
        d_hidden_pre = [d_hidden[h] if self._hidden_pre[h] > 0 else 0.0
                        for h in range(self.hidden_dim)]

        # Backprop through first layer
        for h in range(self.hidden_dim):
            if d_hidden_pre[h] == 0.0:
                continue
            for i in range(self.state_dim):
                self.W1[h][i] -= self.lr * d_hidden_pre[h] * self._input[i]
            self.b1[h] -= self.lr * d_hidden_pre[h]

        return loss

    def copy_from(self, other: "QNetwork") -> None:
        """Copy weights from another network (for target network updates)."""
        for h in range(self.hidden_dim):
            for i in range(self.state_dim):
                self.W1[h][i] = other.W1[h][i]
            self.b1[h] = other.b1[h]
        for a in range(self.num_actions):
            for h in range(self.hidden_dim):
                self.W2[a][h] = other.W2[a][h]
            self.b2[a] = other.b2[a]


# ---------------------------------------------------------------------------
# 4. Actions and Trading Logic
# ---------------------------------------------------------------------------

class Action(IntEnum):
    STRONG_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    STRONG_BUY = 4


# Maps action to target position fraction of max_position
ACTION_POSITION_MAP = {
    Action.STRONG_SELL: -1.0,
    Action.SELL: -0.5,
    Action.HOLD: 0.0,
    Action.BUY: 0.5,
    Action.STRONG_BUY: 1.0,
}


# ---------------------------------------------------------------------------
# 5. Risk Manager — The non-negotiable safety layer
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    max_position_pct: float = 0.20       # max 20% of portfolio in one position
    daily_loss_limit_pct: float = 0.03   # stop trading after 3% daily loss
    max_drawdown_pct: float = 0.15       # scale down at 15% drawdown
    cooldown_after_losses: int = 3       # consecutive losses before cooldown
    cooldown_ticks: int = 10             # how long to cool down
    fractional_kelly: float = 0.25       # use quarter-Kelly for sizing


class RiskManager:
    """Enforces hard risk constraints. The RL agent proposes actions; the risk
    manager decides what actually gets executed. This separation is critical —
    an RL agent optimizing returns will happily take leveraged bets unless
    something stops it."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.daily_start_equity: Optional[float] = None
        self.peak_equity = 0.0
        self.trade_history: list[float] = []  # recent PnLs for Kelly

    def reset_daily(self, equity: float) -> None:
        self.daily_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def kelly_fraction(self) -> float:
        """Compute fractional Kelly criterion from recent trade history.

        f* = (p * b - q) / b, then multiply by fractional_kelly.

        If we don't have enough history, return a conservative default."""
        if len(self.trade_history) < 10:
            return self.config.fractional_kelly * 0.5  # very conservative initially

        wins = [pnl for pnl in self.trade_history if pnl > 0]
        losses = [pnl for pnl in self.trade_history if pnl < 0]

        if not wins or not losses:
            return self.config.fractional_kelly * 0.5

        p = len(wins) / len(self.trade_history)
        q = 1.0 - p
        avg_win = sum(wins) / len(wins)
        avg_loss = -sum(losses) / len(losses)  # make positive

        b = avg_win / max(avg_loss, 1e-8)  # win/loss ratio

        kelly = (p * b - q) / max(b, 1e-8)
        kelly = max(0.0, kelly)  # never negative (would mean don't trade)

        return kelly * self.config.fractional_kelly

    def drawdown_scale(self, equity: float) -> float:
        """Linear scaling: full size at 0% drawdown, zero at max_drawdown.
        This progressively reduces risk as losses accumulate."""
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity <= 0:
            return 0.0
        drawdown = (self.peak_equity - equity) / self.peak_equity
        scale = 1.0 - drawdown / self.config.max_drawdown_pct
        return max(0.0, min(1.0, scale))

    def filter_action(self, action: Action, equity: float, current_price: float) -> tuple[Action, float]:
        """Apply risk constraints to the proposed action.

        Returns (filtered_action, position_size_in_units).
        position_size can be negative (short) or positive (long)."""

        # --- Cooldown check ---
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return Action.HOLD, 0.0

        # --- Daily loss limit ---
        if self.daily_start_equity is not None:
            daily_pnl = (equity - self.daily_start_equity) / self.daily_start_equity
            if daily_pnl < -self.config.daily_loss_limit_pct:
                return Action.HOLD, 0.0

        # --- If HOLD, no position change ---
        if action == Action.HOLD:
            return Action.HOLD, 0.0

        # --- Compute position size ---
        kelly = self.kelly_fraction()
        dd_scale = self.drawdown_scale(equity)

        # Target fraction of portfolio to allocate
        action_intensity = abs(ACTION_POSITION_MAP[action])
        target_fraction = min(
            self.config.max_position_pct,
            kelly * action_intensity * dd_scale,
        )

        # Convert to units (shares/tokens)
        target_value = equity * target_fraction
        position_size = target_value / max(current_price, 1e-8)

        # Apply direction
        if action in (Action.SELL, Action.STRONG_SELL):
            position_size = -position_size

        return action, position_size

    def record_trade(self, pnl: float) -> None:
        """Record a trade result for Kelly calculation and cooldown logic."""
        self.trade_history.append(pnl)
        # Keep only recent history (non-stationarity means old trades are less relevant)
        if len(self.trade_history) > 50:
            self.trade_history.pop(0)

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.cooldown_after_losses:
                self.cooldown_remaining = self.config.cooldown_ticks
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0


# ---------------------------------------------------------------------------
# 6. Execution Engine — Simulates fills with slippage and fees
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    """Result of executing an order."""
    timestamp: int
    side: str           # "BUY" or "SELL"
    size: float         # absolute units
    price: float        # execution price (includes slippage)
    fee: float          # trading fee
    slippage: float     # price impact


class ExecutionEngine:
    """Simulates order execution with realistic market microstructure effects.

    Slippage model: price impact proportional to order_size / volume.
    This captures the intuition that large orders move the market more,
    and illiquid markets have worse execution."""

    FEE_BPS = 10  # 10 basis points = 0.1% per trade

    def __init__(self, slippage_factor: float = 0.001):
        self.slippage_factor = slippage_factor

    def execute(self, timestamp: int, price: float, size: float, volume: float) -> Optional[Fill]:
        """Execute a trade. size > 0 = buy, size < 0 = sell."""
        if abs(size) < 1e-10:
            return None

        # Slippage: proportional to (order_size / volume)
        # A buy pushes price up; a sell pushes price down
        impact = self.slippage_factor * abs(size) * price / max(volume, 1.0)
        if size > 0:
            exec_price = price * (1.0 + impact)
            side = "BUY"
        else:
            exec_price = price * (1.0 - impact)
            side = "SELL"

        fee = abs(size) * exec_price * self.FEE_BPS / 10000

        return Fill(
            timestamp=timestamp,
            side=side,
            size=abs(size),
            price=exec_price,
            fee=fee,
            slippage=abs(exec_price - price),
        )


# ---------------------------------------------------------------------------
# 7. Portfolio Tracker
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    """Tracks cash, positions, and performance metrics."""
    initial_cash: float = 10000.0
    cash: float = 10000.0
    position: float = 0.0          # units held (negative = short)
    avg_entry_price: float = 0.0
    total_fees: float = 0.0
    total_trades: int = 0
    equity_history: list[float] = field(default_factory=list)

    def equity(self, current_price: float) -> float:
        return self.cash + self.position * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        if abs(self.position) < 1e-10:
            return 0.0
        return self.position * (current_price - self.avg_entry_price)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        eq = self.equity(current_price)
        if eq <= 0:
            return 0.0
        return self.unrealized_pnl(current_price) / eq

    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill to the portfolio. Returns realized PnL from this trade."""
        realized_pnl = 0.0
        signed_size = fill.size if fill.side == "BUY" else -fill.size

        # Check if we're closing/reducing an existing position
        if self.position * signed_size < 0:
            # Reducing or flipping position
            close_size = min(abs(self.position), abs(signed_size))
            if self.position > 0:
                # Closing long: sell at fill price, bought at avg_entry
                realized_pnl = close_size * (fill.price - self.avg_entry_price)
            else:
                # Closing short: buy at fill price, sold at avg_entry
                realized_pnl = close_size * (self.avg_entry_price - fill.price)

        # Update position and average entry
        new_position = self.position + signed_size
        if abs(new_position) < 1e-10:
            self.avg_entry_price = 0.0
        elif (self.position <= 0 and new_position > 0) or (self.position >= 0 and new_position < 0):
            # Flipped sides — new avg entry is the fill price
            self.avg_entry_price = fill.price
        elif abs(new_position) > abs(self.position):
            # Adding to position — weighted average entry
            old_value = abs(self.position) * self.avg_entry_price
            new_value = fill.size * fill.price
            self.avg_entry_price = (old_value + new_value) / abs(new_position)

        self.position = new_position
        self.cash -= signed_size * fill.price + fill.fee
        self.total_fees += fill.fee
        self.total_trades += 1

        return realized_pnl


# ---------------------------------------------------------------------------
# 8. Experience Replay Buffer
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """Fixed-size buffer that stores experiences for off-policy learning.

    Why replay? Two reasons:
    1. Breaks temporal correlation between consecutive samples (i.i.d. assumption)
    2. Reuses rare experiences (e.g., large drawdowns) multiple times for learning
    """

    def __init__(self, capacity: int = 5000, seed: int = 42):
        self.capacity = capacity
        self.buffer: list[Experience] = []
        self.rng = random.Random(seed)

    def push(self, exp: Experience) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        batch_size = min(batch_size, len(self.buffer))
        return self.rng.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 9. The Trading Agent — Ties everything together
# ---------------------------------------------------------------------------

class TradingAgent:
    """Q-learning agent with epsilon-greedy exploration, target network,
    and experience replay. This is essentially DQN adapted for trading."""

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

        # Online network (updated every step)
        self.q_net = QNetwork(state_dim, num_actions, hidden_dim, lr, seed)
        # Target network (updated periodically for stability)
        self.target_net = QNetwork(state_dim, num_actions, hidden_dim, lr, seed + 1)
        self.target_net.copy_from(self.q_net)

        self.replay = ReplayBuffer(seed=seed)
        self.steps = 0

    def select_action(self, state: list[float]) -> Action:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return Action(self.rng.randint(0, self.num_actions - 1))
        q_values = self.q_net.forward(state)
        best_action = max(range(self.num_actions), key=lambda a: q_values[a])
        return Action(best_action)

    def learn_step(self) -> float:
        """Sample a batch from replay and do one SGD update. Returns avg loss."""
        if len(self.replay) < self.batch_size:
            return 0.0

        batch = self.replay.sample(self.batch_size)
        total_loss = 0.0

        for exp in batch:
            # Compute TD target: r + gamma * max_a' Q_target(s', a')
            if exp.done:
                target = exp.reward
            else:
                next_q = self.target_net.forward(exp.next_state)
                target = exp.reward + self.gamma * max(next_q)

            loss = self.q_net.update(exp.state, exp.action, target)
            total_loss += loss

        # Periodically sync target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.copy_from(self.q_net)

        return total_loss / len(batch)

    def decay_epsilon(self) -> None:
        """Decay exploration rate. Call once per episode, not per step."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store_experience(self, exp: Experience) -> None:
        self.replay.push(exp)


# ---------------------------------------------------------------------------
# 10. Reward Function
# ---------------------------------------------------------------------------

def compute_reward(prev_equity: float, curr_equity: float, volatility: float,
                   drawdown_pct: float) -> float:
    """Risk-adjusted reward that penalizes drawdowns.

    reward = return / max(vol, eps) - drawdown_penalty

    This encourages the agent to find strategies with high Sharpe ratios,
    not just high returns. The drawdown penalty discourages strategies
    that achieve good averages through occasional catastrophic losses."""
    if prev_equity <= 0:
        return -1.0
    ret = (curr_equity - prev_equity) / prev_equity
    risk_adj = ret / max(volatility, 0.001)
    drawdown_penalty = 2.0 * max(0.0, drawdown_pct - 0.05)
    return risk_adj - drawdown_penalty


# ---------------------------------------------------------------------------
# 11. Baseline Strategies (for comparison)
# ---------------------------------------------------------------------------

def run_buy_and_hold(ticks: list[MarketTick], initial_cash: float = 10000.0) -> dict:
    """Buy everything at the start, hold until the end."""
    buy_price = ticks[0].price
    units = initial_cash / buy_price
    final_equity = units * ticks[-1].price

    # Compute max drawdown
    peak = initial_cash
    max_dd = 0.0
    for tick in ticks:
        equity = units * tick.price
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    total_return = (final_equity - initial_cash) / initial_cash
    return {"strategy": "Buy & Hold", "final_equity": final_equity,
            "return_pct": total_return * 100, "max_drawdown_pct": max_dd * 100,
            "num_trades": 1}


def run_ma_crossover(ticks: list[MarketTick], fast: int = 10, slow: int = 30,
                     initial_cash: float = 10000.0) -> dict:
    """Simple moving average crossover: buy when fast MA > slow MA, sell otherwise."""
    cash = initial_cash
    position = 0.0
    prices = []
    num_trades = 0
    peak_equity = initial_cash
    max_dd = 0.0

    for tick in ticks:
        prices.append(tick.price)
        equity = cash + position * tick.price
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity
        max_dd = max(max_dd, dd)

        if len(prices) < slow:
            continue

        fast_ma = sum(prices[-fast:]) / fast
        slow_ma = sum(prices[-slow:]) / slow

        if fast_ma > slow_ma and position <= 0:
            # Buy signal
            units_to_buy = cash * 0.95 / tick.price  # keep 5% cash reserve
            if units_to_buy > 0:
                position += units_to_buy
                cash -= units_to_buy * tick.price
                num_trades += 1
        elif fast_ma < slow_ma and position > 0:
            # Sell signal
            cash += position * tick.price
            position = 0.0
            num_trades += 1

    final_equity = cash + position * ticks[-1].price
    total_return = (final_equity - initial_cash) / initial_cash
    return {"strategy": "MA Crossover", "final_equity": final_equity,
            "return_pct": total_return * 100, "max_drawdown_pct": max_dd * 100,
            "num_trades": num_trades}


def run_random_trader(ticks: list[MarketTick], initial_cash: float = 10000.0,
                      seed: int = 99) -> dict:
    """Random trading with equal probability of buy/sell/hold."""
    rng = random.Random(seed)
    cash = initial_cash
    position = 0.0
    num_trades = 0
    peak_equity = initial_cash
    max_dd = 0.0

    for tick in ticks:
        equity = cash + position * tick.price
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity
        max_dd = max(max_dd, dd)

        action = rng.choice(["buy", "sell", "hold"])
        if action == "buy" and cash > tick.price:
            units = (cash * 0.1) / tick.price
            position += units
            cash -= units * tick.price
            num_trades += 1
        elif action == "sell" and position > 0:
            sell_units = position * 0.2
            cash += sell_units * tick.price
            position -= sell_units
            num_trades += 1

    final_equity = cash + position * ticks[-1].price
    total_return = (final_equity - initial_cash) / initial_cash
    return {"strategy": "Random", "final_equity": final_equity,
            "return_pct": total_return * 100, "max_drawdown_pct": max_dd * 100,
            "num_trades": num_trades}


# ---------------------------------------------------------------------------
# 12. Training and Evaluation
# ---------------------------------------------------------------------------

def train_agent(num_episodes: int = 15, ticks_per_episode: int = 2000,
                initial_cash: float = 10000.0, verbose: bool = True) -> tuple[TradingAgent, list[dict]]:
    """Train the trading agent across multiple episodes of simulated markets."""

    agent = TradingAgent(
        lr=0.0005,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.85,  # per-episode decay: 15 episodes to reach ~0.08
        target_update_freq=100,
        batch_size=32,
    )
    exec_engine = ExecutionEngine()
    feature_engine = FeatureEngine(lookback=20)

    episode_results: list[dict] = []

    for ep in range(num_episodes):
        # Each episode gets a different market (different seed = different regimes)
        sim = MarketSimulator(seed=ep * 17 + 7, initial_price=100.0, num_ticks=ticks_per_episode)
        ticks = sim.generate()

        portfolio = Portfolio(initial_cash=initial_cash, cash=initial_cash)
        feature_engine.reset()
        # Fresh risk manager each episode so early losses don't permanently suppress trading
        risk_mgr = RiskManager(RiskConfig())
        risk_mgr.daily_start_equity = initial_cash
        risk_mgr.peak_equity = initial_cash

        prev_state: Optional[list[float]] = None
        prev_action: Optional[int] = None
        prev_equity = initial_cash
        peak_equity = initial_cash
        total_reward = 0.0
        ep_trades = 0

        warmup = 25  # need enough ticks to compute features

        for i, tick in enumerate(ticks):
            current_equity = portfolio.equity(tick.price)
            portfolio.equity_history.append(current_equity)
            peak_equity = max(peak_equity, current_equity)
            drawdown_pct = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0

            pos_ratio = 0.0
            max_pos_value = current_equity * risk_mgr.config.max_position_pct
            if max_pos_value > 0 and tick.price > 0:
                pos_ratio = (portfolio.position * tick.price) / max_pos_value

            state = feature_engine.update(tick, pos_ratio, portfolio.unrealized_pnl_pct(tick.price))

            if i < warmup:
                prev_equity = current_equity
                continue

            # Compute reward from previous step
            # Use rolling volatility for risk adjustment
            recent_equities = portfolio.equity_history[-20:]
            if len(recent_equities) >= 2:
                rets = [(recent_equities[j] - recent_equities[j - 1]) / max(recent_equities[j - 1], 1)
                        for j in range(1, len(recent_equities))]
                vol = math.sqrt(sum(r ** 2 for r in rets) / len(rets)) if rets else 0.01
            else:
                vol = 0.01

            reward = compute_reward(prev_equity, current_equity, vol, drawdown_pct)
            total_reward += reward

            # Store experience from previous step
            if prev_state is not None and prev_action is not None:
                done = (i == len(ticks) - 1)
                agent.store_experience(Experience(prev_state, prev_action, reward, state, done))

            # Select and filter action
            raw_action = agent.select_action(state)
            filtered_action, size = risk_mgr.filter_action(raw_action, current_equity, tick.price)

            # Execute trade
            if abs(size) > 1e-10:
                # We need to compute the change in position
                target_pos = size if filtered_action in (Action.BUY, Action.STRONG_BUY) else size
                delta = target_pos - portfolio.position
                if abs(delta) > 1e-10:
                    fill = exec_engine.execute(tick.timestamp, tick.price, delta, tick.volume)
                    if fill:
                        realized_pnl = portfolio.apply_fill(fill)
                        if abs(realized_pnl) > 1e-10:
                            risk_mgr.record_trade(realized_pnl)
                        ep_trades += 1

            # Learn from replay buffer
            if len(agent.replay) >= agent.batch_size:
                agent.learn_step()

            prev_state = state
            prev_action = raw_action.value
            prev_equity = current_equity

        # Decay epsilon once per episode (not per step)
        agent.decay_epsilon()

        final_equity = portfolio.equity(ticks[-1].price)
        total_return = (final_equity - initial_cash) / initial_cash
        max_dd = 0.0
        peak = initial_cash
        for eq in portfolio.equity_history:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        result = {
            "episode": ep + 1,
            "final_equity": final_equity,
            "return_pct": total_return * 100,
            "max_drawdown_pct": max_dd * 100,
            "num_trades": ep_trades,
            "total_reward": total_reward,
            "epsilon": agent.epsilon,
            "total_fees": portfolio.total_fees,
        }
        episode_results.append(result)

        if verbose:
            print(f"  Episode {ep + 1:>2}/{num_episodes}:  "
                  f"Return={total_return * 100:+7.2f}%  "
                  f"MaxDD={max_dd * 100:5.2f}%  "
                  f"Trades={ep_trades:>3}  "
                  f"Fees=${portfolio.total_fees:.2f}  "
                  f"Epsilon={agent.epsilon:.3f}")

    return agent, episode_results


def evaluate_agent(agent: TradingAgent, seed: int = 999, num_ticks: int = 2000,
                   initial_cash: float = 10000.0) -> dict:
    """Evaluate the trained agent on a fresh market (no learning, greedy policy)."""
    sim = MarketSimulator(seed=seed, initial_price=100.0, num_ticks=num_ticks)
    ticks = sim.generate()

    risk_mgr = RiskManager(RiskConfig())
    exec_engine = ExecutionEngine()
    feature_engine = FeatureEngine(lookback=20)

    portfolio = Portfolio(initial_cash=initial_cash, cash=initial_cash)
    risk_mgr.daily_start_equity = initial_cash
    risk_mgr.peak_equity = initial_cash

    # Greedy evaluation — no exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    warmup = 25

    for i, tick in enumerate(ticks):
        current_equity = portfolio.equity(tick.price)
        portfolio.equity_history.append(current_equity)

        max_pos_value = current_equity * risk_mgr.config.max_position_pct
        pos_ratio = 0.0
        if max_pos_value > 0 and tick.price > 0:
            pos_ratio = (portfolio.position * tick.price) / max_pos_value

        state = feature_engine.update(tick, pos_ratio, portfolio.unrealized_pnl_pct(tick.price))

        if i < warmup:
            continue

        raw_action = agent.select_action(state)
        filtered_action, size = risk_mgr.filter_action(raw_action, current_equity, tick.price)

        if abs(size) > 1e-10:
            target_pos = size
            delta = target_pos - portfolio.position
            if abs(delta) > 1e-10:
                fill = exec_engine.execute(tick.timestamp, tick.price, delta, tick.volume)
                if fill:
                    realized_pnl = portfolio.apply_fill(fill)
                    if abs(realized_pnl) > 1e-10:
                        risk_mgr.record_trade(realized_pnl)

    agent.epsilon = old_epsilon

    final_equity = portfolio.equity(ticks[-1].price)
    total_return = (final_equity - initial_cash) / initial_cash
    peak = initial_cash
    max_dd = 0.0
    for eq in portfolio.equity_history:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    return {
        "strategy": "RL Agent (trained)",
        "final_equity": final_equity,
        "return_pct": total_return * 100,
        "max_drawdown_pct": max_dd * 100,
        "num_trades": portfolio.total_trades,
        "total_fees": portfolio.total_fees,
        "ticks": ticks,  # for baselines to use
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  Day 075: Full Autonomous Trading Agent")
    print("=" * 72)

    # --- Phase 1: Train the agent ---
    print("\n--- Phase 1: Training the RL Agent ---\n")
    agent, training_results = train_agent(num_episodes=15, ticks_per_episode=2000)

    # Show training progression
    print("\n--- Training Summary ---")
    first_3 = training_results[:3]
    last_3 = training_results[-3:]
    avg_first = sum(r["return_pct"] for r in first_3) / len(first_3)
    avg_last = sum(r["return_pct"] for r in last_3) / len(last_3)
    print(f"  Avg return (first 3 eps): {avg_first:+.2f}%")
    print(f"  Avg return (last 3 eps):  {avg_last:+.2f}%")
    print(f"  Final epsilon:            {training_results[-1]['epsilon']:.4f}")

    # --- Phase 2: Evaluate on unseen market ---
    print("\n--- Phase 2: Evaluation on Unseen Market ---\n")
    eval_result = evaluate_agent(agent, seed=999)
    ticks = eval_result.pop("ticks")

    # Run baselines on the same market
    bh_result = run_buy_and_hold(ticks)
    ma_result = run_ma_crossover(ticks)
    rand_result = run_random_trader(ticks)

    # --- Phase 3: Comparison ---
    print("--- Strategy Comparison (Evaluation Market) ---\n")
    strategies = [eval_result, bh_result, ma_result, rand_result]

    header = f"  {'Strategy':<22} {'Return':>10} {'MaxDD':>10} {'Trades':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in strategies:
        print(f"  {s['strategy']:<22} {s['return_pct']:>+9.2f}% {s['max_drawdown_pct']:>9.2f}% {s['num_trades']:>8}")

    # --- Phase 4: Risk management in action ---
    print("\n--- Risk Management Analysis ---\n")
    print(f"  RL Agent fees paid:       ${eval_result.get('total_fees', 0):.2f}")
    print(f"  RL Agent max drawdown:    {eval_result['max_drawdown_pct']:.2f}%")

    # Show Kelly criterion example
    print("\n--- Kelly Criterion Example ---")
    rm = RiskManager(RiskConfig(fractional_kelly=0.25))
    # Simulate a trader with 55% win rate, 1.2:1 win/loss ratio
    for _ in range(30):
        rm.record_trade(random.choice([1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
                                        1.1, 1.3, 1.5, 1.0, 1.2, -1.0,
                                        -1.0, -1.0, -1.0, -1.0, -0.8,
                                        -0.9, -1.1, -1.2]))
    kelly = rm.kelly_fraction()
    print(f"  Win rate ~60%, avg win/loss ~1.2:1")
    print(f"  Full Kelly fraction:      {kelly / rm.config.fractional_kelly:.4f}")
    print(f"  Quarter-Kelly fraction:   {kelly:.4f}")
    print(f"  On $10,000 portfolio:     ${kelly * 10000:.2f} per trade")

    # --- Phase 5: Architecture overview ---
    print("\n--- Architecture ---\n")
    print("  Market Simulator ──> Feature Engine ──> RL Agent (Q-Network)")
    print("       │                                       │")
    print("       │                                  [raw action]")
    print("       │                                       │")
    print("       │                                  Risk Manager")
    print("       │                              (Kelly + drawdown)")
    print("       │                                       │")
    print("       │                              [filtered action]")
    print("       │                                       │")
    print("       └──────────────────────────> Execution Engine")
    print("                                    (slippage + fees)")
    print("                                         │")
    print("                                    Portfolio Update")
    print("                                         │")
    print("                                    Reward ──> Agent.learn()")

    print("\n" + "=" * 72)
    print("  Training and evaluation complete!")
    print("=" * 72)


if __name__ == "__main__":
    main()
