# Day 075: Full Autonomous Trading Agent

## Overview

Build a complete autonomous trading agent that combines **reinforcement learning** for strategy optimization, **on-chain data analysis** for market intelligence, and **risk management** with position sizing — all tied together in an event-driven architecture. This is the culmination of concepts from ML (days 57-58), crypto analytics (days 72-73), and the trading bot skeleton (day 19).

In production, autonomous trading agents operate in adversarial environments where milliseconds matter, slippage eats profits, and a single miscalibrated parameter can blow up an account. This challenge teaches you to reason about all three simultaneously: *when* to trade (RL policy), *what* the market is doing (on-chain signals), and *how much* to risk (Kelly criterion + drawdown limits).

## Core Concepts

### 1. Reinforcement Learning for Trading Decisions

A trading agent maps **states** (market features) to **actions** (buy/sell/hold) to maximize cumulative reward (PnL). We use Q-learning (day 57) with function approximation:

- **State space**: A feature vector encoding price momentum, volatility, on-chain metrics, and current position. Each feature is normalized to [-1, 1] for stable learning.
- **Action space**: Discrete actions {STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL} with position sizing determined by action intensity.
- **Reward function**: Risk-adjusted returns using the Sharpe-like ratio:

  ```
  reward = (portfolio_return - risk_free_rate) / max(volatility, epsilon)
  ```

  This trains the agent to prefer *consistent* returns over lucky large bets. We add a penalty for drawdowns to discourage reckless behavior.

- **Exploration vs exploitation**: Epsilon-greedy with decay, starting at high exploration (epsilon=1.0) and annealing to a floor (epsilon=0.05). Early episodes explore randomly; later episodes exploit learned Q-values.

### 2. On-Chain Signal Generation

On-chain data provides signals invisible to price-only analysis:

- **Volume momentum**: Rolling z-score of trading volume. Unusually high volume often precedes price moves.
- **Large transaction detection**: When whale-sized transfers spike, it signals institutional repositioning.
- **Network activity**: Active address counts and transaction frequency as proxies for organic demand vs wash trading.
- **DEX flow imbalance**: Net buy/sell pressure on decentralized exchanges, computed as (buy_volume - sell_volume) / total_volume.

Each signal is normalized and combined into a composite on-chain score in [-1, 1], where positive = bullish, negative = bearish.

### 3. Kelly Criterion for Position Sizing

The Kelly criterion answers: "Given my edge and risk, what fraction of capital should I bet?"

```
f* = (p * b - q) / b

where:
  p = probability of winning (estimated from recent win rate)
  b = win/loss ratio (average win / average loss)
  q = 1 - p (probability of losing)
```

Full Kelly is too aggressive for real trading (a 50% drawdown requires a 100% gain to recover). We use **fractional Kelly** (typically 0.25-0.5x) to trade off growth rate for reduced variance.

### 4. Risk Management: The Non-Negotiable Layer

No signal is worth a blown account. The risk manager enforces hard constraints:

- **Maximum position size**: Never exceed X% of portfolio in a single position.
- **Daily loss limit**: Stop trading if cumulative daily loss exceeds a threshold.
- **Maximum drawdown**: Reduce position sizes as drawdown deepens (linear scaling from max size at 0% drawdown to zero at max allowed drawdown).
- **Cooldown after losses**: After N consecutive losses, force a hold period to avoid tilt-driven revenge trading.

These rules override the RL policy. The agent proposes; the risk manager disposes.

### 5. Event-Driven Architecture

The agent runs on a simple event loop:

```
while market_open:
    event = next_event()  # price tick, on-chain update, fill, etc.
    state = update_state(event)
    action = policy.decide(state)
    sized_action = risk_manager.filter(action, portfolio)
    execute(sized_action)
    reward = compute_reward(portfolio)
    policy.learn(state, action, reward, new_state)
```

Events are processed in timestamp order. The separation of concerns (signal -> decision -> risk filter -> execution) makes each component independently testable.

## Step-by-Step Breakdown

### Step 1: Market Simulator
Build a synthetic market that generates realistic price series with regime changes (trending, mean-reverting, volatile). This is our training environment — the agent never sees real exchange APIs, but the simulator captures the statistical properties that matter.

### Step 2: On-Chain Signal Generator
Create a signal generator that produces correlated on-chain metrics alongside price data. Volume leads price, whale movements precede volatility, and network activity correlates with trend strength.

### Step 3: Feature Engineering
Transform raw market + on-chain data into a normalized state vector. Include momentum (rate of change), volatility (rolling std), on-chain composite score, and current position/PnL as features.

### Step 4: Q-Learning Agent
Implement a Q-learning agent with a simple neural network (2-layer MLP) for Q-value approximation. The agent takes the state vector and outputs Q-values for each action.

### Step 5: Risk Manager
Build the risk management layer with Kelly sizing, drawdown limits, daily loss limits, and cooldown logic. This wraps the agent's raw decisions.

### Step 6: Execution Engine
Simulate order execution with realistic slippage (proportional to order size and inversely proportional to volume) and trading fees.

### Step 7: Training Loop
Train the agent over multiple episodes (market simulations), tracking cumulative PnL, Sharpe ratio, max drawdown, and win rate.

### Step 8: Evaluation
Compare the trained agent against baselines: buy-and-hold, random trading, and a simple moving average crossover strategy.

## Learning Objectives

- Integrate RL, on-chain analytics, and risk management into a cohesive system
- Implement Kelly criterion position sizing with fractional scaling
- Build event-driven trading architecture with clean separation of concerns
- Design reward functions that optimize for risk-adjusted returns, not raw PnL
- Understand why risk management must be a hard constraint layer, not a soft suggestion

## Going Deeper

- **Multi-asset portfolios**: Extend to trade multiple correlated assets with portfolio-level risk constraints (VaR, correlation-aware sizing).
- **Market microstructure**: Add order book simulation with bid-ask dynamics, queue position, and market impact models.
- **Regime detection**: Use Hidden Markov Models to detect market regime changes and adapt strategy parameters accordingly.
- **MEV awareness**: Integrate MEV detection (day 73) to avoid periods of high sandwich attack risk.
- **Live deployment**: Replace the simulator with exchange API adapters. The architecture is designed so only the execution layer changes.
- **Meta-learning**: Train across multiple market conditions so the agent generalizes rather than overfitting to one regime.
