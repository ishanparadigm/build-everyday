# Day 019: Autonomous Trading Bot Skeleton

## Overview

Build an autonomous trading bot framework that integrates **AI signal generation**, **crypto market simulation**, and **control-system risk management** into a single cohesive architecture. This isn't about making money — it's about understanding the software architecture behind autonomous decision-making systems that operate in adversarial, real-time environments.

Real trading bots at firms like Jump Crypto or Wintermute share a common architecture: a **signal pipeline** (ML models generating trade ideas), a **risk engine** (enforcing position limits, drawdown stops, and exposure constraints), and an **execution layer** (translating signals into orders against an exchange). The risk engine is the most critical component — it's the PID controller of finance, constantly measuring deviation from acceptable parameters and applying corrective action.

This challenge ties together concepts from previous days:
- **AI track**: Using prediction models (Days 1, 3, 12, 15) to generate trading signals
- **Crypto track**: Understanding transaction structures (Day 2, 13) and market mechanics
- **Robotics track**: Applying control theory (Day 6, 7) to risk management — treating portfolio state like a physical system that must be kept within bounds

## Core Concepts

### 1. Event-Driven Architecture

Trading bots are fundamentally **reactive systems**. Rather than running in a loop polling for data, professional bots use an event-driven architecture where market data events trigger a cascade of computations:

```
MarketData event -> Signal update -> Risk check -> Order decision -> Execution
```

This is the **observer pattern** applied to finance. Each component subscribes to events it cares about and emits events for downstream consumers. The key insight is **separation of concerns**: the signal generator doesn't know about risk limits, the risk engine doesn't know about ML models, and the executor doesn't know about either.

Why event-driven? Because markets generate data at irregular intervals (trades happen when they happen, not on a fixed clock), and you need to process each event with minimal latency. A polling architecture wastes time waiting and risks missing data between polls.

### 2. Signal Generation with Moving Average Crossover

The simplest systematic trading signal is the **moving average crossover**. Given a price series P(t):

- **Fast MA** (short window, e.g., 10 periods): SMA_fast = (1/k) * sum(P(t-i) for i in 0..k-1)
- **Slow MA** (long window, e.g., 30 periods): SMA_slow = (1/n) * sum(P(t-i) for i in 0..n-1)

**Signal logic**:
- When SMA_fast crosses above SMA_slow -> **BUY signal** (short-term momentum is bullish)
- When SMA_fast crosses below SMA_slow -> **SELL signal** (short-term momentum is bearish)

The intuition: a fast MA reacts quickly to recent price changes while a slow MA smooths out noise. When the fast MA rises above the slow MA, it means recent prices are higher than the longer-term average — momentum is shifting upward.

**Why this works (sometimes)**: Prices exhibit **serial correlation** (momentum) over short horizons. A crossover detects regime changes. Why it fails: in sideways/choppy markets, you get **whipsawed** — repeatedly buying high and selling low as the MAs oscillate around each other.

We also implement an **RSI (Relative Strength Index)** filter:

RSI = 100 - (100 / (1 + RS))   where RS = avg_gain / avg_loss over N periods

RSI ranges from 0-100. Above 70 = overbought (don't buy), below 30 = oversold (don't sell). This acts as a **confirmation filter** to reduce whipsaw trades.

### 3. Risk Management as a Control System

This is where robotics meets finance. Think of your portfolio as a **physical system**:
- **State**: current positions, unrealized P&L, cash balance
- **Setpoint**: target risk levels (max position size, max drawdown, max exposure)
- **Control output**: position adjustments (reduce size, close positions, halt trading)

The risk engine operates like a **governor** (the mechanical device that prevents engines from spinning too fast). It has hard limits that trigger immediate action:

- **Max position size**: No single position > X% of portfolio (prevents concentration risk)
- **Max drawdown**: If portfolio drops > Y% from peak, halt all trading (circuit breaker)
- **Daily loss limit**: If daily P&L < -Z, stop opening new positions
- **Exposure limit**: Total absolute position value < W% of portfolio

These are **safety constraints**, analogous to joint limits on a robot arm (Day 8). Violating them doesn't just lose money — it can be catastrophic (see: LTCM, Knight Capital).

### 4. Order Book Simulation

Real exchanges use a **limit order book** (LOB): a sorted list of buy orders (bids) and sell orders (asks). The **spread** is the gap between the best bid and best ask. When you place a market order, you **cross the spread** — buying at the ask price or selling at the bid price.

We simulate this with a simplified model:
- **Mid price**: the "true" price, following a random walk with drift
- **Spread**: the cost of immediacy, modeled as a function of volatility
- **Slippage**: larger orders move the price against you (market impact)
- **Latency**: orders take time to reach the exchange, during which the price may move

### 5. Performance Metrics

How do you know if a strategy is any good? Key metrics:

- **Sharpe Ratio** = mean(returns) / std(returns) * sqrt(252). Risk-adjusted return. Above 1.0 is decent, above 2.0 is very good. The sqrt(252) annualizes daily returns.
- **Max Drawdown** = max peak-to-trough decline. Measures worst-case pain.
- **Win Rate** = profitable trades / total trades. Above 50% isn't necessary if winners are bigger than losers.
- **Profit Factor** = gross_profit / gross_loss. Above 1.0 means profitable overall.

## Step-by-Step Breakdown

1. **Build the event system**: Create an EventBus that allows components to publish and subscribe to typed events. This decouples all components.

2. **Implement market simulation**: Generate realistic price data with configurable volatility, trend, and noise. Emit MarketData events on each tick.

3. **Build the signal generator**: Consume MarketData events, maintain rolling windows for MAs and RSI, emit Signal events when crossover conditions are met.

4. **Implement the risk engine**: Track portfolio state, enforce position limits, drawdown stops, and daily loss limits. Can veto or modify orders before execution.

5. **Build the execution engine**: Receive vetted orders, simulate fills with spread and slippage, update portfolio state, emit Fill events.

6. **Create the portfolio tracker**: Maintain positions, cash, P&L, and compute performance metrics. Listen for Fill events.

7. **Wire it all together**: Connect components via the EventBus, run a simulation, and analyze results.

## Learning Objectives

- Design event-driven architectures with loose coupling between components
- Implement moving average crossover and RSI trading signals from first principles
- Build a risk management system that enforces portfolio constraints (analogous to control systems)
- Simulate order execution with realistic market microstructure (spread, slippage, latency)
- Compute and interpret standard trading performance metrics (Sharpe, drawdown, win rate)
- Understand how AI, crypto, and control theory concepts integrate in autonomous trading systems

## Going Deeper

- **Market microstructure**: Real order books have queue priority, hidden orders, and complex matching rules. Look into Cont, Stoikov & Talreja (2010) for LOB modeling.
- **Adaptive signals**: Replace fixed MAs with Kalman filters (Day upcoming) for adaptive smoothing.
- **Kelly criterion**: Optimal position sizing based on edge and variance: f* = (bp - q) / b where b is payoff odds, p is win probability.
- **Execution algorithms**: TWAP, VWAP, and implementation shortfall algorithms minimize market impact for large orders.
- **Backtesting pitfalls**: Lookahead bias, survivorship bias, overfitting to historical data. Walk-forward optimization addresses some of these.
- **Production concerns**: Exchange API rate limits, websocket reconnection, order state reconciliation, and the critical importance of kill switches.
