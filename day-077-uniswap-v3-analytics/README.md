# Day 77: Uniswap V3 Pool Analytics

## Overview

Build a complete analytics engine for Uniswap V3 concentrated liquidity pools. Unlike V2's uniform liquidity distribution, V3 lets liquidity providers (LPs) concentrate capital within specific price ranges — a fundamental shift that creates rich analytical opportunities and complex position math.

This matters in the real world because concentrated liquidity is now the dominant DEX design pattern. Understanding V3 pool mechanics is essential for anyone building DeFi analytics dashboards, LP strategy tools, MEV bots, or on-chain risk systems.

## Core Concepts

### Concentrated Liquidity: The Key Innovation

In Uniswap V2, liquidity is spread uniformly across the entire price curve from 0 to ∞. This means most capital sits idle — if ETH/USDC trades at $3,000, liquidity at $0.01 or $100,000 is wasted.

V3 fixes this by letting LPs choose a price range `[p_a, p_b]` for their liquidity. Within that range, their capital behaves as if it were amplified:

```
Capital Efficiency = full_range_liquidity / concentrated_liquidity
                   = (p_b - p_a) / (√p_b - √p_a)²  (simplified)
```

A position concentrated in a ±1% range around the current price provides ~200x the capital efficiency of a V2 position.

### Ticks: Discretizing the Price Space

V3 doesn't allow arbitrary price ranges — it discretizes price space into **ticks**. Each tick `i` corresponds to a price:

```
p(i) = 1.0001^i
```

This means each tick represents a 0.01% (1 basis point) price change. Key relationships:

```
tick_from_price(p) = floor(log(p) / log(1.0001))
price_from_tick(i) = 1.0001^i
```

Ticks are grouped into **tick spacings** that depend on the pool's fee tier:
- 0.01% fee → tick spacing 1
- 0.05% fee → tick spacing 10
- 0.30% fee → tick spacing 60
- 1.00% fee → tick spacing 200

Positions can only start/end at initialized tick boundaries (multiples of tick spacing).

### The √P Representation

V3 internally tracks `√P` (square root of price) rather than price directly. This is a brilliant design choice because the core swap math becomes linear in `√P`:

```
For token0 → token1 (price decreasing):
  Δtoken0 = L × (1/√P_new - 1/√P_old)
  Δtoken1 = L × (√P_new - √P_old)

For token1 → token0 (price increasing):
  Same formulas, opposite signs
```

Where `L` is the liquidity — the geometric mean of the virtual reserves:
```
L = √(x × y)    (in the active range)
```

### Liquidity Math for Positions

When an LP adds liquidity in range `[p_a, p_b]` with amounts `x` (token0) and `y` (token1):

**Case 1: Current price below range** (p < p_a) — position is entirely token0:
```
L = x × (√p_a × √p_b) / (√p_b - √p_a)
```

**Case 2: Current price above range** (p > p_b) — position is entirely token1:
```
L = y / (√p_b - √p_a)
```

**Case 3: Current price in range** (p_a ≤ p ≤ p_b):
```
L = min(
    x × (√p × √p_b) / (√p_b - √p),
    y / (√p - √p_a)
)
```

### Fee Accumulation

Fees in V3 are tracked per-unit-of-liquidity using global fee growth accumulators:

```
feeGrowthGlobal0 += fee0 / L_active    (for token0 fees)
feeGrowthGlobal1 += fee1 / L_active    (for token1 fees)
```

Each tick stores `feeGrowthOutside` — the fee growth on the "other side" of that tick. To compute fees earned by a position in range `[tick_lower, tick_upper]`:

```
feeGrowthInside = feeGrowthGlobal - feeGrowthBelow(tick_lower) - feeGrowthAbove(tick_upper)

fees_earned = L_position × (feeGrowthInside_current - feeGrowthInside_at_last_collection)
```

### Impermanent Loss in V3

IL is amplified in V3 because capital is concentrated. For a position in range `[p_a, p_b]` when price moves from `p_0` to `p_1`:

```
IL_v3 = value_hold / value_lp - 1

Where:
  value_lp uses the concentrated position formulas
  value_hold is simply holding the original token amounts
```

The narrower the range, the higher the IL for the same price move — but also the higher the fee income. This is the fundamental LP tradeoff in V3.

## Step-by-Step Breakdown

### Step 1: Tick and Price Utilities
Build conversion functions between ticks, prices, and √P values. These are the atomic building blocks — every other calculation depends on them. Without precise tick math, position values will be wrong.

### Step 2: Pool State Model
Model a V3 pool's state: current tick, √P, liquidity, fee growth accumulators, and initialized ticks with their liquidity deltas. This represents the on-chain state you'd read from a real pool contract.

### Step 3: Position Modeling
Represent LP positions with their tick ranges and liquidity amounts. Calculate the token amounts (reserves) held by each position given the current price. This is essential for portfolio tracking.

### Step 4: Swap Simulation
Simulate swaps through the pool — this is where the tick-crossing logic lives. As a swap moves price through ticks, active liquidity changes. Getting this right means understanding how V3's "virtual AMM" slides between tick boundaries.

### Step 5: Fee Analytics
Track fee accumulation across swaps and compute per-position fee earnings. This requires the feeGrowthInside accounting described above.

### Step 6: Impermanent Loss Calculator
Compare LP position value vs. holding to quantify IL. Show how range width affects IL magnitude.

### Step 7: Pool Analytics Dashboard
Combine everything into a comprehensive analytics output: liquidity distribution across ticks, capital efficiency metrics, fee APR estimates, and position P&L.

## Learning Objectives

- Understand concentrated liquidity math and why V3 uses √P internally
- Master tick-based price discretization and tick spacing mechanics
- Implement swap simulation with tick-crossing and liquidity transitions
- Calculate LP position values, fee earnings, and impermanent loss
- Build analytical tools for real DeFi pool data

## Going Deeper

- **Real data integration**: Connect to an Ethereum node or The Graph to pull actual V3 pool state and validate your math against on-chain values
- **Optimal range strategies**: Given historical volatility, what tick range maximizes fee income minus IL? This is an active research area
- **Just-in-time (JIT) liquidity**: MEV searchers add concentrated liquidity right before large swaps and remove it after — your swap simulator can model this
- **Multi-pool routing**: V3 has multiple fee tiers for the same pair; optimal routing splits swaps across pools
- **Liquidity mining incentives**: Many protocols incentivize V3 positions — analytics need to factor in reward APR alongside fee APR
- **Builds on**: Day 51 (AMM Constant Product), Day 72 (On-chain Data Analysis), Day 73 (MEV Detection)
