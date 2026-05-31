# Day 51: Automated Market Maker - Constant Product Formula

## Overview

Build a complete Automated Market Maker (AMM) simulation implementing the constant product formula (x * y = k) used by Uniswap V2 and similar DEXes. You'll implement liquidity provision, token swaps, price impact calculation, impermanent loss tracking, and arbitrage detection — all from first principles.

**Why this matters:** AMMs are the backbone of decentralized finance. Unlike traditional order books where buyers and sellers are matched, AMMs use mathematical formulas to determine prices algorithmically. Understanding the constant product formula deeply — its elegance, its limitations, and its economic properties — is essential for anyone building in DeFi. Uniswap alone has processed over $2 trillion in cumulative volume using exactly this math.

## Core Concepts

### The Constant Product Formula: x * y = k

The fundamental invariant is deceptively simple. A liquidity pool holds reserves of two tokens (x and y). The product of these reserves must remain constant (k) after every trade.

**The math:**
- Pool state: (x, y) where x = reserve of token A, y = reserve of token B
- Invariant: x * y = k
- If a trader adds dx of token A, they receive dy of token B such that:
  - (x + dx) * (y - dy) = k
  - Solving for dy: dy = y - k/(x + dx) = y * dx / (x + dx)

**Why this formula works as a market:**
- As you buy token B (removing it from the pool), its price increases — the pool has less of it
- As you sell token A (adding it to the pool), its price decreases — the pool has more of it
- The price at any point is the ratio of reserves: price_A_in_B = y/x
- The curve is a hyperbola — it never reaches zero on either axis, meaning the pool can never be fully drained of either token

**The intuition:** Think of the pool as a see-saw. The product x*y is like the moment of inertia. When one side goes up (more of token A), the other must go down (less of token B) to keep the "balance" constant. Bigger pools (higher k) are more stable — they require larger trades to move the price significantly.

### Fees and Their Role

Real AMMs charge a fee (typically 0.3%) on each swap. The fee is kept in the pool, which means k actually *increases* slightly after each trade. This is how liquidity providers earn yield.

**With fees:**
- Effective input: dx_effective = dx * (1 - fee_rate)
- dy = y * dx_effective / (x + dx_effective)
- New k = (x + dx) * (y - dy) > old k (the fee stays in the pool)

### Price Impact and Slippage

The constant product formula creates *price impact* — larger trades get worse prices because they move along the curve more. This is fundamentally different from a limit order book.

**Price impact formula:**
- Spot price (infinitesimal trade): P = y/x
- Execution price (actual trade of size dx): P_exec = dy/dx = y/(x + dx)
- Price impact = 1 - (P_exec / P_spot) = dx/(x + dx)

Notice: price impact depends *only* on the trade size relative to the pool, not on the absolute token prices. A $100 trade in a $1M pool has the same impact as a $100K trade in a $1B pool.

### Impermanent Loss

When token prices change, liquidity providers would have been better off simply holding their tokens — this difference is called *impermanent loss*. It's "impermanent" because it reverses if prices return to the original ratio.

**The math:**
- If the price of token A relative to token B changes by a factor of r:
- IL = 2*sqrt(r) / (1 + r) - 1
- At r=2 (price doubles): IL = -5.7%
- At r=5 (price quintuples): IL = -25.5%
- IL is always negative (or zero) — the LP always underperforms pure holding

**Why it happens:** Arbitrageurs continuously trade against the pool to align its price with external markets. Every arbitrage trade extracts value from LPs. The pool always sells the appreciating asset too cheaply and buys the depreciating asset too expensively.

### Liquidity Provider (LP) Tokens

When you add liquidity, you receive LP tokens proportional to your share of the pool. These tokens represent your claim on the pool's reserves (plus accumulated fees).

**LP token math:**
- Initial deposit: LP_tokens = sqrt(x_deposited * y_deposited)  (geometric mean)
- Subsequent deposits: LP_tokens = total_supply * min(dx/x, dy/y)
- When withdrawing: you get back (your_LP / total_LP) * each reserve

The geometric mean for initial deposits is a deliberate design choice — it makes the number of LP tokens independent of the price ratio, preventing manipulation.

## Step-by-Step Breakdown

### Step 1: Pool Initialization
Create a LiquidityPool class that stores reserves and LP token balances. The pool starts empty — the first liquidity provider sets the initial price by choosing the ratio of tokens deposited.

### Step 2: Adding Liquidity
Implement proportional liquidity addition. After the first deposit, subsequent LPs must add tokens in the current ratio to avoid changing the price. Calculate and mint LP tokens.

### Step 3: Token Swaps
Implement the constant product swap with fees. Given an input amount of one token, calculate the output amount of the other token. Update reserves. Verify k has not decreased.

### Step 4: Removing Liquidity
Burn LP tokens and return proportional reserves. The LP gets back their share of both tokens — which may differ from what they originally deposited due to trades changing the ratio.

### Step 5: Price Impact Analysis
Build functions to calculate spot price, execution price, and price impact for trades of varying sizes. Show how impact scales with trade size relative to pool depth.

### Step 6: Impermanent Loss Calculator
Given an initial deposit and a price change, calculate the IL. Compare the value of LP position vs. simply holding the original tokens.

### Step 7: Arbitrage Simulation
Simulate an external price change and calculate the optimal arbitrage trade to realign the pool. Show how arbitrage drives IL.

### Step 8: Multi-swap Simulation
Run a series of random trades and track pool metrics over time: k growth from fees, LP returns, price trajectory, and cumulative impermanent loss.

## Learning Objectives

- **Constant product invariant**: Understand x*y=k deeply — why it works as a pricing mechanism, its mathematical properties, and its limitations
- **DeFi economics**: See how fees, impermanent loss, and arbitrage interact to create a self-sustaining market
- **Numerical precision**: Handle the practical challenges of financial math — rounding, minimum amounts, and overflow prevention
- **LP token mechanics**: Understand how ownership shares are tracked and how deposits/withdrawals work
- **Price impact**: Quantify the cost of trading against a finite-liquidity pool

## Going Deeper

- **Concentrated liquidity (Uniswap V3)**: Instead of spreading liquidity across the entire price curve, LPs provide liquidity in specific price ranges, dramatically improving capital efficiency. The math shifts from x*y=k to a virtual reserves model.
- **Stableswap (Curve)**: For assets that should trade near 1:1 (like USDC/USDT), the constant product formula is wasteful. Curve's StableSwap uses a hybrid formula combining constant product and constant sum (x+y=k) for lower slippage near the peg.
- **Multi-asset pools (Balancer)**: Generalize to n tokens with weighted geometric means: product(x_i^w_i) = k.
- **MEV and sandwich attacks**: Traders can observe pending swaps and sandwich them — buying before and selling after — to extract value from price impact. Understanding AMM math is key to understanding (and mitigating) MEV.
- **Oracle manipulation**: Using the spot price (y/x) as a price oracle is dangerous because it can be manipulated in a single transaction. Uniswap V2 introduced time-weighted average prices (TWAPs) to mitigate this.
