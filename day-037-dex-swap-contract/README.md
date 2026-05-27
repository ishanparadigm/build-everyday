# Day 037: DEX Swap Contract — Building an Automated Market Maker

## Overview

You're building a **Decentralized Exchange (DEX) swap contract** — the core mechanism behind protocols like Uniswap that enable trustless token trading without order books. Instead of matching buyers and sellers, an Automated Market Maker (AMM) uses a mathematical formula and liquidity pools to determine prices algorithmically.

**Why this matters:** AMMs process billions of dollars in daily volume. Understanding how they work — from the constant product formula to slippage mechanics — is essential for anyone building in DeFi. This is the engine that powers decentralized finance.

## Core Concepts

### The Constant Product Formula: x * y = k

Traditional exchanges use order books: buyers post bids, sellers post asks, and the exchange matches them. AMMs replace this entirely with a simple invariant:

```
x * y = k
```

Where:
- `x` = reserve of token A in the pool
- `y` = reserve of token B in the pool  
- `k` = a constant that must be maintained after every trade

**The intuition:** Imagine a pool holding 1000 ETH and 2,000,000 USDC. k = 2,000,000,000. If someone buys ETH (removing it from the pool), the ETH reserve decreases. To maintain k, the USDC reserve must increase — meaning the buyer deposits USDC. The *ratio* of reserves determines the price.

**The math of a swap:**
When a user wants to swap `dx` of token A for token B:
```
(x + dx) * (y - dy) = k
dy = y - k / (x + dx)
dy = y * dx / (x + dx)       # simplified
```

This means the output amount `dy` depends on both the input amount and the current reserves. Larger trades relative to pool size get worse prices — this IS slippage, emerging naturally from the math.

### Price Impact and Slippage

The **spot price** of token B in terms of token A is simply `x / y`. But this is only the price for an infinitesimally small trade. Any real trade moves the reserves and thus moves the price.

**Price impact** = how much YOUR trade moves the price:
```
price_impact = dx / (x + dx)
```

A trade of 1% of the reserve has ~1% price impact. A trade of 10% has ~10%. This is the AMM's built-in protection against draining the pool.

### Liquidity Provision and LP Tokens

Liquidity providers deposit both tokens in the current ratio and receive LP tokens representing their share of the pool. When they withdraw, they get back their proportional share — which may differ from what they deposited due to trades shifting the ratio.

**LP token math:**
```
lp_minted = total_lp_supply * min(dx/x, dy/y)     # for existing pools
lp_minted = sqrt(dx * dy)                          # for initial deposit (geometric mean)
```

The geometric mean for initial deposits ensures the LP token value is independent of the arbitrary initial ratio.

### Impermanent Loss

If the external price of token A doubles relative to token B, arbitrageurs will buy the cheap A from the pool until the pool price matches the market. The LP now holds less A and more B than if they'd just held. This "loss" vs holding is called **impermanent loss** because it reverses if the price returns to the original ratio.

```
IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
```

At 2x price change: IL ≈ -5.7%. At 5x: IL ≈ -25.5%.

### Trading Fees

Real AMMs charge a fee (typically 0.3%) on each swap. The fee is added to reserves, growing k over time and compensating LPs for impermanent loss:

```
dx_after_fee = dx * (1 - fee_rate)
dy = y * dx_after_fee / (x + dx_after_fee)
```

The fee accumulates inside the pool, automatically increasing each LP token's claim on the reserves.

## Step-by-Step Breakdown

1. **Token representation**: Model two ERC-20-like tokens with balances and transfer mechanics. We simulate this in Python with dictionaries tracking balances.

2. **Pool initialization**: Create the liquidity pool with initial deposits of both tokens. Mint LP tokens using the geometric mean formula. This first deposit sets the initial price ratio.

3. **Adding liquidity**: Subsequent deposits must match the current reserve ratio. Calculate LP tokens to mint using the minimum ratio to prevent manipulation. Any excess tokens are returned.

4. **Swap execution**: Implement the constant product swap with fees. Validate the invariant holds after every operation. Calculate output amounts, update reserves, and transfer tokens.

5. **Removing liquidity**: Burn LP tokens and return proportional reserves. The LP gets back the current ratio, not their original deposit ratio — this is where impermanent loss materializes.

6. **Price oracle**: Track cumulative prices over time for time-weighted average price (TWAP) calculations — critical for other protocols that need manipulation-resistant price feeds.

7. **Slippage protection**: Implement minimum output amounts so users don't get worse-than-expected prices from front-running or concurrent trades.

## Learning Objectives

- Implement the constant product AMM formula and understand its mathematical properties
- Build liquidity provision/withdrawal with LP token accounting
- Calculate and visualize price impact, slippage, and impermanent loss
- Implement trading fees and understand how they accrue to LPs
- Add slippage protection (minimum output amounts)
- Understand the economic incentives that keep AMMs functioning

## Going Deeper

- **Concentrated liquidity** (Uniswap V3): Instead of spreading liquidity across all prices, LPs choose a range. This is dramatically more capital efficient but introduces complex position management.
- **Multi-token pools**: Balancer extends AMMs to weighted pools with 2-8 tokens using a generalized invariant.
- **Stableswap invariant** (Curve): For tokens that should be close in price (USDC/USDT), a modified curve that's flatter near 1:1 reduces slippage dramatically.
- **MEV and sandwich attacks**: Front-runners can detect pending swaps and sandwich them — buying before and selling after to extract value. Understanding this is critical for production AMMs.
- **Flash swaps**: Borrow tokens from the pool, use them, and return them (plus fee) in a single transaction — enabling arbitrage without capital.
