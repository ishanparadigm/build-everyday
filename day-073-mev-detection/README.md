# Day 073: MEV Detection Script

## What You're Building

A **Maximal Extractable Value (MEV) detection engine** that analyzes Ethereum transactions to identify common MEV strategies: sandwich attacks, arbitrage, and liquidations. MEV is one of the most important (and controversial) phenomena in blockchain — it represents the profit that block producers and searchers can extract by reordering, inserting, or censoring transactions within a block.

Understanding MEV matters because it directly affects every DeFi user. When you swap tokens on Uniswap, a searcher might sandwich your trade — buying before you and selling after — extracting value from your slippage tolerance. MEV extraction totals billions of dollars annually and fundamentally shapes blockchain protocol design (it's why Flashbots exists, why PBS was introduced in Ethereum, and why MEV-aware transaction ordering is an active research area).

## Core Concepts

### What is MEV?

MEV (originally "Miner Extractable Value," now "Maximal Extractable Value" post-merge) is the maximum value that can be extracted from block production beyond the standard block reward and gas fees, by including, excluding, or reordering transactions.

The key insight: **transaction ordering matters**. In traditional finance, orders are processed FIFO. In blockchain, the block producer chooses the order. This creates an auction for transaction positioning.

### Sandwich Attacks

A sandwich attack exploits a pending swap transaction in the mempool:

1. **Front-run**: Attacker sees victim's swap (e.g., buy ETH with USDC). Attacker buys ETH first, pushing the price up.
2. **Victim's trade executes**: Victim buys ETH at a now-higher price (within their slippage tolerance).
3. **Back-run**: Attacker sells ETH at the inflated price, pocketing the difference.

**Detection pattern**: Three transactions in sequence where:
- Tx1 and Tx3 are from the same address (the attacker)
- Tx2 is from a different address (the victim)
- All interact with the same DEX pool
- Tx1 buys token X, Tx2 buys token X (pushing price higher), Tx3 sells token X
- The attacker profits: `sell_amount - buy_amount - gas_costs > 0`

**Math**: If the victim's trade moves the price by `delta_p`, and the attacker front-runs with amount `A`:
- Attacker buys at price `p`, moves price to `p + delta_a`
- Victim buys at `p + delta_a`, moves to `p + delta_a + delta_v`
- Attacker sells at `p + delta_a + delta_v`
- Profit ~= `A * delta_v / p` (minus gas)

### Arbitrage

Arbitrage exploits price differences for the same asset across different venues:

- Token X is $100 on Uniswap and $102 on SushiSwap
- Buy on Uniswap, sell on SushiSwap, profit $2 minus gas
- Often done atomically in a single transaction using flash loans

**Detection pattern**: A single transaction that:
- Interacts with 2+ DEX pools for the same token pair
- Starts and ends with the same token (circular path)
- Results in a net positive balance for the sender

### Liquidations

In lending protocols (Aave, Compound), borrowers must maintain a collateral ratio. When the ratio drops below the liquidation threshold, anyone can repay part of the debt and receive collateral at a discount.

**Detection pattern**: A transaction that:
- Calls a lending protocol's `liquidate` function
- Receives collateral tokens at below-market price
- Often preceded by price oracle updates that trigger the liquidation

### Gas Price Analysis

MEV transactions typically use elevated gas prices (priority fees) to ensure favorable ordering. Analyzing gas price distributions within a block reveals MEV activity — transactions paying significantly above the block's median gas price are likely MEV-related.

## Step-by-Step Breakdown

1. **Model the transaction and block structure**: Define data classes for transactions, DEX swaps, and blocks. Each transaction needs: sender, receiver, value, gas price, input data (decoded), and position in block.

2. **Build a DEX swap decoder**: Parse transaction data to identify swap events — which tokens were traded, amounts in/out, which pool, and the effective price. This is the foundation for all detection.

3. **Implement sandwich detection**: Scan blocks for the three-transaction pattern. Group swaps by pool, look for same-sender bookends around a victim trade. Calculate attacker profit.

4. **Implement arbitrage detection**: Look for circular swap paths within a single transaction (or tightly grouped transactions). Track token flows to verify net-positive outcome.

5. **Implement liquidation detection**: Identify calls to known liquidation functions, calculate the liquidation bonus (discount on collateral).

6. **Build the analysis engine**: Aggregate results across blocks, compute statistics (total MEV extracted, frequency of each type, gas premium analysis).

7. **Generate reports**: Output structured analysis with per-block and aggregate MEV metrics.

## Learning Objectives

- Understand MEV taxonomy: sandwich attacks, arbitrage, liquidations, and JIT liquidity
- Learn transaction ordering economics and why mempool visibility creates extraction opportunities
- Implement pattern-matching algorithms for detecting structured transaction sequences
- Analyze gas price dynamics as MEV signals
- Understand the game theory between searchers, builders, and validators

## Going Deeper

- **Flashbots & MEV-Share**: How the MEV supply chain is being restructured to return value to users
- **PBS (Proposer-Builder Separation)**: Ethereum's architectural response to MEV centralization
- **Cross-domain MEV**: Extraction across L1/L2 boundaries and across chains
- **Order Flow Auctions (OFAs)**: MEV-aware transaction routing (e.g., MEV Blocker, CoW Protocol)
- **CEX-DEX arbitrage**: The dominant form of MEV today, linking on-chain and off-chain liquidity
- **Time-weighted vs. spot pricing**: How TWAP oracles reduce MEV in liquidations
