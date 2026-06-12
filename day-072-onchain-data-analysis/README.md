# Day 072: On-Chain Data Analysis with Python

## Overview

Every transaction on a public blockchain is a permanent, timestamped record of economic activity. On-chain data analysis is the practice of extracting, transforming, and interpreting this data to understand protocol health, user behavior, token flows, and market dynamics. It's how analysts detect wash trading, track whale movements, measure DeFi TVL, and identify MEV extraction — all without relying on centralized data providers.

In this challenge, you'll build an on-chain analytics engine from scratch that processes raw blockchain transaction data to produce actionable insights: wallet profiling, token flow analysis, network graph construction, and anomaly detection.

## Core Concepts

### 1. Transaction Data Model

Every Ethereum-style transaction contains:
- **from/to addresses**: The sender and recipient (or contract being called)
- **value**: Native token transferred (in wei, where 1 ETH = 10^18 wei)
- **gas used / gas price**: Computational cost and price paid per unit
- **block number / timestamp**: When the transaction was included
- **input data**: For contract calls, the encoded function signature and arguments

The key insight: a single "user action" (like a Uniswap swap) might involve multiple internal transactions and token transfers. Raw transaction data is a low-level log — analysis requires reconstructing higher-level economic events.

### 2. Address Profiling and Clustering

Not all addresses behave the same. By analyzing transaction patterns, you can classify addresses:
- **EOAs (Externally Owned Accounts)**: Human wallets with irregular activity patterns
- **Contracts**: Deterministic behavior, often receiving many calls
- **Exchange hot wallets**: High volume, many unique counterparties
- **Whale wallets**: Large balances, infrequent but high-value transactions
- **Bot/MEV addresses**: High frequency, gas price bidding patterns

Metrics that distinguish these:
- **Transaction frequency**: txns per time window
- **Unique counterparties**: how many distinct addresses interact with this one
- **Average value**: typical transaction size
- **Gas price behavior**: do they bid aggressively (MEV) or use standard gas?
- **In/out ratio**: net flow direction indicates accumulation vs distribution

### 3. Token Flow Analysis (Graph-Based)

Token movements form a directed graph where:
- Nodes = addresses
- Edges = token transfers (weighted by amount)

Key graph metrics:
- **In-degree / out-degree**: Number of unique senders/receivers
- **PageRank**: Identifies "important" addresses in the flow network
- **Connected components**: Groups of addresses that interact with each other
- **Flow concentration**: What fraction of total volume goes through the top N addresses (Gini coefficient)

This is how analysts trace funds through mixing services, identify wash trading loops, and measure protocol decentralization.

### 4. Time-Series Analytics

On-chain data has a natural time dimension (block timestamps). Key time-series analyses:
- **Volume over time**: Total value transferred per block/hour/day
- **Active addresses**: Unique addresses transacting per period
- **Gas price trends**: Network congestion indicators
- **Moving averages**: Smoothed metrics for trend detection
- **Anomaly detection**: Sudden spikes in volume or gas that deviate from the rolling mean

The standard approach: compute rolling statistics (mean, std) over a window, then flag points where the z-score exceeds a threshold. This catches flash crashes, exploit transactions, and unusual whale movements.

### 5. The Gini Coefficient

The Gini coefficient measures inequality in a distribution, ranging from 0 (perfect equality) to 1 (one entity has everything). For on-chain data:

```
Gini = (2 * Σᵢ i * xᵢ) / (n * Σᵢ xᵢ) - (n + 1) / n
```

Where x is the sorted array of values. A Gini of 0.8+ for token holdings means extreme concentration — a few whales hold most of the supply. This is a critical metric for assessing protocol decentralization and governance power distribution.

## Step-by-Step Breakdown

1. **Data generation**: Simulate realistic blockchain transaction data with multiple address types (whales, bots, regular users, contracts), varying transaction patterns, and token transfers
2. **Transaction parsing**: Build data structures to efficiently index and query transactions by address, block, time range, and value
3. **Address profiling**: Compute per-address statistics (tx count, volume, unique counterparties, avg gas) and classify addresses into behavioral categories
4. **Token flow graph**: Construct a directed weighted graph from token transfers, compute graph metrics (degree distribution, PageRank, connected components)
5. **Time-series analysis**: Aggregate metrics over time windows, compute rolling statistics, detect anomalies using z-score thresholds
6. **Wealth distribution**: Calculate Gini coefficient and Lorenz curve data for token holdings
7. **Report generation**: Produce a summary dashboard with key protocol health metrics

## Learning Objectives

- Parse and index blockchain transaction data structures efficiently
- Profile addresses by behavioral patterns using statistical features
- Build and analyze directed weighted graphs from token flow data
- Apply time-series anomaly detection to on-chain metrics
- Compute distribution inequality metrics (Gini coefficient)
- Understand how real-world on-chain analytics platforms work under the hood

## Going Deeper

- **Real data sources**: Etherscan API, The Graph subgraphs, Dune Analytics, Flipside Crypto
- **MEV detection**: Look for sandwich attacks (buy before, sell after a target tx in the same block) — see Day 073+
- **Clustering with heuristics**: Common-input-ownership for UTXO chains, deposit address reuse for exchange identification
- **Privacy protocols**: How Tornado Cash and similar mixers break the graph analysis (and how statistical analysis can sometimes still de-anonymize)
- **Production analytics**: Tools like Nansen, Arkham, and Chainalysis use these same primitives at massive scale with entity labeling databases
