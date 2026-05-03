# Day 032: Smart Contract That Uses Oracle ML Predictions

## Overview

You're building a system where a smart contract consumes machine learning predictions delivered through an oracle pattern. This is how real-world DeFi protocols integrate off-chain intelligence: price prediction models, risk scores, fraud detectors, and sentiment classifiers all feed into on-chain logic through oracle mechanisms.

The core problem: blockchains are deterministic execution environments. They can't call external APIs, run ML models, or access off-chain data directly. Oracles bridge this gap by having trusted (or trustlessly aggregated) off-chain reporters push data on-chain. Your contract then uses that data to make autonomous decisions — approving loans, adjusting interest rates, or triggering liquidations.

We'll build:
1. An **ML prediction model** (logistic regression for credit risk scoring)
2. An **oracle contract** that accepts signed predictions from authorized reporters
3. A **lending contract** that uses oracle-delivered risk scores to approve/deny loans and set interest rates
4. A **commit-reveal scheme** to prevent front-running of oracle updates

## Core Concepts

### The Oracle Problem

Smart contracts execute in a sandboxed VM with no I/O. Every node must reach the same result given the same inputs, which means no network calls, no randomness, no timestamps beyond block time. This is called the **oracle problem**: how do you get real-world data into a deterministic system?

Solutions exist on a spectrum:
- **Centralized oracle**: A single trusted entity pushes data on-chain. Simple but single point of failure.
- **Multi-oracle consensus**: N reporters submit values; the contract takes the median. More robust but slower.
- **Commit-reveal**: Reporters first commit a hash of their value, then reveal it. Prevents front-running — if you can see the oracle update in the mempool before it's mined, you could trade ahead of it.

The math behind commit-reveal:

```
commit_hash = keccak256(value || salt || sender_address)
```

Including the sender address prevents one reporter from copying another's commit. The salt adds entropy. During the reveal phase, the contract verifies:

```
keccak256(revealed_value || revealed_salt || msg.sender) == stored_commit
```

### ML Risk Scoring for On-Chain Decisions

Credit risk scoring maps borrower features to a probability of default. We use logistic regression:

```
P(default) = sigmoid(w^T * x + b) = 1 / (1 + exp(-(w^T * x + b)))
```

Where `x` is the feature vector (credit utilization, transaction history, account age, etc.) and `w` are learned weights. The output is a probability in [0, 1].

**Why logistic regression for oracles?** Interpretability. When a loan is denied, you need to explain why. A logistic regression model lets you point to specific features and their weights. Neural networks might be more accurate, but regulators and users demand explainability.

**Quantization for on-chain use**: Solidity has no floating point. We convert probabilities to basis points (0-10000) representing 0.00%-100.00%:

```
risk_score_bps = int(P(default) * 10000)
```

This gives us 0.01% precision — more than sufficient for lending decisions.

### Interest Rate Curves

Given a risk score, the lending contract sets an interest rate. We use a piecewise linear curve:

```
if risk_score <= 2000 bps (20%):  rate = base_rate + risk_score * 0.5
if risk_score <= 5000 bps (50%):  rate = base_rate + 1000 + (risk_score - 2000) * 1.0
if risk_score > 5000 bps (50%):   DENIED — too risky
```

This creates increasing cost for riskier borrowers while hard-cutting off the riskiest. The curve parameters are governable — a DAO could vote to adjust them.

### Staleness and Liveness

Oracle data has a shelf life. A risk score computed yesterday might be dangerously stale today. The contract enforces:

```
require(block.timestamp - oracle.last_update <= MAX_STALENESS)
```

If the oracle hasn't been updated within the staleness window, all lending operations pause. This is a critical safety mechanism — Compound's oracle failure in 2022 led to $80M in bad debt because stale prices were used for liquidation calculations.

## Step-by-Step Breakdown

### Step 1: ML Credit Risk Model (Off-Chain)
Train a logistic regression model on synthetic borrower data. Features: credit utilization ratio, number of past defaults, account age in days, loan-to-value ratio, monthly income. The model outputs P(default) which gets quantized to basis points.

**Why synthetic data?** Real credit data is regulated (FCRA, GDPR). For learning, synthetic data with known distributions lets us verify the model is learning the right patterns.

### Step 2: Oracle Contract
Build a contract that:
- Maintains a whitelist of authorized reporters
- Accepts commit-reveal predictions to prevent front-running
- Stores the latest risk score with a timestamp
- Supports multi-oracle aggregation (median of N reports)
- Enforces a minimum update frequency

### Step 3: Lending Contract
Build a contract that:
- Queries the oracle for a borrower's risk score
- Applies the interest rate curve
- Approves or denies the loan
- Handles collateral deposits and tracks outstanding loans
- Refuses to operate on stale oracle data

### Step 4: Front-Running Attack Demonstration
Show how without commit-reveal, a malicious actor could:
1. See a "high risk" oracle update in the mempool
2. Front-run it by borrowing at the current (lower) rate
3. Profit from the arbitrage

Then show how commit-reveal prevents this.

## Learning Objectives

- Understand the oracle problem and why blockchains need external data bridges
- Implement commit-reveal schemes to prevent front-running
- Integrate off-chain ML predictions with on-chain smart contract logic
- Handle fixed-point arithmetic (basis points) for on-chain financial math
- Design staleness checks and liveness guarantees for oracle data
- Build a multi-oracle aggregation system with median consensus
- Connect ML model outputs to real contract decisions (lending, interest rates)

## Going Deeper

- **Chainlink VRF & Price Feeds**: Production oracles use decentralized reporter networks with staking and slashing for misbehavior. Study how Chainlink's aggregator contract works.
- **Optimistic oracles (UMA)**: Instead of pushing data proactively, anyone can propose a value and it's accepted unless disputed within a challenge period. Lower cost, higher latency.
- **ZK-ML**: Use zero-knowledge proofs to prove an ML model was executed correctly without revealing the model weights. This is cutting-edge — projects like EZKL and Modulus are building this.
- **MEV and oracle extractable value (OEV)**: Oracle updates create MEV opportunities. API3's OEV auction lets protocols capture this value instead of losing it to searchers.
- **Model drift**: ML models degrade over time as data distributions shift. Production systems need monitoring for when the oracle's underlying model needs retraining.
