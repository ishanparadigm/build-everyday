# Day 038: Staking Contract

## Overview

Build a complete ERC-20 token staking system from scratch in Python. Users deposit tokens, earn rewards proportional to their stake over time, and can withdraw at will. This is the economic engine behind DeFi protocols like Aave, Lido, and Synthetix — staking aligns incentives by rewarding participants who lock capital, providing liquidity or security to the protocol.

Understanding staking mechanics is essential because virtually every DeFi protocol uses some variant: liquidity mining, validator staking, governance staking, or yield farming all share the same core math.

## Core Concepts

### 1. Reward Distribution: The "Reward Per Token" Accumulator

The naive approach to staking rewards — iterating over every staker each block to distribute rewards — is O(n) per update and doesn't scale. Synthetix solved this with a **global accumulator pattern**:

```
rewardPerTokenStored += (rewardRate * timeDelta) / totalStaked
```

Each user tracks `userRewardPerTokenPaid` — the value of the accumulator when they last interacted. Their pending reward is:

```
earned = balance * (rewardPerTokenAccumulated - userRewardPerTokenPaid) + previouslyEarned
```

**Why this works**: Instead of distributing to N users, we maintain one global number. Each user "catches up" to the current state on their next interaction. This is O(1) per update regardless of how many stakers exist.

**The math, from first principles**: If the protocol emits R tokens/second and there are S tokens staked total, each staked token earns R/S per second. Over a time interval dt, each staked token earns (R * dt) / S. Summing these intervals gives the cumulative reward per token — that's exactly what `rewardPerTokenStored` tracks.

### 2. Time-Weighted Staking

Rewards must be proportional to both **amount staked** and **duration staked**. A user who stakes 100 tokens for 10 days should earn the same as a user who stakes 1000 tokens for 1 day (assuming constant total supply). The accumulator pattern naturally handles this because it integrates reward rate over time.

**Tradeoff**: Some protocols add lock-up periods or multipliers for longer staking. This increases capital efficiency for the protocol but reduces user flexibility. Our implementation uses the simpler "stake anytime, withdraw anytime" model.

### 3. Reentrancy and State Ordering

In Solidity, the order of operations matters critically:

1. **Update global state** (rewardPerToken)
2. **Update user state** (earned rewards, paid checkpoint)
3. **Transfer tokens** (the external call)

This is the **checks-effects-interactions** pattern. If you transfer before updating state, a malicious contract can re-enter and claim rewards multiple times. We simulate this concern in Python to build the right mental model.

### 4. Reward Rate and Duration

A staking contract typically has a fixed reward budget distributed over a fixed duration:

```
rewardRate = totalRewardBudget / durationInSeconds
```

If new rewards are added before the current period ends, the remaining undistributed rewards roll into the new period:

```
leftover = (periodEnd - now) * oldRewardRate
newRewardRate = (leftover + newRewards) / newDuration
```

This prevents reward dilution and ensures smooth distribution.

## Step-by-Step Breakdown

### Step 1: Token Balances
Simulate ERC-20 token balances for both the staking token and the reward token. We need `transfer`, `balanceOf`, and `approve/transferFrom` semantics. Without this foundation, we can't track who owns what.

### Step 2: Core Staking State
Track `totalStaked`, per-user `stakedBalance`, the global `rewardPerTokenStored`, per-user `userRewardPerTokenPaid`, and per-user `rewards` (earned but unclaimed). These five variables are the complete state of the system.

### Step 3: The Update Reward Modifier
Before every stake/withdraw/claim operation, update the global accumulator and the user's earned rewards. This is the critical invariant — if you forget this step, rewards will be miscalculated. In Solidity this is typically a `modifier`; in Python we'll use a decorator or explicit call.

### Step 4: Stake
Transfer staking tokens from user to contract, increase their balance and totalStaked. Must call updateReward first so existing rewards are checkpointed before the balance changes.

### Step 5: Withdraw
Decrease balance and totalStaked, transfer tokens back. Same updateReward requirement. If we updated state after transfer, a reentrant call could withdraw again with stale balances.

### Step 6: Claim Rewards
Calculate earned rewards, zero out the user's pending rewards, transfer reward tokens. Separate from withdraw so users can compound or claim independently.

### Step 7: Notify Reward Amount
Admin function to start/extend a reward period. Sets the reward rate and period end time. If called mid-period, remaining rewards are folded in.

### Step 8: Time Simulation
Since we're in Python (not on-chain), we simulate block timestamps to test reward accrual over time.

## Learning Objectives

- Implement the Synthetix reward accumulator pattern — the most widely-used staking math in DeFi
- Understand time-weighted reward distribution and why naive approaches don't scale
- Practice the checks-effects-interactions pattern for state safety
- Build intuition for reward rate calculations and period management
- Learn how staking contracts handle edge cases: zero total supply, mid-period reward additions, partial withdrawals

## Going Deeper

- **Compounding**: Auto-compound rewards by re-staking claimed tokens. This changes the APY calculation significantly (continuous compounding vs simple interest).
- **Lock-up periods**: Add minimum staking durations with early withdrawal penalties. Curve's ve-tokenomics uses time-locked staking to boost governance power.
- **Multiple reward tokens**: Synthetix V2 supports multiple reward tokens per staking pool. Each needs its own accumulator.
- **Slashing**: Validator staking (Ethereum PoS) includes slashing for misbehavior — reducing staked balances as punishment.
- **Gauge voting**: Curve/Balancer let governance token holders vote on how rewards are distributed across pools, creating "bribe markets."
- **EIP-4626 Tokenized Vaults**: The modern standard for yield-bearing tokens, which generalizes staking into a composable primitive.
