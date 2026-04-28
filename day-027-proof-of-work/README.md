# Day 027: Proof of Work Simulation

## Overview

Build a complete Proof of Work (PoW) consensus mechanism from scratch — the same fundamental algorithm that secured Bitcoin for over a decade and still underpins many blockchain networks. You'll implement the mining loop, difficulty adjustment, and chain validation that make trustless consensus possible.

**Why it matters:** Proof of Work solved the Byzantine Generals Problem for open networks. Before PoW, there was no way for anonymous, untrusted participants to agree on a shared ledger without a central authority. Understanding PoW deeply means understanding *why* blockchains work, not just *that* they work. Every alternative consensus mechanism (PoS, DPoS, PBFT) is best understood in contrast to PoW.

## Core Concepts

### 1. The Mining Puzzle: Partial Hash Collision

At its heart, PoW is a **partial hash preimage search**. Given a block header `H`, find a nonce `n` such that:

```
SHA-256(H || n) < target
```

The `target` is a 256-bit number. A smaller target means fewer valid hashes exist in the output space, making the puzzle harder. If the target requires `d` leading zero bits, then the probability of any single hash attempt succeeding is:

```
P(success) = 2^(256-d) / 2^256 = 1 / 2^d
```

The expected number of attempts is therefore `2^d`. This is what makes PoW expensive — there's no shortcut. Each hash attempt is independent (memoryless), so the only strategy is brute force.

**Key insight:** The puzzle is *asymmetrically hard*. Finding a valid nonce takes ~2^d attempts, but *verifying* a solution takes exactly 1 hash. This asymmetry is what makes decentralized consensus possible — anyone can cheaply verify that a miner did expensive work.

### 2. Difficulty and Target

Difficulty is typically expressed as a ratio relative to the easiest possible target:

```
difficulty = max_target / current_target
```

In Bitcoin, the max target (difficulty 1) has 32 leading zero bits. The actual difficulty adjusts every 2016 blocks to maintain a 10-minute average block time:

```
new_difficulty = old_difficulty * (expected_time / actual_time)
```

Where `expected_time = 2016 * 10 minutes` and `actual_time` is how long the last 2016 blocks actually took.

**The math behind target adjustment:** If blocks arrived twice as fast as expected, `actual_time = expected_time / 2`, so `new_difficulty = old_difficulty * 2`. The network self-corrects regardless of how much hash power joins or leaves.

### 3. Block Structure

Each block contains:
- **Index**: Position in the chain
- **Timestamp**: When mining began
- **Data**: Transactions or payload
- **Previous hash**: SHA-256 of the prior block (this creates the *chain*)
- **Nonce**: The value miners search for
- **Hash**: SHA-256 of all the above fields concatenated

The previous hash linkage means tampering with block N invalidates blocks N+1, N+2, ... — an attacker must redo the PoW for *every subsequent block*, which becomes exponentially impractical as more blocks are added.

### 4. Chain Validation

A valid chain must satisfy:
1. Genesis block has index 0 and previous_hash = "0"
2. Each block's stored hash matches the recomputed hash of its contents
3. Each block's hash meets the difficulty target at the time it was mined
4. Each block's previous_hash matches the prior block's hash
5. Indices are sequential

### 5. The 51% Attack

If an attacker controls >50% of the network's hash rate, they can:
- Mine a private fork faster than the honest chain
- Release it to orphan honest blocks
- Double-spend transactions

The probability of an attacker with hash fraction `q < 0.5` catching up from `z` blocks behind follows a Poisson distribution (Nakamoto's original analysis):

```
P(catch up) = (q/p)^z  where p = 1-q
```

For q=0.3 and z=6: P ≈ 0.0024 (0.24%) — this is why 6 confirmations is the standard for Bitcoin.

## Step-by-Step Breakdown

### Step 1: Block Data Structure
Define a Block class with all necessary fields. The block hash is computed from the concatenation of index, timestamp, data, previous_hash, and nonce. Using SHA-256 ensures collision resistance — two different blocks will (practically) never produce the same hash.

### Step 2: Mining Function
Implement the nonce search loop. Start nonce at 0, increment, hash, check against target. Track attempts for performance analysis. Without this brute-force search, blocks could be created for free, destroying the economic security model.

### Step 3: Difficulty Adjustment
After a configurable number of blocks, recalculate the target based on actual vs. expected mining time. Without adjustment, adding hash power would make blocks arrive too fast (reducing security), and losing hash power would stall the chain.

### Step 4: Chain Validation
Verify every link in the chain: hash integrity, difficulty compliance, and sequential linkage. This is what allows any new node to independently verify the entire history without trusting anyone.

### Step 5: Mining Simulation
Run a multi-block mining simulation with difficulty adjustment. Measure hash rates, observe difficulty changes, and demonstrate the self-correcting nature of the system.

### Step 6: Attack Simulation
Simulate a 51% attack scenario to demonstrate the security model — show how an attacker with majority hash power can rewrite history, and how the probability drops rapidly for minority attackers.

## Learning Objectives

- Implement SHA-256-based partial hash collision search
- Understand the mathematical relationship between difficulty, target, and expected work
- Build dynamic difficulty adjustment that maintains target block times
- Validate blockchain integrity through hash chain verification
- Analyze the security model and attack economics of PoW
- Measure and reason about mining performance and hash rates

## Going Deeper

- **Selfish mining**: A strategy where miners with >33% hash power can gain disproportionate rewards by strategically withholding blocks. See Eyal & Sirer (2014).
- **ASIC resistance**: Algorithms like Ethash (memory-hard) or RandomX (CPU-optimized) attempt to resist specialized hardware. The tradeoff: ASIC-friendly = more energy efficient but centralized; ASIC-resistant = more decentralized but less efficient.
- **Stratum protocol**: How real mining pools distribute work to individual miners, splitting the nonce space.
- **Merged mining**: Mining multiple chains simultaneously (e.g., Bitcoin + Namecoin) by embedding commitment hashes.
- **Energy analysis**: Bitcoin's PoW consumes ~150 TWh/year. Calculate: if each hash costs X joules and difficulty requires Y hashes per block, what's the energy per transaction?
- **Connection to Day 002 (SHA-256)**: You built the hash function itself; now you're using it as a computational puzzle. The avalanche effect you observed is exactly what makes mining unpredictable.
- **Connection to Day 013 (Merkle Tree)**: Real blocks use Merkle trees to commit to transactions efficiently. A miner can prove any transaction is in a block with O(log n) hashes.
