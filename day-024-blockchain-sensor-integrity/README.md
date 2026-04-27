# Day 024: Blockchain-Verified Sensor Data Pipeline

## Overview

Build a system where a simulated robot produces sensor readings (temperature, pressure, acceleration), each reading is hashed and chained into a tamper-evident blockchain ledger, and a simple ML anomaly detector flags suspicious readings. This is the foundation of **trusted autonomous systems** — robots that can prove their sensor history hasn't been tampered with, while automatically flagging readings that look wrong.

**Why this matters:** In industrial IoT, autonomous vehicles, and supply chain robotics, sensor data drives critical decisions. If an attacker modifies historical sensor logs, they can hide equipment failures, mask environmental hazards, or falsify compliance records. A blockchain-style integrity chain makes tampering detectable. Adding ML anomaly detection catches both sensor malfunctions and adversarial data injection in real time.

## Core Concepts

### 1. Sensor Data Streams

Real robots produce continuous streams of timestamped readings from multiple sensors. Each reading is a vector:

```
reading = (timestamp, sensor_id, value, unit)
```

In production, these arrive at 10-1000 Hz. We simulate this with a data generator that produces realistic readings with occasional anomalies (spikes, drift, stuck sensors).

### 2. Blockchain-Style Data Chaining

Each sensor reading gets a **block** containing:

```
block = {
    index: n,
    timestamp: t,
    data: sensor_reading,
    previous_hash: hash(block[n-1]),
    nonce: 0,        # simplified — no mining
    hash: SHA256(index + timestamp + data + previous_hash + nonce)
}
```

The key insight: because each block includes the hash of the previous block, modifying any historical reading changes its hash, which breaks the chain from that point forward. This is **hash chaining** — the same principle that makes Bitcoin's ledger tamper-evident.

**The math:** SHA-256 produces a 256-bit digest. The probability of finding a collision (two different inputs with the same hash) is approximately 2^(-128) due to the birthday paradox. For practical purposes, this is impossible — it would take ~10^38 operations.

**Tradeoff:** We skip proof-of-work (no mining) because our use case is a single trusted producer. In a multi-robot system, you'd need consensus (PoW, PoS, or BFT) to prevent a rogue robot from creating a fake chain.

### 3. Anomaly Detection with Z-Score

The simplest statistical anomaly detector uses the **z-score**:

```
z = (x - mu) / sigma
```

Where `mu` is the rolling mean and `sigma` is the rolling standard deviation of recent readings. A reading with |z| > 3 means it's more than 3 standard deviations from the mean — this happens with probability < 0.3% for normally distributed data.

**Why rolling statistics?** Sensor readings drift over time (temperature changes throughout the day). A global mean would flag normal daily variation as anomalous. A rolling window (e.g., last 50 readings) adapts to gradual changes while still catching sudden spikes.

**Tradeoff vs. ML models:** Z-score is fast (O(1) per reading with running stats) and interpretable. More complex models (Isolation Forest, autoencoders) catch subtler anomalies but are harder to debug and slower. Start simple, add complexity when simple fails.

### 4. Chain Integrity Verification

To verify the chain hasn't been tampered with:

1. Recompute each block's hash from its contents
2. Check that the stored hash matches the recomputed hash
3. Check that each block's `previous_hash` matches the prior block's hash

If any check fails, the chain is broken at that point — everything from the break onward is suspect.

## Step-by-Step Breakdown

### Step 1: Sensor Simulator
Generate realistic sensor data with configurable noise and injected anomalies. We use Gaussian noise around a base value, with occasional spikes (large deviations) and drift (slow baseline shift). Without realistic data, our anomaly detector would have nothing meaningful to train on.

### Step 2: Block and Blockchain Classes
Implement the hash-chained ledger. Each block hashes its contents plus the previous block's hash. The chain provides `add_reading()` and `verify_integrity()` methods. Without proper hash chaining, the entire integrity guarantee collapses.

### Step 3: Anomaly Detector
Implement rolling z-score anomaly detection. Maintain a sliding window of recent values, compute running mean and standard deviation, flag readings beyond the threshold. Without this, anomalous readings would be faithfully recorded but never flagged.

### Step 4: Pipeline Orchestrator
Wire it all together: sensor produces reading -> anomaly detector scores it -> block is created with reading + anomaly flag -> block is added to chain. This is the integration layer that makes the three components work as a system.

### Step 5: Tamper Detection Demo
Deliberately modify a historical reading in the chain and show that `verify_integrity()` catches the tampering. This proves the blockchain property actually works.

## Learning Objectives

- Implement hash-chained data structures (blockchain fundamentals)
- Build streaming anomaly detection with rolling statistics
- Design a data pipeline that integrates sensing, integrity, and analysis
- Understand tamper-evidence vs. tamper-resistance
- Practice simulation-driven development for robotics systems

## Going Deeper

- **Multi-sensor correlation:** Real anomaly detection correlates across sensors. A temperature spike + pressure drop might be normal (valve opening), while temperature spike alone is anomalous.
- **Merkle trees for batch verification:** Instead of checking every block, group blocks into a Merkle tree for O(log n) verification of any single reading.
- **Consensus for multi-robot systems:** If multiple robots contribute to the same chain, you need Byzantine fault tolerance. Look into PBFT or Raft.
- **Time-series models:** Replace z-score with LSTM autoencoders or Prophet for seasonality-aware anomaly detection.
- **Edge deployment:** In production, this runs on embedded hardware (Raspberry Pi, Jetson Nano) with strict memory/compute budgets — the rolling window size becomes a critical parameter.
