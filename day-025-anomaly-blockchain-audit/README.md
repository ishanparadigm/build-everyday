# Day 025: AI Anomaly Detection with Blockchain Audit Trail

## Overview

Build a system that uses machine learning to detect anomalies in streaming data and records every detection decision on an immutable blockchain-style audit ledger. This is a real-world pattern used in industrial IoT, financial compliance, and cybersecurity — wherever you need to prove *when* an anomaly was detected, *what* evidence triggered it, and that no one tampered with the record after the fact.

The system integrates three foundational concepts:
1. **Isolation Forest** — an unsupervised ML algorithm for anomaly detection
2. **Hash-chained audit ledger** — blockchain fundamentals (hash linking, tamper evidence)
3. **Streaming evaluation pipeline** — tying detection and logging together in real-time

## Core Concepts

### Isolation Forest: Why Anomalies Are Easy to Isolate

Most anomaly detection methods model "normal" and then flag deviations. Isolation Forest flips this: it directly targets anomalies by exploiting one key insight — **anomalous points are few and different, so they are easier to isolate**.

**How it works:**

1. Build a binary tree by randomly selecting a feature and a random split value within that feature's range.
2. Repeat until every point is isolated (in its own leaf) or a max depth is reached.
3. Anomalies end up in shorter paths because they sit in sparse regions — fewer splits needed to separate them.

**The math — anomaly score:**

For a point x with average path length E(h(x)) across T trees:

```
s(x, n) = 2^(-E(h(x)) / c(n))
```

Where c(n) is the average path length of an unsuccessful search in a Binary Search Tree:

```
c(n) = 2 * H(n-1) - 2(n-1)/n
H(i) = ln(i) + 0.5772156649  (Euler-Mascheroni constant)
```

- s(x, n) → 1: strong anomaly (short path)
- s(x, n) → 0.5: normal (average path)
- s(x, n) → 0: very "inlier" (longer than average path)

**Why random splits work:** In high-density regions, random splits rarely isolate a point because neighbors exist on both sides. In low-density regions (where anomalies live), a single random split often puts the anomaly alone on one side. Over many trees, this statistical pattern becomes robust.

**Tradeoffs vs. other approaches:**
- vs. Z-score: Z-score assumes Gaussian distribution. Isolation Forest works on any distribution shape.
- vs. DBSCAN: DBSCAN requires tuning epsilon and minPts. Isolation Forest has fewer hyperparameters and scales better.
- vs. Autoencoders: Neural approaches need more data and compute. Isolation Forest works well with small-to-medium datasets.

### Hash-Chained Audit Ledger

Each audit record contains:
- **Timestamp** and **data snapshot** (the evidence)
- **Detection result** (anomaly score, decision, threshold)
- **Previous hash** — the SHA-256 hash of the prior record
- **Current hash** — SHA-256 of (previous_hash + record_contents)

This creates a chain where modifying any past record invalidates all subsequent hashes. To verify integrity, simply recompute hashes from genesis and compare.

**Why this matters in practice:** Regulatory frameworks (SOX, HIPAA, GDPR) often require demonstrable integrity of audit logs. A hash chain provides *cryptographic* proof that records haven't been altered — stronger than filesystem permissions or database ACLs.

### Streaming Evaluation Pipeline

The pipeline processes data points one at a time:
1. **Ingest** — receive a new data point
2. **Detect** — score it with the Isolation Forest (trained on historical "normal" data)
3. **Decide** — classify as anomaly or normal based on threshold
4. **Record** — append the decision + evidence to the audit ledger
5. **Verify** — periodically validate the entire chain's integrity

This mirrors how production systems work: models are trained offline on historical data, then deployed to score incoming data in real-time.

## Step-by-Step Breakdown

### Step 1: Implement Isolation Tree

Build a single binary tree that recursively splits data on random features and random split values. Each node stores the split feature and value; leaves store the number of data points that reached them. Track path lengths during prediction.

*Why random splits specifically?* Deliberate splits (like in decision trees) would optimize for class separation. We want random splits because their expected behavior differs between dense and sparse regions — that asymmetry is the signal.

### Step 2: Build the Isolation Forest

Create an ensemble of T isolation trees, each trained on a random subsample of size ψ (psi). The subsampling is critical — it:
- Reduces computational cost from O(n²) to O(n·ψ)
- Introduces diversity between trees (like bagging)
- The recommended ψ=256 works well across most datasets

### Step 3: Anomaly Scoring

For each point, compute its average path length across all trees, then convert to the anomaly score using the formula above. The c(n) normalization ensures scores are comparable across different dataset sizes.

### Step 4: Build the Audit Ledger

Implement a hash-chain where each entry contains the detection decision and its cryptographic link to the previous entry. The genesis block has a zeroed previous hash.

### Step 5: Streaming Pipeline

Wire everything together: generate synthetic data with injected anomalies, run the detector, log every decision to the ledger, and verify the chain at the end.

### Step 6: Demonstrate Tamper Detection

Modify a record in the middle of the chain and show that verification catches it — this is the whole point of the blockchain layer.

## Learning Objectives

- Implement Isolation Forest from scratch, understanding why random partitioning isolates anomalies
- Calculate anomaly scores using average path length normalization
- Build a hash-chained audit ledger with SHA-256 linking
- Integrate ML inference with cryptographic logging in a streaming pipeline
- Demonstrate tamper detection on audit records
- Understand the tradeoffs between anomaly detection algorithms

## Going Deeper

- **Streaming Isolation Forest (iForestASD):** Replace static trees with ones that adapt to concept drift — new trees replace old ones as the data distribution shifts.
- **Extended Isolation Forest:** Use hyperplane splits instead of axis-aligned splits to handle correlated features.
- **Merkle Tree audit:** Instead of a linear chain, organize audit records in a Merkle tree for O(log n) inclusion proofs — useful when you need to prove a specific record exists without revealing the entire chain.
- **Production deployment:** In real systems, the model is versioned and the model hash is included in each audit record, creating a full provenance chain from model training → deployment → inference → audit.
- **Multi-variate anomalies:** Points that are normal in each dimension individually but anomalous in combination (e.g., high temperature + low pressure). Isolation Forest handles these naturally since splits operate in the full feature space.
