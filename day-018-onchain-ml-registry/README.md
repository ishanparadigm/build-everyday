# Day 018: On-Chain ML Model Registry

## Overview

You're building a **model registry on a blockchain** — a system that records ML model metadata (architecture, hyperparameters, training data hash, performance metrics) as immutable, tamper-proof entries on a simulated blockchain. Anyone can verify that a model hasn't been silently swapped, its training data hasn't changed, and its claimed accuracy is the one that was attested at registration time.

### Why This Matters

In production ML, **model provenance** is a growing problem. When a model makes a high-stakes decision (credit scoring, medical diagnosis, autonomous driving), regulators and auditors need to answer:

- *Which exact model made this prediction?*
- *What data was it trained on?*
- *Has anyone tampered with it since deployment?*

Traditional model registries (MLflow, Weights & Biases) store this metadata in centralized databases — a single admin can silently edit entries. An on-chain registry makes the record **append-only and tamper-evident**: once a model is registered, its metadata is cryptographically sealed into the chain. Any modification breaks the hash chain and is immediately detectable.

This challenge integrates concepts from three tracks:
- **AI**: Model metadata, evaluation metrics, serialization hashing
- **Crypto**: Blockchain structure (Day 002's SHA-256, Day 013's Merkle trees), digital signatures, immutability
- **Robotics**: Think of this as the "black box recorder" for any autonomous system — logging which model version controlled the robot at what time

## Core Concepts

### 1. Model Identity via Content Hashing

A model's identity should be derived from its *content*, not a human-assigned name. We hash the serialized model weights to produce a unique fingerprint:

```
model_hash = SHA-256(serialized_weights)
```

If a single weight changes by 1e-15, the hash changes completely (avalanche effect from Day 002). This gives us a **content-addressed** registry — the hash *is* the model's identity.

For our simulation, we'll hash a dictionary of model metadata + a simulated weights blob to produce this fingerprint.

### 2. Model Metadata Schema

Each registry entry contains:

| Field | Type | Purpose |
|-------|------|---------|
| `model_hash` | str | SHA-256 of serialized weights |
| `name` | str | Human-readable model name |
| `version` | str | Semantic version (e.g., "1.2.0") |
| `architecture` | str | Model type ("CNN", "RandomForest", etc.) |
| `hyperparameters` | dict | Training config |
| `training_data_hash` | str | SHA-256 of training dataset |
| `metrics` | dict | Evaluation results (accuracy, F1, etc.) |
| `owner` | str | Public key of the registrant |
| `signature` | str | Owner's digital signature over the entry |
| `timestamp` | float | Registration time |

### 3. Digital Signatures for Ownership

We use **ECDSA** (Elliptic Curve Digital Signature Algorithm) to prove *who* registered a model. The flow:

1. Owner generates a key pair: `(private_key, public_key)`
2. Owner hashes the model metadata: `entry_hash = SHA-256(canonical_json(metadata))`
3. Owner signs: `signature = ECDSA_sign(private_key, entry_hash)`
4. Anyone can verify: `ECDSA_verify(public_key, entry_hash, signature) -> bool`

This prevents impersonation — you can't register a model under someone else's identity without their private key.

For simplicity, we'll use HMAC-based signatures (simulating ECDSA with a shared-secret model) so we don't need external crypto libraries.

### 4. Blockchain Storage

Each block in our chain contains:
- **Block header**: index, timestamp, previous block hash, nonce
- **Merkle root**: Root hash of all model entries in the block (from Day 013)
- **Model entries**: One or more model registration records

The Merkle root means we can efficiently prove a specific model exists in a block without downloading every entry — an O(log n) inclusion proof.

### 5. Verification Pipeline

The registry supports three types of verification:

1. **Chain integrity**: Walk the chain and verify each block's `previous_hash` matches the actual hash of the prior block
2. **Entry integrity**: Recompute the Merkle root from entries and verify it matches the stored root
3. **Ownership verification**: Verify the digital signature on each model entry against the claimed owner's public key

If any check fails, we know exactly *where* tampering occurred.

## Step-by-Step Breakdown

### Step 1: Build the Model Entry structure
Create a dataclass that holds all model metadata. Implement canonical JSON serialization (sorted keys, no whitespace) so the hash is deterministic. Compute the entry hash from this canonical form.

### Step 2: Implement the signing system
Create a simple key pair generator and HMAC-based sign/verify functions. The owner signs the entry hash with their private key.

### Step 3: Build the Merkle tree for block entries
Reuse the Merkle tree concept from Day 013 — hash all entry hashes in a block into a single root. This enables efficient inclusion proofs.

### Step 4: Build the Block and Blockchain
Each block stores entries + Merkle root + link to previous block. Implement `add_block()` with optional proof-of-work (lightweight, just a few leading zeros).

### Step 5: Build the Registry API
High-level functions: `register_model()`, `get_model()`, `verify_chain()`, `verify_model()`, `list_models()`. This is the interface users interact with.

### Step 6: Demonstrate tamper detection
Register several models, then deliberately tamper with an entry and show how verification catches it.

## Learning Objectives

- Understand **content-addressed storage** and why hashing model weights creates tamper-evident identity
- Implement a **domain-specific blockchain** (not a general-purpose one) optimized for ML metadata
- Connect **digital signatures** to ownership and non-repudiation in model governance
- Use **Merkle trees** for efficient inclusion proofs of model entries within blocks
- Build a complete **verification pipeline** that detects tampering at chain, block, and entry levels
- See how AI model lifecycle management intersects with cryptographic integrity guarantees

## Going Deeper

- **Real ECDSA**: Replace HMAC signatures with actual ECDSA using Python's `cryptography` library for production-grade ownership proofs
- **IPFS integration**: Store actual model weights on IPFS and record the CID on-chain — content-addressed storage at both layers
- **Zero-knowledge proofs**: Prove a model achieves >90% accuracy without revealing the model weights (connects to ZK concepts in weeks 9-12)
- **Model lineage DAG**: Track parent models (fine-tuned from base) as a directed acyclic graph on-chain
- **Smart contract registry**: Port this to Solidity as an ERC-721 where each model is an NFT with metadata URI pointing to the on-chain record
- **Federated registries**: Multiple organizations maintain their own chains but cross-reference via Merkle proofs across chains
