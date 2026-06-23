# Day 83: Zero-Knowledge Proofs from First Principles

## Overview

Build a zero-knowledge proof system in Python that lets a prover convince a verifier they know a secret **without revealing the secret itself**. This is one of the most powerful ideas in modern cryptography and the backbone of privacy-preserving blockchains (Zcash, zkSync, StarkNet), private identity verification, and scalable L2 rollups.

We'll implement three ZK proof systems of increasing sophistication:
1. **Schnorr's Protocol** — prove you know a discrete logarithm
2. **Sigma Protocol for Graph Isomorphism** — a classic NP-complete ZK proof
3. **zk-SNARK-style Arithmetic Circuit Verification** — prove you know a satisfying assignment to an arithmetic circuit (the foundation of modern ZK systems)

## Core Concepts

### What is a Zero-Knowledge Proof?

A zero-knowledge proof is an interactive protocol between two parties:
- **Prover (P)**: knows a secret and wants to convince the verifier
- **Verifier (V)**: wants to be convinced the prover knows the secret, without learning the secret

Three properties must hold:
1. **Completeness**: If the prover knows the secret, the verifier always accepts
2. **Soundness**: If the prover does NOT know the secret, the verifier rejects with overwhelming probability
3. **Zero-knowledge**: The verifier learns nothing beyond the fact that the statement is true

### The Discrete Logarithm Problem

Given a prime `p`, a generator `g` of the multiplicative group Z*_p, and a value `y = g^x mod p`, finding `x` given `(g, p, y)` is computationally hard for large primes. This is the basis of Schnorr's protocol.

**The math:**
- Public parameters: prime `p`, generator `g`, public key `y = g^x mod p`
- Secret: `x` (the discrete log)
- The prover wants to prove they know `x` such that `y = g^x mod p`

### Schnorr's Protocol (Sigma Protocol)

This is a 3-move protocol (commit → challenge → response):

1. **Commit**: Prover picks random `r`, computes `t = g^r mod p`, sends `t` to verifier
2. **Challenge**: Verifier picks random challenge `c` and sends it to prover  
3. **Response**: Prover computes `s = r + c*x mod (p-1)` and sends `s` to verifier
4. **Verify**: Verifier checks that `g^s ≡ t * y^c (mod p)`

**Why this works:**
- **Completeness**: `g^s = g^(r + cx) = g^r * g^(cx) = t * (g^x)^c = t * y^c` ✓
- **Soundness**: Without knowing `x`, the prover can't produce a valid `s` for a random `c`
- **Zero-knowledge**: The transcript `(t, c, s)` can be simulated without knowing `x` — pick random `s, c`, compute `t = g^s * y^(-c)`. This simulated transcript is indistinguishable from a real one.

### The Fiat-Shamir Heuristic

Interactive proofs require back-and-forth communication. The **Fiat-Shamir transform** makes them non-interactive by replacing the verifier's random challenge with a hash of the commitment:

```
c = H(g || y || t)
```

This is secure in the **random oracle model** — the hash function is unpredictable, so the prover can't choose `t` to get a favorable `c`. This transforms our interactive Schnorr protocol into a **digital signature scheme** (Schnorr signatures).

### Arithmetic Circuits and R1CS

Modern ZK systems (Groth16, PLONK, STARKs) operate on **arithmetic circuits** — computations expressed as addition and multiplication gates over a finite field.

**Rank-1 Constraint System (R1CS):**
An R1CS encodes a computation as a set of constraints of the form:
```
(a · w) * (b · w) = (c · w)
```
where `w` is the witness vector (public inputs + private inputs + intermediate values), and `a, b, c` are coefficient vectors.

For example, to prove `x^3 + x + 5 = 35` (proving we know x=3):
- Introduce intermediate variables: `v1 = x*x`, `v2 = v1*x`, `v3 = v2 + x + 5`
- Witness: `w = [1, 35, x, v1, v2, v3]` = `[1, 35, 3, 9, 27, 35]`
- Constraints:
  - `x * x = v1` → `(0,0,1,0,0,0) · w * (0,0,1,0,0,0) · w = (0,0,0,1,0,0) · w`
  - `v1 * x = v2` → `(0,0,0,1,0,0) · w * (0,0,1,0,0,0) · w = (0,0,0,0,1,0) · w`
  - `(v2 + x + 5) * 1 = v3` → `(5,0,1,0,1,0) · w * (1,0,0,0,0,0) · w = (0,0,0,0,0,1) · w`

### Why ZK Proofs Matter in Crypto

1. **Privacy**: Zcash uses zk-SNARKs to prove a transaction is valid without revealing sender, receiver, or amount
2. **Scalability**: zkRollups (zkSync, StarkNet) batch thousands of transactions and generate one proof that the batch is valid — verified cheaply on L1
3. **Identity**: Prove you're over 18 without revealing your birthdate
4. **Compliance**: Prove your funds are from legitimate sources without revealing your transaction history

## Step-by-Step Breakdown

### Step 1: Schnorr's Interactive ZK Proof
Implement the 3-round protocol with proper modular arithmetic. We need a safe prime `p` where `(p-1)/2` is also prime, ensuring the group has no small subgroups that could leak information.

### Step 2: Fiat-Shamir Non-Interactive Proof
Replace the verifier's challenge with `H(g || p || y || t)` using SHA-256. This produces a proof that anyone can verify without interaction.

### Step 3: Sigma Protocol Composition
Implement AND-composition (prove you know BOTH secrets) and OR-composition (prove you know AT LEAST ONE secret without revealing which).

### Step 4: Arithmetic Circuit ZK Verification
Build an R1CS system, create witness vectors, and verify that a prover knows a satisfying assignment without revealing it. This is the conceptual foundation of zk-SNARKs.

### Step 5: Practical Application — ZK Range Proof
Prove a value lies in a range [0, 2^n) without revealing the value. This is used in confidential transactions to prove amounts are non-negative.

## Learning Objectives

- Understand the three properties of ZK proofs (completeness, soundness, zero-knowledge)
- Implement Schnorr's protocol with correct modular arithmetic
- Apply the Fiat-Shamir heuristic to make interactive proofs non-interactive
- Build R1CS constraint systems for arithmetic circuits
- Understand how these primitives compose into production ZK systems
- Connect ZK proofs to real blockchain applications (zkRollups, private transactions)

## Going Deeper

- **Groth16**: The most commonly deployed zk-SNARK — requires a trusted setup but produces tiny proofs (3 group elements). Used in Zcash.
- **PLONK**: Universal trusted setup that works for any circuit. Used in zkSync and Aztec.
- **STARKs**: No trusted setup, post-quantum secure, but larger proofs. Used in StarkNet.
- **Bulletproofs**: No trusted setup, used for range proofs in Monero. Proof size is logarithmic in the circuit size.
- **Recursive proofs**: A proof that verifies another proof — enables infinite composability (Mina Protocol's constant-size blockchain).
- **KZG commitments**: Polynomial commitment scheme underlying PLONK and EIP-4844 (proto-danksharding).

Building on previous days: This connects to Day 20 (ECDSA — same discrete log assumption), Day 2 (SHA-256 — used in Fiat-Shamir), and Day 27 (Proof of Work — computational hardness assumptions).
