# Day 002: SHA-256 Hash Implementation from Scratch

## Overview

You're building the SHA-256 cryptographic hash function from raw bitwise operations — no hashlib, no shortcuts. SHA-256 is the backbone of Bitcoin's proof-of-work, SSL/TLS certificate verification, digital signatures, and virtually every modern integrity-checking system. Understanding it at the bit level reveals how a fixed set of simple operations can produce outputs that are computationally infeasible to reverse.

## Core Concepts

### What is a cryptographic hash function?

A hash function maps arbitrary-length input to a fixed-length output (256 bits for SHA-256). A *cryptographic* hash must satisfy three properties:

1. **Pre-image resistance**: Given hash `h`, it's infeasible to find any message `m` such that `H(m) = h`. There's no shortcut — you'd need to try ~2^256 inputs.
2. **Second pre-image resistance**: Given `m1`, it's infeasible to find `m2 ≠ m1` with `H(m1) = H(m2)`.
3. **Collision resistance**: It's infeasible to find *any* pair `(m1, m2)` where `H(m1) = H(m2)`. By the birthday paradox, this requires ~2^128 work, not 2^256.

### The Merkle-Damgård construction

SHA-256 follows the Merkle-Damgård structure:

```
message → [pad] → [block₁] → [block₂] → ... → [blockₙ] → digest
                     ↓           ↓                 ↓
              compress(H₀) → compress(H₁) → ... → compress(Hₙ) = final hash
```

The message is padded to a multiple of 512 bits, split into blocks, and each block is processed through a **compression function** that updates an internal state. The security of the whole scheme reduces to the security of the compression function.

**Why padding matters**: SHA-256 appends a `1` bit, enough `0` bits to reach 448 mod 512, then the original message length as a 64-bit big-endian integer. This ensures:
- Different-length messages can't collide trivially
- The length encoding prevents *length extension attacks* (well, partially — SHA-256 is still vulnerable to length extension, which is why HMAC exists)

### The compression function — where the magic happens

Each 512-bit block is processed in 64 rounds. The state is eight 32-bit words (a, b, c, d, e, f, g, h), initialized to specific fractional parts of square roots of the first 8 primes.

Each round applies:
```
T1 = h + Σ1(e) + Ch(e,f,g) + K[i] + W[i]
T2 = Σ0(a) + Maj(a,b,c)
h = g; g = f; f = e; e = d + T1; d = c; c = b; b = a; a = T1 + T2
```

Where:
- **Ch(e,f,g)** = (e AND f) XOR (NOT e AND g) — "choice": e picks bits from f or g
- **Maj(a,b,c)** = (a AND b) XOR (a AND c) XOR (b AND c) — "majority": majority vote of each bit position
- **Σ0(a)** = ROTR(2,a) XOR ROTR(13,a) XOR ROTR(22,a) — scrambles bits via rotation
- **Σ1(e)** = ROTR(6,e) XOR ROTR(11,e) XOR ROTR(25,e)
- **K[i]** = round constants (fractional parts of cube roots of first 64 primes)
- **W[i]** = message schedule (first 16 words from block, rest derived)

### The message schedule

The first 16 words W[0..15] come directly from the 512-bit block. The remaining 48 are:
```
W[i] = σ1(W[i-2]) + W[i-7] + σ0(W[i-15]) + W[i-16]
```
Where σ0 and σ1 are "small sigma" functions (different rotations than the Σ functions). This ensures every input bit influences the entire output — changing one bit of input cascades through all 64 rounds.

### Why these specific operations?

The rotation amounts, the choice of primes for constants, and the combination of XOR/AND/NOT were chosen to:
- Maximize **diffusion** (every input bit affects every output bit)
- Maximize **confusion** (the relationship between input and output is complex)
- Use only operations efficient in hardware: addition, bitwise ops, and rotation
- The constants from prime numbers ensure there are no "nothing up my sleeve" backdoors

### The avalanche effect

Change one bit of input → ~50% of output bits flip. This is measurable and is a key design goal. We'll demonstrate this in our implementation.

## Step-by-Step Breakdown

### Step 1: Message preprocessing
Convert the input to bytes, append the padding (1-bit + zeros + length). This creates complete 512-bit blocks. Without proper padding, messages of different lengths could produce the same padded blocks.

### Step 2: Initialize hash values
Set H[0..7] to the fractional parts of the square roots of the first 8 primes (2, 3, 5, 7, 11, 13, 17, 19). These are not arbitrary — using irrational numbers ensures no hidden structure.

### Step 3: Initialize round constants
K[0..63] are the fractional parts of the cube roots of the first 64 primes. Same rationale.

### Step 4: Process each block
For each 512-bit block:
1. Prepare the message schedule W[0..63]
2. Initialize working variables (a..h) from current hash state
3. Run 64 rounds of the compression function
4. Add the compressed values back to the hash state (modular addition)

### Step 5: Produce final hash
Concatenate H[0..7] as 32-bit big-endian integers → 256-bit hash.

## Learning Objectives

- Understand the internal structure of SHA-256 at the bit manipulation level
- Implement Merkle-Damgård construction and see why padding schemes matter
- Work with 32-bit modular arithmetic, bitwise rotations, and logical functions
- Verify correctness against known test vectors (NIST FIPS 180-4)
- Measure and visualize the avalanche effect
- Connect hash functions to their role in blockchain (Bitcoin mining uses double SHA-256)

## Going Deeper

- **Length extension attacks**: Because SHA-256 outputs its internal state directly, knowing `H(m)` and `len(m)` lets you compute `H(m || padding || m')` without knowing `m`. This is why HMAC wraps the hash in `H(K⊕opad || H(K⊕ipad || m))`.
- **Bitcoin's double hashing**: Bitcoin uses `SHA-256(SHA-256(block_header))` partly to mitigate length extension and partly as defense-in-depth.
- **Hardware optimization**: SHA-256's operations map directly to CPU instructions (Intel SHA Extensions). Mining ASICs implement the compression function in dedicated circuits.
- **SHA-3 (Keccak)**: Uses a completely different construction (sponge) that avoids the Merkle-Damgård weaknesses. Not a replacement for SHA-256 — a complement.
- **Next up**: Day's Merkle tree challenge will build directly on this hash function to create tamper-evident data structures.
