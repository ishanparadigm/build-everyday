# Day 020: Digital Signatures (ECDSA Basics)

## What You're Building

A complete Elliptic Curve Digital Signature Algorithm (ECDSA) implementation from scratch. This is the exact cryptographic primitive that secures every Bitcoin and Ethereum transaction — when you send crypto, your wallet uses ECDSA to prove you authorized the transfer without revealing your private key. Understanding ECDSA deeply means understanding the mathematical trust layer beneath all of blockchain.

## Core Concepts

### Elliptic Curves Over Finite Fields

An elliptic curve is defined by the equation:

```
y² = x³ + ax + b  (mod p)
```

In regular math, this traces a smooth curve. But we work over a **finite field** F_p (integers mod a prime p), which turns it into a scattered set of discrete points. The magic: these points still form a **group** — you can "add" two points and get another point on the curve.

**Why finite fields?** Real-number arithmetic has rounding errors and is continuous — useless for cryptography. Finite fields give us exact arithmetic with a fixed number of discrete elements, making the math deterministic and the discrete logarithm problem hard.

**secp256k1** (used by Bitcoin/Ethereum) uses:
- `a = 0, b = 7` → `y² = x³ + 7`
- `p = 2²⁵⁶ - 2³² - 977` (a massive prime)
- A generator point `G` with order `n` (the number of times you can add G to itself before getting back to the identity)

### Point Addition and Scalar Multiplication

**Point addition** (P + Q where P ≠ Q):
1. Draw a line through P and Q
2. It intersects the curve at a third point R'
3. Reflect R' over the x-axis to get R = P + Q

Algebraically over F_p:
```
slope = (y₂ - y₁) · (x₂ - x₁)⁻¹  mod p
x₃ = slope² - x₁ - x₂  mod p
y₃ = slope · (x₁ - x₃) - y₁  mod p
```

**Point doubling** (P + P):
```
slope = (3x₁² + a) · (2y₁)⁻¹  mod p
```

**Scalar multiplication** (k · G): Add G to itself k times. We use the **double-and-add** algorithm (analogous to square-and-multiply in RSA) to do this in O(log k) steps instead of O(k).

### The Discrete Logarithm Problem (DLP)

Given points G and Q = k·G on the curve, finding k is computationally infeasible. This is the **trapdoor**: multiplication is easy (O(log k) point additions), but the reverse is essentially impossible for 256-bit k. This asymmetry is the entire foundation of ECDSA security.

### The ECDSA Algorithm

**Key Generation:**
1. Pick a random private key `d` (a 256-bit integer, 1 < d < n)
2. Compute public key `Q = d · G`

**Signing a message m:**
1. Hash the message: `z = hash(m)` (truncated to bit-length of n)
2. Pick a random nonce `k` (1 < k < n) — **this MUST be truly random and unique per signature**
3. Compute `R = k · G`, let `r = R.x mod n`. If r = 0, pick new k.
4. Compute `s = k⁻¹ · (z + r · d) mod n`. If s = 0, pick new k.
5. Signature is `(r, s)`

**Verifying signature (r, s) on message m with public key Q:**
1. Compute `z = hash(m)`
2. Compute `w = s⁻¹ mod n`
3. Compute `u₁ = z · w mod n` and `u₂ = r · w mod n`
4. Compute `R' = u₁ · G + u₂ · Q`
5. Signature is valid iff `R'.x mod n == r`

**Why verification works:** Substituting back, `u₁·G + u₂·Q = (z·s⁻¹)·G + (r·s⁻¹)·(d·G) = s⁻¹·(z + r·d)·G = k·G = R`. The math closes the loop.

### The Nonce Catastrophe

If you ever reuse a nonce k for two different messages, an attacker can recover your private key:

```
s₁ = k⁻¹(z₁ + r·d)
s₂ = k⁻¹(z₂ + r·d)
s₁ - s₂ = k⁻¹(z₁ - z₂)
k = (z₁ - z₂) · (s₁ - s₂)⁻¹
d = (s₁·k - z₁) · r⁻¹
```

This is not theoretical — Sony's PlayStation 3 signing key was extracted this way in 2010 because they used a static nonce. In production, RFC 6979 deterministic nonces solve this.

## Step-by-Step Breakdown

1. **Implement modular arithmetic helpers** — modular inverse using extended Euclidean algorithm. Every division in elliptic curve math is actually multiplication by a modular inverse.

2. **Implement the Point class** — represent points on the curve, including the "point at infinity" (the identity element, like zero for addition).

3. **Implement point addition and doubling** — the core group operation. Get this wrong and nothing else works. Must handle edge cases: adding infinity, adding a point to its inverse.

4. **Implement scalar multiplication** — double-and-add algorithm. This is the performance-critical operation — used in key generation, signing, and verification.

5. **Implement key generation** — random private key → public key via scalar multiplication of the generator point.

6. **Implement signing** — hash the message, pick a nonce, compute r and s. The nonce generation is the most security-critical step.

7. **Implement verification** — recompute the point from the signature and public key, check if it matches.

8. **Demonstrate the nonce-reuse attack** — show concretely how reusing a nonce leaks the private key.

## Learning Objectives

- Understand elliptic curve arithmetic over finite fields (point addition, doubling, scalar multiplication)
- Implement modular inverse using the extended Euclidean algorithm
- Build ECDSA key generation, signing, and verification from first principles
- Understand why nonce security is critical and demonstrate a nonce-reuse attack
- Connect the math to real-world blockchain transaction signing

## Going Deeper

- **RFC 6979**: Deterministic nonce generation eliminates the nonce-reuse risk entirely by deriving k from the private key and message hash via HMAC-DRBG
- **Schnorr signatures** (BIP 340): Simpler, support native key/signature aggregation (MuSig2), adopted by Bitcoin in the Taproot upgrade
- **EdDSA / Ed25519**: Uses twisted Edwards curves for faster, constant-time implementations — used by Solana, Cardano, and SSH
- **Signature malleability**: Given valid (r, s), (r, n-s) is also valid. Ethereum uses `v` (recovery id) and enforces low-s to prevent this
- **Key recovery**: From a signature + recovery id, you can recover the public key without it being provided — this is how Ethereum's `ecrecover` works, saving 32 bytes per transaction
