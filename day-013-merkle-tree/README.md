# Day 013: Merkle Tree from Scratch

## Overview

A **Merkle tree** (or hash tree) is the data structure that makes blockchains, Git, and distributed file systems possible. It answers a deceptively hard question: *how do you prove that a piece of data belongs to a large dataset without transmitting the entire dataset?*

Bitcoin uses Merkle trees to summarize every transaction in a block into a single 32-byte root hash. Git uses them to detect which files changed between commits. IPFS uses them to verify file chunks downloaded from untrusted peers. The pattern is universal: whenever you need **tamper-evident, efficiently-verifiable data integrity**, you reach for a Merkle tree.

Today we build one from scratch — including tree construction, root computation, proof generation, and proof verification.

## Core Concepts

### Cryptographic Hash Functions (Recap from Day 002)

A hash function `H(x)` maps arbitrary-length input to a fixed-length digest. For Merkle trees we need two properties:

1. **Collision resistance**: It's computationally infeasible to find `x != y` such that `H(x) = H(y)`.
2. **Avalanche effect**: Changing a single bit of input flips ~50% of output bits.

We'll use SHA-256, which gives us 256-bit (32-byte) digests. This connects directly to what you built in Day 002.

### From Flat Lists to Trees

Suppose you have 8 data blocks and want to verify block #3 is authentic. The naive approach: hash all 8 blocks and send all 8 hashes. That's O(n) communication.

The Merkle insight: arrange the hashes in a **binary tree**. Each leaf is `H(data_i)`. Each internal node is `H(left_child || right_child)` — the hash of its two children concatenated. The root summarizes the entire dataset in a single hash.

```
        Root = H(H01 || H23)
       /                    \
   H01 = H(H0 || H1)    H23 = H(H2 || H3)
    /       \             /       \
  H0=H(A)  H1=H(B)    H2=H(C)  H3=H(D)
   |         |          |         |
   A         B          C         D
```

### The Math: Why O(log n) Proofs Work

To prove leaf `i` belongs to a tree with `n` leaves, you need a **Merkle proof** (also called an **audit path**): the sibling hash at each level from the leaf to the root.

- Tree height: `ceil(log2(n))`
- Proof size: `ceil(log2(n))` hashes = O(log n)
- Verification: `ceil(log2(n))` hash computations = O(log n)

For Bitcoin with ~2,000 transactions per block, a proof is only ~11 hashes (352 bytes) instead of all 2,000 transactions. For a tree with 1 million leaves, you need only ~20 hashes. This logarithmic scaling is what makes Merkle proofs practical.

### Handling Odd Numbers of Leaves

What if the number of leaves isn't a power of 2? There are two common strategies:

1. **Duplicate the last leaf** (Bitcoin's approach): If a level has an odd number of nodes, duplicate the last one so it can be paired with itself.
2. **Promote the unpaired node** (our approach): Carry the unpaired node up to the next level without hashing. This avoids a subtle vulnerability where duplicating can create two different trees with the same root.

We'll implement option 2 because it's more secure and forces us to handle the edge case explicitly.

### Proof Structure

A Merkle proof for leaf at index `i` is a list of `(hash, direction)` pairs:

- `direction = 'left'` means the sibling hash goes on the LEFT when concatenating
- `direction = 'right'` means the sibling hash goes on the RIGHT

To verify: start with `H(leaf_data)`, then at each level combine with the proof element according to its direction, and hash the result. If the final hash equals the root, the proof is valid.

## Step-by-Step Breakdown

### Step 1: Hash the Leaves

Convert each data block to its SHA-256 hash. These become the bottom level of the tree.

*Why*: Raw data can be any size. Hashing normalizes everything to 32 bytes and provides collision resistance.

### Step 2: Build the Tree Bottom-Up

Pair adjacent hashes and compute parent hashes. If a level has an odd count, promote the last hash to the next level.

*Why*: Each parent commits to exactly two children. Changing any leaf forces a cascade of hash changes all the way to the root.

### Step 3: Generate a Proof

Walk from the target leaf to the root, collecting the sibling hash and its position at each level.

*Why*: The verifier doesn't have the full tree. They need exactly the sibling at each level to reconstruct the path to the root.

### Step 4: Verify a Proof

Starting from `H(leaf_data)`, iteratively hash with each proof element. Compare the result to the known root.

*Why*: If ANY node in the path was tampered with, the final hash won't match. The chain of hashes is unbreakable without finding a collision.

### Step 5: Demonstrate Tamper Detection

Modify a leaf and show that the root changes and old proofs become invalid.

*Why*: This is the whole point — Merkle trees provide **tamper evidence**. It's not enough to build the tree; you need to see it catch fraud.

## Learning Objectives

- Understand how binary hash trees provide O(log n) data integrity proofs
- Implement tree construction with proper handling of odd-sized levels
- Generate and verify Merkle proofs (audit paths)
- See the connection between Merkle trees and blockchain transaction verification
- Build on the SHA-256 knowledge from Day 002

## Going Deeper

- **Merkle Patricia Tries**: Ethereum uses a more complex variant that combines Merkle proofs with a trie (prefix tree) to support key-value lookups. This is how Ethereum proves account state.
- **Sparse Merkle Trees**: Used in ZK-rollups, these represent the full 2^256 key space with efficient proofs for both inclusion AND exclusion.
- **Certificate Transparency**: Google's CT logs use Merkle trees to create an append-only, publicly auditable log of every TLS certificate issued.
- **Consistency Proofs**: Beyond audit proofs (is this leaf in the tree?), you can prove that one tree is a consistent extension of another (no history was rewritten). This is critical for append-only logs.
- **Bitcoin SPV**: Simple Payment Verification lets lightweight clients verify transactions using only block headers + Merkle proofs, without downloading the full blockchain.
