# Day 039: Multisig Wallet

## Overview

Build a multi-signature wallet contract from scratch — a smart contract that requires M-of-N owners to approve a transaction before it executes. Multisig wallets are the backbone of treasury security in crypto: Gnosis Safe alone secures over $100B in assets. Every serious DeFi protocol, DAO treasury, and institutional custodian uses some form of multisig.

The key insight: instead of a single private key being a catastrophic single point of failure, a multisig distributes trust across multiple parties. Even if one key is compromised, an attacker cannot drain funds without colluding with enough other signers to reach the threshold.

## Core Concepts

### M-of-N Threshold Signatures

A multisig wallet has N total owners and requires M confirmations (where 1 ≤ M ≤ N) for any transaction to execute. Common configurations:

- **2-of-3**: Personal security (you, hardware wallet, backup) — lose one key and you can still recover
- **3-of-5**: DAO treasury — prevents any small faction from unilaterally moving funds
- **4-of-7**: Protocol governance — high security for upgrade keys

The threshold M is a critical security parameter:
- Too low (1-of-N): barely better than a single signer
- Too high (N-of-N): a single lost key bricks the wallet forever
- Sweet spot: `ceil(N/2) + 1` gives majority control while tolerating one lost key

### Transaction Lifecycle

A multisig transaction goes through a state machine:

```
PROPOSED → CONFIRMED (gathering sigs) → EXECUTED
                                      → REVOKED (if an owner withdraws confirmation)
```

Each transaction stores:
- **destination**: the target address
- **value**: ETH amount to send
- **data**: calldata for contract interactions (e.g., ERC-20 transfer)
- **confirmations**: set of owners who have approved
- **executed**: whether it has already been sent

### Nonce and Replay Protection

Every transaction gets a sequential nonce. This prevents:
1. **Replay attacks**: executing the same transaction twice
2. **Ordering ambiguity**: owners know exactly which transaction they're signing

In on-chain multisigs (like Gnosis Safe), the nonce is stored in contract state. In off-chain signature aggregation schemes, the nonce is part of the signed message hash.

### Access Control Patterns

The wallet enforces multiple layers of access control:
- **onlyOwner**: only registered owners can submit/confirm
- **onlyWallet**: certain admin functions (add/remove owner, change threshold) can only be called by the wallet itself — meaning they must go through the multisig approval process
- **notExecuted**: prevents double-execution
- **notConfirmed/confirmed**: prevents double-voting and ensures threshold is met

### The "Self-Call" Pattern

Admin operations (adding owners, changing threshold) are executed as transactions *to the wallet itself*. This is elegant: the same M-of-N approval process that protects fund transfers also protects configuration changes. No special admin role needed — governance changes are just transactions.

### Gas Considerations and Execution

When the final confirming owner calls `executeTransaction`:
- The wallet performs a low-level `call` with the stored value and data
- If the call reverts, the transaction is marked as failed (not executed), allowing retry
- Gas for execution is paid by whoever submits the final confirmation

## Step-by-Step Breakdown

### Step 1: Owner Management
Initialize the wallet with a list of owner addresses and the required confirmation threshold. Store owners in both a list (for enumeration) and a mapping (for O(1) lookup). Validate: no zero addresses, no duplicates, threshold in valid range.

### Step 2: Transaction Submission
Any owner can propose a transaction by specifying destination, value, and data. The transaction gets a sequential ID and is stored with zero confirmations. The submitter's confirmation is NOT automatically added — they must explicitly confirm (keeps the logic clean and auditable).

### Step 3: Confirmation Collection
Owners confirm transactions by ID. Each owner can only confirm once per transaction. Track confirmations in a nested mapping: `confirmations[txId][owner] = true`. Count confirmations to check against threshold.

### Step 4: Execution
Once a transaction has M confirmations, any owner can trigger execution. The wallet performs a low-level call. If it succeeds, mark as executed. If it reverts, the transaction remains unexecuted so owners can retry or revoke.

### Step 5: Revocation
Owners can withdraw their confirmation before execution. This handles the case where an owner confirms but later realizes the transaction is problematic. The confirmation count decreases accordingly.

### Step 6: Admin Operations via Self-Call
Implement addOwner, removeOwner, and changeThreshold as functions that can only be called by the wallet itself. To add an owner, someone must submit a transaction targeting the wallet address with the appropriate calldata, and M owners must confirm it.

## Learning Objectives

- Implement M-of-N threshold approval logic with proper access control
- Understand transaction lifecycle management (propose → confirm → execute)
- Apply the self-call pattern for governance-protected admin operations
- Handle edge cases: duplicate confirmations, executed transactions, threshold changes
- Build production-grade guard modifiers for contract security

## Going Deeper

- **Off-chain signatures**: Gnosis Safe collects ECDSA signatures off-chain and submits them in one transaction, saving gas. Each signer signs a typed data hash (EIP-712) containing the transaction details.
- **Time locks**: Many multisigs add a delay between confirmation and execution, giving owners a window to veto malicious transactions.
- **Delegate calls**: Gnosis Safe supports `delegatecall`, allowing the multisig to execute arbitrary logic in its own context — powerful but dangerous.
- **Account abstraction (ERC-4337)**: The next evolution — multisig logic moves into "user operation" validation, enabling gasless transactions and social recovery.
- **Key rotation**: Production multisigs need owner replacement without changing the wallet address, which this implementation supports via the self-call pattern.
