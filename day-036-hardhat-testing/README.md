# Day 036: Smart Contract Testing Framework in Python

## Overview

Testing smart contracts isn't just "writing unit tests for Solidity" — it's about simulating an entire blockchain environment, managing state transitions, handling gas economics, and verifying that code controlling real money behaves exactly as specified. In production, a single untested edge case in a DeFi contract can (and has) led to hundreds of millions in losses.

Today we build a **smart contract testing framework from scratch in Python**. Instead of relying on Hardhat (the standard JS-based tool), we'll construct the core abstractions ourselves: a simulated EVM environment, account management, contract deployment, transaction execution, and assertion utilities. This teaches you what testing tools like Hardhat, Foundry, and Brownie actually do under the hood.

## Core Concepts

### 1. The EVM as a State Machine

The Ethereum Virtual Machine is a deterministic state machine. Every transaction transforms the world state:

```
S(t+1) = T(S(t), tx)
```

Where `S(t)` is the state at time `t`, `tx` is a transaction, and `T` is the state transition function. State includes:
- **Account balances** (mapping: address → wei)
- **Contract storage** (mapping: address → mapping: slot → value)
- **Contract code** (mapping: address → bytecode)
- **Nonces** (mapping: address → transaction count)

Testing means verifying that `T` produces the expected `S(t+1)` for every relevant `tx`.

### 2. ABI Encoding: The Contract Communication Protocol

Smart contracts don't receive human-readable function calls. They receive raw bytes. The **Application Binary Interface (ABI)** defines how to encode:

- **Function selector**: First 4 bytes of `keccak256("functionName(type1,type2)")`. For `transfer(address,uint256)`, the selector is `0xa9059cbb`.
- **Arguments**: Each argument is padded to 32 bytes (256 bits). Integers are big-endian, addresses are left-padded with zeros.

Example encoding of `transfer(0xABC..., 1000)`:
```
0xa9059cbb                                                          # selector
000000000000000000000000ABC0000000000000000000000000000000000000      # address (32 bytes)
00000000000000000000000000000000000000000000000000000000000003E8      # 1000 in hex (32 bytes)
```

This is why testing frameworks must handle ABI encoding/decoding — you're testing at the byte level.

### 3. Test Fixtures and State Isolation

Each test must start from a known state. In blockchain testing, this means:
- **Snapshot/Revert**: Take a snapshot of EVM state before each test, revert after. This is O(1) with copy-on-write data structures.
- **Fresh deployment**: Redeploy contracts for each test. Slower but guarantees isolation.
- **Account funding**: Each test account starts with a known ETH balance.

The tradeoff: snapshot/revert is fast but tests can leak state if you forget to revert. Fresh deployment is slow but foolproof.

### 4. Event Emission Verification

Solidity events are logged via EVM LOG opcodes (LOG0 through LOG4). Each event has:
- **Topics**: Indexed parameters (up to 3) + the event signature hash. Topics are searchable.
- **Data**: Non-indexed parameters, ABI-encoded.

Testing events means parsing transaction receipts and verifying the correct LOG entries exist with the correct topics and data.

### 5. Revert Testing

Contracts revert with `require()`, `revert()`, or `assert()`. Each produces:
- **require/revert with message**: Returns `Error(string)` selector (`0x08c379a0`) + ABI-encoded string
- **Custom errors (Solidity 0.8.4+)**: Returns the custom error selector + encoded parameters
- **assert**: Returns empty data (panic code)

A good testing framework must verify both that a transaction reverts AND that it reverts with the expected reason.

## Step-by-Step Approach

### Step 1: Build the Simulated EVM Environment
Create an in-memory blockchain state that tracks accounts, balances, contract storage, and block metadata. This is the foundation everything else builds on.

### Step 2: Implement ABI Encoding/Decoding
Build functions to encode function calls (selector + arguments) and decode return values and events. Without this, you can't communicate with contracts.

### Step 3: Contract Deployment Simulation
Implement the deployment flow: compile contract representation → generate address → store code → execute constructor → return contract instance.

### Step 4: Transaction Execution Engine
Build the transaction processor: validate sender, check balance, execute contract code (simulated), update state, generate receipt with events and gas used.

### Step 5: Assertion Utilities
Create testing helpers: `expect_revert()`, `expect_event()`, `expect_balance_change()`, `expect_storage_change()`. These are the high-level API that makes tests readable.

### Step 6: Test a Complete ERC-20 Flow
Use the framework to test a simulated ERC-20 token: deployment, transfers, approvals, and edge cases (insufficient balance, zero address, overflow).

## Learning Objectives

- Understand EVM state transitions and how testing frameworks simulate them
- Master ABI encoding/decoding at the byte level
- Learn test isolation patterns (snapshot/revert) specific to blockchain
- Build assertion utilities for events, reverts, and state changes
- Understand gas accounting and its role in testing
- Practice testing DeFi-style contracts for edge cases and security

## Going Deeper

- **Fuzz testing**: Generate random inputs to find edge cases. Foundry's `forge fuzz` does this for Solidity — building a Python equivalent teaches you property-based testing.
- **Symbolic execution**: Tools like Mythril and Manticore explore all possible execution paths. Understanding the testing framework helps you understand what symbolic execution automates.
- **Fork testing**: Hardhat can fork mainnet state. This means your tests run against real deployed contracts — crucial for testing integrations with existing DeFi protocols.
- **Gas optimization testing**: Assert that gas usage stays below thresholds. Gas regression tests catch expensive storage patterns before deployment.
- **Invariant testing**: Define properties that must always hold (e.g., "total supply equals sum of all balances") and verify them after every state change.
- **Time manipulation**: Many DeFi contracts depend on `block.timestamp` (vesting, lockups, interest accrual). Testing frameworks must let you fast-forward time.
