# Day 29: Hello World Solidity Contract

## What You're Building

A Python simulation of the Ethereum smart contract execution model. Instead of just writing Solidity syntax, you'll build a **mini contract virtual machine** that demonstrates how smart contracts actually work under the hood: deployment, storage, function dispatch, state mutation, events, and gas accounting.

This matters because understanding smart contracts at the execution level — not just the syntax level — is what separates developers who can write contracts from developers who can audit, optimize, and debug them. Every exploit in DeFi history traces back to someone misunderstanding execution semantics.

## Core Concepts

### 1. What Is a Smart Contract?

A smart contract is just **code stored at an address on a blockchain** that executes deterministically when called. The key properties:

- **Deterministic**: Given the same state + input, every node gets the same output. No randomness, no network calls, no filesystem access.
- **State-bearing**: Contracts have persistent storage (a key-value map) that survives between calls.
- **Trustless**: The blockchain guarantees execution — no party can prevent or alter it.

Mathematically, a contract is a state transition function:

```
f(S, msg) -> (S', output, events)
```

Where `S` is the contract's storage state, `msg` is the incoming call (sender, value, data), `S'` is the new state, and `events` are emitted logs.

### 2. Contract Deployment vs. Execution

Deployment and execution are fundamentally different operations:

- **Deployment**: The contract's bytecode is stored at a new address. The constructor runs once and initializes storage. The address is derived from `hash(deployer_address, nonce)`.
- **Execution**: A transaction targets an existing contract address. The EVM loads the contract's code and storage, runs the requested function, and writes back any storage changes.

### 3. Function Selector / Dispatch

In the EVM, function calls are encoded as:
- First 4 bytes: `keccak256(function_signature)[:4]` — the **function selector**
- Remaining bytes: ABI-encoded arguments

The contract's entry point is essentially a big if/else (or jump table) that routes based on the selector. This is why Solidity generates a dispatcher at the top of every contract.

### 4. Storage Model

EVM storage is a mapping from 256-bit keys to 256-bit values:

```
storage: uint256 -> uint256
```

Each storage slot costs gas to read (SLOAD = 2100 gas cold, 100 warm) and write (SSTORE = 20000 for zero-to-nonzero, 5000 for nonzero-to-nonzero). This cost model drives every storage optimization pattern in Solidity (packing, mappings vs arrays, etc.).

### 5. Gas Accounting

Every operation costs gas. Gas serves two purposes:
1. **DoS prevention**: Without gas, an attacker could submit an infinite loop and halt the network.
2. **Resource pricing**: Storage and computation have real costs; gas makes callers pay proportionally.

A transaction specifies a gas limit. If execution exceeds it, everything reverts — but the gas is still consumed (the miner did the work).

### 6. Events / Logs

Events are cheap write-only data structures. They're stored in transaction receipts (not contract storage), so they're much cheaper than SSTORE. Contracts emit events for off-chain indexing — this is how block explorers, The Graph, and dApps track what happened.

### 7. Access Control

The most common pattern: `msg.sender` checks. The contract records who deployed it (the owner) and restricts certain functions to that address. This is the foundation of every access control system in DeFi: `require(msg.sender == owner)`.

## Step-by-Step Breakdown

### Step 1: Define the Contract ABI

Define a contract's interface — its functions, their parameter types, and return types. This is the contract's "API schema." Without a clear ABI, callers wouldn't know how to encode their calls.

### Step 2: Build the Storage Engine

Implement a key-value store that maps slot numbers to values. Track reads and writes separately for gas accounting. This models the EVM's SLOAD/SSTORE behavior.

### Step 3: Implement Function Dispatch

Given a function name and arguments, resolve which function to execute using a selector mechanism. This teaches how the EVM routes calls — a concept invisible in Solidity but critical for understanding proxy patterns and low-level calls.

### Step 4: Build the Execution Context

Create a `msg` object carrying sender, value (ETH sent), and call data. Every function execution happens within this context. Understanding msg.sender is fundamental to all access control.

### Step 5: Implement Gas Metering

Track gas consumption per operation. If gas runs out mid-execution, revert all state changes. This teaches the atomicity guarantee: either the entire transaction succeeds, or nothing changes (except gas is spent).

### Step 6: Add Event Emission

Allow contracts to emit named events with indexed parameters. Store them in a receipt log. This demonstrates the separation between on-chain storage (expensive, queryable by contracts) and event logs (cheap, queryable only off-chain).

### Step 7: Build a Sample "Hello World" Contract

Implement a contract with:
- A `greeting` storage variable (initialized in constructor)
- `getGreeting()` — read the stored greeting
- `setGreeting(new_greeting)` — update it (owner-only)
- `Transfer` event emission
- Owner-only access control

### Step 8: Simulate a Full Lifecycle

Deploy the contract, call functions, try unauthorized access, run out of gas, and observe the event log. This end-to-end simulation ties every concept together.

## Learning Objectives

- Understand how smart contracts execute at the VM level, not just the syntax level
- Implement function dispatch using selector hashing (the mechanism behind Solidity's ABI)
- Build a gas metering system that enforces computational limits and reverts on exhaustion
- Implement the storage model (slot-based key-value) that underlies all Solidity state variables
- Understand events as cheap, write-only, off-chain-indexed data structures
- Implement owner-based access control — the pattern behind OpenZeppelin's `Ownable`
- See how deployment differs from execution and why constructors only run once

## Going Deeper

- **Proxy patterns**: Contracts can't be upgraded, but you can deploy a proxy that delegates calls to an implementation contract. Understanding dispatch is prerequisite to understanding `delegatecall` and proxy patterns (EIP-1967).
- **Storage collisions**: In proxy patterns, the proxy and implementation share storage. If their slot layouts don't match, catastrophic bugs occur. This is why EIP-1967 uses pseudo-random storage slots.
- **Reentrancy**: If a contract calls an external contract before updating its own state, the external contract can call back in. The `checks-effects-interactions` pattern prevents this. Our gas model shows why reentrancy costs extra gas (warm vs cold access).
- **Gas optimization**: In production, gas optimization is critical. Techniques include: storage packing (multiple values in one 256-bit slot), using events instead of storage for data that's only read off-chain, and minimizing SSTORE operations.
- **Formal verification**: Because contracts are deterministic state machines, they're amenable to formal verification. Tools like Certora and Halmos mathematically prove properties about contract behavior.
