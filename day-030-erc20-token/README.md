# Day 030: ERC-20 Token Implementation

## Overview

Build an ERC-20 token from scratch — the standard that powers every fungible token on Ethereum (USDC, LINK, UNI, and thousands more). You'll implement the complete interface that wallets, DEXs, and DeFi protocols rely on to move value. This isn't just a smart contract exercise; ERC-20 is the API of programmable money, and understanding it deeply is essential for any serious work in DeFi.

We'll implement this in Python to focus on the **logic and state management** behind the standard, without getting tangled in Solidity syntax or EVM details. Once you understand the state machine, writing it in Solidity (which we did yesterday in Day 029) becomes mechanical.

## Core Concepts

### What Is ERC-20?

ERC-20 (Ethereum Request for Comments #20) defines a standard interface for fungible tokens. "Fungible" means every unit is interchangeable — 1 USDC is the same as any other 1 USDC, just like dollars in a bank account.

The standard specifies **6 functions** and **2 events**:

```
// Read-only
totalSupply() -> uint256
balanceOf(address) -> uint256
allowance(owner, spender) -> uint256

// State-changing
transfer(to, amount) -> bool
approve(spender, amount) -> bool
transferFrom(from, to, amount) -> bool

// Events
Transfer(from, to, amount)
Approval(owner, spender, amount)
```

### The Two-Step Transfer Pattern (approve + transferFrom)

This is the most important design pattern in ERC-20 and the one most beginners struggle with.

**Why can't contracts just call `transfer`?** Because `transfer` moves tokens from `msg.sender`. If you want a DEX contract to take your tokens, *the DEX* would need to be `msg.sender` — but you're the one initiating the transaction.

The solution is a **two-step delegation pattern**:

1. **You** call `approve(dex_address, 100)` — this says "DEX can spend up to 100 of my tokens"
2. **DEX** calls `transferFrom(you, dex, 100)` — the DEX moves tokens on your behalf

This is exactly like signing a check (approve) vs. cashing it (transferFrom). The allowance is the check amount.

```
State after approve(DEX, 100):
  allowances[you][DEX] = 100

State after DEX calls transferFrom(you, DEX, 50):
  balances[you]     -= 50
  balances[DEX]     += 50
  allowances[you][DEX] = 50   # remaining allowance
```

### The Approval Race Condition

A subtle but critical vulnerability: if Alice has approved Bob for 100 tokens and wants to change it to 50, she calls `approve(Bob, 50)`. But if Bob sees the pending transaction, he can front-run it:

1. Bob calls `transferFrom(Alice, Bob, 100)` — drains the old allowance
2. Alice's `approve(Bob, 50)` goes through
3. Bob calls `transferFrom(Alice, Bob, 50)` — drains the new allowance
4. Bob got 150 tokens instead of the intended max of 100

**Mitigation**: Always set allowance to 0 first, then to the new value. Or use `increaseAllowance`/`decreaseAllowance` (not in the original ERC-20, but added by OpenZeppelin).

### Token Decimals and Fixed-Point Arithmetic

Ethereum has no floating point. Tokens use integer amounts with a `decimals` field:

```
decimals = 18 means:
  1 token = 1_000_000_000_000_000_000 (1e18) smallest units
  0.5 tokens = 500_000_000_000_000_000 (5e17)
```

This is fixed-point arithmetic: store integers, divide by 10^decimals for display. USDC uses 6 decimals (matching cents granularity), most tokens use 18 (matching ETH's wei).

**Why this matters**: A `transfer(to, 1)` doesn't send 1 token — it sends 1 *wei-equivalent* of the token (0.000000000000000001 tokens for 18 decimals). Getting this wrong has caused real financial losses.

### Minting and Burning

The base ERC-20 standard doesn't define mint/burn, but almost every real token needs them:

- **Mint**: Create new tokens, increase totalSupply. The `from` address in the Transfer event is `address(0)`.
- **Burn**: Destroy tokens, decrease totalSupply. The `to` address is `address(0)`.

These are privileged operations — typically only the contract owner or specific roles can mint.

## Step-by-Step Breakdown

### Step 1: Token State

The entire ERC-20 is backed by three pieces of state:
- `balances: Dict[address, int]` — how many tokens each address holds
- `allowances: Dict[address, Dict[address, int]]` — how many tokens address A lets address B spend
- `total_supply: int` — sum of all balances

Plus metadata: `name`, `symbol`, `decimals`.

### Step 2: Constructor and Minting

The constructor sets metadata and mints an initial supply to the deployer. Minting is addition to `balances[to]` and `total_supply`, with a Transfer event from address(0).

### Step 3: transfer()

Move tokens from the caller to a recipient:
1. Check sender has sufficient balance (revert if not)
2. Subtract from sender's balance
3. Add to recipient's balance
4. Emit Transfer event

**Why check balance first?** Without this, underflow would create tokens from nothing. In Solidity >=0.8, this would revert automatically due to overflow checks, but we should be explicit.

### Step 4: approve()

Set the allowance for a spender:
1. Store `allowances[owner][spender] = amount`
2. Emit Approval event

That's it. No balance check needed — you're just granting permission, not moving tokens.

### Step 5: transferFrom()

The delegated transfer — most complex function:
1. Check allowance: `allowances[from][caller] >= amount`
2. Check balance: `balances[from] >= amount`
3. Subtract from sender's balance
4. Add to recipient's balance
5. Decrease allowance by amount
6. Emit Transfer event

### Step 6: Events and Logging

In a real blockchain, events are logged to the transaction receipt and indexed for off-chain queries. We simulate this with an event log that records every state change.

## Learning Objectives

- Understand the ERC-20 interface and why each function exists
- Implement the approve/transferFrom delegation pattern
- Handle fixed-point token arithmetic with decimals
- Recognize the approval race condition vulnerability
- Build minting/burning with supply tracking
- Design a clean state machine for financial token logic

## Going Deeper

- **ERC-20 Extensions**: ERC-2612 (permit — gasless approvals via signatures), ERC-4626 (tokenized vaults)
- **Real Vulnerabilities**: The `approve` race condition, missing return value checks (some tokens don't return bool), fee-on-transfer tokens that break DeFi composability
- **Gas Optimization**: Why OpenZeppelin uses `unchecked` blocks, why `transferFrom` skips allowance deduction when allowance is `type(uint256).max`
- **Connections to Previous Days**: Day 020 (ECDSA) — permits use signatures for gasless approvals. Day 029 (Solidity) — translate this to a real on-chain contract. Day 013 (Merkle trees) — how token balances are stored in Ethereum's state trie
