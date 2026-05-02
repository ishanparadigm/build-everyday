# Day 031: ERC-721 NFT Contract

## Overview

Build an ERC-721 Non-Fungible Token (NFT) contract from scratch in Python, simulating the full standard as defined in [EIP-721](https://eips.ethereum.org/EIPS/eip-721). Unlike ERC-20 tokens where every unit is identical, each ERC-721 token is unique — it has a distinct ID and an individual owner. This is the standard that powers NFT marketplaces, on-chain gaming items, digital art, real estate deeds, and any application where you need provably unique digital ownership.

Understanding ERC-721 deeply matters because it introduces patterns you'll see everywhere in smart contract development: operator approvals, safe transfers with receiver callbacks, and metadata URIs. These patterns recur in ERC-1155, ERC-4907 (rental NFTs), and even non-token standards.

## Core Concepts

### 1. Non-Fungibility

Fungible tokens (ERC-20) are interchangeable — 1 USDC = 1 USDC. Non-fungible tokens are **not** interchangeable. Each token has a unique `token_id` and exactly one owner address. The key data structures are:

```
owners: Dict[int, str]          # token_id -> owner address
balances: Dict[str, int]        # address -> count of tokens owned
```

The `balances` mapping is technically redundant (you could count owners), but it exists for O(1) balance lookups — a gas optimization pattern you'll see throughout Ethereum.

### 2. Approval Mechanics

ERC-721 has two approval layers:

- **Per-token approval**: `approve(to, token_id)` — lets address `to` transfer one specific token. Only the owner (or an approved operator) can set this. Approval is cleared on transfer.
- **Operator approval**: `setApprovalForAll(operator, approved)` — lets `operator` manage ALL of the caller's tokens. This is what NFT marketplaces use — you approve OpenSea once, and it can transfer any of your NFTs when a sale happens.

```
token_approvals: Dict[int, str]             # token_id -> approved address
operator_approvals: Dict[(str, str), bool]  # (owner, operator) -> is_approved
```

### 3. Safe Transfers and Receiver Callbacks

The "safe" in `safeTransferFrom` means: if the recipient is a contract, call `onERC721Received()` on it and verify it returns the correct selector. This prevents tokens from being permanently locked in contracts that don't know how to handle them.

```
Interface: onERC721Received(operator, from, tokenId, data) -> bytes4
Expected return: 0x150b7a02 (the function selector)
```

Without this check, sending an NFT to a contract without transfer handling code would lock the token forever — no one could ever move it again.

### 4. Token Metadata (ERC-721Metadata Extension)

Each token can have a URI pointing to its metadata (JSON with name, description, image URL, attributes). The standard defines:

- `name()` — collection name (e.g., "Bored Ape Yacht Club")
- `symbol()` — collection symbol (e.g., "BAYC")
- `tokenURI(token_id)` — returns the metadata URI for a specific token

The metadata itself typically follows this JSON schema:
```json
{
  "name": "Token #1",
  "description": "A unique digital asset",
  "image": "ipfs://Qm.../1.png",
  "attributes": [{"trait_type": "Color", "value": "Blue"}]
}
```

### 5. Minting and Burning

Not part of the core ERC-721 standard, but nearly every implementation includes:

- **Minting**: Creating a new token with a unique ID, assigning it to an owner, and incrementing their balance. Emits a `Transfer` event from address(0).
- **Burning**: Destroying a token — clearing its owner, approvals, and decrementing the balance. Emits a `Transfer` event to address(0).

### 6. Events

ERC-721 defines three events that must be emitted:

- `Transfer(from, to, token_id)` — on every ownership change (including mint/burn)
- `Approval(owner, approved, token_id)` — when per-token approval is set
- `ApprovalForAll(owner, operator, approved)` — when operator approval changes

Events create an indexed log that off-chain systems (marketplaces, wallets, indexers) use to track ownership history.

## Step-by-Step Breakdown

### Step 1: Core State
Define the four core mappings: owners, balances, token_approvals, operator_approvals. Also track collection metadata (name, symbol) and token URIs.

### Step 2: Query Functions
Implement `balanceOf(owner)`, `ownerOf(token_id)`, `getApproved(token_id)`, `isApprovedForAll(owner, operator)`. These are the read-only functions that wallets and dApps call constantly.

### Step 3: Internal Transfer Logic
Build `_transfer(from, to, token_id)` with all the checks: valid addresses, correct ownership, balance updates, approval clearing, and event emission. This is the core — get it right and everything else is simple.

### Step 4: Approval Functions
Implement `approve(to, token_id)` and `setApprovalForAll(operator, approved)`. The key subtlety: `approve` must verify the caller is the owner OR an approved operator.

### Step 5: Transfer Functions
Build `transferFrom(from, to, token_id)` with authorization checks, and `safeTransferFrom` with the receiver callback pattern. The authorization check: caller must be owner, approved for this token, OR an approved operator.

### Step 6: Mint and Burn
Implement `mint(to, token_id)` and `burn(token_id)` as special cases of transfer (from/to the zero address). Include existence checks to prevent double-minting.

### Step 7: Metadata Extension
Add `name()`, `symbol()`, `tokenURI(token_id)`, and `setTokenURI(token_id, uri)`.

### Step 8: Enumeration (Bonus)
The ERC-721Enumerable extension tracks all tokens and per-owner token lists. This enables `totalSupply()`, `tokenByIndex(index)`, and `tokenOfOwnerByIndex(owner, index)`.

## Learning Objectives

- Understand the ERC-721 standard and how non-fungible ownership works on-chain
- Implement dual-layer approval mechanics (per-token and operator)
- Build safe transfer patterns with receiver callbacks
- Handle minting, burning, and metadata management
- Learn event-driven architecture for off-chain indexing
- Practice defensive programming with ownership and authorization checks

## Going Deeper

- **ERC-1155 (Multi-Token)**: Combines fungible and non-fungible tokens in one contract — batch transfers, shared approval, more gas-efficient for games
- **ERC-4907 (Rental NFTs)**: Adds a "user" role separate from owner — enables NFT rentals without transferring ownership
- **On-chain vs Off-chain Metadata**: Storing metadata on IPFS (content-addressed, immutable) vs centralized servers (mutable, faster) vs fully on-chain (most expensive, most durable)
- **Soulbound Tokens (ERC-5192)**: Non-transferable NFTs for credentials, reputation, identity
- **Gas Optimization**: ERC-721A batches sequential mints to save ~50% gas by deferring storage writes
- **Royalty Standard (ERC-2981)**: Adds `royaltyInfo()` so marketplaces can pay creators on secondary sales
