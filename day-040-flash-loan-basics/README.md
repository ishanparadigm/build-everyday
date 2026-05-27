# Day 040: Flash Loan Basics

## What You're Building

A complete flash loan implementation from scratch — the mechanism that lets you borrow millions of dollars with zero collateral, as long as you repay within a single transaction. Flash loans are one of DeFi's most powerful (and controversial) primitives: they enable arbitrage, liquidations, collateral swaps, and self-liquidation — but they've also been used in some of the largest exploits in crypto history.

You'll build a flash loan pool, a borrower contract that executes arbitrage, and understand the atomic transaction guarantees that make it all work.

## Core Concepts

### Atomic Transactions and the EVM Execution Model

Every Ethereum transaction is **atomic**: either every state change commits, or the entire transaction reverts. There's no partial execution. This is enforced by the EVM — when a `REVERT` opcode fires, all state changes (storage writes, ETH transfers, token moves) are rolled back, and only gas is consumed.

Flash loans exploit this guarantee:

```
1. Pool lends 1,000,000 tokens to Borrower  (state: pool -1M, borrower +1M)
2. Borrower does arbitrary operations         (arbitrage, swap, liquidation...)
3. Borrower repays 1,000,000 + fee to Pool   (state: pool +1M+fee, borrower 0)
4. Pool VERIFIES repayment                    (if balance < original + fee → REVERT)
```

If step 4 fails, steps 1-3 **never happened**. The lender has zero risk of losing funds (ignoring smart contract bugs). This is impossible in traditional finance — you can't "un-send" a wire transfer.

### The Flash Loan Protocol Pattern

The standard pattern (used by Aave, dYdX, Uniswap) follows a **callback architecture**:

1. **Borrower** calls `flashLoan(amount)` on the **Pool**
2. **Pool** transfers tokens to Borrower
3. **Pool** calls `executeOperation()` on the Borrower (the callback)
4. Inside the callback, Borrower executes their strategy
5. Borrower transfers tokens + fee back to Pool
6. **Pool** checks its balance — if insufficient, the entire tx reverts

The callback pattern is critical because the Pool must maintain control flow. If the borrower just received tokens and was expected to "call back later," there'd be no atomicity guarantee.

### The Math: Fee Calculation

Flash loan fees are typically tiny (Aave charges 0.09%, Uniswap 0.3% for swaps):

```
repayment_amount = borrowed_amount * (1 + fee_rate)
fee = borrowed_amount * fee_rate

Example: Borrow 1,000,000 USDC at 0.09% fee
Fee = 1,000,000 * 0.0009 = 900 USDC
Must repay: 1,000,900 USDC
```

The borrower only profits if their strategy generates more than the fee. For arbitrage:

```
profit = arbitrage_spread - flash_loan_fee - gas_cost
```

### Reentrancy and Security Considerations

Flash loans introduce a dangerous pattern: the pool **calls untrusted code** (the borrower's callback). This is a classic reentrancy vector. The pool must:

1. Use checks-effects-interactions pattern
2. Track the loan state before the callback
3. Verify repayment after the callback returns
4. Consider whether the borrower could manipulate price oracles during the callback

## Step-by-Step Breakdown

### Step 1: Build the Token

We need a simple ERC-20-like token to serve as the lending asset. In our Python simulation, this is a `Token` class tracking balances — the same accounting an ERC-20 contract does on-chain.

### Step 2: Build the Flash Loan Pool

The pool holds token reserves and exposes a `flash_loan()` method. Key design decisions:

- **Fee rate**: Configurable, stored as basis points for precision (9 bps = 0.09%)
- **Balance check**: Must compare post-callback balance against pre-loan balance + fee
- **Callback interface**: The borrower must implement `execute_operation(token, amount, fee, pool)`
- **Event logging**: Track all loans for transparency

### Step 3: Build the Borrower (Arbitrage Example)

The borrower implements the callback interface. In our simulation, we model a simple arbitrage between two exchanges with different prices:

1. Receive flash-loaned tokens
2. Sell on the expensive exchange
3. Buy back on the cheap exchange
4. Repay loan + fee
5. Keep the difference as profit

### Step 4: Simulate Price Discrepancies

Create two mock exchanges with slightly different prices. This models real-world arbitrage opportunities that flash loans can capture atomically.

### Step 5: Demonstrate Failure Cases

Show what happens when:
- The borrower can't repay (transaction reverts)
- The arbitrage spread is smaller than the fee (unprofitable)
- A malicious borrower tries to keep the funds

## Learning Objectives

- Understand atomic transaction guarantees and how DeFi exploits them
- Implement the callback pattern used by real flash loan protocols
- Build a fee calculation system with basis point precision
- Model arbitrage profit/loss including fees and slippage
- Understand the security implications of calling untrusted code
- See how flash loans connect to MEV, liquidations, and DeFi composability

## Going Deeper

- **Real protocols**: Study Aave's `FlashLoanSimpleReceiverBase` and Uniswap V2/V3's `flash()` function — they use this exact callback pattern
- **Flash loan attacks**: The bZx attacks (2020), Cream Finance ($130M), and Euler Finance ($197M) all involved flash loans manipulating price oracles
- **MEV and flashbots**: Flash loans are a core tool for MEV searchers who atomically capture arbitrage on-chain
- **EIP-3156**: The standardized flash loan interface for Ethereum — defines `maxFlashLoan()`, `flashFee()`, and `flashLoan()` with a `onFlashLoan` callback
- **Capital efficiency**: Flash loans prove that in a blockchain context, capital can be "free" if you can guarantee repayment — a concept that doesn't exist in traditional finance
- **Composability**: Flash loans can be nested, combined with other DeFi protocols, and used in ways their creators never anticipated — this is DeFi's "money lego" thesis in action
