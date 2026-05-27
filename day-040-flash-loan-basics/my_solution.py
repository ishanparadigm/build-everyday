"""
Day 040: Flash Loan Basics — Your Implementation

Build a flash loan system from scratch. You'll implement:
1. A Token class (ERC-20 accounting)
2. A FlashLoanPool that lends with atomic guarantees
3. An ArbitrageBorrower that profits from price discrepancies
4. A MaliciousBorrower to prove the system is safe

Key concepts to remember:
- Flash loans rely on atomic transactions (all-or-nothing)
- The callback pattern gives the pool control over execution flow
- Fees are calculated in basis points (1 bp = 0.01%)
- If repayment fails, ALL state changes must be rolled back
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Optional


# ---------------------------------------------------------------------------
# Token — minimal ERC-20 accounting
# ---------------------------------------------------------------------------

class Token:
    """Minimal ERC-20 token simulation with balance tracking.

    Hint: You need a dict to track balances, and transfer() must check
    that the sender has enough before moving tokens.
    """

    def __init__(self, name: str, symbol: str, initial_supply: int, deployer: str) -> None:
        self.name = name
        self.symbol = symbol
        self.total_supply = initial_supply
        # TODO: Initialize balance tracking — give deployer the initial supply
        raise NotImplementedError("TODO: implement this")

    def balance_of(self, account: str) -> int:
        """Return the token balance for an account (0 if unknown)."""
        raise NotImplementedError("TODO: implement this")

    def transfer(self, sender: str, recipient: str, amount: int) -> None:
        """Transfer tokens from sender to recipient.

        Hint: Check balance BEFORE mutating state (checks-effects-interactions).
        Raise ValueError if insufficient balance.
        """
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Transaction context
# ---------------------------------------------------------------------------

class TransactionReverted(Exception):
    """Raised when a flash loan transaction must revert."""
    pass


@dataclass
class FlashLoanEvent:
    """Log entry for a flash loan."""
    borrower: str
    amount: int
    fee: int
    success: bool
    profit: Optional[int] = None


# ---------------------------------------------------------------------------
# Flash Loan Borrower interface
# ---------------------------------------------------------------------------

class IFlashLoanReceiver(Protocol):
    """Interface that borrowers must implement."""

    def execute_operation(
        self,
        token: Token,
        amount: int,
        fee: int,
        pool_address: str,
    ) -> bool:
        ...


# ---------------------------------------------------------------------------
# Flash Loan Pool
# ---------------------------------------------------------------------------

class FlashLoanPool:
    """Flash loan liquidity pool.

    Hint: The flash_loan() method must:
      1. Snapshot state (copy token balances dict)
      2. Transfer tokens to borrower
      3. Call borrower.execute_operation()
      4. Check pool balance >= balance_before + fee
      5. If check fails → restore snapshot and raise TransactionReverted
    """

    def __init__(self, token: Token, pool_address: str, fee_bps: int = 9) -> None:
        self.token = token
        self.address = pool_address
        self.fee_bps = fee_bps
        self.events: list[FlashLoanEvent] = []
        self.total_fees_earned: int = 0

    @property
    def available_liquidity(self) -> int:
        """How much the pool can lend."""
        raise NotImplementedError("TODO: implement this")

    def fee_for(self, amount: int) -> int:
        """Calculate fee in token units.

        Hint: Use ceiling division so the fee is never rounded down to 0.
        Formula: ceil(amount * fee_bps / 10_000)
        """
        raise NotImplementedError("TODO: implement this")

    def flash_loan(self, borrower: IFlashLoanReceiver, borrower_address: str, amount: int) -> FlashLoanEvent:
        """Execute a flash loan with atomic guarantees.

        Hint: The key insight is SNAPSHOTTING state before the loan and
        RESTORING it on failure. This simulates what the EVM does automatically.

        Steps:
          1. Validate amount > 0 and <= available liquidity
          2. Calculate fee
          3. Record balance_before
          4. Snapshot all token balances (deep copy the dict!)
          5. Transfer tokens to borrower
          6. Call borrower.execute_operation()
          7. Check: pool balance >= balance_before + fee
          8. If any step fails: restore snapshot, raise TransactionReverted
        """
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Mock Exchange
# ---------------------------------------------------------------------------

class MockExchange:
    """A simple exchange with a fixed price.

    Hint: sell_tokens transfers tokens FROM the seller TO the exchange
    and returns value = amount * price.
    buy_tokens does the reverse.
    """

    def __init__(self, name: str, token: Token, price: float, address: str) -> None:
        self.name = name
        self.token = token
        self.price = price
        self.address = address

    def sell_tokens(self, seller: str, amount: int) -> int:
        """Sell tokens to exchange, receive value. Returns value received."""
        raise NotImplementedError("TODO: implement this")

    def buy_tokens(self, buyer: str, value: int) -> int:
        """Buy tokens from exchange using value. Returns tokens received."""
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Arbitrage Borrower
# ---------------------------------------------------------------------------

class ArbitrageBorrower:
    """Flash loan borrower that executes cross-exchange arbitrage.

    Hint: In execute_operation:
      1. Determine which exchange has higher/lower price
      2. Sell ALL borrowed tokens on the expensive exchange
      3. Buy tokens back on the cheap exchange
      4. Repay loan + fee to the pool
      5. Return True if successful
    """

    def __init__(self, address: str, exchange_a: MockExchange, exchange_b: MockExchange) -> None:
        self.address = address
        self.exchange_a = exchange_a
        self.exchange_b = exchange_b

    def execute_operation(
        self,
        token: Token,
        amount: int,
        fee: int,
        pool_address: str,
    ) -> bool:
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Malicious Borrower
# ---------------------------------------------------------------------------

class MaliciousBorrower:
    """A borrower that tries to steal funds by not repaying.

    Hint: Just... don't repay. Return True anyway. Watch what happens.
    """

    def __init__(self, address: str) -> None:
        self.address = address

    def execute_operation(
        self,
        token: Token,
        amount: int,
        fee: int,
        pool_address: str,
    ) -> bool:
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Test your implementation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Setting up flash loan simulation...\n")

    # Create token and distribute
    token = Token("USD Coin", "USDC", 10_000_000, "deployer")
    token.transfer("deployer", "flash_pool", 5_000_000)
    token.transfer("deployer", "exchange_a", 2_000_000)
    token.transfer("deployer", "exchange_b", 2_000_000)

    # Create pool
    pool = FlashLoanPool(token, "flash_pool", fee_bps=9)
    print(f"Pool liquidity: {pool.available_liquidity:,}")
    print(f"Fee: {pool.fee_bps} bps")

    # Test 1: Profitable arbitrage
    print("\n--- Test 1: Profitable Arbitrage ---")
    ex_a = MockExchange("UniSwap", token, price=1.02, address="exchange_a")
    ex_b = MockExchange("SushiSwap", token, price=1.00, address="exchange_b")
    arb = ArbitrageBorrower("arb1", ex_a, ex_b)

    event = pool.flash_loan(arb, "arb1", 1_000_000)
    print(f"Success: {event.success}, Profit: {event.profit}")

    # Test 2: Malicious borrower
    print("\n--- Test 2: Malicious Borrower ---")
    thief = MaliciousBorrower("thief")
    try:
        pool.flash_loan(thief, "thief", 1_000_000)
        print("ERROR: Should have reverted!")
    except TransactionReverted:
        print("Correctly reverted — theft prevented!")
        print(f"Pool balance unchanged: {pool.available_liquidity:,}")

    print("\nAll tests passed!")
