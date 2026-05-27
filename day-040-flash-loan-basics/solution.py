"""
Day 040: Flash Loan Basics — Complete Implementation

Simulates flash loans with atomic transaction guarantees, callback-based
borrowing, arbitrage execution, and fee accounting. Models the exact
pattern used by Aave, dYdX, and Uniswap.
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

    On-chain, this would be a Solidity contract with `mapping(address => uint256)`.
    We use a Python dict keyed by string addresses for the same effect.
    """

    def __init__(self, name: str, symbol: str, initial_supply: int, deployer: str) -> None:
        self.name = name
        self.symbol = symbol
        # Balances are stored as integers (like on-chain, where ERC-20s use uint256)
        # In production, amounts are scaled by decimals (e.g., 1 USDC = 1_000_000)
        self._balances: dict[str, int] = {deployer: initial_supply}
        self.total_supply = initial_supply

    def balance_of(self, account: str) -> int:
        return self._balances.get(account, 0)

    def transfer(self, sender: str, recipient: str, amount: int) -> None:
        """Transfer tokens between accounts.

        Mirrors ERC-20 `transfer()` — reverts (raises) if insufficient balance.
        We check balance BEFORE mutating state (checks-effects-interactions pattern).
        """
        if amount < 0:
            raise ValueError("Transfer amount must be non-negative")
        if self.balance_of(sender) < amount:
            raise ValueError(
                f"Insufficient balance: {sender} has {self.balance_of(sender)} "
                f"but tried to send {amount} {self.symbol}"
            )
        self._balances[sender] = self.balance_of(sender) - amount
        self._balances[recipient] = self.balance_of(recipient) + amount

    def __repr__(self) -> str:
        return f"Token({self.name}, supply={self.total_supply})"


# ---------------------------------------------------------------------------
# Transaction context — simulates EVM atomic execution
# ---------------------------------------------------------------------------

class TransactionReverted(Exception):
    """Raised when a flash loan transaction must revert.

    On the EVM, this triggers the REVERT opcode — all state changes are rolled back.
    In our simulation, we snapshot state before the loan and restore on failure.
    """
    pass


@dataclass
class FlashLoanEvent:
    """Log entry for a flash loan — mirrors Solidity events."""
    borrower: str
    amount: int
    fee: int
    success: bool
    profit: Optional[int] = None


# ---------------------------------------------------------------------------
# Flash Loan Borrower interface — the callback contract
# ---------------------------------------------------------------------------

class IFlashLoanReceiver(Protocol):
    """Interface that borrowers must implement.

    This is the Python equivalent of Aave's `IFlashLoanSimpleReceiver` or
    EIP-3156's `IERC3156FlashBorrower`. The pool calls this after transferring
    tokens — the borrower executes their strategy inside this callback.
    """

    def execute_operation(
        self,
        token: Token,
        amount: int,
        fee: int,
        pool_address: str,
    ) -> bool:
        """Called by the pool after tokens are transferred.

        Args:
            token: The borrowed token
            amount: Amount borrowed
            fee: Fee that must be paid on top of the borrowed amount
            pool_address: Address of the lending pool (to repay)

        Returns:
            True if the operation succeeded and repayment was made
        """
        ...


# ---------------------------------------------------------------------------
# Flash Loan Pool — the lending protocol
# ---------------------------------------------------------------------------

class FlashLoanPool:
    """Flash loan liquidity pool.

    Holds token reserves and lends them out for single-transaction use.
    Follows the Aave/EIP-3156 pattern:
      1. Transfer tokens to borrower
      2. Call borrower's callback
      3. Verify repayment (revert if insufficient)

    Fee is specified in basis points (1 bp = 0.01%).
    Aave uses 9 bps (0.09%), we default to the same.
    """

    def __init__(self, token: Token, pool_address: str, fee_bps: int = 9) -> None:
        self.token = token
        self.address = pool_address
        self.fee_bps = fee_bps  # basis points: 9 = 0.09%
        self.events: list[FlashLoanEvent] = []
        self.total_fees_earned: int = 0

    @property
    def available_liquidity(self) -> int:
        """How much the pool can lend — its entire token balance."""
        return self.token.balance_of(self.address)

    def fee_for(self, amount: int) -> int:
        """Calculate fee in token units.

        Uses integer math with ceiling division to avoid rounding down to zero
        on small amounts. On-chain, Solidity does the same with:
            fee = (amount * feeBps + 9999) / 10000
        We use math.ceil for clarity.
        """
        # fee = ceil(amount * fee_bps / 10_000)
        # Ceiling ensures the pool always gets at least 1 unit on non-zero amounts
        return math.ceil(amount * self.fee_bps / 10_000)

    def flash_loan(self, borrower: IFlashLoanReceiver, borrower_address: str, amount: int) -> FlashLoanEvent:
        """Execute a flash loan with atomic guarantees.

        This is the core function. The sequence mirrors on-chain execution:
          1. Snapshot state (for rollback on failure)
          2. Transfer tokens to borrower
          3. Call borrower's executeOperation callback
          4. Verify repayment
          5. If verification fails → revert (restore snapshot)

        Args:
            borrower: Contract implementing IFlashLoanReceiver
            borrower_address: Address identifier for the borrower
            amount: Number of tokens to borrow

        Returns:
            FlashLoanEvent with details of the loan

        Raises:
            TransactionReverted: If repayment is insufficient (simulates EVM revert)
        """
        # --- Checks ---
        if amount <= 0:
            raise ValueError("Flash loan amount must be positive")
        if amount > self.available_liquidity:
            raise ValueError(
                f"Insufficient pool liquidity: requested {amount}, "
                f"available {self.available_liquidity}"
            )

        fee = self.fee_for(amount)
        balance_before = self.available_liquidity

        # --- Snapshot state for atomic rollback ---
        # On the EVM, the VM handles this automatically. In Python, we manually
        # snapshot all token balances so we can restore them if the loan fails.
        snapshot = dict(self.token._balances)

        try:
            # --- Step 1: Transfer tokens to borrower ---
            # This is the "effects" part — state is mutated before the external call.
            # The pool trusts the atomicity guarantee to protect against loss.
            print(f"  [Pool] Lending {amount} {self.token.symbol} to {borrower_address}")
            print(f"  [Pool] Fee: {fee} {self.token.symbol} ({self.fee_bps} bps)")
            self.token.transfer(self.address, borrower_address, amount)

            # --- Step 2: Call borrower's callback (INTERACTION) ---
            # This is the dangerous part — we're calling untrusted external code.
            # The borrower can do ANYTHING here: arbitrage, liquidations, oracle
            # manipulation, nested flash loans, etc.
            print(f"  [Pool] Calling executeOperation on {borrower_address}...")
            success = borrower.execute_operation(
                token=self.token,
                amount=amount,
                fee=fee,
                pool_address=self.address,
            )

            if not success:
                raise TransactionReverted("Borrower's executeOperation returned False")

            # --- Step 3: Verify repayment ---
            # The pool checks its balance AFTER the callback returns.
            # It must have at least (balance_before + fee) tokens.
            balance_after = self.available_liquidity
            required = balance_before + fee

            print(f"  [Pool] Verifying repayment: balance_before={balance_before}, "
                  f"balance_after={balance_after}, required={required}")

            if balance_after < required:
                raise TransactionReverted(
                    f"Flash loan not repaid: pool has {balance_after} but needs {required}. "
                    f"Shortfall: {required - balance_after} {self.token.symbol}"
                )

            # --- Success: record the loan ---
            self.total_fees_earned += fee
            borrower_profit = self.token.balance_of(borrower_address) - 0  # net gain
            event = FlashLoanEvent(
                borrower=borrower_address,
                amount=amount,
                fee=fee,
                success=True,
                profit=borrower_profit,
            )
            self.events.append(event)
            print(f"  [Pool] Flash loan successful! Fee earned: {fee} {self.token.symbol}")
            return event

        except (TransactionReverted, ValueError) as e:
            # --- REVERT: Restore state snapshot ---
            # On the EVM, this happens automatically when REVERT fires.
            # All storage writes, token transfers, and ETH movements are undone.
            print(f"  [Pool] TRANSACTION REVERTED: {e}")
            print(f"  [Pool] Rolling back all state changes...")
            self.token._balances = snapshot

            event = FlashLoanEvent(
                borrower=borrower_address,
                amount=amount,
                fee=fee,
                success=False,
            )
            self.events.append(event)
            # Wrap all failures as TransactionReverted — on the EVM, any
            # failure during execution triggers REVERT regardless of cause.
            if isinstance(e, TransactionReverted):
                raise
            raise TransactionReverted(str(e)) from e


# ---------------------------------------------------------------------------
# Mock Exchange — simulates a DEX with a price
# ---------------------------------------------------------------------------

class MockExchange:
    """A simple exchange with a fixed price for token/ETH swaps.

    Models a DEX like Uniswap, but with a fixed price for simplicity.
    In reality, DEXs use AMM curves (x*y=k) which cause slippage.
    The price difference between two exchanges is the arbitrage opportunity.
    """

    def __init__(self, name: str, token: Token, price: float, address: str) -> None:
        """
        Args:
            name: Exchange name (e.g., "Uniswap", "Sushiswap")
            token: The token being traded
            price: How many "units of value" per token (e.g., 1.00 = $1 per token)
            address: Address that holds the exchange's token reserves
        """
        self.name = name
        self.token = token
        self.price = price  # value per token
        self.address = address

    def sell_tokens(self, seller: str, amount: int) -> int:
        """Sell tokens to this exchange, receive value units.

        Returns the value received (in abstract "value units").
        The exchange takes the tokens and gives back value.
        """
        self.token.transfer(seller, self.address, amount)
        value = int(amount * self.price)
        return value

    def buy_tokens(self, buyer: str, value: int) -> int:
        """Buy tokens from this exchange using value units.

        Returns the number of tokens received.
        The exchange gives tokens and takes value.
        """
        token_amount = int(value / self.price)
        self.token.transfer(self.address, buyer, token_amount)
        return token_amount


# ---------------------------------------------------------------------------
# Arbitrage Borrower — profits from price discrepancies
# ---------------------------------------------------------------------------

class ArbitrageBorrower:
    """Flash loan borrower that executes cross-exchange arbitrage.

    Strategy:
      1. Receive borrowed tokens from pool
      2. Sell tokens on the EXPENSIVE exchange (higher price → more value)
      3. Buy tokens back on the CHEAP exchange (lower price → more tokens)
      4. Repay the loan + fee
      5. Keep the difference as profit

    This only works when:
      price_spread > flash_loan_fee + gas_cost
    """

    def __init__(
        self,
        address: str,
        exchange_a: MockExchange,
        exchange_b: MockExchange,
    ) -> None:
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
        """Callback from the flash loan pool — execute the arbitrage.

        This function is called AFTER we've received the borrowed tokens.
        We must repay amount + fee before returning, or the tx reverts.
        """
        print(f"    [Arbitrageur] Received {amount} {token.symbol}")
        print(f"    [Arbitrageur] Must repay {amount + fee} {token.symbol}")

        # Determine which exchange is expensive (sell there) and which is cheap (buy there)
        if self.exchange_a.price > self.exchange_b.price:
            sell_exchange = self.exchange_a
            buy_exchange = self.exchange_b
        else:
            sell_exchange = self.exchange_b
            buy_exchange = self.exchange_a

        print(f"    [Arbitrageur] Selling on {sell_exchange.name} @ {sell_exchange.price}")
        print(f"    [Arbitrageur] Buying on {buy_exchange.name} @ {buy_exchange.price}")

        # Step 1: Sell all borrowed tokens on the expensive exchange
        value_received = sell_exchange.sell_tokens(self.address, amount)
        print(f"    [Arbitrageur] Sold {amount} tokens for {value_received} value")

        # Step 2: Buy back tokens on the cheap exchange
        tokens_bought = buy_exchange.buy_tokens(self.address, value_received)
        print(f"    [Arbitrageur] Bought {tokens_bought} tokens for {value_received} value")

        # Step 3: Repay the loan + fee
        repay_amount = amount + fee
        my_balance = token.balance_of(self.address)
        print(f"    [Arbitrageur] Balance: {my_balance}, need to repay: {repay_amount}")

        if my_balance < repay_amount:
            print(f"    [Arbitrageur] INSUFFICIENT FUNDS — cannot repay!")
            return False

        token.transfer(self.address, pool_address, repay_amount)
        profit = token.balance_of(self.address)
        print(f"    [Arbitrageur] Repaid {repay_amount}, profit: {profit} {token.symbol}")

        return True


# ---------------------------------------------------------------------------
# Malicious Borrower — tries to keep the funds (will fail)
# ---------------------------------------------------------------------------

class MaliciousBorrower:
    """A borrower that attempts to steal flash-loaned funds.

    This demonstrates WHY flash loans are safe for lenders: the borrower
    simply doesn't repay, so the pool's balance check fails, and the
    entire transaction reverts — the tokens were never actually sent.
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
        print(f"    [Malicious] Received {amount} {token.symbol}... keeping them!")
        print(f"    [Malicious] Not repaying. Let's see what happens...")
        # Deliberately not repaying — the pool will revert the entire transaction
        return True  # Claims success but didn't actually repay


# ---------------------------------------------------------------------------
# Main — demonstrate flash loans in action
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("FLASH LOAN BASICS — Day 040")
    print("=" * 70)

    # --- Setup: Create token and distribute to pool + exchanges ---
    TOTAL_SUPPLY = 10_000_000  # 10M tokens
    token = Token("USD Coin", "USDC", TOTAL_SUPPLY, "deployer")

    # Fund the flash loan pool with 5M tokens
    token.transfer("deployer", "flash_pool", 5_000_000)

    # Fund two exchanges with different amounts
    token.transfer("deployer", "exchange_a", 2_000_000)
    token.transfer("deployer", "exchange_b", 2_000_000)

    print(f"\nToken: {token.name} ({token.symbol})")
    print(f"Total supply: {TOTAL_SUPPLY:,}")
    print(f"Pool balance: {token.balance_of('flash_pool'):,}")
    print(f"Exchange A balance: {token.balance_of('exchange_a'):,}")
    print(f"Exchange B balance: {token.balance_of('exchange_b'):,}")

    # --- Create the flash loan pool ---
    pool = FlashLoanPool(token, pool_address="flash_pool", fee_bps=9)
    print(f"\nFlash Loan Pool: fee = {pool.fee_bps} bps ({pool.fee_bps/100:.2f}%)")

    # ===================================================================
    # Scenario 1: Profitable Arbitrage
    # ===================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 1: Profitable Arbitrage (2% price spread)")
    print("=" * 70)

    # Exchange A sells at $1.02, Exchange B sells at $1.00
    # Arbitrage: buy cheap on B, sell expensive on A
    exchange_a = MockExchange("UniSwap", token, price=1.02, address="exchange_a")
    exchange_b = MockExchange("SushiSwap", token, price=1.00, address="exchange_b")

    arb = ArbitrageBorrower("arbitrageur", exchange_a, exchange_b)

    borrow_amount = 1_000_000  # Borrow 1M tokens
    print(f"\nBorrowing {borrow_amount:,} {token.symbol}...")
    print(f"Expected fee: {pool.fee_for(borrow_amount):,} {token.symbol}")
    print(f"Price spread: {abs(exchange_a.price - exchange_b.price) / min(exchange_a.price, exchange_b.price) * 100:.2f}%")
    print()

    event = pool.flash_loan(arb, "arbitrageur", borrow_amount)
    print(f"\n  Result: {'SUCCESS' if event.success else 'FAILED'}")
    print(f"  Profit: {event.profit:,} {token.symbol}")
    print(f"  Pool fees earned: {pool.total_fees_earned:,} {token.symbol}")
    print(f"  Pool balance: {pool.available_liquidity:,} {token.symbol}")

    # ===================================================================
    # Scenario 2: Unprofitable Arbitrage (spread < fee)
    # ===================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 2: Unprofitable Arbitrage (0.05% spread < 0.09% fee)")
    print("=" * 70)

    # Reset borrower balance
    leftover = token.balance_of("arbitrageur2")
    # Create exchanges with tiny spread
    exchange_c = MockExchange("Exchange C", token, price=1.0005, address="exchange_a")
    exchange_d = MockExchange("Exchange D", token, price=1.0000, address="exchange_b")

    arb2 = ArbitrageBorrower("arbitrageur2", exchange_c, exchange_d)

    print(f"\nBorrowing {borrow_amount:,} {token.symbol}...")
    print(f"Expected fee: {pool.fee_for(borrow_amount):,} {token.symbol}")
    print(f"Price spread: {abs(exchange_c.price - exchange_d.price) / min(exchange_c.price, exchange_d.price) * 100:.4f}%")
    print(f"Fee rate: {pool.fee_bps / 100:.2f}%")
    print(f"Spread < Fee → This will REVERT")
    print()

    try:
        pool.flash_loan(arb2, "arbitrageur2", borrow_amount)
    except TransactionReverted:
        print(f"\n  Result: REVERTED (as expected)")
        print(f"  Pool balance unchanged: {pool.available_liquidity:,} {token.symbol}")
        print(f"  Arbitrageur balance: {token.balance_of('arbitrageur2'):,} (unchanged)")

    # ===================================================================
    # Scenario 3: Malicious Borrower (tries to steal funds)
    # ===================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 3: Malicious Borrower (doesn't repay)")
    print("=" * 70)

    thief = MaliciousBorrower("thief")
    pool_balance_before = pool.available_liquidity
    print(f"\nPool balance before: {pool_balance_before:,}")
    print(f"Thief tries to borrow {borrow_amount:,} without repaying...")
    print()

    try:
        pool.flash_loan(thief, "thief", borrow_amount)
    except TransactionReverted:
        print(f"\n  Result: REVERTED — theft prevented by atomic guarantees!")
        print(f"  Pool balance after: {pool.available_liquidity:,} (unchanged)")
        print(f"  Thief balance: {token.balance_of('thief'):,} (got nothing)")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("FLASH LOAN POOL SUMMARY")
    print("=" * 70)
    print(f"Total loans attempted: {len(pool.events)}")
    print(f"Successful: {sum(1 for e in pool.events if e.success)}")
    print(f"Reverted: {sum(1 for e in pool.events if not e.success)}")
    print(f"Total fees earned: {pool.total_fees_earned:,} {token.symbol}")
    print(f"Current pool balance: {pool.available_liquidity:,} {token.symbol}")

    print("\nLoan history:")
    for i, event in enumerate(pool.events, 1):
        status = "OK" if event.success else "REVERTED"
        print(f"  {i}. {event.borrower}: borrowed {event.amount:,}, "
              f"fee {event.fee:,}, status={status}"
              + (f", profit={event.profit:,}" if event.profit is not None else ""))

    # ===================================================================
    # Key Insight
    # ===================================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Flash loans are only possible because blockchains provide atomic transactions.
In traditional finance, lending requires collateral because you can't "undo"
a wire transfer. On Ethereum, if the borrower doesn't repay, the entire
transaction reverts — the loan never happened. This means:

  1. ZERO risk for lenders (the funds literally can't be lost*)
  2. ZERO collateral needed from borrowers
  3. Anyone can borrow ANY amount for a single transaction
  4. The only cost is gas + a tiny fee (0.09% on Aave)

*Assuming the smart contract is bug-free — which is a big assumption.
Smart contract bugs have led to hundreds of millions in losses.

This is why flash loans are one of DeFi's most powerful primitives:
they democratize access to capital for atomic operations.
""")


if __name__ == "__main__":
    main()
