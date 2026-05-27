"""
Day 037: DEX Swap Contract — Your Implementation

Build a constant-product Automated Market Maker (AMM) from scratch.
This is the core mechanism behind Uniswap-style decentralized exchanges.

Key formula: x * y = k

Run tests with: python3 -m pytest tests.py -v
"""

import math
from typing import Optional


class Token:
    """
    Simulates an ERC-20 token with balances and transfer mechanics.

    Tracks balances as a dictionary: address (string) -> amount (float).
    Supports mint, burn, transfer, and balance queries.
    """

    def __init__(self, name: str, symbol: str, initial_supply: float = 0.0):
        self.name = name
        self.symbol = symbol
        self.balances: dict[str, float] = {}
        self.total_supply = initial_supply

    def mint(self, to: str, amount: float) -> None:
        """Create new tokens and assign to an address."""
        raise NotImplementedError("TODO: implement this")

    def burn(self, from_addr: str, amount: float) -> None:
        """Destroy tokens from an address. Raise ValueError if insufficient balance."""
        raise NotImplementedError("TODO: implement this")

    def balance_of(self, addr: str) -> float:
        """Return balance of an address (0.0 if not found)."""
        raise NotImplementedError("TODO: implement this")

    def transfer(self, from_addr: str, to_addr: str, amount: float) -> None:
        """Transfer tokens between addresses. Raise ValueError if insufficient balance."""
        raise NotImplementedError("TODO: implement this")


class LiquidityPool:
    """
    Constant-product AMM (x * y = k).

    Hint: The core insight is that after every swap, reserve_a * reserve_b
    must be >= what it was before (it grows slightly due to fees).
    """

    POOL_ADDRESS = "pool"

    def __init__(
        self,
        token_a: Token,
        token_b: Token,
        fee_rate: float = 0.003,
    ):
        self.token_a = token_a
        self.token_b = token_b
        self.fee_rate = fee_rate
        self.reserve_a: float = 0.0
        self.reserve_b: float = 0.0
        self.lp_token = Token("LP Token", "LP")
        self.total_fees_a: float = 0.0
        self.total_fees_b: float = 0.0
        self.trades: list[dict] = []

    @property
    def k(self) -> float:
        """The constant product invariant."""
        return self.reserve_a * self.reserve_b

    def spot_price_a_in_b(self) -> float:
        """
        Price of token A denominated in token B (marginal/spot price).
        Hint: Think about what ratio of reserves gives you the price.
        """
        raise NotImplementedError("TODO: implement this")

    def spot_price_b_in_a(self) -> float:
        """Price of token B denominated in token A."""
        raise NotImplementedError("TODO: implement this")

    def add_liquidity(
        self,
        provider: str,
        amount_a: float,
        amount_b: float,
    ) -> tuple[float, float, float]:
        """
        Add liquidity to the pool and receive LP tokens.

        First deposit: any ratio accepted, LP tokens = sqrt(a * b)
        Subsequent: must match current ratio, use min() of both ratios for LP calc

        Hint: For subsequent deposits, calculate the optimal amount of one token
        based on the other and the current reserve ratio. The actual deposited
        amounts may differ from what was requested.

        Returns: (actual_a_deposited, actual_b_deposited, lp_tokens_minted)
        """
        raise NotImplementedError("TODO: implement this")

    def remove_liquidity(
        self,
        provider: str,
        lp_amount: float,
    ) -> tuple[float, float]:
        """
        Burn LP tokens and withdraw proportional reserves.

        Hint: Calculate the provider's share as lp_amount / total_lp_supply,
        then return that fraction of both reserves.

        Returns: (amount_a_withdrawn, amount_b_withdrawn)
        """
        raise NotImplementedError("TODO: implement this")

    def get_output_amount(
        self,
        input_amount: float,
        input_reserve: float,
        output_reserve: float,
    ) -> tuple[float, float]:
        """
        Calculate output amount using constant product formula with fees.

        Steps:
        1. Calculate fee: input_amount * fee_rate
        2. input_after_fee = input_amount - fee
        3. output = output_reserve * input_after_fee / (input_reserve + input_after_fee)

        Hint: This formula comes from solving (x + dx)(y - dy) = x*y for dy.

        Returns: (output_amount, fee_amount)
        """
        raise NotImplementedError("TODO: implement this")

    def swap_a_for_b(
        self,
        trader: str,
        amount_a_in: float,
        min_b_out: float = 0.0,
    ) -> float:
        """
        Swap token A for token B.

        Steps:
        1. Validate inputs (positive amount, pool has liquidity)
        2. Calculate output using get_output_amount
        3. Check slippage protection (output >= min_b_out)
        4. Transfer tokens and update reserves
        5. Verify k didn't decrease

        Returns: amount of token B received
        """
        raise NotImplementedError("TODO: implement this")

    def swap_b_for_a(
        self,
        trader: str,
        amount_b_in: float,
        min_a_out: float = 0.0,
    ) -> float:
        """
        Swap token B for token A. Mirror of swap_a_for_b.

        Returns: amount of token A received
        """
        raise NotImplementedError("TODO: implement this")

    def calculate_impermanent_loss(self, price_ratio: float) -> float:
        """
        Calculate impermanent loss for a given price change ratio.

        Formula: 2 * sqrt(price_ratio) / (1 + price_ratio) - 1

        price_ratio: new_price / original_price (e.g., 2.0 means price doubled)
        Returns: a negative number representing the loss vs holding
        """
        raise NotImplementedError("TODO: implement this")

    def pool_status(self) -> str:
        """Return human-readable pool status string."""
        return (
            f"Pool Status:\n"
            f"  {self.token_a.symbol}: {self.reserve_a:.4f}\n"
            f"  {self.token_b.symbol}: {self.reserve_b:.4f}\n"
            f"  k = {self.k:.4f}\n"
            f"  Price: 1 {self.token_a.symbol} = {self.spot_price_a_in_b():.4f} {self.token_b.symbol}\n"
            f"  LP Supply: {self.lp_token.total_supply:.4f}"
        )


def calculate_arbitrage_opportunity(
    pool: LiquidityPool,
    external_price_a_in_b: float,
) -> Optional[dict]:
    """
    Calculate optimal arbitrage when pool price differs from external market.

    Hint: If pool price < external price, buy A from pool (swap B for A).
    Use sqrt(k / price) and sqrt(k * price) to find target reserves.
    Account for fees when calculating the input amount.

    Returns: dict with trade direction, amounts, and expected profit, or None
    """
    raise NotImplementedError("TODO: implement this")


if __name__ == "__main__":
    print("DEX Swap Contract — Testing your implementation")
    print("=" * 50)

    # Setup tokens
    eth = Token("Ethereum", "ETH")
    usdc = Token("USD Coin", "USDC")

    eth.mint("alice", 100.0)
    usdc.mint("alice", 200_000.0)
    eth.mint("bob", 10.0)
    usdc.mint("bob", 5_000.0)

    pool = LiquidityPool(eth, usdc, fee_rate=0.003)

    # Test 1: Add initial liquidity
    print("\n1. Adding initial liquidity (50 ETH + 100,000 USDC)...")
    a, b, lp = pool.add_liquidity("alice", 50.0, 100_000.0)
    print(f"   LP tokens: {lp:.4f}")
    print(pool.pool_status())

    # Test 2: Swap
    print("\n2. Bob swaps 1 ETH for USDC...")
    usdc_out = pool.swap_a_for_b("bob", 1.0)
    print(f"   Received: {usdc_out:.4f} USDC")
    print(f"   Effective price: {usdc_out:.2f} USDC/ETH")

    # Test 3: Reverse swap
    print("\n3. Bob swaps 1000 USDC for ETH...")
    eth_out = pool.swap_b_for_a("bob", 1000.0)
    print(f"   Received: {eth_out:.4f} ETH")

    # Test 4: Remove liquidity
    print("\n4. Alice removes liquidity...")
    alice_lp = pool.lp_token.balance_of("alice")
    eth_back, usdc_back = pool.remove_liquidity("alice", alice_lp)
    print(f"   Got back: {eth_back:.4f} ETH, {usdc_back:.4f} USDC")

    # Test 5: Impermanent loss
    print("\n5. Impermanent loss at 2x price change:")
    il = pool.calculate_impermanent_loss(2.0)
    print(f"   IL = {il * 100:.3f}%")

    print("\nAll tests passed!")
