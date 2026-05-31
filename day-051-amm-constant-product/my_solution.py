"""
Day 51: Automated Market Maker — Constant Product Formula (x * y = k)

YOUR TASK: Implement a complete AMM with the constant product invariant.

Key formulas to remember:
- Invariant: reserve_a * reserve_b = k
- Swap output: dy = reserve_out * dx_eff / (reserve_in + dx_eff)
- Fee: dx_eff = dx * (1 - fee_rate)
- Impermanent loss: IL = 2*sqrt(r) / (1+r) - 1
- LP tokens (first deposit): sqrt(amount_a * amount_b)
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SwapResult:
    """Captures all details of a swap for analysis."""
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    fee_amount: float
    spot_price_before: float
    execution_price: float
    price_impact: float
    spot_price_after: float


@dataclass
class LiquidityEvent:
    """Records a liquidity add/remove event."""
    action: str  # "add" or "remove"
    token_a_amount: float
    token_b_amount: float
    lp_tokens: float
    share_of_pool: float


class LiquidityPool:
    """
    Constant product AMM implementing x * y = k.

    Hint: The pool needs to track reserves for two tokens, LP token balances
    per provider, total LP supply, and accumulated fees.
    """

    def __init__(self, fee_rate: float = 0.003):
        """
        Initialize an empty pool.

        Args:
            fee_rate: Trading fee as a decimal (0.003 = 0.3%)
        """
        raise NotImplementedError("TODO: initialize pool state — reserves, LP tracking, fee config")

    @property
    def k(self) -> float:
        """The constant product invariant."""
        raise NotImplementedError("TODO: return reserve_a * reserve_b")

    @property
    def spot_price_a_in_b(self) -> float:
        """Price of token A in terms of token B (marginal rate)."""
        raise NotImplementedError("TODO: return reserve_b / reserve_a")

    @property
    def spot_price_b_in_a(self) -> float:
        """Price of token B in terms of token A."""
        raise NotImplementedError("TODO: return reserve_a / reserve_b")

    def add_liquidity(
        self, provider: str, amount_a: float, amount_b: float
    ) -> LiquidityEvent:
        """
        Add liquidity to the pool. Mint LP tokens to the provider.

        Hint: First deposit uses geometric mean sqrt(a*b) for LP tokens.
        Subsequent deposits use proportional minting based on the smaller ratio.

        Args:
            provider: Identifier for the LP
            amount_a: Amount of token A to deposit
            amount_b: Amount of token B to deposit

        Returns:
            LiquidityEvent with deposit details
        """
        raise NotImplementedError("TODO: implement liquidity addition with LP token minting")

    def remove_liquidity(self, provider: str, lp_tokens: float) -> LiquidityEvent:
        """
        Remove liquidity by burning LP tokens. Return proportional reserves.

        Hint: share = lp_tokens / total_supply, then multiply by each reserve.

        Args:
            provider: Identifier for the LP
            lp_tokens: Number of LP tokens to burn

        Returns:
            LiquidityEvent with withdrawal details
        """
        raise NotImplementedError("TODO: implement proportional liquidity removal")

    def swap_a_for_b(self, amount_a_in: float) -> SwapResult:
        """Swap token A for token B using constant product formula."""
        raise NotImplementedError("TODO: implement swap with fee, price impact tracking")

    def swap_b_for_a(self, amount_b_in: float) -> SwapResult:
        """Swap token B for token A."""
        raise NotImplementedError("TODO: implement swap (mirror of swap_a_for_b)")

    def get_quote(self, amount_in: float, token_in: str) -> tuple[float, float]:
        """
        Get a swap quote without executing. Returns (amount_out, price_impact).

        Hint: Same math as swap, just don't update state.
        """
        raise NotImplementedError("TODO: implement read-only quote")


def calculate_impermanent_loss(price_ratio: float) -> float:
    """
    Calculate impermanent loss for a given price change ratio.

    Hint: The formula is IL = 2*sqrt(r) / (1+r) - 1
    where r = new_price / old_price.

    Args:
        price_ratio: New price / old price (e.g., 2.0 = price doubled)

    Returns:
        IL as a decimal (always <= 0)
    """
    raise NotImplementedError("TODO: implement the IL formula")


def calculate_arbitrage_trade(
    pool: LiquidityPool, external_price_a_in_b: float
) -> Optional[tuple[str, float]]:
    """
    Calculate optimal arbitrage trade to align pool with external price.

    Hint: Use sqrt(k / P_ext) to find target reserve_a.
    Compare with current reserves to determine direction and size.
    Account for fees in the calculation.

    Args:
        pool: The liquidity pool
        external_price_a_in_b: External market price

    Returns:
        (direction, amount) or None if no profitable arb exists
    """
    raise NotImplementedError("TODO: implement arbitrage calculation")


if __name__ == "__main__":
    print("Testing your AMM implementation...\n")

    # Test 1: Pool initialization and liquidity
    pool = LiquidityPool(fee_rate=0.003)
    event = pool.add_liquidity("alice", 10.0, 20000.0)
    print(f"Pool created: {pool.reserve_a} ETH + {pool.reserve_b} USDC")
    print(f"k = {pool.k}")
    print(f"Spot price: {pool.spot_price_a_in_b} USDC/ETH")
    print(f"LP tokens: {event.lp_tokens}")

    # Test 2: Swap
    result = pool.swap_b_for_a(1000.0)
    print(f"\nSwapped 1000 USDC → {result.amount_out:.6f} ETH")
    print(f"Price impact: {result.price_impact*100:.4f}%")

    # Test 3: Impermanent loss
    il = calculate_impermanent_loss(2.0)
    print(f"\nIL at 2x price: {il*100:.3f}%")

    # Test 4: Arbitrage
    arb_pool = LiquidityPool(fee_rate=0.003)
    arb_pool.add_liquidity("lp", 100.0, 200000.0)
    arb = calculate_arbitrage_trade(arb_pool, 2500.0)
    print(f"\nArbitrage opportunity: {arb}")

    # Test 5: Remove liquidity
    removal = pool.remove_liquidity("alice", event.lp_tokens)
    print(f"\nRemoved liquidity: {removal.token_a_amount:.4f} ETH + {removal.token_b_amount:.2f} USDC")

    print("\nAll basic tests passed!")
