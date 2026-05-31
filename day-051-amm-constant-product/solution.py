"""
Day 51: Automated Market Maker — Constant Product Formula (x * y = k)

A complete AMM simulation implementing Uniswap V2-style constant product pools.
Covers liquidity provision, swaps, price impact, impermanent loss, and arbitrage.
"""

import math
from dataclasses import dataclass, field
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

    This pool holds two tokens (A and B) and allows:
    - Adding liquidity (depositing both tokens proportionally)
    - Swapping one token for another
    - Removing liquidity (withdrawing proportional reserves)

    The key invariant: reserve_a * reserve_b = k (before fees).
    Fees cause k to grow over time, rewarding liquidity providers.
    """

    def __init__(self, fee_rate: float = 0.003):
        """
        Initialize an empty pool.

        Args:
            fee_rate: Trading fee as a decimal (0.003 = 0.3%, matching Uniswap V2).
                      The fee is charged on the input token and stays in the pool.
        """
        self.reserve_a: float = 0.0
        self.reserve_b: float = 0.0
        self.fee_rate: float = fee_rate

        # LP token tracking: maps provider address (string) to their LP token balance
        self.lp_balances: dict[str, float] = {}
        self.total_lp_supply: float = 0.0

        # Metrics tracking
        self.total_fees_a: float = 0.0
        self.total_fees_b: float = 0.0
        self.swap_count: int = 0

    @property
    def k(self) -> float:
        """The constant product invariant. Increases over time due to fees."""
        return self.reserve_a * self.reserve_b

    @property
    def spot_price_a_in_b(self) -> float:
        """Price of token A denominated in token B: how much B you get per A (marginal)."""
        if self.reserve_a == 0:
            return 0.0
        return self.reserve_b / self.reserve_a

    @property
    def spot_price_b_in_a(self) -> float:
        """Price of token B denominated in token A."""
        if self.reserve_b == 0:
            return 0.0
        return self.reserve_a / self.reserve_b

    def add_liquidity(
        self, provider: str, amount_a: float, amount_b: float
    ) -> LiquidityEvent:
        """
        Add liquidity to the pool.

        For the first deposit, the provider sets the initial price ratio.
        For subsequent deposits, tokens must be added in the current reserve ratio
        to avoid changing the price. If the ratio doesn't match, we use the
        limiting token and refund the excess (simplified here by requiring exact ratio).

        The LP tokens minted follow these rules:
        - First deposit: LP = sqrt(amount_a * amount_b)  [geometric mean]
        - Later deposits: LP = total_supply * min(da/a, db/b)

        Why geometric mean for the first deposit? It makes LP tokens independent
        of the price ratio. Whether you deposit (100 ETH, 200K USDC) or
        (200K USDC, 100 ETH), you get the same number of LP tokens.

        Args:
            provider: Identifier for the liquidity provider
            amount_a: Amount of token A to deposit
            amount_b: Amount of token B to deposit

        Returns:
            LiquidityEvent with details of the deposit
        """
        if amount_a <= 0 or amount_b <= 0:
            raise ValueError("Deposit amounts must be positive")

        if self.total_lp_supply == 0:
            # First deposit — use geometric mean for LP token calculation.
            # We subtract a MINIMUM_LIQUIDITY (burned to address(0)) in real implementations
            # to prevent the first LP from manipulating the pool. Simplified here.
            lp_tokens = math.sqrt(amount_a * amount_b)
        else:
            # Subsequent deposits — mint proportional to the smaller ratio.
            # This ensures the deposit doesn't change the price.
            ratio_a = amount_a / self.reserve_a
            ratio_b = amount_b / self.reserve_b
            # Use the minimum ratio — in production, you'd refund the excess of the other token.
            lp_tokens = self.total_lp_supply * min(ratio_a, ratio_b)

        # Update pool state
        self.reserve_a += amount_a
        self.reserve_b += amount_b

        # Mint LP tokens
        self.lp_balances[provider] = self.lp_balances.get(provider, 0.0) + lp_tokens
        self.total_lp_supply += lp_tokens

        share = lp_tokens / self.total_lp_supply

        return LiquidityEvent(
            action="add",
            token_a_amount=amount_a,
            token_b_amount=amount_b,
            lp_tokens=lp_tokens,
            share_of_pool=share,
        )

    def remove_liquidity(self, provider: str, lp_tokens: float) -> LiquidityEvent:
        """
        Remove liquidity by burning LP tokens.

        The provider receives their proportional share of BOTH tokens.
        This share reflects all accumulated fees since their deposit.

        Note: the returned tokens will be in the current ratio, which may
        differ from the deposit ratio if trades have occurred. This ratio
        change is the source of impermanent loss.

        Args:
            provider: Identifier for the liquidity provider
            lp_tokens: Number of LP tokens to burn

        Returns:
            LiquidityEvent with details of the withdrawal
        """
        balance = self.lp_balances.get(provider, 0.0)
        if lp_tokens > balance:
            raise ValueError(
                f"Insufficient LP tokens: have {balance:.6f}, want {lp_tokens:.6f}"
            )
        if lp_tokens <= 0:
            raise ValueError("Must burn positive LP tokens")

        # Calculate proportional share of reserves
        share = lp_tokens / self.total_lp_supply
        amount_a = self.reserve_a * share
        amount_b = self.reserve_b * share

        # Update pool state
        self.reserve_a -= amount_a
        self.reserve_b -= amount_b

        # Burn LP tokens
        self.lp_balances[provider] -= lp_tokens
        self.total_lp_supply -= lp_tokens

        return LiquidityEvent(
            action="remove",
            token_a_amount=amount_a,
            token_b_amount=amount_b,
            lp_tokens=lp_tokens,
            share_of_pool=share,
        )

    def swap_a_for_b(self, amount_a_in: float) -> SwapResult:
        """
        Swap token A for token B.

        The core AMM operation. Given dx of token A:
        1. Charge fee: dx_effective = dx * (1 - fee_rate)
        2. Calculate output: dy = reserve_b * dx_effective / (reserve_a + dx_effective)
        3. Update reserves: reserve_a += dx, reserve_b -= dy

        Note that the FULL input (including fee) is added to reserves.
        The fee portion increases k, growing the pool for LPs.

        Args:
            amount_a_in: Amount of token A to sell

        Returns:
            SwapResult with execution details and price impact metrics
        """
        return self._swap(amount_a_in, "A", "B")

    def swap_b_for_a(self, amount_b_in: float) -> SwapResult:
        """Swap token B for token A. Mirror of swap_a_for_b."""
        return self._swap(amount_b_in, "B", "A")

    def _swap(self, amount_in: float, token_in: str, token_out: str) -> SwapResult:
        """
        Internal swap implementation.

        The constant product formula derivation:
        - Before: x * y = k
        - After:  (x + dx_eff) * (y - dy) = k
        - Therefore: dy = y * dx_eff / (x + dx_eff)

        This can also be written as: dy = y - k / (x + dx_eff)
        Both are algebraically equivalent, but the first form is more numerically stable.
        """
        if amount_in <= 0:
            raise ValueError("Swap amount must be positive")
        if self.reserve_a == 0 or self.reserve_b == 0:
            raise ValueError("Pool has no liquidity")

        # Determine which reserves we're working with
        if token_in == "A":
            reserve_in = self.reserve_a
            reserve_out = self.reserve_b
        else:
            reserve_in = self.reserve_b
            reserve_out = self.reserve_a

        # Record spot price before the swap (price of input token in terms of output token)
        spot_price_before = reserve_out / reserve_in

        # Apply fee — only the effective amount participates in the price calculation.
        # The fee portion stays in the pool, increasing k.
        fee_amount = amount_in * self.fee_rate
        amount_in_after_fee = amount_in - fee_amount

        # Core constant product calculation
        # dy = y * dx / (x + dx) where dx is after fee
        amount_out = reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee)

        # Safety check: can't drain the pool completely
        if amount_out >= reserve_out:
            raise ValueError("Trade too large: would drain the pool")

        # Execution price: what rate the trader actually got
        execution_price = amount_out / amount_in

        # Price impact: how much worse than spot price
        # A positive price impact means the trader got a worse price
        price_impact = 1.0 - (execution_price / spot_price_before)

        # Update reserves — note: FULL amount_in goes in (including fee portion)
        if token_in == "A":
            self.reserve_a += amount_in
            self.reserve_b -= amount_out
            self.total_fees_a += fee_amount
        else:
            self.reserve_b += amount_in
            self.reserve_a -= amount_out
            self.total_fees_b += fee_amount

        self.swap_count += 1

        # Spot price after the swap
        spot_price_after = (
            self.reserve_b / self.reserve_a
            if token_in == "A"
            else self.reserve_a / self.reserve_b
        )

        return SwapResult(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=amount_out,
            fee_amount=fee_amount,
            spot_price_before=spot_price_before,
            execution_price=execution_price,
            price_impact=price_impact,
            spot_price_after=spot_price_after,
        )

    def get_quote(self, amount_in: float, token_in: str) -> tuple[float, float]:
        """
        Get a swap quote without executing. Returns (amount_out, price_impact).

        Useful for frontends to show expected output before the user confirms.
        """
        if token_in == "A":
            reserve_in, reserve_out = self.reserve_a, self.reserve_b
        else:
            reserve_in, reserve_out = self.reserve_b, self.reserve_a

        amount_in_after_fee = amount_in * (1 - self.fee_rate)
        amount_out = reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee)

        spot_price = reserve_out / reserve_in
        exec_price = amount_out / amount_in
        price_impact = 1.0 - (exec_price / spot_price)

        return amount_out, price_impact


def calculate_impermanent_loss(price_ratio: float) -> float:
    """
    Calculate impermanent loss for a given price change ratio.

    The IL formula is derived from comparing:
    1. Value of LP position after price change (pool rebalances via arbitrage)
    2. Value of simply holding the original tokens

    Derivation:
    - Initial: deposit x0 of A and y0 of B at price P0 = y0/x0
    - After price changes to P1 = r * P0:
      - Pool rebalances: x1 = x0/sqrt(r), y1 = y0*sqrt(r)
      - LP value: x1*P1 + y1 = 2 * y0 * sqrt(r)
      - HODL value: x0*P1 + y0 = y0 * (1 + r)
      - IL = LP_value / HODL_value - 1 = 2*sqrt(r)/(1+r) - 1

    Args:
        price_ratio: New price / old price (e.g., 2.0 means price doubled)

    Returns:
        IL as a decimal (always <= 0). E.g., -0.057 means 5.7% loss vs holding.
    """
    if price_ratio <= 0:
        raise ValueError("Price ratio must be positive")
    # The elegant formula: IL depends ONLY on the price ratio, not absolute prices
    return 2.0 * math.sqrt(price_ratio) / (1.0 + price_ratio) - 1.0


def calculate_arbitrage_trade(
    pool: LiquidityPool, external_price_a_in_b: float
) -> Optional[tuple[str, float]]:
    """
    Calculate the optimal arbitrage trade to align pool price with external market.

    When the pool price diverges from the external market, an arbitrageur can profit
    by trading the pool back toward the market price. This is the mechanism that keeps
    AMM prices in sync with the broader market — and the source of impermanent loss.

    Math:
    - Pool price: P_pool = reserve_b / reserve_a
    - External price: P_ext
    - If P_ext > P_pool: token A is underpriced in the pool → buy A (sell B)
    - If P_ext < P_pool: token A is overpriced in the pool → sell A (buy B)

    For buying A with B (when P_ext > P_pool):
    - After arbitrage, pool should satisfy: reserve_b_new / reserve_a_new = P_ext
    - Using k = reserve_a * reserve_b:
      - reserve_a_new = sqrt(k / P_ext)
      - amount_b_needed = reserve_b_new - reserve_b

    We account for fees in the calculation.

    Args:
        pool: The liquidity pool
        external_price_a_in_b: External market price of A in terms of B

    Returns:
        (direction, amount) tuple or None if no profitable arbitrage exists.
        direction is "sell_a" or "sell_b", amount is the input amount.
    """
    if pool.reserve_a == 0 or pool.reserve_b == 0:
        return None

    pool_price = pool.spot_price_a_in_b

    # Calculate target reserves after arbitrage (ignoring fees for target)
    k = pool.k
    # Target: reserve_a_new * external_price = reserve_b_new
    # And: reserve_a_new * reserve_b_new = k (approximately, before fee growth)
    # So: reserve_a_new^2 * external_price = k
    target_reserve_a = math.sqrt(k / external_price_a_in_b)
    target_reserve_b = math.sqrt(k * external_price_a_in_b)

    if external_price_a_in_b > pool_price:
        # Token A is cheap in the pool — buy A by selling B
        # The arbitrageur needs to add B to the pool
        # Account for fees: they need to sell more B because fee is deducted
        amount_b_raw = target_reserve_b - pool.reserve_b
        if amount_b_raw <= 0:
            return None
        # Adjust for fee: actual input needs to be larger since fee is taken
        amount_b_in = amount_b_raw / (1 - pool.fee_rate)

        # Verify profitability: check that the A received is worth more than B spent
        amount_a_out, _ = pool.get_quote(amount_b_in, "B")
        profit = amount_a_out * external_price_a_in_b - amount_b_in
        if profit <= 0:
            return None

        return ("sell_b", amount_b_in)
    else:
        # Token A is expensive in the pool — sell A, buy B
        amount_a_raw = target_reserve_a - pool.reserve_a
        if amount_a_raw <= 0:
            return None
        amount_a_in = amount_a_raw / (1 - pool.fee_rate)

        amount_b_out, _ = pool.get_quote(amount_a_in, "A")
        profit = amount_b_out - amount_a_in * external_price_a_in_b
        if profit <= 0:
            return None

        return ("sell_a", amount_a_in)


def run_price_impact_analysis(pool: LiquidityPool) -> list[tuple[float, float]]:
    """
    Analyze how price impact scales with trade size.

    Returns a list of (trade_size_as_fraction_of_reserve, price_impact) tuples.
    This demonstrates that price impact is purely a function of trade size
    relative to pool depth — not absolute amounts.
    """
    results = []
    fractions = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    for frac in fractions:
        trade_size = pool.reserve_a * frac
        _, impact = pool.get_quote(trade_size, "A")
        results.append((frac, impact))

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("AUTOMATED MARKET MAKER — CONSTANT PRODUCT (x * y = k)")
    print("=" * 70)

    # ── Step 1: Initialize Pool ──────────────────────────────────────────
    print("\n--- Step 1: Pool Initialization ---")
    pool = LiquidityPool(fee_rate=0.003)  # 0.3% fee like Uniswap V2
    print(f"Empty pool created with {pool.fee_rate*100}% fee")

    # ── Step 2: Add Initial Liquidity ────────────────────────────────────
    print("\n--- Step 2: Add Initial Liquidity ---")
    # Simulate: 10 ETH and 20,000 USDC → initial price = 2000 USDC/ETH
    event = pool.add_liquidity("alice", amount_a=10.0, amount_b=20000.0)
    print(f"Alice deposits: 10 ETH + 20,000 USDC")
    print(f"LP tokens minted: {event.lp_tokens:.4f}")
    print(f"Pool share: {event.share_of_pool*100:.2f}%")
    print(f"Initial k = {pool.k:,.2f}")
    print(f"Spot price: 1 ETH = {pool.spot_price_a_in_b:,.2f} USDC")

    # Second LP adds proportionally
    event2 = pool.add_liquidity("bob", amount_a=5.0, amount_b=10000.0)
    print(f"\nBob deposits: 5 ETH + 10,000 USDC")
    print(f"LP tokens minted: {event2.lp_tokens:.4f}")
    print(f"Bob's pool share: {event2.share_of_pool*100:.2f}%")
    print(f"New k = {pool.k:,.2f}")
    print(f"Price unchanged: 1 ETH = {pool.spot_price_a_in_b:,.2f} USDC")

    # ── Step 3: Perform Swaps ────────────────────────────────────────────
    print("\n--- Step 3: Swap Execution ---")

    # Small swap: buy ETH with 500 USDC
    result = pool.swap_b_for_a(500.0)
    print(f"Swap: 500 USDC → {result.amount_out:.6f} ETH")
    print(f"  Spot price before: {result.spot_price_before:.2f} USDC/ETH")
    print(f"  Execution price:   {1/result.execution_price:.2f} USDC/ETH")
    print(f"  Price impact:      {result.price_impact*100:.4f}%")
    print(f"  Fee paid:          {result.fee_amount:.4f} USDC")
    print(f"  New spot price:    {pool.spot_price_a_in_b:.2f} USDC/ETH")
    print(f"  New k:             {pool.k:,.2f} (grew from fees)")

    # Larger swap to show higher price impact
    result2 = pool.swap_b_for_a(5000.0)
    print(f"\nSwap: 5,000 USDC → {result2.amount_out:.6f} ETH")
    print(f"  Price impact:      {result2.price_impact*100:.4f}%")
    print(f"  New spot price:    {pool.spot_price_a_in_b:.2f} USDC/ETH")

    # ── Step 4: Price Impact Analysis ────────────────────────────────────
    print("\n--- Step 4: Price Impact vs Trade Size ---")
    # Reset pool for clean analysis
    analysis_pool = LiquidityPool(fee_rate=0.003)
    analysis_pool.add_liquidity("lp", 100.0, 200000.0)
    print(f"Analysis pool: 100 ETH + 200,000 USDC (k = {analysis_pool.k:,.0f})")
    print(f"{'Trade Size':>15} {'Amount ETH':>12} {'Price Impact':>14}")
    print("-" * 45)
    impacts = run_price_impact_analysis(analysis_pool)
    for frac, impact in impacts:
        trade_amount = analysis_pool.reserve_a * frac
        print(f"{frac*100:>14.1f}% {trade_amount:>12.2f} {impact*100:>13.4f}%")

    # ── Step 5: Impermanent Loss ─────────────────────────────────────────
    print("\n--- Step 5: Impermanent Loss Analysis ---")
    print(f"{'Price Change':>14} {'IL':>10} {'LP Value (HODL=100)':>22}")
    print("-" * 50)
    for ratio in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]:
        il = calculate_impermanent_loss(ratio)
        lp_relative = (1 + il) * 100
        direction = "down" if ratio < 1 else ("flat" if ratio == 1 else "up")
        print(f"  {ratio:>5.2f}x ({direction:>4}) {il*100:>9.3f}% {lp_relative:>20.2f}")

    # ── Step 6: Arbitrage Simulation ─────────────────────────────────────
    print("\n--- Step 6: Arbitrage Simulation ---")
    arb_pool = LiquidityPool(fee_rate=0.003)
    arb_pool.add_liquidity("lp", 100.0, 200000.0)
    print(f"Pool price: {arb_pool.spot_price_a_in_b:.2f} USDC/ETH")
    print(f"External price rises to 2,500 USDC/ETH")

    arb = calculate_arbitrage_trade(arb_pool, 2500.0)
    if arb:
        direction, amount = arb
        print(f"Arbitrage: {direction} with amount {amount:.4f}")

        if direction == "sell_b":
            result = arb_pool.swap_b_for_a(amount)
            print(f"  Bought {result.amount_out:.4f} ETH for {amount:.2f} USDC")
            profit = result.amount_out * 2500.0 - amount
            print(f"  Profit: {profit:.2f} USDC")
        else:
            result = arb_pool.swap_a_for_b(amount)
            print(f"  Sold {amount:.4f} ETH for {result.amount_out:.2f} USDC")
            profit = result.amount_out - amount * 2500.0
            print(f"  Profit: {profit:.2f} USDC")

        print(f"  Pool price after arb: {arb_pool.spot_price_a_in_b:.2f} USDC/ETH")

    # ── Step 7: LP Returns with Fee Accumulation ─────────────────────────
    print("\n--- Step 7: LP Returns Over Time ---")
    import random
    random.seed(42)

    lp_pool = LiquidityPool(fee_rate=0.003)
    lp_pool.add_liquidity("lp", 100.0, 200000.0)
    initial_k = lp_pool.k
    initial_value = 100 * 2000 + 200000  # value at initial prices

    print(f"Initial pool: 100 ETH + 200,000 USDC")
    print(f"Initial k: {initial_k:,.2f}")

    # Simulate 100 random trades
    for _ in range(100):
        # Random trade: 50% chance each direction, random size 0.1% to 5% of reserves
        size_frac = random.uniform(0.001, 0.05)
        if random.random() < 0.5:
            amount = lp_pool.reserve_a * size_frac
            lp_pool.swap_a_for_b(amount)
        else:
            amount = lp_pool.reserve_b * size_frac
            lp_pool.swap_b_for_a(amount)

    print(f"\nAfter 100 random trades:")
    print(f"  Final k: {lp_pool.k:,.2f}")
    print(f"  k growth: {(lp_pool.k / initial_k - 1) * 100:.4f}%")
    print(f"  Total fees collected: {lp_pool.total_fees_a:.4f} ETH + {lp_pool.total_fees_b:.2f} USDC")
    print(f"  Reserves: {lp_pool.reserve_a:.4f} ETH + {lp_pool.reserve_b:.2f} USDC")
    print(f"  Spot price: {lp_pool.spot_price_a_in_b:.2f} USDC/ETH")

    # ── Step 8: Remove Liquidity ─────────────────────────────────────────
    print("\n--- Step 8: Liquidity Removal ---")
    lp_tokens = lp_pool.lp_balances["lp"]
    removal = lp_pool.remove_liquidity("lp", lp_tokens)
    print(f"LP withdraws all {lp_tokens:.4f} LP tokens:")
    print(f"  Received: {removal.token_a_amount:.4f} ETH + {removal.token_b_amount:.2f} USDC")
    current_price = lp_pool.spot_price_a_in_b if lp_pool.reserve_a > 0 else 2000.0
    # Use the price before removal for valuation
    final_value = removal.token_a_amount * 2000 + removal.token_b_amount
    print(f"  Value at original price (2000 USDC/ETH): {final_value:,.2f} USDC")
    print(f"  vs initial: {initial_value:,.2f} USDC")
    print(f"  Net gain from fees: {final_value - initial_value:,.2f} USDC")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("  1. x*y=k ensures prices adjust automatically based on supply/demand")
    print("  2. Price impact scales with trade size relative to pool depth")
    print("  3. Fees grow k over time, rewarding liquidity providers")
    print("  4. Impermanent loss is the cost LPs pay for arbitrageurs keeping prices aligned")
    print("  5. Larger pools = less slippage = better for traders, but lower fee APY for LPs")
    print("=" * 70)
