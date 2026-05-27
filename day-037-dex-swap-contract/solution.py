"""
Day 037: DEX Swap Contract — Automated Market Maker Implementation

A complete constant-product AMM (Uniswap V2-style) implemented in Python.
This simulates the core mechanics: liquidity provision, token swapping with fees,
LP token accounting, and price impact calculation.

Key formula: x * y = k (constant product invariant)
"""

import math
from typing import Optional


class Token:
    """
    Simulates an ERC-20 token with balances and transfer mechanics.

    In a real smart contract, this would be a separate ERC-20 contract.
    Here we model balances as a dictionary mapping addresses to amounts.
    We use float for readability, but production code would use integer
    arithmetic to avoid floating-point precision issues.
    """

    def __init__(self, name: str, symbol: str, initial_supply: float = 0.0):
        self.name = name
        self.symbol = symbol
        # Balances map: address (string) -> amount (float)
        self.balances: dict[str, float] = {}
        self.total_supply = initial_supply

    def mint(self, to: str, amount: float) -> None:
        """Create new tokens and assign to an address."""
        self.balances[to] = self.balances.get(to, 0.0) + amount
        self.total_supply += amount

    def burn(self, from_addr: str, amount: float) -> None:
        """Destroy tokens from an address."""
        if self.balances.get(from_addr, 0.0) < amount - 1e-10:
            raise ValueError(f"Insufficient balance: {from_addr} has {self.balances.get(from_addr, 0.0)}, needs {amount}")
        self.balances[from_addr] -= amount
        self.total_supply -= amount

    def balance_of(self, addr: str) -> float:
        return self.balances.get(addr, 0.0)

    def transfer(self, from_addr: str, to_addr: str, amount: float) -> None:
        """Transfer tokens between addresses. Validates sufficient balance."""
        if self.balances.get(from_addr, 0.0) < amount - 1e-10:
            raise ValueError(
                f"Insufficient {self.symbol} balance: {from_addr} has "
                f"{self.balances.get(from_addr, 0.0):.6f}, needs {amount:.6f}"
            )
        self.balances[from_addr] = self.balances.get(from_addr, 0.0) - amount
        self.balances[to_addr] = self.balances.get(to_addr, 0.0) + amount


class LiquidityPool:
    """
    Constant-product Automated Market Maker (x * y = k).

    This implements the core Uniswap V2 mechanics:
    - Liquidity provision with LP token minting
    - Token swaps with configurable fee
    - Liquidity removal with proportional reserve withdrawal
    - Price impact and slippage calculation

    Architecture decision: We store reserves directly rather than tracking
    token balances of the pool address. This is simpler and equivalent for
    our simulation — in a real contract, reserves are tracked separately
    from balances to prevent donation attacks.
    """

    # The pool's "address" for token transfers
    POOL_ADDRESS = "pool"

    def __init__(
        self,
        token_a: Token,
        token_b: Token,
        fee_rate: float = 0.003,  # 0.3% — standard Uniswap V2 fee
    ):
        self.token_a = token_a
        self.token_b = token_b
        self.fee_rate = fee_rate

        # Pool reserves — tracks how much of each token the pool holds
        self.reserve_a: float = 0.0
        self.reserve_b: float = 0.0

        # LP token tracks liquidity provider shares
        # Using geometric mean for initial mint prevents LP value
        # from depending on the arbitrary initial price ratio
        self.lp_token = Token("LP Token", "LP")

        # Cumulative price tracking for TWAP oracle
        # In production, these use fixed-point math to avoid overflow
        self.cumulative_price_a: float = 0.0  # sum of (reserve_b / reserve_a) over time
        self.cumulative_price_b: float = 0.0  # sum of (reserve_a / reserve_b) over time
        self.last_block_timestamp: int = 0

        # Fee accounting — total fees collected per token
        self.total_fees_a: float = 0.0
        self.total_fees_b: float = 0.0

        # Trade history for analysis
        self.trades: list[dict] = []

    @property
    def k(self) -> float:
        """The constant product invariant. Should only increase (from fees)."""
        return self.reserve_a * self.reserve_b

    def spot_price_a_in_b(self) -> float:
        """
        Price of token A denominated in token B.
        This is the marginal price for an infinitesimally small trade.
        Real trades always get a worse price due to price impact.
        """
        if self.reserve_a == 0:
            return 0.0
        return self.reserve_b / self.reserve_a

    def spot_price_b_in_a(self) -> float:
        """Price of token B denominated in token A."""
        if self.reserve_b == 0:
            return 0.0
        return self.reserve_a / self.reserve_b

    def add_liquidity(
        self,
        provider: str,
        amount_a: float,
        amount_b: float,
    ) -> tuple[float, float, float]:
        """
        Add liquidity to the pool and receive LP tokens.

        For the first deposit, any ratio is accepted and LP tokens = sqrt(a * b).
        For subsequent deposits, tokens must be provided in the current ratio.
        If the ratio doesn't match, we use the minimum ratio to prevent manipulation
        and the excess of one token is NOT deposited (returned to the user).

        Returns: (actual_a_deposited, actual_b_deposited, lp_tokens_minted)
        """
        if self.reserve_a == 0 and self.reserve_b == 0:
            # First deposit: set the initial price ratio
            # LP tokens = geometric mean, making LP value ratio-independent
            # Why sqrt? If we used a+b, someone could inflate LP supply by
            # depositing 1M of a worthless token. sqrt(a*b) means both tokens
            # must have real value for the LP tokens to be valuable.
            lp_minted = math.sqrt(amount_a * amount_b)

            # Transfer tokens into the pool
            self.token_a.transfer(provider, self.POOL_ADDRESS, amount_a)
            self.token_b.transfer(provider, self.POOL_ADDRESS, amount_b)
            self.reserve_a = amount_a
            self.reserve_b = amount_b

            # Mint LP tokens to the provider
            self.lp_token.mint(provider, lp_minted)

            return (amount_a, amount_b, lp_minted)
        else:
            # Subsequent deposits: must match current ratio
            # Calculate the optimal amounts based on current reserves
            optimal_b = amount_a * self.reserve_b / self.reserve_a
            optimal_a = amount_b * self.reserve_a / self.reserve_b

            if optimal_b <= amount_b:
                # Use all of amount_a, adjust amount_b down
                actual_a = amount_a
                actual_b = optimal_b
            else:
                # Use all of amount_b, adjust amount_a down
                actual_a = optimal_a
                actual_b = amount_b

            # LP tokens minted proportional to the share of reserves added
            # Using min() of the two ratios prevents gaming by providing
            # excess of one token to inflate LP share
            share_a = actual_a / self.reserve_a
            share_b = actual_b / self.reserve_b
            share = min(share_a, share_b)
            lp_minted = self.lp_token.total_supply * share

            # Execute transfers
            self.token_a.transfer(provider, self.POOL_ADDRESS, actual_a)
            self.token_b.transfer(provider, self.POOL_ADDRESS, actual_b)
            self.reserve_a += actual_a
            self.reserve_b += actual_b
            self.lp_token.mint(provider, lp_minted)

            return (actual_a, actual_b, lp_minted)

    def remove_liquidity(
        self,
        provider: str,
        lp_amount: float,
    ) -> tuple[float, float]:
        """
        Burn LP tokens and withdraw proportional reserves.

        The provider gets back their SHARE of the pool, not their original deposit.
        If the price ratio has shifted since deposit, they'll get a different mix
        of tokens — this is where impermanent loss materializes.

        Returns: (amount_a_withdrawn, amount_b_withdrawn)
        """
        if lp_amount <= 0:
            raise ValueError("Must remove positive liquidity amount")
        if self.lp_token.balance_of(provider) < lp_amount - 1e-10:
            raise ValueError("Insufficient LP token balance")

        # Calculate the provider's share of the pool
        share = lp_amount / self.lp_token.total_supply

        # Calculate proportional amounts to return
        amount_a = self.reserve_a * share
        amount_b = self.reserve_b * share

        # Burn LP tokens first (checks-effects-interactions pattern)
        self.lp_token.burn(provider, lp_amount)

        # Update reserves and transfer tokens
        self.reserve_a -= amount_a
        self.reserve_b -= amount_b
        self.token_a.transfer(self.POOL_ADDRESS, provider, amount_a)
        self.token_b.transfer(self.POOL_ADDRESS, provider, amount_b)

        return (amount_a, amount_b)

    def get_output_amount(
        self,
        input_amount: float,
        input_reserve: float,
        output_reserve: float,
    ) -> tuple[float, float]:
        """
        Calculate output amount for a given input using the constant product formula.

        The fee is taken from the input amount BEFORE computing the swap.
        This means the fee stays in the pool, increasing k and benefiting all LPs.

        Math:
            input_after_fee = input_amount * (1 - fee_rate)
            output = output_reserve * input_after_fee / (input_reserve + input_after_fee)

        This is algebraically derived from:
            (input_reserve + input_after_fee) * (output_reserve - output) = input_reserve * output_reserve

        Returns: (output_amount, fee_amount)
        """
        fee = input_amount * self.fee_rate
        input_after_fee = input_amount - fee

        # Constant product formula, solved for output amount
        # The denominator (input_reserve + input_after_fee) ensures that larger
        # trades get progressively worse prices — this IS the slippage mechanism
        numerator = output_reserve * input_after_fee
        denominator = input_reserve + input_after_fee
        output_amount = numerator / denominator

        return (output_amount, fee)

    def swap_a_for_b(
        self,
        trader: str,
        amount_a_in: float,
        min_b_out: float = 0.0,
        block_timestamp: int = 0,
    ) -> float:
        """
        Swap token A for token B.

        The min_b_out parameter is slippage protection — the trade reverts if
        the output would be less than this amount. In production, this protects
        against front-running (sandwich attacks) and price movement between
        transaction submission and execution.

        Returns: amount of token B received
        """
        if amount_a_in <= 0:
            raise ValueError("Input amount must be positive")
        if self.reserve_a == 0 or self.reserve_b == 0:
            raise ValueError("Pool has no liquidity")

        # Update price oracle before state changes
        self._update_oracle(block_timestamp)

        # Record state before swap for analysis
        k_before = self.k
        price_before = self.spot_price_a_in_b()

        # Calculate output
        amount_b_out, fee = self.get_output_amount(
            amount_a_in, self.reserve_a, self.reserve_b
        )

        # Slippage protection: revert if output is below minimum
        if amount_b_out < min_b_out:
            raise ValueError(
                f"Slippage exceeded: would receive {amount_b_out:.6f} {self.token_b.symbol}, "
                f"minimum is {min_b_out:.6f}"
            )

        # Execute the swap: transfer tokens and update reserves
        self.token_a.transfer(trader, self.POOL_ADDRESS, amount_a_in)
        self.token_b.transfer(self.POOL_ADDRESS, trader, amount_b_out)
        self.reserve_a += amount_a_in
        self.reserve_b -= amount_b_out

        # Track fees
        self.total_fees_a += fee

        # Verify invariant: k should increase (from fees) or stay constant
        k_after = self.k
        assert k_after >= k_before - 1e-6, (
            f"Invariant violated: k went from {k_before} to {k_after}"
        )

        # Calculate actual price impact for this trade
        # Price impact = how much worse the effective price is vs the spot price before
        effective_price = amount_b_out / amount_a_in  # USDC per ETH received
        price_impact = 1.0 - (effective_price / price_before)

        # Record trade
        self.trades.append({
            "direction": "A->B",
            "input": amount_a_in,
            "output": amount_b_out,
            "fee": fee,
            "price_impact": abs(price_impact),
            "effective_price": effective_price,
            "k_before": k_before,
            "k_after": k_after,
        })

        return amount_b_out

    def swap_b_for_a(
        self,
        trader: str,
        amount_b_in: float,
        min_a_out: float = 0.0,
        block_timestamp: int = 0,
    ) -> float:
        """
        Swap token B for token A. Mirror of swap_a_for_b.

        Returns: amount of token A received
        """
        if amount_b_in <= 0:
            raise ValueError("Input amount must be positive")
        if self.reserve_a == 0 or self.reserve_b == 0:
            raise ValueError("Pool has no liquidity")

        self._update_oracle(block_timestamp)

        k_before = self.k
        price_before = self.spot_price_b_in_a()

        amount_a_out, fee = self.get_output_amount(
            amount_b_in, self.reserve_b, self.reserve_a
        )

        if amount_a_out < min_a_out:
            raise ValueError(
                f"Slippage exceeded: would receive {amount_a_out:.6f} {self.token_a.symbol}, "
                f"minimum is {min_a_out:.6f}"
            )

        self.token_b.transfer(trader, self.POOL_ADDRESS, amount_b_in)
        self.token_a.transfer(self.POOL_ADDRESS, trader, amount_a_out)
        self.reserve_b += amount_b_in
        self.reserve_a -= amount_a_out

        self.total_fees_b += fee

        k_after = self.k
        assert k_after >= k_before - 1e-6, (
            f"Invariant violated: k went from {k_before} to {k_after}"
        )

        effective_price = amount_a_out / amount_b_in  # ETH per USDC received
        price_impact = 1.0 - (effective_price / price_before)

        self.trades.append({
            "direction": "B->A",
            "input": amount_b_in,
            "output": amount_a_out,
            "fee": fee,
            "price_impact": abs(price_impact),
            "effective_price": effective_price,
            "k_before": k_before,
            "k_after": k_after,
        })

        return amount_a_out

    def _update_oracle(self, block_timestamp: int) -> None:
        """
        Update cumulative price oracle. In Uniswap V2, this tracks the
        time-weighted sum of prices, allowing external contracts to compute
        TWAP (Time-Weighted Average Price) over any period.

        TWAP is manipulation-resistant because an attacker would need to
        hold the price at an extreme for the entire averaging period.
        """
        if block_timestamp > self.last_block_timestamp and self.reserve_a > 0 and self.reserve_b > 0:
            time_elapsed = block_timestamp - self.last_block_timestamp
            self.cumulative_price_a += (self.reserve_b / self.reserve_a) * time_elapsed
            self.cumulative_price_b += (self.reserve_a / self.reserve_b) * time_elapsed
            self.last_block_timestamp = block_timestamp

    def get_twap(
        self,
        cumulative_start_a: float,
        cumulative_start_b: float,
        time_elapsed: int,
    ) -> tuple[float, float]:
        """
        Calculate Time-Weighted Average Price over a period.

        Returns: (twap_a_in_b, twap_b_in_a)
        """
        if time_elapsed == 0:
            return (self.spot_price_a_in_b(), self.spot_price_b_in_a())
        twap_a = (self.cumulative_price_a - cumulative_start_a) / time_elapsed
        twap_b = (self.cumulative_price_b - cumulative_start_b) / time_elapsed
        return (twap_a, twap_b)

    def calculate_impermanent_loss(self, price_ratio: float) -> float:
        """
        Calculate impermanent loss for a given price change.

        price_ratio: new_price / original_price (e.g., 2.0 means price doubled)

        IL formula: 2 * sqrt(r) / (1 + r) - 1

        This always returns a negative number (it's a loss).
        At r=1 (no change): IL = 0
        At r=2 (2x): IL ≈ -5.7%
        At r=0.5 (halved): IL ≈ -5.7% (symmetric!)
        At r=5 (5x): IL ≈ -25.5%
        """
        return 2 * math.sqrt(price_ratio) / (1 + price_ratio) - 1

    def pool_status(self) -> str:
        """Human-readable pool status for debugging and display."""
        return (
            f"Pool Status:\n"
            f"  {self.token_a.symbol}: {self.reserve_a:.4f}\n"
            f"  {self.token_b.symbol}: {self.reserve_b:.4f}\n"
            f"  k = {self.k:.4f}\n"
            f"  Price: 1 {self.token_a.symbol} = {self.spot_price_a_in_b():.4f} {self.token_b.symbol}\n"
            f"  LP Supply: {self.lp_token.total_supply:.4f}\n"
            f"  Total Fees ({self.token_a.symbol}): {self.total_fees_a:.6f}\n"
            f"  Total Fees ({self.token_b.symbol}): {self.total_fees_b:.6f}\n"
            f"  Trades executed: {len(self.trades)}"
        )


def calculate_arbitrage_opportunity(
    pool: LiquidityPool,
    external_price_a_in_b: float,
) -> Optional[dict]:
    """
    Calculate the optimal arbitrage trade when pool price differs from external market.

    This is how AMM prices stay aligned with the broader market — arbitrageurs
    profit from the discrepancy, and in doing so, push the pool price toward
    the market price. This is a fundamental mechanism of DeFi.

    Returns dict with trade direction and optimal size, or None if no opportunity.
    """
    pool_price = pool.spot_price_a_in_b()

    if abs(pool_price - external_price_a_in_b) / external_price_a_in_b < 0.001:
        return None  # Less than 0.1% difference, not worth arbing

    if pool_price < external_price_a_in_b:
        # A is cheaper in the pool than on the market
        # Strategy: buy A from pool (swap B for A), sell A on market
        # Optimal amount: solve for the trade that brings pool price to market price
        # After trade: reserve_a_new / reserve_b_new = 1 / external_price
        # Using: (reserve_b + dy) * (reserve_a - dx) = k
        # And: (reserve_b + dy) / (reserve_a - dx) = external_price
        target_reserve_a = math.sqrt(pool.k / external_price_a_in_b)
        target_reserve_b = math.sqrt(pool.k * external_price_a_in_b)
        amount_b_in = target_reserve_b - pool.reserve_b
        # Account for fees
        amount_b_in = amount_b_in / (1 - pool.fee_rate)
        if amount_b_in <= 0:
            return None
        expected_a_out, _ = pool.get_output_amount(
            amount_b_in, pool.reserve_b, pool.reserve_a
        )
        profit_in_b = expected_a_out * external_price_a_in_b - amount_b_in
        return {
            "direction": "buy_A_sell_on_market",
            "input_b": amount_b_in,
            "expected_a_out": expected_a_out,
            "profit_in_b": profit_in_b,
        }
    else:
        # A is more expensive in the pool
        # Strategy: buy A on market, sell A to pool (swap A for B)
        target_reserve_a = math.sqrt(pool.k / external_price_a_in_b)
        target_reserve_b = math.sqrt(pool.k * external_price_a_in_b)
        amount_a_in = target_reserve_a - pool.reserve_a
        amount_a_in = amount_a_in / (1 - pool.fee_rate)
        if amount_a_in <= 0:
            return None
        expected_b_out, _ = pool.get_output_amount(
            amount_a_in, pool.reserve_a, pool.reserve_b
        )
        cost_in_b = amount_a_in * external_price_a_in_b
        profit_in_b = expected_b_out - cost_in_b
        return {
            "direction": "buy_A_on_market_sell_to_pool",
            "input_a": amount_a_in,
            "expected_b_out": expected_b_out,
            "profit_in_b": profit_in_b,
        }


if __name__ == "__main__":
    print("=" * 70)
    print("DEX SWAP CONTRACT — Automated Market Maker Simulation")
    print("=" * 70)

    # --- Setup tokens and initial balances ---
    eth = Token("Ethereum", "ETH")
    usdc = Token("USD Coin", "USDC")

    # Mint tokens to participants
    # Alice: liquidity provider with lots of both tokens
    eth.mint("alice", 100.0)
    usdc.mint("alice", 200_000.0)

    # Bob: a trader who wants to swap
    eth.mint("bob", 10.0)
    usdc.mint("bob", 5_000.0)

    # Charlie: another LP
    eth.mint("charlie", 50.0)
    usdc.mint("charlie", 100_000.0)

    pool = LiquidityPool(eth, usdc, fee_rate=0.003)

    # --- Step 1: Initial Liquidity ---
    print("\n--- Step 1: Alice provides initial liquidity ---")
    print(f"Alice deposits: 50 ETH + 100,000 USDC (setting price to 2000 USDC/ETH)")
    a_dep, b_dep, lp = pool.add_liquidity("alice", 50.0, 100_000.0)
    print(f"Deposited: {a_dep:.2f} ETH, {b_dep:.2f} USDC")
    print(f"LP tokens minted: {lp:.4f} (sqrt(50 * 100000) = {math.sqrt(50 * 100000):.4f})")
    print(pool.pool_status())

    # --- Step 2: A swap ---
    print("\n--- Step 2: Bob swaps 1 ETH for USDC ---")
    price_before = pool.spot_price_a_in_b()
    print(f"Spot price before: 1 ETH = {price_before:.2f} USDC")

    # Calculate expected output for demonstration
    expected_out, fee = pool.get_output_amount(1.0, pool.reserve_a, pool.reserve_b)
    print(f"Expected output: {expected_out:.4f} USDC (fee: {fee:.4f} ETH)")
    print(f"Effective price: {1.0 / expected_out * pool.reserve_b:.2f} USDC/ETH")

    usdc_received = pool.swap_a_for_b("bob", 1.0, min_b_out=1900.0)
    print(f"Bob received: {usdc_received:.4f} USDC for 1 ETH")
    print(f"Effective price: {usdc_received:.2f} USDC per ETH")
    print(f"Spot price after: 1 ETH = {pool.spot_price_a_in_b():.2f} USDC")

    # Show price impact
    price_impact = 1.0 - usdc_received / price_before
    print(f"Price impact: {price_impact * 100:.3f}%")
    print(f"k before: {pool.trades[-1]['k_before']:.2f}, k after: {pool.trades[-1]['k_after']:.2f} (increased by fees)")

    # --- Step 3: Larger swap to show slippage ---
    print("\n--- Step 3: Bob swaps 5 ETH — observe larger slippage ---")
    price_before = pool.spot_price_a_in_b()
    usdc_received_2 = pool.swap_a_for_b("bob", 5.0, min_b_out=8000.0)
    print(f"Bob received: {usdc_received_2:.4f} USDC for 5 ETH")
    print(f"Effective price: {usdc_received_2 / 5:.2f} USDC per ETH")
    print(f"vs spot price was: {price_before:.2f} USDC per ETH")
    price_impact_2 = 1.0 - (usdc_received_2 / 5) / price_before
    print(f"Price impact: {price_impact_2 * 100:.3f}% (much worse than 1 ETH swap!)")
    print(pool.pool_status())

    # --- Step 4: Charlie adds liquidity at new price ---
    print("\n--- Step 4: Charlie adds liquidity at current price ---")
    current_price = pool.spot_price_a_in_b()
    print(f"Current pool price: 1 ETH = {current_price:.2f} USDC")
    charlie_eth = 20.0
    charlie_usdc = charlie_eth * current_price  # Match ratio
    a_dep, b_dep, lp = pool.add_liquidity("charlie", charlie_eth, charlie_usdc)
    print(f"Charlie deposited: {a_dep:.4f} ETH, {b_dep:.4f} USDC")
    print(f"Charlie received: {lp:.4f} LP tokens")
    print(f"Alice LP: {pool.lp_token.balance_of('alice'):.4f}, Charlie LP: {pool.lp_token.balance_of('charlie'):.4f}")
    print(pool.pool_status())

    # --- Step 5: Swap in reverse direction ---
    print("\n--- Step 5: Bob buys ETH with 5000 USDC ---")
    price_before = pool.spot_price_b_in_a()
    eth_received = pool.swap_b_for_a("bob", 5000.0, min_a_out=2.0)
    print(f"Bob received: {eth_received:.4f} ETH for 5000 USDC")
    print(f"Effective price: {5000 / eth_received:.2f} USDC per ETH")
    print(pool.pool_status())

    # --- Step 6: Impermanent Loss calculation ---
    print("\n--- Step 6: Impermanent Loss Analysis ---")
    print("If ETH price changes after providing liquidity:")
    for ratio in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]:
        il = pool.calculate_impermanent_loss(ratio)
        direction = "down" if ratio < 1 else "up" if ratio > 1 else "unchanged"
        print(f"  {ratio:.2f}x ({direction:>9s}): IL = {il * 100:+.3f}%")

    # --- Step 7: Slippage protection demo ---
    print("\n--- Step 7: Slippage Protection ---")
    try:
        # Try to swap with unreasonably high minimum
        pool.swap_a_for_b("bob", 1.0, min_b_out=5000.0)
    except ValueError as e:
        print(f"Trade reverted (as expected): {e}")

    # --- Step 8: Arbitrage calculation ---
    print("\n--- Step 8: Arbitrage Opportunity ---")
    pool_price = pool.spot_price_a_in_b()
    external_price = pool_price * 1.05  # External market has 5% higher ETH price
    print(f"Pool price: {pool_price:.2f} USDC/ETH")
    print(f"External market: {external_price:.2f} USDC/ETH (5% higher)")

    arb = calculate_arbitrage_opportunity(pool, external_price)
    if arb:
        print(f"Arbitrage: {arb['direction']}")
        if "input_b" in arb:
            print(f"  Trade: swap {arb['input_b']:.4f} USDC for ~{arb['expected_a_out']:.4f} ETH in pool")
        else:
            print(f"  Trade: swap {arb['input_a']:.4f} ETH for ~{arb['expected_b_out']:.4f} USDC in pool")
        print(f"  Expected profit: {arb['profit_in_b']:.4f} USDC")

    # --- Step 9: Alice removes liquidity ---
    print("\n--- Step 9: Alice removes all liquidity ---")
    alice_lp = pool.lp_token.balance_of("alice")
    print(f"Alice has {alice_lp:.4f} LP tokens")
    print(f"Pool reserves before: {pool.reserve_a:.4f} ETH, {pool.reserve_b:.4f} USDC")
    eth_out, usdc_out = pool.remove_liquidity("alice", alice_lp)
    print(f"Alice received: {eth_out:.4f} ETH, {usdc_out:.4f} USDC")
    print(f"Alice originally deposited: 50.0000 ETH, 100000.0000 USDC")
    print(f"Difference: {eth_out - 50:.4f} ETH, {usdc_out - 100000:.4f} USDC")
    print(f"(Ratio changed due to trades; fees partially compensate)")
    print(pool.pool_status())

    # --- Summary ---
    print("\n--- Trade History ---")
    for i, trade in enumerate(pool.trades):
        print(
            f"  Trade {i + 1}: {trade['direction']} | "
            f"in={trade['input']:.4f} out={trade['output']:.4f} | "
            f"fee={trade['fee']:.4f} | impact={trade['price_impact'] * 100:.3f}%"
        )

    print("\n--- Total Fees Collected ---")
    print(f"  {eth.symbol}: {pool.total_fees_a:.6f}")
    print(f"  {usdc.symbol}: {pool.total_fees_b:.6f}")
    print("\nDone! The AMM correctly maintained the constant product invariant")
    print("while enabling trustless token swaps with fee accrual to LPs.")
