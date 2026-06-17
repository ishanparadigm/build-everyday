"""
Day 77: Uniswap V3 Pool Analytics Engine — Your Implementation

Build a complete analytics engine for Uniswap V3 concentrated liquidity pools.
Implement tick math, position modeling, swap simulation, and fee/IL analysis.

Key concepts to remember:
- V3 uses √P (square root of price) internally, not price directly
- Each tick i represents price p(i) = 1.0001^i
- Liquidity L is constant between ticks, changes at tick boundaries
- Fees accumulate per unit of active liquidity
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# Constants
TICK_BASE = 1.0001
FEE_TIERS = {
    100: 1,       # 0.01% fee, tick spacing 1
    500: 10,      # 0.05% fee, tick spacing 10
    3000: 60,     # 0.30% fee, tick spacing 60
    10000: 200,   # 1.00% fee, tick spacing 200
}
MIN_TICK = -887272
MAX_TICK = 887272


# =============================================================================
# Step 1: Tick and Price Utilities
# Hint: The fundamental relationship is p(i) = 1.0001^i
# Hint: V3 stores √P because swap math becomes linear in √P
# =============================================================================

def tick_to_price(tick: int) -> float:
    """
    Convert a tick index to a price.
    p(i) = 1.0001^i
    """
    raise NotImplementedError("TODO: implement this")


def price_to_tick(price: float) -> int:
    """
    Convert a price to the nearest tick index (rounding down).
    i = floor(log(p) / log(1.0001))
    """
    raise NotImplementedError("TODO: implement this")


def tick_to_sqrt_price(tick: int) -> float:
    """
    Convert a tick to √P (square root of price).
    √P(i) = 1.0001^(i/2)
    """
    raise NotImplementedError("TODO: implement this")


def sqrt_price_to_tick(sqrt_price: float) -> int:
    """
    Convert √P back to a tick index.
    Hint: √P² = P, then use price_to_tick logic.
    """
    raise NotImplementedError("TODO: implement this")


def nearest_usable_tick(tick: int, tick_spacing: int) -> int:
    """
    Round a tick down to the nearest multiple of tick_spacing.
    Hint: Integer division then multiply back.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Step 2: Pool State Model
# Hint: The pool tracks current √P, active liquidity, and tick-level deltas
# =============================================================================

@dataclass
class TickState:
    """State stored at each initialized tick boundary."""
    liquidity_gross: float = 0.0
    liquidity_net: float = 0.0
    fee_growth_outside_0: float = 0.0
    fee_growth_outside_1: float = 0.0


@dataclass
class Pool:
    """Complete state of a Uniswap V3 pool."""
    token0_symbol: str
    token1_symbol: str
    fee: int
    tick_spacing: int
    sqrt_price: float
    tick: int
    liquidity: float
    fee_growth_global_0: float = 0.0
    fee_growth_global_1: float = 0.0
    ticks: dict[int, TickState] = field(default_factory=dict)
    volume_token0: float = 0.0
    volume_token1: float = 0.0

    @property
    def current_price(self) -> float:
        """Price of token0 in terms of token1."""
        raise NotImplementedError("TODO: implement this")

    def initialize_tick(self, tick: int) -> TickState:
        """Get or create tick state."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Step 3: Position Modeling
# Hint: Token amounts depend on whether price is below, in, or above the range
# Hint: Below range = all token0; Above range = all token1; In range = mix
# =============================================================================

@dataclass
class Position:
    """An LP position in a V3 pool."""
    owner: str
    tick_lower: int
    tick_upper: int
    liquidity: float
    fee_growth_inside_last_0: float = 0.0
    fee_growth_inside_last_1: float = 0.0
    tokens_owed_0: float = 0.0
    tokens_owed_1: float = 0.0

    @property
    def price_lower(self) -> float:
        return tick_to_price(self.tick_lower)

    @property
    def price_upper(self) -> float:
        return tick_to_price(self.tick_upper)


def calculate_position_amounts(
    liquidity: float,
    sqrt_price_current: float,
    sqrt_price_lower: float,
    sqrt_price_upper: float,
) -> tuple[float, float]:
    """
    Calculate the token0 and token1 amounts held by a position.

    Three cases based on where current price falls relative to range:
    - Below range: x = L × (1/√p_a - 1/√p_b), y = 0
    - Above range: x = 0, y = L × (√p_b - √p_a)
    - In range: x = L × (1/√p - 1/√p_b), y = L × (√p - √p_a)

    Hint: Think about WHY these formulas work — as price rises through
    the range, token0 is continuously sold for token1.
    """
    raise NotImplementedError("TODO: implement this")


def calculate_liquidity_from_amounts(
    amount0: float,
    amount1: float,
    sqrt_price_current: float,
    sqrt_price_lower: float,
    sqrt_price_upper: float,
) -> float:
    """
    Calculate the maximum liquidity mintable from given token amounts.

    Hint: This is the inverse of calculate_position_amounts.
    Hint: For in-range, take the MINIMUM of the two liquidity values.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Step 4: Swap Simulation
# Hint: Between ticks, the pool is a simple constant-product AMM
# Hint: At tick boundaries, active liquidity changes
# Hint: The swap loop processes one tick range at a time
# =============================================================================

@dataclass
class SwapResult:
    """Result of a swap through the pool."""
    amount_in: float
    amount_out: float
    fee_amount: float
    price_before: float
    price_after: float
    ticks_crossed: int
    price_impact: float


def swap(pool: Pool, amount_in: float, zero_for_one: bool) -> SwapResult:
    """
    Simulate a swap through a V3 pool.

    Algorithm:
    1. Start at current √P with current liquidity L
    2. Compute input needed to reach next tick boundary
    3. If enough input: cross tick, update L, continue
       If not: compute final √P within this range, stop
    4. Accumulate fees and output along the way

    Hint: For zero_for_one (selling token0):
      - Price DECREASES (√P goes down)
      - Δx = L × |1/√P_new - 1/√P_old|  (token0 input)
      - Δy = L × |√P_old - √P_new|       (token1 output)

    Hint: For one_for_zero (selling token1):
      - Price INCREASES (√P goes up)
      - Δy = L × |√P_new - √P_old|       (token1 input)
      - Δx = L × |1/√P_old - 1/√P_new|   (token0 output)
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Step 5: Liquidity Management
# Hint: Update tick states at position boundaries (liquidity_net)
# Hint: Only add to pool.liquidity if position is currently in range
# =============================================================================

def mint_position(
    pool: Pool,
    owner: str,
    tick_lower: int,
    tick_upper: int,
    amount0_desired: float,
    amount1_desired: float,
) -> tuple[Position, float, float]:
    """
    Add a new liquidity position to the pool.

    Steps:
    1. Snap ticks to valid spacing
    2. Calculate liquidity from amounts
    3. Update tick states (liquidity_net at boundaries)
    4. Update active liquidity if in range
    5. Return position and actual amounts used

    Hint: At tick_lower, liquidity_net += L (enters when crossing up)
    Hint: At tick_upper, liquidity_net -= L (exits when crossing up)
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Step 6: Fee Analytics
# Hint: Fees per position = L × (feeGrowthInside_now - feeGrowthInside_last)
# =============================================================================

def calculate_position_fees(pool: Pool, position: Position) -> tuple[float, float]:
    """
    Calculate uncollected fees for a position.

    Hint: Use feeGrowthGlobal as an approximation for feeGrowthInside
    when the position is in range. Full implementation would track
    feeGrowthOutside per tick.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Step 7: Impermanent Loss Calculator
# Hint: IL = (value_LP / value_HOLD) - 1
# Hint: Value_HOLD = initial amounts × current price
# Hint: Value_LP = current position amounts × current price
# =============================================================================

def calculate_impermanent_loss(
    liquidity: float,
    sqrt_price_entry: float,
    sqrt_price_current: float,
    sqrt_price_lower: float,
    sqrt_price_upper: float,
) -> dict:
    """
    Calculate impermanent loss for a V3 position.

    Returns dict with: price_entry, price_current, price_change_pct,
    value_lp, value_hold, impermanent_loss_pct, amounts_entry, amounts_current
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Main: Test your implementation
# =============================================================================

if __name__ == "__main__":
    print("=== Testing Tick Math ===")
    # Test tick-price conversions
    assert abs(tick_to_price(0) - 1.0) < 1e-10, "Tick 0 should be price 1.0"
    print(f"  Tick 0 → Price: {tick_to_price(0)}")
    print(f"  Tick 6000 → Price: {tick_to_price(6000):.4f}")
    print(f"  Price 3000 → Tick: {price_to_tick(3000.0)}")

    print("\n=== Creating Pool ===")
    initial_tick = price_to_tick(3000.0)
    pool = Pool(
        token0_symbol="ETH",
        token1_symbol="USDC",
        fee=3000,
        tick_spacing=60,
        sqrt_price=tick_to_sqrt_price(initial_tick),
        tick=initial_tick,
        liquidity=0.0,
    )
    print(f"  Price: {pool.current_price:.2f}")

    print("\n=== Adding Liquidity ===")
    tick_lower = nearest_usable_tick(price_to_tick(2700), 60)
    tick_upper = nearest_usable_tick(price_to_tick(3300), 60)
    pos, amt0, amt1 = mint_position(
        pool, "Alice", tick_lower, tick_upper, 10.0, 30000.0
    )
    print(f"  Deposited: {amt0:.4f} ETH + {amt1:.2f} USDC")
    print(f"  Liquidity: {pos.liquidity:.2f}")

    print("\n=== Simulating Swap ===")
    result = swap(pool, 5000.0, zero_for_one=False)
    print(f"  Bought {result.amount_out:.6f} ETH for 5000 USDC")
    print(f"  Price: {result.price_before:.2f} → {result.price_after:.2f}")
    print(f"  Fee: {result.fee_amount:.4f}")

    print("\n=== Impermanent Loss ===")
    il = calculate_impermanent_loss(
        pos.liquidity,
        tick_to_sqrt_price(initial_tick),
        pool.sqrt_price,
        tick_to_sqrt_price(pos.tick_lower),
        tick_to_sqrt_price(pos.tick_upper),
    )
    print(f"  Price change: {il['price_change_pct']:.2f}%")
    print(f"  IL: {il['impermanent_loss_pct']:.4f}%")

    print("\n✓ All checks passed!")
