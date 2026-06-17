"""
Day 77: Uniswap V3 Pool Analytics — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import unittest
from my_solution import (
    tick_to_price,
    price_to_tick,
    tick_to_sqrt_price,
    sqrt_price_to_tick,
    nearest_usable_tick,
    Pool,
    TickState,
    Position,
    calculate_position_amounts,
    calculate_liquidity_from_amounts,
    swap,
    mint_position,
    calculate_impermanent_loss,
)


class TestTickMath(unittest.TestCase):
    """Test tick ↔ price conversion utilities."""

    def test_tick_zero_is_price_one(self):
        """Tick 0 corresponds to price 1.0 (the identity)."""
        self.assertAlmostEqual(tick_to_price(0), 1.0, places=10)

    def test_positive_tick_gives_price_above_one(self):
        """Positive ticks give prices > 1 (1.0001^positive > 1)."""
        price = tick_to_price(1000)
        self.assertGreater(price, 1.0)
        expected = 1.0001 ** 1000
        self.assertAlmostEqual(price, expected, places=6)

    def test_negative_tick_gives_price_below_one(self):
        """Negative ticks give prices < 1 (1.0001^negative < 1)."""
        price = tick_to_price(-1000)
        self.assertLess(price, 1.0)
        expected = 1.0001 ** (-1000)
        self.assertAlmostEqual(price, expected, places=6)

    def test_price_to_tick_roundtrip(self):
        """Converting price → tick → price should be close to original."""
        for target_price in [0.5, 1.0, 100.0, 3000.0, 50000.0]:
            tick = price_to_tick(target_price)
            recovered = tick_to_price(tick)
            # Floor rounding means recovered ≤ target_price
            self.assertLessEqual(recovered, target_price * 1.001)
            self.assertGreaterEqual(recovered, target_price * 0.999)

    def test_sqrt_price_consistency(self):
        """√P should be the square root of the price at that tick."""
        for tick in [-5000, 0, 5000, 80000]:
            price = tick_to_price(tick)
            sqrt_p = tick_to_sqrt_price(tick)
            self.assertAlmostEqual(sqrt_p ** 2, price, places=4)

    def test_sqrt_price_to_tick_roundtrip(self):
        """sqrt_price_to_tick should invert tick_to_sqrt_price."""
        for tick in [-6000, -60, 0, 60, 6000]:
            sqrt_p = tick_to_sqrt_price(tick)
            recovered_tick = sqrt_price_to_tick(sqrt_p)
            self.assertEqual(recovered_tick, tick)

    def test_nearest_usable_tick(self):
        """Ticks should snap down to nearest multiple of spacing."""
        self.assertEqual(nearest_usable_tick(65, 60), 60)
        self.assertEqual(nearest_usable_tick(119, 60), 60)
        self.assertEqual(nearest_usable_tick(120, 60), 120)
        self.assertEqual(nearest_usable_tick(-1, 60), -60)
        self.assertEqual(nearest_usable_tick(0, 60), 0)


class TestPositionMath(unittest.TestCase):
    """Test position amount calculations."""

    def _make_pool_at_price(self, price: float) -> Pool:
        tick = price_to_tick(price)
        return Pool("A", "B", 3000, 60,
                     tick_to_sqrt_price(tick), tick, 0.0)

    def test_below_range_all_token0(self):
        """When price is below range, position is entirely token0."""
        sqrt_lower = tick_to_sqrt_price(price_to_tick(2000.0))
        sqrt_upper = tick_to_sqrt_price(price_to_tick(3000.0))
        sqrt_current = tick_to_sqrt_price(price_to_tick(1500.0))  # Below range

        amt0, amt1 = calculate_position_amounts(1000.0, sqrt_current, sqrt_lower, sqrt_upper)
        self.assertGreater(amt0, 0)
        self.assertAlmostEqual(amt1, 0.0, places=10)

    def test_above_range_all_token1(self):
        """When price is above range, position is entirely token1."""
        sqrt_lower = tick_to_sqrt_price(price_to_tick(2000.0))
        sqrt_upper = tick_to_sqrt_price(price_to_tick(3000.0))
        sqrt_current = tick_to_sqrt_price(price_to_tick(4000.0))  # Above range

        amt0, amt1 = calculate_position_amounts(1000.0, sqrt_current, sqrt_lower, sqrt_upper)
        self.assertAlmostEqual(amt0, 0.0, places=10)
        self.assertGreater(amt1, 0)

    def test_in_range_both_tokens(self):
        """When price is in range, position holds both tokens."""
        sqrt_lower = tick_to_sqrt_price(price_to_tick(2000.0))
        sqrt_upper = tick_to_sqrt_price(price_to_tick(4000.0))
        sqrt_current = tick_to_sqrt_price(price_to_tick(3000.0))  # In range

        amt0, amt1 = calculate_position_amounts(100000.0, sqrt_current, sqrt_lower, sqrt_upper)
        self.assertGreater(amt0, 0)
        self.assertGreater(amt1, 0)

    def test_liquidity_from_amounts_roundtrip(self):
        """Minting then checking amounts should be consistent."""
        sqrt_lower = tick_to_sqrt_price(price_to_tick(2500.0))
        sqrt_upper = tick_to_sqrt_price(price_to_tick(3500.0))
        sqrt_current = tick_to_sqrt_price(price_to_tick(3000.0))

        liq = calculate_liquidity_from_amounts(
            10.0, 30000.0, sqrt_current, sqrt_lower, sqrt_upper
        )
        amt0, amt1 = calculate_position_amounts(liq, sqrt_current, sqrt_lower, sqrt_upper)
        # At least one of the amounts should match what we put in (the binding one)
        self.assertTrue(amt0 <= 10.0 + 0.01 and amt1 <= 30000.0 + 1.0)


class TestMintPosition(unittest.TestCase):
    """Test adding liquidity to a pool."""

    def test_mint_creates_position_with_liquidity(self):
        """Minting should return a position with positive liquidity."""
        tick = price_to_tick(3000.0)
        pool = Pool("ETH", "USDC", 3000, 60,
                     tick_to_sqrt_price(tick), tick, 0.0)

        tick_lower = nearest_usable_tick(price_to_tick(2700), 60)
        tick_upper = nearest_usable_tick(price_to_tick(3300), 60)

        pos, amt0, amt1 = mint_position(pool, "Alice", tick_lower, tick_upper, 10.0, 30000.0)
        self.assertGreater(pos.liquidity, 0)
        self.assertGreater(amt0, 0)
        self.assertGreater(amt1, 0)

    def test_mint_updates_pool_liquidity(self):
        """In-range mint should increase pool's active liquidity."""
        tick = price_to_tick(3000.0)
        pool = Pool("ETH", "USDC", 3000, 60,
                     tick_to_sqrt_price(tick), tick, 0.0)

        tick_lower = nearest_usable_tick(price_to_tick(2700), 60)
        tick_upper = nearest_usable_tick(price_to_tick(3300), 60)

        pos, _, _ = mint_position(pool, "Alice", tick_lower, tick_upper, 10.0, 30000.0)
        self.assertAlmostEqual(pool.liquidity, pos.liquidity, places=2)

    def test_mint_out_of_range_no_active_liquidity(self):
        """Out-of-range mint should NOT change pool's active liquidity."""
        tick = price_to_tick(3000.0)
        pool = Pool("ETH", "USDC", 3000, 60,
                     tick_to_sqrt_price(tick), tick, 0.0)

        # Position entirely above current price
        tick_lower = nearest_usable_tick(price_to_tick(3500), 60)
        tick_upper = nearest_usable_tick(price_to_tick(4000), 60)

        mint_position(pool, "Bob", tick_lower, tick_upper, 10.0, 0.0)
        self.assertAlmostEqual(pool.liquidity, 0.0, places=2)


class TestSwap(unittest.TestCase):
    """Test swap simulation."""

    def _make_pool_with_liquidity(self) -> Pool:
        tick = price_to_tick(3000.0)
        pool = Pool("ETH", "USDC", 3000, 60,
                     tick_to_sqrt_price(tick), tick, 0.0)
        tick_lower = nearest_usable_tick(price_to_tick(2000), 60)
        tick_upper = nearest_usable_tick(price_to_tick(5000), 60)
        mint_position(pool, "LP", tick_lower, tick_upper, 100.0, 300000.0)
        return pool

    def test_swap_produces_output(self):
        """A swap should produce non-zero output."""
        pool = self._make_pool_with_liquidity()
        result = swap(pool, 1000.0, zero_for_one=False)
        self.assertGreater(result.amount_out, 0)

    def test_swap_charges_fee(self):
        """A swap should charge fees."""
        pool = self._make_pool_with_liquidity()
        result = swap(pool, 1000.0, zero_for_one=False)
        self.assertGreater(result.fee_amount, 0)
        # Fee should be approximately 0.3% of input
        expected_fee = 1000.0 * 0.003
        self.assertAlmostEqual(result.fee_amount, expected_fee, delta=expected_fee * 0.1)

    def test_swap_moves_price_correctly(self):
        """Buying token0 (zero_for_one=False) should increase price."""
        pool = self._make_pool_with_liquidity()
        price_before = pool.current_price
        swap(pool, 5000.0, zero_for_one=False)
        self.assertGreater(pool.current_price, price_before)

    def test_swap_price_decrease(self):
        """Selling token0 (zero_for_one=True) should decrease price."""
        pool = self._make_pool_with_liquidity()
        price_before = pool.current_price
        swap(pool, 2.0, zero_for_one=True)
        self.assertLess(pool.current_price, price_before)


class TestImpermanentLoss(unittest.TestCase):
    """Test impermanent loss calculations."""

    def test_no_price_change_no_il(self):
        """If price hasn't moved, IL should be zero."""
        sqrt_entry = tick_to_sqrt_price(price_to_tick(3000.0))
        sqrt_lower = tick_to_sqrt_price(price_to_tick(2500.0))
        sqrt_upper = tick_to_sqrt_price(price_to_tick(3500.0))

        il = calculate_impermanent_loss(1000.0, sqrt_entry, sqrt_entry, sqrt_lower, sqrt_upper)
        self.assertAlmostEqual(il["impermanent_loss_pct"], 0.0, places=2)

    def test_price_change_causes_il(self):
        """A price change should cause negative IL."""
        sqrt_entry = tick_to_sqrt_price(price_to_tick(3000.0))
        sqrt_current = tick_to_sqrt_price(price_to_tick(3500.0))
        sqrt_lower = tick_to_sqrt_price(price_to_tick(2500.0))
        sqrt_upper = tick_to_sqrt_price(price_to_tick(4000.0))

        il = calculate_impermanent_loss(1000.0, sqrt_entry, sqrt_current, sqrt_lower, sqrt_upper)
        self.assertLess(il["impermanent_loss_pct"], 0)


if __name__ == "__main__":
    unittest.main()
