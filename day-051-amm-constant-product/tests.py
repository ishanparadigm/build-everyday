"""
Tests for Day 51: Automated Market Maker — Constant Product Formula

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import unittest
from my_solution import (
    LiquidityPool,
    SwapResult,
    LiquidityEvent,
    calculate_impermanent_loss,
    calculate_arbitrage_trade,
)


class TestLiquidityPool(unittest.TestCase):
    """Test the core AMM pool functionality."""

    def setUp(self):
        """Create a standard pool for testing."""
        self.pool = LiquidityPool(fee_rate=0.003)
        self.pool.add_liquidity("alice", 100.0, 200000.0)

    def test_initial_reserves(self):
        """Pool should hold the deposited reserves."""
        self.assertAlmostEqual(self.pool.reserve_a, 100.0)
        self.assertAlmostEqual(self.pool.reserve_b, 200000.0)

    def test_constant_product(self):
        """k should equal reserve_a * reserve_b."""
        self.assertAlmostEqual(self.pool.k, 100.0 * 200000.0)

    def test_spot_price(self):
        """Spot price should be the ratio of reserves."""
        self.assertAlmostEqual(self.pool.spot_price_a_in_b, 2000.0)
        self.assertAlmostEqual(self.pool.spot_price_b_in_a, 0.0005)

    def test_lp_tokens_first_deposit(self):
        """First deposit should mint sqrt(a * b) LP tokens."""
        pool = LiquidityPool()
        event = pool.add_liquidity("alice", 100.0, 200000.0)
        expected = math.sqrt(100.0 * 200000.0)
        self.assertAlmostEqual(event.lp_tokens, expected)

    def test_lp_tokens_proportional_deposit(self):
        """Second deposit should mint proportional LP tokens."""
        initial_supply = self.pool.total_lp_supply
        event = self.pool.add_liquidity("bob", 50.0, 100000.0)
        # 50% of existing reserves → 50% of existing supply
        self.assertAlmostEqual(event.lp_tokens, initial_supply * 0.5, places=5)

    def test_swap_output_correct(self):
        """Swap should follow constant product formula with fees."""
        amount_in = 1000.0  # 1000 USDC
        fee = amount_in * 0.003
        effective = amount_in - fee
        expected_out = 100.0 * effective / (200000.0 + effective)

        result = self.pool.swap_b_for_a(1000.0)
        self.assertAlmostEqual(result.amount_out, expected_out, places=8)

    def test_k_increases_after_swap(self):
        """k should increase after a swap due to fees staying in the pool."""
        k_before = self.pool.k
        self.pool.swap_a_for_b(1.0)
        self.assertGreater(self.pool.k, k_before)

    def test_swap_price_impact_increases_with_size(self):
        """Larger trades should have more price impact."""
        pool1 = LiquidityPool(fee_rate=0.003)
        pool1.add_liquidity("lp", 100.0, 200000.0)
        pool2 = LiquidityPool(fee_rate=0.003)
        pool2.add_liquidity("lp", 100.0, 200000.0)

        small = pool1.swap_a_for_b(1.0)
        large = pool2.swap_a_for_b(10.0)
        self.assertGreater(large.price_impact, small.price_impact)

    def test_remove_liquidity_returns_proportional(self):
        """Removing all LP tokens should drain the pool."""
        lp_tokens = self.pool.lp_balances["alice"]
        event = self.pool.remove_liquidity("alice", lp_tokens)
        self.assertAlmostEqual(self.pool.reserve_a, 0.0, places=10)
        self.assertAlmostEqual(self.pool.reserve_b, 0.0, places=10)

    def test_remove_partial_liquidity(self):
        """Removing half LP tokens should return half of reserves."""
        reserve_a_before = self.pool.reserve_a
        reserve_b_before = self.pool.reserve_b
        lp_tokens = self.pool.lp_balances["alice"]

        event = self.pool.remove_liquidity("alice", lp_tokens / 2)
        self.assertAlmostEqual(event.token_a_amount, reserve_a_before / 2, places=8)
        self.assertAlmostEqual(event.token_b_amount, reserve_b_before / 2, places=8)


class TestImpermanentLoss(unittest.TestCase):
    """Test the impermanent loss calculator."""

    def test_no_price_change(self):
        """No price change = no impermanent loss."""
        self.assertAlmostEqual(calculate_impermanent_loss(1.0), 0.0)

    def test_price_doubles(self):
        """2x price → ~5.7% IL."""
        il = calculate_impermanent_loss(2.0)
        self.assertAlmostEqual(il, 2 * math.sqrt(2) / 3 - 1, places=10)
        self.assertAlmostEqual(il, -0.05719, places=4)

    def test_price_halves(self):
        """0.5x price should give same IL as 2x (symmetry around sqrt)."""
        il_up = calculate_impermanent_loss(2.0)
        il_down = calculate_impermanent_loss(0.5)
        self.assertAlmostEqual(il_up, il_down, places=10)

    def test_il_always_negative(self):
        """IL should always be <= 0 for any price ratio."""
        for r in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            self.assertLessEqual(calculate_impermanent_loss(r), 0.0 + 1e-15)

    def test_extreme_price_change(self):
        """5x price → ~25.5% IL."""
        il = calculate_impermanent_loss(5.0)
        self.assertAlmostEqual(il, -0.25347, places=4)


class TestArbitrage(unittest.TestCase):
    """Test the arbitrage calculator."""

    def test_no_arb_at_fair_price(self):
        """No arbitrage when pool price matches external price."""
        pool = LiquidityPool(fee_rate=0.003)
        pool.add_liquidity("lp", 100.0, 200000.0)
        result = calculate_arbitrage_trade(pool, 2000.0)
        self.assertIsNone(result)

    def test_arb_when_price_higher(self):
        """Should sell B (buy A) when external A price is higher."""
        pool = LiquidityPool(fee_rate=0.003)
        pool.add_liquidity("lp", 100.0, 200000.0)
        result = calculate_arbitrage_trade(pool, 2500.0)
        self.assertIsNotNone(result)
        direction, amount = result
        self.assertEqual(direction, "sell_b")
        self.assertGreater(amount, 0)

    def test_arb_moves_price_toward_external(self):
        """Executing the arb trade should move pool price toward external."""
        pool = LiquidityPool(fee_rate=0.003)
        pool.add_liquidity("lp", 100.0, 200000.0)
        external = 2500.0

        result = calculate_arbitrage_trade(pool, external)
        if result:
            direction, amount = result
            if direction == "sell_b":
                pool.swap_b_for_a(amount)
            else:
                pool.swap_a_for_b(amount)

            # Price should be closer to external after arb
            self.assertGreater(pool.spot_price_a_in_b, 2000.0)


class TestEdgeCases(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_swap_on_empty_pool(self):
        """Swapping on an empty pool should raise."""
        pool = LiquidityPool()
        with self.assertRaises(ValueError):
            pool.swap_a_for_b(1.0)

    def test_negative_deposit(self):
        """Negative deposit amounts should raise."""
        pool = LiquidityPool()
        with self.assertRaises(ValueError):
            pool.add_liquidity("alice", -1.0, 100.0)

    def test_remove_more_than_balance(self):
        """Removing more LP tokens than balance should raise."""
        pool = LiquidityPool()
        pool.add_liquidity("alice", 100.0, 100.0)
        with self.assertRaises(ValueError):
            pool.remove_liquidity("alice", 999999.0)

    def test_get_quote_matches_swap(self):
        """Quote should match actual swap output."""
        pool = LiquidityPool(fee_rate=0.003)
        pool.add_liquidity("lp", 100.0, 200000.0)

        quote_out, quote_impact = pool.get_quote(1000.0, "B")
        result = pool.swap_b_for_a(1000.0)

        self.assertAlmostEqual(quote_out, result.amount_out, places=8)


if __name__ == "__main__":
    unittest.main()
