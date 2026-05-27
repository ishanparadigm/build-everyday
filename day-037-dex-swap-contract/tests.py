"""
Day 037: DEX Swap Contract — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import math
import unittest
from my_solution import Token, LiquidityPool, calculate_arbitrage_opportunity


class TestToken(unittest.TestCase):
    """Test ERC-20 token simulation."""

    def setUp(self):
        self.token = Token("Test", "TST")

    def test_mint_and_balance(self):
        """Minting should increase balance and total supply."""
        self.token.mint("alice", 100.0)
        self.assertEqual(self.token.balance_of("alice"), 100.0)
        self.assertEqual(self.token.total_supply, 100.0)

    def test_transfer(self):
        """Transfer should move tokens between addresses."""
        self.token.mint("alice", 100.0)
        self.token.transfer("alice", "bob", 30.0)
        self.assertAlmostEqual(self.token.balance_of("alice"), 70.0)
        self.assertAlmostEqual(self.token.balance_of("bob"), 30.0)

    def test_transfer_insufficient_balance(self):
        """Transfer should fail with insufficient balance."""
        self.token.mint("alice", 10.0)
        with self.assertRaises(ValueError):
            self.token.transfer("alice", "bob", 20.0)

    def test_burn(self):
        """Burn should reduce balance and total supply."""
        self.token.mint("alice", 100.0)
        self.token.burn("alice", 40.0)
        self.assertAlmostEqual(self.token.balance_of("alice"), 60.0)
        self.assertAlmostEqual(self.token.total_supply, 60.0)

    def test_zero_balance_for_unknown_address(self):
        """Unknown addresses should have zero balance."""
        self.assertEqual(self.token.balance_of("nobody"), 0.0)


class TestLiquidityPool(unittest.TestCase):
    """Test the constant-product AMM."""

    def setUp(self):
        """Create a pool with ETH/USDC and fund participants."""
        self.eth = Token("Ethereum", "ETH")
        self.usdc = Token("USD Coin", "USDC")

        self.eth.mint("alice", 200.0)
        self.usdc.mint("alice", 400_000.0)
        self.eth.mint("bob", 50.0)
        self.usdc.mint("bob", 100_000.0)

        self.pool = LiquidityPool(self.eth, self.usdc, fee_rate=0.003)

    def _add_initial_liquidity(self):
        """Helper: Alice adds 50 ETH + 100k USDC."""
        return self.pool.add_liquidity("alice", 50.0, 100_000.0)

    def test_initial_liquidity(self):
        """First deposit should set reserves and mint sqrt(a*b) LP tokens."""
        a, b, lp = self._add_initial_liquidity()
        self.assertAlmostEqual(a, 50.0)
        self.assertAlmostEqual(b, 100_000.0)
        self.assertAlmostEqual(lp, math.sqrt(50.0 * 100_000.0), places=4)
        self.assertAlmostEqual(self.pool.reserve_a, 50.0)
        self.assertAlmostEqual(self.pool.reserve_b, 100_000.0)

    def test_spot_price(self):
        """Spot price should be reserve_b / reserve_a."""
        self._add_initial_liquidity()
        self.assertAlmostEqual(self.pool.spot_price_a_in_b(), 2000.0)
        self.assertAlmostEqual(self.pool.spot_price_b_in_a(), 0.0005)

    def test_swap_a_for_b(self):
        """Swapping A for B should give less than spot price * amount (due to slippage)."""
        self._add_initial_liquidity()
        usdc_out = self.pool.swap_a_for_b("bob", 1.0)
        # Should be less than 2000 (spot price) due to slippage + fee
        self.assertGreater(usdc_out, 1900.0)
        self.assertLess(usdc_out, 2000.0)

    def test_swap_b_for_a(self):
        """Swapping B for A should work symmetrically."""
        self._add_initial_liquidity()
        eth_out = self.pool.swap_b_for_a("bob", 2000.0)
        # Should be less than 1.0 ETH (what spot price would give)
        self.assertGreater(eth_out, 0.9)
        self.assertLess(eth_out, 1.0)

    def test_constant_product_invariant(self):
        """k should never decrease after a swap (fees increase it)."""
        self._add_initial_liquidity()
        k_before = self.pool.k
        self.pool.swap_a_for_b("bob", 5.0)
        k_after = self.pool.k
        self.assertGreaterEqual(k_after, k_before - 1e-6)

    def test_larger_swap_more_slippage(self):
        """A larger swap should have more slippage (worse effective price)."""
        self._add_initial_liquidity()

        # Small swap
        small_out = self.pool.swap_a_for_b("bob", 1.0)
        small_price = small_out / 1.0

        # Reset pool for fair comparison
        self.pool = LiquidityPool(self.eth, self.usdc, fee_rate=0.003)
        # Re-fund pool address balances
        self.eth.mint("alice", 200.0)
        self.usdc.mint("alice", 400_000.0)
        self.pool.add_liquidity("alice", 50.0, 100_000.0)

        # Large swap
        large_out = self.pool.swap_a_for_b("bob", 10.0)
        large_price = large_out / 10.0

        self.assertGreater(small_price, large_price)

    def test_slippage_protection(self):
        """Swap should revert if output is below minimum."""
        self._add_initial_liquidity()
        with self.assertRaises(ValueError):
            self.pool.swap_a_for_b("bob", 1.0, min_b_out=5000.0)

    def test_add_liquidity_subsequent(self):
        """Subsequent liquidity additions should match current ratio."""
        self._add_initial_liquidity()
        # Bob adds liquidity
        a, b, lp = self.pool.add_liquidity("bob", 10.0, 20_000.0)
        self.assertAlmostEqual(a, 10.0)
        self.assertAlmostEqual(b, 20_000.0)
        self.assertGreater(lp, 0)

    def test_remove_liquidity(self):
        """Removing liquidity should return proportional reserves."""
        self._add_initial_liquidity()
        alice_lp = self.pool.lp_token.balance_of("alice")
        eth_out, usdc_out = self.pool.remove_liquidity("alice", alice_lp)
        self.assertAlmostEqual(eth_out, 50.0, places=4)
        self.assertAlmostEqual(usdc_out, 100_000.0, places=2)

    def test_fees_accrue_to_lps(self):
        """After trades, LPs should be able to withdraw more than they deposited (from fees)."""
        self._add_initial_liquidity()

        # Execute several trades to accumulate fees
        for _ in range(5):
            self.pool.swap_a_for_b("bob", 2.0)
            self.pool.swap_b_for_a("bob", 3000.0)

        # Alice removes all liquidity
        alice_lp = self.pool.lp_token.balance_of("alice")
        eth_out, usdc_out = self.pool.remove_liquidity("alice", alice_lp)

        # The total value should be slightly more than initial deposit
        # because fees were collected
        # Value everything in USDC at current pool price
        total_value_out = eth_out * 2000 + usdc_out
        total_value_in = 50.0 * 2000 + 100_000.0
        self.assertGreater(total_value_out, total_value_in * 0.999)

    def test_impermanent_loss(self):
        """IL formula should return correct values for known price ratios."""
        self._add_initial_liquidity()
        # At 2x price change, IL should be approximately -5.7%
        il = self.pool.calculate_impermanent_loss(2.0)
        self.assertAlmostEqual(il, -0.05719, places=4)
        # At no price change, IL should be 0
        il_flat = self.pool.calculate_impermanent_loss(1.0)
        self.assertAlmostEqual(il_flat, 0.0, places=6)


class TestArbitrage(unittest.TestCase):
    """Test arbitrage opportunity detection."""

    def setUp(self):
        self.eth = Token("Ethereum", "ETH")
        self.usdc = Token("USD Coin", "USDC")
        self.eth.mint("alice", 100.0)
        self.usdc.mint("alice", 200_000.0)
        self.pool = LiquidityPool(self.eth, self.usdc, fee_rate=0.003)
        self.pool.add_liquidity("alice", 50.0, 100_000.0)

    def test_no_arb_when_prices_match(self):
        """No arbitrage when pool price matches external price."""
        result = calculate_arbitrage_opportunity(self.pool, 2000.0)
        self.assertIsNone(result)

    def test_arb_when_pool_underpriced(self):
        """Should detect arb when pool price is lower than external."""
        result = calculate_arbitrage_opportunity(self.pool, 2200.0)
        self.assertIsNotNone(result)
        self.assertIn("direction", result)
        self.assertGreater(result.get("profit_in_b", 0), 0)

    def test_arb_when_pool_overpriced(self):
        """Should detect arb when pool price is higher than external."""
        result = calculate_arbitrage_opportunity(self.pool, 1800.0)
        self.assertIsNotNone(result)
        self.assertGreater(result.get("profit_in_b", 0), 0)


if __name__ == "__main__":
    unittest.main()
