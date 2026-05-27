"""
Tests for Day 040: Flash Loan Basics

Run with:
    python3 -m pytest tests.py
    python3 tests.py
"""

import unittest
import math
from my_solution import (
    Token,
    FlashLoanPool,
    MockExchange,
    ArbitrageBorrower,
    MaliciousBorrower,
    TransactionReverted,
    FlashLoanEvent,
)


class TestToken(unittest.TestCase):
    """Test the ERC-20 token simulation."""

    def test_initial_balance(self):
        """Deployer receives the full initial supply."""
        token = Token("Test", "TST", 1000, "alice")
        self.assertEqual(token.balance_of("alice"), 1000)

    def test_zero_balance_for_unknown(self):
        """Unknown accounts have zero balance."""
        token = Token("Test", "TST", 1000, "alice")
        self.assertEqual(token.balance_of("bob"), 0)

    def test_transfer(self):
        """Basic transfer moves tokens correctly."""
        token = Token("Test", "TST", 1000, "alice")
        token.transfer("alice", "bob", 400)
        self.assertEqual(token.balance_of("alice"), 600)
        self.assertEqual(token.balance_of("bob"), 400)

    def test_transfer_insufficient_balance(self):
        """Transfer fails if sender doesn't have enough."""
        token = Token("Test", "TST", 1000, "alice")
        with self.assertRaises(ValueError):
            token.transfer("alice", "bob", 1001)

    def test_transfer_negative_amount(self):
        """Transfer fails for negative amounts."""
        token = Token("Test", "TST", 1000, "alice")
        with self.assertRaises(ValueError):
            token.transfer("alice", "bob", -1)


class TestFlashLoanPool(unittest.TestCase):
    """Test the flash loan pool mechanics."""

    def setUp(self):
        self.token = Token("USDC", "USDC", 10_000_000, "deployer")
        self.token.transfer("deployer", "pool", 5_000_000)
        self.token.transfer("deployer", "exchange_a", 2_000_000)
        self.token.transfer("deployer", "exchange_b", 2_000_000)
        self.pool = FlashLoanPool(self.token, "pool", fee_bps=9)

    def test_available_liquidity(self):
        """Pool reports its token balance as available liquidity."""
        self.assertEqual(self.pool.available_liquidity, 5_000_000)

    def test_fee_calculation(self):
        """Fee is ceil(amount * bps / 10000)."""
        # 1,000,000 * 9 / 10000 = 900
        self.assertEqual(self.pool.fee_for(1_000_000), 900)

    def test_fee_ceiling(self):
        """Fee rounds up so pool never gets zero fee on non-zero amounts."""
        # 100 * 9 / 10000 = 0.09 → ceil = 1
        self.assertEqual(self.pool.fee_for(100), 1)

    def test_profitable_arbitrage(self):
        """Arbitrage with sufficient spread succeeds."""
        ex_a = MockExchange("A", self.token, 1.02, "exchange_a")
        ex_b = MockExchange("B", self.token, 1.00, "exchange_b")
        arb = ArbitrageBorrower("arb", ex_a, ex_b)

        event = self.pool.flash_loan(arb, "arb", 1_000_000)
        self.assertTrue(event.success)
        self.assertGreater(event.profit, 0)
        # Pool should have earned the fee
        self.assertEqual(self.pool.total_fees_earned, 900)

    def test_pool_balance_increases(self):
        """Pool balance increases by exactly the fee after a successful loan."""
        balance_before = self.pool.available_liquidity
        ex_a = MockExchange("A", self.token, 1.02, "exchange_a")
        ex_b = MockExchange("B", self.token, 1.00, "exchange_b")
        arb = ArbitrageBorrower("arb", ex_a, ex_b)

        self.pool.flash_loan(arb, "arb", 1_000_000)
        self.assertGreaterEqual(self.pool.available_liquidity, balance_before + 900)

    def test_malicious_borrower_reverts(self):
        """Borrower that doesn't repay causes revert."""
        thief = MaliciousBorrower("thief")
        balance_before = self.pool.available_liquidity

        with self.assertRaises(TransactionReverted):
            self.pool.flash_loan(thief, "thief", 1_000_000)

        # State must be fully restored
        self.assertEqual(self.pool.available_liquidity, balance_before)
        self.assertEqual(self.token.balance_of("thief"), 0)

    def test_insufficient_liquidity(self):
        """Borrowing more than pool has raises ValueError."""
        thief = MaliciousBorrower("thief")
        with self.assertRaises(ValueError):
            self.pool.flash_loan(thief, "thief", 6_000_000)

    def test_unprofitable_arb_reverts(self):
        """Arbitrage with spread < fee reverts."""
        # 0.05% spread < 0.09% fee
        ex_a = MockExchange("A", self.token, 1.0005, "exchange_a")
        ex_b = MockExchange("B", self.token, 1.0000, "exchange_b")
        arb = ArbitrageBorrower("arb", ex_a, ex_b)

        balance_before = self.pool.available_liquidity
        with self.assertRaises((TransactionReverted, ValueError)):
            self.pool.flash_loan(arb, "arb", 1_000_000)

        self.assertEqual(self.pool.available_liquidity, balance_before)

    def test_event_logging(self):
        """Pool records events for both successful and failed loans."""
        ex_a = MockExchange("A", self.token, 1.02, "exchange_a")
        ex_b = MockExchange("B", self.token, 1.00, "exchange_b")
        arb = ArbitrageBorrower("arb", ex_a, ex_b)
        thief = MaliciousBorrower("thief")

        self.pool.flash_loan(arb, "arb", 1_000_000)
        try:
            self.pool.flash_loan(thief, "thief", 500_000)
        except TransactionReverted:
            pass

        self.assertEqual(len(self.pool.events), 2)
        self.assertTrue(self.pool.events[0].success)
        self.assertFalse(self.pool.events[1].success)


if __name__ == "__main__":
    unittest.main()
