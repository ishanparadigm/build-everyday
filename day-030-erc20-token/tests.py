"""
Day 030: ERC-20 Token Tests

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
from my_solution import ERC20Token, TransferEvent, ApprovalEvent, ZERO_ADDRESS


ONE_TOKEN = 10 ** 18


class TestERC20Token(unittest.TestCase):
    """Comprehensive tests for the ERC-20 token implementation."""

    def setUp(self):
        """Deploy a fresh token for each test."""
        self.token = ERC20Token("TestToken", "TST", decimals=18, owner="owner")
        # Mint an initial supply of 1M tokens to the owner
        self.token.mint("owner", "owner", 1_000_000 * ONE_TOKEN)

    # ----- Basic metadata and state -----

    def test_metadata(self):
        """Token metadata should be set correctly at deployment."""
        self.assertEqual(self.token.name, "TestToken")
        self.assertEqual(self.token.symbol, "TST")
        self.assertEqual(self.token.decimals, 18)

    def test_initial_supply(self):
        """After minting, total supply and balance should reflect minted amount."""
        self.assertEqual(self.token.total_supply(), 1_000_000 * ONE_TOKEN)
        self.assertEqual(self.token.balance_of("owner"), 1_000_000 * ONE_TOKEN)

    def test_balance_of_unknown_address(self):
        """Unknown addresses should have zero balance, not raise errors."""
        self.assertEqual(self.token.balance_of("nonexistent"), 0)

    # ----- Transfer -----

    def test_transfer_basic(self):
        """Basic transfer should move tokens and update both balances."""
        self.token.transfer("owner", "alice", 100 * ONE_TOKEN)
        self.assertEqual(self.token.balance_of("alice"), 100 * ONE_TOKEN)
        self.assertEqual(self.token.balance_of("owner"), 999_900 * ONE_TOKEN)

    def test_transfer_insufficient_balance(self):
        """Transfer should fail if sender doesn't have enough tokens."""
        with self.assertRaises(ValueError):
            self.token.transfer("alice", "bob", 1)  # alice has 0

    def test_transfer_to_zero_address(self):
        """Transfer to zero address should be rejected (use burn instead)."""
        with self.assertRaises(ValueError):
            self.token.transfer("owner", ZERO_ADDRESS, 100)

    def test_transfer_zero_amount(self):
        """Transferring 0 tokens should succeed (it's a valid no-op)."""
        self.token.transfer("owner", "alice", 0)
        self.assertEqual(self.token.balance_of("alice"), 0)

    # ----- Approve + TransferFrom -----

    def test_approve_and_allowance(self):
        """Approve should set the correct allowance."""
        self.token.approve("owner", "spender", 500 * ONE_TOKEN)
        self.assertEqual(self.token.allowance("owner", "spender"), 500 * ONE_TOKEN)

    def test_transfer_from_basic(self):
        """TransferFrom should move tokens using the spender's allowance."""
        self.token.approve("owner", "spender", 500 * ONE_TOKEN)
        self.token.transfer_from("spender", "owner", "alice", 200 * ONE_TOKEN)

        self.assertEqual(self.token.balance_of("alice"), 200 * ONE_TOKEN)
        self.assertEqual(self.token.balance_of("owner"), 999_800 * ONE_TOKEN)
        # Allowance should decrease
        self.assertEqual(self.token.allowance("owner", "spender"), 300 * ONE_TOKEN)

    def test_transfer_from_exceeds_allowance(self):
        """TransferFrom should fail if amount exceeds allowance."""
        self.token.approve("owner", "spender", 100 * ONE_TOKEN)
        with self.assertRaises(ValueError):
            self.token.transfer_from("spender", "owner", "alice", 200 * ONE_TOKEN)

    def test_transfer_from_exceeds_balance(self):
        """TransferFrom should fail if owner doesn't have enough tokens, even with allowance."""
        # Give alice a small balance but large allowance
        self.token.transfer("owner", "alice", 10 * ONE_TOKEN)
        self.token.approve("alice", "spender", 1_000 * ONE_TOKEN)

        with self.assertRaises(ValueError):
            self.token.transfer_from("spender", "alice", "bob", 100 * ONE_TOKEN)

    # ----- Mint and Burn -----

    def test_mint_only_owner(self):
        """Only the contract owner should be able to mint."""
        with self.assertRaises(PermissionError):
            self.token.mint("alice", "alice", 100 * ONE_TOKEN)

    def test_burn_reduces_supply(self):
        """Burning should reduce both balance and total supply."""
        self.token.burn("owner", 100_000 * ONE_TOKEN)
        self.assertEqual(self.token.total_supply(), 900_000 * ONE_TOKEN)
        self.assertEqual(self.token.balance_of("owner"), 900_000 * ONE_TOKEN)

    def test_burn_exceeds_balance(self):
        """Cannot burn more tokens than you hold."""
        with self.assertRaises(ValueError):
            self.token.burn("alice", 1)  # alice has 0

    # ----- Allowance helpers -----

    def test_increase_allowance(self):
        """increaseAllowance should add to existing allowance."""
        self.token.approve("owner", "spender", 100 * ONE_TOKEN)
        self.token.increase_allowance("owner", "spender", 50 * ONE_TOKEN)
        self.assertEqual(self.token.allowance("owner", "spender"), 150 * ONE_TOKEN)

    def test_decrease_allowance(self):
        """decreaseAllowance should subtract from existing allowance."""
        self.token.approve("owner", "spender", 100 * ONE_TOKEN)
        self.token.decrease_allowance("owner", "spender", 30 * ONE_TOKEN)
        self.assertEqual(self.token.allowance("owner", "spender"), 70 * ONE_TOKEN)

    def test_decrease_allowance_below_zero(self):
        """decreaseAllowance should fail if it would go below zero."""
        self.token.approve("owner", "spender", 100 * ONE_TOKEN)
        with self.assertRaises(ValueError):
            self.token.decrease_allowance("owner", "spender", 200 * ONE_TOKEN)

    # ----- Events -----

    def test_events_emitted(self):
        """Transfers and approvals should emit the correct events."""
        token = ERC20Token("T", "T", 18, "owner")
        token.mint("owner", "owner", 100)
        token.transfer("owner", "alice", 50)
        token.approve("alice", "bob", 30)

        events = token.get_events()
        self.assertEqual(len(events), 3)
        self.assertIsInstance(events[0], TransferEvent)  # mint
        self.assertEqual(events[0].from_addr, ZERO_ADDRESS)
        self.assertIsInstance(events[1], TransferEvent)  # transfer
        self.assertIsInstance(events[2], ApprovalEvent)  # approve

    # ----- Invariant -----

    def test_supply_invariant(self):
        """Sum of all balances must always equal total supply."""
        self.token.transfer("owner", "alice", 100 * ONE_TOKEN)
        self.token.transfer("owner", "bob", 200 * ONE_TOKEN)
        self.token.burn("owner", 50 * ONE_TOKEN)
        self.token.mint("owner", "charlie", 25 * ONE_TOKEN)

        total = sum(self.token.balance_of(a) for a in ["owner", "alice", "bob", "charlie"])
        self.assertEqual(total, self.token.total_supply())


if __name__ == "__main__":
    unittest.main()
