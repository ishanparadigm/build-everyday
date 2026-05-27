"""
Day 039: Multisig Wallet — Test Suite

Run with: python3 -m pytest tests.py -v
     or: python3 tests.py
"""

import unittest
from my_solution import MultisigWallet, TxStatus


class TestWalletCreation(unittest.TestCase):
    """Test wallet initialization and validation."""

    def test_basic_creation(self):
        """2-of-3 wallet initializes correctly."""
        w = MultisigWallet(owners=["alice", "bob", "carol"], required=2)
        self.assertEqual(w.owners, ["alice", "bob", "carol"])
        self.assertEqual(w.required, 2)
        self.assertEqual(w.balance, 0)
        self.assertEqual(w.transaction_count, 0)

    def test_no_owners_rejected(self):
        """Empty owner list is rejected."""
        with self.assertRaises(ValueError):
            MultisigWallet(owners=[], required=1)

    def test_threshold_too_high(self):
        """Required > owner count is rejected."""
        with self.assertRaises(ValueError):
            MultisigWallet(owners=["alice", "bob"], required=3)

    def test_threshold_zero_rejected(self):
        """Required = 0 is rejected."""
        with self.assertRaises(ValueError):
            MultisigWallet(owners=["alice"], required=0)

    def test_duplicate_owners_rejected(self):
        """Duplicate owner addresses are rejected."""
        with self.assertRaises(ValueError):
            MultisigWallet(owners=["alice", "alice"], required=1)

    def test_empty_address_rejected(self):
        """Empty string owner address is rejected."""
        with self.assertRaises(ValueError):
            MultisigWallet(owners=["alice", ""], required=1)


class TestDeposit(unittest.TestCase):
    """Test deposit functionality."""

    def test_deposit_increases_balance(self):
        w = MultisigWallet(owners=["alice"], required=1)
        w.deposit("anyone", 500)
        self.assertEqual(w.balance, 500)

    def test_multiple_deposits(self):
        w = MultisigWallet(owners=["alice"], required=1)
        w.deposit("a", 100)
        w.deposit("b", 200)
        self.assertEqual(w.balance, 300)

    def test_zero_deposit_rejected(self):
        w = MultisigWallet(owners=["alice"], required=1)
        with self.assertRaises(ValueError):
            w.deposit("anyone", 0)


class TestTransactionLifecycle(unittest.TestCase):
    """Test the full submit → confirm → execute flow."""

    def setUp(self):
        self.w = MultisigWallet(owners=["alice", "bob", "carol"], required=2)
        self.w.deposit("funder", 1000)

    def test_submit_returns_sequential_ids(self):
        id0 = self.w.submit_transaction("alice", "dest1", value=10)
        id1 = self.w.submit_transaction("bob", "dest2", value=20)
        self.assertEqual(id0, 0)
        self.assertEqual(id1, 1)

    def test_non_owner_cannot_submit(self):
        with self.assertRaises(PermissionError):
            self.w.submit_transaction("mallory", "dest", value=10)

    def test_confirm_and_execute(self):
        """Full lifecycle: submit, confirm x2, execute."""
        tx_id = self.w.submit_transaction("alice", "dave", value=300)
        self.w.confirm_transaction("alice", tx_id)
        self.w.confirm_transaction("bob", tx_id)
        success = self.w.execute_transaction("alice", tx_id)
        self.assertTrue(success)
        self.assertEqual(self.w.balance, 700)
        self.assertEqual(self.w.get_transaction(tx_id).status, TxStatus.EXECUTED)

    def test_cannot_execute_without_threshold(self):
        """Execution fails if confirmations < required."""
        tx_id = self.w.submit_transaction("alice", "dave", value=100)
        self.w.confirm_transaction("alice", tx_id)
        with self.assertRaises(ValueError):
            self.w.execute_transaction("alice", tx_id)

    def test_double_confirm_rejected(self):
        """Same owner cannot confirm twice."""
        tx_id = self.w.submit_transaction("alice", "dave", value=100)
        self.w.confirm_transaction("alice", tx_id)
        with self.assertRaises(ValueError):
            self.w.confirm_transaction("alice", tx_id)

    def test_double_execute_rejected(self):
        """Cannot execute an already-executed transaction."""
        tx_id = self.w.submit_transaction("alice", "dave", value=100)
        self.w.confirm_transaction("alice", tx_id)
        self.w.confirm_transaction("bob", tx_id)
        self.w.execute_transaction("alice", tx_id)
        with self.assertRaises(ValueError):
            self.w.execute_transaction("bob", tx_id)

    def test_insufficient_balance_fails(self):
        """Execution fails (returns False) when balance is too low."""
        tx_id = self.w.submit_transaction("alice", "greedy", value=9999)
        self.w.confirm_transaction("alice", tx_id)
        self.w.confirm_transaction("bob", tx_id)
        success = self.w.execute_transaction("alice", tx_id)
        self.assertFalse(success)
        self.assertEqual(self.w.get_transaction(tx_id).status, TxStatus.FAILED)
        self.assertEqual(self.w.balance, 1000)  # Balance unchanged


class TestRevocation(unittest.TestCase):
    """Test confirmation revocation."""

    def setUp(self):
        self.w = MultisigWallet(owners=["alice", "bob", "carol"], required=2)
        self.w.deposit("funder", 1000)

    def test_revoke_removes_confirmation(self):
        tx_id = self.w.submit_transaction("alice", "dave", value=100)
        self.w.confirm_transaction("alice", tx_id)
        self.w.confirm_transaction("bob", tx_id)
        self.w.revoke_confirmation("bob", tx_id)
        self.assertEqual(self.w.get_transaction(tx_id).confirmation_count, 1)

    def test_revoke_prevents_execution(self):
        tx_id = self.w.submit_transaction("alice", "dave", value=100)
        self.w.confirm_transaction("alice", tx_id)
        self.w.confirm_transaction("bob", tx_id)
        self.w.revoke_confirmation("bob", tx_id)
        with self.assertRaises(ValueError):
            self.w.execute_transaction("alice", tx_id)

    def test_cannot_revoke_without_confirming(self):
        tx_id = self.w.submit_transaction("alice", "dave", value=100)
        with self.assertRaises(ValueError):
            self.w.revoke_confirmation("alice", tx_id)


class TestAdminOperations(unittest.TestCase):
    """Test self-call admin functions: add/remove owner, change threshold."""

    def setUp(self):
        self.w = MultisigWallet(owners=["alice", "bob", "carol"], required=2)

    def _approve_and_execute(self, tx_id: int) -> bool:
        self.w.confirm_transaction("alice", tx_id)
        self.w.confirm_transaction("bob", tx_id)
        return self.w.execute_transaction("alice", tx_id)

    def test_add_owner(self):
        tx_id = self.w.submit_transaction(
            "alice", MultisigWallet.SELF_ADDRESS,
            data=("addOwner", {"owner": "dave"}),
        )
        self._approve_and_execute(tx_id)
        self.assertIn("dave", self.w.owners)
        self.assertEqual(len(self.w.owners), 4)

    def test_remove_owner(self):
        tx_id = self.w.submit_transaction(
            "alice", MultisigWallet.SELF_ADDRESS,
            data=("removeOwner", {"owner": "carol"}),
        )
        self._approve_and_execute(tx_id)
        self.assertNotIn("carol", self.w.owners)
        self.assertEqual(len(self.w.owners), 2)

    def test_remove_owner_auto_adjusts_threshold(self):
        """Removing owner when required > new_owner_count auto-lowers threshold."""
        w = MultisigWallet(owners=["alice", "bob", "carol"], required=3)
        tx_id = w.submit_transaction(
            "alice", MultisigWallet.SELF_ADDRESS,
            data=("removeOwner", {"owner": "carol"}),
        )
        w.confirm_transaction("alice", tx_id)
        w.confirm_transaction("bob", tx_id)
        w.confirm_transaction("carol", tx_id)
        w.execute_transaction("alice", tx_id)
        self.assertEqual(w.required, 2)  # Auto-adjusted from 3

    def test_change_requirement(self):
        tx_id = self.w.submit_transaction(
            "alice", MultisigWallet.SELF_ADDRESS,
            data=("changeRequirement", {"required": 3}),
        )
        self._approve_and_execute(tx_id)
        self.assertEqual(self.w.required, 3)

    def test_cannot_remove_last_owner(self):
        w = MultisigWallet(owners=["alice"], required=1)
        tx_id = w.submit_transaction(
            "alice", MultisigWallet.SELF_ADDRESS,
            data=("removeOwner", {"owner": "alice"}),
        )
        w.confirm_transaction("alice", tx_id)
        success = w.execute_transaction("alice", tx_id)
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()
