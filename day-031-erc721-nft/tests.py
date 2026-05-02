"""
Day 031: ERC-721 NFT Contract Tests

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
from my_solution import ERC721, ZERO_ADDRESS


class TestERC721Metadata(unittest.TestCase):
    """Test collection and token metadata."""

    def setUp(self):
        self.nft = ERC721(name="TestNFT", symbol="TNFT")
        self.alice = "0xAlice"

    def test_collection_name_and_symbol(self):
        self.assertEqual(self.nft.name(), "TestNFT")
        self.assertEqual(self.nft.symbol(), "TNFT")

    def test_token_uri(self):
        self.nft.set_caller(self.alice)
        self.nft.mint(self.alice, 1)
        self.nft.set_token_uri(1, "ipfs://test/1.json")
        self.assertEqual(self.nft.token_uri(1), "ipfs://test/1.json")

    def test_token_uri_nonexistent_reverts(self):
        with self.assertRaises(ValueError):
            self.nft.token_uri(999)


class TestERC721Minting(unittest.TestCase):
    """Test minting behavior."""

    def setUp(self):
        self.nft = ERC721(name="TestNFT", symbol="TNFT")
        self.alice = "0xAlice"
        self.nft.set_caller(self.alice)

    def test_mint_assigns_ownership(self):
        self.nft.mint(self.alice, 1)
        self.assertEqual(self.nft.owner_of(1), self.alice)

    def test_mint_updates_balance(self):
        self.nft.mint(self.alice, 1)
        self.nft.mint(self.alice, 2)
        self.assertEqual(self.nft.balance_of(self.alice), 2)

    def test_mint_updates_total_supply(self):
        self.nft.mint(self.alice, 1)
        self.assertEqual(self.nft.total_supply(), 1)

    def test_mint_duplicate_reverts(self):
        self.nft.mint(self.alice, 1)
        with self.assertRaises(ValueError):
            self.nft.mint(self.alice, 1)

    def test_mint_to_zero_address_reverts(self):
        with self.assertRaises(ValueError):
            self.nft.mint(ZERO_ADDRESS, 1)

    def test_mint_emits_transfer_event(self):
        self.nft.mint(self.alice, 1)
        transfer_events = [e for e in self.nft.events if e.name == "Transfer"]
        self.assertEqual(len(transfer_events), 1)
        self.assertEqual(transfer_events[0].args["from_addr"], ZERO_ADDRESS)
        self.assertEqual(transfer_events[0].args["to_addr"], self.alice)


class TestERC721Transfer(unittest.TestCase):
    """Test transfer mechanics."""

    def setUp(self):
        self.nft = ERC721(name="TestNFT", symbol="TNFT")
        self.alice = "0xAlice"
        self.bob = "0xBob"
        self.nft.set_caller(self.alice)
        self.nft.mint(self.alice, 1)

    def test_owner_can_transfer(self):
        self.nft.transfer_from(self.alice, self.bob, 1)
        self.assertEqual(self.nft.owner_of(1), self.bob)

    def test_transfer_updates_balances(self):
        self.nft.transfer_from(self.alice, self.bob, 1)
        self.assertEqual(self.nft.balance_of(self.alice), 0)
        self.assertEqual(self.nft.balance_of(self.bob), 1)

    def test_unauthorized_transfer_reverts(self):
        self.nft.set_caller(self.bob)
        with self.assertRaises(ValueError):
            self.nft.transfer_from(self.alice, self.bob, 1)

    def test_transfer_to_zero_address_reverts(self):
        with self.assertRaises(ValueError):
            self.nft.transfer_from(self.alice, ZERO_ADDRESS, 1)

    def test_transfer_clears_approval(self):
        self.nft.approve(self.bob, 1)
        self.nft.transfer_from(self.alice, self.bob, 1)
        self.assertEqual(self.nft.get_approved(1), ZERO_ADDRESS)


class TestERC721Approval(unittest.TestCase):
    """Test per-token and operator approval."""

    def setUp(self):
        self.nft = ERC721(name="TestNFT", symbol="TNFT")
        self.alice = "0xAlice"
        self.bob = "0xBob"
        self.charlie = "0xCharlie"
        self.nft.set_caller(self.alice)
        self.nft.mint(self.alice, 1)

    def test_approve_and_transfer(self):
        self.nft.approve(self.bob, 1)
        self.assertEqual(self.nft.get_approved(1), self.bob)
        self.nft.set_caller(self.bob)
        self.nft.transfer_from(self.alice, self.bob, 1)
        self.assertEqual(self.nft.owner_of(1), self.bob)

    def test_approve_to_owner_reverts(self):
        with self.assertRaises(ValueError):
            self.nft.approve(self.alice, 1)

    def test_non_owner_approve_reverts(self):
        self.nft.set_caller(self.bob)
        with self.assertRaises(ValueError):
            self.nft.approve(self.charlie, 1)

    def test_operator_approval(self):
        self.nft.set_approval_for_all(self.bob, True)
        self.assertTrue(self.nft.is_approved_for_all(self.alice, self.bob))
        self.nft.set_caller(self.bob)
        self.nft.transfer_from(self.alice, self.charlie, 1)
        self.assertEqual(self.nft.owner_of(1), self.charlie)

    def test_operator_can_approve_tokens(self):
        """An approved operator should be able to set per-token approvals."""
        self.nft.set_approval_for_all(self.bob, True)
        self.nft.set_caller(self.bob)
        self.nft.approve(self.charlie, 1)
        self.assertEqual(self.nft.get_approved(1), self.charlie)

    def test_revoke_operator(self):
        self.nft.set_approval_for_all(self.bob, True)
        self.nft.set_approval_for_all(self.bob, False)
        self.assertFalse(self.nft.is_approved_for_all(self.alice, self.bob))


class TestERC721Burn(unittest.TestCase):
    """Test burning tokens."""

    def setUp(self):
        self.nft = ERC721(name="TestNFT", symbol="TNFT")
        self.alice = "0xAlice"
        self.bob = "0xBob"
        self.nft.set_caller(self.alice)
        self.nft.mint(self.alice, 1)
        self.nft.mint(self.alice, 2)

    def test_burn_removes_token(self):
        self.nft.burn(1)
        with self.assertRaises(ValueError):
            self.nft.owner_of(1)

    def test_burn_updates_balance_and_supply(self):
        self.nft.burn(1)
        self.assertEqual(self.nft.balance_of(self.alice), 1)
        self.assertEqual(self.nft.total_supply(), 1)

    def test_unauthorized_burn_reverts(self):
        self.nft.set_caller(self.bob)
        with self.assertRaises(ValueError):
            self.nft.burn(1)


class TestERC721Enumerable(unittest.TestCase):
    """Test enumeration functions."""

    def setUp(self):
        self.nft = ERC721(name="TestNFT", symbol="TNFT")
        self.alice = "0xAlice"
        self.bob = "0xBob"
        self.nft.set_caller(self.alice)
        self.nft.mint(self.alice, 10)
        self.nft.mint(self.alice, 20)
        self.nft.mint(self.bob, 30)

    def test_total_supply(self):
        self.assertEqual(self.nft.total_supply(), 3)

    def test_token_by_index(self):
        tokens = {self.nft.token_by_index(i) for i in range(3)}
        self.assertEqual(tokens, {10, 20, 30})

    def test_token_of_owner_by_index(self):
        alice_tokens = {self.nft.token_of_owner_by_index(self.alice, i) for i in range(2)}
        self.assertEqual(alice_tokens, {10, 20})

    def test_index_out_of_bounds(self):
        with self.assertRaises(IndexError):
            self.nft.token_by_index(99)

    def test_enumeration_after_transfer(self):
        """After transferring a token, enumeration should reflect new ownership."""
        self.nft.transfer_from(self.alice, self.bob, 10)
        self.assertEqual(self.nft.balance_of(self.alice), 1)
        self.assertEqual(self.nft.balance_of(self.bob), 2)
        bob_tokens = {self.nft.token_of_owner_by_index(self.bob, i) for i in range(2)}
        self.assertIn(10, bob_tokens)
        self.assertIn(30, bob_tokens)


if __name__ == "__main__":
    unittest.main()
