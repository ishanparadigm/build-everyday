"""
Tests for Day 036: Smart Contract Testing Framework

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
from my_solution import (
    keccak256, compute_function_selector, abi_encode_uint256, abi_encode_address,
    abi_encode_string, abi_decode_uint256, abi_decode_address, abi_encode_call,
    SimulatedEVM, ERC20Token, RevertError, TestContext, EventLog
)


class TestABIEncoding(unittest.TestCase):
    """Test ABI encoding and decoding functions."""

    def test_uint256_zero(self):
        """Zero encodes as 32 zero bytes."""
        encoded = abi_encode_uint256(0)
        self.assertEqual(len(encoded), 32)
        self.assertEqual(encoded, b'\x00' * 32)

    def test_uint256_roundtrip(self):
        """Encode then decode returns the original value."""
        for value in [0, 1, 255, 1000, 2**128, 2**256 - 1]:
            encoded = abi_encode_uint256(value)
            decoded = abi_decode_uint256(encoded)
            self.assertEqual(decoded, value, f"Failed for {value}")

    def test_address_encoding_32_bytes(self):
        """Address encodes to exactly 32 bytes."""
        addr = "0x" + "ab" * 20
        self.assertEqual(len(abi_encode_address(addr)), 32)

    def test_address_left_padded(self):
        """Address is left-padded with 12 zero bytes."""
        addr = "0x" + "ff" * 20
        encoded = abi_encode_address(addr)
        self.assertEqual(encoded[:12], b'\x00' * 12)
        self.assertEqual(encoded[12:], b'\xff' * 20)

    def test_address_roundtrip(self):
        """Encode then decode returns the original address."""
        addr = "0x" + "ab" * 20
        encoded = abi_encode_address(addr)
        decoded = abi_decode_address(encoded)
        self.assertEqual(decoded, addr)

    def test_function_selector_is_4_bytes(self):
        """Function selector is exactly 4 bytes."""
        selector = compute_function_selector("transfer(address,uint256)")
        self.assertEqual(len(selector), 4)

    def test_function_selector_deterministic(self):
        """Same signature always produces the same selector."""
        sig = "transfer(address,uint256)"
        self.assertEqual(compute_function_selector(sig), compute_function_selector(sig))

    def test_full_call_encoding_length(self):
        """Full call = 4 selector + 32 per arg."""
        calldata = abi_encode_call(
            "transfer(address,uint256)",
            ("address", "0x" + "ab" * 20),
            ("uint256", 1000)
        )
        self.assertEqual(len(calldata), 68)  # 4 + 32 + 32


class TestSimulatedEVM(unittest.TestCase):
    """Test the EVM state machine."""

    def setUp(self):
        self.evm = SimulatedEVM()
        self.deployer = "0x" + "1" * 40
        self.alice = "0x" + "2" * 40
        self.evm.create_account(self.deployer, 100 * 10**18)
        self.evm.create_account(self.alice, 50 * 10**18)

    def test_create_account_balance(self):
        """Created account has the specified balance."""
        self.assertEqual(self.evm.get_balance(self.deployer), 100 * 10**18)

    def test_nonexistent_account_zero_balance(self):
        """Non-existent account returns 0 balance."""
        self.assertEqual(self.evm.get_balance("0x" + "9" * 40), 0)

    def test_set_balance(self):
        """set_balance updates the balance."""
        self.evm.set_balance(self.deployer, 200)
        self.assertEqual(self.evm.get_balance(self.deployer), 200)

    def test_eth_transfer(self):
        """ETH transfer moves funds between accounts."""
        amount = 10 * 10**18
        receipt = self.evm.transfer_eth(self.deployer, self.alice, amount)
        self.assertTrue(receipt.success)
        self.assertEqual(self.evm.get_balance(self.deployer), 90 * 10**18)
        self.assertEqual(self.evm.get_balance(self.alice), 60 * 10**18)

    def test_eth_transfer_insufficient_balance(self):
        """ETH transfer with insufficient balance fails."""
        receipt = self.evm.transfer_eth(self.alice, self.deployer, 999 * 10**18)
        self.assertFalse(receipt.success)

    def test_snapshot_revert(self):
        """Snapshot and revert restores state."""
        snap = self.evm.snapshot()
        self.evm.set_balance(self.deployer, 0)
        self.assertEqual(self.evm.get_balance(self.deployer), 0)
        self.evm.revert(snap)
        self.assertEqual(self.evm.get_balance(self.deployer), 100 * 10**18)

    def test_advance_time(self):
        """advance_time increments block timestamp."""
        ts = self.evm.block.timestamp
        self.evm.advance_time(3600)
        self.assertEqual(self.evm.block.timestamp, ts + 3600)

    def test_advance_blocks(self):
        """advance_blocks increments block number and timestamp."""
        bn = self.evm.block.number
        self.evm.advance_blocks(10)
        self.assertEqual(self.evm.block.number, bn + 10)


class TestERC20Contract(unittest.TestCase):
    """Test the ERC-20 token contract."""

    DEPLOYER = "0x" + "1" * 40
    ALICE = "0x" + "2" * 40
    BOB = "0x" + "3" * 40
    ZERO = "0x" + "0" * 40
    SUPPLY = 1_000_000 * 10**18

    def setUp(self):
        self.evm = SimulatedEVM()
        self.token_logic = ERC20Token()
        self.ctx = TestContext(self.evm)

        self.evm.create_account(self.DEPLOYER, 100 * 10**18)
        self.evm.create_account(self.ALICE, 100 * 10**18)
        self.evm.create_account(self.BOB, 100 * 10**18)

        self.token_addr, receipt = self.evm.deploy_contract(
            self.DEPLOYER, "Token", self.token_logic,
            {"initial_supply": self.SUPPLY, "name": "T", "symbol": "T"}
        )
        self.assertTrue(receipt.success, f"Deploy failed: {receipt.revert_reason}")

    def test_total_supply(self):
        """Total supply matches initial supply."""
        r = self.evm.call_contract(self.DEPLOYER, self.token_addr, "totalSupply")
        self.assertEqual(abi_decode_uint256(r.return_data), self.SUPPLY)

    def test_deployer_holds_all_tokens(self):
        """Deployer receives full supply."""
        self.ctx.expect_token_balance(self.token_addr, self.token_logic, self.DEPLOYER, self.SUPPLY)

    def test_transfer(self):
        """Transfer moves tokens and updates both balances."""
        amt = 1000 * 10**18
        r = self.evm.call_contract(self.DEPLOYER, self.token_addr, "transfer",
                                   {"to": self.ALICE, "amount": amt})
        self.assertTrue(r.success)
        self.ctx.expect_token_balance(self.token_addr, self.token_logic, self.ALICE, amt)
        self.ctx.expect_token_balance(self.token_addr, self.token_logic, self.DEPLOYER, self.SUPPLY - amt)

    def test_transfer_emits_event(self):
        """Transfer emits a Transfer event."""
        r = self.evm.call_contract(self.DEPLOYER, self.token_addr, "transfer",
                                   {"to": self.ALICE, "amount": 500})
        events = self.ctx.expect_event(r, "Transfer", count=1)
        self.assertEqual(abi_decode_uint256(events[0].data), 500)

    def test_transfer_insufficient_balance_reverts(self):
        """Transfer with insufficient balance reverts."""
        r = self.evm.call_contract(self.ALICE, self.token_addr, "transfer",
                                   {"to": self.BOB, "amount": 1})
        self.ctx.expect_revert(r, "exceeds balance")

    def test_transfer_to_zero_reverts(self):
        """Transfer to zero address reverts."""
        r = self.evm.call_contract(self.DEPLOYER, self.token_addr, "transfer",
                                   {"to": self.ZERO, "amount": 100})
        self.ctx.expect_revert(r, "zero address")

    def test_approve_and_transfer_from(self):
        """Approve then transferFrom works correctly."""
        amt = 5000
        self.evm.call_contract(self.DEPLOYER, self.token_addr, "approve",
                               {"spender": self.ALICE, "amount": amt})
        r = self.evm.call_contract(self.ALICE, self.token_addr, "transferFrom",
                                   {"from": self.DEPLOYER, "to": self.BOB, "amount": 3000})
        self.assertTrue(r.success)
        self.ctx.expect_token_balance(self.token_addr, self.token_logic, self.BOB, 3000)

        # Allowance should decrease
        r = self.evm.call_contract(self.DEPLOYER, self.token_addr, "allowance",
                                   {"owner": self.DEPLOYER, "spender": self.ALICE})
        self.assertEqual(abi_decode_uint256(r.return_data), 2000)

    def test_transfer_from_exceeds_allowance_reverts(self):
        """transferFrom over allowance reverts."""
        self.evm.call_contract(self.DEPLOYER, self.token_addr, "approve",
                               {"spender": self.ALICE, "amount": 100})
        r = self.evm.call_contract(self.ALICE, self.token_addr, "transferFrom",
                                   {"from": self.DEPLOYER, "to": self.BOB, "amount": 200})
        self.ctx.expect_revert(r, "insufficient allowance")

    def test_zero_transfer_succeeds(self):
        """Zero-amount transfer should succeed."""
        r = self.evm.call_contract(self.DEPLOYER, self.token_addr, "transfer",
                                   {"to": self.ALICE, "amount": 0})
        self.assertTrue(r.success)


if __name__ == "__main__":
    unittest.main()
