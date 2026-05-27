"""
Day 036: Smart Contract Testing Framework in Python — Your Implementation

Build a smart contract testing framework from scratch. You'll implement:
1. ABI encoding/decoding (the wire protocol for contract calls)
2. A simulated EVM state machine (accounts, balances, storage)
3. Contract deployment and transaction execution
4. Assertion utilities for testing (reverts, events, balances)
5. A test suite for an ERC-20 token

Think about: What does a testing framework need to simulate to be useful?
Why is test isolation (snapshot/revert) critical for blockchain testing?
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import time


# =============================================================================
# Part 1: ABI Encoding/Decoding
# =============================================================================
# Hint: ABI encoding pads everything to 32-byte boundaries.
# uint256 → big-endian 32 bytes. address → 12 zero bytes + 20 address bytes.
# Function selector → first 4 bytes of hash("functionName(type1,type2)").

def keccak256(data: bytes) -> bytes:
    """Simulate keccak256 using SHA-256 (same interface, avoids external deps)."""
    return hashlib.sha256(data).digest()


def compute_function_selector(signature: str) -> bytes:
    """
    Compute the 4-byte function selector from a signature string.

    Example: "transfer(address,uint256)" → first 4 bytes of keccak256(signature)

    Hint: Hash the signature bytes and take the first 4 bytes.
    """
    raise NotImplementedError("TODO: implement this")


def abi_encode_uint256(value: int) -> bytes:
    """
    Encode a uint256 as 32 bytes, big-endian.

    Hint: Python's int.to_bytes(32, byteorder='big') does exactly this.
    """
    raise NotImplementedError("TODO: implement this")


def abi_encode_address(address: str) -> bytes:
    """
    Encode an Ethereum address as 32 bytes (left-padded with zeros).

    Hint: Strip "0x", convert hex to bytes, left-pad to 32 bytes.
    """
    raise NotImplementedError("TODO: implement this")


def abi_encode_string(value: str) -> bytes:
    """
    Encode a string: 32-byte length prefix + data padded to 32-byte boundary.
    """
    raise NotImplementedError("TODO: implement this")


def abi_decode_uint256(data: bytes) -> int:
    """Decode 32 bytes into a uint256 integer."""
    raise NotImplementedError("TODO: implement this")


def abi_decode_address(data: bytes) -> str:
    """Decode 32 bytes into an address string (last 20 bytes)."""
    raise NotImplementedError("TODO: implement this")


def abi_encode_call(signature: str, *args: tuple[str, Any]) -> bytes:
    """
    Encode a complete function call: selector + encoded arguments.

    Args:
        signature: e.g., "transfer(address,uint256)"
        *args: Tuples of (type, value), e.g., ("address", "0x123..."), ("uint256", 1000)

    Hint: Concatenate the selector with each ABI-encoded argument.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 2: EVM State Machine
# =============================================================================
# Hint: The EVM is a state machine where each transaction transforms the world
# state. You need to track: accounts (balance, nonce, code, storage), blocks,
# and provide snapshot/revert for test isolation.

@dataclass
class Account:
    """Ethereum account: balance, nonce, optional code and storage."""
    balance: int = 0
    nonce: int = 0
    code: bytes = b''
    storage: dict = field(default_factory=dict)


@dataclass
class EventLog:
    """An event emitted by a contract."""
    address: str
    topics: list[bytes]
    data: bytes
    event_name: str = ""


@dataclass
class TransactionReceipt:
    """Result of executing a transaction."""
    success: bool
    gas_used: int
    events: list[EventLog]
    return_data: bytes = b''
    revert_reason: str = ""
    sender: str = ""
    to: str = ""
    value: int = 0


@dataclass
class BlockInfo:
    """Simulated block metadata."""
    number: int = 1
    timestamp: int = field(default_factory=lambda: int(time.time()))
    base_fee: int = 1_000_000_000


class SimulatedEVM:
    """
    Simulated Ethereum Virtual Machine.

    Implement: account management, contract deployment, transaction execution,
    snapshot/revert for test isolation, and time manipulation.

    Hint: snapshot() should deep-copy all state; revert() should restore it.
    Contract logic is stored as Python objects, not bytecode.
    """

    def __init__(self):
        self.accounts: dict[str, Account] = {}
        self.block: BlockInfo = BlockInfo()
        self._snapshots: dict[int, dict] = {}
        self._next_snapshot_id: int = 0
        self._contract_logic: dict[str, Any] = {}
        self._contract_names: dict[str, str] = {}

    def create_account(self, address: str, balance: int = 0) -> None:
        """Create an EOA with the given balance."""
        raise NotImplementedError("TODO: implement this")

    def get_balance(self, address: str) -> int:
        """Get account balance. Returns 0 for non-existent accounts."""
        raise NotImplementedError("TODO: implement this")

    def set_balance(self, address: str, balance: int) -> None:
        """Directly set an account's balance (testing cheatcode)."""
        raise NotImplementedError("TODO: implement this")

    def get_storage(self, address: str, slot: int) -> int:
        """Read a contract storage slot. Returns 0 if uninitialized."""
        raise NotImplementedError("TODO: implement this")

    def set_storage(self, address: str, slot: int, value: int) -> None:
        """Write to a contract storage slot."""
        raise NotImplementedError("TODO: implement this")

    def deploy_contract(
        self,
        deployer: str,
        contract_name: str,
        contract_logic: 'ContractLogic',
        constructor_args: Optional[dict] = None,
        value: int = 0
    ) -> tuple[str, TransactionReceipt]:
        """
        Deploy a contract.

        Steps:
        1. Verify deployer exists and has enough balance
        2. Generate contract address from deployer + nonce
        3. Create contract account, store logic
        4. Execute constructor
        5. Handle RevertError if constructor fails

        Hint: Use keccak256(f"{deployer}{nonce}") for address generation.
        """
        raise NotImplementedError("TODO: implement this")

    def call_contract(
        self,
        sender: str,
        contract_address: str,
        function_name: str,
        args: Optional[dict] = None,
        value: int = 0
    ) -> TransactionReceipt:
        """
        Call a function on a deployed contract.

        Steps:
        1. Validate sender and contract exist
        2. Transfer ETH value (before execution)
        3. Call contract_logic.handle_call()
        4. If RevertError: undo value transfer, return failed receipt
        5. If success: update nonce, return receipt with events

        Hint: Value transfer happens BEFORE code execution. Revert must undo it.
        """
        raise NotImplementedError("TODO: implement this")

    def transfer_eth(self, sender: str, to: str, amount: int) -> TransactionReceipt:
        """Simple ETH transfer between accounts."""
        raise NotImplementedError("TODO: implement this")

    def snapshot(self) -> int:
        """
        Take a snapshot of all EVM state. Returns snapshot ID.

        Hint: Use copy.deepcopy on accounts and block. Store in _snapshots dict.
        """
        raise NotImplementedError("TODO: implement this")

    def revert(self, snapshot_id: int) -> bool:
        """
        Revert to a snapshot. Returns True if successful.

        Hint: Restore all state from the snapshot, then delete it.
        """
        raise NotImplementedError("TODO: implement this")

    def advance_time(self, seconds: int) -> None:
        """Advance the block timestamp."""
        raise NotImplementedError("TODO: implement this")

    def advance_blocks(self, count: int) -> None:
        """Advance the block number (each block ~12 seconds)."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 3: Contract Logic Base Class
# =============================================================================

class RevertError(Exception):
    """Raised when a contract reverts. Simulates the EVM REVERT opcode."""
    pass


class ContractLogic:
    """
    Base class for simulated contracts.

    Implement constructor() and handle_call() in subclasses.
    Raise RevertError to simulate a revert.
    """

    def constructor(self, evm: SimulatedEVM, address: str, sender: str,
                    args: dict, events: list[EventLog]) -> None:
        pass

    def handle_call(self, evm: SimulatedEVM, address: str, sender: str,
                    function_name: str, args: dict, value: int,
                    events: list[EventLog]) -> Optional[bytes]:
        raise RevertError(f"Function {function_name} not found")


# =============================================================================
# Part 4: ERC-20 Token Contract
# =============================================================================
# Hint: Use storage slots to store balances and allowances.
# Use keccak256 to derive unique slots for each (owner, spender) pair.
# Emit EventLog entries for Transfer and Approval events.

class ERC20Token(ContractLogic):
    """
    Simplified ERC-20 token. Implement:
    - constructor: mint initial supply to deployer
    - balanceOf, totalSupply: read-only queries
    - transfer: move tokens, check balance, emit Transfer
    - approve: set allowance, emit Approval
    - transferFrom: spend allowance, check both allowance and balance
    """

    TOTAL_SUPPLY_SLOT = 0
    BALANCE_PREFIX = 1
    ALLOWANCE_PREFIX = 2

    def _balance_slot(self, owner: str) -> int:
        """Compute storage slot for a balance mapping."""
        return int.from_bytes(keccak256(f"{owner}{self.BALANCE_PREFIX}".encode())[:8], 'big')

    def _allowance_slot(self, owner: str, spender: str) -> int:
        """Compute storage slot for an allowance mapping (nested)."""
        return int.from_bytes(keccak256(f"{owner}{spender}{self.ALLOWANCE_PREFIX}".encode())[:8], 'big')

    def constructor(self, evm: SimulatedEVM, address: str, sender: str,
                    args: dict, events: list[EventLog]) -> None:
        """
        Mint initial supply to deployer.

        Hint: Store total supply in slot 0, deployer balance in _balance_slot(sender).
        Emit a Transfer event from the zero address to sender.
        """
        raise NotImplementedError("TODO: implement this")

    def handle_call(self, evm: SimulatedEVM, address: str, sender: str,
                    function_name: str, args: dict, value: int,
                    events: list[EventLog]) -> Optional[bytes]:
        """
        Route function calls. Implement dispatching to:
        - balanceOf, totalSupply, transfer, approve, allowance, transferFrom
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 5: Testing Assertion Utilities
# =============================================================================
# Hint: These wrap common assertion patterns to make tests readable.

class TestContext:
    """High-level testing assertions for blockchain transactions."""

    def __init__(self, evm: SimulatedEVM):
        self.evm = evm

    def expect_revert(self, receipt: TransactionReceipt, reason: str = "") -> None:
        """Assert transaction reverted, optionally with a specific reason string."""
        raise NotImplementedError("TODO: implement this")

    def expect_success(self, receipt: TransactionReceipt) -> None:
        """Assert transaction succeeded."""
        raise NotImplementedError("TODO: implement this")

    def expect_event(self, receipt: TransactionReceipt, event_name: str,
                     count: int = 1) -> list[EventLog]:
        """Assert specific event was emitted the expected number of times."""
        raise NotImplementedError("TODO: implement this")

    def expect_balance_change(
        self,
        address: str,
        expected_change: int,
        action: Callable[[], TransactionReceipt]
    ) -> TransactionReceipt:
        """Assert an action changes ETH balance by expected amount."""
        raise NotImplementedError("TODO: implement this")

    def expect_token_balance(self, token_address: str, token_logic: ERC20Token,
                             owner: str, expected: int) -> None:
        """Assert an address holds the expected token balance."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    print("Testing your Smart Contract Testing Framework...\n")

    # Test ABI encoding
    print("--- ABI Encoding ---")
    selector = compute_function_selector("transfer(address,uint256)")
    print(f"Selector: 0x{selector.hex()}")

    encoded = abi_encode_uint256(1000)
    decoded = abi_decode_uint256(encoded)
    print(f"Uint256 roundtrip: 1000 → {decoded}")

    addr = "0x" + "ab" * 20
    encoded_addr = abi_encode_address(addr)
    decoded_addr = abi_decode_address(encoded_addr)
    print(f"Address roundtrip: {addr} → {decoded_addr}")

    # Test EVM
    print("\n--- EVM ---")
    evm = SimulatedEVM()
    deployer = "0x" + "1" * 40
    alice = "0x" + "2" * 40
    bob = "0x" + "3" * 40

    evm.create_account(deployer, 100 * 10**18)
    evm.create_account(alice, 100 * 10**18)
    evm.create_account(bob, 100 * 10**18)
    print(f"Deployer balance: {evm.get_balance(deployer)}")

    # Deploy ERC-20
    print("\n--- Deploy ERC-20 ---")
    token_logic = ERC20Token()
    token_addr, receipt = evm.deploy_contract(
        deployer, "TestToken", token_logic,
        {"initial_supply": 1_000_000 * 10**18, "name": "TestToken", "symbol": "TT"}
    )
    print(f"Deployed at: {token_addr}")
    print(f"Success: {receipt.success}")

    # Test transfer
    print("\n--- Transfer ---")
    receipt = evm.call_contract(
        deployer, token_addr, "transfer",
        {"to": alice, "amount": 1000 * 10**18}
    )
    print(f"Transfer: {'SUCCESS' if receipt.success else 'FAILED'}")

    # Test revert
    receipt = evm.call_contract(
        bob, token_addr, "transfer",
        {"to": alice, "amount": 1}
    )
    print(f"Bob transfer (should fail): {'SUCCESS' if receipt.success else 'REVERTED'}")
    print(f"Reason: {receipt.revert_reason}")

    # Test snapshot/revert
    print("\n--- Snapshot/Revert ---")
    snap = evm.snapshot()
    evm.call_contract(deployer, token_addr, "transfer", {"to": bob, "amount": 50000 * 10**18})

    r = evm.call_contract(deployer, token_addr, "balanceOf", {"owner": bob})
    print(f"Bob before revert: {abi_decode_uint256(r.return_data)}")

    evm.revert(snap)
    r = evm.call_contract(deployer, token_addr, "balanceOf", {"owner": bob})
    print(f"Bob after revert: {abi_decode_uint256(r.return_data)}")

    print("\nAll manual tests complete!")
