"""
Day 036: Smart Contract Testing Framework in Python

A from-scratch implementation of a smart contract testing framework that simulates
the core functionality of tools like Hardhat, Foundry, and Brownie. We build:
  1. A simulated EVM state machine (accounts, balances, storage, blocks)
  2. ABI encoding/decoding for contract communication
  3. Contract deployment and transaction execution
  4. Rich assertion utilities (reverts, events, balance changes)
  5. Test fixture management with snapshot/revert

This teaches you what professional testing frameworks do under the hood,
and why each piece is necessary for testing code that controls real money.
"""

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# =============================================================================
# ABI Encoding/Decoding
# =============================================================================
# The ABI is the wire protocol between callers and contracts. Understanding it
# at the byte level is essential because every vulnerability scanner, debugger,
# and testing tool must parse these bytes correctly.

def keccak256(data: bytes) -> bytes:
    """
    Simulate keccak256 hashing using SHA-256 for our testing framework.

    In production Ethereum, keccak256 (SHA-3 family) is used everywhere:
    function selectors, event signatures, storage slot computation, address
    derivation. We use SHA-256 here to avoid external dependencies while
    preserving the same interface — 32 bytes in, 32 bytes out.

    The key insight: what matters for testing is that the hash function is
    deterministic and collision-resistant, not which specific algorithm is used.
    """
    return hashlib.sha256(data).digest()


def compute_function_selector(signature: str) -> bytes:
    """
    Compute the 4-byte function selector from a function signature.

    The selector is the first 4 bytes of keccak256("functionName(type1,type2)").
    This is how the EVM routes calls to the correct function — when you call
    transfer(address,uint256), the EVM sees 0xa9059cbb and jumps to that function.

    Why only 4 bytes? It's a tradeoff: 4 bytes = 2^32 possible selectors. With
    ~10 functions per contract, collision probability is negligible (~1 in 400M).
    But selector collisions DO exist across different contracts and have been
    exploited in proxy patterns.

    Args:
        signature: Function signature like "transfer(address,uint256)"

    Returns:
        4-byte selector
    """
    return keccak256(signature.encode())[:4]


def abi_encode_uint256(value: int) -> bytes:
    """
    Encode a uint256 value as 32 bytes, big-endian.

    Every ABI-encoded value occupies exactly 32 bytes (256 bits). This is because
    the EVM's word size is 256 bits — every stack slot, every storage slot, every
    memory access is 32-byte aligned. Smaller types (uint8, bool) are still
    padded to 32 bytes.

    Big-endian means the most significant byte comes first:
    1000 decimal = 0x03E8 = 0x00...03E8 (30 zero bytes + 0x03E8)
    """
    # value.to_bytes handles the conversion; 'big' = most significant byte first
    return value.to_bytes(32, byteorder='big')


def abi_encode_address(address: str) -> bytes:
    """
    Encode an Ethereum address as 32 bytes (left-padded with zeros).

    Addresses are 20 bytes (160 bits) but ABI encoding pads them to 32 bytes.
    The address goes in the rightmost 20 bytes, with 12 zero bytes on the left.

    Why 20 bytes for addresses? They're derived from the last 20 bytes of the
    keccak256 hash of the public key. 20 bytes = 160 bits gives ~10^48 possible
    addresses — enough that random collision is astronomically unlikely.
    """
    # Strip the "0x" prefix if present, convert hex string to bytes
    addr_bytes = bytes.fromhex(address.replace("0x", ""))
    # Left-pad to 32 bytes: 12 zero bytes + 20 address bytes
    return b'\x00' * (32 - len(addr_bytes)) + addr_bytes


def abi_encode_string(value: str) -> bytes:
    """
    Encode a string using ABI dynamic encoding.

    Dynamic types (string, bytes, arrays) use indirect encoding:
    1. The argument slot contains an OFFSET pointing to where the data lives
    2. At that offset: first 32 bytes = length, then the actual data (padded to 32-byte boundary)

    This indirection allows the EVM to skip over dynamic data when accessing
    later fixed-size arguments — crucial for gas efficiency.
    """
    encoded = value.encode('utf-8')
    # Length prefix (32 bytes) + data padded to 32-byte boundary
    padded_length = ((len(encoded) + 31) // 32) * 32
    return abi_encode_uint256(len(encoded)) + encoded.ljust(padded_length, b'\x00')


def abi_decode_uint256(data: bytes) -> int:
    """Decode 32 bytes into a uint256 integer."""
    return int.from_bytes(data[:32], byteorder='big')


def abi_decode_address(data: bytes) -> str:
    """Decode 32 bytes into an address string (last 20 bytes)."""
    return "0x" + data[12:32].hex()


def abi_encode_call(signature: str, *args: tuple[str, Any]) -> bytes:
    """
    Encode a complete function call: selector + encoded arguments.

    This is what gets sent as transaction calldata. The EVM reads the first
    4 bytes to determine which function to call, then decodes the remaining
    bytes according to that function's parameter types.

    Args:
        signature: Function signature, e.g., "transfer(address,uint256)"
        *args: Tuples of (type, value), e.g., ("address", "0x123..."), ("uint256", 1000)

    Returns:
        Complete calldata bytes
    """
    selector = compute_function_selector(signature)
    encoded_args = b''
    for arg_type, arg_value in args:
        if arg_type == "uint256":
            encoded_args += abi_encode_uint256(arg_value)
        elif arg_type == "address":
            encoded_args += abi_encode_address(arg_value)
        elif arg_type == "string":
            encoded_args += abi_encode_string(arg_value)
        else:
            raise ValueError(f"Unsupported ABI type: {arg_type}")
    return selector + encoded_args


# =============================================================================
# EVM State Machine
# =============================================================================

@dataclass
class Account:
    """
    Represents an Ethereum account in our simulated EVM.

    There are two types of accounts in Ethereum:
    1. Externally Owned Accounts (EOAs): Controlled by private keys. Have balance
       and nonce but no code or storage. These are user wallets.
    2. Contract Accounts: Have code and storage in addition to balance and nonce.
       They can only act when called by a transaction.

    The nonce prevents replay attacks: each transaction must have the next
    sequential nonce, so you can't replay an old transaction.
    """
    balance: int = 0           # Balance in wei (1 ETH = 10^18 wei)
    nonce: int = 0             # Transaction count (for EOAs) or contracts created (for contracts)
    code: bytes = b''          # Contract bytecode (empty for EOAs)
    storage: dict = field(default_factory=dict)  # Contract storage slots (slot → value)


@dataclass
class EventLog:
    """
    Represents an event emitted by a contract.

    Events are the primary way contracts communicate with off-chain systems.
    They're stored in transaction receipts (not in contract storage) and are
    indexed by bloom filters for efficient searching.

    In Solidity: event Transfer(address indexed from, address indexed to, uint256 value);
    - "indexed" parameters become topics (searchable, max 3 per event)
    - Non-indexed parameters go into data (not directly searchable)
    - topic[0] is always the event signature hash
    """
    address: str           # Contract that emitted the event
    topics: list[bytes]    # Indexed parameters (topic[0] = event signature hash)
    data: bytes            # Non-indexed parameters, ABI-encoded
    event_name: str = ""   # Human-readable name for our testing convenience


@dataclass
class TransactionReceipt:
    """
    The receipt of an executed transaction.

    Receipts are how you know what happened: did it succeed? How much gas was
    used? What events were emitted? In production, receipts are stored in the
    blockchain and queryable via eth_getTransactionReceipt.
    """
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
    """
    Simulated block metadata.

    Many contracts depend on block properties:
    - block.timestamp: Used for time locks, vesting schedules, interest calculations
    - block.number: Used for governance voting periods, randomness (poorly)
    - block.basefee: Used for gas price estimation

    A testing framework MUST let you manipulate these values — otherwise you
    can't test time-dependent logic without waiting real-time.
    """
    number: int = 1
    timestamp: int = field(default_factory=lambda: int(time.time()))
    base_fee: int = 1_000_000_000  # 1 gwei default base fee


class SimulatedEVM:
    """
    A simulated Ethereum Virtual Machine for testing.

    This is the core of our testing framework. It maintains the world state
    (all accounts, balances, storage) and processes transactions. Think of it
    as a simplified version of what Geth or Erigon do.

    Key design decisions:
    - We simulate contract LOGIC in Python rather than executing actual bytecode.
      This is because our goal is testing the framework, not building an EVM interpreter.
    - We track gas as estimates, not exact calculations. Exact gas accounting
      requires opcode-level simulation which would triple the complexity.
    - Snapshot/revert uses deep copies. Production EVMs use copy-on-write tries
      for O(1) snapshots, but deep copy is simpler and correct for testing.
    """

    def __init__(self):
        self.accounts: dict[str, Account] = {}
        self.block: BlockInfo = BlockInfo()
        self._snapshots: dict[int, dict] = {}  # snapshot_id → state copy
        self._next_snapshot_id: int = 0
        self._contract_logic: dict[str, Any] = {}  # address → contract handler
        self._contract_names: dict[str, str] = {}   # address → human name
        self._next_contract_nonce: int = 0

    def create_account(self, address: str, balance: int = 0) -> None:
        """
        Create an externally owned account (EOA) with the given balance.

        In real Ethereum, accounts are created implicitly when they first receive
        ETH or send a transaction. In testing, we create them explicitly so each
        test starts from a known state.
        """
        self.accounts[address] = Account(balance=balance)

    def get_balance(self, address: str) -> int:
        """Get account balance. Returns 0 for non-existent accounts (matches EVM behavior)."""
        if address in self.accounts:
            return self.accounts[address].balance
        return 0

    def set_balance(self, address: str, balance: int) -> None:
        """
        Directly set an account's balance. This is a testing-only operation —
        in real Ethereum, balances can only change via transactions or block rewards.
        Tools like Hardhat expose this as hardhat_setBalance for testing.
        """
        if address not in self.accounts:
            self.create_account(address, balance)
        else:
            self.accounts[address].balance = balance

    def get_storage(self, address: str, slot: int) -> int:
        """
        Read a storage slot for a contract.

        Contract storage is a mapping from 256-bit keys to 256-bit values.
        Uninitialized slots return 0. Storage is the most expensive EVM resource:
        writing to a fresh slot costs 20,000 gas, reading costs 2,100 gas.
        This economic pressure shapes how smart contracts are designed.
        """
        if address in self.accounts:
            return self.accounts[address].storage.get(slot, 0)
        return 0

    def set_storage(self, address: str, slot: int, value: int) -> None:
        """
        Write to a storage slot. In testing, we can write directly.
        In production, only the contract's own code can write to its storage.
        """
        if address in self.accounts:
            self.accounts[address].storage[slot] = value

    def deploy_contract(
        self,
        deployer: str,
        contract_name: str,
        contract_logic: 'ContractLogic',
        constructor_args: Optional[dict] = None,
        value: int = 0
    ) -> tuple[str, TransactionReceipt]:
        """
        Deploy a contract to the simulated EVM.

        The deployment process in real Ethereum:
        1. Sender creates a transaction with empty 'to' field and bytecode as data
        2. EVM generates contract address: keccak256(rlp([sender, nonce]))[12:]
        3. EVM executes the constructor (init code)
        4. The RETURN value of the constructor becomes the contract's runtime code
        5. Contract is now live at the generated address

        We simulate this by:
        1. Generating a deterministic address from deployer + nonce
        2. Running the contract_logic's constructor method
        3. Storing the contract_logic as the "code"
        """
        if deployer not in self.accounts:
            raise ValueError(f"Deployer {deployer} does not exist")

        deployer_account = self.accounts[deployer]

        # Check deployer has enough ETH for the value transfer
        if deployer_account.balance < value:
            return "", TransactionReceipt(
                success=False, gas_used=21000, events=[],
                revert_reason="Insufficient balance for deployment",
                sender=deployer, to="", value=value
            )

        # Generate contract address deterministically (simulating CREATE opcode)
        # Real: keccak256(rlp([sender, nonce]))[12:]
        address_bytes = keccak256(f"{deployer}{deployer_account.nonce}".encode())
        contract_address = "0x" + address_bytes[12:].hex()

        # Update deployer nonce and transfer value
        deployer_account.nonce += 1
        deployer_account.balance -= value

        # Create the contract account
        self.accounts[contract_address] = Account(
            balance=value,
            code=b'\x01'  # Non-empty to mark as contract
        )

        # Store contract logic handler
        self._contract_logic[contract_address] = contract_logic
        self._contract_names[contract_address] = contract_name

        # Execute constructor
        events = []
        try:
            contract_logic.constructor(
                evm=self,
                address=contract_address,
                sender=deployer,
                args=constructor_args or {},
                events=events
            )
        except RevertError as e:
            # Constructor failed — remove the contract
            del self.accounts[contract_address]
            del self._contract_logic[contract_address]
            del self._contract_names[contract_address]
            deployer_account.balance += value
            return "", TransactionReceipt(
                success=False, gas_used=21000, events=[],
                revert_reason=str(e), sender=deployer, to="", value=value
            )

        # Estimate gas: base deployment cost + storage writes
        gas_used = 32000 + len(self.accounts[contract_address].storage) * 20000

        receipt = TransactionReceipt(
            success=True,
            gas_used=gas_used,
            events=events,
            sender=deployer,
            to=contract_address,
            value=value
        )

        return contract_address, receipt

    def call_contract(
        self,
        sender: str,
        contract_address: str,
        function_name: str,
        args: Optional[dict] = None,
        value: int = 0
    ) -> TransactionReceipt:
        """
        Execute a function on a deployed contract.

        In real Ethereum, this involves:
        1. ABI-encode the function call (selector + arguments)
        2. Create a transaction: {from, to, data, value, gasLimit}
        3. EVM loads contract code, jumps to the selector match
        4. Executes opcodes, modifying storage/emitting events
        5. Returns success + return data OR reverts

        We simulate this by calling the contract_logic's handle_call method,
        which dispatches to the appropriate Python function.
        """
        if contract_address not in self._contract_logic:
            return TransactionReceipt(
                success=False, gas_used=21000, events=[],
                revert_reason=f"No contract at {contract_address}",
                sender=sender, to=contract_address, value=value
            )

        if sender not in self.accounts:
            return TransactionReceipt(
                success=False, gas_used=21000, events=[],
                revert_reason=f"Sender {sender} does not exist",
                sender=sender, to=contract_address, value=value
            )

        sender_account = self.accounts[sender]

        # Check sender has enough ETH for value transfer
        if sender_account.balance < value:
            return TransactionReceipt(
                success=False, gas_used=21000, events=[],
                revert_reason="Insufficient balance",
                sender=sender, to=contract_address, value=value
            )

        # Transfer ETH value (this happens before code execution in the EVM)
        # This is important: if the contract reverts, the value must be refunded
        sender_account.balance -= value
        self.accounts[contract_address].balance += value

        contract_logic = self._contract_logic[contract_address]
        events: list[EventLog] = []

        try:
            return_data = contract_logic.handle_call(
                evm=self,
                address=contract_address,
                sender=sender,
                function_name=function_name,
                args=args or {},
                value=value,
                events=events
            )

            # Update sender nonce on successful transaction
            sender_account.nonce += 1

            gas_used = 21000 + 5000 * len(events)  # Base + per-event estimate

            return TransactionReceipt(
                success=True,
                gas_used=gas_used,
                events=events,
                return_data=return_data if isinstance(return_data, bytes) else b'',
                sender=sender,
                to=contract_address,
                value=value
            )

        except RevertError as e:
            # Revert: undo the value transfer (gas is still consumed)
            sender_account.balance += value
            self.accounts[contract_address].balance -= value

            return TransactionReceipt(
                success=False,
                gas_used=21000,
                events=[],
                revert_reason=str(e),
                sender=sender,
                to=contract_address,
                value=value
            )

    def transfer_eth(self, sender: str, to: str, amount: int) -> TransactionReceipt:
        """
        Simple ETH transfer between accounts.

        This is the simplest transaction type: just move ETH from one account
        to another. Gas cost is exactly 21,000 for a simple transfer (the minimum).
        """
        if sender not in self.accounts:
            return TransactionReceipt(
                success=False, gas_used=21000, events=[],
                revert_reason="Sender does not exist",
                sender=sender, to=to, value=amount
            )

        if self.accounts[sender].balance < amount:
            return TransactionReceipt(
                success=False, gas_used=21000, events=[],
                revert_reason="Insufficient balance",
                sender=sender, to=to, value=amount
            )

        self.accounts[sender].balance -= amount
        if to not in self.accounts:
            self.create_account(to)
        self.accounts[to].balance += amount
        self.accounts[sender].nonce += 1

        return TransactionReceipt(
            success=True, gas_used=21000, events=[],
            sender=sender, to=to, value=amount
        )

    # -------------------------------------------------------------------------
    # Snapshot/Revert — Essential for test isolation
    # -------------------------------------------------------------------------

    def snapshot(self) -> int:
        """
        Take a snapshot of the current EVM state.

        This captures EVERYTHING: all account balances, storage, nonces, and
        contract deployments. After running a test, you revert to this snapshot
        to get a clean state for the next test.

        Production EVMs (like Geth) use a Merkle Patricia Trie for state, which
        supports O(1) snapshots via structural sharing. We use a simple deep copy.
        """
        import copy
        snapshot_id = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._snapshots[snapshot_id] = {
            'accounts': copy.deepcopy(self.accounts),
            'block': copy.deepcopy(self.block),
            'contract_logic': dict(self._contract_logic),  # Shallow copy of logic refs
            'contract_names': dict(self._contract_names),
        }
        return snapshot_id

    def revert(self, snapshot_id: int) -> bool:
        """
        Revert to a previous snapshot, restoring all state.

        Returns True if successful, False if the snapshot doesn't exist.
        Note: the snapshot is consumed (deleted) after reverting — you can't
        revert to the same snapshot twice. Take a new snapshot if needed.
        """
        if snapshot_id not in self._snapshots:
            return False

        state = self._snapshots.pop(snapshot_id)
        self.accounts = state['accounts']
        self.block = state['block']
        self._contract_logic = state['contract_logic']
        self._contract_names = state['contract_names']
        return True

    # -------------------------------------------------------------------------
    # Time manipulation — Critical for testing time-dependent contracts
    # -------------------------------------------------------------------------

    def advance_time(self, seconds: int) -> None:
        """
        Advance the block timestamp. Contracts that use block.timestamp for
        time locks, vesting, or interest calculations need this for testing.
        """
        self.block.timestamp += seconds

    def advance_blocks(self, count: int) -> None:
        """Advance the block number. Used for governance voting periods, etc."""
        self.block.number += count
        # Each block is ~12 seconds on Ethereum mainnet
        self.block.timestamp += count * 12


# =============================================================================
# Contract Logic Base Class
# =============================================================================

class RevertError(Exception):
    """
    Raised when a contract reverts execution.

    In the real EVM, REVERT (opcode 0xFD) halts execution, refunds remaining gas,
    and returns error data. Solidity's require() and revert() compile to this opcode.

    We simulate this with a Python exception — the testing framework catches it
    and creates a failed TransactionReceipt.
    """
    pass


class ContractLogic:
    """
    Base class for simulated smart contract logic.

    Instead of writing Solidity and compiling to bytecode, we define contract
    behavior in Python. This lets us focus on testing patterns without needing
    an actual Solidity compiler.

    Each contract must implement:
    - constructor(): Called once during deployment
    - handle_call(): Dispatches function calls to the appropriate method
    """

    def constructor(self, evm: SimulatedEVM, address: str, sender: str,
                    args: dict, events: list[EventLog]) -> None:
        """Called during deployment. Override to initialize contract state."""
        pass

    def handle_call(self, evm: SimulatedEVM, address: str, sender: str,
                    function_name: str, args: dict, value: int,
                    events: list[EventLog]) -> Optional[bytes]:
        """
        Dispatch a function call. Override to implement contract functions.

        Returns:
            Optional bytes of return data (e.g., ABI-encoded uint256 for balanceOf)
        """
        raise RevertError(f"Function {function_name} not found")


# =============================================================================
# Example Contract: ERC-20 Token (Simplified)
# =============================================================================

class ERC20Token(ContractLogic):
    """
    A simplified ERC-20 token implementation for testing.

    ERC-20 is the standard interface for fungible tokens on Ethereum. It defines:
    - balanceOf(address): How many tokens an address holds
    - transfer(to, amount): Send tokens directly
    - approve(spender, amount): Allow someone else to spend your tokens
    - transferFrom(from, to, amount): Spend tokens you've been approved for
    - totalSupply(): Total tokens in existence

    This is the most battle-tested standard in crypto — every DEX, lending
    protocol, and DeFi aggregator depends on these 6 functions behaving
    exactly as specified. Getting the edge cases right is critical.

    Storage Layout:
    - Slot 0: totalSupply
    - Slot hash(owner, 1): balanceOf[owner]
    - Slot hash(owner, spender, 2): allowance[owner][spender]
    """

    TOTAL_SUPPLY_SLOT = 0
    BALANCE_PREFIX = 1
    ALLOWANCE_PREFIX = 2

    def _balance_slot(self, owner: str) -> int:
        """Compute storage slot for a balance. Simulates Solidity's mapping storage."""
        return int.from_bytes(keccak256(f"{owner}{self.BALANCE_PREFIX}".encode())[:8], 'big')

    def _allowance_slot(self, owner: str, spender: str) -> int:
        """Compute storage slot for an allowance. Nested mapping = double hash."""
        return int.from_bytes(keccak256(f"{owner}{spender}{self.ALLOWANCE_PREFIX}".encode())[:8], 'big')

    def constructor(self, evm: SimulatedEVM, address: str, sender: str,
                    args: dict, events: list[EventLog]) -> None:
        """
        Mint initial supply to the deployer.

        In most real ERC-20s, the constructor mints the total supply and assigns
        it to the deployer (or a treasury address). Some tokens have dynamic supply
        (mint/burn functions), but the initial distribution happens here.
        """
        initial_supply = args.get("initial_supply", 1_000_000 * 10**18)
        name = args.get("name", "TestToken")
        symbol = args.get("symbol", "TT")

        # Store total supply
        evm.set_storage(address, self.TOTAL_SUPPLY_SLOT, initial_supply)

        # Assign all tokens to deployer
        balance_slot = self._balance_slot(sender)
        evm.set_storage(address, balance_slot, initial_supply)

        # Emit Transfer event from zero address (standard for minting)
        events.append(EventLog(
            address=address,
            topics=[
                keccak256(b"Transfer(address,address,uint256)"),
                abi_encode_address("0x" + "0" * 40),
                abi_encode_address(sender)
            ],
            data=abi_encode_uint256(initial_supply),
            event_name="Transfer"
        ))

    def handle_call(self, evm: SimulatedEVM, address: str, sender: str,
                    function_name: str, args: dict, value: int,
                    events: list[EventLog]) -> Optional[bytes]:
        """Route function calls to implementations."""

        if function_name == "balanceOf":
            owner = args["owner"]
            balance = evm.get_storage(address, self._balance_slot(owner))
            return abi_encode_uint256(balance)

        elif function_name == "totalSupply":
            supply = evm.get_storage(address, self.TOTAL_SUPPLY_SLOT)
            return abi_encode_uint256(supply)

        elif function_name == "transfer":
            return self._transfer(evm, address, sender, args["to"], args["amount"], events)

        elif function_name == "approve":
            return self._approve(evm, address, sender, args["spender"], args["amount"], events)

        elif function_name == "allowance":
            slot = self._allowance_slot(args["owner"], args["spender"])
            return abi_encode_uint256(evm.get_storage(address, slot))

        elif function_name == "transferFrom":
            return self._transfer_from(
                evm, address, sender,
                args["from"], args["to"], args["amount"],
                events
            )

        else:
            raise RevertError(f"Function {function_name} not found")

    def _transfer(self, evm: SimulatedEVM, address: str, sender: str,
                  to: str, amount: int, events: list[EventLog]) -> Optional[bytes]:
        """
        Transfer tokens from sender to recipient.

        Critical checks:
        1. Cannot send to zero address (tokens would be unrecoverable)
        2. Sender must have sufficient balance (prevent underflow)
        3. Balance updates must be atomic (no partial state changes on failure)

        Note: We check and update balances in the correct order to prevent
        reentrancy. In the real EVM, the checks-effects-interactions pattern
        is essential — we update state BEFORE any external calls.
        """
        zero_address = "0x" + "0" * 40
        if to == zero_address:
            raise RevertError("ERC20: transfer to the zero address")

        sender_slot = self._balance_slot(sender)
        sender_balance = evm.get_storage(address, sender_slot)

        if sender_balance < amount:
            raise RevertError("ERC20: transfer amount exceeds balance")

        # Update balances (checks-effects pattern)
        evm.set_storage(address, sender_slot, sender_balance - amount)

        to_slot = self._balance_slot(to)
        to_balance = evm.get_storage(address, to_slot)
        evm.set_storage(address, to_slot, to_balance + amount)

        # Emit Transfer event
        events.append(EventLog(
            address=address,
            topics=[
                keccak256(b"Transfer(address,address,uint256)"),
                abi_encode_address(sender),
                abi_encode_address(to)
            ],
            data=abi_encode_uint256(amount),
            event_name="Transfer"
        ))

        return abi_encode_uint256(1)  # Return true

    def _approve(self, evm: SimulatedEVM, address: str, owner: str,
                 spender: str, amount: int, events: list[EventLog]) -> Optional[bytes]:
        """
        Approve a spender to transfer tokens on behalf of the owner.

        The approve/transferFrom pattern enables DEXes: you approve the DEX
        contract to spend your tokens, then the DEX transfers them during a swap.

        Security note: The "approve race condition" — if you change an approval
        from 100 to 50, the spender could front-run by spending 100 before the
        new approval, then spend the new 50, getting 150 total. Mitigation:
        approve to 0 first, then to the new amount. ERC-20 spec doesn't enforce this.
        """
        zero_address = "0x" + "0" * 40
        if spender == zero_address:
            raise RevertError("ERC20: approve to the zero address")

        slot = self._allowance_slot(owner, spender)
        evm.set_storage(address, slot, amount)

        events.append(EventLog(
            address=address,
            topics=[
                keccak256(b"Approval(address,address,uint256)"),
                abi_encode_address(owner),
                abi_encode_address(spender)
            ],
            data=abi_encode_uint256(amount),
            event_name="Approval"
        ))

        return abi_encode_uint256(1)  # Return true

    def _transfer_from(self, evm: SimulatedEVM, address: str, spender: str,
                       from_addr: str, to: str, amount: int,
                       events: list[EventLog]) -> Optional[bytes]:
        """
        Transfer tokens from one address to another using the allowance mechanism.

        This is the second half of the approve/transferFrom pattern. The spender
        must have sufficient allowance AND the from_addr must have sufficient balance.
        After the transfer, the allowance is reduced by the transferred amount.
        """
        zero_address = "0x" + "0" * 40
        if to == zero_address:
            raise RevertError("ERC20: transfer to the zero address")

        # Check allowance
        allowance_slot = self._allowance_slot(from_addr, spender)
        current_allowance = evm.get_storage(address, allowance_slot)

        if current_allowance < amount:
            raise RevertError("ERC20: insufficient allowance")

        # Check balance
        from_slot = self._balance_slot(from_addr)
        from_balance = evm.get_storage(address, from_slot)

        if from_balance < amount:
            raise RevertError("ERC20: transfer amount exceeds balance")

        # Update allowance
        evm.set_storage(address, allowance_slot, current_allowance - amount)

        # Update balances
        evm.set_storage(address, from_slot, from_balance - amount)
        to_slot = self._balance_slot(to)
        to_balance = evm.get_storage(address, to_slot)
        evm.set_storage(address, to_slot, to_balance + amount)

        # Emit Transfer event
        events.append(EventLog(
            address=address,
            topics=[
                keccak256(b"Transfer(address,address,uint256)"),
                abi_encode_address(from_addr),
                abi_encode_address(to)
            ],
            data=abi_encode_uint256(amount),
            event_name="Transfer"
        ))

        return abi_encode_uint256(1)


# =============================================================================
# Testing Assertion Utilities
# =============================================================================

class TestContext:
    """
    High-level testing utilities that make test code readable and expressive.

    These assertions are what differentiate a testing framework from just
    "running code and checking the output." They encode domain knowledge
    about blockchain semantics: reverts have reasons, events have structure,
    balances change by specific amounts.

    This is modeled after Hardhat's expect() and Foundry's vm.expect* cheatcodes.
    """

    def __init__(self, evm: SimulatedEVM):
        self.evm = evm

    def expect_revert(self, receipt: TransactionReceipt, reason: str = "") -> None:
        """
        Assert that a transaction reverted, optionally with a specific reason.

        Why check the reason, not just that it reverted? Because a transfer
        might revert for "insufficient balance" OR "paused" — if you only check
        that it reverted, you might miss that it's reverting for the wrong reason,
        hiding a real bug.
        """
        assert not receipt.success, (
            f"Expected transaction to revert but it succeeded. "
            f"Events emitted: {[e.event_name for e in receipt.events]}"
        )
        if reason:
            assert reason in receipt.revert_reason, (
                f"Expected revert reason '{reason}' but got '{receipt.revert_reason}'"
            )

    def expect_success(self, receipt: TransactionReceipt) -> None:
        """Assert that a transaction succeeded."""
        assert receipt.success, (
            f"Expected transaction to succeed but it reverted: {receipt.revert_reason}"
        )

    def expect_event(self, receipt: TransactionReceipt, event_name: str,
                     count: int = 1) -> list[EventLog]:
        """
        Assert that a specific event was emitted and return the matching events.

        Checking events is critical for:
        1. Verifying off-chain indexers will work (they depend on events)
        2. Confirming the contract communicated the right information
        3. Testing that events match the actual state changes
        """
        matching = [e for e in receipt.events if e.event_name == event_name]
        assert len(matching) == count, (
            f"Expected {count} '{event_name}' event(s) but found {len(matching)}. "
            f"Events emitted: {[e.event_name for e in receipt.events]}"
        )
        return matching

    def expect_balance_change(
        self,
        address: str,
        expected_change: int,
        action: Callable[[], TransactionReceipt]
    ) -> TransactionReceipt:
        """
        Assert that an action changes an account's ETH balance by the expected amount.

        This pattern wraps an action and checks the before/after balance difference.
        It's essential for testing:
        - ETH payments (value transfers in function calls)
        - Fee collection (protocol takes a cut)
        - Refunds (failed transactions should return ETH)

        The callable pattern ensures we capture the balance BEFORE the action runs.
        """
        balance_before = self.evm.get_balance(address)
        receipt = action()
        balance_after = self.evm.get_balance(address)
        actual_change = balance_after - balance_before

        assert actual_change == expected_change, (
            f"Expected balance change of {expected_change} for {address} "
            f"but got {actual_change} (before: {balance_before}, after: {balance_after})"
        )
        return receipt

    def expect_token_balance(self, token_address: str, token_logic: ERC20Token,
                             owner: str, expected: int) -> None:
        """Assert that an address holds the expected token balance."""
        slot = token_logic._balance_slot(owner)
        actual = self.evm.get_storage(token_address, slot)
        assert actual == expected, (
            f"Expected token balance {expected} for {owner} but got {actual}"
        )


# =============================================================================
# Test Runner
# =============================================================================

class TestRunner:
    """
    Minimal test runner with setup/teardown and snapshot isolation.

    Each test method runs in its own snapshot, so tests can't leak state
    to each other. This is the same pattern Hardhat uses with beforeEach
    and Foundry uses with setUp().
    """

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[tuple[str, str]] = []

    def run(self, test_class: type) -> None:
        """Run all test methods (prefixed with 'test_') in a test class."""
        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        print(f"\n{'='*60}")
        print(f"Running: {test_class.__name__}")
        print(f"{'='*60}")

        for method_name in sorted(test_methods):
            # Setup fresh state for each test
            if hasattr(instance, 'setUp'):
                instance.setUp()

            # Take snapshot for isolation
            evm = getattr(instance, 'evm', None)
            snapshot_id = evm.snapshot() if evm else None

            try:
                getattr(instance, method_name)()
                self.passed += 1
                print(f"  PASS  {method_name}")
            except AssertionError as e:
                self.failed += 1
                self.errors.append((method_name, str(e)))
                print(f"  FAIL  {method_name}: {e}")
            except Exception as e:
                self.failed += 1
                self.errors.append((method_name, str(e)))
                print(f"  ERROR {method_name}: {type(e).__name__}: {e}")
            finally:
                # Revert to snapshot (restore clean state)
                if evm and snapshot_id is not None:
                    evm.revert(snapshot_id)

        print(f"\nResults: {self.passed} passed, {self.failed} failed")

    def summary(self) -> str:
        total = self.passed + self.failed
        return f"{self.passed}/{total} tests passed"


# =============================================================================
# Example Tests: Full ERC-20 Test Suite
# =============================================================================

class TestERC20(object):
    """
    Comprehensive ERC-20 test suite demonstrating the testing framework.

    This covers:
    - Deployment and initial state
    - Basic transfers
    - Approve/transferFrom flow
    - Edge cases (zero address, insufficient balance, zero amount)
    - Event emission verification
    - State isolation via snapshots
    """

    # Test addresses (would be derived from private keys in production)
    DEPLOYER = "0x" + "1" * 40
    ALICE = "0x" + "2" * 40
    BOB = "0x" + "3" * 40
    ZERO = "0x" + "0" * 40

    INITIAL_SUPPLY = 1_000_000 * 10**18  # 1M tokens with 18 decimals

    def setUp(self):
        """Fresh EVM and contract deployment for each test."""
        self.evm = SimulatedEVM()
        self.token_logic = ERC20Token()
        self.ctx = TestContext(self.evm)

        # Fund accounts with ETH (for gas)
        self.evm.create_account(self.DEPLOYER, 100 * 10**18)
        self.evm.create_account(self.ALICE, 100 * 10**18)
        self.evm.create_account(self.BOB, 100 * 10**18)

        # Deploy token
        self.token_addr, receipt = self.evm.deploy_contract(
            deployer=self.DEPLOYER,
            contract_name="TestToken",
            contract_logic=self.token_logic,
            constructor_args={
                "initial_supply": self.INITIAL_SUPPLY,
                "name": "TestToken",
                "symbol": "TT"
            }
        )
        assert receipt.success, f"Deployment failed: {receipt.revert_reason}"

    def test_deployment_mints_total_supply(self):
        """Total supply should equal initial supply after deployment."""
        receipt = self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "totalSupply"
        )
        self.ctx.expect_success(receipt)
        supply = abi_decode_uint256(receipt.return_data)
        assert supply == self.INITIAL_SUPPLY, f"Expected {self.INITIAL_SUPPLY}, got {supply}"

    def test_deployer_receives_all_tokens(self):
        """Deployer should hold all tokens after deployment."""
        self.ctx.expect_token_balance(
            self.token_addr, self.token_logic, self.DEPLOYER, self.INITIAL_SUPPLY
        )

    def test_transfer_moves_tokens(self):
        """Transfer should move tokens from sender to recipient."""
        amount = 1000 * 10**18

        receipt = self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "transfer",
            {"to": self.ALICE, "amount": amount}
        )
        self.ctx.expect_success(receipt)

        # Verify balances
        self.ctx.expect_token_balance(
            self.token_addr, self.token_logic, self.DEPLOYER,
            self.INITIAL_SUPPLY - amount
        )
        self.ctx.expect_token_balance(
            self.token_addr, self.token_logic, self.ALICE, amount
        )

    def test_transfer_emits_event(self):
        """Transfer should emit a Transfer event."""
        amount = 500 * 10**18
        receipt = self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "transfer",
            {"to": self.ALICE, "amount": amount}
        )
        events = self.ctx.expect_event(receipt, "Transfer", count=1)

        # Verify event data contains the correct amount
        event_amount = abi_decode_uint256(events[0].data)
        assert event_amount == amount, f"Event amount {event_amount} != {amount}"

    def test_transfer_insufficient_balance_reverts(self):
        """Transfer more than balance should revert."""
        receipt = self.evm.call_contract(
            self.ALICE, self.token_addr, "transfer",
            {"to": self.BOB, "amount": 1}  # Alice has 0 tokens
        )
        self.ctx.expect_revert(receipt, "transfer amount exceeds balance")

    def test_transfer_to_zero_address_reverts(self):
        """Transfer to zero address should revert (prevents token burning by accident)."""
        receipt = self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "transfer",
            {"to": self.ZERO, "amount": 100}
        )
        self.ctx.expect_revert(receipt, "transfer to the zero address")

    def test_approve_and_transfer_from(self):
        """Full approve/transferFrom flow: deployer approves Alice, Alice transfers."""
        amount = 2000 * 10**18

        # Step 1: Deployer approves Alice
        receipt = self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "approve",
            {"spender": self.ALICE, "amount": amount}
        )
        self.ctx.expect_success(receipt)
        self.ctx.expect_event(receipt, "Approval")

        # Verify allowance
        receipt = self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "allowance",
            {"owner": self.DEPLOYER, "spender": self.ALICE}
        )
        allowance = abi_decode_uint256(receipt.return_data)
        assert allowance == amount

        # Step 2: Alice transfers from Deployer to Bob
        transfer_amount = 1500 * 10**18
        receipt = self.evm.call_contract(
            self.ALICE, self.token_addr, "transferFrom",
            {"from": self.DEPLOYER, "to": self.BOB, "amount": transfer_amount}
        )
        self.ctx.expect_success(receipt)

        # Verify balances
        self.ctx.expect_token_balance(
            self.token_addr, self.token_logic, self.DEPLOYER,
            self.INITIAL_SUPPLY - transfer_amount
        )
        self.ctx.expect_token_balance(
            self.token_addr, self.token_logic, self.BOB, transfer_amount
        )

        # Verify allowance decreased
        receipt = self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "allowance",
            {"owner": self.DEPLOYER, "spender": self.ALICE}
        )
        remaining = abi_decode_uint256(receipt.return_data)
        assert remaining == amount - transfer_amount

    def test_transfer_from_exceeds_allowance_reverts(self):
        """transferFrom more than allowance should revert."""
        # Approve 100 tokens
        self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "approve",
            {"spender": self.ALICE, "amount": 100}
        )

        # Try to transfer 200
        receipt = self.evm.call_contract(
            self.ALICE, self.token_addr, "transferFrom",
            {"from": self.DEPLOYER, "to": self.BOB, "amount": 200}
        )
        self.ctx.expect_revert(receipt, "insufficient allowance")

    def test_snapshot_revert_isolates_state(self):
        """Demonstrate that snapshot/revert properly isolates state."""
        # Transfer tokens
        amount = 5000 * 10**18
        self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "transfer",
            {"to": self.ALICE, "amount": amount}
        )

        # Take snapshot AFTER the transfer
        snap = self.evm.snapshot()

        # Transfer more tokens
        self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "transfer",
            {"to": self.BOB, "amount": amount}
        )

        # Bob should have tokens now
        self.ctx.expect_token_balance(
            self.token_addr, self.token_logic, self.BOB, amount
        )

        # Revert — Bob's transfer should be undone
        self.evm.revert(snap)

        # Bob should have 0 again, Alice should still have her tokens
        self.ctx.expect_token_balance(
            self.token_addr, self.token_logic, self.BOB, 0
        )
        self.ctx.expect_token_balance(
            self.token_addr, self.token_logic, self.ALICE, amount
        )

    def test_zero_transfer_succeeds(self):
        """Transferring 0 tokens should succeed (some protocols rely on this)."""
        receipt = self.evm.call_contract(
            self.DEPLOYER, self.token_addr, "transfer",
            {"to": self.ALICE, "amount": 0}
        )
        self.ctx.expect_success(receipt)


# =============================================================================
# ABI Encoding Tests
# =============================================================================

class TestABIEncoding(object):
    """Test the ABI encoding/decoding utilities directly."""

    def setUp(self):
        self.evm = SimulatedEVM()  # Needed for test runner snapshot

    def test_uint256_encoding_zero(self):
        """Zero should encode as 32 zero bytes."""
        encoded = abi_encode_uint256(0)
        assert len(encoded) == 32
        assert encoded == b'\x00' * 32

    def test_uint256_encoding_roundtrip(self):
        """Encode then decode should return the original value."""
        for value in [0, 1, 255, 1000, 2**128, 2**256 - 1]:
            encoded = abi_encode_uint256(value)
            decoded = abi_decode_uint256(encoded)
            assert decoded == value, f"Roundtrip failed for {value}: got {decoded}"

    def test_address_encoding_length(self):
        """Address should encode to exactly 32 bytes."""
        encoded = abi_encode_address("0x" + "ab" * 20)
        assert len(encoded) == 32

    def test_address_encoding_left_padded(self):
        """Address should be left-padded with 12 zero bytes."""
        addr = "0x" + "ff" * 20
        encoded = abi_encode_address(addr)
        assert encoded[:12] == b'\x00' * 12
        assert encoded[12:] == b'\xff' * 20

    def test_function_selector_deterministic(self):
        """Same signature should always produce the same selector."""
        sig = "transfer(address,uint256)"
        selector1 = compute_function_selector(sig)
        selector2 = compute_function_selector(sig)
        assert selector1 == selector2
        assert len(selector1) == 4

    def test_full_call_encoding(self):
        """Full call encoding should be selector + encoded args."""
        calldata = abi_encode_call(
            "transfer(address,uint256)",
            ("address", "0x" + "ab" * 20),
            ("uint256", 1000)
        )
        # 4 bytes selector + 32 bytes address + 32 bytes uint256 = 68 bytes
        assert len(calldata) == 68


# =============================================================================
# Main: Run all tests and demonstrate the framework
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Smart Contract Testing Framework")
    print("=" * 60)

    # --- Part 1: Demonstrate ABI encoding ---
    print("\n--- ABI Encoding Demo ---")

    sig = "transfer(address,uint256)"
    selector = compute_function_selector(sig)
    print(f"Function: {sig}")
    print(f"Selector: 0x{selector.hex()}")

    addr = "0x" + "ab" * 20
    encoded_addr = abi_encode_address(addr)
    print(f"\nAddress {addr}")
    print(f"Encoded: 0x{encoded_addr.hex()}")

    value = 1000
    encoded_val = abi_encode_uint256(value)
    print(f"\nUint256 {value}")
    print(f"Encoded: 0x{encoded_val.hex()}")

    calldata = abi_encode_call(sig, ("address", addr), ("uint256", value))
    print(f"\nFull calldata ({len(calldata)} bytes):")
    print(f"  Selector: 0x{calldata[:4].hex()}")
    print(f"  Arg 0:    0x{calldata[4:36].hex()}")
    print(f"  Arg 1:    0x{calldata[36:68].hex()}")

    # --- Part 2: Deploy and interact with ERC-20 ---
    print("\n--- ERC-20 Token Demo ---")

    evm = SimulatedEVM()
    deployer = "0x" + "1" * 40
    alice = "0x" + "2" * 40
    bob = "0x" + "3" * 40

    evm.create_account(deployer, 100 * 10**18)
    evm.create_account(alice, 100 * 10**18)
    evm.create_account(bob, 100 * 10**18)

    initial_supply = 1_000_000 * 10**18
    token_logic = ERC20Token()

    token_addr, deploy_receipt = evm.deploy_contract(
        deployer=deployer,
        contract_name="MyToken",
        contract_logic=token_logic,
        constructor_args={"initial_supply": initial_supply, "name": "MyToken", "symbol": "MTK"}
    )
    print(f"Token deployed at: {token_addr}")
    print(f"Deploy gas: {deploy_receipt.gas_used}")
    print(f"Events: {[e.event_name for e in deploy_receipt.events]}")

    # Check deployer balance
    receipt = evm.call_contract(deployer, token_addr, "balanceOf", {"owner": deployer})
    balance = abi_decode_uint256(receipt.return_data)
    print(f"\nDeployer balance: {balance / 10**18:,.0f} MTK")

    # Transfer tokens
    transfer_amount = 50_000 * 10**18
    receipt = evm.call_contract(
        deployer, token_addr, "transfer",
        {"to": alice, "amount": transfer_amount}
    )
    print(f"\nTransfer {transfer_amount / 10**18:,.0f} MTK to Alice: {'SUCCESS' if receipt.success else 'FAILED'}")
    print(f"Events: {[e.event_name for e in receipt.events]}")

    # Check balances
    receipt = evm.call_contract(deployer, token_addr, "balanceOf", {"owner": deployer})
    print(f"Deployer: {abi_decode_uint256(receipt.return_data) / 10**18:,.0f} MTK")

    receipt = evm.call_contract(deployer, token_addr, "balanceOf", {"owner": alice})
    print(f"Alice:    {abi_decode_uint256(receipt.return_data) / 10**18:,.0f} MTK")

    # Approve and transferFrom
    print("\n--- Approve/TransferFrom Flow ---")
    approve_amount = 10_000 * 10**18
    evm.call_contract(deployer, token_addr, "approve", {"spender": alice, "amount": approve_amount})
    print(f"Deployer approved Alice for {approve_amount / 10**18:,.0f} MTK")

    from_amount = 5_000 * 10**18
    receipt = evm.call_contract(
        alice, token_addr, "transferFrom",
        {"from": deployer, "to": bob, "amount": from_amount}
    )
    print(f"Alice transferred {from_amount / 10**18:,.0f} MTK from Deployer to Bob: {'SUCCESS' if receipt.success else 'FAILED'}")

    receipt = evm.call_contract(deployer, token_addr, "balanceOf", {"owner": bob})
    print(f"Bob:      {abi_decode_uint256(receipt.return_data) / 10**18:,.0f} MTK")

    # Demonstrate revert
    print("\n--- Revert Demo ---")
    receipt = evm.call_contract(
        bob, token_addr, "transfer",
        {"to": alice, "amount": 999_999 * 10**18}  # Bob doesn't have this much
    )
    print(f"Bob tries to transfer 999,999 MTK: {'SUCCESS' if receipt.success else 'REVERTED'}")
    print(f"Reason: {receipt.revert_reason}")

    # Demonstrate snapshot/revert
    print("\n--- Snapshot/Revert Demo ---")
    snap_id = evm.snapshot()
    print(f"Snapshot taken (id: {snap_id})")

    evm.call_contract(deployer, token_addr, "transfer", {"to": bob, "amount": 100_000 * 10**18})
    receipt = evm.call_contract(deployer, token_addr, "balanceOf", {"owner": bob})
    print(f"Bob after transfer: {abi_decode_uint256(receipt.return_data) / 10**18:,.0f} MTK")

    evm.revert(snap_id)
    receipt = evm.call_contract(deployer, token_addr, "balanceOf", {"owner": bob})
    print(f"Bob after revert:   {abi_decode_uint256(receipt.return_data) / 10**18:,.0f} MTK")

    # --- Part 3: Run the test suite ---
    runner = TestRunner()
    runner.run(TestABIEncoding)
    runner.run(TestERC20)

    print(f"\n{'='*60}")
    print(f"Final: {runner.summary()}")
    print(f"{'='*60}")
