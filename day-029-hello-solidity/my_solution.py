"""
Day 29: Hello World Solidity Contract — Your Implementation

Build a mini smart contract VM that simulates Ethereum contract execution:
storage, function dispatch, gas metering, events, and access control.

Hints:
- Start with ContractStorage: it's just a dict with gas tracking
- GasMeter is an accumulator that raises when limit exceeded
- Function dispatch maps selector strings to callables
- The contract ties everything together: storage + dispatch + events

Run tests: python3 -m pytest tests.py -v
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# =============================================================================
# Storage Engine
# =============================================================================

class ContractStorage:
    """Simulates EVM contract storage with gas-aware read/write tracking.

    Key insight: EVM storage is a uint256->uint256 mapping. Each read/write
    costs gas, with cold (first access) being more expensive than warm (cached).

    Hint: You need _slots (the data), _warm_slots (access cache), and
    the gas cost constants from EIP-2929.
    """

    SLOAD_COLD_GAS = 2100
    SLOAD_WARM_GAS = 100
    SSTORE_NEW_GAS = 20000
    SSTORE_UPDATE_GAS = 5000

    def __init__(self) -> None:
        raise NotImplementedError("TODO: initialize storage slots and warm cache")

    def sload(self, key: str, gas_meter: "GasMeter") -> Any:
        """Read a storage slot. Charge cold gas on first access, warm on subsequent.

        Hint: Check if key is in _warm_slots to determine cold vs warm.
        Return None for unset slots (like EVM returns 0 for empty slots).
        """
        raise NotImplementedError("TODO: implement storage read with gas accounting")

    def sstore(self, key: str, value: Any, gas_meter: "GasMeter") -> None:
        """Write a storage slot. New slots cost more than updates.

        Hint: Check if the old value was None/0 (new) vs existing (update)
        to determine the gas cost.
        """
        raise NotImplementedError("TODO: implement storage write with gas accounting")

    def snapshot(self) -> dict[str, Any]:
        """Take a snapshot of current state for potential rollback."""
        raise NotImplementedError("TODO: return a copy of _slots")

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore storage to a previous snapshot (used on revert)."""
        raise NotImplementedError("TODO: restore _slots from snapshot")

    def reset_warm_cache(self) -> None:
        """Clear the warm cache between transactions."""
        raise NotImplementedError("TODO: clear _warm_slots")

    def dump(self) -> dict[str, Any]:
        """Return a copy of all storage slots."""
        raise NotImplementedError("TODO: return copy of _slots")


# =============================================================================
# Gas Metering
# =============================================================================

class OutOfGasError(Exception):
    """Raised when execution exceeds the gas limit."""
    pass


class GasMeter:
    """Tracks gas consumption. Raises OutOfGasError if limit exceeded.

    Hint: Track gas_limit, gas_used, and a gas_log list.
    The consume() method is the core — it adds gas and checks the limit.
    """

    BASE_TRANSACTION_GAS = 21000

    def __init__(self, gas_limit: int) -> None:
        raise NotImplementedError("TODO: initialize gas tracking state")

    def consume(self, amount: int, operation: str = "") -> None:
        """Consume gas for an operation. Raise OutOfGasError if over limit.

        Hint: Add amount to gas_used, log it, then check if over limit.
        The check AFTER adding ensures we catch the exact operation that exceeded.
        """
        raise NotImplementedError("TODO: implement gas consumption with limit check")

    @property
    def gas_remaining(self) -> int:
        raise NotImplementedError("TODO: return remaining gas")


# =============================================================================
# Event System
# =============================================================================

EVENT_LOG_GAS = 375

@dataclass
class Event:
    """An emitted event. Like Solidity's `emit Transfer(from, to, amount)`."""
    name: str
    args: dict[str, Any]
    emitter: str
    block_number: int = 0


class EventLog:
    """Collects events during execution.

    Hint: Just a list of Events with emit() and filter-by-name.
    Don't forget to charge gas on emit!
    """

    def __init__(self) -> None:
        raise NotImplementedError("TODO: initialize events list")

    def emit(self, event: Event, gas_meter: GasMeter) -> None:
        raise NotImplementedError("TODO: charge gas and append event")

    def get_events(self, name: Optional[str] = None) -> list[Event]:
        raise NotImplementedError("TODO: return events, optionally filtered by name")


# =============================================================================
# Function Dispatch
# =============================================================================

def compute_selector(signature: str) -> str:
    """Compute a 4-byte function selector from a function signature.

    Hint: Hash the signature with SHA-256, take the first 8 hex chars,
    prefix with '0x'. This mirrors how keccak256 selectors work in the EVM.
    """
    raise NotImplementedError("TODO: implement selector computation")


@dataclass
class FunctionABI:
    """Describes a contract function."""
    name: str
    signature: str
    selector: str
    handler: Callable
    mutates_state: bool
    gas_cost: int


class FunctionRegistry:
    """Maps selectors to function implementations. This IS the dispatcher.

    Hint: Internal dict mapping selector string -> FunctionABI.
    register() computes the selector and stores the mapping.
    dispatch() looks up by selector, resolve_by_name() looks up by name.
    """

    def __init__(self) -> None:
        raise NotImplementedError("TODO: initialize function mapping")

    def register(self, name: str, signature: str, handler: Callable,
                 mutates_state: bool = False, gas_cost: int = 200) -> str:
        raise NotImplementedError("TODO: compute selector, store FunctionABI, return selector")

    def dispatch(self, selector: str) -> Optional[FunctionABI]:
        raise NotImplementedError("TODO: look up function by selector")

    def resolve_by_name(self, name: str) -> Optional[FunctionABI]:
        raise NotImplementedError("TODO: find function by name")

    def list_functions(self) -> list[FunctionABI]:
        raise NotImplementedError("TODO: return all registered functions")


# =============================================================================
# Execution Context
# =============================================================================

@dataclass
class MessageContext:
    """The msg object: sender, value, data."""
    sender: str
    value: int = 0
    data: bytes = b""


# =============================================================================
# Smart Contract Base
# =============================================================================

class RevertError(Exception):
    """Raised when require() fails. Undoes state changes."""
    pass


class SmartContract:
    """Base class for contracts.

    Hint: Store address, deployer (owner), and create instances of
    ContractStorage, FunctionRegistry, and EventLog.
    """

    def __init__(self, address: str, deployer: str) -> None:
        raise NotImplementedError("TODO: initialize contract state")

    def require(self, condition: bool, message: str = "Requirement failed") -> None:
        raise NotImplementedError("TODO: raise RevertError if condition is false")

    def emit_event(self, name: str, args: dict[str, Any], gas_meter: GasMeter) -> None:
        raise NotImplementedError("TODO: create Event and emit through event_log")


# =============================================================================
# Hello World Contract
# =============================================================================

class HelloWorldContract(SmartContract):
    """Your Hello World contract. Deploy with a greeting, read/update it.

    Functions to implement:
    - constructor: init storage (greeting, owner, greetingCount), register functions
    - _get_greeting: read greeting from storage
    - _set_greeting: owner-only greeting update with event emission
    - _get_owner: read owner from storage
    - _transfer_ownership: owner-only ownership transfer
    - _get_greeting_count: read greeting change counter

    Hint: Each function takes (self, msg, gas_meter, **kwargs).
    Use self.storage.sload/sstore for reads/writes.
    Use self.require(msg.sender == owner) for access control.
    """

    def __init__(self, address: str, deployer: str, initial_greeting: str,
                 gas_meter: GasMeter) -> None:
        raise NotImplementedError("TODO: call super().__init__, init storage, register functions")

    def _get_greeting(self, msg: MessageContext, gas_meter: GasMeter) -> str:
        raise NotImplementedError("TODO: sload and return greeting")

    def _set_greeting(self, msg: MessageContext, gas_meter: GasMeter,
                      new_greeting: str = "") -> str:
        raise NotImplementedError("TODO: check owner, update greeting, increment count, emit event")

    def _get_owner(self, msg: MessageContext, gas_meter: GasMeter) -> str:
        raise NotImplementedError("TODO: sload and return owner")

    def _transfer_ownership(self, msg: MessageContext, gas_meter: GasMeter,
                            new_owner: str = "") -> str:
        raise NotImplementedError("TODO: check owner, update owner, emit event")

    def _get_greeting_count(self, msg: MessageContext, gas_meter: GasMeter) -> int:
        raise NotImplementedError("TODO: sload and return greetingCount")


# =============================================================================
# Mini EVM
# =============================================================================

class MiniEVM:
    """Minimal EVM: deploy contracts and execute transactions.

    Hint: Track contracts (address->contract), nonces (for address derivation),
    block_number, and transaction_log.

    deploy_contract: derive address, create contract, handle errors
    call_function: resolve function, snapshot state, execute, rollback on error
    """

    def __init__(self) -> None:
        raise NotImplementedError("TODO: initialize EVM state")

    def _derive_address(self, deployer: str) -> str:
        """Derive contract address from deployer + nonce.

        Hint: hash(f"{deployer}:{nonce}") and take first 40 hex chars.
        """
        raise NotImplementedError("TODO: implement address derivation")

    def deploy_contract(self, deployer: str, initial_greeting: str,
                        gas_limit: int = 300000) -> tuple[Optional[HelloWorldContract], dict]:
        """Deploy a HelloWorld contract. Return (contract, receipt).

        Hint: Create GasMeter, charge base gas, derive address, construct contract.
        On OutOfGasError, return (None, error_receipt).
        """
        raise NotImplementedError("TODO: implement contract deployment")

    def call_function(self, contract_address: str, function_name: str,
                      sender: str, gas_limit: int = 100000,
                      **kwargs: Any) -> dict[str, Any]:
        """Call a function on a deployed contract. Return receipt dict.

        Hint: Look up contract, create gas meter + msg context, resolve function,
        snapshot storage, execute handler, return receipt. On RevertError or
        OutOfGasError, restore snapshot and return error receipt.
        """
        raise NotImplementedError("TODO: implement function call execution")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    evm = MiniEVM()
    alice = "0xAlice"

    # Deploy
    contract, receipt = evm.deploy_contract(alice, "Hello, World!", gas_limit=300000)
    print(f"Deploy status: {receipt['status']}")
    print(f"Contract address: {receipt.get('contract_address', 'N/A')}")

    if contract:
        addr = receipt["contract_address"]

        # Read greeting
        result = evm.call_function(addr, "getGreeting", sender=alice)
        print(f"Greeting: {result.get('result')}")

        # Update greeting
        result = evm.call_function(addr, "setGreeting", sender=alice,
                                   new_greeting="Updated!")
        print(f"Set greeting status: {result['status']}")

        # Unauthorized access
        bob = "0xBob"
        result = evm.call_function(addr, "setGreeting", sender=bob,
                                   new_greeting="Hacked!")
        print(f"Bob's attempt: {result['status']} - {result.get('error', 'no error')}")

        # Check count
        result = evm.call_function(addr, "getGreetingCount", sender=alice)
        print(f"Greeting changes: {result.get('result')}")
