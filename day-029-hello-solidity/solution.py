"""
Day 29: Hello World Solidity Contract — Python Smart Contract VM Simulation

This module builds a mini smart contract execution engine from scratch.
Instead of just writing Solidity syntax, we simulate HOW contracts actually
execute: deployment, storage, function dispatch, gas metering, events, and
access control.

Every concept maps directly to Ethereum's EVM:
- ContractStorage  -> EVM's SLOAD/SSTORE (slot-based key-value store)
- FunctionSelector -> keccak256(signature)[:4] dispatch
- GasMeter         -> 21000 base + per-operation costs
- EventLog         -> LOG0-LOG4 opcodes (transaction receipts)
- ExecutionContext  -> msg.sender, msg.value, msg.data
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# =============================================================================
# Storage Engine
# =============================================================================
# The EVM stores contract state as a mapping: uint256 -> uint256
# Each slot is 256 bits (32 bytes). Reading costs 2100 gas (cold) or 100 (warm).
# Writing costs 20000 (zero->nonzero) or 5000 (nonzero->nonzero).
# Our simulation uses string keys for readability but models the same cost structure.

class ContractStorage:
    """Simulates EVM contract storage with gas-aware read/write tracking."""

    # Gas costs modeled after EIP-2929 (Berlin hardfork)
    SLOAD_COLD_GAS = 2100    # First read of a slot in a transaction
    SLOAD_WARM_GAS = 100     # Subsequent reads of same slot
    SSTORE_NEW_GAS = 20000   # Writing to a slot that was zero
    SSTORE_UPDATE_GAS = 5000 # Updating a non-zero slot

    def __init__(self) -> None:
        self._slots: dict[str, Any] = {}
        # Track which slots have been accessed this transaction (warm vs cold)
        self._warm_slots: set[str] = set()

    def sload(self, key: str, gas_meter: "GasMeter") -> Any:
        """Read a storage slot. Charges cold/warm gas accordingly.

        Why cold vs warm? The EVM caches accessed slots per-transaction.
        First access loads from disk (cold = expensive), subsequent accesses
        hit the cache (warm = cheap). This incentivizes access patterns that
        touch each slot once rather than repeatedly.
        """
        if key in self._warm_slots:
            gas_meter.consume(self.SLOAD_WARM_GAS, f"SLOAD(warm) '{key}'")
        else:
            gas_meter.consume(self.SLOAD_COLD_GAS, f"SLOAD(cold) '{key}'")
            self._warm_slots.add(key)
        return self._slots.get(key)

    def sstore(self, key: str, value: Any, gas_meter: "GasMeter") -> None:
        """Write a storage slot. Gas depends on whether slot was previously zero.

        Why different costs? Writing a new slot allocates storage in the state trie,
        which is expensive (the node must store this forever). Updating an existing
        slot just modifies a leaf, which is cheaper.
        """
        old_value = self._slots.get(key)
        if old_value is None or old_value == 0:
            gas_meter.consume(self.SSTORE_NEW_GAS, f"SSTORE(new) '{key}'={value}")
        else:
            gas_meter.consume(self.SSTORE_UPDATE_GAS, f"SSTORE(update) '{key}'={value}")
        self._slots[key] = value
        self._warm_slots.add(key)

    def snapshot(self) -> dict[str, Any]:
        """Take a snapshot for rollback on revert. Returns a copy of current state."""
        return dict(self._slots)

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore storage to a previous snapshot. Used when execution reverts."""
        self._slots = snapshot

    def reset_warm_cache(self) -> None:
        """Clear warm cache between transactions."""
        self._warm_slots.clear()

    def dump(self) -> dict[str, Any]:
        """Return a copy of all storage slots for inspection."""
        return dict(self._slots)


# =============================================================================
# Gas Metering
# =============================================================================
# Gas is the fundamental resource-limiting mechanism in Ethereum.
# Without it, anyone could submit an infinite loop and halt the network.
# Our meter tracks consumption and raises OutOfGasError if the limit is exceeded.

class OutOfGasError(Exception):
    """Raised when execution exceeds the gas limit.

    In a real blockchain, this causes the entire transaction to revert,
    but the gas fee is still charged (the miner did the computational work).
    """
    pass


class GasMeter:
    """Tracks gas consumption during contract execution."""

    BASE_TRANSACTION_GAS = 21000  # Every transaction costs at least 21000 gas

    def __init__(self, gas_limit: int) -> None:
        self.gas_limit = gas_limit
        self.gas_used = 0
        self.gas_log: list[str] = []  # Detailed log of gas consumption

    def consume(self, amount: int, operation: str = "") -> None:
        """Consume gas for an operation. Reverts if limit exceeded.

        Why revert the entire transaction? Atomicity guarantee.
        Ethereum ensures either ALL state changes in a transaction apply,
        or NONE do. Partial execution would leave contracts in inconsistent states.
        """
        self.gas_used += amount
        self.gas_log.append(f"  {operation}: {amount} gas (total: {self.gas_used})")
        if self.gas_used > self.gas_limit:
            raise OutOfGasError(
                f"Out of gas: used {self.gas_used} > limit {self.gas_limit} "
                f"during '{operation}'"
            )

    @property
    def gas_remaining(self) -> int:
        return max(0, self.gas_limit - self.gas_used)


# =============================================================================
# Event / Log System
# =============================================================================
# Events are the EVM's LOG opcodes. They write data to transaction receipts,
# NOT to contract storage. This makes them ~10x cheaper than SSTORE.
# The tradeoff: contracts cannot read their own events. Events are only
# accessible off-chain (by indexers like The Graph, or block explorers).

EVENT_LOG_GAS = 375  # LOG0 base cost (simplified)

@dataclass
class Event:
    """Represents an emitted event (LOG opcode output).

    In Solidity: `emit Transfer(from, to, amount);`
    The event name is the topic (indexed for fast lookup).
    Args are the data payload.
    """
    name: str
    args: dict[str, Any]
    emitter: str  # Contract address that emitted it
    block_number: int = 0


class EventLog:
    """Collects events emitted during execution. Models transaction receipts."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event, gas_meter: GasMeter) -> None:
        gas_meter.consume(EVENT_LOG_GAS, f"LOG '{event.name}'")
        self.events.append(event)

    def get_events(self, name: Optional[str] = None) -> list[Event]:
        """Filter events by name. Like filtering by topic in eth_getLogs."""
        if name is None:
            return list(self.events)
        return [e for e in self.events if e.name == name]


# =============================================================================
# Function Selector / ABI Dispatch
# =============================================================================
# In the real EVM, calldata starts with a 4-byte function selector:
#   selector = keccak256("functionName(type1,type2)")[:4]
# The contract's code starts with a jump table that routes to the right function.
# We simulate this with a registry that maps selectors to callables.

def compute_selector(signature: str) -> str:
    """Compute 4-byte function selector from signature.

    In production this uses keccak256. We use SHA-256 for simplicity
    (no external dependencies). The principle is identical:
    deterministic hash -> first 4 bytes -> unique function identifier.

    Example: "transfer(address,uint256)" -> "0xa9059cbb"
    """
    hash_bytes = hashlib.sha256(signature.encode()).hexdigest()[:8]
    return f"0x{hash_bytes}"


@dataclass
class FunctionABI:
    """Describes a contract function's interface."""
    name: str
    signature: str       # e.g., "setGreeting(string)"
    selector: str        # 4-byte hex selector
    handler: Callable    # The actual implementation
    mutates_state: bool  # Whether this function writes storage (view vs nonpayable)
    gas_cost: int        # Additional gas for function execution overhead


class FunctionRegistry:
    """Maps function selectors to implementations. This IS the contract's dispatcher."""

    def __init__(self) -> None:
        self._functions: dict[str, FunctionABI] = {}

    def register(self, name: str, signature: str, handler: Callable,
                 mutates_state: bool = False, gas_cost: int = 200) -> str:
        """Register a function. Returns the computed selector.

        Why a registry? In the EVM, the Solidity compiler generates a dispatcher
        at the start of the contract bytecode. It's literally:
            if selector == 0xa9059cbb: jump to transfer
            if selector == 0x70a08231: jump to balanceOf
            ...
            revert  // unknown function
        """
        selector = compute_selector(signature)
        self._functions[selector] = FunctionABI(
            name=name,
            signature=signature,
            selector=selector,
            handler=handler,
            mutates_state=mutates_state,
            gas_cost=gas_cost,
        )
        return selector

    def dispatch(self, selector: str) -> Optional[FunctionABI]:
        """Look up a function by its selector."""
        return self._functions.get(selector)

    def resolve_by_name(self, name: str) -> Optional[FunctionABI]:
        """Convenience: find function by name (not how the EVM works, but useful for our sim)."""
        for fn in self._functions.values():
            if fn.name == name:
                return fn
        return None

    def list_functions(self) -> list[FunctionABI]:
        return list(self._functions.values())


# =============================================================================
# Execution Context (msg object)
# =============================================================================
# Every EVM call has a context: who sent it (msg.sender), how much ETH
# (msg.value), and the calldata. This is how contracts know WHO is calling
# and can enforce access control.

@dataclass
class MessageContext:
    """The msg object available during contract execution.

    msg.sender is THE fundamental security primitive in Ethereum.
    It's cryptographically guaranteed — you can't spoof it because
    transactions are signed with the sender's private key.
    """
    sender: str          # Address of the caller (msg.sender)
    value: int = 0       # Wei sent with the call (msg.value)
    data: bytes = b""    # Raw calldata


# =============================================================================
# Smart Contract Base
# =============================================================================

class RevertError(Exception):
    """Raised when a contract explicitly reverts (require/revert/assert).

    Revert undoes all state changes in the current call but does NOT
    consume all gas (unlike out-of-gas, which does in pre-London EVM).
    Post-EIP-3529, the refund model changed, but reverts still return unused gas.
    """
    pass


class SmartContract:
    """Base class for a smart contract. Manages storage, dispatch, and events."""

    def __init__(self, address: str, deployer: str) -> None:
        self.address = address
        self.storage = ContractStorage()
        self.functions = FunctionRegistry()
        self.event_log = EventLog()
        self.owner = deployer  # Set in constructor, immutable by convention

    def require(self, condition: bool, message: str = "Requirement failed") -> None:
        """Solidity's require() — revert if condition is false.

        This is the primary guard mechanism. Unlike assert() (which consumes
        all gas), require() returns remaining gas on failure.
        """
        if not condition:
            raise RevertError(message)

    def emit_event(self, name: str, args: dict[str, Any], gas_meter: GasMeter) -> None:
        """Emit an event. Wrapper around the event log."""
        event = Event(name=name, args=args, emitter=self.address)
        self.event_log.emit(event, gas_meter)


# =============================================================================
# Hello World Contract Implementation
# =============================================================================
# This is our "Solidity contract" implemented in Python.
# In Solidity it would look like:
#
# contract HelloWorld {
#     string public greeting;
#     address public owner;
#     uint256 public greetingCount;
#
#     event GreetingChanged(address indexed changer, string oldGreeting, string newGreeting);
#     event Funded(address indexed funder, uint256 amount);
#
#     constructor(string memory _greeting) {
#         greeting = _greeting;
#         owner = msg.sender;
#         greetingCount = 0;
#     }
#
#     function getGreeting() public view returns (string memory) { return greeting; }
#     function setGreeting(string memory _new) public { ... }
#     function transferOwnership(address newOwner) public { ... }
# }

class HelloWorldContract(SmartContract):
    """A complete Hello World smart contract with storage, events, and access control."""

    def __init__(self, address: str, deployer: str, initial_greeting: str,
                 gas_meter: GasMeter) -> None:
        super().__init__(address, deployer)

        # === CONSTRUCTOR ===
        # The constructor runs ONCE at deployment. It initializes storage and
        # is never callable again. This is fundamentally different from a regular
        # function — it's part of the deployment transaction, not the runtime code.
        print(f"\n[DEPLOY] Deploying HelloWorld at {address}")
        print(f"  Constructor called by {deployer}")

        # Initialize storage slots
        # In the real EVM, 'greeting' would be at slot 0, 'owner' at slot 1, etc.
        # Solidity assigns slots sequentially based on declaration order.
        self.storage.sstore("greeting", initial_greeting, gas_meter)
        self.storage.sstore("owner", deployer, gas_meter)
        self.storage.sstore("greetingCount", 0, gas_meter)

        # Register functions (build the dispatch table)
        self.functions.register(
            "getGreeting", "getGreeting()", self._get_greeting,
            mutates_state=False, gas_cost=100,
        )
        self.functions.register(
            "setGreeting", "setGreeting(string)", self._set_greeting,
            mutates_state=True, gas_cost=300,
        )
        self.functions.register(
            "getOwner", "getOwner()", self._get_owner,
            mutates_state=False, gas_cost=100,
        )
        self.functions.register(
            "transferOwnership", "transferOwnership(address)", self._transfer_ownership,
            mutates_state=True, gas_cost=200,
        )
        self.functions.register(
            "getGreetingCount", "getGreetingCount()", self._get_greeting_count,
            mutates_state=False, gas_cost=100,
        )

        self.emit_event("Deployed", {
            "owner": deployer,
            "greeting": initial_greeting,
        }, gas_meter)

        print(f"  Greeting initialized to: '{initial_greeting}'")
        print(f"  Owner set to: {deployer}")
        print(f"  Deployment gas used: {gas_meter.gas_used}")

    def _get_greeting(self, msg: MessageContext, gas_meter: GasMeter) -> str:
        """View function — reads storage but doesn't modify it.

        In Solidity, 'view' functions don't cost gas when called off-chain
        (eth_call), but DO cost gas when called from another contract on-chain.
        """
        return self.storage.sload("greeting", gas_meter)

    def _set_greeting(self, msg: MessageContext, gas_meter: GasMeter,
                      new_greeting: str = "") -> str:
        """State-mutating function with access control.

        Only the owner can change the greeting. This pattern (onlyOwner modifier)
        is the foundation of all contract access control. In production, you'd
        use OpenZeppelin's Ownable for battle-tested implementation.
        """
        # Access control check — the most important line in smart contract security
        owner = self.storage.sload("owner", gas_meter)
        self.require(msg.sender == owner, "Only owner can set greeting")

        old_greeting = self.storage.sload("greeting", gas_meter)
        self.storage.sstore("greeting", new_greeting, gas_meter)

        # Increment the greeting counter
        count = self.storage.sload("greetingCount", gas_meter)
        self.storage.sstore("greetingCount", count + 1, gas_meter)

        # Emit event for off-chain indexing
        self.emit_event("GreetingChanged", {
            "changer": msg.sender,
            "oldGreeting": old_greeting,
            "newGreeting": new_greeting,
        }, gas_meter)

        return new_greeting

    def _get_owner(self, msg: MessageContext, gas_meter: GasMeter) -> str:
        return self.storage.sload("owner", gas_meter)

    def _transfer_ownership(self, msg: MessageContext, gas_meter: GasMeter,
                            new_owner: str = "") -> str:
        """Transfer contract ownership. Two-step pattern is safer in production.

        In production, OpenZeppelin's Ownable2Step requires the new owner to
        explicitly accept ownership, preventing accidental transfers to wrong addresses.
        We implement the simpler single-step version here.
        """
        owner = self.storage.sload("owner", gas_meter)
        self.require(msg.sender == owner, "Only owner can transfer ownership")
        self.require(new_owner != "", "New owner cannot be zero address")

        self.storage.sstore("owner", new_owner, gas_meter)

        self.emit_event("OwnershipTransferred", {
            "previousOwner": owner,
            "newOwner": new_owner,
        }, gas_meter)

        return new_owner

    def _get_greeting_count(self, msg: MessageContext, gas_meter: GasMeter) -> int:
        return self.storage.sload("greetingCount", gas_meter)


# =============================================================================
# Blockchain VM (ties everything together)
# =============================================================================

class MiniEVM:
    """A minimal Ethereum Virtual Machine simulation.

    Manages contract deployment, transaction execution, and block state.
    In the real Ethereum, this role is split between the EVM (execution),
    the state trie (storage), and the transaction pool (ordering).
    """

    def __init__(self) -> None:
        self.contracts: dict[str, SmartContract] = {}
        self.nonces: dict[str, int] = {}  # Track nonces for address derivation
        self.block_number = 0
        self.transaction_log: list[dict[str, Any]] = []

    def _derive_address(self, deployer: str) -> str:
        """Derive a contract address from deployer + nonce.

        In Ethereum: address = keccak256(rlp([sender, nonce]))[12:]
        CREATE2 uses: address = keccak256(0xff ++ sender ++ salt ++ keccak256(bytecode))[12:]
        We simplify but preserve the concept: address is deterministic from inputs.
        """
        nonce = self.nonces.get(deployer, 0)
        self.nonces[deployer] = nonce + 1
        raw = f"{deployer}:{nonce}"
        addr = "0x" + hashlib.sha256(raw.encode()).hexdigest()[:40]
        return addr

    def deploy_contract(self, deployer: str, initial_greeting: str,
                        gas_limit: int = 300000) -> tuple[Optional[HelloWorldContract], dict]:
        """Deploy a new HelloWorld contract.

        Returns (contract_or_None, receipt).
        On failure, contract is None and receipt contains the error.
        """
        gas_meter = GasMeter(gas_limit)
        gas_meter.consume(GasMeter.BASE_TRANSACTION_GAS, "Base transaction cost")

        address = self._derive_address(deployer)
        snapshot = {}  # No prior state for new contract

        try:
            contract = HelloWorldContract(address, deployer, initial_greeting, gas_meter)
            self.contracts[address] = contract
            self.block_number += 1

            receipt = {
                "status": "success",
                "contract_address": address,
                "gas_used": gas_meter.gas_used,
                "gas_limit": gas_limit,
                "deployer": deployer,
                "block": self.block_number,
                "events": [
                    {"name": e.name, "args": e.args}
                    for e in contract.event_log.get_events()
                ],
            }
            self.transaction_log.append(receipt)
            return contract, receipt

        except OutOfGasError as e:
            receipt = {
                "status": "reverted (out of gas)",
                "error": str(e),
                "gas_used": gas_meter.gas_used,
                "gas_limit": gas_limit,
            }
            self.transaction_log.append(receipt)
            return None, receipt

    def call_function(self, contract_address: str, function_name: str,
                      sender: str, gas_limit: int = 100000,
                      **kwargs: Any) -> dict[str, Any]:
        """Execute a function on a deployed contract.

        This simulates sending a transaction to a contract:
        1. Create execution context (msg)
        2. Look up the function via dispatch
        3. Take a storage snapshot (for revert)
        4. Execute with gas metering
        5. On success: commit state changes
        6. On failure: rollback to snapshot
        """
        contract = self.contracts.get(contract_address)
        if contract is None:
            return {"status": "error", "error": f"No contract at {contract_address}"}

        gas_meter = GasMeter(gas_limit)
        gas_meter.consume(GasMeter.BASE_TRANSACTION_GAS, "Base transaction cost")

        msg = MessageContext(sender=sender)

        # Resolve the function
        fn_abi = contract.functions.resolve_by_name(function_name)
        if fn_abi is None:
            return {"status": "error", "error": f"Unknown function: {function_name}"}

        # Charge function dispatch gas (simulates JUMPDEST lookup)
        gas_meter.consume(fn_abi.gas_cost, f"Function dispatch: {fn_abi.signature}")

        # Snapshot state for potential rollback
        storage_snapshot = contract.storage.snapshot()
        events_before = len(contract.event_log.events)

        try:
            result = fn_abi.handler(msg, gas_meter, **kwargs)
            self.block_number += 1

            new_events = contract.event_log.events[events_before:]
            receipt = {
                "status": "success",
                "function": fn_abi.signature,
                "result": result,
                "gas_used": gas_meter.gas_used,
                "gas_limit": gas_limit,
                "sender": sender,
                "block": self.block_number,
                "events": [{"name": e.name, "args": e.args} for e in new_events],
                "gas_log": gas_meter.gas_log,
            }
            self.transaction_log.append(receipt)
            return receipt

        except RevertError as e:
            # Revert: undo all state changes but keep gas accounting
            contract.storage.restore(storage_snapshot)
            # Remove any events emitted during the reverted call
            contract.event_log.events = contract.event_log.events[:events_before]

            receipt = {
                "status": "reverted",
                "function": fn_abi.signature,
                "error": str(e),
                "gas_used": gas_meter.gas_used,
                "gas_limit": gas_limit,
                "sender": sender,
            }
            self.transaction_log.append(receipt)
            return receipt

        except OutOfGasError as e:
            # Out of gas: revert AND consume all gas
            contract.storage.restore(storage_snapshot)
            contract.event_log.events = contract.event_log.events[:events_before]

            receipt = {
                "status": "reverted (out of gas)",
                "function": fn_abi.signature,
                "error": str(e),
                "gas_used": gas_limit,  # All gas consumed on OOG
                "gas_limit": gas_limit,
                "sender": sender,
            }
            self.transaction_log.append(receipt)
            return receipt


# =============================================================================
# Main: Full lifecycle demonstration
# =============================================================================

def print_receipt(label: str, receipt: dict) -> None:
    """Pretty-print a transaction receipt."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for key, value in receipt.items():
        if key == "gas_log":
            continue  # Skip verbose gas log in summary
        if key == "events" and value:
            print(f"  {key}:")
            for evt in value:
                print(f"    - {evt['name']}: {evt['args']}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("=" * 60)
    print("  SMART CONTRACT VM SIMULATION")
    print("  Building an Ethereum-like execution environment")
    print("=" * 60)

    # Initialize our mini blockchain
    evm = MiniEVM()

    # === STEP 1: Deploy the contract ===
    alice = "0xAlice_deployer"
    contract, deploy_receipt = evm.deploy_contract(
        deployer=alice,
        initial_greeting="Hello, Blockchain!",
        gas_limit=300000,
    )
    print_receipt("DEPLOYMENT RECEIPT", deploy_receipt)
    assert contract is not None, "Deployment failed!"
    contract_addr = deploy_receipt["contract_address"]

    # === STEP 2: Read the greeting (view function) ===
    result = evm.call_function(contract_addr, "getGreeting", sender=alice)
    print_receipt("CALL: getGreeting()", result)
    print(f"\n  >> Current greeting: '{result['result']}'")
    assert result["result"] == "Hello, Blockchain!"

    # === STEP 3: Update the greeting (state mutation) ===
    result = evm.call_function(
        contract_addr, "setGreeting", sender=alice,
        new_greeting="Hello from Python!",
    )
    print_receipt("CALL: setGreeting('Hello from Python!')", result)
    assert result["status"] == "success"

    # Verify the update
    result = evm.call_function(contract_addr, "getGreeting", sender=alice)
    print(f"\n  >> Updated greeting: '{result['result']}'")
    assert result["result"] == "Hello from Python!"

    # === STEP 4: Unauthorized access (should revert) ===
    bob = "0xBob_attacker"
    print("\n\n--- Testing access control: Bob tries to change the greeting ---")
    result = evm.call_function(
        contract_addr, "setGreeting", sender=bob,
        new_greeting="Hacked!",
    )
    print_receipt("CALL: setGreeting('Hacked!') from Bob", result)
    assert result["status"] == "reverted"
    print("  >> Access control working: Bob's transaction reverted!")

    # Verify greeting unchanged
    result = evm.call_function(contract_addr, "getGreeting", sender=alice)
    assert result["result"] == "Hello from Python!"
    print(f"  >> Greeting still: '{result['result']}' (unchanged)")

    # === STEP 5: Transfer ownership ===
    print("\n\n--- Testing ownership transfer ---")
    result = evm.call_function(
        contract_addr, "transferOwnership", sender=alice,
        new_owner=bob,
    )
    print_receipt("CALL: transferOwnership(Bob)", result)
    assert result["status"] == "success"

    # Now Bob CAN change the greeting
    result = evm.call_function(
        contract_addr, "setGreeting", sender=bob,
        new_greeting="Bob's greeting!",
    )
    print_receipt("CALL: setGreeting('Bob's greeting!') from new owner Bob", result)
    assert result["status"] == "success"

    result = evm.call_function(contract_addr, "getGreeting", sender=bob)
    print(f"\n  >> Bob's greeting: '{result['result']}'")
    assert result["result"] == "Bob's greeting!"

    # Alice can no longer change it
    result = evm.call_function(
        contract_addr, "setGreeting", sender=alice,
        new_greeting="Alice tries again",
    )
    assert result["status"] == "reverted"
    print("  >> Alice can no longer modify (ownership transferred)")

    # === STEP 6: Out of gas simulation ===
    print("\n\n--- Testing out-of-gas scenario ---")
    result = evm.call_function(
        contract_addr, "setGreeting", sender=bob,
        new_greeting="This will fail",
        gas_limit=21500,  # Barely enough for base cost, not enough for execution
    )
    print_receipt("CALL: setGreeting() with insufficient gas", result)
    assert "out of gas" in result["status"]
    print("  >> Transaction reverted due to insufficient gas!")

    # === STEP 7: Check greeting count ===
    result = evm.call_function(contract_addr, "getGreetingCount", sender=alice)
    print(f"\n  >> Total greeting changes: {result['result']}")
    assert result["result"] == 2  # Alice's change + Bob's change

    # === STEP 8: Review event log ===
    print("\n\n" + "=" * 60)
    print("  EVENT LOG (all events emitted by the contract)")
    print("=" * 60)
    for i, event in enumerate(contract.event_log.get_events()):
        print(f"  [{i}] {event.name}: {event.args}")

    # === STEP 9: Inspect storage ===
    print("\n\n" + "=" * 60)
    print("  FINAL STORAGE STATE")
    print("=" * 60)
    for slot, value in contract.storage.dump().items():
        print(f"  slot '{slot}': {value}")

    # === STEP 10: Function selector table ===
    print("\n\n" + "=" * 60)
    print("  FUNCTION DISPATCH TABLE")
    print("=" * 60)
    for fn in contract.functions.list_functions():
        mutability = "view" if not fn.mutates_state else "nonpayable"
        print(f"  {fn.selector} -> {fn.signature} [{mutability}]")

    # === Summary ===
    print("\n\n" + "=" * 60)
    print("  BLOCKCHAIN SUMMARY")
    print("=" * 60)
    print(f"  Total blocks: {evm.block_number}")
    print(f"  Total transactions: {len(evm.transaction_log)}")
    print(f"  Deployed contracts: {len(evm.contracts)}")
    successful = sum(1 for tx in evm.transaction_log if tx["status"] == "success")
    reverted = len(evm.transaction_log) - successful
    print(f"  Successful: {successful}, Reverted: {reverted}")
    print(f"\n  All assertions passed!")
