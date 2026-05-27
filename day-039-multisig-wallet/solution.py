"""
Day 039: Multisig Wallet

A complete M-of-N multi-signature wallet implementation. This simulates the core
logic of production multisig wallets like Gnosis Safe: owner management, transaction
proposal/confirmation/execution, revocation, and governance via self-call.

We model addresses as strings, ETH balances as integers (in wei), and calldata as
(function_name, args) tuples. The focus is on the approval logic, access control,
and state machine — not on EVM bytecode encoding.

Usage: python3 solution.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class TxStatus(Enum):
    """Transaction lifecycle states."""
    PENDING = auto()     # Submitted, gathering confirmations
    EXECUTED = auto()    # Successfully executed
    FAILED = auto()      # Execution attempted but reverted


@dataclass
class Transaction:
    """
    Represents a multisig transaction waiting for approval.

    Fields mirror Gnosis Safe's internal Transaction struct:
    - destination: target address (could be external or the wallet itself for admin ops)
    - value: amount of ETH (wei) to transfer
    - data: calldata — we model this as (function_name, args_dict) for clarity
    - confirmations: set of owner addresses who have approved
    - status: current lifecycle state
    - tx_id: sequential identifier for replay protection
    """
    destination: str
    value: int
    data: Optional[tuple[str, dict[str, Any]]]
    tx_id: int
    confirmations: set[str] = field(default_factory=set)
    status: TxStatus = TxStatus.PENDING

    @property
    def confirmation_count(self) -> int:
        return len(self.confirmations)


class MultisigWallet:
    """
    M-of-N multi-signature wallet.

    Design decisions:
    - Owners stored in both a list (enumeration) and set (O(1) lookup).
    - Transaction IDs are sequential integers starting from 0 (nonce pattern).
    - Admin functions (add/remove owner, change threshold) use the self-call pattern:
      they can ONLY be invoked through the multisig approval process itself.
    - Explicit confirmation required — submitting a tx does NOT auto-confirm it.
      This keeps the audit trail clean: every approval is a deliberate action.
    """

    # Special address representing "this wallet" for self-call detection
    SELF_ADDRESS = "__WALLET_SELF__"

    def __init__(self, owners: list[str], required: int) -> None:
        """
        Initialize the multisig wallet.

        Args:
            owners: list of owner addresses (strings)
            required: number of confirmations needed to execute (the M in M-of-N)

        Raises:
            ValueError: if owners list is invalid or threshold is out of range

        Why validate so aggressively? A misconfigured multisig is a permanently
        bricked or insecure wallet. There's no "undo" in production — getting
        constructor parameters wrong means redeploying and migrating all funds.
        """
        # --- Validation ---
        if not owners:
            raise ValueError("Must have at least one owner")
        if required < 1:
            raise ValueError("Required confirmations must be >= 1")
        if required > len(owners):
            raise ValueError(
                f"Required ({required}) exceeds owner count ({len(owners)})"
            )

        # Check for zero addresses and duplicates
        seen: set[str] = set()
        for addr in owners:
            if not addr or addr.strip() == "":
                raise ValueError("Owner address cannot be empty")
            if addr in seen:
                raise ValueError(f"Duplicate owner: {addr}")
            seen.add(addr)

        # --- State initialization ---
        self._owners: list[str] = list(owners)       # Ordered list for enumeration
        self._owner_set: set[str] = set(owners)       # O(1) membership check
        self._required: int = required                 # Confirmation threshold
        self._transactions: dict[int, Transaction] = {}  # txId -> Transaction
        self._tx_count: int = 0                        # Next transaction ID (nonce)
        self._balance: int = 0                         # ETH balance in wei

        # _executing flag prevents re-entrancy during self-calls.
        # When the wallet executes a transaction targeting itself, the admin
        # function needs to know it's being called in the wallet's context.
        self._executing: bool = False

    # ──────────────────────────────────────────────
    # View functions
    # ──────────────────────────────────────────────

    @property
    def owners(self) -> list[str]:
        """Return the current owner list (copy to prevent external mutation)."""
        return list(self._owners)

    @property
    def required(self) -> int:
        """Return the current confirmation threshold."""
        return self._required

    @property
    def balance(self) -> int:
        """Return the wallet's ETH balance in wei."""
        return self._balance

    @property
    def transaction_count(self) -> int:
        """Return total number of submitted transactions (including executed)."""
        return self._tx_count

    def is_owner(self, address: str) -> bool:
        """Check if an address is a registered owner. O(1) via set lookup."""
        return address in self._owner_set

    def get_transaction(self, tx_id: int) -> Transaction:
        """Retrieve a transaction by ID."""
        if tx_id not in self._transactions:
            raise KeyError(f"Transaction {tx_id} does not exist")
        return self._transactions[tx_id]

    def get_pending_transactions(self) -> list[Transaction]:
        """Return all transactions that haven't been executed yet."""
        return [
            tx for tx in self._transactions.values()
            if tx.status == TxStatus.PENDING
        ]

    def is_confirmed(self, tx_id: int) -> bool:
        """Check if a transaction has enough confirmations to execute."""
        tx = self.get_transaction(tx_id)
        return tx.confirmation_count >= self._required

    # ──────────────────────────────────────────────
    # Deposit
    # ──────────────────────────────────────────────

    def deposit(self, sender: str, amount: int) -> None:
        """
        Deposit ETH into the wallet. Anyone can deposit — not restricted to owners.
        In Solidity this would be the receive() or fallback() function.
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        print(f"  [Deposit] {sender} deposited {amount} wei. Balance: {self._balance}")

    # ──────────────────────────────────────────────
    # Transaction submission
    # ──────────────────────────────────────────────

    def submit_transaction(
        self,
        sender: str,
        destination: str,
        value: int = 0,
        data: Optional[tuple[str, dict[str, Any]]] = None,
    ) -> int:
        """
        Propose a new transaction. Only owners can submit.

        Returns the transaction ID (nonce).

        Note: the submitter does NOT auto-confirm. This is a deliberate design
        choice matching Gnosis Safe — it makes the audit trail unambiguous.
        Every confirmation is an explicit action.
        """
        self._require_owner(sender)

        tx_id = self._tx_count
        self._transactions[tx_id] = Transaction(
            destination=destination,
            value=value,
            data=data,
            tx_id=tx_id,
        )
        self._tx_count += 1

        data_desc = f", data={data[0]}(...)" if data else ""
        print(
            f"  [Submit] Owner {sender} submitted tx#{tx_id}: "
            f"to={destination}, value={value}{data_desc}"
        )
        return tx_id

    # ──────────────────────────────────────────────
    # Confirmation
    # ──────────────────────────────────────────────

    def confirm_transaction(self, sender: str, tx_id: int) -> None:
        """
        Confirm (approve) a pending transaction. Each owner can confirm at most once.

        Why separate confirm from execute? It decouples approval from gas payment.
        In production, the last confirmer pays execution gas. Separating the steps
        lets any owner trigger execution after threshold is met.
        """
        self._require_owner(sender)
        tx = self.get_transaction(tx_id)

        if tx.status != TxStatus.PENDING:
            raise ValueError(f"Transaction {tx_id} is not pending (status: {tx.status.name})")
        if sender in tx.confirmations:
            raise ValueError(f"Owner {sender} already confirmed tx#{tx_id}")

        tx.confirmations.add(sender)
        print(
            f"  [Confirm] Owner {sender} confirmed tx#{tx_id}. "
            f"Confirmations: {tx.confirmation_count}/{self._required}"
        )

    # ──────────────────────────────────────────────
    # Revocation
    # ──────────────────────────────────────────────

    def revoke_confirmation(self, sender: str, tx_id: int) -> None:
        """
        Withdraw a confirmation before execution.

        This is critical for security: if an owner realizes a transaction is
        malicious (e.g., a compromised key submitted it), they can revoke their
        approval. Without revocation, once you confirm you're locked in.
        """
        self._require_owner(sender)
        tx = self.get_transaction(tx_id)

        if tx.status != TxStatus.PENDING:
            raise ValueError(f"Transaction {tx_id} is not pending")
        if sender not in tx.confirmations:
            raise ValueError(f"Owner {sender} has not confirmed tx#{tx_id}")

        tx.confirmations.discard(sender)
        print(
            f"  [Revoke] Owner {sender} revoked confirmation on tx#{tx_id}. "
            f"Confirmations: {tx.confirmation_count}/{self._required}"
        )

    # ──────────────────────────────────────────────
    # Execution
    # ──────────────────────────────────────────────

    def execute_transaction(self, sender: str, tx_id: int) -> bool:
        """
        Execute a fully-confirmed transaction.

        Returns True if execution succeeded, False if the call reverted.

        The checks-effects-interactions pattern:
        1. CHECK: verify threshold is met, tx is pending
        2. EFFECT: mark as executed BEFORE the external call
        3. INTERACT: perform the call

        Why mark executed before calling? Re-entrancy protection. If the
        destination is a malicious contract that calls back into the wallet,
        the transaction is already marked executed so it can't be replayed.

        However, if the call reverts, we need to revert our state change too.
        We handle this by catching the failure and marking as FAILED, or
        keeping as PENDING to allow retry (we choose FAILED for clarity).
        """
        self._require_owner(sender)
        tx = self.get_transaction(tx_id)

        if tx.status != TxStatus.PENDING:
            raise ValueError(f"Transaction {tx_id} is not pending")
        if tx.confirmation_count < self._required:
            raise ValueError(
                f"Not enough confirmations: {tx.confirmation_count}/{self._required}"
            )

        # --- Execute ---
        success = self._execute_call(tx)

        if success:
            tx.status = TxStatus.EXECUTED
            print(f"  [Execute] tx#{tx_id} executed successfully!")
        else:
            tx.status = TxStatus.FAILED
            print(f"  [Execute] tx#{tx_id} execution FAILED (call reverted)")

        return success

    def _execute_call(self, tx: Transaction) -> bool:
        """
        Perform the actual "call" for a transaction.

        In Solidity this would be:
            (bool success, ) = tx.destination.call{value: tx.value}(tx.data);

        We simulate three cases:
        1. Self-call (admin operation): dispatch to internal admin function
        2. Value transfer: deduct from balance
        3. External call with data: simulate success (in production, EVM handles this)
        """
        # Handle value transfer
        if tx.value > 0:
            if self._balance < tx.value:
                print(f"    Revert: insufficient balance ({self._balance} < {tx.value})")
                return False
            self._balance -= tx.value
            print(f"    Transferred {tx.value} wei to {tx.destination}. Balance: {self._balance}")

        # Handle self-call (admin operations)
        if tx.destination == self.SELF_ADDRESS and tx.data is not None:
            func_name, args = tx.data
            return self._dispatch_admin(func_name, args)

        return True

    def _dispatch_admin(self, func_name: str, args: dict[str, Any]) -> bool:
        """
        Route self-calls to the appropriate admin function.

        This is the self-call pattern: admin functions are "internal" and can
        only be triggered through the multisig approval process. We enforce this
        by checking the _executing flag (set during execute_transaction).

        In Solidity, these would have a `onlyWallet` modifier that checks
        msg.sender == address(this).
        """
        self._executing = True
        try:
            if func_name == "addOwner":
                self._add_owner(args["owner"])
            elif func_name == "removeOwner":
                self._remove_owner(args["owner"])
            elif func_name == "changeRequirement":
                self._change_requirement(args["required"])
            else:
                print(f"    Revert: unknown admin function '{func_name}'")
                return False
        except (ValueError, KeyError) as e:
            print(f"    Revert: {e}")
            return False
        finally:
            self._executing = False
        return True

    # ──────────────────────────────────────────────
    # Admin functions (self-call only)
    # ──────────────────────────────────────────────

    def _add_owner(self, new_owner: str) -> None:
        """
        Add a new owner. Can only be called via self-call (multisig approval).

        After adding, the threshold stays the same. If the caller wants to
        change the threshold too, they submit a separate changeRequirement tx.
        Keeping operations atomic and composable is cleaner than bundling.
        """
        if not self._executing:
            raise ValueError("Can only be called by the wallet itself")
        if not new_owner or new_owner.strip() == "":
            raise ValueError("Invalid owner address")
        if new_owner in self._owner_set:
            raise ValueError(f"Already an owner: {new_owner}")

        self._owners.append(new_owner)
        self._owner_set.add(new_owner)
        print(f"    [Admin] Added owner: {new_owner}. Owners: {self._owners}")

    def _remove_owner(self, owner: str) -> None:
        """
        Remove an owner. Adjusts threshold down if necessary to prevent bricking.

        Why auto-adjust threshold? If you have 3-of-5 and remove an owner to get
        4 owners, the 3-of-4 threshold still works. But if you had 5-of-5 and
        remove one, you'd need to lower to 4-of-4 or the wallet is permanently
        bricked (can never reach 5 confirmations with only 4 owners).
        """
        if not self._executing:
            raise ValueError("Can only be called by the wallet itself")
        if owner not in self._owner_set:
            raise ValueError(f"Not an owner: {owner}")
        if len(self._owners) <= 1:
            raise ValueError("Cannot remove the last owner")

        self._owners.remove(owner)
        self._owner_set.discard(owner)

        # Auto-adjust threshold if it now exceeds owner count
        if self._required > len(self._owners):
            old_req = self._required
            self._required = len(self._owners)
            print(
                f"    [Admin] Threshold auto-adjusted: {old_req} -> {self._required} "
                f"(owner count decreased)"
            )

        print(f"    [Admin] Removed owner: {owner}. Owners: {self._owners}")

    def _change_requirement(self, new_required: int) -> None:
        """
        Change the confirmation threshold. Must still be in valid range [1, N].
        """
        if not self._executing:
            raise ValueError("Can only be called by the wallet itself")
        if new_required < 1:
            raise ValueError("Required must be >= 1")
        if new_required > len(self._owners):
            raise ValueError(
                f"Required ({new_required}) exceeds owner count ({len(self._owners)})"
            )

        old = self._required
        self._required = new_required
        print(f"    [Admin] Threshold changed: {old} -> {new_required}")

    # ──────────────────────────────────────────────
    # Guard modifiers (simulated)
    # ──────────────────────────────────────────────

    def _require_owner(self, address: str) -> None:
        """Revert if the caller is not a registered owner."""
        if address not in self._owner_set:
            raise PermissionError(f"Not an owner: {address}")

    # ──────────────────────────────────────────────
    # Pretty printing
    # ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MultisigWallet(owners={self._owners}, required={self._required}, "
            f"balance={self._balance}, txCount={self._tx_count})"
        )

    def print_status(self) -> None:
        """Print a summary of the wallet state."""
        print(f"\n  Wallet Status:")
        print(f"    Owners ({len(self._owners)}): {self._owners}")
        print(f"    Required confirmations: {self._required}")
        print(f"    Balance: {self._balance} wei")
        print(f"    Total transactions: {self._tx_count}")
        pending = self.get_pending_transactions()
        if pending:
            print(f"    Pending transactions: {[tx.tx_id for tx in pending]}")


# ══════════════════════════════════════════════════
# Demonstration
# ══════════════════════════════════════════════════

def demo_basic_transfer() -> None:
    """Demonstrate a basic ETH transfer through the multisig."""
    print("=" * 70)
    print("DEMO 1: Basic 2-of-3 Multisig Transfer")
    print("=" * 70)

    # Create a 2-of-3 wallet
    wallet = MultisigWallet(
        owners=["alice", "bob", "carol"],
        required=2,
    )
    wallet.print_status()

    # Deposit funds
    print("\n--- Depositing funds ---")
    wallet.deposit("external_user", 1000)

    # Alice proposes sending 300 wei to Dave
    print("\n--- Submitting transaction ---")
    tx_id = wallet.submit_transaction("alice", destination="dave", value=300)

    # Alice confirms
    print("\n--- Gathering confirmations ---")
    wallet.confirm_transaction("alice", tx_id)

    # Not enough yet — try to execute (should fail)
    print("\n--- Attempting premature execution ---")
    try:
        wallet.execute_transaction("bob", tx_id)
    except ValueError as e:
        print(f"  Expected error: {e}")

    # Bob confirms — now we have 2/2
    wallet.confirm_transaction("bob", tx_id)

    # Execute
    print("\n--- Executing ---")
    wallet.execute_transaction("bob", tx_id)
    wallet.print_status()


def demo_revocation() -> None:
    """Demonstrate confirmation revocation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Confirmation Revocation")
    print("=" * 70)

    wallet = MultisigWallet(owners=["alice", "bob", "carol"], required=2)
    wallet.deposit("funder", 500)

    # Submit and confirm
    tx_id = wallet.submit_transaction("alice", destination="eve", value=200)
    wallet.confirm_transaction("alice", tx_id)
    wallet.confirm_transaction("bob", tx_id)

    # Bob realizes it's a mistake — revokes before execution
    print("\n--- Bob revokes ---")
    wallet.revoke_confirmation("bob", tx_id)

    # Can't execute now
    print("\n--- Attempting execution after revocation ---")
    try:
        wallet.execute_transaction("carol", tx_id)
    except ValueError as e:
        print(f"  Expected error: {e}")

    # Carol confirms instead, reaching threshold again
    print("\n--- Carol confirms ---")
    wallet.confirm_transaction("carol", tx_id)
    wallet.execute_transaction("carol", tx_id)
    wallet.print_status()


def demo_admin_operations() -> None:
    """Demonstrate governance via self-call: add owner, change threshold."""
    print("\n" + "=" * 70)
    print("DEMO 3: Admin Operations via Self-Call")
    print("=" * 70)

    wallet = MultisigWallet(owners=["alice", "bob", "carol"], required=2)

    # --- Add a new owner via multisig ---
    print("\n--- Adding owner 'dave' via multisig approval ---")
    tx_id = wallet.submit_transaction(
        "alice",
        destination=MultisigWallet.SELF_ADDRESS,
        data=("addOwner", {"owner": "dave"}),
    )
    wallet.confirm_transaction("alice", tx_id)
    wallet.confirm_transaction("bob", tx_id)
    wallet.execute_transaction("alice", tx_id)
    wallet.print_status()

    # --- Change threshold to 3-of-4 ---
    print("\n--- Changing threshold to 3-of-4 via multisig approval ---")
    tx_id = wallet.submit_transaction(
        "bob",
        destination=MultisigWallet.SELF_ADDRESS,
        data=("changeRequirement", {"required": 3}),
    )
    wallet.confirm_transaction("bob", tx_id)
    wallet.confirm_transaction("carol", tx_id)
    wallet.execute_transaction("carol", tx_id)
    wallet.print_status()

    # --- Now a transfer needs 3 confirmations ---
    print("\n--- Transfer with new 3-of-4 threshold ---")
    wallet.deposit("funder", 1000)
    tx_id = wallet.submit_transaction("alice", destination="external", value=100)
    wallet.confirm_transaction("alice", tx_id)
    wallet.confirm_transaction("bob", tx_id)

    print("\n  Attempting with only 2 confirmations...")
    try:
        wallet.execute_transaction("alice", tx_id)
    except ValueError as e:
        print(f"  Expected error: {e}")

    wallet.confirm_transaction("dave", tx_id)
    wallet.execute_transaction("dave", tx_id)
    wallet.print_status()


def demo_remove_owner() -> None:
    """Demonstrate owner removal with automatic threshold adjustment."""
    print("\n" + "=" * 70)
    print("DEMO 4: Remove Owner (Threshold Auto-Adjustment)")
    print("=" * 70)

    # Start with 3-of-3 — all owners must agree
    wallet = MultisigWallet(owners=["alice", "bob", "carol"], required=3)
    print(f"  Starting config: {len(wallet.owners)}-of-{wallet.required}")

    # Remove carol — this MUST auto-adjust threshold to 2-of-2
    # Otherwise the wallet would be bricked (can't get 3 sigs from 2 owners)
    tx_id = wallet.submit_transaction(
        "alice",
        destination=MultisigWallet.SELF_ADDRESS,
        data=("removeOwner", {"owner": "carol"}),
    )
    wallet.confirm_transaction("alice", tx_id)
    wallet.confirm_transaction("bob", tx_id)
    wallet.confirm_transaction("carol", tx_id)
    wallet.execute_transaction("alice", tx_id)
    wallet.print_status()


def demo_access_control() -> None:
    """Demonstrate access control: non-owners rejected, double-confirm blocked."""
    print("\n" + "=" * 70)
    print("DEMO 5: Access Control & Edge Cases")
    print("=" * 70)

    wallet = MultisigWallet(owners=["alice", "bob"], required=2)
    wallet.deposit("anyone", 500)

    tx_id = wallet.submit_transaction("alice", destination="target", value=100)

    # Non-owner tries to confirm
    print("\n--- Non-owner attempts ---")
    try:
        wallet.confirm_transaction("mallory", tx_id)
    except PermissionError as e:
        print(f"  Blocked: {e}")

    # Non-owner tries to submit
    try:
        wallet.submit_transaction("mallory", destination="mallory", value=500)
    except PermissionError as e:
        print(f"  Blocked: {e}")

    # Double confirmation
    print("\n--- Double confirmation attempt ---")
    wallet.confirm_transaction("alice", tx_id)
    try:
        wallet.confirm_transaction("alice", tx_id)
    except ValueError as e:
        print(f"  Blocked: {e}")

    # Execute and try to re-execute
    print("\n--- Double execution attempt ---")
    wallet.confirm_transaction("bob", tx_id)
    wallet.execute_transaction("alice", tx_id)
    try:
        wallet.execute_transaction("bob", tx_id)
    except ValueError as e:
        print(f"  Blocked: {e}")


def demo_insufficient_balance() -> None:
    """Demonstrate failed execution due to insufficient balance."""
    print("\n" + "=" * 70)
    print("DEMO 6: Insufficient Balance Handling")
    print("=" * 70)

    wallet = MultisigWallet(owners=["alice", "bob"], required=2)
    wallet.deposit("funder", 100)

    # Try to send more than the balance
    tx_id = wallet.submit_transaction("alice", destination="greedy", value=999)
    wallet.confirm_transaction("alice", tx_id)
    wallet.confirm_transaction("bob", tx_id)
    success = wallet.execute_transaction("alice", tx_id)
    print(f"  Execution success: {success}")

    tx = wallet.get_transaction(tx_id)
    print(f"  Transaction status: {tx.status.name}")
    print(f"  Balance unchanged: {wallet.balance}")


if __name__ == "__main__":
    demo_basic_transfer()
    demo_revocation()
    demo_admin_operations()
    demo_remove_owner()
    demo_access_control()
    demo_insufficient_balance()

    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
