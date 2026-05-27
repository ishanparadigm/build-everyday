"""
Day 039: Multisig Wallet — Your Implementation

Build an M-of-N multi-signature wallet from scratch.

Key concepts to implement:
- Owner management with O(1) lookup
- Transaction lifecycle: submit → confirm → execute
- Revocation of confirmations before execution
- Self-call pattern for admin operations (add/remove owner, change threshold)
- Access control guards on every public function

Run tests with: python3 -m pytest tests.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class TxStatus(Enum):
    """Transaction lifecycle states."""
    PENDING = auto()
    EXECUTED = auto()
    FAILED = auto()


@dataclass
class Transaction:
    """
    Represents a multisig transaction waiting for approval.

    Fields:
    - destination: target address
    - value: amount of ETH (wei) to transfer
    - data: calldata as (function_name, args_dict) or None
    - tx_id: sequential identifier
    - confirmations: set of owner addresses who approved
    - status: PENDING, EXECUTED, or FAILED
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

    Hint: You'll need to track owners in two data structures — think about
    why a list alone isn't sufficient, and what operations each supports.
    """

    SELF_ADDRESS = "__WALLET_SELF__"

    def __init__(self, owners: list[str], required: int) -> None:
        """
        Initialize the multisig wallet.

        Args:
            owners: list of owner addresses
            required: confirmation threshold (M in M-of-N)

        Validate:
        - At least one owner
        - No empty addresses or duplicates
        - required in range [1, len(owners)]

        Hint: What happens if you skip duplicate checking? Two "votes" from
        one address could count as two confirmations.
        """
        raise NotImplementedError("TODO: implement this")

    # ── View functions ────────────────────────────

    @property
    def owners(self) -> list[str]:
        """Return the current owner list (return a copy to prevent mutation)."""
        raise NotImplementedError("TODO: implement this")

    @property
    def required(self) -> int:
        """Return the current confirmation threshold."""
        raise NotImplementedError("TODO: implement this")

    @property
    def balance(self) -> int:
        """Return the wallet's ETH balance in wei."""
        raise NotImplementedError("TODO: implement this")

    @property
    def transaction_count(self) -> int:
        """Return total number of submitted transactions."""
        raise NotImplementedError("TODO: implement this")

    def is_owner(self, address: str) -> bool:
        """Check if an address is a registered owner."""
        raise NotImplementedError("TODO: implement this")

    def get_transaction(self, tx_id: int) -> Transaction:
        """Retrieve a transaction by ID. Raise KeyError if not found."""
        raise NotImplementedError("TODO: implement this")

    def get_pending_transactions(self) -> list[Transaction]:
        """Return all transactions with PENDING status."""
        raise NotImplementedError("TODO: implement this")

    def is_confirmed(self, tx_id: int) -> bool:
        """Check if a transaction has met the confirmation threshold."""
        raise NotImplementedError("TODO: implement this")

    # ── Deposit ───────────────────────────────────

    def deposit(self, sender: str, amount: int) -> None:
        """
        Deposit ETH into the wallet. Anyone can deposit.
        Raise ValueError if amount <= 0.
        """
        raise NotImplementedError("TODO: implement this")

    # ── Transaction submission ────────────────────

    def submit_transaction(
        self,
        sender: str,
        destination: str,
        value: int = 0,
        data: Optional[tuple[str, dict[str, Any]]] = None,
    ) -> int:
        """
        Propose a new transaction. Only owners can submit.
        Returns the transaction ID.

        Hint: Do NOT auto-confirm for the submitter. Every confirmation
        should be an explicit action for a clean audit trail.
        """
        raise NotImplementedError("TODO: implement this")

    # ── Confirmation ──────────────────────────────

    def confirm_transaction(self, sender: str, tx_id: int) -> None:
        """
        Confirm a pending transaction. Each owner confirms at most once.

        Check:
        - sender is an owner
        - tx is PENDING
        - sender hasn't already confirmed

        Hint: Use a set for confirmations — O(1) add and membership check.
        """
        raise NotImplementedError("TODO: implement this")

    # ── Revocation ────────────────────────────────

    def revoke_confirmation(self, sender: str, tx_id: int) -> None:
        """
        Withdraw a confirmation before execution.

        Check:
        - sender is an owner
        - tx is PENDING
        - sender has actually confirmed this tx
        """
        raise NotImplementedError("TODO: implement this")

    # ── Execution ─────────────────────────────────

    def execute_transaction(self, sender: str, tx_id: int) -> bool:
        """
        Execute a fully-confirmed transaction. Returns True on success.

        Check:
        - sender is an owner
        - tx is PENDING
        - confirmation_count >= required

        Hint: Think about the checks-effects-interactions pattern.
        What if destination is a malicious contract that calls back?
        """
        raise NotImplementedError("TODO: implement this")

    def _execute_call(self, tx: Transaction) -> bool:
        """
        Perform the actual call. Handle three cases:
        1. Value transfer (deduct balance, fail if insufficient)
        2. Self-call (dispatch to admin function)
        3. External call with data (simulate success)
        """
        raise NotImplementedError("TODO: implement this")

    def _dispatch_admin(self, func_name: str, args: dict[str, Any]) -> bool:
        """
        Route self-calls to admin functions: addOwner, removeOwner, changeRequirement.

        Hint: Set a flag before calling admin functions so they can verify
        they're being called through the multisig process, not directly.
        """
        raise NotImplementedError("TODO: implement this")

    # ── Admin functions (self-call only) ──────────

    def _add_owner(self, new_owner: str) -> None:
        """
        Add a new owner. Only callable via self-call.
        Validate: not empty, not already an owner.
        """
        raise NotImplementedError("TODO: implement this")

    def _remove_owner(self, owner: str) -> None:
        """
        Remove an owner. Only callable via self-call.

        Hint: What happens to the threshold if you remove an owner and
        now required > len(owners)? The wallet would be permanently bricked.
        Auto-adjust threshold downward to prevent this.
        """
        raise NotImplementedError("TODO: implement this")

    def _change_requirement(self, new_required: int) -> None:
        """
        Change the confirmation threshold. Only callable via self-call.
        Validate: new_required in [1, len(owners)].
        """
        raise NotImplementedError("TODO: implement this")

    # ── Guards ────────────────────────────────────

    def _require_owner(self, address: str) -> None:
        """Raise PermissionError if address is not an owner."""
        raise NotImplementedError("TODO: implement this")

    def print_status(self) -> None:
        """Print wallet state summary."""
        print(f"\n  Wallet Status:")
        print(f"    Owners ({len(self.owners)}): {self.owners}")
        print(f"    Required confirmations: {self.required}")
        print(f"    Balance: {self.balance} wei")
        print(f"    Total transactions: {self.transaction_count}")


if __name__ == "__main__":
    # Test your implementation step by step

    print("=== Step 1: Create wallet ===")
    wallet = MultisigWallet(owners=["alice", "bob", "carol"], required=2)
    wallet.print_status()

    print("\n=== Step 2: Deposit ===")
    wallet.deposit("funder", 1000)

    print("\n=== Step 3: Submit transaction ===")
    tx_id = wallet.submit_transaction("alice", destination="dave", value=300)

    print("\n=== Step 4: Confirm ===")
    wallet.confirm_transaction("alice", tx_id)
    wallet.confirm_transaction("bob", tx_id)

    print("\n=== Step 5: Execute ===")
    wallet.execute_transaction("bob", tx_id)
    wallet.print_status()

    print("\n=== Step 6: Admin via self-call ===")
    tx_id = wallet.submit_transaction(
        "alice",
        destination=MultisigWallet.SELF_ADDRESS,
        data=("addOwner", {"owner": "dave"}),
    )
    wallet.confirm_transaction("alice", tx_id)
    wallet.confirm_transaction("carol", tx_id)
    wallet.execute_transaction("carol", tx_id)
    wallet.print_status()
