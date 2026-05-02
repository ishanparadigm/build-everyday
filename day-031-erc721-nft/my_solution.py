"""
Day 031: ERC-721 NFT Contract - Your Implementation

Implement the ERC-721 Non-Fungible Token standard from scratch.
Each token is unique (has a distinct ID) with exactly one owner.

Key data structures you'll need:
- owners: maps token_id -> owner address
- balances: maps address -> token count (redundant but O(1) lookups)
- token_approvals: maps token_id -> approved address
- operator_approvals: maps (owner, operator) -> bool

Run tests with: python3 -m pytest tests.py -v
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


@dataclass
class Event:
    """Represents an emitted event from the contract."""
    name: str
    args: dict

    def __repr__(self) -> str:
        formatted = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.name}({formatted})"


class ERC721:
    """
    ERC-721 Non-Fungible Token implementation.

    Hint: Think about what state you need in __init__:
    - Four core mappings (owners, balances, token_approvals, operator_approvals)
    - Metadata storage (name, symbol, token URIs)
    - Enumerable tracking (all tokens list, per-owner token lists)
    - Event log and simulated msg.sender
    """

    def __init__(self, name: str, symbol: str) -> None:
        raise NotImplementedError("TODO: initialize contract state")

    def set_caller(self, address: str) -> None:
        """Set the simulated msg.sender for subsequent calls."""
        raise NotImplementedError("TODO: implement this")

    def _emit(self, name: str, **kwargs) -> None:
        """Emit an event to the log."""
        raise NotImplementedError("TODO: implement this")

    # ── Metadata ──

    def name(self) -> str:
        """Returns the collection name."""
        raise NotImplementedError("TODO: implement this")

    def symbol(self) -> str:
        """Returns the collection symbol."""
        raise NotImplementedError("TODO: implement this")

    def token_uri(self, token_id: int) -> str:
        """
        Returns the metadata URI for a given token.
        Hint: Revert if the token doesn't exist.
        """
        raise NotImplementedError("TODO: implement this")

    def set_token_uri(self, token_id: int, uri: str) -> None:
        """Set the metadata URI for a token."""
        raise NotImplementedError("TODO: implement this")

    # ── Core Query Functions ──

    def balance_of(self, owner: str) -> int:
        """
        Returns the number of NFTs owned by `owner`.
        Hint: Revert for the zero address.
        """
        raise NotImplementedError("TODO: implement this")

    def owner_of(self, token_id: int) -> str:
        """
        Returns the owner of a specific token.
        Hint: Revert if the token doesn't exist.
        """
        raise NotImplementedError("TODO: implement this")

    def _exists(self, token_id: int) -> bool:
        """Check if a token exists (has been minted and not burned)."""
        raise NotImplementedError("TODO: implement this")

    # ── Approval Functions ──

    def get_approved(self, token_id: int) -> str:
        """
        Returns the approved address for a token, or ZERO_ADDRESS if none.
        Hint: Revert for nonexistent tokens.
        """
        raise NotImplementedError("TODO: implement this")

    def is_approved_for_all(self, owner: str, operator: str) -> bool:
        """Check if `operator` is approved to manage all of `owner`'s tokens."""
        raise NotImplementedError("TODO: implement this")

    def approve(self, to: str, token_id: int) -> None:
        """
        Approve `to` to transfer token `token_id`.

        Hints:
        - Can't approve the current owner (no-op)
        - Only owner or approved operator can set approval
        - Emit an Approval event
        """
        raise NotImplementedError("TODO: implement this")

    def set_approval_for_all(self, operator: str, approved: bool) -> None:
        """
        Enable/disable `operator` to manage all caller's tokens.

        Hints:
        - Can't approve yourself as operator
        - Emit an ApprovalForAll event
        """
        raise NotImplementedError("TODO: implement this")

    # ── Authorization ──

    def _is_approved_or_owner(self, spender: str, token_id: int) -> bool:
        """
        Check if `spender` can manage `token_id`.
        Hint: Three ways to be authorized — owner, per-token approval, operator approval.
        """
        raise NotImplementedError("TODO: implement this")

    # ── Transfer Functions ──

    def _transfer(self, from_addr: str, to_addr: str, token_id: int) -> None:
        """
        Internal transfer logic.

        Hints:
        - Verify from_addr actually owns the token
        - Can't transfer to zero address
        - Clear per-token approval on transfer
        - Update balances and ownership
        - Update enumerable indices
        - Emit Transfer event
        """
        raise NotImplementedError("TODO: implement this")

    def transfer_from(self, from_addr: str, to_addr: str, token_id: int) -> None:
        """
        Transfer a token. Caller must be authorized.
        Hint: Check _is_approved_or_owner, then call _transfer.
        """
        raise NotImplementedError("TODO: implement this")

    def safe_transfer_from(
        self, from_addr: str, to_addr: str, token_id: int, data: bytes = b""
    ) -> None:
        """
        Safe transfer: same as transfer_from + receiver callback check.
        Hint: In this simulation, call transfer_from then _check_on_erc721_received.
        """
        raise NotImplementedError("TODO: implement this")

    def _check_on_erc721_received(
        self, from_addr: str, to_addr: str, token_id: int, data: bytes
    ) -> None:
        """Check if recipient can handle ERC-721 tokens (simulated)."""
        pass  # Simplified — no-op in this simulation

    # ── Mint and Burn ──

    def mint(self, to: str, token_id: int) -> None:
        """
        Create a new token assigned to `to`.

        Hints:
        - Can't mint to zero address
        - Can't mint a token that already exists
        - Update balances, owners, enumerable tracking
        - Emit Transfer from ZERO_ADDRESS
        """
        raise NotImplementedError("TODO: implement this")

    def burn(self, token_id: int) -> None:
        """
        Destroy a token.

        Hints:
        - Only owner or approved can burn
        - Clear approvals, metadata, ownership
        - Update balances and enumerable tracking
        - Emit Transfer to ZERO_ADDRESS
        """
        raise NotImplementedError("TODO: implement this")

    # ── Enumerable Extension ──

    def total_supply(self) -> int:
        """Returns the total number of tokens in existence."""
        raise NotImplementedError("TODO: implement this")

    def token_by_index(self, index: int) -> int:
        """Returns the token ID at a given index in the global list."""
        raise NotImplementedError("TODO: implement this")

    def token_of_owner_by_index(self, owner: str, index: int) -> int:
        """Returns the token ID at a given index in an owner's list."""
        raise NotImplementedError("TODO: implement this")

    # ── Enumerable Helpers ──
    # Hint: Use the swap-with-last-then-pop trick for O(1) removal.
    # You'll need index mappings to make this efficient.

    def _add_token_to_all_enumeration(self, token_id: int) -> None:
        raise NotImplementedError("TODO: implement this")

    def _remove_token_from_all_enumeration(self, token_id: int) -> None:
        raise NotImplementedError("TODO: implement this")

    def _add_token_to_owner_enumeration(self, owner: str, token_id: int) -> None:
        raise NotImplementedError("TODO: implement this")

    def _remove_token_from_owner_enumeration(self, owner: str, token_id: int) -> None:
        raise NotImplementedError("TODO: implement this")


# ─── Test your implementation ────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing your ERC-721 implementation...\n")

    nft = ERC721(name="TestNFT", symbol="TNFT")
    print(f"Collection: {nft.name()} ({nft.symbol()})")

    alice = "0xAlice"
    bob = "0xBob"

    # Test minting
    nft.set_caller(alice)
    nft.mint(alice, 1)
    nft.mint(alice, 2)
    nft.mint(bob, 3)
    print(f"Minted 3 tokens. Total supply: {nft.total_supply()}")
    print(f"Alice balance: {nft.balance_of(alice)}, Bob balance: {nft.balance_of(bob)}")

    # Test transfer
    nft.transfer_from(alice, bob, 1)
    print(f"\nAfter transfer: Token #1 owner = {nft.owner_of(1)}")

    # Test approval
    nft.set_caller(alice)
    nft.approve(bob, 2)
    nft.set_caller(bob)
    nft.transfer_from(alice, bob, 2)
    print(f"After approved transfer: Token #2 owner = {nft.owner_of(2)}")

    # Test burn
    nft.set_caller(bob)
    nft.burn(3)
    print(f"\nAfter burning token #3: Total supply = {nft.total_supply()}")

    print("\nAll basic tests passed!")
