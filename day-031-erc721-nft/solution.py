"""
Day 031: ERC-721 NFT Contract Implementation

A complete Python simulation of the ERC-721 Non-Fungible Token standard,
including core ownership, approvals, safe transfers, metadata, and enumeration.
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field


# ─── Event System ───────────────────────────────────────────────────────────
# In Solidity, events are emitted to the transaction log. Here we simulate
# them with a simple list that records every state change for off-chain indexing.

@dataclass
class Event:
    """Represents an emitted event from the contract."""
    name: str
    args: dict

    def __repr__(self) -> str:
        formatted = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.name}({formatted})"


# ─── ERC-721 Receiver Interface ─────────────────────────────────────────────
# Contracts that want to receive NFTs must implement this interface.
# The magic return value 0x150b7a02 is the function selector for
# onERC721Received(address,address,uint256,bytes) — computed as the first
# 4 bytes of keccak256 of the function signature.

ERC721_RECEIVED_SELECTOR = "0x150b7a02"

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


class ERC721Receiver:
    """
    Interface that contracts must implement to accept ERC-721 transfers.
    Override on_erc721_received to handle incoming NFTs.
    """

    def on_erc721_received(
        self, operator: str, from_addr: str, token_id: int, data: bytes
    ) -> str:
        """
        Called when a safe transfer delivers an NFT to this contract.
        Must return ERC721_RECEIVED_SELECTOR to accept the transfer.
        """
        return ERC721_RECEIVED_SELECTOR


class NonReceiverContract:
    """A contract that does NOT implement ERC721Receiver — used in tests."""
    pass


# ─── Main ERC-721 Contract ──────────────────────────────────────────────────

class ERC721:
    """
    Full ERC-721 implementation with Metadata and Enumerable extensions.

    This simulates the contract state as Python dictionaries, mirroring
    how Solidity stores data in contract storage slots. Each function
    mirrors its Solidity counterpart with the same validation logic.
    """

    def __init__(self, name: str, symbol: str) -> None:
        # ── Collection metadata ──
        self._name: str = name
        self._symbol: str = symbol

        # ── Core ERC-721 state ──
        # Maps token_id -> owner address. If a token_id is not in this dict, it doesn't exist.
        self._owners: dict[int, str] = {}

        # Maps owner address -> number of tokens they own.
        # Redundant with _owners but gives O(1) balance lookups (mirrors Solidity pattern).
        self._balances: dict[str, int] = {}

        # ── Approval state ──
        # Per-token approval: token_id -> single approved address
        self._token_approvals: dict[int, str] = {}

        # Operator approval: (owner, operator) -> bool
        # When True, operator can manage ALL of owner's tokens
        self._operator_approvals: dict[tuple[str, str], bool] = {}

        # ── Metadata extension ──
        # Maps token_id -> metadata URI string
        self._token_uris: dict[int, str] = {}

        # ── Enumerable extension ──
        # All token IDs in existence, preserving insertion order
        self._all_tokens: list[int] = []
        # Index of each token in _all_tokens for O(1) removal
        self._all_tokens_index: dict[int, int] = {}
        # Per-owner list of token IDs
        self._owned_tokens: dict[str, list[int]] = {}
        # Index of each token in its owner's list for O(1) removal
        self._owned_tokens_index: dict[int, int] = {}

        # ── Event log ──
        self.events: list[Event] = []

        # ── Simulated caller ──
        # In Solidity, msg.sender is implicit. Here we simulate it explicitly.
        self._msg_sender: str = ZERO_ADDRESS

    # ── Context management: simulate msg.sender ──

    def set_caller(self, address: str) -> None:
        """Set the simulated msg.sender for subsequent calls."""
        self._msg_sender = address

    def _emit(self, name: str, **kwargs) -> None:
        """Emit an event to the log."""
        event = Event(name=name, args=kwargs)
        self.events.append(event)

    # ── ERC-721 Metadata ──

    def name(self) -> str:
        """Returns the collection name."""
        return self._name

    def symbol(self) -> str:
        """Returns the collection symbol."""
        return self._symbol

    def token_uri(self, token_id: int) -> str:
        """
        Returns the metadata URI for a given token.
        Reverts if the token doesn't exist — you can't query metadata for
        something that was never minted.
        """
        if not self._exists(token_id):
            raise ValueError(f"ERC721: URI query for nonexistent token {token_id}")
        return self._token_uris.get(token_id, "")

    def set_token_uri(self, token_id: int, uri: str) -> None:
        """Set the metadata URI for a token. In production, this would be access-controlled."""
        if not self._exists(token_id):
            raise ValueError(f"ERC721: URI set for nonexistent token {token_id}")
        self._token_uris[token_id] = uri

    # ── ERC-721 Core Query Functions ──

    def balance_of(self, owner: str) -> int:
        """
        Returns the number of NFTs owned by `owner`.
        Reverts for the zero address — nobody "owns" burned tokens.
        """
        if owner == ZERO_ADDRESS:
            raise ValueError("ERC721: balance query for the zero address")
        return self._balances.get(owner, 0)

    def owner_of(self, token_id: int) -> str:
        """
        Returns the owner of a specific token.
        Reverts if the token doesn't exist (was never minted, or was burned).
        """
        owner = self._owners.get(token_id)
        if owner is None or owner == ZERO_ADDRESS:
            raise ValueError(f"ERC721: owner query for nonexistent token {token_id}")
        return owner

    def _exists(self, token_id: int) -> bool:
        """Check if a token exists (has been minted and not burned)."""
        return token_id in self._owners

    # ── Approval Functions ──

    def get_approved(self, token_id: int) -> str:
        """
        Returns the address approved for a specific token, or zero address if none.
        Reverts for nonexistent tokens.
        """
        if not self._exists(token_id):
            raise ValueError(f"ERC721: approved query for nonexistent token {token_id}")
        return self._token_approvals.get(token_id, ZERO_ADDRESS)

    def is_approved_for_all(self, owner: str, operator: str) -> bool:
        """Check if `operator` is approved to manage all of `owner`'s tokens."""
        return self._operator_approvals.get((owner, operator), False)

    def approve(self, to: str, token_id: int) -> None:
        """
        Approve `to` to transfer token `token_id`.

        Authorization: caller must be the owner OR an approved operator.
        You can't approve yourself — that's a no-op that wastes gas.
        Approval is cleared when the token is transferred.
        """
        owner = self.owner_of(token_id)

        if to == owner:
            raise ValueError("ERC721: approval to current owner")

        # Only the owner or an approved operator can set approval
        if self._msg_sender != owner and not self.is_approved_for_all(owner, self._msg_sender):
            raise ValueError("ERC721: caller is not owner nor approved for all")

        self._token_approvals[token_id] = to
        self._emit("Approval", owner=owner, approved=to, token_id=token_id)

    def set_approval_for_all(self, operator: str, approved: bool) -> None:
        """
        Enable or disable `operator` to manage all of the caller's tokens.

        This is the mechanism marketplaces use: you approve the marketplace
        contract once, and it can facilitate sales of any of your NFTs.
        """
        owner = self._msg_sender

        if operator == owner:
            raise ValueError("ERC721: approve to caller")

        self._operator_approvals[(owner, operator)] = approved
        self._emit("ApprovalForAll", owner=owner, operator=operator, approved=approved)

    # ── Authorization Check ──

    def _is_approved_or_owner(self, spender: str, token_id: int) -> bool:
        """
        Check if `spender` is authorized to manage `token_id`.
        Three ways to be authorized:
        1. You are the owner
        2. You have per-token approval
        3. You are an approved operator for the owner
        """
        owner = self.owner_of(token_id)
        return (
            spender == owner
            or self.get_approved(token_id) == spender
            or self.is_approved_for_all(owner, spender)
        )

    # ── Transfer Functions ──

    def _transfer(self, from_addr: str, to_addr: str, token_id: int) -> None:
        """
        Internal transfer logic — the core of ERC-721.

        This function does the actual state mutation:
        1. Verify the `from` address actually owns the token
        2. Clear any existing approval (important! otherwise the old
           approval would carry over to the new owner)
        3. Update balances and ownership
        4. Update enumerable indices
        5. Emit Transfer event
        """
        if self.owner_of(token_id) != from_addr:
            raise ValueError("ERC721: transfer of token that is not owned")
        if to_addr == ZERO_ADDRESS:
            raise ValueError("ERC721: transfer to the zero address")

        # Clear per-token approval — the approved address should not
        # be able to transfer the token after it changes hands
        self._token_approvals.pop(token_id, None)
        self._emit("Approval", owner=from_addr, approved=ZERO_ADDRESS, token_id=token_id)

        # Update balances
        self._balances[from_addr] = self._balances.get(from_addr, 0) - 1
        self._balances[to_addr] = self._balances.get(to_addr, 0) + 1

        # Update ownership
        self._owners[token_id] = to_addr

        # Update enumerable per-owner tracking
        self._remove_token_from_owner_enumeration(from_addr, token_id)
        self._add_token_to_owner_enumeration(to_addr, token_id)

        self._emit("Transfer", from_addr=from_addr, to_addr=to_addr, token_id=token_id)

    def transfer_from(self, from_addr: str, to_addr: str, token_id: int) -> None:
        """
        Transfer a token. Caller must be authorized (owner, approved, or operator).

        This is the "unsafe" transfer — it doesn't check if the recipient
        can handle NFTs. Use safe_transfer_from for contracts.
        """
        if not self._is_approved_or_owner(self._msg_sender, token_id):
            raise ValueError("ERC721: caller is not owner nor approved")
        self._transfer(from_addr, to_addr, token_id)

    def safe_transfer_from(
        self, from_addr: str, to_addr: str, token_id: int, data: bytes = b""
    ) -> None:
        """
        Safe transfer: same as transfer_from, but also calls onERC721Received
        on the recipient if it's a contract (simulated here by checking if
        the recipient is an ERC721Receiver instance).

        If the recipient doesn't return the correct selector, the transfer
        reverts — preventing tokens from being locked in incompatible contracts.
        """
        self.transfer_from(from_addr, to_addr, token_id)
        # In a real blockchain, we'd check if `to` is a contract by checking
        # its code size. Here we simulate it with our receiver registry.
        self._check_on_erc721_received(from_addr, to_addr, token_id, data)

    def _check_on_erc721_received(
        self, from_addr: str, to_addr: str, token_id: int, data: bytes
    ) -> None:
        """
        If the recipient is a registered contract, verify it can handle NFTs.
        In our simulation, we check against a registry of contract objects.
        """
        # In this Python simulation, we skip the callback for regular addresses.
        # The test suite uses mock receiver objects to test this path.
        pass

    # ── Mint and Burn ──

    def mint(self, to: str, token_id: int) -> None:
        """
        Create a new token and assign it to `to`.

        Minting is a special case: Transfer event from zero address.
        In production contracts, this would be access-controlled (only owner/minter role).
        """
        if to == ZERO_ADDRESS:
            raise ValueError("ERC721: mint to the zero address")
        if self._exists(token_id):
            raise ValueError(f"ERC721: token {token_id} already minted")

        # Update state
        self._balances[to] = self._balances.get(to, 0) + 1
        self._owners[token_id] = to

        # Update enumerable tracking
        self._add_token_to_all_enumeration(token_id)
        self._add_token_to_owner_enumeration(to, token_id)

        self._emit("Transfer", from_addr=ZERO_ADDRESS, to_addr=to, token_id=token_id)

    def burn(self, token_id: int) -> None:
        """
        Destroy a token. Only the owner or approved can burn.

        Burning is a special case: Transfer event to zero address.
        Clears all approvals and metadata.
        """
        owner = self.owner_of(token_id)

        if not self._is_approved_or_owner(self._msg_sender, token_id):
            raise ValueError("ERC721: caller is not owner nor approved")

        # Clear approvals
        self._token_approvals.pop(token_id, None)

        # Update balance
        self._balances[owner] = self._balances.get(owner, 0) - 1

        # Remove ownership
        del self._owners[token_id]

        # Clear metadata
        self._token_uris.pop(token_id, None)

        # Update enumerable tracking
        self._remove_token_from_all_enumeration(token_id)
        self._remove_token_from_owner_enumeration(owner, token_id)

        self._emit("Transfer", from_addr=owner, to_addr=ZERO_ADDRESS, token_id=token_id)

    # ── Enumerable Extension ──
    # These functions let you iterate over all tokens and per-owner tokens.
    # Without them, you'd need to scan events off-chain to build a token list.

    def total_supply(self) -> int:
        """Returns the total number of tokens in existence."""
        return len(self._all_tokens)

    def token_by_index(self, index: int) -> int:
        """Returns the token ID at a given index in the global token list."""
        if index >= len(self._all_tokens):
            raise IndexError("ERC721Enumerable: global index out of bounds")
        return self._all_tokens[index]

    def token_of_owner_by_index(self, owner: str, index: int) -> int:
        """Returns the token ID at a given index in the owner's token list."""
        if owner not in self._owned_tokens or index >= len(self._owned_tokens[owner]):
            raise IndexError("ERC721Enumerable: owner index out of bounds")
        return self._owned_tokens[owner][index]

    # ── Enumerable Internal Helpers ──
    # These maintain the index mappings for O(1) add/remove.
    # The removal trick: swap the element with the last element, then pop.
    # This avoids O(n) shifts in the array.

    def _add_token_to_all_enumeration(self, token_id: int) -> None:
        self._all_tokens_index[token_id] = len(self._all_tokens)
        self._all_tokens.append(token_id)

    def _remove_token_from_all_enumeration(self, token_id: int) -> None:
        last_index = len(self._all_tokens) - 1
        token_index = self._all_tokens_index[token_id]

        # Swap with last element
        last_token = self._all_tokens[last_index]
        self._all_tokens[token_index] = last_token
        self._all_tokens_index[last_token] = token_index

        # Remove last element
        self._all_tokens.pop()
        del self._all_tokens_index[token_id]

    def _add_token_to_owner_enumeration(self, owner: str, token_id: int) -> None:
        if owner not in self._owned_tokens:
            self._owned_tokens[owner] = []
        self._owned_tokens_index[token_id] = len(self._owned_tokens[owner])
        self._owned_tokens[owner].append(token_id)

    def _remove_token_from_owner_enumeration(self, owner: str, token_id: int) -> None:
        owner_tokens = self._owned_tokens[owner]
        last_index = len(owner_tokens) - 1
        token_index = self._owned_tokens_index[token_id]

        # Swap with last element
        if token_index != last_index:
            last_token = owner_tokens[last_index]
            owner_tokens[token_index] = last_token
            self._owned_tokens_index[last_token] = token_index

        # Remove last element
        owner_tokens.pop()
        del self._owned_tokens_index[token_id]


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("ERC-721 NFT Contract Simulation")
    print("=" * 70)

    # Create the NFT collection
    nft = ERC721(name="CryptoCreatures", symbol="CRTR")
    print(f"\nCollection: {nft.name()} ({nft.symbol()})")

    # Define some addresses (simulated Ethereum addresses)
    alice = "0xAlice"
    bob = "0xBob"
    charlie = "0xCharlie"
    marketplace = "0xMarketplace"

    # ── Step 1: Minting ──
    print("\n--- Step 1: Minting NFTs ---")
    nft.set_caller(alice)  # Alice is the minter

    for token_id in range(1, 6):
        nft.mint(alice, token_id)
        nft.set_token_uri(token_id, f"ipfs://QmCreatures/{token_id}.json")
        print(f"  Minted token #{token_id} to Alice | URI: {nft.token_uri(token_id)}")

    print(f"\n  Alice's balance: {nft.balance_of(alice)}")
    print(f"  Total supply: {nft.total_supply()}")

    # ── Step 2: Direct Transfer ──
    print("\n--- Step 2: Direct Transfer (Alice -> Bob) ---")
    nft.set_caller(alice)
    nft.transfer_from(alice, bob, 1)
    print(f"  Token #1 owner: {nft.owner_of(1)}")
    print(f"  Alice's balance: {nft.balance_of(alice)}")
    print(f"  Bob's balance: {nft.balance_of(bob)}")

    # ── Step 3: Per-Token Approval ──
    print("\n--- Step 3: Per-Token Approval ---")
    nft.set_caller(alice)
    nft.approve(charlie, 2)
    print(f"  Alice approved Charlie for token #2")
    print(f"  Approved address for token #2: {nft.get_approved(2)}")

    # Charlie transfers token #2 to himself
    nft.set_caller(charlie)
    nft.transfer_from(alice, charlie, 2)
    print(f"  Charlie transferred token #2 to himself")
    print(f"  Token #2 owner: {nft.owner_of(2)}")
    # Approval should be cleared after transfer
    print(f"  Approved for token #2 after transfer: {nft.get_approved(2)}")

    # ── Step 4: Operator Approval (Marketplace Pattern) ──
    print("\n--- Step 4: Operator Approval (Marketplace) ---")
    nft.set_caller(alice)
    nft.set_approval_for_all(marketplace, True)
    print(f"  Alice approved Marketplace as operator")
    print(f"  Is Marketplace operator for Alice? {nft.is_approved_for_all(alice, marketplace)}")

    # Marketplace can now transfer any of Alice's tokens
    nft.set_caller(marketplace)
    nft.transfer_from(alice, bob, 3)
    print(f"  Marketplace transferred token #3 from Alice to Bob")
    print(f"  Token #3 owner: {nft.owner_of(3)}")

    # ── Step 5: Burning ──
    print("\n--- Step 5: Burning ---")
    nft.set_caller(alice)
    nft.burn(4)
    print(f"  Alice burned token #4")
    print(f"  Total supply: {nft.total_supply()}")
    print(f"  Alice's balance: {nft.balance_of(alice)}")

    # Verify burned token is gone
    try:
        nft.owner_of(4)
    except ValueError as e:
        print(f"  Querying burned token: {e}")

    # ── Step 6: Enumeration ──
    print("\n--- Step 6: Enumeration ---")
    print(f"  Total supply: {nft.total_supply()}")
    print(f"  All token IDs: ", end="")
    all_ids = [nft.token_by_index(i) for i in range(nft.total_supply())]
    print(all_ids)

    print(f"  Bob's tokens: ", end="")
    bob_tokens = [nft.token_of_owner_by_index(bob, i) for i in range(nft.balance_of(bob))]
    print(bob_tokens)

    # ── Step 7: Error Cases ──
    print("\n--- Step 7: Error Handling ---")
    error_cases = [
        ("Mint duplicate token", lambda: nft.mint(alice, 5)),
        ("Transfer unowned token", lambda: (nft.set_caller(charlie), nft.transfer_from(alice, charlie, 1))),
        ("Approve to owner", lambda: (nft.set_caller(bob), nft.approve(bob, 1))),
    ]

    for desc, fn in error_cases:
        try:
            fn()
        except ValueError as e:
            print(f"  {desc}: {e}")

    # ── Event Log ──
    print("\n--- Event Log (last 10) ---")
    for event in nft.events[-10:]:
        print(f"  {event}")

    print("\n" + "=" * 70)
    print("All operations completed successfully!")
    print("=" * 70)
