"""
Day 030: ERC-20 Token Implementation — Your Solution

Implement the ERC-20 token standard from scratch. The ERC-20 is the interface
that powers every fungible token on Ethereum. Your job: build the state machine
that tracks balances, manages allowances, and enforces transfer rules.

Hint: The entire token is backed by just three data structures:
  - balances: dict mapping address -> amount
  - allowances: dict mapping (owner, spender) -> approved amount
  - total_supply: int tracking total tokens in existence
"""

from __future__ import annotations
from dataclasses import dataclass


# =============================================================================
# Events — these are provided for you
# =============================================================================

@dataclass
class TransferEvent:
    """Emitted on every token movement, including mints (from=0x0) and burns (to=0x0)."""
    from_addr: str
    to_addr: str
    amount: int

    def __repr__(self) -> str:
        return f"Transfer({self.from_addr} -> {self.to_addr}, {self.amount})"


@dataclass
class ApprovalEvent:
    """Emitted when an owner authorizes a spender to use their tokens."""
    owner: str
    spender: str
    amount: int

    def __repr__(self) -> str:
        return f"Approval({self.owner} -> {self.spender}, {self.amount})"


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


# =============================================================================
# ERC-20 Token — Implement this class
# =============================================================================

class ERC20Token:
    """
    Full ERC-20 implementation with mint/burn extensions.

    Think about:
    - What state do you need to track?
    - What invariants must always hold? (hint: sum of balances == total_supply)
    - What checks need to happen before each state mutation?
    """

    def __init__(
        self,
        name: str,
        symbol: str,
        decimals: int = 18,
        owner: str = "deployer",
    ) -> None:
        """
        Initialize the token with metadata and empty state.

        Hint: You need storage for balances (dict), allowances (nested dict),
        total supply (int), events (list), and the owner address.
        """
        raise NotImplementedError("TODO: implement this")

    # -------------------------------------------------------------------------
    # Read-Only Functions
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        raise NotImplementedError("TODO: implement this")

    @property
    def symbol(self) -> str:
        raise NotImplementedError("TODO: implement this")

    @property
    def decimals(self) -> int:
        raise NotImplementedError("TODO: implement this")

    def total_supply(self) -> int:
        """Returns the total number of tokens in existence."""
        raise NotImplementedError("TODO: implement this")

    def balance_of(self, account: str) -> int:
        """
        Returns the token balance for an account.

        Hint: Unknown addresses should return 0, not raise an error.
        """
        raise NotImplementedError("TODO: implement this")

    def allowance(self, owner: str, spender: str) -> int:
        """
        Returns how many tokens `spender` can spend on behalf of `owner`.

        Hint: This is a nested lookup — handle missing keys gracefully.
        """
        raise NotImplementedError("TODO: implement this")

    # -------------------------------------------------------------------------
    # State-Changing Functions
    # -------------------------------------------------------------------------

    def transfer(self, sender: str, to: str, amount: int) -> bool:
        """
        Move `amount` tokens from `sender` to `to`.

        Checks needed:
        - to != zero address
        - amount >= 0
        - sender has sufficient balance

        Hint: Update both balances and emit a TransferEvent.
        """
        raise NotImplementedError("TODO: implement this")

    def approve(self, owner: str, spender: str, amount: int) -> bool:
        """
        Allow `spender` to spend up to `amount` of `owner`'s tokens.

        Hint: No balance check needed here! You can approve more than
        you own — the check happens at transferFrom time.
        Emit an ApprovalEvent.
        """
        raise NotImplementedError("TODO: implement this")

    def transfer_from(self, caller: str, from_addr: str, to: str, amount: int) -> bool:
        """
        Move tokens from `from_addr` to `to`, using `caller`'s allowance.

        This is the key function for the delegation pattern.

        Checks needed:
        - to != zero address
        - amount >= 0
        - caller has sufficient allowance from from_addr
        - from_addr has sufficient balance

        Hint: Decrease allowance FIRST, then move tokens (checks-effects-interactions).
        Emit a TransferEvent.
        """
        raise NotImplementedError("TODO: implement this")

    # -------------------------------------------------------------------------
    # Extensions: Mint, Burn, Allowance Helpers
    # -------------------------------------------------------------------------

    def mint(self, caller: str, to: str, amount: int) -> bool:
        """
        Create new tokens and assign them to `to`. Only the owner can mint.

        Hint: Increase total_supply AND balances[to]. Emit Transfer from ZERO_ADDRESS.
        """
        raise NotImplementedError("TODO: implement this")

    def burn(self, owner: str, amount: int) -> bool:
        """
        Destroy tokens from `owner`'s balance.

        Hint: Decrease both balances[owner] AND total_supply.
        Emit Transfer to ZERO_ADDRESS.
        """
        raise NotImplementedError("TODO: implement this")

    def increase_allowance(self, owner: str, spender: str, added_value: int) -> bool:
        """
        Atomically increase allowance — safe alternative to approve().

        Hint: Read current allowance, then approve(current + added_value).
        """
        raise NotImplementedError("TODO: implement this")

    def decrease_allowance(self, owner: str, spender: str, subtracted_value: int) -> bool:
        """
        Atomically decrease allowance.

        Hint: Check that current allowance >= subtracted_value to prevent underflow.
        """
        raise NotImplementedError("TODO: implement this")

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def format_amount(self, amount: int) -> str:
        """Convert raw amount to human-readable string with proper decimal places."""
        raise NotImplementedError("TODO: implement this")

    def get_events(self) -> list[TransferEvent | ApprovalEvent]:
        """Return all emitted events."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    ONE_TOKEN = 10 ** 18

    # Deploy
    token = ERC20Token("BuildToken", "BLD", decimals=18, owner="deployer")
    print(f"Deployed: {token.name} ({token.symbol}), {token.decimals} decimals")

    # Mint
    token.mint("deployer", "deployer", 1_000_000 * ONE_TOKEN)
    print(f"Minted 1M BLD. Supply: {token.format_amount(token.total_supply())} BLD")

    # Transfer
    token.transfer("deployer", "alice", 50_000 * ONE_TOKEN)
    print(f"Transferred 50k to Alice. Alice: {token.format_amount(token.balance_of('alice'))} BLD")

    # Approve + TransferFrom
    token.approve("alice", "dex", 10_000 * ONE_TOKEN)
    token.transfer_from("dex", "alice", "pool", 3_000 * ONE_TOKEN)
    print(f"DEX swapped 3k from Alice to pool. Remaining allowance: {token.format_amount(token.allowance('alice', 'dex'))} BLD")

    # Burn
    token.burn("deployer", 100_000 * ONE_TOKEN)
    print(f"Burned 100k. New supply: {token.format_amount(token.total_supply())} BLD")

    print("\nAll events:")
    for e in token.get_events():
        print(f"  {e}")

    print("\nAll tests passed!" if token.total_supply() == 900_000 * ONE_TOKEN else "\nSomething went wrong!")
