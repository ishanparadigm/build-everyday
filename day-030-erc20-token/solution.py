"""
Day 030: ERC-20 Token Implementation

A complete ERC-20 token built in Python, focusing on the state management
and logic that powers every fungible token on Ethereum.

Key insight: An ERC-20 token is just a ledger with three maps (balances,
allowances, total_supply) and strict rules about who can modify them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Event Types — In a real blockchain, these go into transaction receipts
# and are indexed for off-chain queries (e.g., tracking all transfers)
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


# =============================================================================
# ERC-20 Token
# =============================================================================

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


class ERC20Token:
    """
    Full ERC-20 implementation with mint/burn extensions.

    State model:
      - balances: maps address -> token balance (in smallest units)
      - allowances: maps (owner, spender) -> approved amount
      - total_supply: sum of all balances, invariant maintained by mint/burn

    All amounts are in the smallest unit (like wei for ETH). To get
    human-readable amounts, divide by 10**decimals.
    """

    def __init__(
        self,
        name: str,
        symbol: str,
        decimals: int = 18,
        owner: str = "deployer",
    ) -> None:
        # Token metadata — immutable after deployment
        self._name = name
        self._symbol = symbol
        self._decimals = decimals

        # The owner has special privileges (minting). In production,
        # you'd use a more sophisticated access control system (roles, multisig).
        self._owner = owner

        # Core state: the entire token is these three data structures
        self._balances: dict[str, int] = {}
        self._allowances: dict[str, dict[str, int]] = {}
        self._total_supply: int = 0

        # Event log — simulates blockchain event emission
        self._events: list[TransferEvent | ApprovalEvent] = []

    # -------------------------------------------------------------------------
    # ERC-20 Read-Only Functions (view/pure in Solidity)
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def decimals(self) -> int:
        return self._decimals

    def total_supply(self) -> int:
        """Returns the total number of tokens in existence."""
        return self._total_supply

    def balance_of(self, account: str) -> int:
        """Returns the token balance for an account. Defaults to 0 for unknown addresses."""
        return self._balances.get(account, 0)

    def allowance(self, owner: str, spender: str) -> int:
        """
        Returns how many tokens `spender` is allowed to spend on behalf of `owner`.

        This is the core of the delegation pattern: owner sets an allowance,
        then spender can call transferFrom up to that amount.
        """
        return self._allowances.get(owner, {}).get(spender, 0)

    # -------------------------------------------------------------------------
    # ERC-20 State-Changing Functions
    # -------------------------------------------------------------------------

    def transfer(self, sender: str, to: str, amount: int) -> bool:
        """
        Move `amount` tokens from `sender` to `to`.

        In Solidity, `sender` would be msg.sender (the transaction caller).
        We pass it explicitly since we don't have a blockchain runtime.

        Why we check balance explicitly: without this guard, a sender with
        0 tokens could "transfer" and create tokens from nothing via underflow.
        Solidity >=0.8 catches this automatically, but being explicit is clearer.
        """
        if to == ZERO_ADDRESS:
            raise ValueError("ERC20: transfer to the zero address — use burn() instead")
        if amount < 0:
            raise ValueError("ERC20: transfer amount must be non-negative")

        sender_balance = self.balance_of(sender)
        if sender_balance < amount:
            raise ValueError(
                f"ERC20: insufficient balance. "
                f"{sender} has {sender_balance}, tried to send {amount}"
            )

        # State mutation — these two lines are the atomic "transfer"
        # In a real blockchain, if anything reverts after the first line,
        # both changes are rolled back (transaction atomicity).
        self._balances[sender] = sender_balance - amount
        self._balances[to] = self.balance_of(to) + amount

        event = TransferEvent(sender, to, amount)
        self._events.append(event)
        return True

    def approve(self, owner: str, spender: str, amount: int) -> bool:
        """
        Allow `spender` to spend up to `amount` tokens on behalf of `owner`.

        IMPORTANT: This sets the allowance to exactly `amount`, replacing
        any previous value. This creates a race condition vulnerability
        (see README). In production, prefer increaseAllowance/decreaseAllowance.

        Note: No balance check here. You can approve more than you own.
        The check happens at transferFrom time. This is intentional — you
        might approve now and receive tokens later.
        """
        if spender == ZERO_ADDRESS:
            raise ValueError("ERC20: approve to the zero address")
        if amount < 0:
            raise ValueError("ERC20: approval amount must be non-negative")

        if owner not in self._allowances:
            self._allowances[owner] = {}
        self._allowances[owner][spender] = amount

        event = ApprovalEvent(owner, spender, amount)
        self._events.append(event)
        return True

    def transfer_from(self, caller: str, from_addr: str, to: str, amount: int) -> bool:
        """
        Move `amount` tokens from `from_addr` to `to`, using `caller`'s allowance.

        This is the second step of the approve+transferFrom pattern:
        1. Owner called approve(caller, amount)
        2. Caller now calls transferFrom(owner, recipient, amount)

        The caller must have sufficient allowance AND the from_addr must
        have sufficient balance. Both are checked.

        Why decrease allowance? Without this, a single approve(100) would
        let the spender drain tokens forever, 100 at a time.
        """
        if to == ZERO_ADDRESS:
            raise ValueError("ERC20: transfer to the zero address")
        if amount < 0:
            raise ValueError("ERC20: transfer amount must be non-negative")

        # Check allowance — does the caller have permission?
        current_allowance = self.allowance(from_addr, caller)
        if current_allowance < amount:
            raise ValueError(
                f"ERC20: insufficient allowance. "
                f"{caller} approved for {current_allowance}, tried to spend {amount}"
            )

        # Check balance — does the owner actually have the tokens?
        from_balance = self.balance_of(from_addr)
        if from_balance < amount:
            raise ValueError(
                f"ERC20: insufficient balance. "
                f"{from_addr} has {from_balance}, tried to send {amount}"
            )

        # Decrease allowance FIRST (checks-effects-interactions pattern).
        # In Solidity, this ordering matters for reentrancy protection.
        # We maintain the pattern even in Python for correctness education.
        self._allowances[from_addr][caller] = current_allowance - amount

        # Execute the transfer
        self._balances[from_addr] = from_balance - amount
        self._balances[to] = self.balance_of(to) + amount

        event = TransferEvent(from_addr, to, amount)
        self._events.append(event)
        return True

    # -------------------------------------------------------------------------
    # Extensions: Mint, Burn, Allowance Helpers
    # -------------------------------------------------------------------------

    def mint(self, caller: str, to: str, amount: int) -> bool:
        """
        Create `amount` new tokens and assign them to `to`.

        Only the owner can mint. In production, you'd use role-based
        access control (e.g., MINTER_ROLE) instead of a single owner.

        Minting is a Transfer from address(0) — this convention lets
        indexers and block explorers detect token creation.
        """
        if caller != self._owner:
            raise PermissionError("ERC20: only owner can mint")
        if to == ZERO_ADDRESS:
            raise ValueError("ERC20: mint to the zero address")
        if amount <= 0:
            raise ValueError("ERC20: mint amount must be positive")

        self._total_supply += amount
        self._balances[to] = self.balance_of(to) + amount

        event = TransferEvent(ZERO_ADDRESS, to, amount)
        self._events.append(event)
        return True

    def burn(self, owner: str, amount: int) -> bool:
        """
        Destroy `amount` tokens from `owner`'s balance.

        Anyone can burn their own tokens. Burning reduces total supply,
        making remaining tokens (theoretically) more scarce.

        Burning is a Transfer to address(0).
        """
        if amount <= 0:
            raise ValueError("ERC20: burn amount must be positive")

        balance = self.balance_of(owner)
        if balance < amount:
            raise ValueError(
                f"ERC20: burn exceeds balance. "
                f"{owner} has {balance}, tried to burn {amount}"
            )

        self._balances[owner] = balance - amount
        self._total_supply -= amount

        event = TransferEvent(owner, ZERO_ADDRESS, amount)
        self._events.append(event)
        return True

    def increase_allowance(self, owner: str, spender: str, added_value: int) -> bool:
        """
        Atomically increase the allowance granted to `spender` by `added_value`.

        This is the safe alternative to approve() that avoids the race condition.
        Instead of setting an absolute value, you add to the current allowance.
        """
        if added_value < 0:
            raise ValueError("ERC20: added value must be non-negative")
        current = self.allowance(owner, spender)
        return self.approve(owner, spender, current + added_value)

    def decrease_allowance(self, owner: str, spender: str, subtracted_value: int) -> bool:
        """
        Atomically decrease the allowance granted to `spender`.

        Reverts if the subtraction would go below zero — this prevents
        the spender from gaining allowance through underflow.
        """
        if subtracted_value < 0:
            raise ValueError("ERC20: subtracted value must be non-negative")
        current = self.allowance(owner, spender)
        if current < subtracted_value:
            raise ValueError(
                f"ERC20: decreased allowance below zero. "
                f"Current: {current}, decrease: {subtracted_value}"
            )
        return self.approve(owner, spender, current - subtracted_value)

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def format_amount(self, amount: int) -> str:
        """Convert raw amount to human-readable string with proper decimal places."""
        if self._decimals == 0:
            return str(amount)
        whole = amount // (10 ** self._decimals)
        frac = amount % (10 ** self._decimals)
        frac_str = str(frac).zfill(self._decimals).rstrip("0") or "0"
        return f"{whole}.{frac_str}"

    def get_events(self) -> list[TransferEvent | ApprovalEvent]:
        """Return all emitted events (simulates reading from blockchain logs)."""
        return list(self._events)

    def __repr__(self) -> str:
        return (
            f"ERC20({self._name} [{self._symbol}], "
            f"supply={self.format_amount(self._total_supply)} {self._symbol})"
        )


# =============================================================================
# Demo: Full Token Lifecycle
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("ERC-20 TOKEN IMPLEMENTATION — FULL LIFECYCLE DEMO")
    print("=" * 70)

    # --- Step 1: Deploy token ---
    print("\n--- Step 1: Deploy Token ---")
    # 18 decimals is the Ethereum convention (matches ETH's wei)
    token = ERC20Token(
        name="BuildToken",
        symbol="BLD",
        decimals=18,
        owner="deployer",
    )
    print(f"Deployed: {token}")
    print(f"Total supply: {token.total_supply()} (no tokens exist yet)")

    # --- Step 2: Mint initial supply ---
    print("\n--- Step 2: Mint Initial Supply ---")
    # 1,000,000 tokens = 1_000_000 * 10^18 smallest units
    ONE_TOKEN = 10 ** 18
    INITIAL_SUPPLY = 1_000_000 * ONE_TOKEN

    token.mint("deployer", "deployer", INITIAL_SUPPLY)
    print(f"Minted {token.format_amount(INITIAL_SUPPLY)} BLD to deployer")
    print(f"Total supply: {token.format_amount(token.total_supply())} BLD")
    print(f"Deployer balance: {token.format_amount(token.balance_of('deployer'))} BLD")

    # --- Step 3: Direct transfer ---
    print("\n--- Step 3: Direct Transfer ---")
    transfer_amount = 50_000 * ONE_TOKEN
    token.transfer("deployer", "alice", transfer_amount)
    print(f"Transferred {token.format_amount(transfer_amount)} BLD from deployer to alice")
    print(f"  Deployer balance: {token.format_amount(token.balance_of('deployer'))} BLD")
    print(f"  Alice balance:    {token.format_amount(token.balance_of('alice'))} BLD")

    # --- Step 4: Approve + TransferFrom (the delegation pattern) ---
    print("\n--- Step 4: Approve + TransferFrom (Delegation Pattern) ---")
    print("Scenario: Alice approves the DEX to spend 10,000 BLD on her behalf")

    approve_amount = 10_000 * ONE_TOKEN
    token.approve("alice", "dex_contract", approve_amount)
    print(f"  Alice approved DEX for {token.format_amount(approve_amount)} BLD")
    print(f"  Allowance: {token.format_amount(token.allowance('alice', 'dex_contract'))} BLD")

    # DEX executes a swap — takes 3,000 BLD from Alice, sends to liquidity pool
    swap_amount = 3_000 * ONE_TOKEN
    token.transfer_from("dex_contract", "alice", "liquidity_pool", swap_amount)
    print(f"\n  DEX swapped {token.format_amount(swap_amount)} BLD from Alice to pool")
    print(f"  Alice balance:    {token.format_amount(token.balance_of('alice'))} BLD")
    print(f"  Pool balance:     {token.format_amount(token.balance_of('liquidity_pool'))} BLD")
    print(f"  Remaining allowance: {token.format_amount(token.allowance('alice', 'dex_contract'))} BLD")

    # --- Step 5: Insufficient balance/allowance errors ---
    print("\n--- Step 5: Error Handling ---")

    # Try to transfer more than balance
    try:
        token.transfer("alice", "bob", 100_000 * ONE_TOKEN)
    except ValueError as e:
        print(f"  [CAUGHT] {e}")

    # Try to transferFrom beyond allowance
    try:
        token.transfer_from("dex_contract", "alice", "bob", 50_000 * ONE_TOKEN)
    except ValueError as e:
        print(f"  [CAUGHT] {e}")

    # Try to mint as non-owner
    try:
        token.mint("alice", "alice", 1_000_000 * ONE_TOKEN)
    except PermissionError as e:
        print(f"  [CAUGHT] {e}")

    # --- Step 6: Burn tokens ---
    print("\n--- Step 6: Burn Tokens ---")
    burn_amount = 100_000 * ONE_TOKEN
    before_supply = token.total_supply()
    token.burn("deployer", burn_amount)
    print(f"  Deployer burned {token.format_amount(burn_amount)} BLD")
    print(f"  Supply before: {token.format_amount(before_supply)} BLD")
    print(f"  Supply after:  {token.format_amount(token.total_supply())} BLD")

    # --- Step 7: Safe allowance management ---
    print("\n--- Step 7: Safe Allowance Management ---")
    print("  Using increaseAllowance/decreaseAllowance to avoid race condition")
    print(f"  Current DEX allowance: {token.format_amount(token.allowance('alice', 'dex_contract'))} BLD")

    token.increase_allowance("alice", "dex_contract", 5_000 * ONE_TOKEN)
    print(f"  After increase by 5,000: {token.format_amount(token.allowance('alice', 'dex_contract'))} BLD")

    token.decrease_allowance("alice", "dex_contract", 2_000 * ONE_TOKEN)
    print(f"  After decrease by 2,000: {token.format_amount(token.allowance('alice', 'dex_contract'))} BLD")

    # --- Step 8: Review event log ---
    print("\n--- Step 8: Event Log ---")
    print("All events emitted (simulates reading blockchain logs):\n")
    for i, event in enumerate(token.get_events()):
        print(f"  [{i}] {event}")

    # --- Step 9: Final state ---
    print("\n--- Final State ---")
    print(f"  {token}")
    accounts = ["deployer", "alice", "liquidity_pool", "dex_contract", "bob"]
    for acct in accounts:
        bal = token.balance_of(acct)
        if bal > 0:
            print(f"  {acct:20s}: {token.format_amount(bal)} BLD")

    # Verify invariant: sum of all balances == total supply
    all_balances = sum(token._balances.values())
    assert all_balances == token.total_supply(), "INVARIANT VIOLATED: balances != supply"
    print(f"\n  Invariant check PASSED: sum(balances) == totalSupply == {token.format_amount(token.total_supply())} BLD")


if __name__ == "__main__":
    main()
