"""
Day 038: Staking Contract — Synthetix-style ERC-20 Staking with Reward Accumulator

This implements the exact math used by Synthetix's StakingRewards contract,
the most forked staking contract in DeFi. The key insight is the O(1) reward
distribution using a global accumulator — no iteration over stakers needed.

Reference: https://github.com/Synthetix-io/synthetix/blob/develop/contracts/StakingRewards.sol
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Simple ERC-20 Token Simulation
# =============================================================================

class Token:
    """Minimal ERC-20 token simulation for staking and reward tokens."""

    def __init__(self, name: str, symbol: str, initial_supply: int, owner: str) -> None:
        self.name = name
        self.symbol = symbol
        # balances maps address -> amount (in smallest unit, like wei)
        self.balances: dict[str, int] = {owner: initial_supply}
        self.total_supply = initial_supply

    def balance_of(self, account: str) -> int:
        return self.balances.get(account, 0)

    def transfer(self, sender: str, recipient: str, amount: int) -> None:
        """Transfer tokens. Reverts (raises) if insufficient balance."""
        if self.balances.get(sender, 0) < amount:
            raise ValueError(
                f"Insufficient balance: {sender} has {self.balances.get(sender, 0)}, "
                f"needs {amount}"
            )
        if amount < 0:
            raise ValueError("Transfer amount must be non-negative")
        self.balances[sender] = self.balances.get(sender, 0) - amount
        self.balances[recipient] = self.balances.get(recipient, 0) + amount

    def mint(self, to: str, amount: int) -> None:
        """Mint new tokens (for testing/reward distribution)."""
        self.balances[to] = self.balances.get(to, 0) + amount
        self.total_supply += amount


# =============================================================================
# Staking Contract — Synthetix RewardPerToken Pattern
# =============================================================================

class StakingContract:
    """
    A Synthetix-style staking rewards contract.

    Users stake `staking_token` and earn `reward_token` over time.
    Rewards are distributed at a fixed rate over a configurable duration.
    The reward-per-token accumulator ensures O(1) updates regardless of
    the number of stakers.
    """

    def __init__(
        self,
        staking_token: Token,
        reward_token: Token,
        owner: str,
    ) -> None:
        self.staking_token = staking_token
        self.reward_token = reward_token
        self.owner = owner  # Admin who can notify reward amounts
        self.contract_address = "staking_contract"  # Simulated contract address

        # --- Global staking state ---
        self.total_staked: int = 0

        # --- Reward distribution state ---
        # reward_rate: reward tokens distributed per second (scaled by 1e18 for precision)
        self.reward_rate: int = 0
        self.reward_duration: int = 0  # Duration in seconds
        self.period_finish: int = 0    # Timestamp when current reward period ends
        self.last_update_time: int = 0 # Last time rewards were calculated

        # The global accumulator: cumulative reward per staked token (scaled by 1e18)
        # This is THE key variable — it tracks how much reward each unit of stake
        # has earned since the contract was deployed
        self.reward_per_token_stored: int = 0

        # --- Per-user state ---
        self.staked_balances: dict[str, int] = {}
        # Snapshot of reward_per_token when user last interacted
        self.user_reward_per_token_paid: dict[str, int] = {}
        # Accumulated but unclaimed rewards
        self.rewards: dict[str, int] = {}

        # --- Time simulation ---
        self._current_time: int = 0

    # =========================================================================
    # Time simulation (replaces block.timestamp in Solidity)
    # =========================================================================

    def set_time(self, timestamp: int) -> None:
        """Advance the simulated clock. Time can only move forward."""
        if timestamp < self._current_time:
            raise ValueError("Time cannot go backwards")
        self._current_time = timestamp

    def advance_time(self, seconds: int) -> None:
        """Move clock forward by `seconds`."""
        self._current_time += seconds

    @property
    def current_time(self) -> int:
        return self._current_time

    # =========================================================================
    # Core reward math — the heart of the contract
    # =========================================================================

    def _last_time_reward_applicable(self) -> int:
        """
        Returns min(now, periodFinish).

        After the reward period ends, no more rewards accrue. This prevents
        the accumulator from growing beyond the budgeted amount.
        """
        return min(self._current_time, self.period_finish)

    def _reward_per_token(self) -> int:
        """
        Calculate the current reward-per-token accumulator value.

        Formula:
            rewardPerToken += (timeDelta * rewardRate * 1e18) / totalStaked

        We scale by 1e18 to maintain precision with integer arithmetic
        (Solidity has no floats). This is the standard DeFi precision pattern.

        Returns the stored value if no tokens are staked (avoids division by zero).
        When totalStaked is 0, no one earns rewards — they effectively "leak"
        (are unclaimable for that period). Some protocols send leaked rewards
        to a treasury instead.
        """
        if self.total_staked == 0:
            return self.reward_per_token_stored

        time_delta = self._last_time_reward_applicable() - self.last_update_time

        # rewardRate is already in reward-tokens-per-second
        # Multiply by 1e18 for precision, divide by totalStaked
        additional = (time_delta * self.reward_rate * 10**18) // self.total_staked

        return self.reward_per_token_stored + additional

    def _earned(self, account: str) -> int:
        """
        Calculate pending (unclaimed) rewards for an account.

        earned = stakedBalance * (rewardPerToken - userRewardPerTokenPaid) / 1e18
                 + previouslyStoredRewards

        The first term captures rewards earned since the user's last interaction.
        The second term carries forward rewards from previous interactions that
        haven't been claimed yet.
        """
        balance = self.staked_balances.get(account, 0)
        per_token_diff = self._reward_per_token() - self.user_reward_per_token_paid.get(account, 0)

        # Divide by 1e18 to reverse the precision scaling
        new_rewards = (balance * per_token_diff) // 10**18
        stored_rewards = self.rewards.get(account, 0)

        return new_rewards + stored_rewards

    def _update_reward(self, account: Optional[str]) -> None:
        """
        The critical state update — must be called BEFORE any balance change.

        This is the equivalent of Solidity's `modifier updateReward(account)`.
        It snapshots the current accumulator state so that subsequent balance
        changes don't retroactively affect past reward calculations.

        Order matters:
        1. Update global accumulator (rewardPerTokenStored)
        2. Update last_update_time
        3. If user specified: snapshot their earned rewards and paid checkpoint
        """
        # Step 1: Capture current accumulator value
        self.reward_per_token_stored = self._reward_per_token()

        # Step 2: Advance the time checkpoint
        self.last_update_time = self._last_time_reward_applicable()

        # Step 3: Update user-specific state
        if account is not None:
            self.rewards[account] = self._earned(account)
            self.user_reward_per_token_paid[account] = self.reward_per_token_stored

    # =========================================================================
    # User-facing operations
    # =========================================================================

    def stake(self, user: str, amount: int) -> None:
        """
        Stake tokens into the contract.

        Flow:
        1. Update reward state (checkpoint existing rewards before balance changes)
        2. Transfer staking tokens from user to contract
        3. Update balances

        The order is checks-effects-interactions:
        - Check: amount > 0, user has sufficient balance (enforced by transfer)
        - Effect: update balances and total
        - Interaction: token transfer (in Solidity, this would be the external call)
        """
        if amount <= 0:
            raise ValueError("Cannot stake zero or negative amount")

        # CRITICAL: Update rewards BEFORE changing balances
        self._update_reward(user)

        # Transfer staking tokens: user -> contract
        self.staking_token.transfer(user, self.contract_address, amount)

        # Update state
        self.staked_balances[user] = self.staked_balances.get(user, 0) + amount
        self.total_staked += amount

        print(f"  [Stake] {user} staked {amount} tokens (total staked: {self.total_staked})")

    def withdraw(self, user: str, amount: int) -> None:
        """
        Withdraw staked tokens.

        Same pattern: update rewards first, then modify state, then transfer.
        """
        if amount <= 0:
            raise ValueError("Cannot withdraw zero or negative amount")
        if self.staked_balances.get(user, 0) < amount:
            raise ValueError(f"{user} only has {self.staked_balances.get(user, 0)} staked")

        # CRITICAL: Update rewards BEFORE changing balances
        self._update_reward(user)

        # Update state first (checks-effects before interaction)
        self.staked_balances[user] -= amount
        self.total_staked -= amount

        # Transfer staking tokens: contract -> user
        self.staking_token.transfer(self.contract_address, user, amount)

        print(f"  [Withdraw] {user} withdrew {amount} tokens (total staked: {self.total_staked})")

    def claim_reward(self, user: str) -> int:
        """
        Claim all pending reward tokens.

        Returns the amount claimed (useful for logging/testing).
        """
        self._update_reward(user)

        reward = self.rewards.get(user, 0)
        if reward > 0:
            self.rewards[user] = 0
            # Transfer reward tokens: contract -> user
            self.reward_token.transfer(self.contract_address, user, reward)
            print(f"  [Claim] {user} claimed {reward} reward tokens")

        return reward

    def exit(self, user: str) -> int:
        """
        Convenience: withdraw all staked tokens and claim rewards in one call.
        """
        amount = self.staked_balances.get(user, 0)
        if amount > 0:
            self.withdraw(user, amount)
        return self.claim_reward(user)

    # =========================================================================
    # Admin operations
    # =========================================================================

    def notify_reward_amount(self, reward_amount: int, duration: int) -> None:
        """
        Start or extend a reward distribution period.

        Called by the contract owner to fund the reward pool.

        If called mid-period, remaining undistributed rewards are folded into
        the new period to prevent waste:
            leftover = (periodFinish - now) * rewardRate
            newRate = (leftover + newRewards) / newDuration

        This is how protocols "top up" staking rewards without losing
        the tail end of the current distribution.
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")

        # Update state to checkpoint current period
        self._update_reward(None)

        # Transfer reward tokens to the contract
        self.reward_token.transfer(self.owner, self.contract_address, reward_amount)

        if self._current_time >= self.period_finish:
            # No active period — simple case
            self.reward_rate = reward_amount // duration
        else:
            # Mid-period: fold remaining rewards into new period
            remaining_time = self.period_finish - self._current_time
            leftover = remaining_time * self.reward_rate
            self.reward_rate = (leftover + reward_amount) // duration

        self.reward_duration = duration
        self.last_update_time = self._current_time
        self.period_finish = self._current_time + duration

        # Sanity check: contract must have enough reward tokens to cover the period
        # In production, you'd verify: reward_rate * duration <= contract balance
        print(
            f"  [Admin] Reward period started: {reward_amount} tokens over {duration}s "
            f"(rate: {self.reward_rate}/s, ends at t={self.period_finish})"
        )

    # =========================================================================
    # View functions (read-only)
    # =========================================================================

    def earned(self, account: str) -> int:
        """Public view: how much reward has this account earned (unclaimed)?"""
        return self._earned(account)

    def staked_balance(self, account: str) -> int:
        """Public view: how much has this account staked?"""
        return self.staked_balances.get(account, 0)

    def get_reward_for_duration(self) -> int:
        """Total rewards distributed over the full reward duration."""
        return self.reward_rate * self.reward_duration

    def status(self) -> str:
        """Human-readable contract status."""
        active = self._current_time < self.period_finish
        return (
            f"StakingContract Status @ t={self._current_time}\n"
            f"  Total staked: {self.total_staked}\n"
            f"  Reward rate: {self.reward_rate}/s\n"
            f"  Period: {'ACTIVE' if active else 'ENDED'} "
            f"(ends at t={self.period_finish})\n"
            f"  RewardPerToken: {self.reward_per_token_stored}"
        )


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STAKING CONTRACT DEMONSTRATION")
    print("Synthetix-style Reward Accumulator Pattern")
    print("=" * 70)

    # --- Setup tokens ---
    # Staking token: users stake this (e.g., LP tokens)
    staking_token = Token("Stake Token", "STK", initial_supply=1_000_000, owner="deployer")
    # Reward token: users earn this (e.g., protocol governance token)
    reward_token = Token("Reward Token", "RWD", initial_supply=1_000_000, owner="deployer")

    # Distribute staking tokens to users
    staking_token.transfer("deployer", "alice", 10_000)
    staking_token.transfer("deployer", "bob", 10_000)
    staking_token.transfer("deployer", "charlie", 10_000)

    # Deploy staking contract
    contract = StakingContract(staking_token, reward_token, owner="deployer")

    # --- Scenario 1: Basic staking with two users ---
    print("\n--- Scenario 1: Two users stake, earn proportional rewards ---")

    # Fund the reward pool: 10,000 reward tokens over 100 seconds
    contract.set_time(0)
    contract.notify_reward_amount(reward_amount=10_000, duration=100)
    # reward_rate = 10_000 / 100 = 100 tokens/second

    # Alice stakes 1000 tokens at t=0
    contract.stake("alice", 1000)

    # Bob stakes 3000 tokens at t=0
    contract.stake("bob", 3000)

    # Alice has 25% of the pool, Bob has 75%
    # After 100 seconds: Alice should earn ~2500, Bob should earn ~7500

    # Advance to t=50 (halfway)
    contract.advance_time(50)
    print(f"\n  At t=50:")
    print(f"    Alice earned: {contract.earned('alice')} (expected ~1250)")
    print(f"    Bob earned:   {contract.earned('bob')} (expected ~3750)")

    # Advance to t=100 (period ends)
    contract.advance_time(50)
    print(f"\n  At t=100 (period ended):")
    print(f"    Alice earned: {contract.earned('alice')} (expected ~2500)")
    print(f"    Bob earned:   {contract.earned('bob')} (expected ~7500)")

    # Claim rewards
    alice_reward = contract.claim_reward("alice")
    bob_reward = contract.claim_reward("bob")
    print(f"\n  Total distributed: {alice_reward + bob_reward} (budget was 10,000)")
    # Small rounding error due to integer division is expected

    # --- Scenario 2: Mid-period staking ---
    print("\n\n--- Scenario 2: User joins mid-period ---")

    # New reward period: 10,000 tokens over 100 seconds starting at t=100
    contract.notify_reward_amount(reward_amount=10_000, duration=100)

    # Alice is still staked (1000 tokens). Bob withdraws.
    contract.withdraw("bob", 3000)

    # At t=100: Alice is the only staker with 1000 tokens
    # She earns ALL rewards: 100 tokens/second

    # Charlie joins at t=150 with 1000 tokens
    contract.advance_time(50)
    print(f"\n  At t=150 (before Charlie):")
    print(f"    Alice earned: {contract.earned('alice')}")
    # Alice earned 50 * 100 = 5000 (she was sole staker)

    contract.stake("charlie", 1000)
    # Now Alice and Charlie each have 1000 staked — 50/50 split

    # Advance to t=200 (period ends)
    contract.advance_time(50)
    print(f"\n  At t=200 (period ended):")
    print(f"    Alice earned:   {contract.earned('alice')}")
    print(f"    Charlie earned: {contract.earned('charlie')}")
    # Alice: 5000 (solo) + 2500 (split) = 7500
    # Charlie: 0 (wasn't staked) + 2500 (split) = 2500

    contract.claim_reward("alice")
    contract.claim_reward("charlie")

    # --- Scenario 3: Reward period extension ---
    print("\n\n--- Scenario 3: Top-up rewards mid-period ---")

    # Start period: 5000 tokens over 100 seconds at t=200
    contract.notify_reward_amount(reward_amount=5_000, duration=100)
    # rate = 50/s, ends at t=300

    contract.advance_time(50)
    # At t=250: 50 seconds left, 50*50 = 2500 remaining

    print(f"\n  At t=250, adding 7500 more rewards for another 100s:")
    contract.notify_reward_amount(reward_amount=7_500, duration=100)
    # leftover = 50 * 50 = 2500
    # new rate = (2500 + 7500) / 100 = 100/s
    # new end = t=350

    contract.advance_time(100)
    print(f"\n  At t=350 (extended period ended):")
    print(f"    Alice earned:   {contract.earned('alice')}")
    print(f"    Charlie earned: {contract.earned('charlie')}")

    # Both still have 1000 staked (50/50), so each earns half
    # First 50s: rate=50/s, 1250 each. Next 100s: rate=100/s, 5000 each.
    # Total each: 1250 + 5000 = 6250

    contract.exit("alice")
    contract.exit("charlie")

    # --- Scenario 4: Edge case — no stakers during reward period ---
    print("\n\n--- Scenario 4: Rewards leak when no one is staked ---")

    contract.notify_reward_amount(reward_amount=5_000, duration=100)
    # Nobody is staked! Rewards are emitted but nobody earns them.

    contract.advance_time(50)
    # Alice stakes halfway through
    contract.stake("alice", 1000)

    contract.advance_time(50)
    print(f"\n  At t={contract.current_time} (period ended):")
    print(f"    Alice earned: {contract.earned('alice')}")
    # Alice only earns for the last 50 seconds: 50 * 50 = 2500
    # The first 2500 tokens are "leaked" — unclaimable
    print(f"    Leaked rewards: ~{5000 - contract.earned('alice')} tokens (no stakers for first 50s)")

    contract.exit("alice")

    # --- Final balances ---
    print("\n\n--- Final Token Balances ---")
    for user in ["alice", "bob", "charlie"]:
        print(
            f"  {user}: "
            f"{staking_token.balance_of(user)} STK, "
            f"{reward_token.balance_of(user)} RWD"
        )
    print(
        f"  contract: "
        f"{staking_token.balance_of('staking_contract')} STK, "
        f"{reward_token.balance_of('staking_contract')} RWD (leaked/rounding)"
    )

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("  1. Reward accumulator makes distribution O(1) per interaction")
    print("  2. Rewards are proportional to stake amount AND duration")
    print("  3. Mid-period joins only earn from their entry point forward")
    print("  4. Rewards 'leak' when totalStaked == 0 (no one to earn them)")
    print("  5. Integer division causes small rounding losses (< 1 token)")
    print("=" * 70)
