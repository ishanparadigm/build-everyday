"""
Day 038: Staking Contract — Your Implementation

Implement a Synthetix-style ERC-20 staking rewards contract.

Key concepts to nail:
- The reward-per-token accumulator (O(1) distribution)
- Time-weighted rewards (amount * duration)
- Checks-effects-interactions ordering
- Handling zero total supply and mid-period changes

Run tests: python3 -m pytest tests.py -v
"""

from __future__ import annotations
from typing import Optional


class Token:
    """Minimal ERC-20 token simulation."""

    def __init__(self, name: str, symbol: str, initial_supply: int, owner: str) -> None:
        """
        Create a token with `initial_supply` assigned to `owner`.

        Hint: You need a dict to track balances per address.
        """
        raise NotImplementedError("TODO: implement this")

    def balance_of(self, account: str) -> int:
        """Return the token balance of `account`."""
        raise NotImplementedError("TODO: implement this")

    def transfer(self, sender: str, recipient: str, amount: int) -> None:
        """
        Transfer `amount` tokens from `sender` to `recipient`.
        Raise ValueError if sender has insufficient balance or amount is negative.
        """
        raise NotImplementedError("TODO: implement this")

    def mint(self, to: str, amount: int) -> None:
        """Mint `amount` new tokens to address `to`."""
        raise NotImplementedError("TODO: implement this")


class StakingContract:
    """
    Synthetix-style staking rewards contract.

    Users stake `staking_token` and earn `reward_token` proportionally over time.

    The key insight: maintain a global accumulator `reward_per_token_stored` that
    tracks cumulative rewards per unit of stake. Each user snapshots this value
    on every interaction, so their earned rewards are:

        earned = balance * (current_accumulator - user_snapshot) / 1e18 + stored_rewards

    Think about: why do we scale by 1e18? What happens if we don't?
    """

    def __init__(
        self,
        staking_token: Token,
        reward_token: Token,
        owner: str,
    ) -> None:
        """
        Initialize the staking contract.

        You need to track:
        - Global: total_staked, reward_rate, period_finish, last_update_time,
                  reward_per_token_stored
        - Per-user: staked_balances, user_reward_per_token_paid, rewards (earned)
        - Meta: staking_token, reward_token, owner, contract_address, _current_time
        """
        raise NotImplementedError("TODO: implement this")

    # --- Time simulation ---

    def set_time(self, timestamp: int) -> None:
        """Set the simulated clock. Time can only move forward."""
        raise NotImplementedError("TODO: implement this")

    def advance_time(self, seconds: int) -> None:
        """Move the clock forward by `seconds`."""
        raise NotImplementedError("TODO: implement this")

    @property
    def current_time(self) -> int:
        """Return the current simulated timestamp."""
        raise NotImplementedError("TODO: implement this")

    # --- Core reward math ---
    # Hint: These three methods ARE the algorithm. Get these right and everything works.

    def _last_time_reward_applicable(self) -> int:
        """
        Return min(now, period_finish).

        Why: After the reward period ends, no more rewards should accrue.
        Without this cap, the accumulator would keep growing forever.
        """
        raise NotImplementedError("TODO: implement this")

    def _reward_per_token(self) -> int:
        """
        Calculate the current value of the reward-per-token accumulator.

        Formula: stored + (timeDelta * rewardRate * 1e18) / totalStaked

        Hint: What should you return if totalStaked == 0? Think about what
        happens to rewards during a period with no stakers.
        """
        raise NotImplementedError("TODO: implement this")

    def _earned(self, account: str) -> int:
        """
        Calculate total unclaimed rewards for an account.

        Formula: balance * (rewardPerToken - userPaid) / 1e18 + storedRewards

        Hint: The division by 1e18 undoes the scaling from _reward_per_token.
        """
        raise NotImplementedError("TODO: implement this")

    def _update_reward(self, account: Optional[str]) -> None:
        """
        Checkpoint reward state. MUST be called before any balance change.

        Steps:
        1. Update reward_per_token_stored to current value
        2. Update last_update_time
        3. If account is not None: snapshot their earned rewards and paid checkpoint

        Hint: This is the most important method. If you change a user's balance
        without calling this first, their reward calculation will be wrong because
        it will apply the new balance retroactively.
        """
        raise NotImplementedError("TODO: implement this")

    # --- User operations ---

    def stake(self, user: str, amount: int) -> None:
        """
        Stake `amount` tokens from `user` into the contract.

        Order: update_reward -> transfer tokens -> update balances
        Hint: Checks-effects-interactions pattern.
        """
        raise NotImplementedError("TODO: implement this")

    def withdraw(self, user: str, amount: int) -> None:
        """
        Withdraw `amount` staked tokens back to `user`.

        Order: update_reward -> update balances -> transfer tokens
        """
        raise NotImplementedError("TODO: implement this")

    def claim_reward(self, user: str) -> int:
        """
        Claim all pending rewards for `user`. Returns amount claimed.

        Order: update_reward -> zero out rewards -> transfer reward tokens
        """
        raise NotImplementedError("TODO: implement this")

    def exit(self, user: str) -> int:
        """Withdraw all staked tokens and claim rewards. Returns reward amount."""
        raise NotImplementedError("TODO: implement this")

    # --- Admin operations ---

    def notify_reward_amount(self, reward_amount: int, duration: int) -> None:
        """
        Start or extend a reward distribution period.

        If mid-period: fold remaining rewards into the new period.
            leftover = (period_finish - now) * reward_rate
            new_rate = (leftover + reward_amount) / duration

        Hint: Don't forget to transfer reward tokens and update all time vars.
        """
        raise NotImplementedError("TODO: implement this")

    # --- View functions ---

    def earned(self, account: str) -> int:
        """Public view: unclaimed rewards for account."""
        return self._earned(account)

    def staked_balance(self, account: str) -> int:
        """Public view: staked token balance for account."""
        raise NotImplementedError("TODO: implement this")

    def get_reward_for_duration(self) -> int:
        """Total rewards over the full reward duration."""
        raise NotImplementedError("TODO: implement this")


if __name__ == "__main__":
    # Quick smoke test — will fail until you implement the classes
    staking_token = Token("Stake", "STK", 100_000, "admin")
    reward_token = Token("Reward", "RWD", 100_000, "admin")

    staking_token.transfer("admin", "alice", 10_000)

    contract = StakingContract(staking_token, reward_token, owner="admin")
    contract.set_time(0)
    contract.notify_reward_amount(reward_amount=10_000, duration=100)

    contract.stake("alice", 1_000)
    contract.advance_time(100)

    print(f"Alice earned: {contract.earned('alice')}")
    print(f"Expected: ~10000 (she was the only staker)")

    claimed = contract.claim_reward("alice")
    print(f"Claimed: {claimed}")
    print("All checks passed!")
