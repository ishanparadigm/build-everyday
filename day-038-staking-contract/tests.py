"""
Day 038: Staking Contract Tests

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
from my_solution import Token, StakingContract


class TestToken(unittest.TestCase):
    """Basic ERC-20 token tests."""

    def setUp(self):
        self.token = Token("Test", "TST", 10_000, "owner")

    def test_initial_balance(self):
        self.assertEqual(self.token.balance_of("owner"), 10_000)
        self.assertEqual(self.token.balance_of("nobody"), 0)

    def test_transfer(self):
        self.token.transfer("owner", "alice", 3_000)
        self.assertEqual(self.token.balance_of("owner"), 7_000)
        self.assertEqual(self.token.balance_of("alice"), 3_000)

    def test_transfer_insufficient_balance(self):
        with self.assertRaises(ValueError):
            self.token.transfer("owner", "alice", 20_000)

    def test_transfer_negative(self):
        with self.assertRaises(ValueError):
            self.token.transfer("owner", "alice", -100)


class TestStakingBasic(unittest.TestCase):
    """Core staking mechanics."""

    def setUp(self):
        self.stk = Token("Stake", "STK", 1_000_000, "deployer")
        self.rwd = Token("Reward", "RWD", 1_000_000, "deployer")
        self.stk.transfer("deployer", "alice", 10_000)
        self.stk.transfer("deployer", "bob", 10_000)
        self.contract = StakingContract(self.stk, self.rwd, owner="deployer")
        self.contract.set_time(0)

    def test_stake_updates_balances(self):
        """Staking should move tokens to contract and update staked balance."""
        self.contract.stake("alice", 1_000)
        self.assertEqual(self.contract.staked_balance("alice"), 1_000)
        self.assertEqual(self.contract.total_staked, 1_000)
        self.assertEqual(self.stk.balance_of("alice"), 9_000)

    def test_withdraw_updates_balances(self):
        """Withdrawing should return tokens and update staked balance."""
        self.contract.stake("alice", 1_000)
        self.contract.withdraw("alice", 500)
        self.assertEqual(self.contract.staked_balance("alice"), 500)
        self.assertEqual(self.contract.total_staked, 500)
        self.assertEqual(self.stk.balance_of("alice"), 9_500)

    def test_cannot_withdraw_more_than_staked(self):
        self.contract.stake("alice", 1_000)
        with self.assertRaises(ValueError):
            self.contract.withdraw("alice", 2_000)

    def test_cannot_stake_zero(self):
        with self.assertRaises(ValueError):
            self.contract.stake("alice", 0)


class TestRewardDistribution(unittest.TestCase):
    """Reward accumulator math tests."""

    def setUp(self):
        self.stk = Token("Stake", "STK", 1_000_000, "deployer")
        self.rwd = Token("Reward", "RWD", 1_000_000, "deployer")
        self.stk.transfer("deployer", "alice", 10_000)
        self.stk.transfer("deployer", "bob", 10_000)
        self.contract = StakingContract(self.stk, self.rwd, owner="deployer")
        self.contract.set_time(0)

    def test_single_staker_full_period(self):
        """Solo staker should earn all rewards (minus rounding)."""
        self.contract.notify_reward_amount(10_000, duration=100)
        self.contract.stake("alice", 1_000)
        self.contract.advance_time(100)
        earned = self.contract.earned("alice")
        # Allow small rounding error from integer division
        self.assertAlmostEqual(earned, 10_000, delta=5)

    def test_proportional_rewards(self):
        """Two stakers should earn proportionally to their stake."""
        self.contract.notify_reward_amount(10_000, duration=100)
        self.contract.stake("alice", 1_000)  # 25%
        self.contract.stake("bob", 3_000)    # 75%
        self.contract.advance_time(100)
        alice_earned = self.contract.earned("alice")
        bob_earned = self.contract.earned("bob")
        # Alice: 25% of 10000 = 2500, Bob: 75% of 10000 = 7500
        self.assertAlmostEqual(alice_earned, 2_500, delta=5)
        self.assertAlmostEqual(bob_earned, 7_500, delta=5)

    def test_mid_period_entry(self):
        """User who joins mid-period only earns from entry point."""
        self.contract.notify_reward_amount(10_000, duration=100)
        self.contract.stake("alice", 1_000)
        # Alice earns alone for 50s, then splits with Bob for 50s
        self.contract.advance_time(50)
        self.contract.stake("bob", 1_000)
        self.contract.advance_time(50)

        alice_earned = self.contract.earned("alice")
        bob_earned = self.contract.earned("bob")
        # Alice: 50*100 (solo) + 50*50 (half) = 5000 + 2500 = 7500
        # Bob: 50*50 (half) = 2500
        self.assertAlmostEqual(alice_earned, 7_500, delta=5)
        self.assertAlmostEqual(bob_earned, 2_500, delta=5)

    def test_no_rewards_after_period(self):
        """Rewards should stop accruing after period ends."""
        self.contract.notify_reward_amount(10_000, duration=100)
        self.contract.stake("alice", 1_000)
        self.contract.advance_time(100)
        earned_at_end = self.contract.earned("alice")
        self.contract.advance_time(1000)  # Way past the end
        earned_later = self.contract.earned("alice")
        self.assertEqual(earned_at_end, earned_later)

    def test_claim_transfers_reward_tokens(self):
        """Claiming should transfer reward tokens to the user."""
        self.contract.notify_reward_amount(10_000, duration=100)
        self.contract.stake("alice", 1_000)
        self.contract.advance_time(100)
        initial_balance = self.rwd.balance_of("alice")
        claimed = self.contract.claim_reward("alice")
        self.assertGreater(claimed, 0)
        self.assertEqual(self.rwd.balance_of("alice"), initial_balance + claimed)

    def test_claim_zeroes_pending(self):
        """After claiming, earned should be 0."""
        self.contract.notify_reward_amount(10_000, duration=100)
        self.contract.stake("alice", 1_000)
        self.contract.advance_time(100)
        self.contract.claim_reward("alice")
        self.assertEqual(self.contract.earned("alice"), 0)

    def test_reward_topup_mid_period(self):
        """Adding rewards mid-period should fold remaining into new period."""
        self.contract.notify_reward_amount(5_000, duration=100)
        # rate = 50/s
        self.contract.stake("alice", 1_000)
        self.contract.advance_time(50)
        # 50s left, 50*50 = 2500 remaining
        self.contract.notify_reward_amount(7_500, duration=100)
        # new rate = (2500 + 7500) / 100 = 100/s
        self.assertEqual(self.contract.reward_rate, 100)

    def test_leaked_rewards_zero_supply(self):
        """Rewards during zero totalStaked are unclaimable ('leaked')."""
        self.contract.notify_reward_amount(10_000, duration=100)
        # No one stakes for first 50 seconds
        self.contract.advance_time(50)
        self.contract.stake("alice", 1_000)
        self.contract.advance_time(50)
        earned = self.contract.earned("alice")
        # Alice only earns for last 50s: 50 * 100 = 5000
        self.assertAlmostEqual(earned, 5_000, delta=5)

    def test_exit_withdraws_and_claims(self):
        """Exit should return staked tokens and claim rewards."""
        self.contract.notify_reward_amount(10_000, duration=100)
        self.contract.stake("alice", 1_000)
        self.contract.advance_time(100)
        stk_before = self.stk.balance_of("alice")
        rwd_before = self.rwd.balance_of("alice")
        reward = self.contract.exit("alice")
        self.assertEqual(self.stk.balance_of("alice"), stk_before + 1_000)
        self.assertEqual(self.rwd.balance_of("alice"), rwd_before + reward)
        self.assertEqual(self.contract.staked_balance("alice"), 0)


if __name__ == "__main__":
    unittest.main()
