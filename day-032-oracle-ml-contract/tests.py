"""
Day 032: Tests for Oracle ML Contract

Run with:
    python3 -m pytest tests.py -v
    python3 tests.py
"""

import hashlib
import math
import os
import unittest

from my_solution import (
    LogisticRegressionModel,
    OracleContract,
    OracleReport,
    LendingContract,
    Loan,
    train_risk_model,
    demonstrate_frontrun_vulnerability,
)


class TestLogisticRegressionModel(unittest.TestCase):
    """Test the ML credit risk model."""

    def setUp(self):
        self.model = LogisticRegressionModel(
            weights=[2.0, 1.5, -1.0, 1.0, -0.5],
            bias=-1.0
        )
        self.feature_names = [
            "credit_utilization", "num_defaults", "account_age_norm",
            "loan_to_value", "income_norm"
        ]

    def test_sigmoid_bounds(self):
        """Sigmoid output must be in (0, 1) for any input."""
        self.assertAlmostEqual(self.model.sigmoid(0), 0.5, places=5)
        self.assertGreater(self.model.sigmoid(-500), 0)
        self.assertLess(self.model.sigmoid(500), 1)
        self.assertGreater(self.model.sigmoid(100), 0.99)
        self.assertLess(self.model.sigmoid(-100), 0.01)

    def test_predict_probability_range(self):
        """P(default) must be in [0, 1]."""
        features = [0.5, 1, 0.5, 0.8, 0.5]
        prob = self.model.predict_probability(features)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_risk_bps_range(self):
        """Risk score must be 0-10000 basis points."""
        features = [0.5, 1, 0.5, 0.8, 0.5]
        bps = self.model.predict_risk_bps(features)
        self.assertGreaterEqual(bps, 0)
        self.assertLessEqual(bps, 10000)

    def test_high_risk_vs_low_risk(self):
        """A risky borrower should have a higher score than a safe one."""
        safe = [0.1, 0, 1.0, 0.3, 0.9]      # Low util, no defaults, old, low LTV, high income
        risky = [0.9, 3, 0.1, 1.5, 0.1]      # High util, many defaults, new, high LTV, low income
        self.assertGreater(
            self.model.predict_risk_bps(risky),
            self.model.predict_risk_bps(safe)
        )

    def test_explain_returns_all_features(self):
        """Explain should return contributions for all features plus bias."""
        features = [0.5, 1, 0.5, 0.8, 0.5]
        contribs = self.model.explain(features, self.feature_names)
        for name in self.feature_names:
            self.assertIn(name, contribs)
        self.assertIn("bias", contribs)
        # Verify contribution calculation
        self.assertAlmostEqual(contribs["credit_utilization"], 2.0 * 0.5)


class TestTrainRiskModel(unittest.TestCase):
    """Test that training produces a sensible model."""

    def test_trained_model_scores_correctly(self):
        """After training, the model should rank borrowers correctly."""
        model = train_risk_model()
        safe = [0.1, 0, 1.0, 0.3, 0.9]
        risky = [0.9, 3, 0.1, 1.8, 0.1]
        self.assertGreater(
            model.predict_risk_bps(risky),
            model.predict_risk_bps(safe)
        )

    def test_trained_model_has_correct_weight_signs(self):
        """Credit utilization and defaults should increase risk (positive weight)."""
        model = train_risk_model()
        # credit_utilization weight should be positive (increases risk)
        self.assertGreater(model.weights[0], 0)
        # num_defaults weight should be positive
        self.assertGreater(model.weights[1], 0)


class TestOracleContract(unittest.TestCase):
    """Test the oracle commit-reveal mechanism."""

    def setUp(self):
        self.base_time = 1000000.0
        self.oracle = OracleContract(owner="deployer")
        self.oracle.add_reporter("r1", caller="deployer")
        self.oracle.add_reporter("r2", caller="deployer")

    def test_only_owner_adds_reporters(self):
        """Non-owner cannot add reporters."""
        with self.assertRaises(AssertionError):
            self.oracle.add_reporter("r3", caller="random_user")

    def test_commit_reveal_happy_path(self):
        """Full commit-reveal cycle should produce correct aggregated score."""
        round_id = self.oracle.start_round(caller="deployer")
        self.oracle.commit_deadline = self.base_time + 60
        self.oracle.reveal_deadline = self.base_time + 120

        value = 1500
        salt = "test_salt_123"
        ch = OracleContract.compute_commit_hash(value, salt, "r1")
        self.oracle.commit(ch, "r1", round_id, current_time=self.base_time + 10)

        self.oracle.reveal(value, salt, "borrower_a", "r1", round_id,
                           current_time=self.base_time + 70)

        score = self.oracle.aggregate("borrower_a", round_id,
                                      current_time=self.base_time + 130)
        self.assertEqual(score, 1500)

    def test_commit_hash_fraud_detection(self):
        """Revealing a different value than committed should fail."""
        round_id = self.oracle.start_round(caller="deployer")
        self.oracle.commit_deadline = self.base_time + 60
        self.oracle.reveal_deadline = self.base_time + 120

        committed_value = 3000
        cheat_value = 500
        salt = "my_salt"
        ch = OracleContract.compute_commit_hash(committed_value, salt, "r1")
        self.oracle.commit(ch, "r1", round_id, current_time=self.base_time + 10)

        with self.assertRaises(AssertionError):
            self.oracle.reveal(cheat_value, salt, "borrower", "r1", round_id,
                               current_time=self.base_time + 70)

    def test_median_aggregation(self):
        """Median of multiple reports should be robust to outliers."""
        round_id = self.oracle.start_round(caller="deployer")
        self.oracle.add_reporter("r3", caller="deployer")
        self.oracle.commit_deadline = self.base_time + 60
        self.oracle.reveal_deadline = self.base_time + 120

        # Three reporters: 1000, 1050, 9999 (one outlier)
        for reporter, value in [("r1", 1000), ("r2", 1050), ("r3", 9999)]:
            salt = f"salt_{reporter}"
            ch = OracleContract.compute_commit_hash(value, salt, reporter)
            self.oracle.commit(ch, reporter, round_id, current_time=self.base_time + 10)
            self.oracle.reveal(value, salt, "bob", reporter, round_id,
                               current_time=self.base_time + 70)

        score = self.oracle.aggregate("bob", round_id, current_time=self.base_time + 130)
        self.assertEqual(score, 1050)  # Median, not skewed by 9999

    def test_staleness_check(self):
        """Score older than MAX_STALENESS should be rejected."""
        round_id = self.oracle.start_round(caller="deployer")
        self.oracle.commit_deadline = self.base_time + 60
        self.oracle.reveal_deadline = self.base_time + 120

        salt = "s"
        ch = OracleContract.compute_commit_hash(2000, salt, "r1")
        self.oracle.commit(ch, "r1", round_id, current_time=self.base_time + 10)
        self.oracle.reveal(2000, salt, "stale_user", "r1", round_id,
                           current_time=self.base_time + 70)
        agg_time = self.base_time + 130
        self.oracle.aggregate("stale_user", round_id, current_time=agg_time)

        # Within staleness window — should work
        self.oracle.get_risk_score("stale_user", current_time=agg_time + 100)

        # Beyond staleness window — should fail
        with self.assertRaises(AssertionError):
            self.oracle.get_risk_score(
                "stale_user",
                current_time=agg_time + self.oracle.MAX_STALENESS + 1
            )


class TestLendingContract(unittest.TestCase):
    """Test the lending contract's use of oracle data."""

    def setUp(self):
        self.base_time = 1000000.0
        self.oracle = OracleContract(owner="deployer")
        self.oracle.add_reporter("r1", caller="deployer")
        self.lending = LendingContract(oracle=self.oracle, owner="deployer")

        # Set up a valid oracle score for "good_borrower"
        round_id = self.oracle.start_round(caller="deployer")
        self.oracle.commit_deadline = self.base_time + 60
        self.oracle.reveal_deadline = self.base_time + 120
        salt = "lending_test"
        score = 1500
        ch = OracleContract.compute_commit_hash(score, salt, "r1")
        self.oracle.commit(ch, "r1", round_id, current_time=self.base_time + 10)
        self.oracle.reveal(score, salt, "good_borrower", "r1", round_id,
                           current_time=self.base_time + 70)
        self.agg_time = self.base_time + 130
        self.oracle.aggregate("good_borrower", round_id, current_time=self.agg_time)

    def test_interest_rate_low_risk(self):
        """Low risk borrower should get a low interest rate."""
        rate = self.lending.compute_interest_rate(1000)
        expected = 200 + int(1000 * 0.5)  # base + tier1
        self.assertEqual(rate, expected)

    def test_interest_rate_high_risk(self):
        """Higher risk should result in higher rate."""
        rate_low = self.lending.compute_interest_rate(1000)
        rate_high = self.lending.compute_interest_rate(4000)
        self.assertGreater(rate_high, rate_low)

    def test_deny_excessive_risk(self):
        """Risk above MAX_RISK_BPS should be denied."""
        with self.assertRaises(ValueError):
            self.lending.compute_interest_rate(6000)

    def test_loan_approval(self):
        """Loan should be approved with valid oracle score and sufficient collateral."""
        self.lending.deposit_collateral("good_borrower", 12000)
        loan = self.lending.request_loan("good_borrower", 10000,
                                         current_time=self.agg_time + 10)
        self.assertEqual(loan.principal, 10000)
        self.assertEqual(loan.risk_score_bps, 1500)
        self.assertGreater(loan.interest_rate_bps, 0)

    def test_insufficient_collateral(self):
        """Loan should be denied if collateral is insufficient."""
        self.lending.deposit_collateral("good_borrower", 100)  # Way too little
        with self.assertRaises(AssertionError):
            self.lending.request_loan("good_borrower", 10000,
                                      current_time=self.agg_time + 10)


class TestFrontRunDemo(unittest.TestCase):
    """Test the front-running demonstration."""

    def test_returns_expected_structure(self):
        """Should return dict with steps and prevention keys."""
        result = demonstrate_frontrun_vulnerability()
        self.assertIn("steps", result)
        self.assertIn("prevention", result)
        self.assertIsInstance(result["steps"], list)
        self.assertIsInstance(result["prevention"], list)
        self.assertGreater(len(result["steps"]), 0)
        self.assertGreater(len(result["prevention"]), 0)


if __name__ == "__main__":
    unittest.main()
