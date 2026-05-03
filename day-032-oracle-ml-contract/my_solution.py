"""
Day 032: Smart Contract That Uses Oracle ML Predictions

YOUR TASK: Implement the full pipeline:
1. ML credit risk model (logistic regression)
2. Oracle contract with commit-reveal
3. Lending contract that uses oracle risk scores
4. Front-running attack demonstration

Key concepts to think about:
- Why can't smart contracts call ML models directly?
- How does commit-reveal prevent information leakage?
- Why use median aggregation instead of mean?
- Why do oracle scores have a staleness window?
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# PART 1: Off-Chain ML Credit Risk Model
# =============================================================================

@dataclass
class LogisticRegressionModel:
    """
    Logistic regression for credit risk scoring.

    Features (in order):
        - credit_utilization: [0, 1]
        - num_past_defaults: [0, inf)
        - account_age_days: normalized [0, 1]
        - loan_to_value: [0, inf)
        - monthly_income: normalized [0, 1]
    """
    weights: list[float]
    bias: float

    def sigmoid(self, z: float) -> float:
        """
        Logistic sigmoid: 1 / (1 + exp(-z))
        Clamp z to [-500, 500] to avoid overflow.

        Hint: This is the same sigmoid from Day 003.
        """
        raise NotImplementedError("TODO: implement sigmoid")

    def predict_probability(self, features: list[float]) -> float:
        """
        Compute P(default) = sigmoid(w^T * x + b).

        Hint: Compute the dot product of weights and features,
        add the bias, then apply sigmoid.
        """
        raise NotImplementedError("TODO: implement predict_probability")

    def predict_risk_bps(self, features: list[float]) -> int:
        """
        Return risk score in basis points (0-10000).
        1 bps = 0.01%. Multiply probability by 10000 and truncate to int.
        """
        raise NotImplementedError("TODO: implement predict_risk_bps")

    def explain(self, features: list[float], feature_names: list[str]) -> dict:
        """
        Return a dict mapping feature_name -> weight * feature_value.
        Include "bias" -> self.bias as well.

        Hint: This is the interpretability advantage of logistic regression.
        """
        raise NotImplementedError("TODO: implement explain")


def train_risk_model() -> LogisticRegressionModel:
    """
    Train logistic regression with gradient descent on synthetic data.

    Steps:
    1. Generate 500 synthetic borrower samples with known risk relationship
    2. Normalize features to [0, 1]
    3. Run gradient descent: for each epoch, compute predictions,
       cross-entropy loss, and update weights with gradient (pred - label) * feature

    Hint: The gradient of binary cross-entropy w.r.t. weights has a
    beautifully simple form: (prediction - label) * feature_value.
    This is because the sigmoid derivative cancels with the cross-entropy derivative.

    Use: lr=0.5, epochs=200, random.seed(42)
    """
    raise NotImplementedError("TODO: implement train_risk_model")


# =============================================================================
# PART 2: Oracle Contract with Commit-Reveal
# =============================================================================

@dataclass
class Commit:
    """A committed (hidden) oracle value awaiting reveal."""
    commit_hash: str
    reporter: str
    timestamp: float
    revealed: bool = False


@dataclass
class OracleReport:
    """A revealed oracle value from a single reporter."""
    reporter: str
    borrower: str
    risk_score_bps: int
    timestamp: float
    round_id: int


@dataclass
class OracleContract:
    """
    Oracle contract with commit-reveal and multi-reporter aggregation.

    Hint: Think about WHY each phase exists:
    - Commit: hide values so reporters can't copy each other
    - Reveal: expose values and verify against commits
    - Aggregate: combine multiple reports into one canonical value
    - Staleness: prevent using outdated information
    """
    owner: str
    authorized_reporters: set[str] = field(default_factory=set)
    commits: dict[tuple[str, int], Commit] = field(default_factory=dict)
    reveals: dict[int, list[OracleReport]] = field(default_factory=dict)
    aggregated_scores: dict[str, tuple[int, float, int]] = field(default_factory=dict)
    current_round: int = 0
    commit_deadline: float = 0.0
    reveal_deadline: float = 0.0

    MAX_STALENESS: float = 3600.0
    COMMIT_WINDOW: float = 60.0
    REVEAL_WINDOW: float = 60.0
    MIN_REPORTERS: int = 1

    def add_reporter(self, reporter: str, caller: str) -> None:
        """Only the owner can authorize new reporters."""
        raise NotImplementedError("TODO: implement add_reporter")

    def start_round(self, caller: str) -> int:
        """
        Begin a new oracle round. Returns the round ID.

        Hint: Increment current_round, set deadlines based on
        COMMIT_WINDOW and REVEAL_WINDOW, initialize reveals list.
        """
        raise NotImplementedError("TODO: implement start_round")

    @staticmethod
    def compute_commit_hash(value_bps: int, salt: str, reporter: str) -> str:
        """
        Hash the value with salt and reporter address.
        Format: SHA-256("{value_bps}||{salt}||{reporter}")

        Hint: Why include the reporter address? To prevent one reporter
        from copying another's commit hash when they happen to submit
        the same value.
        """
        raise NotImplementedError("TODO: implement compute_commit_hash")

    def commit(self, commit_hash: str, reporter: str, round_id: int,
               current_time: Optional[float] = None) -> None:
        """
        Submit a commit for a round.

        Verify: reporter is authorized, round_id is current,
        within commit deadline, hasn't already committed.
        """
        raise NotImplementedError("TODO: implement commit")

    def reveal(self, value_bps: int, salt: str, borrower: str,
               reporter: str, round_id: int,
               current_time: Optional[float] = None) -> None:
        """
        Reveal a previously committed value.

        Verify: hash(value||salt||reporter) matches stored commit.
        This is the core security property — reporters cannot change
        their value after committing.

        Hint: Recompute the hash and compare to the stored commit_hash.
        """
        raise NotImplementedError("TODO: implement reveal")

    def aggregate(self, borrower: str, round_id: int,
                  current_time: Optional[float] = None) -> int:
        """
        Compute median risk score from all reveals in a round.

        Hint: Sort the scores and take the middle value.
        For even N, take the lower of the two middle values (conservative).

        Why median instead of mean? Median is robust to outliers.
        One malicious reporter can't skew the result.
        """
        raise NotImplementedError("TODO: implement aggregate")

    def get_risk_score(self, borrower: str,
                       current_time: Optional[float] = None) -> tuple[int, float, int]:
        """
        Get the latest aggregated risk score.
        Returns (score_bps, timestamp, round_id).

        MUST check staleness: reject if older than MAX_STALENESS.
        """
        raise NotImplementedError("TODO: implement get_risk_score")


# =============================================================================
# PART 3: Lending Contract
# =============================================================================

@dataclass
class Loan:
    """An outstanding loan with its terms."""
    borrower: str
    principal: float
    collateral: float
    interest_rate_bps: int
    risk_score_bps: int
    origination_time: float
    oracle_round_id: int


@dataclass
class LendingContract:
    """
    Lending contract that uses oracle ML risk scores.

    Interest rate curve (piecewise linear):
    - Risk 0-2000 bps:  BASE_RATE + risk * 0.5
    - Risk 2000-5000 bps: BASE_RATE + 1000 + (risk-2000) * 1.5
    - Risk > 5000 bps: DENIED

    Collateral requirements:
    - Risk 0-2000: 110%
    - Risk 2000-5000: 150%
    """
    oracle: OracleContract
    owner: str
    pool_balance: float = 100000.0
    loans: dict[str, Loan] = field(default_factory=dict)
    collateral: dict[str, float] = field(default_factory=dict)

    BASE_RATE_BPS: int = 200
    MAX_RISK_BPS: int = 5000
    RISK_TIER1_CUTOFF: int = 2000
    TIER1_MULTIPLIER: float = 0.5
    TIER2_MULTIPLIER: float = 1.5

    def compute_interest_rate(self, risk_score_bps: int) -> int:
        """
        Map risk score to interest rate using the piecewise linear curve.

        Hint: Check which tier the risk falls in, then apply the formula.
        Raise ValueError if risk exceeds MAX_RISK_BPS.
        """
        raise NotImplementedError("TODO: implement compute_interest_rate")

    def required_collateral_ratio(self, risk_score_bps: int) -> float:
        """Higher risk → more collateral. Returns ratio (e.g., 1.1 = 110%)."""
        raise NotImplementedError("TODO: implement required_collateral_ratio")

    def deposit_collateral(self, borrower: str, amount: float) -> None:
        """Deposit collateral before requesting a loan."""
        raise NotImplementedError("TODO: implement deposit_collateral")

    def request_loan(self, borrower: str, amount: float,
                     current_time: Optional[float] = None) -> Loan:
        """
        Request a loan. Steps:
        1. Fetch risk score from oracle (checks staleness)
        2. Compute interest rate (denies if risk too high)
        3. Verify collateral is sufficient
        4. Issue the loan

        Hint: This is where everything comes together — ML prediction
        flows through the oracle into an autonomous financial decision.
        """
        raise NotImplementedError("TODO: implement request_loan")


# =============================================================================
# PART 4: Front-Running Attack Demonstration
# =============================================================================

def demonstrate_frontrun_vulnerability() -> dict:
    """
    Show how without commit-reveal, oracle updates can be front-run.

    Return a dict with:
    - "steps": list of strings describing the attack
    - "prevention": list of strings explaining how commit-reveal stops it

    Hint: The key insight is that in a naive oracle, the value is visible
    in the mempool before it's mined. With commit-reveal, only the hash
    is visible — and SHA-256 is preimage resistant.
    """
    raise NotImplementedError("TODO: implement demonstrate_frontrun_vulnerability")


# =============================================================================
# MAIN: Test your implementation
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 032: SMART CONTRACT WITH ORACLE ML PREDICTIONS")
    print("=" * 70)

    # Step 1: Train model
    print("\n--- Training ML Risk Model ---")
    model = train_risk_model()
    feature_names = [
        "credit_utilization", "num_defaults", "account_age_norm",
        "loan_to_value", "income_norm"
    ]
    print(f"Weights: {[f'{w:.4f}' for w in model.weights]}")
    print(f"Bias: {model.bias:.4f}")

    # Step 2: Score borrowers
    print("\n--- Scoring Borrowers ---")
    alice_features = [0.2, 0, 1.0, 0.5, 0.8]  # Low risk
    bob_features = [0.5, 1, 0.3, 1.0, 0.4]     # Medium risk
    carol_features = [0.9, 3, 0.1, 1.8, 0.2]   # High risk

    for name, features in [("Alice", alice_features), ("Bob", bob_features), ("Carol", carol_features)]:
        score = model.predict_risk_bps(features)
        prob = model.predict_probability(features)
        print(f"  {name}: P(default)={prob:.4f}, risk={score} bps")

    # Step 3: Oracle commit-reveal
    print("\n--- Oracle Commit-Reveal ---")
    base_time = 1000000.0
    oracle = OracleContract(owner="deployer")
    oracle.add_reporter("reporter_1", caller="deployer")

    round_id = oracle.start_round(caller="deployer")
    oracle.commit_deadline = base_time + oracle.COMMIT_WINDOW
    oracle.reveal_deadline = oracle.commit_deadline + oracle.REVEAL_WINDOW

    alice_score = model.predict_risk_bps(alice_features)
    salt = os.urandom(16).hex()
    ch = OracleContract.compute_commit_hash(alice_score, salt, "reporter_1")
    oracle.commit(ch, "reporter_1", round_id, current_time=base_time + 10)
    print(f"  Committed hash: {ch[:24]}...")

    oracle.reveal(alice_score, salt, "alice", "reporter_1", round_id,
                  current_time=oracle.commit_deadline + 5)
    print(f"  Revealed: {alice_score} bps (verified!)")

    agg_time = oracle.reveal_deadline + 1
    median = oracle.aggregate("alice", round_id, current_time=agg_time)
    print(f"  Aggregated score: {median} bps")

    # Step 4: Lending
    print("\n--- Lending Contract ---")
    lending = LendingContract(oracle=oracle, owner="deployer")
    lending.deposit_collateral("alice", 12000)
    loan = lending.request_loan("alice", 10000, current_time=agg_time + 5)
    print(f"  Loan approved: {loan.principal:.0f} at {loan.interest_rate_bps} bps ({loan.interest_rate_bps/100:.2f}%)")

    # Step 5: Front-running demo
    print("\n--- Front-Running Demo ---")
    result = demonstrate_frontrun_vulnerability()
    for step in result["steps"]:
        print(f"  {step}")
    print("  Prevention:")
    for line in result["prevention"]:
        print(f"    {line}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED — Your implementation works!")
    print("=" * 70)
