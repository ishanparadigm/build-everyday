"""
Day 032: Smart Contract That Uses Oracle ML Predictions

A complete system integrating:
1. Off-chain ML credit risk model (logistic regression)
2. Oracle contract with commit-reveal and multi-reporter aggregation
3. Lending contract that uses oracle-delivered risk scores
4. Front-running attack demonstration and prevention

This simulates the full pipeline that DeFi protocols like Aave and Compound
use to integrate external data — except we're feeding ML predictions instead
of just price feeds.
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
# In production, this runs on a server. The model is trained once, then used
# to score borrowers. Only the score (not the model) goes on-chain.

@dataclass
class LogisticRegressionModel:
    """
    Logistic regression for credit risk scoring.

    Why logistic regression instead of a neural network?
    1. Interpretability: each weight tells you how much a feature contributes
       to default risk. Regulators demand this.
    2. Monotonicity: higher credit utilization always means higher risk.
       Neural networks can learn non-monotonic artifacts from noisy data.
    3. Calibration: the sigmoid output is a genuine probability estimate
       when the model is well-calibrated. P(default)=0.3 means ~30% of
       similar borrowers actually defaulted historically.

    Features (in order):
        - credit_utilization: ratio of used credit to total credit limit [0, 1]
        - num_past_defaults: count of historical defaults [0, inf)
        - account_age_days: how long the account has existed [0, inf)
        - loan_to_value: requested loan / collateral value [0, inf)
        - monthly_income: normalized monthly income [0, inf)
    """
    weights: list[float]
    bias: float

    def sigmoid(self, z: float) -> float:
        """
        Logistic sigmoid: maps any real number to (0, 1).

        We clamp z to [-500, 500] to avoid overflow in exp().
        For z < -500, sigmoid ≈ 0. For z > 500, sigmoid ≈ 1.
        """
        z = max(-500.0, min(500.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_probability(self, features: list[float]) -> float:
        """
        Compute P(default) for a borrower.

        The linear combination w^T * x + b captures how each feature
        contributes to default risk. Positive weights increase risk;
        negative weights decrease it. The sigmoid squashes this to a
        probability.
        """
        assert len(features) == len(self.weights), \
            f"Expected {len(self.weights)} features, got {len(features)}"

        # Linear combination: z = w0*x0 + w1*x1 + ... + wn*xn + b
        z = sum(w * x for w, x in zip(self.weights, features)) + self.bias
        return self.sigmoid(z)

    def predict_risk_bps(self, features: list[float]) -> int:
        """
        Return risk score in basis points (0-10000).

        Basis points are the standard for on-chain financial math because
        Solidity has no floating point. 1 basis point = 0.01%.
        10000 bps = 100.00%.

        This quantization loses at most 0.005% precision — negligible
        for lending decisions.
        """
        prob = self.predict_probability(features)
        return int(prob * 10000)

    def explain(self, features: list[float], feature_names: list[str]) -> dict:
        """
        Explain which features contributed most to the risk score.

        For each feature, the contribution is weight * feature_value.
        This is the key advantage of logistic regression — transparent
        attributions. A borrower denied a loan can see exactly why.
        """
        contributions = {}
        for name, w, x in zip(feature_names, self.weights, features):
            contributions[name] = w * x
        contributions["bias"] = self.bias
        return contributions


def train_risk_model() -> LogisticRegressionModel:
    """
    Train a logistic regression model using gradient descent on synthetic data.

    In production, you'd use scikit-learn or a proper ML pipeline. We train
    from scratch here to show what's happening under the hood — this connects
    back to Day 001 (linear regression) and Day 003 (logistic regression).

    The training loop:
    1. Forward pass: compute predictions
    2. Compute binary cross-entropy loss
    3. Backward pass: compute gradients
    4. Update weights with gradient descent
    """
    # --- Generate synthetic training data ---
    # Each row: [credit_util, num_defaults, account_age_days, ltv, monthly_income]
    # We construct data where the true relationship is known, so we can
    # verify the model learns the right patterns.
    import random
    random.seed(42)

    data: list[tuple[list[float], int]] = []
    for _ in range(500):
        credit_util = random.uniform(0, 1)
        num_defaults = random.randint(0, 5)
        account_age = random.uniform(30, 3650)  # 1 month to 10 years
        ltv = random.uniform(0.1, 2.0)
        income = random.uniform(1000, 20000)

        # True default probability — our ground truth
        # High utilization, many defaults, high LTV → higher risk
        # Older accounts, higher income → lower risk
        z_true = (2.0 * credit_util
                  + 1.5 * num_defaults
                  - 0.001 * account_age
                  + 1.0 * ltv
                  - 0.0002 * income
                  - 1.0)
        p_default = 1 / (1 + math.exp(-z_true))
        label = 1 if random.random() < p_default else 0
        data.append(([credit_util, num_defaults, account_age / 3650, ltv, income / 20000], label))

    # --- Normalize features to [0, 1] range ---
    # account_age and income are already normalized in the feature vector above.
    # This is critical for gradient descent convergence — if features have
    # wildly different scales, the loss landscape is elongated and GD oscillates.

    # --- Gradient descent training ---
    n_features = 5
    weights = [0.0] * n_features
    bias = 0.0
    lr = 0.5  # Learning rate — larger than typical because our data is clean
    n_epochs = 200

    for epoch in range(n_epochs):
        # Accumulate gradients over the full dataset (batch gradient descent)
        dw = [0.0] * n_features
        db = 0.0
        total_loss = 0.0

        for features, label in data:
            # Forward pass
            z = sum(w * x for w, x in zip(weights, features)) + bias
            z = max(-500, min(500, z))
            pred = 1 / (1 + math.exp(-z))

            # Binary cross-entropy loss: -[y*log(p) + (1-y)*log(1-p)]
            # Clamp pred to avoid log(0)
            pred_clamped = max(1e-15, min(1 - 1e-15, pred))
            total_loss += -(label * math.log(pred_clamped)
                           + (1 - label) * math.log(1 - pred_clamped))

            # Gradient of BCE w.r.t. weights: (pred - label) * feature
            # This elegant form comes from the derivative of sigmoid composed
            # with cross-entropy — the sigmoid's derivative cancels nicely.
            error = pred - label
            for i in range(n_features):
                dw[i] += error * features[i]
            db += error

        # Average gradients and update
        n = len(data)
        for i in range(n_features):
            weights[i] -= lr * dw[i] / n
        bias -= lr * db / n

    return LogisticRegressionModel(weights=weights, bias=bias)


# =============================================================================
# PART 2: Oracle Contract with Commit-Reveal
# =============================================================================
# The oracle bridges off-chain ML predictions to on-chain contracts.
# Commit-reveal prevents front-running: reporters first commit a hash,
# then reveal the actual value in a separate transaction.

@dataclass
class Commit:
    """A committed (hidden) oracle value awaiting reveal."""
    commit_hash: str       # keccak256(value || salt || reporter)
    reporter: str          # address of the reporter
    timestamp: float       # when the commit was made
    revealed: bool = False


@dataclass
class OracleReport:
    """A revealed oracle value from a single reporter."""
    reporter: str
    borrower: str
    risk_score_bps: int    # 0-10000
    timestamp: float
    round_id: int


@dataclass
class OracleContract:
    """
    Oracle contract that accepts ML predictions via commit-reveal.

    Architecture:
    - Multiple authorized reporters can submit predictions
    - Commit phase: reporters submit hash(value || salt || address)
    - Reveal phase: reporters reveal value + salt, contract verifies
    - Aggregation: median of all reveals in a round becomes the canonical value
    - Staleness: data expires after MAX_STALENESS seconds

    Why median aggregation?
    Mean is vulnerable to outliers — one malicious reporter could skew the
    result dramatically. Median requires >50% of reporters to collude for
    manipulation. This is the same reason Chainlink uses median.
    """
    owner: str
    authorized_reporters: set[str] = field(default_factory=set)
    commits: dict[tuple[str, int], Commit] = field(default_factory=dict)  # (reporter, round_id) -> Commit
    reveals: dict[int, list[OracleReport]] = field(default_factory=dict)  # round_id -> [reports]
    aggregated_scores: dict[str, tuple[int, float, int]] = field(default_factory=dict)  # borrower -> (score, timestamp, round_id)
    current_round: int = 0
    commit_deadline: float = 0.0
    reveal_deadline: float = 0.0

    # Configuration
    MAX_STALENESS: float = 3600.0      # 1 hour — scores older than this are rejected
    COMMIT_WINDOW: float = 60.0        # 60 seconds to submit commits
    REVEAL_WINDOW: float = 60.0        # 60 seconds to reveal after commit window
    MIN_REPORTERS: int = 1             # minimum reveals needed for a valid round

    def add_reporter(self, reporter: str, caller: str) -> None:
        """Only the owner can authorize new reporters."""
        assert caller == self.owner, "Only owner can add reporters"
        self.authorized_reporters.add(reporter)

    def start_round(self, caller: str) -> int:
        """
        Begin a new oracle round. Returns the round ID.

        A round has two phases:
        1. Commit phase: reporters submit hashes (COMMIT_WINDOW seconds)
        2. Reveal phase: reporters reveal values (REVEAL_WINDOW seconds)

        Separating these phases is critical — if commit and reveal happened
        in the same transaction, a reporter could see others' reveals and
        adjust their value. The temporal separation forces commitment.
        """
        assert caller == self.owner or caller in self.authorized_reporters
        self.current_round += 1
        now = time.time()
        self.commit_deadline = now + self.COMMIT_WINDOW
        self.reveal_deadline = self.commit_deadline + self.REVEAL_WINDOW
        self.reveals[self.current_round] = []
        return self.current_round

    @staticmethod
    def compute_commit_hash(value_bps: int, salt: str, reporter: str) -> str:
        """
        Hash the value with salt and reporter address.

        Including the reporter address prevents one reporter from copying
        another's commit hash. The salt adds randomness so identical values
        from different reporters produce different hashes — without this,
        you could infer that two reporters submitted the same value.
        """
        preimage = f"{value_bps}||{salt}||{reporter}"
        return hashlib.sha256(preimage.encode()).hexdigest()

    def commit(self, commit_hash: str, reporter: str, round_id: int,
               current_time: Optional[float] = None) -> None:
        """
        Submit a commit for a round.

        The reporter sends only the hash — the actual value is hidden.
        This is Phase 1 of the commit-reveal protocol.
        """
        now = current_time or time.time()
        assert reporter in self.authorized_reporters, "Not an authorized reporter"
        assert round_id == self.current_round, "Wrong round"
        assert now <= self.commit_deadline, "Commit window closed"
        assert (reporter, round_id) not in self.commits, "Already committed this round"

        self.commits[(reporter, round_id)] = Commit(
            commit_hash=commit_hash,
            reporter=reporter,
            timestamp=now
        )

    def reveal(self, value_bps: int, salt: str, borrower: str,
               reporter: str, round_id: int,
               current_time: Optional[float] = None) -> None:
        """
        Reveal a previously committed value.

        The contract verifies that hash(value || salt || reporter) matches
        the stored commit. If it doesn't match, the reporter is trying to
        change their prediction after seeing others' commits — rejected.

        Phase 2 of commit-reveal: now the actual values become visible.
        """
        now = current_time or time.time()
        assert reporter in self.authorized_reporters, "Not authorized"
        assert round_id == self.current_round, "Wrong round"
        assert now > self.commit_deadline, "Commit window still open"
        assert now <= self.reveal_deadline, "Reveal window closed"

        key = (reporter, round_id)
        assert key in self.commits, "No commit found"
        assert not self.commits[key].revealed, "Already revealed"

        # Verify the commit hash matches
        expected_hash = self.compute_commit_hash(value_bps, salt, reporter)
        assert expected_hash == self.commits[key].commit_hash, \
            f"Hash mismatch — commit fraud detected! Expected {self.commits[key].commit_hash}, got {expected_hash}"

        self.commits[key].revealed = True
        assert 0 <= value_bps <= 10000, "Risk score must be 0-10000 bps"

        self.reveals[round_id].append(OracleReport(
            reporter=reporter,
            borrower=borrower,
            risk_score_bps=value_bps,
            timestamp=now,
            round_id=round_id
        ))

    def aggregate(self, borrower: str, round_id: int,
                  current_time: Optional[float] = None) -> int:
        """
        Compute the median risk score from all reveals in a round.

        Median is robust to up to floor(n/2) malicious reporters.
        If 5 reporters submit [200, 210, 215, 9999, 10000], the median
        is 215 — the two outliers have zero effect.

        For even number of reporters, we take the lower of the two middle
        values (conservative — slightly lower risk score means slightly
        more favorable terms for the borrower, which is the safe direction).
        """
        now = current_time or time.time()
        assert now > self.reveal_deadline, "Reveal window still open"

        reports = [r for r in self.reveals.get(round_id, []) if r.borrower == borrower]
        assert len(reports) >= self.MIN_REPORTERS, \
            f"Need at least {self.MIN_REPORTERS} reports, got {len(reports)}"

        # Sort scores and take median
        scores = sorted(r.risk_score_bps for r in reports)
        n = len(scores)
        if n % 2 == 1:
            median_score = scores[n // 2]
        else:
            # Conservative: take lower of two middle values
            median_score = scores[n // 2 - 1]

        self.aggregated_scores[borrower] = (median_score, now, round_id)
        return median_score

    def get_risk_score(self, borrower: str,
                       current_time: Optional[float] = None) -> tuple[int, float, int]:
        """
        Get the latest aggregated risk score for a borrower.

        Returns (score_bps, timestamp, round_id).
        Raises if no score exists or if the score is stale.
        """
        now = current_time or time.time()
        assert borrower in self.aggregated_scores, f"No score for {borrower}"

        score, ts, round_id = self.aggregated_scores[borrower]
        assert now - ts <= self.MAX_STALENESS, \
            f"Score is stale: {now - ts:.0f}s old (max {self.MAX_STALENESS}s)"

        return score, ts, round_id


# =============================================================================
# PART 3: Lending Contract
# =============================================================================
# Uses oracle-delivered risk scores to make autonomous lending decisions.

@dataclass
class Loan:
    """An outstanding loan with its terms."""
    borrower: str
    principal: float           # Amount borrowed
    collateral: float          # Collateral deposited
    interest_rate_bps: int     # Annual interest rate in basis points
    risk_score_bps: int        # Risk score at time of approval
    origination_time: float    # When the loan was issued
    oracle_round_id: int       # Which oracle round was used


@dataclass
class LendingContract:
    """
    Lending contract that uses oracle ML risk scores.

    The interest rate curve is piecewise linear:
    - Risk 0-2000 bps (0-20%):   base_rate + risk * 0.5
    - Risk 2000-5000 bps (20-50%): base_rate + 1000 + (risk-2000) * 1.5
    - Risk > 5000 bps (50%+):    DENIED

    This creates a convex curve — risk premium accelerates for riskier
    borrowers. The cutoff at 50% risk is a hard stop — beyond this,
    the expected loss exceeds the interest revenue at any reasonable rate.

    Collateral requirements also scale with risk:
    - Risk 0-2000:  110% collateral (10% cushion)
    - Risk 2000-5000: 150% collateral (50% cushion)

    Higher risk → more collateral → more protection for the protocol.
    """
    oracle: OracleContract
    owner: str
    pool_balance: float = 100000.0    # Available liquidity
    loans: dict[str, Loan] = field(default_factory=dict)
    collateral: dict[str, float] = field(default_factory=dict)

    # Rate curve parameters (all in basis points)
    BASE_RATE_BPS: int = 200          # 2% base rate
    MAX_RISK_BPS: int = 5000          # 50% — deny above this
    RISK_TIER1_CUTOFF: int = 2000     # 20% — low risk boundary
    TIER1_MULTIPLIER: float = 0.5     # Rate growth per bps of risk (low tier)
    TIER2_MULTIPLIER: float = 1.5     # Rate growth per bps of risk (high tier)

    def compute_interest_rate(self, risk_score_bps: int) -> int:
        """
        Map risk score to interest rate using the piecewise linear curve.

        Returns interest rate in basis points.
        Raises if risk is too high (loan denied).

        The curve is designed so that:
        - A perfect borrower (risk=0) pays just the base rate (2%)
        - A moderate borrower (risk=2000/20%) pays 2% + 10% = 12%
        - A risky borrower (risk=4000/40%) pays 2% + 10% + 30% = 42%
        - Beyond 50% risk: denied entirely

        These numbers are intentionally aggressive for demonstration.
        Real protocols use much more sophisticated curves calibrated
        to historical default rates and recovery values.
        """
        if risk_score_bps > self.MAX_RISK_BPS:
            raise ValueError(
                f"Risk score {risk_score_bps} bps exceeds maximum {self.MAX_RISK_BPS} bps — loan denied"
            )

        if risk_score_bps <= self.RISK_TIER1_CUTOFF:
            rate = self.BASE_RATE_BPS + int(risk_score_bps * self.TIER1_MULTIPLIER)
        else:
            tier1_premium = int(self.RISK_TIER1_CUTOFF * self.TIER1_MULTIPLIER)
            tier2_premium = int((risk_score_bps - self.RISK_TIER1_CUTOFF) * self.TIER2_MULTIPLIER)
            rate = self.BASE_RATE_BPS + tier1_premium + tier2_premium

        return rate

    def required_collateral_ratio(self, risk_score_bps: int) -> float:
        """
        Higher risk → more collateral required.

        Returns the ratio (e.g., 1.1 means 110% collateral).
        """
        if risk_score_bps <= self.RISK_TIER1_CUTOFF:
            return 1.10  # 110%
        else:
            return 1.50  # 150%

    def deposit_collateral(self, borrower: str, amount: float) -> None:
        """Deposit collateral before requesting a loan."""
        assert amount > 0, "Must deposit positive amount"
        self.collateral[borrower] = self.collateral.get(borrower, 0) + amount

    def request_loan(self, borrower: str, amount: float,
                     current_time: Optional[float] = None) -> Loan:
        """
        Request a loan. The contract:
        1. Fetches the borrower's risk score from the oracle
        2. Checks the score isn't stale
        3. Computes the interest rate from the risk curve
        4. Verifies sufficient collateral is deposited
        5. Approves and funds the loan

        This is the key integration point — off-chain ML feeds into
        on-chain financial decisions with no human in the loop.
        """
        assert borrower not in self.loans, "Borrower already has an active loan"
        assert amount > 0, "Loan amount must be positive"
        assert amount <= self.pool_balance, "Insufficient pool liquidity"

        # Step 1: Get oracle risk score (will fail if stale or missing)
        risk_score, oracle_ts, round_id = self.oracle.get_risk_score(
            borrower, current_time=current_time
        )

        # Step 2: Compute interest rate (will raise if risk too high)
        interest_rate = self.compute_interest_rate(risk_score)

        # Step 3: Check collateral
        required_ratio = self.required_collateral_ratio(risk_score)
        required_collateral = amount * required_ratio
        actual_collateral = self.collateral.get(borrower, 0)
        assert actual_collateral >= required_collateral, \
            f"Insufficient collateral: need {required_collateral:.2f}, have {actual_collateral:.2f}"

        # Step 4: Issue the loan
        now = current_time or time.time()
        loan = Loan(
            borrower=borrower,
            principal=amount,
            collateral=actual_collateral,
            interest_rate_bps=interest_rate,
            risk_score_bps=risk_score,
            origination_time=now,
            oracle_round_id=round_id
        )
        self.loans[borrower] = loan
        self.pool_balance -= amount

        return loan


# =============================================================================
# PART 4: Front-Running Attack Demonstration
# =============================================================================

def demonstrate_frontrun_vulnerability() -> dict:
    """
    Show how a naive (non-commit-reveal) oracle is vulnerable to front-running.

    The attack:
    1. Borrower sees oracle update tx in mempool: risk score going UP
    2. Borrower front-runs with a borrow tx at the current LOWER rate
    3. By the time the oracle update is mined, the loan is already issued

    This is a form of MEV (Maximal Extractable Value). In real blockchains,
    searchers run bots that monitor the mempool for exactly these opportunities.
    """
    results = {"steps": []}

    # Simulate naive oracle (no commit-reveal)
    results["steps"].append(
        "1. Current oracle risk score for Alice: 1500 bps (15%) → interest rate: 950 bps (9.5%)"
    )
    results["steps"].append(
        "2. Reporter submits update: risk score changing to 4000 bps (40%)"
    )
    results["steps"].append(
        "   This update sits in the mempool, visible to everyone..."
    )
    results["steps"].append(
        "3. Alice's bot sees the pending update in the mempool!"
    )
    results["steps"].append(
        "   New rate would be: 3200 bps (32.0%) — much more expensive"
    )
    results["steps"].append(
        "4. Alice front-runs: submits borrow tx with higher gas to get mined BEFORE the oracle update"
    )
    results["steps"].append(
        "   Alice borrows at 950 bps (9.5%) instead of 3200 bps (32.0%)"
    )
    results["steps"].append(
        f"   Savings on a 10,000 loan: {(3200 - 950) * 10000 / 10000:.0f} bps/year = {(3200 - 950) * 10000 / 10000 / 100:.1f}%/year"
    )

    # How commit-reveal prevents this
    results["prevention"] = [
        "With COMMIT-REVEAL, the oracle update is hidden:",
        "- Phase 1 (COMMIT): Reporter sends hash(4000 || salt || reporter_addr)",
        "  → Alice sees the commit tx but CANNOT extract the value 4000 from the hash",
        "  → SHA-256 is preimage resistant: given H, finding x such that SHA-256(x)=H is infeasible",
        "- Phase 2 (REVEAL): Reporter reveals 4000 + salt",
        "  → Contract verifies hash matches → value becomes official",
        "  → By this point, the commit window has closed — Alice can't time a borrow between phases",
        "RESULT: Alice cannot extract information from the commit to front-run the update"
    ]

    return results


# =============================================================================
# MAIN: Full Pipeline Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 032: SMART CONTRACT WITH ORACLE ML PREDICTIONS")
    print("=" * 70)

    # --------------------------------------------------
    # Step 1: Train the ML risk model
    # --------------------------------------------------
    print("\n--- STEP 1: Training ML Credit Risk Model ---\n")
    model = train_risk_model()

    feature_names = [
        "credit_utilization", "num_defaults", "account_age_norm",
        "loan_to_value", "income_norm"
    ]
    print(f"Learned weights:")
    for name, w in zip(feature_names, model.weights):
        direction = "↑ risk" if w > 0 else "↓ risk"
        print(f"  {name:25s}: {w:+.4f} ({direction})")
    print(f"  {'bias':25s}: {model.bias:+.4f}")

    # Score some example borrowers
    print("\n--- Scoring Example Borrowers ---\n")
    borrowers = {
        "alice": {
            "features": [0.2, 0, 1.0, 0.5, 0.8],  # Low util, no defaults, old account, low LTV, high income
            "description": "Low risk: low utilization, no defaults, established account"
        },
        "bob": {
            "features": [0.5, 1, 0.3, 1.0, 0.4],  # Medium util, 1 default, newer account, medium LTV
            "description": "Medium risk: moderate utilization, 1 past default"
        },
        "carol": {
            "features": [0.9, 3, 0.1, 1.8, 0.2],  # High util, many defaults, new account, high LTV, low income
            "description": "High risk: maxed credit, multiple defaults, new account"
        }
    }

    risk_scores = {}
    for name, info in borrowers.items():
        score_bps = model.predict_risk_bps(info["features"])
        prob = model.predict_probability(info["features"])
        risk_scores[name] = score_bps

        print(f"  {name.upper():8s}: {info['description']}")
        print(f"           P(default) = {prob:.4f} ({prob*100:.2f}%)")
        print(f"           Risk score = {score_bps} bps")

        # Show feature contributions
        contribs = model.explain(info["features"], feature_names)
        top_contrib = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        contrib_str = ", ".join(f"{k}={v:+.3f}" for k, v in top_contrib)
        print(f"           Top factors: {contrib_str}")
        print()

    # --------------------------------------------------
    # Step 2: Oracle - Commit-Reveal Flow
    # --------------------------------------------------
    print("--- STEP 2: Oracle Commit-Reveal Flow ---\n")

    # Use controlled timestamps for deterministic demo
    base_time = 1000000.0

    oracle = OracleContract(owner="deployer")
    oracle.add_reporter("reporter_1", caller="deployer")
    oracle.add_reporter("reporter_2", caller="deployer")
    oracle.add_reporter("reporter_3", caller="deployer")
    print(f"Oracle deployed with 3 authorized reporters")

    # Start a round for Alice
    round_id = oracle.start_round(caller="deployer")
    # Override deadlines for demo
    oracle.commit_deadline = base_time + oracle.COMMIT_WINDOW
    oracle.reveal_deadline = oracle.commit_deadline + oracle.REVEAL_WINDOW
    print(f"Round {round_id} started")

    # Each reporter computes their own ML score (simulating independent models)
    # Small variations because different models/data
    reporter_scores = {
        "reporter_1": risk_scores["alice"],
        "reporter_2": risk_scores["alice"] + 50,   # Slightly different model
        "reporter_3": risk_scores["alice"] - 30,    # Slightly different model
    }

    # Phase 1: COMMIT
    print(f"\n  COMMIT PHASE (values hidden):")
    salts = {}
    for reporter, score in reporter_scores.items():
        salt = os.urandom(16).hex()
        salts[reporter] = salt
        commit_hash = OracleContract.compute_commit_hash(score, salt, reporter)
        oracle.commit(commit_hash, reporter, round_id, current_time=base_time + 10)
        print(f"    {reporter} committed: hash={commit_hash[:16]}... (value hidden)")

    # Phase 2: REVEAL
    print(f"\n  REVEAL PHASE (values exposed):")
    reveal_time = oracle.commit_deadline + 10
    for reporter, score in reporter_scores.items():
        oracle.reveal(
            score, salts[reporter], "alice", reporter, round_id,
            current_time=reveal_time
        )
        print(f"    {reporter} revealed: {score} bps (verified against commit)")

    # Aggregation
    agg_time = oracle.reveal_deadline + 1
    median_score = oracle.aggregate("alice", round_id, current_time=agg_time)
    print(f"\n  AGGREGATED (median): Alice's risk score = {median_score} bps")

    # --------------------------------------------------
    # Step 3: Lending Contract
    # --------------------------------------------------
    print("\n--- STEP 3: Lending Contract Decisions ---\n")

    lending = LendingContract(oracle=oracle, owner="deployer")

    # Show the interest rate curve
    print("  Interest Rate Curve:")
    print("  Risk (bps) | Interest Rate (bps) | Rate %")
    print("  -----------|---------------------|-------")
    for risk in [0, 500, 1000, 1500, 2000, 3000, 4000, 5000]:
        rate = lending.compute_interest_rate(risk)
        print(f"  {risk:10d} | {rate:19d} | {rate/100:.2f}%")
    print(f"  {'> 5000':>10s} | {'DENIED':>19s} | —")

    # Process loan for Alice (low risk)
    query_time = agg_time + 10  # Within staleness window
    print(f"\n  Processing loan for ALICE (risk={median_score} bps):")
    lending.deposit_collateral("alice", 12000)
    rate = lending.compute_interest_rate(median_score)
    req_ratio = lending.required_collateral_ratio(median_score)
    print(f"    Interest rate: {rate} bps ({rate/100:.2f}%)")
    print(f"    Collateral requirement: {req_ratio*100:.0f}%")
    print(f"    Collateral deposited: 12,000")

    loan = lending.request_loan("alice", 10000, current_time=query_time)
    print(f"    APPROVED! Loan: {loan.principal:.0f} at {loan.interest_rate_bps} bps ({loan.interest_rate_bps/100:.2f}%)")

    # Now do a round for Carol (high risk)
    print(f"\n  Processing CAROL (high risk):")
    round_id_2 = oracle.start_round(caller="deployer")
    oracle.commit_deadline = base_time + 200 + oracle.COMMIT_WINDOW
    oracle.reveal_deadline = oracle.commit_deadline + oracle.REVEAL_WINDOW

    carol_score = risk_scores["carol"]
    salt_carol = os.urandom(16).hex()
    commit_hash_carol = OracleContract.compute_commit_hash(carol_score, salt_carol, "reporter_1")
    oracle.commit(commit_hash_carol, "reporter_1", round_id_2, current_time=base_time + 210)
    oracle.reveal(carol_score, salt_carol, "carol", "reporter_1", round_id_2,
                  current_time=oracle.commit_deadline + 5)
    carol_agg = oracle.aggregate("carol", round_id_2, current_time=oracle.reveal_deadline + 1)
    agg_time_2 = oracle.reveal_deadline + 1

    print(f"    Oracle risk score: {carol_agg} bps ({carol_agg/100:.2f}%)")
    if carol_agg > lending.MAX_RISK_BPS:
        print(f"    DENIED — risk score {carol_agg} exceeds maximum {lending.MAX_RISK_BPS} bps")
        print(f"    The ML model flagged Carol as too risky for lending")
    else:
        rate_carol = lending.compute_interest_rate(carol_agg)
        print(f"    Interest rate: {rate_carol} bps ({rate_carol/100:.2f}%)")

    # --------------------------------------------------
    # Step 4: Staleness Protection Demo
    # --------------------------------------------------
    print("\n--- STEP 4: Staleness Protection ---\n")

    stale_time = agg_time + oracle.MAX_STALENESS + 100  # Well past staleness window
    print(f"  Attempting to use Alice's score {oracle.MAX_STALENESS + 100:.0f}s after aggregation...")
    try:
        oracle.get_risk_score("alice", current_time=stale_time)
        print("  ERROR: Should have been rejected!")
    except AssertionError as e:
        print(f"  BLOCKED: {e}")
        print(f"  Stale oracle data cannot be used for lending decisions.")
        print(f"  This prevents Compound-style incidents where stale prices caused $80M in bad debt.")

    # --------------------------------------------------
    # Step 5: Front-Running Demo
    # --------------------------------------------------
    print("\n--- STEP 5: Front-Running Attack & Prevention ---\n")

    frontrun = demonstrate_frontrun_vulnerability()
    print("  ATTACK SCENARIO (without commit-reveal):")
    for step in frontrun["steps"]:
        print(f"    {step}")
    print()
    print("  PREVENTION (with commit-reveal):")
    for line in frontrun["prevention"]:
        print(f"    {line}")

    # --------------------------------------------------
    # Step 6: Commit-Reveal Fraud Detection
    # --------------------------------------------------
    print("\n--- STEP 6: Commit-Reveal Fraud Detection ---\n")

    round_id_3 = oracle.start_round(caller="deployer")
    oracle.commit_deadline = base_time + 400 + oracle.COMMIT_WINDOW
    oracle.reveal_deadline = oracle.commit_deadline + oracle.REVEAL_WINDOW

    # Reporter commits 3000 bps...
    original_score = 3000
    cheat_score = 500
    salt_fraud = os.urandom(16).hex()
    commit_hash_honest = OracleContract.compute_commit_hash(original_score, salt_fraud, "reporter_1")
    oracle.commit(commit_hash_honest, "reporter_1", round_id_3, current_time=base_time + 410)
    print(f"  Reporter committed hash for value {original_score} bps")

    # ...then tries to reveal a DIFFERENT value
    print(f"  Reporter tries to reveal {cheat_score} bps instead...")
    try:
        oracle.reveal(cheat_score, salt_fraud, "dave", "reporter_1", round_id_3,
                      current_time=oracle.commit_deadline + 5)
        print("  ERROR: Should have caught fraud!")
    except AssertionError as e:
        print(f"  CAUGHT: {e}")
        print(f"  The hash of {cheat_score}||salt||reporter doesn't match the committed hash")
        print(f"  Commit-reveal ensures reporters cannot change their mind after seeing others' commits")

    print("\n" + "=" * 70)
    print("COMPLETE: ML model → Oracle → Lending contract pipeline demonstrated")
    print("=" * 70)
