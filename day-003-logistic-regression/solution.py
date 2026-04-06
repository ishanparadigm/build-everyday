"""
Day 003: Logistic Regression from Scratch

Builds directly on Day 001's gradient descent framework, but replaces the linear
prediction with sigmoid and MSE with cross-entropy loss. This single change —
wrapping a linear model in a nonlinearity — is the conceptual leap that takes us
from regression to classification, and from statistics to neural networks.
"""

import numpy as np
from typing import Tuple


class LogisticRegression:
    """
    Binary classifier using logistic regression with gradient descent.

    The model: P(y=1|x) = sigmoid(Xw + b)
    The loss: binary cross-entropy (negative log-likelihood)
    The optimizer: batch gradient descent with optional L2 regularization
    """

    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000,
                 lambda_reg: float = 0.0):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg  # L2 regularization strength
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        # Store scaling parameters for transforming new data
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.

        Naive implementation 1/(1+exp(-z)) overflows when z is very negative
        (exp(-z) → ∞). Instead, we use:
        - For z >= 0: 1 / (1 + exp(-z))          [standard form]
        - For z < 0:  exp(z) / (1 + exp(z))      [equivalent, avoids overflow]

        This is the same mathematical function, just rearranged to keep the
        exponent's argument negative, preventing overflow in either direction.
        """
        result = np.zeros_like(z, dtype=np.float64)
        pos_mask = z >= 0
        neg_mask = ~pos_mask
        # Standard form for positive z
        result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
        # Overflow-safe form for negative z
        exp_z = np.exp(z[neg_mask])
        result[neg_mask] = exp_z / (1.0 + exp_z)
        return result

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Standardize features to zero mean and unit variance.

        Same motivation as Day 001: gradient descent on un-scaled features creates
        elongated loss contours → slow, zigzagging convergence. Standardization
        makes contours more circular → faster, more direct convergence.

        We store mean/std at fit time so we can apply the SAME transformation
        to test data — a common mistake is re-fitting the scaler on test data,
        which leaks information and changes the feature space.
        """
        if fit:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0)
            # Avoid division by zero for constant features
            self._std[self._std == 0] = 1.0
        return (X - self._mean) / self._std

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary cross-entropy loss: -1/m * Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]

        This is the negative log-likelihood of the data under our model.
        Minimizing this = maximizing the probability our model assigns to the
        correct labels.

        We clip predictions away from 0 and 1 to avoid log(0) = -inf.
        The clip range 1e-15 is small enough to not affect predictions but
        large enough to keep the loss finite.
        """
        m = len(y)
        # Clip to avoid numerical issues with log(0)
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        # Cross-entropy: measures "surprise" when using predicted distribution
        # to encode the true labels
        loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))

        # L2 regularization term: penalizes large weights to prevent overfitting.
        # Note: we don't regularize the bias — it's a location parameter, not
        # a complexity parameter. Regularizing it would be like penalizing the
        # model for having a non-zero base rate.
        if self.lambda_reg > 0:
            loss += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)

        return loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train the model using batch gradient descent.

        The gradient of cross-entropy loss w.r.t. weights is:
            ∂J/∂w = 1/m · Xᵀ(ŷ - y)
            ∂J/∂b = 1/m · Σ(ŷᵢ - yᵢ)

        This is identical in form to linear regression's gradient (Day 001)!
        The sigmoid "absorbs" into the derivative so cleanly because of its
        special property: σ'(z) = σ(z)(1-σ(z)), which cancels terms in the
        chain rule expansion.
        """
        X_scaled = self._standardize(X, fit=True)
        m, n_features = X_scaled.shape

        # Initialize weights to zero. For logistic regression (convex problem),
        # all initializations converge to the same solution. Zero is conventional
        # and means we start with P(y=1) = sigmoid(0) = 0.5 for all inputs —
        # the maximally uncertain prediction.
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            # --- Forward pass ---
            # Linear combination: z = Xw + b (same as linear regression)
            z = X_scaled @ self.weights + self.bias
            # Squash through sigmoid to get probabilities (the key difference)
            y_pred = self._sigmoid(z)

            # --- Compute and store loss ---
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # --- Backward pass (gradient computation) ---
            # The "error signal": how wrong are our predictions?
            # Positive = we predicted too high, negative = too low
            error = y_pred - y

            # Gradient w.r.t. weights: correlates features with errors
            # High gradient for a feature = that feature is consistently associated
            # with prediction errors → needs a bigger weight update
            dw = (1 / m) * (X_scaled.T @ error)
            db = (1 / m) * np.sum(error)

            # Add L2 regularization gradient: pulls weights toward zero
            # at a rate proportional to their magnitude
            if self.lambda_reg > 0:
                dw += (self.lambda_reg / m) * self.weights

            # --- Parameter update ---
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities P(y=1|x) for each sample."""
        X_scaled = self._standardize(X)
        z = X_scaled @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return class predictions (0 or 1).

        The threshold of 0.5 is the natural choice (predict the more likely class),
        but in practice you might adjust it:
        - Lower threshold → more positive predictions (higher recall, lower precision)
        - Higher threshold → fewer positive predictions (lower recall, higher precision)
        This is the precision-recall tradeoff, governed by the ROC curve.
        """
        return (self.predict_proba(X) >= threshold).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics from the confusion matrix.

    Accuracy alone is misleading for imbalanced datasets. If 99% of emails are
    not spam, predicting "not spam" always gives 99% accuracy but catches zero spam.

    Precision: Of everything we predicted positive, what fraction actually was?
    Recall: Of everything that actually was positive, what fraction did we catch?
    F1: Harmonic mean of precision and recall — balances both concerns.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / len(y_true)
    # Guard against division by zero when no positive predictions are made
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
    }


def generate_binary_dataset(
    n_samples: int = 300, n_features: int = 2, separation: float = 1.5, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic binary classification dataset.

    Creates two Gaussian clusters with controllable separation.
    Lower separation → more overlap → harder classification problem.
    Returns train/test split (80/20).
    """
    rng = np.random.RandomState(seed)

    n_per_class = n_samples // 2

    # Class 0: centered at origin
    X0 = rng.randn(n_per_class, n_features)
    # Class 1: shifted by 'separation' along each axis
    X1 = rng.randn(n_per_class, n_features) + separation

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # Shuffle to avoid ordered classes (important for mini-batch methods later)
    shuffle_idx = rng.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Train/test split
    split = int(0.8 * n_samples)
    return X[:split], X[split:], y[:split], y[split:]


def demonstrate_sigmoid():
    """Show the sigmoid function's key properties."""
    print("=" * 60)
    print("SIGMOID FUNCTION PROPERTIES")
    print("=" * 60)

    z_values = np.array([-10, -5, -2, -1, 0, 1, 2, 5, 10])
    sigmoid = LogisticRegression._sigmoid(z_values)

    print(f"\n{'z':>6}  {'σ(z)':>8}  {'1-σ(z)':>8}  {'σ(z)·(1-σ(z))':>14}")
    print("-" * 42)
    for z, s in zip(z_values, sigmoid):
        deriv = s * (1 - s)
        print(f"{z:6.1f}  {s:8.5f}  {1 - s:8.5f}  {deriv:14.5f}")

    print("\nKey observations:")
    print("  - σ(0) = 0.5 (maximum uncertainty)")
    print("  - σ(-z) = 1 - σ(z) (perfect symmetry)")
    print("  - Derivative peaks at z=0 (steepest learning near the boundary)")
    print("  - Saturates at extremes (confident predictions change slowly)")


def demonstrate_training():
    """Full training pipeline with detailed output."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION TRAINING")
    print("=" * 60)

    # Generate data with moderate separation — not trivial, not impossible
    X_train, X_test, y_train, y_test = generate_binary_dataset(
        n_samples=300, separation=1.5, seed=42
    )
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class balance (train): {np.mean(y_train):.1%} positive")

    # Train the model
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X_train, y_train)

    # Show loss convergence at key checkpoints
    print("\nTraining convergence:")
    print(f"  {'Iteration':>10}  {'Loss':>10}")
    print("  " + "-" * 24)
    checkpoints = [0, 10, 50, 100, 250, 499]
    for i in checkpoints:
        print(f"  {i + 1:>10}  {model.loss_history[i]:>10.6f}")

    # Learned parameters (in standardized space)
    print(f"\nLearned weights: {model.weights}")
    print(f"Learned bias: {model.bias:.4f}")
    print("(Weights are in standardized feature space)")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print(f"\n--- Test Set Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 Score:  {metrics['f1_score']:.1%}")

    cm = metrics["confusion_matrix"]
    print(f"\nConfusion Matrix:")
    print(f"              Predicted 0  Predicted 1")
    print(f"  Actual 0:   {cm['tn']:>10}  {cm['fp']:>10}")
    print(f"  Actual 1:   {cm['fn']:>10}  {cm['tp']:>10}")

    # Show some individual predictions with confidence
    print(f"\nSample predictions (first 10 test points):")
    print(f"  {'True':>5} {'Pred':>5} {'P(y=1)':>8} {'Correct':>8}")
    print("  " + "-" * 30)
    for i in range(min(10, len(y_test))):
        correct = "✓" if y_pred[i] == y_test[i] else "✗"
        print(f"  {int(y_test[i]):>5} {y_pred[i]:>5} {y_proba[i]:>8.4f} {correct:>8}")

    return model


def demonstrate_regularization():
    """Show how L2 regularization affects the model."""
    print("\n" + "=" * 60)
    print("EFFECT OF L2 REGULARIZATION")
    print("=" * 60)

    X_train, X_test, y_train, y_test = generate_binary_dataset(
        n_samples=300, separation=1.5, seed=42
    )

    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    print(f"\n{'Lambda':>8} {'Train Acc':>10} {'Test Acc':>10} {'||w||':>10}")
    print("-" * 42)

    for lam in lambdas:
        model = LogisticRegression(learning_rate=0.1, n_iterations=500, lambda_reg=lam)
        model.fit(X_train, y_train)

        train_acc = np.mean(model.predict(X_train) == y_train)
        test_acc = np.mean(model.predict(X_test) == y_test)
        weight_norm = np.linalg.norm(model.weights)

        print(f"{lam:>8.2f} {train_acc:>10.1%} {test_acc:>10.1%} {weight_norm:>10.4f}")

    print("\nObservations:")
    print("  - Small λ: minimal effect — model fits freely")
    print("  - Moderate λ: weights shrink, may generalize better")
    print("  - Large λ: underfitting — weights crushed toward zero")
    print("  - The 'right' λ balances fit quality vs model complexity")


def demonstrate_decision_boundary():
    """
    Explain the decision boundary geometrically.

    The boundary is where P(y=1|x) = 0.5, i.e., where wᵀx + b = 0.
    In 2D with features x₁ and x₂:
        w₁·x₁ + w₂·x₂ + b = 0
        x₂ = -(w₁/w₂)·x₁ - b/w₂
    This is a straight line — logistic regression always produces linear boundaries.
    """
    print("\n" + "=" * 60)
    print("DECISION BOUNDARY ANALYSIS")
    print("=" * 60)

    X_train, _, y_train, _ = generate_binary_dataset(n_samples=300, separation=1.5, seed=42)

    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X_train, y_train)

    w = model.weights
    b = model.bias

    print(f"\nIn standardized feature space:")
    print(f"  w = [{w[0]:.4f}, {w[1]:.4f}], b = {b:.4f}")
    print(f"\n  Decision boundary: {w[0]:.4f}·x₁ + {w[1]:.4f}·x₂ + {b:.4f} = 0")

    if abs(w[1]) > 1e-10:
        slope = -w[0] / w[1]
        intercept = -b / w[1]
        print(f"  As a line: x₂ = {slope:.4f}·x₁ + {intercept:.4f}")

    # Show predictions at various distances from boundary
    print(f"\n  Distance from boundary → Confidence:")
    print(f"  {'Distance':>10} {'P(y=1)':>8}")
    print("  " + "-" * 20)
    # Points along the weight vector direction (perpendicular to boundary)
    w_norm = w / np.linalg.norm(w)
    for dist in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]:
        # Point at 'dist' units from boundary along normal direction
        # The boundary passes through -b*w_norm/||w||, but for simplicity
        # we compute the sigmoid of the signed distance
        z = dist * np.linalg.norm(w)
        prob = 1.0 / (1.0 + np.exp(-z))
        marker = " ← boundary" if dist == 0 else ""
        print(f"  {dist:>10.1f} {prob:>8.4f}{marker}")

    print("\n  Further from boundary = more confident prediction")
    print("  The sigmoid maps distance to probability nonlinearly")


if __name__ == "__main__":
    demonstrate_sigmoid()
    model = demonstrate_training()
    demonstrate_regularization()
    demonstrate_decision_boundary()
