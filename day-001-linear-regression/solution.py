"""
Day 001: Linear Regression from Scratch

Two complete implementations:
1. Normal Equation (closed-form) — exact solution via matrix algebra
2. Gradient Descent (iterative) — the optimization loop that powers all of modern ML

No sklearn, no frameworks — just NumPy and understanding.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(
    n_samples: int = 200,
    n_features: int = 2,
    true_weights: Optional[np.ndarray] = None,
    true_bias: float = 2.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate synthetic linear data: y = Xw + b + noise.

    We use known weights so we can verify our model recovers them.
    In real ML, you never know the true weights — but for learning,
    this is invaluable for debugging.
    """
    rng = np.random.default_rng(seed)

    if true_weights is None:
        true_weights = np.array([3.0, 7.0])  # ground truth we'll try to recover
        n_features = len(true_weights)

    X = rng.standard_normal((n_samples, n_features))

    # y = Xw + b + noise
    # The noise simulates irreducible error — even a perfect model can't predict this
    y = X @ true_weights + true_bias + rng.normal(0, noise_std, n_samples)

    return X, y, true_weights, true_bias


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets by shuffling indices."""
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =============================================================================
# Feature Augmentation
# =============================================================================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """
    Prepend a column of ones to X, turning [x1, x2] into [1, x1, x2].

    This lets us absorb the bias term into the weight vector:
        y = w0*1 + w1*x1 + w2*x2 = [1, x1, x2] @ [w0, w1, w2]

    Now w0 IS the bias, and we only have one matrix multiply to worry about.
    """
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


# =============================================================================
# Solution 1: Normal Equation
# =============================================================================

def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form solution: w* = (X^T X)^{-1} X^T y

    This solves the system of linear equations ∇L = 0 directly.

    Complexity: O(d³) for the matrix inverse, O(nd²) for X^T X.
    For d < ~10,000 features and well-conditioned data, this is fast and exact.

    We use np.linalg.solve instead of np.linalg.inv for better numerical stability.
    solve(A, b) computes A^{-1} b without explicitly forming the inverse — it uses
    LU decomposition under the hood, which is both faster and more numerically stable.
    """
    # X^T X is the Gram matrix — a d×d matrix capturing feature correlations
    # X^T y is the cross-correlation between features and the target
    gram = X.T @ X
    cross = X.T @ y

    # Solve the normal equations: (X^T X) w = X^T y
    # This is equivalent to w = (X^T X)^{-1} X^T y but numerically better
    w = np.linalg.solve(gram, cross)
    return w


# =============================================================================
# Solution 2: Gradient Descent
# =============================================================================

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, list]:
    """
    Batch gradient descent: update weights using the full dataset each iteration.

    The gradient of MSE w.r.t. weights is:
        ∇L = (2/n) * X^T (Xw - y)

    Intuition: (Xw - y) is the vector of errors. X^T maps those errors back to
    the feature space, telling us how much each feature contributed to the error.
    Multiply by 2/n to get the average gradient.

    We subtract α * ∇L from w because the gradient points UPHILL — we want to go
    downhill, so we go in the opposite direction.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # start at the origin — any starting point works for convex problems
    loss_history = []

    for i in range(n_iterations):
        # Forward pass: compute predictions
        predictions = X @ w

        # Compute error vector (residuals)
        errors = predictions - y

        # MSE loss for tracking convergence
        mse = np.mean(errors ** 2)
        loss_history.append(mse)

        # Gradient: the direction of steepest ASCENT
        gradient = (2 / n_samples) * (X.T @ errors)

        # Update: step in the opposite direction (steepest descent)
        w = w - learning_rate * gradient

        if verbose and (i % 100 == 0 or i == n_iterations - 1):
            print(f"  Iteration {i:4d} | MSE: {mse:.6f} | ||grad||: {np.linalg.norm(gradient):.6f}")

    return w, loss_history


# =============================================================================
# Feature Scaling
# =============================================================================

def standardize(
    X: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit variance: z = (x - μ) / σ

    Why this matters for gradient descent:
    - Without scaling, features with large ranges dominate the gradient.
    - The loss surface becomes an elongated ellipsoid — gradient descent zigzags
      instead of heading straight to the minimum.
    - With scaling, all features contribute equally, the loss surface is nearly
      spherical, and gradient descent converges fast.

    Returns the mean and std so we can apply the SAME transformation to test data.
    Using test data statistics would be data leakage — a subtle but critical mistake.
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero for constant features

    return (X - mean) / std, mean, std


# =============================================================================
# Evaluation Metrics
# =============================================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error — average of squared residuals."""
    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (coefficient of determination) — proportion of variance explained.

    R² = 1 - (SS_res / SS_tot)

    SS_res = Σ(y - ŷ)² — residual sum of squares (unexplained variance)
    SS_tot = Σ(y - ȳ)² — total sum of squares (total variance)

    If SS_res = 0, we explain everything → R² = 1
    If SS_res = SS_tot, we explain nothing → R² = 0
    If SS_res > SS_tot, we're worse than the mean → R² < 0
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# =============================================================================
# Main: Demonstrate Everything
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 001: LINEAR REGRESSION FROM SCRATCH")
    print("=" * 70)

    # --- Generate and split data ---
    true_w = np.array([3.0, 7.0])
    true_b = 2.0
    X, y, true_w, true_b = generate_data(
        n_samples=300, true_weights=true_w, true_bias=true_b, noise_std=0.5
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Ground truth: weights = {true_w}, bias = {true_b}")

    # --- Normal Equation ---
    print("\n" + "=" * 70)
    print("METHOD 1: NORMAL EQUATION (Closed-Form)")
    print("=" * 70)

    X_train_aug = add_bias_column(X_train)
    X_test_aug = add_bias_column(X_test)

    w_normal = normal_equation(X_train_aug, y_train)

    # w_normal[0] is the bias (coefficient of the ones column)
    # w_normal[1:] are the feature weights
    print(f"\nRecovered weights: {w_normal[1:]}")
    print(f"Recovered bias:    {w_normal[0]:.4f}")
    print(f"True weights:      {true_w}")
    print(f"True bias:         {true_b}")

    y_pred_normal = X_test_aug @ w_normal
    print(f"\nTest MSE:  {mse(y_test, y_pred_normal):.6f}")
    print(f"Test R²:   {r_squared(y_test, y_pred_normal):.6f}")

    # --- Gradient Descent (without scaling) ---
    print("\n" + "=" * 70)
    print("METHOD 2: GRADIENT DESCENT (Without Feature Scaling)")
    print("=" * 70)
    print("\nRunning gradient descent with lr=0.01, 1000 iterations...")

    w_gd_raw, loss_raw = gradient_descent(
        X_train_aug, y_train, learning_rate=0.01, n_iterations=1000, verbose=True
    )

    print(f"\nRecovered weights: {w_gd_raw[1:]}")
    print(f"Recovered bias:    {w_gd_raw[0]:.4f}")

    y_pred_gd_raw = X_test_aug @ w_gd_raw
    print(f"Test MSE:  {mse(y_test, y_pred_gd_raw):.6f}")
    print(f"Test R²:   {r_squared(y_test, y_pred_gd_raw):.6f}")

    # --- Gradient Descent (with scaling) ---
    print("\n" + "=" * 70)
    print("METHOD 3: GRADIENT DESCENT (With Feature Scaling)")
    print("=" * 70)

    # Scale features BEFORE adding bias column
    # Important: fit statistics on train data only, then apply to test
    X_train_scaled, train_mean, train_std = standardize(X_train)
    X_test_scaled, _, _ = standardize(X_test, train_mean, train_std)

    X_train_scaled_aug = add_bias_column(X_train_scaled)
    X_test_scaled_aug = add_bias_column(X_test_scaled)

    print("\nRunning gradient descent with lr=0.1, 500 iterations...")
    print("(Scaling lets us use a 10x larger learning rate and fewer iterations)\n")

    w_gd_scaled, loss_scaled = gradient_descent(
        X_train_scaled_aug, y_train, learning_rate=0.1, n_iterations=500, verbose=True
    )

    # Note: these weights are in the SCALED feature space.
    # To interpret them in original space: w_original = w_scaled / std, b_original = b_scaled - (w_scaled / std) @ mean
    w_original = w_gd_scaled[1:] / train_std
    b_original = w_gd_scaled[0] - (w_gd_scaled[1:] / train_std) @ train_mean

    print(f"\nRecovered weights (transformed back): {w_original}")
    print(f"Recovered bias (transformed back):    {b_original:.4f}")

    y_pred_gd_scaled = X_test_scaled_aug @ w_gd_scaled
    print(f"Test MSE:  {mse(y_test, y_pred_gd_scaled):.6f}")
    print(f"Test R²:   {r_squared(y_test, y_pred_gd_scaled):.6f}")

    # --- Convergence comparison ---
    print("\n" + "=" * 70)
    print("CONVERGENCE COMPARISON")
    print("=" * 70)
    print(f"\nWithout scaling (lr=0.01, 1000 iters):")
    print(f"  Initial MSE: {loss_raw[0]:.4f}")
    print(f"  Final MSE:   {loss_raw[-1]:.6f}")
    print(f"  Iterations to reach MSE < 0.5: ", end="")
    below_threshold = [i for i, l in enumerate(loss_raw) if l < 0.5]
    print(f"{below_threshold[0]}" if below_threshold else "never reached")

    print(f"\nWith scaling (lr=0.1, 500 iters):")
    print(f"  Initial MSE: {loss_scaled[0]:.4f}")
    print(f"  Final MSE:   {loss_scaled[-1]:.6f}")
    print(f"  Iterations to reach MSE < 0.5: ", end="")
    below_threshold = [i for i, l in enumerate(loss_scaled) if l < 0.5]
    print(f"{below_threshold[0]}" if below_threshold else "never reached")

    # --- Final comparison ---
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: ALL METHODS")
    print("=" * 70)
    print(f"\n{'Method':<35} {'MSE':>10} {'R²':>10}")
    print("-" * 57)
    print(f"{'Normal Equation':<35} {mse(y_test, y_pred_normal):>10.6f} {r_squared(y_test, y_pred_normal):>10.6f}")
    print(f"{'GD (no scaling, lr=0.01, 1000it)':<35} {mse(y_test, y_pred_gd_raw):>10.6f} {r_squared(y_test, y_pred_gd_raw):>10.6f}")
    print(f"{'GD (scaled, lr=0.1, 500it)':<35} {mse(y_test, y_pred_gd_scaled):>10.6f} {r_squared(y_test, y_pred_gd_scaled):>10.6f}")

    print("\nKey takeaway: All three methods converge to the same solution.")
    print("The Normal Equation gets there instantly; gradient descent iterates.")
    print("Feature scaling makes gradient descent converge ~10x faster.")
    print("But gradient descent generalizes to problems where no closed-form exists.")
