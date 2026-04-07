"""
Day 001: Linear Regression from Scratch

Implement two methods:
1. Normal Equation (closed-form) -- exact solution via matrix algebra
2. Gradient Descent (iterative) -- the optimization loop that powers all of modern ML

No sklearn, no frameworks -- just NumPy and understanding.
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

    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        true_weights: The ground-truth weight vector
        true_bias: The ground-truth bias scalar
    """
    rng = np.random.default_rng(seed)
    if true_weights is None:
        true_weights = rng.standard_normal(n_features)
    else:
        true_weights = np.asarray(true_weights, dtype=float).reshape(n_features)

    X = rng.standard_normal((n_samples, n_features))
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = X @ true_weights + true_bias + noise
    return X, y, true_weights, float(true_bias)


# =============================================================================
# Feature Augmentation
# =============================================================================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """
    Prepend a column of ones to X, turning [x1, x2] into [1, x1, x2].

    This lets us absorb the bias term into the weight vector:
        y = w0*1 + w1*x1 + w2*x2 = [1, x1, x2] @ [w0, w1, w2]
    """
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])


# =============================================================================
# Solution 1: Normal Equation
# =============================================================================

def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form solution: w* = (X^T X)^{-1} X^T y

    Args:
        X: Feature matrix with bias column, shape (n_samples, n_features + 1)
        y: Target vector, shape (n_samples,)

    Returns:
        w: Weight vector, shape (n_features + 1,)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    gram = X.T @ X
    rhs = X.T @ y
    return np.linalg.solve(gram, rhs)


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
    Batch gradient descent for linear regression.

    The gradient of MSE w.r.t. weights is:
        grad_L = (2/n) * X^T (Xw - y)

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Target vector, shape (n_samples,)
        learning_rate: Step size alpha
        n_iterations: Number of gradient steps
        verbose: Print progress every 100 iterations

    Returns:
        w: Learned weight vector
        loss_history: List of MSE values at each iteration
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    loss_history: list[float] = []

    for i in range(n_iterations):
        residual = X @ w - y
        loss_history.append(float(np.mean(residual**2)))
        if verbose and i % 100 == 0:
            print(f"  iter {i}: MSE = {loss_history[-1]:.6f}")
        grad = (2.0 / n_samples) * (X.T @ residual)
        w -= learning_rate * grad

    return w, loss_history


# =============================================================================
# Feature Scaling
# =============================================================================

def standardize(
    X: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit variance: z = (x - mu) / sigma

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        mean: Pre-computed mean (use None to compute from X)
        std: Pre-computed std (use None to compute from X)

    Returns:
        X_scaled: Standardized feature matrix
        mean: Mean used for scaling
        std: Std used for scaling
    """
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = np.mean(X, axis=0)
    else:
        mean = np.asarray(mean, dtype=float)
    if std is None:
        std = np.std(X, axis=0, ddof=0)
    else:
        std = np.asarray(std, dtype=float)
    std_safe = np.where(std == 0, 1.0, std)
    X_scaled = (X - mean) / std_safe
    return X_scaled, mean, std_safe


# =============================================================================
# Evaluation Metrics
# =============================================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error -- average of squared residuals.

    MSE = (1/n) * sum((y_true - y_pred)^2)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 (coefficient of determination) -- proportion of variance explained.

    R^2 = 1 - (SS_res / SS_tot)
    where SS_res = sum((y - y_hat)^2), SS_tot = sum((y - y_mean)^2)

    Returns 1.0 for perfect predictions, 0.0 for mean-level predictions.
    """
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot == 0:
        return 1.0 if np.isclose(ss_res, 0.0) else 0.0
    return float(1.0 - ss_res / ss_tot)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 001: LINEAR REGRESSION FROM SCRATCH")
    print("=" * 70)

    # Generate synthetic data with known weights
    true_w = np.array([3.0, 7.0])
    true_b = 2.0
    X, y, true_w, true_b = generate_data(
        n_samples=300, true_weights=true_w, true_bias=true_b, noise_std=0.5
    )
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Ground truth: weights = {true_w}, bias = {true_b}")

    # --- Normal Equation ---
    print("\n--- Normal Equation ---")
    X_aug = add_bias_column(X)
    w_normal = normal_equation(X_aug, y)
    print(f"Recovered weights: {w_normal[1:]}")
    print(f"Recovered bias:    {w_normal[0]:.4f}")

    y_pred_normal = X_aug @ w_normal
    print(f"MSE:  {mse(y, y_pred_normal):.6f}")
    print(f"R^2:  {r_squared(y, y_pred_normal):.6f}")

    # --- Gradient Descent (with scaling) ---
    print("\n--- Gradient Descent (with feature scaling) ---")
    X_scaled, train_mean, train_std = standardize(X)
    X_scaled_aug = add_bias_column(X_scaled)
    w_gd, loss_history = gradient_descent(
        X_scaled_aug, y, learning_rate=0.1, n_iterations=500, verbose=True
    )
    print(f"Recovered weights (scaled space): {w_gd[1:]}")
    print(f"Recovered bias:                   {w_gd[0]:.4f}")
    print(f"Final MSE: {loss_history[-1]:.6f}")
