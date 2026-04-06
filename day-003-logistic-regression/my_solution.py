"""
Day 003: Logistic Regression from Scratch

Builds on Day 001's gradient descent framework. The key change: wrap a linear
model in a sigmoid nonlinearity and replace MSE with cross-entropy loss.
This takes us from regression to classification.
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
        """
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of gradient descent iterations
            lambda_reg: L2 regularization strength (0 = no regularization)
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function: sigma(z) = 1 / (1 + exp(-z))

        Must handle both large positive and large negative z without overflow.

        Args:
            z: Input array of any shape

        Returns:
            Array of same shape with values in (0, 1)
        """
        # Hint: for z >= 0 use 1/(1+exp(-z)); for z < 0 use exp(z)/(1+exp(z))
        # Hint: this avoids overflow from exp() of large positive numbers
        raise NotImplementedError("TODO: implement this")

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Standardize features to zero mean and unit variance.

        When fit=True, compute and store mean/std from X.
        When fit=False, use previously stored mean/std.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            fit: Whether to compute new statistics or use stored ones

        Returns:
            Standardized feature matrix
        """
        # Hint: same as Day 001's standardize, but stores mean/std as instance vars
        # Hint: handle constant features (std == 0) to avoid division by zero
        raise NotImplementedError("TODO: implement this")

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary cross-entropy loss: -1/m * sum[y*log(y_hat) + (1-y)*log(1-y_hat)]

        Optionally includes L2 regularization: + (lambda/2m) * ||w||^2

        Args:
            y: True labels, shape (m,)
            y_pred: Predicted probabilities, shape (m,)

        Returns:
            Scalar loss value
        """
        # Hint: clip y_pred away from 0 and 1 to avoid log(0)
        # Hint: don't regularize the bias, only the weights
        raise NotImplementedError("TODO: implement this")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train the model using batch gradient descent.

        The gradient of cross-entropy loss w.r.t. weights is:
            dw = (1/m) * X^T (y_hat - y)
            db = (1/m) * sum(y_hat - y)

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels (0 or 1), shape (n_samples,)

        Returns:
            self (for method chaining)
        """
        # Hint: standardize X first (fit=True)
        # Hint: initialize weights to zeros, bias to 0
        # Hint: each iteration: z = Xw + b -> sigmoid -> loss -> gradients -> update
        # Hint: if lambda_reg > 0, add regularization term to weight gradient
        raise NotImplementedError("TODO: implement this")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probabilities P(y=1|x) for each sample.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Probability array, shape (n_samples,), values in [0, 1]
        """
        # Hint: standardize using stored mean/std (fit=False), then sigmoid(Xw + b)
        raise NotImplementedError("TODO: implement this")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return class predictions (0 or 1).

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            threshold: Decision boundary (default 0.5)

        Returns:
            Integer array of predictions, shape (n_samples,)
        """
        # Hint: predict_proba >= threshold -> 1, else -> 0
        raise NotImplementedError("TODO: implement this")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics from predictions.

    Returns dict with keys: accuracy, precision, recall, f1_score, confusion_matrix.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Dictionary of metrics
    """
    # Hint: compute TP, TN, FP, FN first
    # Hint: precision = TP / (TP + FP), recall = TP / (TP + FN)
    # Hint: handle division by zero when no positive predictions are made
    raise NotImplementedError("TODO: implement this")


def generate_binary_dataset(
    n_samples: int = 300, n_features: int = 2, separation: float = 1.5, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic binary classification dataset.

    Creates two Gaussian clusters with controllable separation.

    Args:
        n_samples: Total number of samples
        n_features: Number of features
        separation: Distance between cluster centers
        seed: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Hint: class 0 centered at origin, class 1 shifted by 'separation'
    # Hint: use np.random.RandomState(seed) for reproducibility
    # Hint: shuffle before splitting 80/20
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 003: LOGISTIC REGRESSION FROM SCRATCH")
    print("=" * 60)

    # Generate data
    X_train, X_test, y_train, y_test = generate_binary_dataset(
        n_samples=300, separation=1.5, seed=42
    )
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test")
    print(f"Class balance (train): {np.mean(y_train):.1%} positive")

    # Train
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X_train, y_train)

    print(f"\nFinal loss: {model.loss_history[-1]:.6f}")
    print(f"Learned weights: {model.weights}")
    print(f"Learned bias: {model.bias:.4f}")

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print(f"\nTest Accuracy:  {metrics['accuracy']:.1%}")
    print(f"Test Precision: {metrics['precision']:.1%}")
    print(f"Test Recall:    {metrics['recall']:.1%}")
    print(f"Test F1:        {metrics['f1_score']:.1%}")
