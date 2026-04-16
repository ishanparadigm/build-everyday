"""
Day 015: Simple Neural Network — Forward Pass (Your Implementation)

Build a feedforward neural network from scratch using only NumPy.
Implement activation functions, weight initialization, and the forward pass.

Hint: The forward pass is just alternating matrix multiplications and
element-wise nonlinearities. Think shapes: (batch, n_prev) @ (n_prev, n_curr).
"""

import numpy as np
from typing import List, Tuple, Dict


# =============================================================================
# Activation Functions
# =============================================================================

def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU activation: f(z) = max(0, z)

    Hint: np.maximum does element-wise max. One line.
    """
    raise NotImplementedError("TODO: implement this")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: f(z) = 1 / (1 + exp(-z))

    Hint: Handle numerical stability — for very negative z, exp(-z) overflows.
    Consider using different formulas for z >= 0 and z < 0.
    """
    raise NotImplementedError("TODO: implement this")


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax: converts logits to probabilities. Output sums to 1 per row.

    Hint: Subtract max(z) per row before exp() for numerical stability.
    Use keepdims=True to maintain broadcasting dimensions.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Weight Initialization
# =============================================================================

def initialize_weights(layer_sizes: List[int], seed: int = 42) -> List[Dict[str, np.ndarray]]:
    """
    Initialize network parameters using He initialization.

    He init: W ~ N(0, sqrt(2/n_in))
    Biases: zeros with shape (1, n_out)

    Args:
        layer_sizes: e.g. [2, 64, 32, 3]
        seed: Random seed

    Returns:
        List of {'W': weight_matrix, 'b': bias_vector} per layer

    Hint: Loop over consecutive pairs in layer_sizes.
    Use np.random.RandomState(seed) for reproducibility.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Neural Network Class
# =============================================================================

class NeuralNetwork:
    """
    Feedforward neural network for multi-class classification.

    Architecture: Input → [Hidden with ReLU] → Output with Softmax
    """

    def __init__(self, layer_sizes: List[int], seed: int = 42):
        """
        Hint: Store layer_sizes, compute n_layers, initialize params.
        """
        raise NotImplementedError("TODO: implement this")

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
        """
        Forward pass through all layers.

        For each layer:
            z = a_prev @ W + b    (linear transformation)
            a = activation(z)      (ReLU for hidden, softmax for output)

        Args:
            X: shape (batch_size, n_features)

        Returns:
            output: shape (batch_size, n_classes) — probabilities
            cache: list of {'z': ..., 'a_prev': ...} per layer

        Hint: Track the current activation (starts as X). Loop through layers.
        Last layer uses softmax, all others use ReLU.
        """
        raise NotImplementedError("TODO: implement this")

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Cross-entropy loss: L = -1/m * sum(y_true * log(y_pred))

        Args:
            y_pred: Softmax output, shape (m, n_classes)
            y_true: One-hot labels, shape (m, n_classes)

        Hint: Clip y_pred to avoid log(0). y_true is one-hot so only
        one term per row contributes to the sum.
        """
        raise NotImplementedError("TODO: implement this")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (integers).

        Hint: Forward pass → argmax of output probabilities.
        """
        raise NotImplementedError("TODO: implement this")

    def accuracy(self, X: np.ndarray, y_true_labels: np.ndarray) -> float:
        """
        Fraction of correct predictions.

        Args:
            X: Input data
            y_true_labels: True class indices (NOT one-hot)
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Data Utilities
# =============================================================================

def one_hot_encode(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot vectors.

    Example: label=2, n_classes=4 → [0, 0, 1, 0]

    Hint: Create a zeros array, then use advanced indexing to set the 1s.
    """
    raise NotImplementedError("TODO: implement this")


def generate_spiral_data(
    n_samples_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spiral dataset for nonlinear classification testing.

    Hint: For each class, create points along a spiral arm using
    parametric equations: x = r*sin(theta), y = r*cos(theta)
    where r increases with t and theta is offset per class.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Test Your Implementation
# =============================================================================

if __name__ == '__main__':
    # Test activations
    test = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    print("ReLU:", relu(test))
    print("Sigmoid:", sigmoid(test))
    print("Softmax:", softmax(np.array([[2.0, 1.0, 0.1]])))

    # Generate data
    X, y = generate_spiral_data()
    y_oh = one_hot_encode(y, 3)
    print(f"\nData: X={X.shape}, y={y.shape}")

    # Build and run network
    nn = NeuralNetwork([2, 64, 32, 3])
    output, cache = nn.forward(X)
    loss = nn.compute_loss(output, y_oh)
    acc = nn.accuracy(X, y)

    print(f"Output shape: {output.shape}")
    print(f"Probabilities sum to 1? {np.allclose(output.sum(axis=1), 1.0)}")
    print(f"Loss: {loss:.4f} (random baseline: {-np.log(1/3):.4f})")
    print(f"Accuracy: {acc*100:.1f}% (random baseline: 33.3%)")
