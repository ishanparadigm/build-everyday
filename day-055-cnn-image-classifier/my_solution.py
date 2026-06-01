"""
Day 55: Image Classifier with CNN — Your Implementation

Build a CNN from scratch using only NumPy. Implement each layer's forward and
backward pass, then train on synthetic digit data.

Architecture:
  Input (1×28×28) → Conv(8 filters, 3×3) → ReLU → MaxPool(2×2) → FC(1352→10) → Softmax

Run tests: python3 -m pytest tests.py -v
"""

import numpy as np
from typing import Tuple, Dict


# =============================================================================
# Layer implementations — implement each forward and backward pass
# =============================================================================

def conv2d_forward(input: np.ndarray, kernels: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """
    2D convolution forward pass (cross-correlation, no padding, stride=1).

    Args:
        input: shape (C_in, H, W) — single image with C_in channels
        kernels: shape (C_out, C_in, KH, KW) — C_out filters
        biases: shape (C_out,)

    Returns:
        output: shape (C_out, H_out, W_out) where H_out = H - KH + 1

    Hint: Slide the kernel across each spatial position. At each position,
    compute the element-wise product of the kernel and the input patch,
    then sum over all dimensions (C_in, KH, KW).
    """
    raise NotImplementedError("TODO: implement conv2d forward pass")


def conv2d_backward(
    input: np.ndarray,
    kernels: np.ndarray,
    d_output: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backprop through convolution layer.

    Args:
        input: shape (C_in, H, W)
        kernels: shape (C_out, C_in, KH, KW)
        d_output: shape (C_out, H_out, W_out) — upstream gradient

    Returns:
        d_input, d_kernels, d_biases

    Hint: At each spatial position (i, j), the output gradient d_output[:, i, j]
    tells you how much error flowed through that position.
    - d_kernels: accumulate input_patch * d_output for each position
    - d_input: accumulate kernels * d_output for each position
    - d_biases: sum d_output over spatial dims
    """
    raise NotImplementedError("TODO: implement conv2d backward pass")


def relu_forward(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation: max(0, x), applied element-wise.

    Hint: One line with np.maximum.
    """
    raise NotImplementedError("TODO: implement ReLU forward")


def relu_backward(d_output: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    ReLU backward: gradient is 1 where x > 0, else 0.

    Hint: Multiply d_output by a mask of where x was positive.
    """
    raise NotImplementedError("TODO: implement ReLU backward")


def maxpool2d_forward(input: np.ndarray, pool_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Max pooling with non-overlapping windows.

    Args:
        input: shape (C, H, W)
        pool_size: window size (default 2)

    Returns:
        output: shape (C, H//pool_size, W//pool_size)
        max_indices: which position in each window had the max (for backprop)

    Hint: For each window, reshape to flat, find argmax and max.
    Store argmax indices — you need them in the backward pass.
    """
    raise NotImplementedError("TODO: implement max pooling forward")


def maxpool2d_backward(
    d_output: np.ndarray,
    max_indices: np.ndarray,
    input_shape: Tuple[int, int, int],
    pool_size: int = 2
) -> np.ndarray:
    """
    Max pooling backward: route gradient to the max position in each window.

    Hint: Create a zero array of input_shape. For each output position, convert
    the stored max_index back to 2D coordinates and place the gradient there.
    """
    raise NotImplementedError("TODO: implement max pooling backward")


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.

    Hint: Subtract max(logits) before exp to prevent overflow.
    """
    raise NotImplementedError("TODO: implement softmax")


def cross_entropy_loss(predictions: np.ndarray, label: int) -> float:
    """
    Cross-entropy loss for single sample with integer label.

    Hint: -log(predictions[label]), but clip to avoid log(0).
    """
    raise NotImplementedError("TODO: implement cross-entropy loss")


# =============================================================================
# CNN class
# =============================================================================

class CNN:
    """
    Conv(3×3, 8 filters) → ReLU → MaxPool(2×2) → FC(1352→10) → Softmax

    Hint: Initialize kernels with He initialization: N(0, sqrt(2/fan_in))
    """

    def __init__(self, num_filters: int = 8, num_classes: int = 10):
        raise NotImplementedError("TODO: initialize CNN parameters")

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Forward pass: image (1, 28, 28) → class probabilities (10,)

        Hint: Chain the layers: conv → relu → pool → flatten → FC → softmax.
        Cache intermediate values for backprop.
        """
        raise NotImplementedError("TODO: implement CNN forward pass")

    def backward(self, label: int) -> Dict[str, np.ndarray]:
        """
        Backward pass: compute gradients for all parameters.

        Hint: Start from softmax+CE gradient: d_logits = probs - one_hot(label)
        Then propagate backwards through FC → unflatten → pool → relu → conv.
        """
        raise NotImplementedError("TODO: implement CNN backward pass")

    def update(self, grads: Dict[str, np.ndarray], lr: float = 0.005):
        """SGD update: param -= lr * grad"""
        raise NotImplementedError("TODO: implement parameter update")


# =============================================================================
# Data generation (provided — same as solution.py)
# =============================================================================

def generate_digit_patterns(num_samples_per_class: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 28×28 digit-like images for 10 classes."""
    rng = np.random.RandomState(seed)
    images = []
    labels = []

    for class_id in range(10):
        for _ in range(num_samples_per_class):
            img = np.zeros((28, 28))
            if class_id == 0:
                img[4:7, 6:22] = 1.0
            elif class_id == 1:
                img[4:24, 12:15] = 1.0
            elif class_id == 2:
                for k in range(20):
                    img[4+k, 4+k] = 1.0
                    if 4+k+1 < 28: img[4+k, 4+k+1] = 0.7
            elif class_id == 3:
                img[4:22, 6:9] = 1.0
                img[19:22, 6:20] = 1.0
            elif class_id == 4:
                img[12:16, 4:24] = 1.0
                img[4:24, 12:16] = 1.0
            elif class_id == 5:
                img[4:14, 14:24] = 1.0
            elif class_id == 6:
                img[14:24, 4:14] = 1.0
            elif class_id == 7:
                img[6:9, 6:22] = 1.0
                img[19:22, 6:22] = 1.0
            elif class_id == 8:
                img[4:24, 4:6] = 1.0
                img[4:24, 22:24] = 1.0
                img[4:6, 4:24] = 1.0
                img[22:24, 4:24] = 1.0
            elif class_id == 9:
                cx, cy = 14, 14
                for x in range(28):
                    for y in range(28):
                        if abs(x - cx) + abs(y - cy) <= 8 and abs(x - cx) + abs(y - cy) >= 6:
                            img[x, y] = 1.0

            noise = rng.randn(28, 28) * 0.15
            img = np.clip(img + noise, 0, 1)
            images.append(img)
            labels.append(class_id)

    images = np.array(images)[:, np.newaxis, :, :]
    labels = np.array(labels)
    idx = rng.permutation(len(labels))
    return images[idx], labels[idx]


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == '__main__':
    print("Generating data...")
    images, labels = generate_digit_patterns(num_samples_per_class=50, seed=42)
    split = int(0.8 * len(labels))
    train_images, test_images = images[:split], images[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    print(f"Train: {len(train_labels)}, Test: {len(test_labels)}")

    print("\nInitializing CNN...")
    cnn = CNN(num_filters=8, num_classes=10)

    print("\nTraining for 3 epochs...")
    for epoch in range(3):
        total_loss = 0.0
        correct = 0
        idx = np.random.permutation(len(train_labels))
        for i in idx:
            probs = cnn.forward(train_images[i])
            total_loss += cross_entropy_loss(probs, train_labels[i])
            if np.argmax(probs) == train_labels[i]:
                correct += 1
            grads = cnn.backward(train_labels[i])
            cnn.update(grads, lr=0.005)

        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_labels):.4f} "
              f"acc={correct/len(train_labels):.1%}")

    # Test
    correct = 0
    for i in range(len(test_labels)):
        probs = cnn.forward(test_images[i])
        if np.argmax(probs) == test_labels[i]:
            correct += 1
    print(f"\nTest accuracy: {correct/len(test_labels):.1%}")
