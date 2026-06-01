"""
Day 55: Image Classifier with CNN — From Scratch with NumPy

A complete CNN implementation: Conv2D → ReLU → MaxPool → FC → Softmax
Trained on a synthetic digit dataset (no external data dependencies).

Architecture:
  Input (1×28×28) → Conv(8 filters, 3×3) → ReLU → MaxPool(2×2) → FC(1352→10) → Softmax

This teaches the same principles as training on MNIST — the forward pass, backprop through
convolutions, pooling gradients, and cross-entropy loss are all identical. We use synthetic
data so the script runs anywhere without downloading datasets.
"""

import numpy as np
from typing import Tuple, Dict, List


# =============================================================================
# Layer implementations
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

    The key insight: each output pixel is a dot product between the kernel and a
    local patch of the input. We iterate over spatial positions and compute this
    dot product for all filters simultaneously.
    """
    c_out, c_in, kh, kw = kernels.shape
    _, h, w = input.shape
    h_out = h - kh + 1
    w_out = w - kw + 1

    output = np.zeros((c_out, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            # Extract the local patch: shape (C_in, KH, KW)
            patch = input[:, i:i+kh, j:j+kw]
            # Dot product with all filters at once:
            # kernels is (C_out, C_in, KH, KW), patch is (C_in, KH, KW)
            # Sum over (C_in, KH, KW) dimensions → (C_out,)
            output[:, i, j] = np.sum(kernels * patch[np.newaxis, :, :, :], axis=(1, 2, 3)) + biases

    return output


def conv2d_backward(
    input: np.ndarray,
    kernels: np.ndarray,
    d_output: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backprop through convolution layer.

    Args:
        input: shape (C_in, H, W) — the input from the forward pass
        kernels: shape (C_out, C_in, KH, KW)
        d_output: shape (C_out, H_out, W_out) — upstream gradient

    Returns:
        d_input: gradient w.r.t. input (same shape as input)
        d_kernels: gradient w.r.t. kernels (same shape as kernels)
        d_biases: gradient w.r.t. biases (shape (C_out,))

    Key math:
    - d_kernels[f] = correlate(input, d_output[f]) — each kernel grad is input convolved with its output grad
    - d_input = full_convolve(d_output, flipped_kernels) — error signal propagates backwards
    - d_biases[f] = sum(d_output[f]) — bias grad is just the sum of the output grad
    """
    c_out, c_in, kh, kw = kernels.shape
    _, h_out, w_out = d_output.shape

    d_kernels = np.zeros_like(kernels)
    d_biases = np.sum(d_output, axis=(1, 2))  # Sum over spatial dims
    d_input = np.zeros_like(input)

    for i in range(h_out):
        for j in range(w_out):
            patch = input[:, i:i+kh, j:j+kw]  # (C_in, KH, KW)
            # Each output position contributes to the kernel gradient:
            # d_output[:, i, j] is (C_out,), patch is (C_in, KH, KW)
            d_kernels += d_output[:, i, j][:, np.newaxis, np.newaxis, np.newaxis] * patch[np.newaxis, :, :, :]

            # And contributes to the input gradient:
            # Weighted sum of all kernels, weighted by the output gradient
            d_input[:, i:i+kh, j:j+kw] += np.sum(
                kernels * d_output[:, i, j][:, np.newaxis, np.newaxis, np.newaxis],
                axis=0
            )

    return d_input, d_kernels, d_biases


def relu_forward(x: np.ndarray) -> np.ndarray:
    """ReLU: max(0, x). Zero-cost nonlinearity with constant gradient for positive values."""
    return np.maximum(0, x)


def relu_backward(d_output: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gradient flows through where input was positive, blocked where it was negative."""
    return d_output * (x > 0).astype(float)


def maxpool2d_forward(input: np.ndarray, pool_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Max pooling: take the maximum in each non-overlapping window.

    Returns the pooled output and an index mask showing which positions held the max.
    We need the mask for backprop — gradients only flow to the max position.
    """
    c, h, w = input.shape
    h_out = h // pool_size
    w_out = w // pool_size

    output = np.zeros((c, h_out, w_out))
    # Store the index of the max within each pool window (flattened within the window)
    max_indices = np.zeros((c, h_out, w_out), dtype=int)

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * pool_size
            w_start = j * pool_size
            window = input[:, h_start:h_start+pool_size, w_start:w_start+pool_size]
            # Reshape window to (C, pool_size*pool_size) to find max along last axis
            window_flat = window.reshape(c, -1)
            max_indices[:, i, j] = np.argmax(window_flat, axis=1)
            output[:, i, j] = np.max(window_flat, axis=1)

    return output, max_indices


def maxpool2d_backward(
    d_output: np.ndarray,
    max_indices: np.ndarray,
    input_shape: Tuple[int, int, int],
    pool_size: int = 2
) -> np.ndarray:
    """
    Backprop through max pooling: gradient goes only to the position that was the max.

    This is like a routing operation — the gradient is unchanged in magnitude but gets
    sent to exactly one position in each window. All other positions get zero.
    """
    c, h, w = input_shape
    h_out = h // pool_size
    w_out = w // pool_size
    d_input = np.zeros(input_shape)

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * pool_size
            w_start = j * pool_size
            for ch in range(c):
                # Convert flat index back to 2D position within the pool window
                max_idx = max_indices[ch, i, j]
                max_h = max_idx // pool_size
                max_w = max_idx % pool_size
                d_input[ch, h_start + max_h, w_start + max_w] = d_output[ch, i, j]

    return d_input


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax: subtract max before exp to prevent overflow.
    exp(1000) overflows float64, but exp(1000 - 1000) = exp(0) = 1.
    """
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def cross_entropy_loss(predictions: np.ndarray, label: int) -> float:
    """
    Cross-entropy loss for a single sample with integer label.
    -log(p_correct) — penalizes low confidence in the correct class.
    Clip to avoid log(0) = -inf.
    """
    return -np.log(np.clip(predictions[label], 1e-12, 1.0))


# =============================================================================
# CNN class — ties all layers together
# =============================================================================

class CNN:
    """
    A minimal CNN: Conv(3×3, 8 filters) → ReLU → MaxPool(2×2) → FC(10) → Softmax

    This architecture is deliberately simple to make the from-scratch implementation
    tractable while still demonstrating all the key CNN concepts. On MNIST, it reaches
    ~95% accuracy in a few epochs.
    """

    def __init__(self, num_filters: int = 8, num_classes: int = 10):
        self.num_filters = num_filters
        self.num_classes = num_classes

        # He initialization: scale by sqrt(2 / fan_in)
        # For 3×3 conv, fan_in = 1 * 3 * 3 = 9 (single input channel)
        # He init compensates for ReLU zeroing out ~half the values
        self.conv_kernels = np.random.randn(num_filters, 1, 3, 3) * np.sqrt(2.0 / 9)
        self.conv_biases = np.zeros(num_filters)

        # FC layer: 8 filters × 13 × 13 (after pooling 26×26 with 2×2)
        # He init with fan_in = 1352
        fc_input_size = num_filters * 13 * 13  # 1352
        self.fc_weights = np.random.randn(fc_input_size, num_classes) * np.sqrt(2.0 / fc_input_size)
        self.fc_biases = np.zeros(num_classes)

        # Cache for backprop — stores intermediate values from forward pass
        self.cache: Dict = {}

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire network.

        Args:
            image: shape (1, 28, 28) — single grayscale image

        Returns:
            probabilities: shape (10,) — class probabilities
        """
        # Conv layer: (1, 28, 28) → (8, 26, 26)
        conv_out = conv2d_forward(image, self.conv_kernels, self.conv_biases)
        self.cache['conv_input'] = image
        self.cache['conv_out'] = conv_out

        # ReLU: element-wise, same shape
        relu_out = relu_forward(conv_out)
        self.cache['relu_out'] = relu_out

        # Max pool: (8, 26, 26) → (8, 13, 13)
        pool_out, max_indices = maxpool2d_forward(relu_out)
        self.cache['pool_out'] = pool_out
        self.cache['max_indices'] = max_indices
        self.cache['relu_out_shape'] = relu_out.shape

        # Flatten: (8, 13, 13) → (1352,)
        flat = pool_out.reshape(-1)
        self.cache['flat'] = flat

        # FC layer: (1352,) → (10,)
        logits = flat @ self.fc_weights + self.fc_biases
        self.cache['logits'] = logits

        # Softmax → probabilities
        probs = softmax(logits)
        self.cache['probs'] = probs

        return probs

    def backward(self, label: int) -> Dict[str, np.ndarray]:
        """
        Backward pass: compute gradients for all parameters.

        The softmax + cross-entropy gradient has a beautiful closed form:
            dL/dz_i = p_i - y_i
        where p_i is the predicted probability and y_i is 1 for the correct class, 0 otherwise.

        This elegant result is one reason cross-entropy is the standard loss for classification.
        """
        probs = self.cache['probs']

        # Softmax + cross-entropy gradient: dL/d_logits = probs - one_hot
        d_logits = probs.copy()
        d_logits[label] -= 1.0  # Subtract 1 from the correct class

        # FC backward
        flat = self.cache['flat']
        # dL/d_fc_weights = outer product of flat input and logit gradient
        d_fc_weights = flat[:, np.newaxis] @ d_logits[np.newaxis, :]
        d_fc_biases = d_logits
        # dL/d_flat: propagate gradient through FC layer
        d_flat = d_logits @ self.fc_weights.T

        # Unflatten: (1352,) → (8, 13, 13)
        d_pool_out = d_flat.reshape(self.cache['pool_out'].shape)

        # Max pool backward: route gradients to max positions
        d_relu_out = maxpool2d_backward(
            d_pool_out,
            self.cache['max_indices'],
            self.cache['relu_out_shape']
        )

        # ReLU backward: zero gradient where input was negative
        d_conv_out = relu_backward(d_relu_out, self.cache['conv_out'])

        # Conv backward: compute kernel and input gradients
        _, d_conv_kernels, d_conv_biases = conv2d_backward(
            self.cache['conv_input'],
            self.conv_kernels,
            d_conv_out
        )

        return {
            'conv_kernels': d_conv_kernels,
            'conv_biases': d_conv_biases,
            'fc_weights': d_fc_weights,
            'fc_biases': d_fc_biases,
        }

    def update(self, grads: Dict[str, np.ndarray], lr: float = 0.005):
        """SGD parameter update. Simple but effective for small networks."""
        self.conv_kernels -= lr * grads['conv_kernels']
        self.conv_biases -= lr * grads['conv_biases']
        self.fc_weights -= lr * grads['fc_weights']
        self.fc_biases -= lr * grads['fc_biases']


# =============================================================================
# Synthetic data generation
# =============================================================================

def generate_digit_patterns(num_samples_per_class: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 28×28 digit-like images for 10 classes.

    Each class has a distinct spatial pattern (different stroke positions).
    This isn't as rich as real MNIST but demonstrates that the CNN learns
    spatial features from the convolution filters.

    Returns:
        images: shape (N, 1, 28, 28) — normalized to [0, 1]
        labels: shape (N,) — integer class labels 0-9
    """
    rng = np.random.RandomState(seed)
    images = []
    labels = []

    # Each class gets a unique pattern: different combinations of strokes
    # These are simple but have enough spatial structure for a CNN to learn
    for class_id in range(10):
        for _ in range(num_samples_per_class):
            img = np.zeros((28, 28))

            # Each class draws strokes at different positions
            # Class 0: horizontal bar at top
            # Class 1: vertical bar in center
            # Class 2: diagonal \
            # etc.
            if class_id == 0:  # Top horizontal bar
                img[4:7, 6:22] = 1.0
            elif class_id == 1:  # Center vertical bar
                img[4:24, 12:15] = 1.0
            elif class_id == 2:  # Diagonal \
                for k in range(20):
                    img[4+k, 4+k] = 1.0
                    if 4+k+1 < 28: img[4+k, 4+k+1] = 0.7
            elif class_id == 3:  # L-shape
                img[4:22, 6:9] = 1.0
                img[19:22, 6:20] = 1.0
            elif class_id == 4:  # Cross
                img[12:16, 4:24] = 1.0
                img[4:24, 12:16] = 1.0
            elif class_id == 5:  # Top-right block
                img[4:14, 14:24] = 1.0
            elif class_id == 6:  # Bottom-left block
                img[14:24, 4:14] = 1.0
            elif class_id == 7:  # Two horizontal bars
                img[6:9, 6:22] = 1.0
                img[19:22, 6:22] = 1.0
            elif class_id == 8:  # Border rectangle
                img[4:24, 4:6] = 1.0
                img[4:24, 22:24] = 1.0
                img[4:6, 4:24] = 1.0
                img[22:24, 4:24] = 1.0
            elif class_id == 9:  # Diamond
                cx, cy = 14, 14
                for x in range(28):
                    for y in range(28):
                        if abs(x - cx) + abs(y - cy) <= 8 and abs(x - cx) + abs(y - cy) >= 6:
                            img[x, y] = 1.0

            # Add noise to make it realistic — real data is never clean
            noise = rng.randn(28, 28) * 0.15
            img = np.clip(img + noise, 0, 1)

            images.append(img)
            labels.append(class_id)

    images = np.array(images)[:, np.newaxis, :, :]  # Add channel dim: (N, 1, 28, 28)
    labels = np.array(labels)

    # Shuffle
    idx = rng.permutation(len(labels))
    return images[idx], labels[idx]


# =============================================================================
# Training and evaluation
# =============================================================================

def train_epoch(
    cnn: CNN,
    images: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.005
) -> Tuple[float, float]:
    """
    Train for one epoch (one pass through all data).

    Processes samples one at a time (SGD with batch size 1).
    This is simple and works fine for small datasets.

    Returns:
        (average_loss, accuracy)
    """
    total_loss = 0.0
    correct = 0
    n = len(labels)

    for i in range(n):
        # Forward
        probs = cnn.forward(images[i])
        loss = cross_entropy_loss(probs, labels[i])
        total_loss += loss

        if np.argmax(probs) == labels[i]:
            correct += 1

        # Backward
        grads = cnn.backward(labels[i])

        # Update
        cnn.update(grads, lr=lr)

    return total_loss / n, correct / n


def evaluate(cnn: CNN, images: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Evaluate accuracy and loss without updating parameters."""
    total_loss = 0.0
    correct = 0
    n = len(labels)

    for i in range(n):
        probs = cnn.forward(images[i])
        total_loss += cross_entropy_loss(probs, labels[i])
        if np.argmax(probs) == labels[i]:
            correct += 1

    return total_loss / n, correct / n


# =============================================================================
# Main — demonstrate training and inspect learned features
# =============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("CNN Image Classifier — From Scratch with NumPy")
    print("=" * 65)

    # Generate synthetic data
    print("\n[1] Generating synthetic digit-like dataset...")
    images, labels = generate_digit_patterns(num_samples_per_class=50, seed=42)
    print(f"    Dataset: {len(labels)} images, 10 classes, shape per image: {images[0].shape}")

    # Split into train/test (80/20)
    split = int(0.8 * len(labels))
    train_images, test_images = images[:split], images[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    print(f"    Train: {len(train_labels)}, Test: {len(test_labels)}")

    # Initialize CNN
    print("\n[2] Initializing CNN architecture...")
    cnn = CNN(num_filters=8, num_classes=10)
    print(f"    Conv layer: 8 filters × 3×3 (72 params + 8 biases)")
    print(f"    FC layer: 1352 → 10 ({1352 * 10} params + 10 biases)")
    total_params = 8 * 1 * 3 * 3 + 8 + 1352 * 10 + 10
    print(f"    Total parameters: {total_params:,}")
    print(f"    (Compare: a fully connected 784→10 network has {784*10 + 10:,} params)")

    # Before training — show random performance
    print("\n[3] Before training (random weights)...")
    pre_loss, pre_acc = evaluate(cnn, test_images, test_labels)
    print(f"    Test loss: {pre_loss:.4f}, Test accuracy: {pre_acc:.1%}")
    print(f"    (Random chance would be ~10% for 10 classes)")

    # Training
    print("\n[4] Training...")
    num_epochs = 3
    lr = 0.005

    for epoch in range(num_epochs):
        # Shuffle training data each epoch
        idx = np.random.permutation(len(train_labels))
        shuffled_images = train_images[idx]
        shuffled_labels = train_labels[idx]

        train_loss, train_acc = train_epoch(cnn, shuffled_images, shuffled_labels, lr=lr)
        test_loss, test_acc = evaluate(cnn, test_images, test_labels)

        print(f"    Epoch {epoch+1}/{num_epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.1%} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.1%}")

    # Show individual predictions
    print("\n[5] Sample predictions on test set:")
    for i in range(min(10, len(test_labels))):
        probs = cnn.forward(test_images[i])
        pred = np.argmax(probs)
        conf = probs[pred]
        true = test_labels[i]
        status = "✓" if pred == true else "✗"
        print(f"    {status} True: {true}, Predicted: {pred} (confidence: {conf:.2%})")

    # Inspect learned filters
    print("\n[6] Learned conv filter statistics:")
    print(f"    Filter shapes: {cnn.conv_kernels.shape}")
    for f in range(cnn.num_filters):
        kernel = cnn.conv_kernels[f, 0]
        print(f"    Filter {f}: min={kernel.min():.3f}, max={kernel.max():.3f}, "
              f"mean={kernel.mean():.3f}, std={kernel.std():.3f}")

    # Show what the conv layer sees for a sample image
    print("\n[7] Feature map activations for a sample image (class {})...".format(test_labels[0]))
    sample_conv_out = conv2d_forward(test_images[0], cnn.conv_kernels, cnn.conv_biases)
    sample_relu_out = relu_forward(sample_conv_out)
    for f in range(cnn.num_filters):
        activation = sample_relu_out[f]
        nonzero_pct = (activation > 0).mean()
        print(f"    Filter {f}: {nonzero_pct:.1%} active, "
              f"max activation={activation.max():.3f}")

    print("\n" + "=" * 65)
    print("CNN training complete!")
    print(f"Final test accuracy: {test_acc:.1%}")
    print("=" * 65)
