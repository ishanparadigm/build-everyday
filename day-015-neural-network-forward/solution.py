"""
Day 015: Simple Neural Network — Forward Pass

A complete feedforward neural network implementation using only NumPy.
We build the forward pass from scratch: weight initialization, activation
functions, layer-by-layer computation, loss calculation, and prediction.

This implementation handles multi-class classification on synthetic data,
demonstrating how raw numbers flow through a network to become predictions.
"""

import numpy as np
from typing import List, Tuple, Dict


# =============================================================================
# Activation Functions
# =============================================================================
# Each activation returns both the output and caches the input (needed for
# backpropagation in future challenges). We implement them as pure functions
# operating on NumPy arrays.

def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU: f(z) = max(0, z)

    The simplest nonlinearity. Passes positive values unchanged,
    kills negative values. This creates sparse activations (many zeros),
    which is actually beneficial — it acts as a form of implicit regularization.

    Gradient: 1 if z > 0, 0 if z <= 0
    Problem: "dying ReLU" — if a neuron's weights push it permanently negative,
    it outputs 0 forever and receives 0 gradient. It can never recover.
    """
    return np.maximum(0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid: f(z) = 1 / (1 + exp(-z))

    Squashes any real number into (0, 1). Historically the default activation,
    now mostly used in output layers for binary classification.

    Gradient: sigmoid(z) * (1 - sigmoid(z))
    Problem: For |z| > 5, the gradient is essentially 0 ("vanishing gradient").
    In deep networks, this compounds across layers, making early layers untrainable.

    Numerical stability: For very negative z, exp(-z) overflows. We use the
    identity: sigmoid(z) = exp(z) / (1 + exp(z)) when z < 0 to avoid overflow.
    """
    # Clip to avoid overflow in exp
    # For z >= 0: 1 / (1 + exp(-z)) — standard form
    # For z < 0: exp(z) / (1 + exp(z)) — numerically stable form
    result = np.zeros_like(z, dtype=np.float64)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[neg_mask])
    result[neg_mask] = exp_z / (1.0 + exp_z)
    return result


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax: softmax(z_i) = exp(z_i) / sum(exp(z_j))

    Converts a vector of raw scores (logits) into a probability distribution.
    All outputs are positive and sum to 1.

    The key numerical stability trick: subtract max(z) before exponentiating.
    This doesn't change the result (it cancels out in numerator/denominator)
    but prevents exp() from overflowing:

        exp(z_i - max(z)) / sum(exp(z_j - max(z))) = exp(z_i) / sum(exp(z_j))

    We apply softmax row-wise: each sample in the batch gets its own distribution.
    """
    # Subtract max for numerical stability (per row for batched input)
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# =============================================================================
# Weight Initialization
# =============================================================================

def initialize_weights(layer_sizes: List[int], seed: int = 42) -> List[Dict[str, np.ndarray]]:
    """
    Initialize network parameters using He initialization.

    He initialization: W ~ N(0, sqrt(2/n_in))

    Why sqrt(2/n_in)? Consider a neuron computing z = w1*x1 + w2*x2 + ... + wn*xn.
    If inputs have variance 1 and weights have variance sigma^2, then:
        Var(z) = n_in * sigma^2 * Var(x)

    For ReLU, half the outputs are zeroed, so we need 2x the variance to compensate:
        sigma^2 = 2 / n_in

    This keeps the variance of activations roughly constant across layers,
    preventing signals from exploding or vanishing before we even start training.

    Args:
        layer_sizes: List of layer widths, e.g. [784, 128, 64, 10]
        seed: Random seed for reproducibility

    Returns:
        List of dicts, each containing 'W' (weight matrix) and 'b' (bias vector)
    """
    rng = np.random.RandomState(seed)
    params = []

    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]

        # He initialization: scale by sqrt(2/n_in) for ReLU activations
        W = rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

        # Biases start at zero — they don't have the symmetry-breaking problem
        # that weights do, because each neuron's bias is independent
        b = np.zeros((1, n_out))

        params.append({'W': W, 'b': b})

    return params


# =============================================================================
# Neural Network Class
# =============================================================================

class NeuralNetwork:
    """
    A feedforward neural network for multi-class classification.

    Architecture: Input → [Hidden layers with ReLU] → Output with Softmax

    The network stores all intermediate computations during the forward pass
    in a 'cache' — this is essential for backpropagation (computing gradients),
    which we'll implement in a future challenge.

    Design decisions:
    - ReLU for hidden layers: fast, effective, standard choice
    - Softmax output: produces valid probability distributions for classification
    - Cross-entropy loss: the natural pairing with softmax (they simplify nicely
      when computing gradients: dL/dz = y_pred - y_true)
    """

    def __init__(self, layer_sizes: List[int], seed: int = 42):
        """
        Args:
            layer_sizes: Dimensions of each layer including input and output.
                         Example: [784, 128, 64, 10] for MNIST
                         - 784 input features (28x28 pixels)
                         - 128 neurons in first hidden layer
                         - 64 neurons in second hidden layer
                         - 10 output classes (digits 0-9)
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # Number of weight matrices
        self.params = initialize_weights(layer_sizes, seed)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
        """
        Forward pass: propagate input through all layers to produce predictions.

        For each layer l:
            z_l = a_{l-1} @ W_l + b_l    (linear transformation)
            a_l = activation(z_l)          (nonlinear activation)

        Hidden layers use ReLU, output layer uses softmax.

        Args:
            X: Input data, shape (batch_size, n_features)

        Returns:
            output: Predicted probabilities, shape (batch_size, n_classes)
            cache: List of intermediate values {z, a_prev} per layer (for backprop)
        """
        cache = []
        a = X  # First "activation" is just the input

        for l in range(self.n_layers):
            a_prev = a
            W = self.params[l]['W']
            b = self.params[l]['b']

            # Linear transformation: z = a_prev @ W + b
            # Shape: (batch_size, n_prev) @ (n_prev, n_current) = (batch_size, n_current)
            # Broadcasting handles the bias: (batch_size, n_current) + (1, n_current)
            z = a_prev @ W + b

            # Apply activation function
            if l < self.n_layers - 1:
                # Hidden layers: ReLU
                a = relu(z)
            else:
                # Output layer: softmax for classification
                a = softmax(z)

            # Cache everything needed for backpropagation
            cache.append({
                'z': z,           # Pre-activation values
                'a_prev': a_prev  # Input to this layer
            })

        return a, cache

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Cross-entropy loss for multi-class classification.

        L = -1/m * sum(sum(y_true * log(y_pred)))

        Where y_true is one-hot encoded and y_pred is softmax output.

        For a single sample with true class k:
            L = -log(y_pred[k])

        This means: if the network assigns probability 0.9 to the correct class,
        loss = -log(0.9) = 0.105. If it assigns 0.01, loss = -log(0.01) = 4.6.
        The penalty grows *logarithmically* as confidence in the correct class drops.

        We clip y_pred to avoid log(0) = -inf. In practice, softmax never outputs
        exactly 0, but floating point can round very small values to 0.

        Args:
            y_pred: Predicted probabilities, shape (m, n_classes)
            y_true: One-hot encoded labels, shape (m, n_classes)

        Returns:
            Scalar loss value (average over the batch)
        """
        m = y_true.shape[0]
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        # Cross-entropy: only the true class contributes (y_true is one-hot)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Runs the forward pass, then takes argmax of output probabilities.
        argmax returns the index of the highest probability — the predicted class.

        Args:
            X: Input data, shape (batch_size, n_features)

        Returns:
            Predicted class indices, shape (batch_size,)
        """
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y_true_labels: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Input data
            y_true_labels: True class indices (NOT one-hot), shape (batch_size,)

        Returns:
            Fraction of correct predictions
        """
        predictions = self.predict(X)
        return np.mean(predictions == y_true_labels)


# =============================================================================
# Data Utilities
# =============================================================================

def one_hot_encode(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.

    Example: label=2, n_classes=4 → [0, 0, 1, 0]

    One-hot encoding is necessary because cross-entropy loss expects a
    probability distribution as the target (even though it's a degenerate
    distribution with all mass on one class).
    """
    m = labels.shape[0]
    one_hot = np.zeros((m, n_classes))
    one_hot[np.arange(m), labels] = 1.0
    return one_hot


def generate_spiral_data(
    n_samples_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a spiral dataset — a classic nonlinear classification problem.

    Each class forms a spiral arm in 2D space. A linear classifier CANNOT
    separate these classes — you need nonlinear decision boundaries,
    which is exactly what neural networks provide through their activation functions.

    This is a perfect test case because:
    1. It's simple to visualize (2D)
    2. It requires nonlinear decision boundaries
    3. Difficulty scales with noise level
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples_per_class * n_classes, 2))
    y = np.zeros(n_samples_per_class * n_classes, dtype=int)

    for class_idx in range(n_classes):
        start = n_samples_per_class * class_idx
        end = n_samples_per_class * (class_idx + 1)

        # Parametric spiral: r increases with t, angle offset by class
        t = np.linspace(0, 1, n_samples_per_class)
        r = t * 5  # Radius grows linearly
        theta = t * 4 * np.pi + (2 * np.pi / n_classes) * class_idx  # 2 full turns

        X[start:end, 0] = r * np.sin(theta) + rng.randn(n_samples_per_class) * noise
        X[start:end, 1] = r * np.cos(theta) + rng.randn(n_samples_per_class) * noise
        y[start:end] = class_idx

    return X, y


# =============================================================================
# Main: Demonstrate the Forward Pass
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("NEURAL NETWORK FORWARD PASS — FROM SCRATCH")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Generate Data
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Generate Spiral Dataset ---")
    X, y = generate_spiral_data(n_samples_per_class=100, n_classes=3, noise=0.3)
    y_one_hot = one_hot_encode(y, n_classes=3)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)} with {np.bincount(y)} samples each")
    print(f"Feature ranges: x1=[{X[:,0].min():.2f}, {X[:,0].max():.2f}], "
          f"x2=[{X[:,1].min():.2f}, {X[:,1].max():.2f}]")

    # -------------------------------------------------------------------------
    # Step 2: Create Network
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Initialize Neural Network ---")
    # Architecture: 2 inputs → 64 hidden → 32 hidden → 3 outputs
    # Why these sizes?
    # - 2 inputs: our data is 2D
    # - 64, 32 hidden: enough capacity for spiral boundaries,
    #   decreasing width encourages information compression
    # - 3 outputs: one per class
    layer_sizes = [2, 64, 32, 3]
    nn = NeuralNetwork(layer_sizes, seed=42)

    print(f"Architecture: {layer_sizes}")
    print(f"Number of layers: {nn.n_layers}")
    for i, p in enumerate(nn.params):
        n_params = p['W'].size + p['b'].size
        print(f"  Layer {i+1}: W{p['W'].shape} + b{p['b'].shape} = {n_params} parameters")
    total_params = sum(p['W'].size + p['b'].size for p in nn.params)
    print(f"Total parameters: {total_params}")

    # -------------------------------------------------------------------------
    # Step 3: Examine Weight Initialization
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Weight Initialization Statistics ---")
    print("He initialization keeps variance ≈ 2/n_in across layers:")
    for i, p in enumerate(nn.params):
        n_in = p['W'].shape[0]
        expected_std = np.sqrt(2.0 / n_in)
        actual_std = np.std(p['W'])
        print(f"  Layer {i+1}: expected std={expected_std:.4f}, "
              f"actual std={actual_std:.4f}, "
              f"mean={np.mean(p['W']):.6f}")

    # -------------------------------------------------------------------------
    # Step 4: Demonstrate Activation Functions
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Activation Functions ---")
    test_vals = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    print(f"Input:   {test_vals[0]}")
    print(f"ReLU:    {relu(test_vals)[0]}")
    print(f"Sigmoid: {np.round(sigmoid(test_vals), 4)[0]}")

    test_logits = np.array([[2.0, 1.0, 0.1]])
    sm = softmax(test_logits)
    print(f"\nSoftmax demo:")
    print(f"  Input logits: {test_logits[0]}")
    print(f"  Output probs: {np.round(sm, 4)[0]}")
    print(f"  Sum to 1?     {np.sum(sm):.6f}")

    # -------------------------------------------------------------------------
    # Step 5: Run Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Forward Pass ---")

    # Process a small batch to show intermediate values
    X_batch = X[:5]
    y_batch = y[:5]
    y_batch_one_hot = y_one_hot[:5]

    output, cache = nn.forward(X_batch)

    print(f"Input batch shape: {X_batch.shape}")
    print(f"Input (first sample): {X_batch[0]}")

    for i, c in enumerate(cache):
        act_name = "ReLU" if i < nn.n_layers - 1 else "Softmax"
        z_stats = f"mean={np.mean(c['z']):.4f}, std={np.std(c['z']):.4f}"
        print(f"\n  Layer {i+1} ({act_name}):")
        print(f"    Pre-activation z shape: {c['z'].shape}, {z_stats}")
        if i < nn.n_layers - 1:
            activated = relu(c['z'])
            pct_dead = np.mean(activated == 0) * 100
            print(f"    Dead neurons (z <= 0): {pct_dead:.1f}%")

    print(f"\nOutput probabilities (first 5 samples):")
    for i in range(5):
        pred_class = np.argmax(output[i])
        true_class = y_batch[i]
        conf = output[i][pred_class]
        print(f"  Sample {i}: probs={np.round(output[i], 4)}, "
              f"predicted={pred_class}, true={true_class}, "
              f"confidence={conf:.4f}")

    # -------------------------------------------------------------------------
    # Step 6: Compute Loss and Accuracy
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Loss and Accuracy (full dataset) ---")

    full_output, _ = nn.forward(X)
    loss = nn.compute_loss(full_output, y_one_hot)
    acc = nn.accuracy(X, y)

    # For 3 classes, random guessing gives 33.3% accuracy
    # and loss of -log(1/3) = 1.099
    random_loss = -np.log(1.0 / 3)

    print(f"Cross-entropy loss: {loss:.4f}")
    print(f"Random baseline loss: {random_loss:.4f}")
    print(f"Classification accuracy: {acc*100:.1f}%")
    print(f"Random baseline accuracy: 33.3%")
    print(f"\nNote: Without training, the network performs near random chance.")
    print(f"The forward pass works correctly — it just needs trained weights")
    print(f"(via backpropagation) to make accurate predictions.")

    # -------------------------------------------------------------------------
    # Step 7: Show Effect of Network Depth
    # -------------------------------------------------------------------------
    print("\n--- Step 7: Effect of Architecture on Output ---")

    architectures = [
        [2, 3],            # No hidden layers — linear classifier
        [2, 16, 3],        # 1 hidden layer
        [2, 64, 32, 3],    # 2 hidden layers (our network)
        [2, 128, 64, 32, 3]  # 3 hidden layers — deeper
    ]

    for arch in architectures:
        net = NeuralNetwork(arch, seed=42)
        out, _ = net.forward(X)
        loss_val = net.compute_loss(out, y_one_hot)
        acc_val = net.accuracy(X, y)
        n_params = sum(p['W'].size + p['b'].size for p in net.params)
        print(f"  {str(arch):25s} | params={n_params:5d} | "
              f"loss={loss_val:.4f} | acc={acc_val*100:.1f}%")

    print(f"\nAll architectures perform similarly before training — the structure")
    print(f"determines capacity (what CAN be learned), not initial performance.")

    # -------------------------------------------------------------------------
    # Step 8: Mini-batch Processing
    # -------------------------------------------------------------------------
    print("\n--- Step 8: Mini-batch Processing ---")
    batch_size = 32
    n_batches = len(X) // batch_size

    print(f"Dataset size: {len(X)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {n_batches}")

    # Shuffle and create batches
    rng = np.random.RandomState(0)
    indices = rng.permutation(len(X))

    batch_losses = []
    for b in range(min(n_batches, 5)):  # Show first 5 batches
        batch_idx = indices[b * batch_size:(b + 1) * batch_size]
        X_b = X[batch_idx]
        y_b = y_one_hot[batch_idx]

        out_b, _ = nn.forward(X_b)
        loss_b = nn.compute_loss(out_b, y_b)
        batch_losses.append(loss_b)
        print(f"  Batch {b+1}: loss={loss_b:.4f}, "
              f"mean_confidence={np.max(out_b, axis=1).mean():.4f}")

    print(f"\n  Loss variance across batches: {np.var(batch_losses):.6f}")
    print(f"  (Low variance = consistent predictions across the dataset)")

    print("\n" + "=" * 70)
    print("FORWARD PASS COMPLETE")
    print("=" * 70)
    print("\nThe network can process inputs and produce predictions.")
    print("Next step: backpropagation to compute gradients and train the weights.")
