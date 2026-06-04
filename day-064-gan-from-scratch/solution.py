"""
Day 064: Generative Adversarial Network (GAN) from Scratch

A complete GAN training loop using only NumPy. A generator learns to produce
2D data matching a mixture of Gaussians, while a discriminator tries to tell
real from fake. Implements manual backpropagation, non-saturating generator
loss, and training diagnostics.
"""

import numpy as np
from typing import List, Tuple, Dict


# =============================================================================
# Activation functions and their derivatives
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0."""
    return (x > 0).astype(float)

def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """LeakyReLU: x if x > 0, else alpha * x.

    Why LeakyReLU for the discriminator? Standard ReLU kills gradients for
    negative inputs. Since the discriminator sees data centered around zero,
    many pre-activations will be negative. LeakyReLU preserves gradient flow
    through these neurons, keeping the discriminator trainable.
    """
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Derivative of LeakyReLU: 1 if x > 0, else alpha."""
    return np.where(x > 0, 1.0, alpha)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-x)).

    Clips input to [-500, 500] to prevent overflow. For very negative x,
    sigmoid ≈ 0; for very positive x, sigmoid ≈ 1. We never need values
    outside this range in practice.
    """
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation: bounds output to [-1, 1].

    Used as the generator's output activation. This bounds the generated
    data, which helps training stability — the discriminator doesn't need
    to handle arbitrarily large inputs.
    """
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh²(x)."""
    t = np.tanh(x)
    return 1.0 - t ** 2


# =============================================================================
# Binary Cross-Entropy Loss
# =============================================================================

def bce_loss(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy loss: -[y*log(p) + (1-y)*log(1-p)].

    Args:
        predictions: Discriminator outputs (probabilities), shape (batch, 1)
        targets: Labels (1 for real, 0 for fake), shape (batch, 1)
        eps: Small constant to prevent log(0)

    Returns:
        Scalar loss value (mean over batch)

    The eps clipping is crucial — without it, log(0) = -inf would produce
    NaN gradients and crash training.
    """
    p = np.clip(predictions, eps, 1 - eps)
    return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))

def bce_loss_gradient(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Gradient of BCE loss w.r.t. predictions: -(y/p - (1-y)/(1-p)) / batch_size.

    This is the starting point of backpropagation — the gradient flows from
    the loss backward through the discriminator (and optionally through the
    generator when training G).
    """
    p = np.clip(predictions, eps, 1 - eps)
    return (-(targets / p) + (1 - targets) / (1 - p)) / predictions.shape[0]


# =============================================================================
# Neural Network Layer
# =============================================================================

class Layer:
    """A single fully-connected layer with optional activation.

    Stores pre-activation (z) and post-activation (a) values during forward
    pass for use in backpropagation. Weight gradients are accumulated and
    applied during the update step.
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = 'relu'):
        """Initialize with He initialization for ReLU-family activations.

        He initialization: W ~ N(0, sqrt(2/fan_in)). This ensures the
        variance of activations stays roughly constant across layers,
        preventing signal from vanishing or exploding during forward pass.
        For tanh/sigmoid output layers, we use Xavier: sqrt(1/fan_in).
        """
        if activation in ('relu', 'leaky_relu'):
            scale = np.sqrt(2.0 / in_features)  # He init
        else:
            scale = np.sqrt(1.0 / in_features)   # Xavier init

        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))
        self.activation = activation

        # Cache for backprop — stored during forward, consumed during backward
        self.input: np.ndarray = None   # input to this layer
        self.z: np.ndarray = None       # pre-activation: input @ W + b
        self.a: np.ndarray = None       # post-activation: act(z)

        # Gradient accumulators
        self.dW: np.ndarray = None
        self.db: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: z = xW + b, a = activation(z).

        Caches input, z, and a for use in backward pass. The caching is
        essential — backprop needs to know what the activations were at
        each layer to compute correct gradients.
        """
        self.input = x
        self.z = x @ self.W + self.b

        if self.activation == 'relu':
            self.a = relu(self.z)
        elif self.activation == 'leaky_relu':
            self.a = leaky_relu(self.z)
        elif self.activation == 'sigmoid':
            self.a = sigmoid(self.z)
        elif self.activation == 'tanh':
            self.a = tanh(self.z)
        elif self.activation == 'none':
            self.a = self.z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        return self.a

    def backward(self, da: np.ndarray) -> np.ndarray:
        """Backward pass: compute dW, db, and propagate gradient to input.

        Chain rule decomposition:
        1. da → dz: multiply by activation derivative
        2. dz → dW: dW = input.T @ dz (how weights affect output)
        3. dz → db: db = sum(dz) (bias gradient is just the upstream gradient)
        4. dz → dx: dx = dz @ W.T (propagate to previous layer)

        Returns dx so the previous layer can continue backprop.
        """
        # Step 1: Activation derivative
        if self.activation == 'relu':
            dz = da * relu_derivative(self.z)
        elif self.activation == 'leaky_relu':
            dz = da * leaky_relu_derivative(self.z)
        elif self.activation == 'sigmoid':
            s = sigmoid(self.z)
            dz = da * s * (1 - s)
        elif self.activation == 'tanh':
            dz = da * tanh_derivative(self.z)
        elif self.activation == 'none':
            dz = da
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Step 2-4: Parameter and input gradients
        self.dW = self.input.T @ dz
        self.db = np.sum(dz, axis=0, keepdims=True)
        dx = dz @ self.W.T

        return dx

    def update(self, lr: float) -> None:
        """SGD weight update: W -= lr * dW."""
        self.W -= lr * self.dW
        self.b -= lr * self.db


# =============================================================================
# Generator Network
# =============================================================================

class Generator:
    """Generator: maps latent noise z to data space.

    Architecture: z → Linear(ReLU) → Linear(ReLU) → Linear(tanh)

    The latent dimension is typically much smaller than data dimension —
    the generator must learn to "unfold" this compressed representation
    into the full data space. Think of it as a learned decompressor.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        self.latent_dim = latent_dim
        self.layers = [
            Layer(latent_dim, hidden_dim, activation='relu'),
            Layer(hidden_dim, hidden_dim, activation='relu'),
            Layer(hidden_dim, output_dim, activation='tanh'),
        ]

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Generate fake data from noise vector z."""
        x = z
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        """Backpropagate gradient through all generator layers.

        The gradient comes from the discriminator — it tells the generator
        how to change its output to better fool D. We propagate this signal
        backward through G's layers to update G's weights.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr: float) -> None:
        """Update all generator weights."""
        for layer in self.layers:
            layer.update(lr)

    def sample(self, n: int) -> np.ndarray:
        """Generate n samples by drawing random noise and forwarding."""
        z = np.random.randn(n, self.latent_dim)
        return self.forward(z)


# =============================================================================
# Discriminator Network
# =============================================================================

class Discriminator:
    """Discriminator: classifies data as real (1) or fake (0).

    Architecture: x → Linear(LeakyReLU) → Linear(LeakyReLU) → Linear(sigmoid)

    Uses LeakyReLU instead of ReLU because the discriminator needs to
    maintain gradient flow for both positive and negative activations.
    Data centered near zero means many pre-activations are negative.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        self.layers = [
            Layer(input_dim, hidden_dim, activation='leaky_relu'),
            Layer(hidden_dim, hidden_dim, activation='leaky_relu'),
            Layer(hidden_dim, 1, activation='sigmoid'),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Classify input as real (→1) or fake (→0)."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate and return gradient w.r.t. input.

        Returns the input gradient so it can flow into the generator
        when training G. When training D alone, this return value is unused.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad  # gradient w.r.t. D's input — needed for G training

    def update(self, lr: float) -> None:
        """Update all discriminator weights."""
        for layer in self.layers:
            layer.update(lr)


# =============================================================================
# Data Generation
# =============================================================================

def make_mixture_of_gaussians(n: int, n_clusters: int = 4,
                               radius: float = 0.6, std: float = 0.05,
                               seed: int = None) -> np.ndarray:
    """Generate 2D data from a mixture of Gaussians arranged in a circle.

    This creates a clear, multi-modal distribution. A well-trained generator
    should place samples around ALL clusters. If it only covers some, that's
    mode collapse — the most common GAN failure mode.

    Args:
        n: Number of samples
        n_clusters: Number of Gaussian clusters
        radius: Radius of the circle on which cluster centers sit
        std: Standard deviation of each cluster
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n, 2) with 2D data points
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # Place cluster centers evenly on a circle
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    # Assign each sample to a random cluster, then add Gaussian noise
    assignments = rng.randint(0, n_clusters, size=n)
    data = centers[assignments] + rng.randn(n, 2) * std

    return data


# =============================================================================
# Training Metrics
# =============================================================================

def compute_mode_coverage(fake_data: np.ndarray, centers: np.ndarray,
                          threshold: float = 0.15) -> Tuple[int, int]:
    """Count how many modes (cluster centers) are covered by generated samples.

    A mode is "covered" if at least one generated sample falls within
    `threshold` distance of the cluster center. Low coverage = mode collapse.

    Returns:
        (covered_modes, total_modes)
    """
    covered = 0
    for center in centers:
        distances = np.linalg.norm(fake_data - center, axis=1)
        if np.any(distances < threshold):
            covered += 1
    return covered, len(centers)


def wasserstein_estimate(real: np.ndarray, fake: np.ndarray) -> float:
    """Rough 1D Wasserstein distance estimate (sliced Wasserstein).

    Projects both distributions onto random 1D directions and computes
    the average 1D Wasserstein distance (= difference of sorted values).
    This is a practical approximation — exact Wasserstein in 2D is O(n³).
    Lower values indicate the generated distribution is closer to real.
    """
    n_projections = 50
    distances = []

    for _ in range(n_projections):
        # Random unit direction
        direction = np.random.randn(2)
        direction /= np.linalg.norm(direction)

        # Project and sort
        real_proj = np.sort(real @ direction)
        fake_proj = np.sort(fake @ direction)

        # Subsample to same size for comparison
        min_n = min(len(real_proj), len(fake_proj))
        distances.append(np.mean(np.abs(real_proj[:min_n] - fake_proj[:min_n])))

    return np.mean(distances)


# =============================================================================
# GAN Training
# =============================================================================

def train_gan(
    n_epochs: int = 2000,
    batch_size: int = 256,
    latent_dim: int = 8,
    hidden_dim: int = 64,
    lr_g: float = 0.0002,
    lr_d: float = 0.0002,
    n_clusters: int = 4,
    d_steps: int = 1,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Train a GAN to generate 2D mixture of Gaussians data.

    Args:
        n_epochs: Number of training iterations
        batch_size: Samples per batch
        latent_dim: Dimension of noise vector z
        hidden_dim: Hidden layer size for both G and D
        lr_g: Generator learning rate
        lr_d: Discriminator learning rate
        n_clusters: Number of Gaussian clusters in target distribution
        d_steps: Discriminator updates per generator update
        seed: Random seed
        verbose: Print training progress

    Returns:
        Dictionary with training history and final models

    Training protocol:
    1. Train D for d_steps on real+fake data (standard BCE)
    2. Train G for 1 step using non-saturating loss
    3. Log metrics every 200 epochs

    The d_steps parameter controls the D/G training ratio. If D is too weak,
    G gets no useful gradient signal. If D is too strong, G gradients vanish.
    d_steps=1 works well for this problem.
    """
    np.random.seed(seed)

    # Initialize networks
    data_dim = 2  # 2D data
    G = Generator(latent_dim, hidden_dim, data_dim)
    D = Discriminator(data_dim, hidden_dim)

    # Cluster centers for mode coverage tracking
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = np.column_stack([0.6 * np.cos(angles), 0.6 * np.sin(angles)])

    # Training history
    history = {
        'g_loss': [], 'd_loss': [], 'd_real_acc': [], 'd_fake_acc': [],
        'mode_coverage': [], 'wasserstein': [], 'epochs': []
    }

    for epoch in range(n_epochs):
        # =====================================================================
        # Step 1: Train Discriminator
        # =====================================================================
        for _ in range(d_steps):
            # Sample real data
            real_data = make_mixture_of_gaussians(batch_size, n_clusters)

            # Sample fake data (detached from G's computation graph)
            z = np.random.randn(batch_size, latent_dim)
            fake_data = G.forward(z)

            # D forward pass on real data
            d_real = D.forward(real_data)
            real_labels = np.ones((batch_size, 1))
            d_loss_real = bce_loss(d_real, real_labels)

            # Backprop on real — gradient of loss w.r.t. D's output
            grad_real = bce_loss_gradient(d_real, real_labels)
            # Multiply by sigmoid derivative to get gradient w.r.t. pre-sigmoid
            # Actually, our backward already handles this through the sigmoid layer
            D.backward(grad_real)

            # D forward pass on fake data
            d_fake = D.forward(fake_data)
            fake_labels = np.zeros((batch_size, 1))
            d_loss_fake = bce_loss(d_fake, fake_labels)

            # Backprop on fake
            grad_fake = bce_loss_gradient(d_fake, fake_labels)
            # Save gradients from real pass
            saved_dW = [layer.dW.copy() for layer in D.layers]
            saved_db = [layer.db.copy() for layer in D.layers]

            D.backward(grad_fake)

            # Accumulate gradients from both real and fake passes
            for i, layer in enumerate(D.layers):
                layer.dW += saved_dW[i]
                layer.db += saved_db[i]

            # Update D
            D.update(lr_d)

            d_loss = d_loss_real + d_loss_fake

        # =====================================================================
        # Step 2: Train Generator
        # =====================================================================
        # Sample fresh noise — important to use new noise each time
        z = np.random.randn(batch_size, latent_dim)
        fake_data = G.forward(z)

        # Pass fakes through D
        d_fake_for_g = D.forward(fake_data)

        # Non-saturating loss: G maximizes log(D(G(z))) instead of
        # minimizing log(1 - D(G(z))). We compute -log(D(G(z))) and minimize.
        # This gives stronger gradients early in training when D easily rejects fakes.
        g_labels = np.ones((batch_size, 1))  # G wants D to output 1
        g_loss = bce_loss(d_fake_for_g, g_labels)

        # Backprop through D (frozen — we don't update D's weights)
        grad_g = bce_loss_gradient(d_fake_for_g, g_labels)
        grad_to_g = D.backward(grad_g)  # gradient w.r.t. D's input = G's output

        # Backprop through G
        G.backward(grad_to_g)
        G.update(lr_g)

        # =====================================================================
        # Step 3: Log metrics
        # =====================================================================
        if epoch % 200 == 0 or epoch == n_epochs - 1:
            # Discriminator accuracy
            d_real_acc = np.mean(d_real > 0.5)
            d_fake_acc = np.mean(d_fake < 0.5)

            # Mode coverage
            eval_fake = G.sample(500)
            covered, total = compute_mode_coverage(eval_fake, centers)

            # Wasserstein estimate
            eval_real = make_mixture_of_gaussians(500, n_clusters)
            w_dist = wasserstein_estimate(eval_real, eval_fake)

            history['g_loss'].append(g_loss)
            history['d_loss'].append(d_loss)
            history['d_real_acc'].append(d_real_acc)
            history['d_fake_acc'].append(d_fake_acc)
            history['mode_coverage'].append(covered / total)
            history['wasserstein'].append(w_dist)
            history['epochs'].append(epoch)

            if verbose:
                print(f"Epoch {epoch:5d} | D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f} | "
                      f"D_acc(real): {d_real_acc:.2f} | D_acc(fake): {d_fake_acc:.2f} | "
                      f"Modes: {covered}/{total} | W_dist: {w_dist:.4f}")

    return {
        'generator': G,
        'discriminator': D,
        'history': history,
        'centers': centers,
    }


# =============================================================================
# Visualization (text-based)
# =============================================================================

def ascii_scatter(points: np.ndarray, width: int = 60, height: int = 25,
                  title: str = "") -> str:
    """Render a 2D scatter plot as ASCII art.

    Maps points to a character grid. Density is shown by character weight:
    '.' for sparse, 'o' for medium, '#' for dense regions.
    """
    if len(points) == 0:
        return "No points to plot"

    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0

    grid = [[' '] * width for _ in range(height)]
    counts = [[0] * width for _ in range(height)]

    for px, py in points:
        col = int((px - x_min) / (x_max - x_min) * (width - 1))
        row = int((y_max - py) / (y_max - y_min) * (height - 1))  # flip y
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        counts[row][col] += 1

    for r in range(height):
        for c in range(width):
            if counts[r][c] >= 3:
                grid[r][c] = '#'
            elif counts[r][c] >= 1:
                grid[r][c] = 'o'

    lines = []
    if title:
        lines.append(f"  {title}")
        lines.append(f"  {'─' * width}")
    for row in grid:
        lines.append(f"  │{''.join(row)}│")
    lines.append(f"  {'─' * width}")

    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Day 064: Generative Adversarial Network (GAN) from Scratch")
    print("=" * 70)

    # --- Show real data distribution ---
    print("\n1. TARGET DISTRIBUTION (Mixture of 4 Gaussians)")
    print("-" * 50)
    real_data = make_mixture_of_gaussians(500, n_clusters=4, seed=42)
    print(ascii_scatter(real_data, title="Real Data"))
    print(f"   Mean: ({real_data[:,0].mean():.3f}, {real_data[:,1].mean():.3f})")
    print(f"   Std:  ({real_data[:,0].std():.3f}, {real_data[:,1].std():.3f})")

    # --- Train the GAN ---
    print("\n2. TRAINING GAN")
    print("-" * 50)
    print("Training with non-saturating generator loss...")
    print("Watch D_acc — it should hover around 0.5 when balanced.\n")

    result = train_gan(
        n_epochs=3000,
        batch_size=256,
        latent_dim=8,
        hidden_dim=64,
        lr_g=0.0003,
        lr_d=0.0001,  # Slower D learning rate to prevent it from overpowering G
        n_clusters=4,
        d_steps=1,
        seed=42,
    )

    G = result['generator']
    D = result['discriminator']
    history = result['history']
    centers = result['centers']

    # --- Show generated data ---
    print("\n3. GENERATED SAMPLES (after training)")
    print("-" * 50)
    fake_samples = G.sample(500)
    print(ascii_scatter(fake_samples, title="Generated Data"))
    print(f"   Mean: ({fake_samples[:,0].mean():.3f}, {fake_samples[:,1].mean():.3f})")
    print(f"   Std:  ({fake_samples[:,0].std():.3f}, {fake_samples[:,1].std():.3f})")

    # --- Final evaluation ---
    print("\n4. FINAL EVALUATION")
    print("-" * 50)

    # Mode coverage
    covered, total = compute_mode_coverage(fake_samples, centers, threshold=0.2)
    print(f"   Mode coverage: {covered}/{total} clusters")

    # Wasserstein distance
    real_eval = make_mixture_of_gaussians(1000, n_clusters=4, seed=99)
    fake_eval = G.sample(1000)
    w_dist = wasserstein_estimate(real_eval, fake_eval)
    print(f"   Wasserstein distance: {w_dist:.4f}")

    # Discriminator equilibrium check
    d_on_real = D.forward(real_eval)
    d_on_fake = D.forward(fake_eval)
    print(f"   D(real) mean: {d_on_real.mean():.4f} (ideal: 0.5)")
    print(f"   D(fake) mean: {d_on_fake.mean():.4f} (ideal: 0.5)")

    # --- Training dynamics analysis ---
    print("\n5. TRAINING DYNAMICS")
    print("-" * 50)
    print("   Epoch  | G_loss | D_loss | D_acc | Coverage | W_dist")
    print("   " + "-" * 60)
    for i in range(len(history['epochs'])):
        e = history['epochs'][i]
        gl = history['g_loss'][i]
        dl = history['d_loss'][i]
        da = (history['d_real_acc'][i] + history['d_fake_acc'][i]) / 2
        mc = history['mode_coverage'][i]
        wd = history['wasserstein'][i]
        print(f"   {e:5d}  | {gl:.4f} | {dl:.4f} | {da:.2f}  | {mc:.2f}     | {wd:.4f}")

    # --- Key takeaways ---
    print("\n6. KEY OBSERVATIONS")
    print("-" * 50)

    final_coverage = history['mode_coverage'][-1]
    final_w = history['wasserstein'][-1]
    initial_w = history['wasserstein'][0]

    print(f"   • Wasserstein distance decreased from {initial_w:.4f} → {final_w:.4f} "
          f"({(1 - final_w/initial_w)*100:.0f}% improvement)")
    print(f"   • Final mode coverage: {final_coverage*100:.0f}%")

    if final_coverage >= 0.75:
        print("   • Generator successfully learned the multi-modal distribution")
    else:
        print("   • Partial mode collapse detected — G only captures some modes")

    d_acc_final = (history['d_real_acc'][-1] + history['d_fake_acc'][-1]) / 2
    if 0.4 <= d_acc_final <= 0.65:
        print(f"   • D accuracy ({d_acc_final:.2f}) near equilibrium — healthy training")
    elif d_acc_final > 0.8:
        print(f"   • D accuracy ({d_acc_final:.2f}) too high — D overpowering G")
    else:
        print(f"   • D accuracy ({d_acc_final:.2f}) below chance — G dominating")

    print("\n   Compare with Day 060 (Diffusion): GANs train via adversarial dynamics")
    print("   (two competing networks), while diffusion models learn to denoise via")
    print("   a single regression objective. GANs are harder to train but can be")
    print("   faster at inference (single forward pass vs. iterative denoising).")

    print("\n" + "=" * 70)
    print("GAN training complete!")
    print("=" * 70)
