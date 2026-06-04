"""
Day 064: Generative Adversarial Network (GAN) from Scratch — Your Implementation

Build a GAN that generates 2D data matching a mixture of Gaussians.
Implement the generator, discriminator, loss functions, and training loop.

Hints:
- Start with the activation functions — they're the building blocks
- The Layer class handles forward/backward for a single linear + activation layer
- Generator: noise → ReLU layers → tanh output (bounded data)
- Discriminator: data → LeakyReLU layers → sigmoid output (probability)
- BCE loss gradient is the starting point of backpropagation
- Non-saturating G loss: use labels=1 for fake data when training G
- Key insight: when training G, backprop flows through D (frozen) into G
"""

import numpy as np
from typing import List, Tuple, Dict


# =============================================================================
# Activation functions and their derivatives
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    raise NotImplementedError("TODO: implement ReLU")

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0."""
    raise NotImplementedError("TODO: implement ReLU derivative")

def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """LeakyReLU: x if x > 0, else alpha * x.

    Hint: np.where is your friend here.
    """
    raise NotImplementedError("TODO: implement LeakyReLU")

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Derivative of LeakyReLU: 1 if x > 0, else alpha."""
    raise NotImplementedError("TODO: implement LeakyReLU derivative")

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-x)).

    Hint: clip x to [-500, 500] to prevent overflow.
    """
    raise NotImplementedError("TODO: implement sigmoid")

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation: bounds output to [-1, 1]."""
    raise NotImplementedError("TODO: implement tanh")

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh²(x)."""
    raise NotImplementedError("TODO: implement tanh derivative")


# =============================================================================
# Binary Cross-Entropy Loss
# =============================================================================

def bce_loss(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy loss: -mean[y*log(p) + (1-y)*log(1-p)].

    Args:
        predictions: Discriminator outputs (probabilities), shape (batch, 1)
        targets: Labels (1 for real, 0 for fake), shape (batch, 1)
        eps: Small constant to prevent log(0)

    Hint: clip predictions to [eps, 1-eps] before taking log.
    """
    raise NotImplementedError("TODO: implement BCE loss")

def bce_loss_gradient(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Gradient of BCE loss w.r.t. predictions.

    Formula: -(y/p - (1-y)/(1-p)) / batch_size

    Hint: don't forget to divide by batch_size for mean gradient.
    """
    raise NotImplementedError("TODO: implement BCE loss gradient")


# =============================================================================
# Neural Network Layer
# =============================================================================

class Layer:
    """A single fully-connected layer with optional activation.

    Hint: store input, pre-activation (z), and post-activation (a) during
    forward pass — you'll need them for backpropagation.
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = 'relu'):
        """Initialize weights with He initialization for ReLU-family activations.

        Hint: He init scale = sqrt(2/fan_in) for ReLU, sqrt(1/fan_in) for others.
        """
        raise NotImplementedError("TODO: initialize weights and biases")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: z = xW + b, a = activation(z).

        Hint: cache self.input, self.z, self.a for backward pass.
        """
        raise NotImplementedError("TODO: implement forward pass")

    def backward(self, da: np.ndarray) -> np.ndarray:
        """Backward pass: compute dW, db, and return gradient w.r.t. input.

        Steps:
        1. dz = da * activation_derivative(z)
        2. dW = input.T @ dz
        3. db = sum(dz, axis=0)
        4. dx = dz @ W.T  (return this)
        """
        raise NotImplementedError("TODO: implement backward pass")

    def update(self, lr: float) -> None:
        """SGD update: W -= lr * dW, b -= lr * db."""
        raise NotImplementedError("TODO: implement weight update")


# =============================================================================
# Generator Network
# =============================================================================

class Generator:
    """Generator: maps latent noise z to data space.

    Architecture: z → Linear(ReLU) → Linear(ReLU) → Linear(tanh)

    Hint: the output activation is tanh to bound generated data to [-1, 1].
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        raise NotImplementedError("TODO: create generator layers")

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Generate fake data from noise vector z."""
        raise NotImplementedError("TODO: forward through all layers")

    def backward(self, grad: np.ndarray) -> None:
        """Backpropagate gradient through all generator layers."""
        raise NotImplementedError("TODO: backward through all layers in reverse")

    def update(self, lr: float) -> None:
        """Update all generator weights."""
        raise NotImplementedError("TODO: update all layers")

    def sample(self, n: int) -> np.ndarray:
        """Generate n samples by drawing random noise and forwarding."""
        raise NotImplementedError("TODO: sample noise and forward")


# =============================================================================
# Discriminator Network
# =============================================================================

class Discriminator:
    """Discriminator: classifies data as real (1) or fake (0).

    Architecture: x → Linear(LeakyReLU) → Linear(LeakyReLU) → Linear(sigmoid)

    Hint: uses LeakyReLU, not ReLU — important for gradient flow.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        raise NotImplementedError("TODO: create discriminator layers")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Classify input as real (→1) or fake (→0)."""
        raise NotImplementedError("TODO: forward through all layers")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate and return gradient w.r.t. input.

        Hint: the returned gradient flows into the generator during G training.
        """
        raise NotImplementedError("TODO: backward through all layers, return input grad")

    def update(self, lr: float) -> None:
        """Update all discriminator weights."""
        raise NotImplementedError("TODO: update all layers")


# =============================================================================
# Data Generation
# =============================================================================

def make_mixture_of_gaussians(n: int, n_clusters: int = 4,
                               radius: float = 0.6, std: float = 0.05,
                               seed: int = None) -> np.ndarray:
    """Generate 2D data from a mixture of Gaussians arranged in a circle.

    Args:
        n: Number of samples
        n_clusters: Number of Gaussian clusters
        radius: Radius of the circle on which cluster centers sit
        std: Standard deviation of each cluster
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n, 2)

    Hint: place centers on a circle using cos/sin, then add Gaussian noise.
    """
    raise NotImplementedError("TODO: generate mixture of Gaussians data")


# =============================================================================
# Training Metrics
# =============================================================================

def compute_mode_coverage(fake_data: np.ndarray, centers: np.ndarray,
                          threshold: float = 0.15) -> Tuple[int, int]:
    """Count how many cluster centers have at least one nearby generated sample.

    Hint: for each center, check if any fake sample is within threshold distance.
    """
    raise NotImplementedError("TODO: implement mode coverage check")


def wasserstein_estimate(real: np.ndarray, fake: np.ndarray) -> float:
    """Sliced Wasserstein distance estimate.

    Hint: project both distributions onto random 1D directions, sort,
    and compute mean absolute difference of sorted values.
    """
    raise NotImplementedError("TODO: implement sliced Wasserstein estimate")


# =============================================================================
# GAN Training Loop
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

    The training loop alternates between:
    1. Train D: real data (label=1) + fake data (label=0), BCE loss
    2. Train G: fake data through D, non-saturating loss (label=1)

    Key insight for G training:
    - Forward: z → G → fake → D → prediction
    - Backward: loss_grad → D.backward() → gets grad_to_G → G.backward()
    - Only update G's weights, NOT D's weights

    Hint: accumulate D gradients from both real and fake passes before updating.
    """
    raise NotImplementedError("TODO: implement the full GAN training loop")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Day 064: GAN from Scratch — Your Implementation")
    print("=" * 70)

    # Test data generation
    print("\n1. Testing data generation...")
    try:
        data = make_mixture_of_gaussians(100, n_clusters=4, seed=42)
        print(f"   Generated {len(data)} points, shape: {data.shape}")
        print(f"   Mean: ({data[:,0].mean():.3f}, {data[:,1].mean():.3f})")
    except NotImplementedError as e:
        print(f"   {e}")

    # Test activations
    print("\n2. Testing activation functions...")
    test_x = np.array([-1.0, 0.0, 1.0])
    for name, fn in [('relu', relu), ('leaky_relu', leaky_relu),
                      ('sigmoid', sigmoid), ('tanh', tanh)]:
        try:
            print(f"   {name}({test_x}) = {fn(test_x)}")
        except NotImplementedError as e:
            print(f"   {name}: {e}")

    # Test layer
    print("\n3. Testing Layer...")
    try:
        layer = Layer(2, 4, activation='relu')
        out = layer.forward(np.random.randn(3, 2))
        print(f"   Layer(2→4, ReLU) output shape: {out.shape}")
    except NotImplementedError as e:
        print(f"   {e}")

    # Test networks
    print("\n4. Testing Generator and Discriminator...")
    try:
        G = Generator(latent_dim=8, hidden_dim=64, output_dim=2)
        fake = G.sample(10)
        print(f"   Generator output shape: {fake.shape}")
    except NotImplementedError as e:
        print(f"   Generator: {e}")

    try:
        D = Discriminator(input_dim=2, hidden_dim=64)
        pred = D.forward(np.random.randn(10, 2))
        print(f"   Discriminator output shape: {pred.shape}")
    except NotImplementedError as e:
        print(f"   Discriminator: {e}")

    # Train GAN
    print("\n5. Training GAN...")
    try:
        result = train_gan(n_epochs=2000, verbose=True)
        print("\n   Training complete!")

        G = result['generator']
        fake = G.sample(500)
        covered, total = compute_mode_coverage(fake, result['centers'], threshold=0.2)
        print(f"   Mode coverage: {covered}/{total}")
    except NotImplementedError as e:
        print(f"   {e}")

    print("\n" + "=" * 70)
    print("Done! Check solution.py for reference implementation.")
    print("=" * 70)
