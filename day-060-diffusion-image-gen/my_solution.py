"""
Day 060: Denoising Diffusion Probabilistic Model (DDPM) — Your Implementation

Implement a complete diffusion model from scratch using NumPy.
You'll build the forward noising process, the reverse denoising network,
and the sampling loop to generate images from pure noise.

Run tests with: python3 -m pytest tests.py -v
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class NoiseSchedule:
    """
    Precomputes all diffusion constants for T timesteps.

    You need to compute:
    - betas: the noise schedule (linear or cosine)
    - alphas: 1 - betas
    - alpha_bars: cumulative product of alphas
    - Various derived quantities for the forward and reverse processes

    Hint: The cosine schedule uses alpha_bar_t = cos^2((t/T + s)/(1+s) * pi/2)
    and then derives betas from consecutive alpha_bars.
    """

    def __init__(self, num_timesteps: int = 100, schedule_type: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = num_timesteps
        raise NotImplementedError("TODO: implement noise schedule computation")

    def get_forward_params(self, t: int) -> Tuple[float, float]:
        """Return (sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t) for timestep t."""
        raise NotImplementedError("TODO: implement forward parameter lookup")

    def info(self) -> Dict[str, float]:
        """Summary statistics about the schedule."""
        raise NotImplementedError("TODO: implement schedule info")


def forward_diffusion(x0: np.ndarray, t: int, schedule: NoiseSchedule,
                      noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add noise to a clean image x0 at timestep t.

    Use the closed-form formula:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    Hint: You don't need to apply noise step by step — the math lets you
    jump directly to any timestep t.

    Args:
        x0: Clean image, shape (H*W,)
        t: Timestep (0 to T-1)
        schedule: Precomputed noise schedule
        noise: Optional pre-sampled noise

    Returns:
        (x_t, epsilon): Noisy image and the noise that was added
    """
    raise NotImplementedError("TODO: implement forward diffusion")


def sinusoidal_embedding(t: int, dim: int) -> np.ndarray:
    """
    Encode a scalar timestep into a vector using sinusoidal functions.

    Hint: Use log-spaced frequencies from 1 to 1/10000, then
    concatenate sin and cos values.

    Args:
        t: Integer timestep
        dim: Embedding dimension (must be even)

    Returns:
        Embedding vector of shape (dim,)
    """
    raise NotImplementedError("TODO: implement sinusoidal timestep embedding")


class DenoisingMLP:
    """
    MLP that predicts the noise added to an image at timestep t.

    Architecture: [noisy_image + time_embed] -> FC -> ReLU -> FC -> ReLU -> FC -> noise_pred

    Hint: Use He initialization (scale by sqrt(2/fan_in)) for ReLU layers.
    Store activations in self._cache for backpropagation.
    """

    def __init__(self, image_dim: int, time_embed_dim: int = 32,
                 hidden_dim: int = 128, seed: int = 42):
        self.image_dim = image_dim
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim
        self._cache = {}
        raise NotImplementedError("TODO: initialize network weights")

    def forward(self, x_flat: np.ndarray, t_embed: np.ndarray) -> np.ndarray:
        """
        Predict noise from noisy image + timestep embedding.

        Hint: Concatenate x_flat and t_embed, then pass through 3 linear
        layers with ReLU after the first two. No activation on the output.

        Args:
            x_flat: Flattened noisy image, shape (image_dim,)
            t_embed: Timestep embedding, shape (time_embed_dim,)

        Returns:
            Predicted noise, shape (image_dim,)
        """
        raise NotImplementedError("TODO: implement forward pass")

    def backward(self, d_out: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backpropagation through the MLP.

        Hint: Work backward from the output layer. For ReLU, the gradient
        is 1 where z > 0 and 0 otherwise. Use np.outer for weight gradients.

        Args:
            d_out: Gradient of loss w.r.t. output, shape (image_dim,)

        Returns:
            Dictionary of gradients for all parameters
        """
        raise NotImplementedError("TODO: implement backpropagation")

    def update(self, grads: Dict[str, np.ndarray], lr: float):
        """SGD update with gradient clipping."""
        raise NotImplementedError("TODO: implement parameter update")

    def get_param_count(self) -> int:
        """Total trainable parameters."""
        raise NotImplementedError("TODO: implement parameter counting")


def train_diffusion(model: DenoisingMLP, schedule: NoiseSchedule,
                    data: np.ndarray, num_epochs: int = 50,
                    lr: float = 1e-3, seed: int = 42) -> List[float]:
    """
    Train the denoising model using the DDPM objective.

    Training loop:
    1. Pick random image from dataset
    2. Pick random timestep t
    3. Sample noise epsilon
    4. Compute x_t using forward_diffusion
    5. Predict noise with model
    6. Loss = MSE(epsilon, epsilon_theta)
    7. Backprop and update

    Hint: The loss gradient d_out = 2*(noise_pred - noise)/dim

    Args:
        model: Denoising network
        schedule: Noise schedule
        data: Training images, shape (N, image_dim)
        num_epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed

    Returns:
        List of average losses per epoch
    """
    raise NotImplementedError("TODO: implement training loop")


def sample_ddpm(model: DenoisingMLP, schedule: NoiseSchedule,
                image_shape: int, num_samples: int = 4,
                seed: int = 123) -> List[np.ndarray]:
    """
    Generate images by iteratively denoising from pure noise.

    Sampling (reverse process):
    1. x_T ~ N(0, I)
    2. For t = T-1 down to 0:
       a. eps_theta = model(x_t, t)
       b. mu = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
       c. x_{t-1} = mu + sigma_t * z  (z=0 for t=0)

    Hint: sigma_t = sqrt(posterior_variance[t])

    Args:
        model: Trained denoising network
        schedule: Noise schedule
        image_shape: Flattened image dimension
        num_samples: Number of images to generate
        seed: Random seed

    Returns:
        List of generated images (flattened)
    """
    raise NotImplementedError("TODO: implement DDPM sampling")


def create_dataset(num_samples: int = 200, image_size: int = 8,
                   seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """
    Create a dataset of simple grayscale patterns.

    Patterns: horizontal bars, vertical bars, cross, diagonal, border.
    Each with random variations. Normalize to [-1, 1].

    Args:
        num_samples: Total number of training images
        image_size: Width and height
        seed: Random seed

    Returns:
        (data, labels): Flattened images shape (N, H*W), pattern names
    """
    raise NotImplementedError("TODO: implement dataset creation")


def render_ascii(image_flat: np.ndarray, size: int = 8) -> str:
    """Render a flattened image as ASCII art. Values in [-1, 1]."""
    raise NotImplementedError("TODO: implement ASCII rendering")


def compute_distribution_stats(data: np.ndarray, samples: List[np.ndarray]) -> Dict[str, float]:
    """Compare statistics between real data and generated samples."""
    raise NotImplementedError("TODO: implement distribution comparison")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    IMAGE_SIZE = 8
    IMAGE_DIM = IMAGE_SIZE * IMAGE_SIZE
    T = 50

    print("Testing your DDPM implementation...\n")

    # Test 1: Noise schedule
    print("1. Creating noise schedule...")
    schedule = NoiseSchedule(T, "cosine")
    print(f"   alpha_bar at t=0: {schedule.alpha_bars[0]:.4f} (should be ~1.0)")
    print(f"   alpha_bar at t=T: {schedule.alpha_bars[-1]:.4f} (should be ~0.0)")

    # Test 2: Forward diffusion
    print("\n2. Testing forward diffusion...")
    x0 = np.random.randn(IMAGE_DIM)
    x_t, noise = forward_diffusion(x0, T - 1, schedule)
    print(f"   Input norm: {np.linalg.norm(x0):.4f}")
    print(f"   Noised norm: {np.linalg.norm(x_t):.4f}")

    # Test 3: Timestep embedding
    print("\n3. Testing timestep embedding...")
    emb = sinusoidal_embedding(10, 32)
    print(f"   Embedding shape: {emb.shape} (should be (32,))")

    # Test 4: Model
    print("\n4. Building model...")
    model = DenoisingMLP(IMAGE_DIM, time_embed_dim=32, hidden_dim=128)
    print(f"   Parameters: {model.get_param_count():,}")

    # Test 5: Training
    print("\n5. Training...")
    data, labels = create_dataset(100, IMAGE_SIZE)
    losses = train_diffusion(model, schedule, data, num_epochs=5, lr=5e-4)
    print(f"   Loss went from {losses[0]:.4f} to {losses[-1]:.4f}")

    # Test 6: Sampling
    print("\n6. Generating samples...")
    samples = sample_ddpm(model, schedule, IMAGE_DIM, num_samples=2)
    print(f"   Generated {len(samples)} images")
    print(f"   Sample shape: {samples[0].shape}")

    print("\nAll components working! Run the full training for better results.")
