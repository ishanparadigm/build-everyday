"""
Day 060: Denoising Diffusion Probabilistic Model (DDPM) from Scratch

A complete implementation of a diffusion model using only NumPy.
We train on simple 2D patterns (circles, crosses) to demonstrate
the full forward noising → reverse denoising → generation pipeline.

Since we're working without GPU frameworks, we use small grayscale images
(8x8) and an MLP instead of a U-Net. The math is identical to full-scale
DDPM — only the architecture is simplified.
"""

import warnings
import numpy as np
from typing import Tuple, List, Dict, Optional

# Suppress matmul overflow warnings that occur in early training epochs
# before gradient clipping stabilizes the weights. The nan_to_num calls
# in the forward/backward passes handle these cases safely.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")


# =============================================================================
# NOISE SCHEDULE
# =============================================================================

class NoiseSchedule:
    """
    Precomputes all diffusion constants for T timesteps.

    The noise schedule defines how quickly we add noise during the forward
    process. All derived quantities (alpha, alpha_bar, posterior variance)
    are precomputed here for efficiency — they're used in every training
    step and every sampling step.
    """

    def __init__(self, num_timesteps: int = 100, schedule_type: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = num_timesteps

        # Compute beta schedule
        if schedule_type == "linear":
            # Linear interpolation from beta_start to beta_end
            # Simple but destroys information too fast in early steps
            self.betas = np.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            # Preserves more structure in early steps → better quality
            s = 0.008  # small offset to prevent beta from being exactly 0
            steps = np.arange(num_timesteps + 1) / num_timesteps
            alpha_bar = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]  # normalize so alpha_bar[0] = 1
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = np.clip(betas, 1e-6, 0.999)  # clip for numerical stability
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Derived quantities — these come directly from the math
        # alpha_t = 1 - beta_t: how much signal is retained at each step
        self.alphas = 1.0 - self.betas

        # alpha_bar_t = product of alphas from 1 to t
        # This is the key quantity: it tells us the signal-to-noise ratio at step t
        # alpha_bar close to 1 → mostly signal; close to 0 → mostly noise
        self.alpha_bars = np.cumprod(self.alphas)

        # Square roots used in the closed-form forward process:
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1.0 - self.alpha_bars)

        # For the reverse process mean computation:
        # mu_theta = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
        self.sqrt_alphas = np.sqrt(self.alphas)
        self.sqrt_recip_alphas = 1.0 / self.sqrt_alphas

        # Posterior variance: sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        # This is the variance of the true reverse process q(x_{t-1}|x_t, x_0)
        alpha_bars_prev = np.append(1.0, self.alpha_bars[:-1])
        self.posterior_variance = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        # Clip at t=0 where it would be 0/0
        self.posterior_variance = np.clip(self.posterior_variance, 1e-20, None)

    def get_forward_params(self, t: int) -> Tuple[float, float]:
        """Return (sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t) for timestep t."""
        return self.sqrt_alpha_bars[t], self.sqrt_one_minus_alpha_bars[t]

    def info(self) -> Dict[str, float]:
        """Summary statistics about the schedule."""
        return {
            "T": self.T,
            "beta_range": (self.betas[0], self.betas[-1]),
            "alpha_bar_start": self.alpha_bars[0],
            "alpha_bar_end": self.alpha_bars[-1],
            "signal_preserved_at_T": self.alpha_bars[-1],
        }


# =============================================================================
# FORWARD PROCESS
# =============================================================================

def forward_diffusion(x0: np.ndarray, t: int, schedule: NoiseSchedule,
                      noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add noise to a clean image x0 at timestep t using the closed-form formula.

    The key insight: we don't need to apply noise sequentially for t steps.
    The cumulative product alpha_bar lets us jump directly to any timestep:

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    Args:
        x0: Clean image, shape (H, W) or (H*W,)
        t: Timestep (0 to T-1)
        schedule: Precomputed noise schedule
        noise: Optional pre-sampled noise (for reproducibility)

    Returns:
        (x_t, epsilon): The noisy image and the noise that was added
    """
    if noise is None:
        noise = np.random.randn(*x0.shape)

    sqrt_ab, sqrt_one_minus_ab = schedule.get_forward_params(t)

    # This is the entire forward process in one line
    # sqrt_ab scales down the signal, sqrt_one_minus_ab scales up the noise
    # At t=0: mostly signal. At t=T: mostly noise.
    x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise

    return x_t, noise


# =============================================================================
# TIMESTEP EMBEDDING
# =============================================================================

def sinusoidal_embedding(t: int, dim: int) -> np.ndarray:
    """
    Encode a scalar timestep into a vector using sinusoidal functions.

    Same idea as positional encodings in Transformers — different frequencies
    allow the network to distinguish timesteps at multiple scales.

    For example, high-frequency components help distinguish t=50 from t=51,
    while low-frequency components help distinguish t=10 from t=90.

    Args:
        t: Integer timestep
        dim: Embedding dimension (must be even)

    Returns:
        Embedding vector of shape (dim,)
    """
    half_dim = dim // 2
    # Frequencies span from 1 to 1/10000, logarithmically spaced
    freqs = np.exp(-np.log(10000) * np.arange(half_dim) / half_dim)
    angles = t * freqs
    # Concatenate sin and cos — each provides complementary information
    return np.concatenate([np.sin(angles), np.cos(angles)])


# =============================================================================
# NEURAL NETWORK (MLP for noise prediction)
# =============================================================================

class DenoisingMLP:
    """
    A simple MLP that predicts the noise added to an image at timestep t.

    Architecture: [noisy_image + time_embed] → FC → ReLU → FC → ReLU → FC → noise_pred

    In a real DDPM you'd use a U-Net with skip connections, attention, and
    group normalization. The MLP works for small images and demonstrates
    the same training objective.

    Key design choices:
    - Input is concatenation of flattened noisy image + timestep embedding
    - Output has same dimension as the image (predicting noise per pixel)
    - He initialization for ReLU networks (scale by sqrt(2/fan_in))
    """

    def __init__(self, image_dim: int, time_embed_dim: int = 32,
                 hidden_dim: int = 128, seed: int = 42):
        self.image_dim = image_dim
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim

        rng = np.random.RandomState(seed)
        input_dim = image_dim + time_embed_dim

        # He initialization: scale = sqrt(2/fan_in)
        # This keeps variance stable through ReLU layers
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W2 = rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)

        self.W3 = rng.randn(hidden_dim, image_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(image_dim)

        # Store activations for backprop
        self._cache = {}

    def forward(self, x_flat: np.ndarray, t_embed: np.ndarray) -> np.ndarray:
        """
        Forward pass: predict noise from noisy image + timestep.

        Args:
            x_flat: Flattened noisy image, shape (image_dim,)
            t_embed: Timestep embedding, shape (time_embed_dim,)

        Returns:
            Predicted noise, shape (image_dim,)
        """
        # Concatenate image and timestep info
        inp = np.concatenate([x_flat, t_embed])
        self._cache['inp'] = inp

        # Layer 1
        # nan_to_num + clip prevent numerical blowup in early training epochs
        # when the cosine schedule's high-beta timesteps produce extreme gradients
        z1 = np.nan_to_num(inp @ self.W1 + self.b1, nan=0.0, posinf=50.0, neginf=-50.0)
        z1 = np.clip(z1, -50, 50)
        a1 = np.maximum(0, z1)  # ReLU
        self._cache['z1'] = z1
        self._cache['a1'] = a1

        # Layer 2
        z2 = np.nan_to_num(a1 @ self.W2 + self.b2, nan=0.0, posinf=50.0, neginf=-50.0)
        z2 = np.clip(z2, -50, 50)
        a2 = np.maximum(0, z2)  # ReLU
        self._cache['z2'] = z2
        self._cache['a2'] = a2

        # Output layer (linear — no activation, since noise can be any value)
        out = np.nan_to_num(a2 @ self.W3 + self.b3, nan=0.0, posinf=50.0, neginf=-50.0)
        out = np.clip(out, -50, 50)
        self._cache['out'] = out

        return out

    def backward(self, d_out: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backpropagation through the MLP.

        Args:
            d_out: Gradient of loss w.r.t. output, shape (image_dim,)

        Returns:
            Dictionary of gradients for all parameters
        """
        a2 = self._cache['a2']
        z2 = self._cache['z2']
        a1 = self._cache['a1']
        z1 = self._cache['z1']
        inp = self._cache['inp']

        # Output layer gradients
        # out = a2 @ W3 + b3
        dW3 = np.nan_to_num(np.outer(a2, d_out))
        db3 = np.nan_to_num(d_out.copy())
        da2 = np.nan_to_num(d_out @ self.W3.T)

        # Layer 2 gradients (ReLU derivative: 1 if z>0, 0 otherwise)
        dz2 = da2 * (z2 > 0).astype(float)
        dW2 = np.nan_to_num(np.outer(a1, dz2))
        db2 = dz2.copy()
        da1 = np.nan_to_num(dz2 @ self.W2.T)

        # Layer 1 gradients
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = np.nan_to_num(np.outer(inp, dz1))
        db1 = dz1.copy()

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def update(self, grads: Dict[str, np.ndarray], lr: float):
        """SGD parameter update with global gradient clipping for stability."""
        # Global gradient clipping: compute total norm across all params,
        # then scale all gradients uniformly. This preserves relative magnitudes
        # while preventing any single large gradient from destabilizing training.
        max_norm = 1.0
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for key in grads:
                grads[key] = grads[key] * scale

        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']
        self.W3 -= lr * grads['W3']
        self.b3 -= lr * grads['b3']

    def get_param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.size for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])


# =============================================================================
# TRAINING
# =============================================================================

def train_diffusion(model: DenoisingMLP, schedule: NoiseSchedule,
                    data: np.ndarray, num_epochs: int = 50,
                    lr: float = 1e-3, seed: int = 42) -> List[float]:
    """
    Train the denoising model using the DDPM objective.

    The training loop is elegantly simple:
    1. Pick a random clean image from the dataset
    2. Pick a random timestep t
    3. Sample noise epsilon ~ N(0, I)
    4. Compute noisy image: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
    5. Predict noise: epsilon_theta = model(x_t, t)
    6. Loss = MSE(epsilon, epsilon_theta)
    7. Backprop and update

    This is it — no discriminator, no adversarial loss, no mode collapse.
    The model simply learns to predict which noise was added, at every noise level.

    Args:
        model: The denoising network
        schedule: Noise schedule with precomputed constants
        data: Training images, shape (N, image_dim)
        num_epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed

    Returns:
        List of average losses per epoch
    """
    rng = np.random.RandomState(seed)
    losses = []
    N = len(data)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Shuffle data each epoch
        indices = rng.permutation(N)

        for i in indices:
            x0 = data[i]  # Clean image, shape (image_dim,)

            # Sample random timestep — uniform over [0, T)
            # Each timestep teaches the model a different noise level
            t = rng.randint(0, schedule.T)

            # Sample noise and create noisy image
            noise = rng.randn(*x0.shape)
            x_t, _ = forward_diffusion(x0, t, schedule, noise=noise)

            # Get timestep embedding
            t_embed = sinusoidal_embedding(t, model.time_embed_dim)

            # Forward pass: predict the noise
            noise_pred = model.forward(x_t, t_embed)

            # MSE loss: L = ||epsilon - epsilon_theta||^2
            # The gradient is simply 2*(pred - target)/dim
            diff = noise_pred - noise
            loss = np.mean(diff ** 2)
            epoch_loss += loss

            # Backward pass
            d_out = 2.0 * diff / len(diff)  # gradient of MSE
            grads = model.backward(d_out)

            # Update parameters
            model.update(grads, lr)

        avg_loss = epoch_loss / N
        losses.append(avg_loss)

    return losses


# =============================================================================
# SAMPLING (GENERATION)
# =============================================================================

def sample_ddpm(model: DenoisingMLP, schedule: NoiseSchedule,
                image_shape: Tuple[int, ...], num_samples: int = 4,
                seed: int = 123) -> List[np.ndarray]:
    """
    Generate images by iteratively denoising from pure Gaussian noise.

    The sampling algorithm (Algorithm 2 from Ho et al., 2020):
    1. Start with x_T ~ N(0, I)
    2. For t = T-1, T-2, ..., 1, 0:
       a. Predict noise: eps_theta = model(x_t, t)
       b. Compute mean: mu = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
       c. Sample: x_{t-1} = mu + sigma_t * z  (z ~ N(0,I) for t>0, z=0 for t=0)

    The variance sigma_t^2 = beta_t is the simplest choice.
    Using the posterior variance (beta_t * (1-alpha_bar_{t-1})/(1-alpha_bar_t)) is better.

    Args:
        model: Trained denoising network
        schedule: Noise schedule
        image_shape: Shape of images to generate (H*W for flattened)
        num_samples: Number of images to generate
        seed: Random seed

    Returns:
        List of generated images (flattened)
    """
    rng = np.random.RandomState(seed)
    samples = []

    for s in range(num_samples):
        # Start from pure noise
        x = rng.randn(image_shape[0] if isinstance(image_shape, tuple) else image_shape)

        # Iterate backward through all timesteps
        for t in reversed(range(schedule.T)):
            t_embed = sinusoidal_embedding(t, model.time_embed_dim)

            # Predict noise at this timestep
            eps_theta = model.forward(x, t_embed)

            # Compute the mean of the reverse distribution
            # mu = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps_theta)
            coeff = schedule.betas[t] / schedule.sqrt_one_minus_alpha_bars[t]
            mu = schedule.sqrt_recip_alphas[t] * (x - coeff * eps_theta)

            if t > 0:
                # Add noise for all steps except the last one
                # Using posterior variance for better quality
                sigma = np.sqrt(schedule.posterior_variance[t])
                z = rng.randn(*x.shape)
                x = mu + sigma * z
            else:
                # Final step: no noise added (deterministic)
                x = mu

            # Clip to prevent numerical drift during long sampling chains.
            # Real implementations handle this via better architectures (U-Net
            # with normalization), but for our MLP we need explicit bounds.
            x = np.clip(x, -3.0, 3.0)

        samples.append(x)

    return samples


# =============================================================================
# DATASET: Simple 2D Patterns
# =============================================================================

def create_dataset(num_samples: int = 200, image_size: int = 8,
                   seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """
    Create a dataset of simple grayscale patterns for training.

    We use tiny images (8x8) so training completes in seconds with NumPy.
    Patterns include: horizontal bars, vertical bars, diagonal, cross, border.

    Each pattern has slight random variations (noise, shifts) so the model
    learns a distribution, not just memorization of a few templates.

    Args:
        num_samples: Total number of training images
        image_size: Width and height of images
        seed: Random seed

    Returns:
        (data, labels): Flattened images shape (N, H*W), pattern names
    """
    rng = np.random.RandomState(seed)
    dim = image_size * image_size
    data = []
    labels = []
    patterns = ["horizontal", "vertical", "cross", "diagonal", "border"]

    for _ in range(num_samples):
        pattern = patterns[rng.randint(len(patterns))]
        img = np.zeros((image_size, image_size))

        if pattern == "horizontal":
            # Horizontal bar at random height
            y = rng.randint(1, image_size - 1)
            thickness = rng.randint(1, 3)
            for dy in range(-thickness // 2, thickness // 2 + 1):
                row = y + dy
                if 0 <= row < image_size:
                    img[row, :] = 1.0

        elif pattern == "vertical":
            # Vertical bar at random position
            x = rng.randint(1, image_size - 1)
            thickness = rng.randint(1, 3)
            for dx in range(-thickness // 2, thickness // 2 + 1):
                col = x + dx
                if 0 <= col < image_size:
                    img[:, col] = 1.0

        elif pattern == "cross":
            # Cross pattern (horizontal + vertical)
            y = image_size // 2 + rng.randint(-1, 2)
            x = image_size // 2 + rng.randint(-1, 2)
            img[y, :] = 1.0
            img[:, x] = 1.0

        elif pattern == "diagonal":
            # Diagonal line
            direction = rng.choice([-1, 1])
            for i in range(image_size):
                j = i if direction == 1 else (image_size - 1 - i)
                img[i, j] = 1.0
                # Add some thickness
                if j + 1 < image_size:
                    img[i, j + 1] = 0.5

        elif pattern == "border":
            # Rectangle border
            margin = rng.randint(0, 2)
            img[margin, margin:image_size - margin] = 1.0
            img[image_size - 1 - margin, margin:image_size - margin] = 1.0
            img[margin:image_size - margin, margin] = 1.0
            img[margin:image_size - margin, image_size - 1 - margin] = 1.0

        # Add slight random noise for variation
        img += rng.randn(image_size, image_size) * 0.05
        # Normalize to [-1, 1] range (standard for diffusion models)
        img = np.clip(img, 0, 1) * 2 - 1

        data.append(img.flatten())
        labels.append(pattern)

    return np.array(data), labels


# =============================================================================
# VISUALIZATION (ASCII art for terminal output)
# =============================================================================

def render_ascii(image_flat: np.ndarray, size: int = 8) -> str:
    """Render a flattened image as ASCII art. Values in [-1, 1]."""
    img = image_flat.reshape(size, size)
    # Map [-1, 1] → characters
    chars = " .:-=+*#@"
    result = []
    for row in img:
        line = ""
        for val in row:
            # Map from [-1,1] to [0, len(chars)-1]
            idx = int(np.clip((val + 1) / 2 * (len(chars) - 1), 0, len(chars) - 1))
            line += chars[idx] + " "
        result.append(line)
    return "\n".join(result)


def compute_distribution_stats(data: np.ndarray, samples: List[np.ndarray]) -> Dict[str, float]:
    """
    Compare statistics between real data and generated samples.

    A basic quality check: if the model learned the distribution well,
    generated samples should have similar mean, std, and pixel value ranges
    as the training data.
    """
    samples_arr = np.array(samples)
    return {
        "data_mean": float(np.mean(data)),
        "data_std": float(np.std(data)),
        "gen_mean": float(np.mean(samples_arr)),
        "gen_std": float(np.std(samples_arr)),
        "data_min": float(np.min(data)),
        "data_max": float(np.max(data)),
        "gen_min": float(np.min(samples_arr)),
        "gen_max": float(np.max(samples_arr)),
        "mean_abs_pixel_diff": float(np.mean(np.abs(np.mean(data, axis=0) - np.mean(samples_arr, axis=0)))),
    }


# =============================================================================
# MAIN: Full DDPM Pipeline Demo
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    IMAGE_SIZE = 8
    IMAGE_DIM = IMAGE_SIZE * IMAGE_SIZE  # 64 pixels
    T = 50  # Number of diffusion steps (small for fast demo; real DDPM uses 1000)

    print("=" * 60)
    print("DENOISING DIFFUSION PROBABILISTIC MODEL (DDPM)")
    print("=" * 60)

    # ── Step 1: Create noise schedules and compare ──
    print("\n── Step 1: Noise Schedule ──")
    linear_schedule = NoiseSchedule(T, "linear")
    cosine_schedule = NoiseSchedule(T, "cosine")

    print(f"Timesteps: {T}")
    print(f"\nLinear schedule:")
    print(f"  beta range: [{linear_schedule.betas[0]:.6f}, {linear_schedule.betas[-1]:.6f}]")
    print(f"  alpha_bar at T: {linear_schedule.alpha_bars[-1]:.6f} (signal remaining)")
    print(f"\nCosine schedule:")
    print(f"  beta range: [{cosine_schedule.betas[0]:.6f}, {cosine_schedule.betas[-1]:.6f}]")
    print(f"  alpha_bar at T: {cosine_schedule.alpha_bars[-1]:.6f} (signal remaining)")

    # Show alpha_bar progression
    print(f"\nalpha_bar at key timesteps (cosine):")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(frac * (T - 1)), T - 1)
        print(f"  t={idx:3d}: alpha_bar={cosine_schedule.alpha_bars[idx]:.4f} "
              f"(signal: {cosine_schedule.alpha_bars[idx]*100:.1f}%, noise: {(1-cosine_schedule.alpha_bars[idx])*100:.1f}%)")

    # Use cosine schedule for training
    schedule = cosine_schedule

    # ── Step 2: Create dataset ──
    print("\n── Step 2: Dataset ──")
    data, labels = create_dataset(num_samples=200, image_size=IMAGE_SIZE)
    print(f"Created {len(data)} training images ({IMAGE_SIZE}x{IMAGE_SIZE})")
    print(f"Patterns: {set(labels)}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

    # Show a sample image
    print(f"\nSample training image ('{labels[0]}'):")
    print(render_ascii(data[0], IMAGE_SIZE))

    # ── Step 3: Demonstrate forward diffusion ──
    print("\n── Step 3: Forward Diffusion ──")
    print("Adding noise at increasing timesteps:")
    x0 = data[0]
    for t in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        x_t, _ = forward_diffusion(x0, t, schedule)
        snr = schedule.alpha_bars[t] / (1 - schedule.alpha_bars[t])
        print(f"\nt={t} (SNR={snr:.2f}):")
        print(render_ascii(x_t, IMAGE_SIZE))

    # ── Step 4: Build and train model ──
    print("\n── Step 4: Training ──")
    TIME_EMBED_DIM = 32
    HIDDEN_DIM = 128

    model = DenoisingMLP(
        image_dim=IMAGE_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        seed=42
    )
    print(f"Model parameters: {model.get_param_count():,}")
    print(f"Architecture: [{IMAGE_DIM}+{TIME_EMBED_DIM}] → {HIDDEN_DIM} → {HIDDEN_DIM} → {IMAGE_DIM}")

    print("\nTraining...")
    losses = train_diffusion(model, schedule, data, num_epochs=40, lr=5e-4, seed=42)

    print(f"  Epoch  1 loss: {losses[0]:.4f}")
    print(f"  Epoch 10 loss: {losses[9]:.4f}")
    print(f"  Epoch 20 loss: {losses[19]:.4f}")
    print(f"  Epoch 40 loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")

    # ── Step 5: Generate samples ──
    print("\n── Step 5: Sampling (Generation) ──")
    print(f"Generating 4 images by denoising from pure noise ({T} steps each)...")

    samples = sample_ddpm(model, schedule, IMAGE_DIM, num_samples=4, seed=123)

    for i, sample in enumerate(samples):
        print(f"\nGenerated sample {i+1}:")
        print(render_ascii(sample, IMAGE_SIZE))

    # ── Step 6: Compare distributions ──
    print("\n── Step 6: Distribution Comparison ──")
    more_samples = sample_ddpm(model, schedule, IMAGE_DIM, num_samples=50, seed=456)
    stats = compute_distribution_stats(data, more_samples)

    print(f"                Training Data    Generated")
    print(f"  Mean:         {stats['data_mean']:+.4f}          {stats['gen_mean']:+.4f}")
    print(f"  Std:           {stats['data_std']:.4f}           {stats['gen_std']:.4f}")
    print(f"  Range:        [{stats['data_min']:.2f}, {stats['data_max']:.2f}]    [{stats['gen_min']:.2f}, {stats['gen_max']:.2f}]")
    print(f"  Mean abs pixel diff: {stats['mean_abs_pixel_diff']:.4f}")

    # ── Step 7: Visualize denoising trajectory ──
    print("\n── Step 7: Denoising Trajectory ──")
    print("Watching a single sample evolve from noise to image:")

    rng = np.random.RandomState(999)
    x = rng.randn(IMAGE_DIM)

    # Show snapshots during denoising
    checkpoints = [T - 1, 3 * T // 4, T // 2, T // 4, 0]
    for t in reversed(range(schedule.T)):
        t_embed = sinusoidal_embedding(t, model.time_embed_dim)
        eps_theta = model.forward(x, t_embed)
        coeff = schedule.betas[t] / schedule.sqrt_one_minus_alpha_bars[t]
        mu = schedule.sqrt_recip_alphas[t] * (x - coeff * eps_theta)

        if t > 0:
            sigma = np.sqrt(schedule.posterior_variance[t])
            z = rng.randn(*x.shape)
            x = mu + sigma * z
        else:
            x = mu

        x = np.clip(x, -3.0, 3.0)

        if t in checkpoints:
            print(f"\nt={t} (denoising step {T - 1 - t + 1}/{T}):")
            print(render_ascii(x, IMAGE_SIZE))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Schedule: Cosine, {T} timesteps")
    print(f"  Model: 3-layer MLP, {model.get_param_count():,} parameters")
    print(f"  Training: 40 epochs on {len(data)} images")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Key insight: The model learns to predict noise at each")
    print(f"  timestep, then generation reverses the noising process.")
    print(f"  No adversarial training, no mode collapse — just MSE loss.")
