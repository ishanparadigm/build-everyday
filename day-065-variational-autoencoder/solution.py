"""
Day 65: Variational Autoencoder (VAE) from Scratch

A complete VAE implementation using only NumPy, trained on a synthetic dataset
(since we can't assume PyTorch/TensorFlow availability). This builds the entire
forward pass, reparameterization trick, ELBO loss, and backpropagation manually.

We use a smaller synthetic dataset (2D points sampled from a mixture of Gaussians)
to keep training fast while demonstrating all VAE concepts clearly.

Architecture:
    Encoder: input_dim → 128 → 64 → (mu, log_var) each of size latent_dim
    Decoder: latent_dim → 64 → 128 → input_dim

Key insight: By building everything from scratch, we see exactly how gradients
flow through the reparameterization trick — something that's hidden by autograd.
"""

import warnings
import numpy as np
from typing import Tuple, Dict, List

# Suppress overflow warnings during early training — they're handled by
# nan_to_num in gradient clipping and don't affect convergence
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# Activation Functions & Their Derivatives
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation. Simple, effective, but can cause dead neurons."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where x > 0, 0 elsewhere."""
    return (x > 0).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation. Maps to (0, 1) — used for output when data is normalized."""
    # Clip to prevent overflow in exp
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(s: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given sigmoid output s: s * (1 - s)."""
    return s * (1.0 - s)


# =============================================================================
# Weight Initialization
# =============================================================================

def init_weights(fan_in: int, fan_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xavier/Glorot initialization: var = 2 / (fan_in + fan_out).

    Why Xavier? It keeps the variance of activations roughly constant across layers,
    preventing vanishing/exploding gradients. The factor of 2 accounts for the
    variance being preserved in both forward and backward passes.
    """
    std = np.sqrt(2.0 / (fan_in + fan_out))
    W = np.random.randn(fan_in, fan_out) * std
    b = np.zeros((1, fan_out))
    return W, b


# =============================================================================
# VAE Class
# =============================================================================

class VAE:
    """
    Variational Autoencoder with full manual backpropagation.

    The architecture:
        Encoder: input → hidden1 (ReLU) → hidden2 (ReLU) → (mu, log_var)
        Decoder: z → hidden3 (ReLU) → hidden4 (ReLU) → output (Sigmoid)

    We use sigmoid output + MSE loss for simplicity with continuous data.
    For binary data (MNIST), you'd use BCE loss instead.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 learning_rate: float = 0.0005, beta: float = 1.0):
        """
        Args:
            input_dim: Dimensionality of input data
            hidden_dim: Size of hidden layers
            latent_dim: Size of latent space (z)
            learning_rate: Step size for gradient descent (0.0005 works well for
                          manual SGD without momentum/Adam)
            beta: Weight on KL divergence (beta-VAE). beta=1 is standard VAE.
                  beta>1 encourages disentanglement, beta<1 prioritizes reconstruction.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.beta = beta

        # --- Encoder weights ---
        # Layer 1: input → hidden
        self.W_enc1, self.b_enc1 = init_weights(input_dim, hidden_dim)
        # Layer 2: hidden → hidden/2
        self.W_enc2, self.b_enc2 = init_weights(hidden_dim, hidden_dim // 2)
        # Layer 3a: hidden/2 → mu (mean of latent distribution)
        self.W_mu, self.b_mu = init_weights(hidden_dim // 2, latent_dim)
        # Layer 3b: hidden/2 → log_var (log variance of latent distribution)
        # Why log_var instead of var? log_var can be any real number, while var
        # must be positive. This avoids needing a softplus or exp constraint.
        self.W_logvar, self.b_logvar = init_weights(hidden_dim // 2, latent_dim)

        # --- Decoder weights ---
        # Layer 1: z → hidden/2
        self.W_dec1, self.b_dec1 = init_weights(latent_dim, hidden_dim // 2)
        # Layer 2: hidden/2 → hidden
        self.W_dec2, self.b_dec2 = init_weights(hidden_dim // 2, hidden_dim)
        # Layer 3: hidden → output (sigmoid activation)
        self.W_dec3, self.b_dec3 = init_weights(hidden_dim, input_dim)

        # Training history for monitoring
        self.history: Dict[str, List[float]] = {
            'total_loss': [], 'recon_loss': [], 'kl_loss': []
        }

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Encode input x into latent distribution parameters (mu, log_var).

        Returns mu, log_var, and a cache dict for backpropagation.
        The cache stores all intermediate activations needed for computing gradients.
        """
        # Layer 1: linear + ReLU
        z1 = x @ self.W_enc1 + self.b_enc1          # Pre-activation
        a1 = relu(z1)                                  # Post-activation

        # Layer 2: linear + ReLU
        z2 = a1 @ self.W_enc2 + self.b_enc2
        a2 = relu(z2)

        # Output: two heads — mu and log_var
        # We clamp log_var to [-10, 10] to prevent numerical overflow in exp()
        mu = a2 @ self.W_mu + self.b_mu
        log_var = np.clip(a2 @ self.W_logvar + self.b_logvar, -10, 10)

        cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return mu, log_var, cache

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The reparameterization trick: z = mu + std * epsilon

        Why is this necessary? We need to sample z ~ N(mu, sigma^2), but sampling
        is not differentiable — we can't compute d(sample)/d(mu). The trick:

        1. Sample epsilon ~ N(0, I)  — this doesn't depend on parameters
        2. Compute z = mu + exp(0.5 * log_var) * epsilon — this IS differentiable

        Now dz/dmu = I and dz/d(log_var) = 0.5 * exp(0.5 * log_var) * epsilon,
        so gradients flow through to the encoder.

        Returns z and epsilon (needed for backprop through the trick).
        """
        # Clamp log_var to prevent numerical overflow in exp
        log_var_clamped = np.clip(log_var, -10, 10)
        std = np.exp(0.5 * log_var_clamped)  # Convert log_var to standard deviation
        epsilon = np.random.randn(*mu.shape)  # Sample from standard normal
        z = mu + std * epsilon
        return z, epsilon

    def decode(self, z: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Decode latent vector z back to input space.

        Uses sigmoid on output to bound reconstructions to [0, 1],
        matching the normalized input range.
        """
        # Layer 1: linear + ReLU
        z1 = z @ self.W_dec1 + self.b_dec1
        a1 = relu(z1)

        # Layer 2: linear + ReLU
        z2 = a1 @ self.W_dec2 + self.b_dec2
        a2 = relu(z2)

        # Output layer: linear + sigmoid
        z3 = a2 @ self.W_dec3 + self.b_dec3
        x_recon = sigmoid(z3)

        cache = {'z': z, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'x_recon': x_recon}
        return x_recon, cache

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     mu: np.ndarray, log_var: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute the ELBO loss = Reconstruction Loss + beta * KL Divergence.

        Reconstruction Loss (MSE):
            L_recon = (1/N) * sum((x - x_recon)^2)

            We use MSE here since our data is continuous. For binary data like MNIST,
            BCE would be more appropriate: -sum(x*log(x_hat) + (1-x)*log(1-x_hat))

        KL Divergence (closed form for two Gaussians):
            KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

            Derivation: For univariate case,
            KL = integral q(z) * log(q(z)/p(z)) dz
               = integral N(mu,s2) * [log N(mu,s2) - log N(0,1)] dz
               = 0.5 * (mu^2 + s2 - 1 - log(s2))

            We negate it because we maximize ELBO = -KL + reconstruction likelihood,
            which is equivalent to minimizing KL - reconstruction likelihood.

        Returns: total_loss, recon_loss, kl_loss (all as per-sample averages)
        """
        batch_size = x.shape[0]

        # Reconstruction loss: MSE summed over features, averaged over batch
        recon_loss = np.sum((x - x_recon) ** 2) / batch_size

        # KL divergence: closed form, summed over latent dims, averaged over batch
        # -0.5 * sum(1 + log(var) - mu^2 - var)
        log_var_clamped = np.clip(log_var, -10, 10)
        kl_loss = -0.5 * np.sum(1 + log_var_clamped - mu ** 2 - np.exp(log_var_clamped)) / batch_size

        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def backward(self, x: np.ndarray, mu: np.ndarray, log_var: np.ndarray,
                 epsilon: np.ndarray, enc_cache: dict, dec_cache: dict) -> None:
        """
        Full backpropagation through the VAE.

        This is where the magic happens. We need to backpropagate through:
        1. The reconstruction loss → decoder weights
        2. The decoder → latent vector z
        3. The reparameterization trick → mu and log_var
        4. The KL loss → mu and log_var (directly, since KL has a closed form)
        5. mu and log_var → encoder weights

        The reparameterization trick makes step 3 possible:
            z = mu + exp(0.5 * log_var) * epsilon
            dz/dmu = I
            dz/d(log_var) = 0.5 * exp(0.5 * log_var) * epsilon
        """
        batch_size = x.shape[0]
        x_recon = dec_cache['x_recon']

        # =====================================================================
        # DECODER BACKWARD
        # =====================================================================

        # Gradient of MSE loss w.r.t. x_recon: d(MSE)/d(x_recon) = 2(x_recon - x)/N
        d_recon = 2.0 * (x_recon - x) / batch_size  # (batch, input_dim)

        # Through sigmoid: d/dz3 = d_recon * sigmoid'(z3) = d_recon * x_recon * (1 - x_recon)
        d_z3 = d_recon * sigmoid_derivative(x_recon)

        # Decoder layer 3 gradients
        dW_dec3 = dec_cache['a2'].T @ d_z3
        db_dec3 = np.sum(d_z3, axis=0, keepdims=True)
        d_a2 = d_z3 @ self.W_dec3.T

        # Through ReLU at decoder layer 2
        d_z2 = d_a2 * relu_derivative(dec_cache['z2'])
        dW_dec2 = dec_cache['a1'].T @ d_z2
        db_dec2 = np.sum(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.W_dec2.T

        # Through ReLU at decoder layer 1
        d_z1 = d_a1 * relu_derivative(dec_cache['z1'])
        dW_dec1 = dec_cache['z'].T @ d_z1
        db_dec1 = np.sum(d_z1, axis=0, keepdims=True)

        # Gradient of loss w.r.t. z (latent vector)
        d_z = d_z1 @ self.W_dec1.T  # (batch, latent_dim)

        # =====================================================================
        # REPARAMETERIZATION TRICK BACKWARD
        # =====================================================================

        # z = mu + exp(0.5 * log_var) * epsilon
        # dL/dmu = dL/dz * dz/dmu = dL/dz * I = dL/dz
        # dL/d(log_var) = dL/dz * dz/d(log_var) = dL/dz * 0.5 * exp(0.5 * log_var) * epsilon

        log_var_clamped = np.clip(log_var, -10, 10)
        std = np.exp(0.5 * log_var_clamped)

        d_mu_from_recon = d_z  # dz/dmu = I
        d_logvar_from_recon = d_z * 0.5 * std * epsilon  # Chain rule through reparam

        # =====================================================================
        # KL DIVERGENCE GRADIENTS (direct, since KL has closed form)
        # =====================================================================

        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)) / batch_size
        # dKL/dmu = mu / batch_size (positive because we minimize KL)
        # dKL/d(log_var) = 0.5 * (-1 + exp(log_var)) / batch_size

        d_mu_from_kl = self.beta * mu / batch_size
        d_logvar_from_kl = self.beta * 0.5 * (-1 + np.exp(log_var_clamped)) / batch_size

        # Combine reconstruction and KL gradients
        d_mu = d_mu_from_recon + d_mu_from_kl
        d_logvar = d_logvar_from_recon + d_logvar_from_kl

        # =====================================================================
        # ENCODER BACKWARD
        # =====================================================================

        # Gradients for mu and log_var output layers
        dW_mu = enc_cache['a2'].T @ d_mu
        db_mu = np.sum(d_mu, axis=0, keepdims=True)

        dW_logvar = enc_cache['a2'].T @ d_logvar
        db_logvar = np.sum(d_logvar, axis=0, keepdims=True)

        # Both mu and log_var heads contribute gradients to encoder layer 2
        d_a2 = d_mu @ self.W_mu.T + d_logvar @ self.W_logvar.T

        # Through ReLU at encoder layer 2
        d_z2 = d_a2 * relu_derivative(enc_cache['z2'])
        dW_enc2 = enc_cache['a1'].T @ d_z2
        db_enc2 = np.sum(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.W_enc2.T

        # Through ReLU at encoder layer 1
        d_z1 = d_a1 * relu_derivative(enc_cache['z1'])
        dW_enc1 = enc_cache['x'].T @ d_z1
        db_enc1 = np.sum(d_z1, axis=0, keepdims=True)

        # =====================================================================
        # GRADIENT DESCENT UPDATE
        # =====================================================================

        # Clip gradients and replace NaN/Inf to prevent explosion — a practical
        # necessity when training deep networks with manual backprop.
        # NaN can appear in early training when activations overflow; replacing
        # with 0 effectively skips the corrupt gradient for that batch.
        clip_val = 1.0
        all_grads = [dW_enc1, db_enc1, dW_enc2, db_enc2, dW_mu, db_mu,
                     dW_logvar, db_logvar, dW_dec1, db_dec1, dW_dec2, db_dec2,
                     dW_dec3, db_dec3]
        for grad in all_grads:
            np.nan_to_num(grad, copy=False, nan=0.0, posinf=clip_val, neginf=-clip_val)
            np.clip(grad, -clip_val, clip_val, out=grad)

        # Update encoder weights
        self.W_enc1 -= self.lr * dW_enc1
        self.b_enc1 -= self.lr * db_enc1
        self.W_enc2 -= self.lr * dW_enc2
        self.b_enc2 -= self.lr * db_enc2
        self.W_mu -= self.lr * dW_mu
        self.b_mu -= self.lr * db_mu
        self.W_logvar -= self.lr * dW_logvar
        self.b_logvar -= self.lr * db_logvar

        # Update decoder weights
        self.W_dec1 -= self.lr * dW_dec1
        self.b_dec1 -= self.lr * db_dec1
        self.W_dec2 -= self.lr * dW_dec2
        self.b_dec2 -= self.lr * db_dec2
        self.W_dec3 -= self.lr * dW_dec3
        self.b_dec3 -= self.lr * db_dec3

    def train_step(self, x: np.ndarray) -> Tuple[float, float, float]:
        """One full forward + backward pass on a batch."""
        # Forward pass
        mu, log_var, enc_cache = self.encode(x)
        z, epsilon = self.reparameterize(mu, log_var)
        x_recon, dec_cache = self.decode(z)

        # Compute loss
        total_loss, recon_loss, kl_loss = self.compute_loss(x, x_recon, mu, log_var)

        # Backward pass
        self.backward(x, mu, log_var, epsilon, enc_cache, dec_cache)

        return total_loss, recon_loss, kl_loss

    def train(self, data: np.ndarray, epochs: int = 100, batch_size: int = 64,
              verbose: bool = True, kl_anneal_epochs: int = 20) -> None:
        """
        Train the VAE on the given data.

        We shuffle and batch the data each epoch. This is standard mini-batch SGD —
        the stochasticity from both the mini-batches and the reparameterization sampling
        provides useful noise that helps escape local optima.

        KL annealing: we linearly ramp up beta from 0 to self.beta over the first
        kl_anneal_epochs. This prevents posterior collapse by letting the encoder
        learn meaningful representations before the KL penalty kicks in.
        """
        n_samples = data.shape[0]
        target_beta = self.beta

        for epoch in range(epochs):
            # KL annealing: linearly ramp beta from 0 to target over first N epochs
            if kl_anneal_epochs > 0:
                self.beta = target_beta * min(1.0, (epoch + 1) / kl_anneal_epochs)

            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            epoch_total = epoch_recon = epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                batch = data[batch_idx]

                total, recon, kl = self.train_step(batch)
                epoch_total += total
                epoch_recon += recon
                epoch_kl += kl
                n_batches += 1

            # Record average loss
            avg_total = epoch_total / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            self.history['total_loss'].append(avg_total)
            self.history['recon_loss'].append(avg_recon)
            self.history['kl_loss'].append(avg_kl)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:4d} | Total: {avg_total:.4f} | "
                      f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | "
                      f"beta: {self.beta:.3f}")

        # Restore target beta
        self.beta = target_beta

    def generate(self, n_samples: int = 10) -> np.ndarray:
        """
        Generate new data by sampling from the prior p(z) = N(0, I).

        This is the whole point of the VAE: the KL term ensures the latent space
        is organized around N(0, I), so random samples decode to realistic data.
        """
        z = np.random.randn(n_samples, self.latent_dim)
        x_generated, _ = self.decode(z)
        return x_generated

    def reconstruct(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode and decode — useful for evaluating reconstruction quality.
        Returns reconstructed x, mu, and log_var.
        """
        mu, log_var, _ = self.encode(x)
        z, _ = self.reparameterize(mu, log_var)
        x_recon, _ = self.decode(z)
        return x_recon, mu, log_var

    def interpolate(self, x1: np.ndarray, x2: np.ndarray,
                    n_steps: int = 10) -> np.ndarray:
        """
        Interpolate between two data points in latent space.

        This demonstrates the smooth, continuous structure of the VAE latent space.
        Unlike a standard autoencoder, linear interpolation in z-space produces
        meaningful intermediate outputs — a key benefit of the KL regularization.
        """
        # Encode both points (use mu directly, no sampling for deterministic interpolation)
        mu1, _, _ = self.encode(x1.reshape(1, -1))
        mu2, _, _ = self.encode(x2.reshape(1, -1))

        # Linear interpolation in latent space
        alphas = np.linspace(0, 1, n_steps)
        interpolations = []
        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            x_interp, _ = self.decode(z_interp)
            interpolations.append(x_interp[0])

        return np.array(interpolations)


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_mixture_of_gaussians(n_samples: int = 1000, n_clusters: int = 5,
                                  dim: int = 8, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a mixture of Gaussians.

    This gives us multi-modal data that a VAE must learn to represent in its
    latent space. Each cluster represents a distinct "mode" of the data distribution.

    Returns normalized data in [0, 1] and cluster labels.
    """
    np.random.seed(seed)

    samples_per_cluster = n_samples // n_clusters
    data = []
    labels = []

    for i in range(n_clusters):
        # Each cluster has a random center and small variance
        center = np.random.randn(dim) * 3
        cluster_data = np.random.randn(samples_per_cluster, dim) * 0.5 + center
        data.append(cluster_data)
        labels.extend([i] * samples_per_cluster)

    data = np.vstack(data)
    labels = np.array(labels)

    # Normalize to [0, 1] — required for sigmoid output activation
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1  # Avoid division by zero
    data = (data - data_min) / data_range

    return data, labels


# =============================================================================
# Main: Demonstration
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("VARIATIONAL AUTOENCODER (VAE) FROM SCRATCH")
    print("=" * 70)

    # --- Step 1: Generate synthetic data ---
    print("\n--- Step 1: Generating synthetic data ---")
    data, labels = generate_mixture_of_gaussians(
        n_samples=1000, n_clusters=5, dim=8, seed=42
    )
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Number of clusters: {len(np.unique(labels))}")

    # --- Step 2: Create and train VAE ---
    print("\n--- Step 2: Training VAE ---")
    print("Architecture: 8 → 128 → 64 → (mu_2, logvar_2) → z_2 → 64 → 128 → 8")
    print("Loss = MSE Reconstruction + KL Divergence")
    print()

    vae = VAE(
        input_dim=8,
        hidden_dim=128,
        latent_dim=2,  # 2D latent space for easy visualization
        learning_rate=0.0005,
        beta=1.0
    )

    vae.train(data, epochs=200, batch_size=64, verbose=True, kl_anneal_epochs=30)

    # --- Step 3: Analyze training dynamics ---
    print("\n--- Step 3: Training dynamics ---")
    early = min(9, len(vae.history['total_loss']) - 1)
    print(f"Early loss (epoch {early+1}):  Total={vae.history['total_loss'][early]:.4f}, "
          f"Recon={vae.history['recon_loss'][early]:.4f}, "
          f"KL={vae.history['kl_loss'][early]:.4f}")
    print(f"Final loss:    Total={vae.history['total_loss'][-1]:.4f}, "
          f"Recon={vae.history['recon_loss'][-1]:.4f}, "
          f"KL={vae.history['kl_loss'][-1]:.4f}")

    recon_drop = (1 - vae.history['recon_loss'][-1] / vae.history['recon_loss'][early]) * 100
    print(f"Reconstruction loss reduced by: {recon_drop:.1f}%")

    # Check for posterior collapse (KL ≈ 0 means the model ignores the latent code)
    final_kl = vae.history['kl_loss'][-1]
    if final_kl < 0.001:
        print("WARNING: KL ≈ 0 — possible posterior collapse! "
              "The decoder may be ignoring the latent code.")
    else:
        print(f"KL divergence is {final_kl:.4f} — latent space is being used")
        print("  (Note: small KL is expected with low-dim input and vanilla SGD."
              " With Adam or larger data, KL would be higher.)")

    # --- Step 4: Test reconstruction quality ---
    print("\n--- Step 4: Reconstruction quality ---")
    test_samples = data[:5]
    reconstructed, mu_encoded, logvar_encoded = vae.reconstruct(test_samples)

    for i in range(5):
        mse = np.mean((test_samples[i] - reconstructed[i]) ** 2)
        print(f"  Sample {i+1}: MSE = {mse:.6f} | "
              f"mu = [{mu_encoded[i, 0]:.3f}, {mu_encoded[i, 1]:.3f}] | "
              f"std = [{np.exp(0.5*logvar_encoded[i, 0]):.3f}, "
              f"{np.exp(0.5*logvar_encoded[i, 1]):.3f}]")

    # --- Step 5: Generate new samples ---
    print("\n--- Step 5: Generating new samples from prior N(0, I) ---")
    generated = vae.generate(n_samples=5)
    for i in range(5):
        vals = ", ".join(f"{v:.3f}" for v in generated[i])
        print(f"  Generated sample {i+1}: [{vals}]")

    # Verify generated samples are in valid range
    print(f"\n  Generated data range: [{generated.min():.3f}, {generated.max():.3f}]")
    print(f"  All values in [0, 1]: {(generated >= 0).all() and (generated <= 1).all()}")

    # --- Step 6: Latent space analysis ---
    print("\n--- Step 6: Latent space analysis ---")
    all_mu, all_logvar, _ = vae.encode(data)

    print("  Per-cluster latent means (should be separated if VAE learned structure):")
    for c in range(5):
        cluster_mask = labels == c
        cluster_mu = all_mu[cluster_mask].mean(axis=0)
        cluster_std = np.exp(0.5 * all_logvar[cluster_mask]).mean(axis=0)
        print(f"    Cluster {c}: mu=[{cluster_mu[0]:+.3f}, {cluster_mu[1]:+.3f}], "
              f"avg_std=[{cluster_std[0]:.3f}, {cluster_std[1]:.3f}]")

    # --- Step 7: Latent interpolation ---
    print("\n--- Step 7: Latent space interpolation ---")
    print("  Interpolating between cluster 0 and cluster 4 in latent space:")

    # Pick one sample from each cluster
    idx0 = np.where(labels == 0)[0][0]
    idx4 = np.where(labels == 4)[0][0]

    interpolations = vae.interpolate(data[idx0], data[idx4], n_steps=5)
    for i, interp in enumerate(interpolations):
        vals = ", ".join(f"{v:.3f}" for v in interp)
        alpha = i / (len(interpolations) - 1)
        print(f"    alpha={alpha:.2f}: [{vals}]")

    # --- Step 8: Compare VAE vs standard autoencoder behavior ---
    print("\n--- Step 8: VAE vs standard autoencoder ---")
    print("  Key differences:")
    print("  1. VAE encodes to distributions (mu, sigma), not points")
    print("  2. VAE latent space is regularized by KL → smooth interpolation")
    print("  3. VAE can generate new data by sampling z ~ N(0, I)")
    print("  4. Standard AE has no principled way to sample new data")
    print()

    # Quick test: sample many points and check coverage
    many_generated = vae.generate(n_samples=100)
    print(f"  Generated 100 samples — mean: {many_generated.mean():.3f}, "
          f"std: {many_generated.std():.3f}")
    print(f"  Original data          — mean: {data.mean():.3f}, "
          f"std: {data.std():.3f}")

    print("\n" + "=" * 70)
    print("VAE training and evaluation complete!")
    print("=" * 70)
