"""
Tests for Day 060: Denoising Diffusion Probabilistic Model (DDPM)

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    NoiseSchedule, forward_diffusion, sinusoidal_embedding,
    DenoisingMLP, train_diffusion, sample_ddpm, create_dataset,
    render_ascii, compute_distribution_stats,
)


class TestNoiseSchedule(unittest.TestCase):
    """Tests for the noise schedule computation."""

    def test_linear_schedule_shape(self):
        """Beta array should have T elements."""
        schedule = NoiseSchedule(100, "linear")
        self.assertEqual(len(schedule.betas), 100)
        self.assertEqual(len(schedule.alphas), 100)
        self.assertEqual(len(schedule.alpha_bars), 100)

    def test_alpha_bar_monotonically_decreasing(self):
        """alpha_bar should decrease over time (signal fades)."""
        schedule = NoiseSchedule(100, "linear")
        for i in range(1, len(schedule.alpha_bars)):
            self.assertLess(schedule.alpha_bars[i], schedule.alpha_bars[i - 1])

    def test_alpha_bar_boundaries(self):
        """alpha_bar should start near 1 and end near 0."""
        schedule = NoiseSchedule(100, "cosine")
        self.assertGreater(schedule.alpha_bars[0], 0.9)
        self.assertLess(schedule.alpha_bars[-1], 0.1)

    def test_cosine_schedule_smoother(self):
        """Cosine schedule should preserve more signal in early steps than linear."""
        linear = NoiseSchedule(100, "linear")
        cosine = NoiseSchedule(100, "cosine")
        # At t=25 (quarter way), cosine should have higher alpha_bar
        self.assertGreater(cosine.alpha_bars[25], linear.alpha_bars[25])

    def test_get_forward_params(self):
        """get_forward_params should return correct sqrt values."""
        schedule = NoiseSchedule(50, "linear")
        sqrt_ab, sqrt_1_ab = schedule.get_forward_params(10)
        self.assertAlmostEqual(sqrt_ab ** 2, schedule.alpha_bars[10], places=5)
        self.assertAlmostEqual(sqrt_1_ab ** 2, 1 - schedule.alpha_bars[10], places=5)


class TestForwardDiffusion(unittest.TestCase):
    """Tests for the forward noising process."""

    def test_output_shape(self):
        """Output should match input shape."""
        schedule = NoiseSchedule(50, "linear")
        x0 = np.random.randn(64)
        x_t, noise = forward_diffusion(x0, 10, schedule)
        self.assertEqual(x_t.shape, x0.shape)
        self.assertEqual(noise.shape, x0.shape)

    def test_t0_preserves_signal(self):
        """At t=0, the noisy image should be very close to the original."""
        schedule = NoiseSchedule(50, "cosine")
        x0 = np.ones(64) * 0.5
        noise = np.random.randn(64) * 0.1
        x_t, _ = forward_diffusion(x0, 0, schedule, noise=noise)
        # At t=0, alpha_bar is close to 1, so x_t ≈ x_0
        self.assertLess(np.mean((x_t - x0) ** 2), 0.1)

    def test_large_t_approaches_noise(self):
        """At t=T-1, the result should be dominated by noise."""
        schedule = NoiseSchedule(100, "linear")
        x0 = np.ones(64)
        x_t, noise = forward_diffusion(x0, 99, schedule)
        # Correlation with noise should be high
        corr_noise = np.corrcoef(x_t, noise)[0, 1]
        self.assertGreater(abs(corr_noise), 0.5)

    def test_deterministic_with_fixed_noise(self):
        """Same noise should give same result."""
        schedule = NoiseSchedule(50, "linear")
        x0 = np.random.randn(64)
        noise = np.random.randn(64)
        x_t1, _ = forward_diffusion(x0, 25, schedule, noise=noise)
        x_t2, _ = forward_diffusion(x0, 25, schedule, noise=noise)
        np.testing.assert_array_almost_equal(x_t1, x_t2)


class TestSinusoidalEmbedding(unittest.TestCase):
    """Tests for timestep embeddings."""

    def test_output_shape(self):
        """Embedding should have the requested dimension."""
        emb = sinusoidal_embedding(10, 32)
        self.assertEqual(emb.shape, (32,))

    def test_different_timesteps_different_embeddings(self):
        """Different timesteps should produce different embeddings."""
        e1 = sinusoidal_embedding(5, 32)
        e2 = sinusoidal_embedding(50, 32)
        self.assertGreater(np.linalg.norm(e1 - e2), 0.1)

    def test_bounded_values(self):
        """Sin/cos values should be in [-1, 1]."""
        emb = sinusoidal_embedding(42, 64)
        self.assertTrue(np.all(emb >= -1.0))
        self.assertTrue(np.all(emb <= 1.0))


class TestDenoisingMLP(unittest.TestCase):
    """Tests for the neural network."""

    def test_forward_output_shape(self):
        """Output should match image dimension."""
        model = DenoisingMLP(64, time_embed_dim=32, hidden_dim=64)
        x = np.random.randn(64)
        t_emb = sinusoidal_embedding(5, 32)
        out = model.forward(x, t_emb)
        self.assertEqual(out.shape, (64,))

    def test_backward_gradient_shapes(self):
        """Gradients should match parameter shapes."""
        model = DenoisingMLP(64, time_embed_dim=32, hidden_dim=64)
        x = np.random.randn(64)
        t_emb = sinusoidal_embedding(5, 32)
        model.forward(x, t_emb)
        d_out = np.random.randn(64)
        grads = model.backward(d_out)
        self.assertEqual(grads['W1'].shape, model.W1.shape)
        self.assertEqual(grads['W2'].shape, model.W2.shape)
        self.assertEqual(grads['W3'].shape, model.W3.shape)
        self.assertEqual(grads['b1'].shape, model.b1.shape)
        self.assertEqual(grads['b2'].shape, model.b2.shape)
        self.assertEqual(grads['b3'].shape, model.b3.shape)

    def test_param_count(self):
        """Parameter count should reflect the architecture."""
        model = DenoisingMLP(64, time_embed_dim=32, hidden_dim=128)
        # W1: (96, 128) + b1: 128 + W2: (128, 128) + b2: 128 + W3: (128, 64) + b3: 64
        expected = 96 * 128 + 128 + 128 * 128 + 128 + 128 * 64 + 64
        self.assertEqual(model.get_param_count(), expected)


class TestTrainingAndSampling(unittest.TestCase):
    """Tests for the training loop and sampling."""

    def test_training_reduces_loss(self):
        """Loss should decrease over training epochs."""
        schedule = NoiseSchedule(20, "linear")
        data, _ = create_dataset(50, image_size=8, seed=42)
        model = DenoisingMLP(64, time_embed_dim=32, hidden_dim=64, seed=42)
        losses = train_diffusion(model, schedule, data, num_epochs=10, lr=5e-4, seed=42)
        self.assertGreater(len(losses), 0)
        # Loss should generally decrease (allow some noise)
        self.assertLess(losses[-1], losses[0])

    def test_sampling_returns_correct_count(self):
        """Should return the requested number of samples."""
        schedule = NoiseSchedule(10, "linear")
        data, _ = create_dataset(30, image_size=8, seed=42)
        model = DenoisingMLP(64, time_embed_dim=32, hidden_dim=64, seed=42)
        train_diffusion(model, schedule, data, num_epochs=2, lr=5e-4, seed=42)
        samples = sample_ddpm(model, schedule, 64, num_samples=3, seed=42)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (64,))

    def test_samples_finite(self):
        """Generated samples should not contain NaN or Inf."""
        schedule = NoiseSchedule(10, "linear")
        data, _ = create_dataset(30, image_size=8, seed=42)
        model = DenoisingMLP(64, time_embed_dim=32, hidden_dim=64, seed=42)
        train_diffusion(model, schedule, data, num_epochs=3, lr=5e-4, seed=42)
        samples = sample_ddpm(model, schedule, 64, num_samples=2, seed=42)
        for s in samples:
            self.assertTrue(np.all(np.isfinite(s)))


if __name__ == "__main__":
    unittest.main()
