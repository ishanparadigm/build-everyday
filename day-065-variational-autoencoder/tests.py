"""
Tests for Day 65: Variational Autoencoder (VAE)

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    relu, relu_derivative, sigmoid, sigmoid_derivative,
    init_weights, VAE, generate_mixture_of_gaussians
)


class TestActivationFunctions(unittest.TestCase):
    """Test activation functions and their derivatives."""

    def test_relu_positive(self):
        """ReLU should pass through positive values unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_relu_negative(self):
        """ReLU should zero out negative values."""
        x = np.array([-1.0, -2.0, 0.0, 1.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(relu(x), expected)

    def test_relu_derivative_values(self):
        """ReLU derivative: 1 for positive, 0 for negative."""
        x = np.array([-2.0, -0.1, 0.0, 0.1, 5.0])
        result = relu_derivative(x)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[3], 1.0)
        self.assertEqual(result[4], 1.0)

    def test_sigmoid_bounds(self):
        """Sigmoid output should always be in (0, 1)."""
        x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        result = sigmoid(x)
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(result < 1))
        self.assertAlmostEqual(result[2], 0.5, places=5)

    def test_sigmoid_derivative_at_half(self):
        """sigmoid'(0.5) = 0.5 * 0.5 = 0.25"""
        s = np.array([0.5])
        self.assertAlmostEqual(sigmoid_derivative(s)[0], 0.25, places=5)


class TestWeightInit(unittest.TestCase):
    """Test Xavier weight initialization."""

    def test_init_shapes(self):
        """Weights and biases should have correct shapes."""
        W, b = init_weights(10, 5)
        self.assertEqual(W.shape, (10, 5))
        self.assertEqual(b.shape, (1, 5))

    def test_init_bias_zeros(self):
        """Biases should be initialized to zeros."""
        _, b = init_weights(10, 5)
        np.testing.assert_array_equal(b, np.zeros((1, 5)))

    def test_init_variance(self):
        """Weight variance should approximate 2 / (fan_in + fan_out)."""
        W, _ = init_weights(1000, 1000)
        expected_var = 2.0 / 2000
        actual_var = np.var(W)
        self.assertAlmostEqual(actual_var, expected_var, places=3)


class TestDataGeneration(unittest.TestCase):
    """Test synthetic data generation."""

    def test_shape(self):
        """Generated data should have correct shape."""
        data, labels = generate_mixture_of_gaussians(n_samples=100, n_clusters=5, dim=8)
        self.assertEqual(data.shape, (100, 8))
        self.assertEqual(labels.shape, (100,))

    def test_normalized_range(self):
        """Data should be normalized to [0, 1]."""
        data, _ = generate_mixture_of_gaussians(n_samples=500, n_clusters=5, dim=8)
        self.assertGreaterEqual(data.min(), 0.0)
        self.assertLessEqual(data.max(), 1.0)

    def test_cluster_count(self):
        """Should have the correct number of unique labels."""
        _, labels = generate_mixture_of_gaussians(n_samples=100, n_clusters=5, dim=8)
        self.assertEqual(len(np.unique(labels)), 5)


class TestVAE(unittest.TestCase):
    """Test VAE architecture and training."""

    def setUp(self):
        """Create a small VAE for testing."""
        np.random.seed(42)
        self.vae = VAE(input_dim=8, hidden_dim=32, latent_dim=2, learning_rate=0.001)
        self.data, self.labels = generate_mixture_of_gaussians(
            n_samples=200, n_clusters=4, dim=8, seed=42
        )

    def test_encode_shapes(self):
        """Encoder should output mu and log_var with correct shapes."""
        x = self.data[:10]
        mu, log_var, cache = self.vae.encode(x)
        self.assertEqual(mu.shape, (10, 2))
        self.assertEqual(log_var.shape, (10, 2))

    def test_reparameterize_shape(self):
        """Reparameterized z should have same shape as mu."""
        mu = np.zeros((10, 2))
        log_var = np.zeros((10, 2))
        z, epsilon = self.vae.reparameterize(mu, log_var)
        self.assertEqual(z.shape, (10, 2))
        self.assertEqual(epsilon.shape, (10, 2))

    def test_reparameterize_with_zero_variance(self):
        """With log_var=0 (var=1), z should equal mu + epsilon."""
        mu = np.ones((5, 2)) * 3.0
        log_var = np.zeros((5, 2))  # var = 1, std = 1
        z, epsilon = self.vae.reparameterize(mu, log_var)
        np.testing.assert_array_almost_equal(z, mu + epsilon)

    def test_decode_shapes(self):
        """Decoder output should match input dimensionality."""
        z = np.random.randn(10, 2)
        x_recon, cache = self.vae.decode(z)
        self.assertEqual(x_recon.shape, (10, 8))

    def test_decode_range(self):
        """Decoder output should be in [0, 1] due to sigmoid."""
        z = np.random.randn(50, 2) * 3  # Large z values
        x_recon, _ = self.vae.decode(z)
        self.assertTrue(np.all(x_recon >= 0))
        self.assertTrue(np.all(x_recon <= 1))

    def test_loss_components(self):
        """Loss should have non-negative reconstruction and KL terms."""
        x = self.data[:10]
        mu, log_var, _ = self.vae.encode(x)
        z, _ = self.vae.reparameterize(mu, log_var)
        x_recon, _ = self.vae.decode(z)

        total, recon, kl = self.vae.compute_loss(x, x_recon, mu, log_var)
        self.assertGreater(recon, 0, "Reconstruction loss should be positive")
        self.assertGreaterEqual(kl, 0, "KL divergence should be non-negative")
        self.assertAlmostEqual(total, recon + kl, places=5)

    def test_kl_zero_for_standard_normal(self):
        """KL(N(0,I) || N(0,I)) should be 0."""
        x = self.data[:10]
        x_recon = x.copy()  # Perfect reconstruction
        mu = np.zeros((10, 2))
        log_var = np.zeros((10, 2))  # var = 1

        _, _, kl = self.vae.compute_loss(x, x_recon, mu, log_var)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_training_reduces_loss(self):
        """Training should reduce the total loss."""
        # Get initial loss
        x = self.data[:64]
        mu, log_var, _ = self.vae.encode(x)
        z, _ = self.vae.reparameterize(mu, log_var)
        x_recon, _ = self.vae.decode(z)
        initial_loss, _, _ = self.vae.compute_loss(x, x_recon, mu, log_var)

        # Train for a few epochs
        self.vae.train(self.data, epochs=30, batch_size=64, verbose=False)

        # Check loss decreased
        final_loss = self.vae.history['total_loss'][-1]
        self.assertLess(final_loss, initial_loss,
                        "Training should reduce loss")

    def test_generate_shape(self):
        """Generated samples should have correct shape."""
        generated = self.vae.generate(n_samples=10)
        self.assertEqual(generated.shape, (10, 8))

    def test_interpolate_shape(self):
        """Interpolation should produce the right number of steps."""
        interps = self.vae.interpolate(self.data[0], self.data[1], n_steps=7)
        self.assertEqual(interps.shape, (7, 8))


if __name__ == '__main__':
    unittest.main()
