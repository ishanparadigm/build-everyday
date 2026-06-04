"""
Day 064: GAN from Scratch — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np

from my_solution import (
    relu, relu_derivative, leaky_relu, leaky_relu_derivative,
    sigmoid, tanh, tanh_derivative,
    bce_loss, bce_loss_gradient,
    Layer, Generator, Discriminator,
    make_mixture_of_gaussians, compute_mode_coverage,
    wasserstein_estimate, train_gan,
)


class TestActivations(unittest.TestCase):
    """Test activation functions and their derivatives."""

    def test_relu(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(relu(x), expected)

    def test_relu_derivative(self):
        x = np.array([-2.0, -0.1, 0.1, 2.0])
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(relu_derivative(x), expected)

    def test_leaky_relu(self):
        x = np.array([-2.0, 0.0, 2.0])
        result = leaky_relu(x, alpha=0.2)
        expected = np.array([-0.4, 0.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_leaky_relu_derivative(self):
        x = np.array([-1.0, 1.0])
        result = leaky_relu_derivative(x, alpha=0.2)
        expected = np.array([0.2, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sigmoid_range(self):
        x = np.array([-100.0, 0.0, 100.0])
        result = sigmoid(x)
        self.assertAlmostEqual(result[1], 0.5, places=5)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))

    def test_tanh_range(self):
        x = np.array([-100.0, 0.0, 100.0])
        result = tanh(x)
        self.assertAlmostEqual(result[1], 0.0, places=5)
        self.assertTrue(np.all(result >= -1) and np.all(result <= 1))

    def test_tanh_derivative(self):
        x = np.array([0.0])
        result = tanh_derivative(x)
        self.assertAlmostEqual(result[0], 1.0, places=5)


class TestBCELoss(unittest.TestCase):
    """Test binary cross-entropy loss and gradient."""

    def test_perfect_predictions(self):
        preds = np.array([[0.99], [0.99]])
        targets = np.array([[1.0], [1.0]])
        loss = bce_loss(preds, targets)
        self.assertLess(loss, 0.05)

    def test_worst_predictions(self):
        preds = np.array([[0.01], [0.01]])
        targets = np.array([[1.0], [1.0]])
        loss = bce_loss(preds, targets)
        self.assertGreater(loss, 3.0)

    def test_gradient_shape(self):
        preds = np.array([[0.7], [0.3], [0.5]])
        targets = np.array([[1.0], [0.0], [1.0]])
        grad = bce_loss_gradient(preds, targets)
        self.assertEqual(grad.shape, preds.shape)


class TestLayer(unittest.TestCase):
    """Test neural network layer forward and backward passes."""

    def test_forward_shape(self):
        layer = Layer(4, 8, activation='relu')
        x = np.random.randn(16, 4)
        out = layer.forward(x)
        self.assertEqual(out.shape, (16, 8))

    def test_relu_output_nonnegative(self):
        layer = Layer(4, 8, activation='relu')
        x = np.random.randn(16, 4)
        out = layer.forward(x)
        self.assertTrue(np.all(out >= 0))

    def test_sigmoid_output_range(self):
        layer = Layer(4, 1, activation='sigmoid')
        x = np.random.randn(16, 4)
        out = layer.forward(x)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1))

    def test_backward_gradient_shape(self):
        layer = Layer(4, 8, activation='relu')
        x = np.random.randn(16, 4)
        layer.forward(x)
        grad_out = np.random.randn(16, 8)
        grad_in = layer.backward(grad_out)
        self.assertEqual(grad_in.shape, (16, 4))
        self.assertEqual(layer.dW.shape, (4, 8))
        self.assertEqual(layer.db.shape, (1, 8))


class TestGenerator(unittest.TestCase):
    """Test generator network."""

    def test_output_shape(self):
        G = Generator(latent_dim=8, hidden_dim=32, output_dim=2)
        z = np.random.randn(16, 8)
        out = G.forward(z)
        self.assertEqual(out.shape, (16, 2))

    def test_output_bounded(self):
        """Generator uses tanh output, so values should be in [-1, 1]."""
        G = Generator(latent_dim=8, hidden_dim=32, output_dim=2)
        out = G.sample(100)
        self.assertTrue(np.all(out >= -1) and np.all(out <= 1))

    def test_sample_shape(self):
        G = Generator(latent_dim=8, hidden_dim=32, output_dim=2)
        samples = G.sample(50)
        self.assertEqual(samples.shape, (50, 2))


class TestDiscriminator(unittest.TestCase):
    """Test discriminator network."""

    def test_output_shape(self):
        D = Discriminator(input_dim=2, hidden_dim=32)
        x = np.random.randn(16, 2)
        out = D.forward(x)
        self.assertEqual(out.shape, (16, 1))

    def test_output_probability(self):
        """Discriminator uses sigmoid output, so values should be in [0, 1]."""
        D = Discriminator(input_dim=2, hidden_dim=32)
        x = np.random.randn(100, 2)
        out = D.forward(x)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1))


class TestDataGeneration(unittest.TestCase):
    """Test mixture of Gaussians data generation."""

    def test_shape(self):
        data = make_mixture_of_gaussians(100, n_clusters=4)
        self.assertEqual(data.shape, (100, 2))

    def test_reproducibility(self):
        d1 = make_mixture_of_gaussians(50, seed=42)
        d2 = make_mixture_of_gaussians(50, seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_cluster_structure(self):
        """Data should be centered near origin (clusters cancel out)."""
        data = make_mixture_of_gaussians(10000, n_clusters=4, seed=42)
        self.assertAlmostEqual(data[:, 0].mean(), 0.0, places=1)
        self.assertAlmostEqual(data[:, 1].mean(), 0.0, places=1)


class TestTraining(unittest.TestCase):
    """Test that GAN training improves over time."""

    def test_training_reduces_wasserstein(self):
        """After training, Wasserstein distance should decrease."""
        result = train_gan(
            n_epochs=1000, batch_size=128, latent_dim=8,
            hidden_dim=32, lr_g=0.0003, lr_d=0.0001,
            n_clusters=4, seed=42, verbose=False,
        )
        history = result['history']
        # Wasserstein should generally decrease (compare first vs last)
        self.assertLess(history['wasserstein'][-1], history['wasserstein'][0])

    def test_mode_coverage_improves(self):
        """Generator should cover more modes after training than before."""
        result = train_gan(
            n_epochs=1500, batch_size=128, latent_dim=8,
            hidden_dim=32, lr_g=0.0003, lr_d=0.0001,
            n_clusters=4, seed=42, verbose=False,
        )
        # Check final coverage
        G = result['generator']
        fake = G.sample(500)
        covered, total = compute_mode_coverage(fake, result['centers'], threshold=0.2)
        self.assertGreaterEqual(covered, 2, "Generator should cover at least 2 modes")


if __name__ == '__main__':
    unittest.main()
