"""
Day 55: CNN Image Classifier — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    conv2d_forward, conv2d_backward,
    relu_forward, relu_backward,
    maxpool2d_forward, maxpool2d_backward,
    softmax, cross_entropy_loss,
    CNN, generate_digit_patterns
)


class TestConv2D(unittest.TestCase):
    """Test convolution forward and backward passes."""

    def test_output_shape(self):
        """Conv with 4 filters on 1×8×8 input should produce 4×6×6 output."""
        input = np.random.randn(1, 8, 8)
        kernels = np.random.randn(4, 1, 3, 3)
        biases = np.zeros(4)
        output = conv2d_forward(input, kernels, biases)
        self.assertEqual(output.shape, (4, 6, 6))

    def test_single_filter_known_values(self):
        """Identity-like filter should reproduce center of input."""
        # A kernel that picks out just the center pixel
        input = np.ones((1, 5, 5))
        kernels = np.zeros((1, 1, 3, 3))
        kernels[0, 0, 1, 1] = 1.0  # Center only
        biases = np.zeros(1)
        output = conv2d_forward(input, kernels, biases)
        np.testing.assert_allclose(output, np.ones((1, 3, 3)))

    def test_bias_effect(self):
        """Bias should shift all output values by the bias amount."""
        input = np.zeros((1, 5, 5))
        kernels = np.zeros((2, 1, 3, 3))
        biases = np.array([3.0, -1.0])
        output = conv2d_forward(input, kernels, biases)
        np.testing.assert_allclose(output[0], 3.0 * np.ones((3, 3)))
        np.testing.assert_allclose(output[1], -1.0 * np.ones((3, 3)))

    def test_backward_kernel_gradient_shape(self):
        """Backward pass should produce gradients matching parameter shapes."""
        input = np.random.randn(1, 8, 8)
        kernels = np.random.randn(4, 1, 3, 3)
        d_output = np.random.randn(4, 6, 6)
        d_input, d_kernels, d_biases = conv2d_backward(input, kernels, d_output)
        self.assertEqual(d_input.shape, input.shape)
        self.assertEqual(d_kernels.shape, kernels.shape)
        self.assertEqual(d_biases.shape, (4,))


class TestReLU(unittest.TestCase):
    """Test ReLU activation."""

    def test_positive_passthrough(self):
        """Positive values should pass through unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu_forward(x), x)

    def test_negative_zeroed(self):
        """Negative values should become zero."""
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        expected = np.array([0.0, 0.0, 0.0, 0.5, 1.0])
        np.testing.assert_array_equal(relu_forward(x), expected)

    def test_backward_gradient_mask(self):
        """Gradient should be zero where input was negative."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        d_output = np.ones(5)
        d_input = relu_backward(d_output, x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(d_input, expected)


class TestMaxPool(unittest.TestCase):
    """Test max pooling forward and backward."""

    def test_output_shape(self):
        """2×2 pooling on 2×4×4 should give 2×2×2."""
        input = np.random.randn(2, 4, 4)
        output, _ = maxpool2d_forward(input, pool_size=2)
        self.assertEqual(output.shape, (2, 2, 2))

    def test_picks_max(self):
        """Should select the maximum value in each 2×2 window."""
        input = np.array([[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]], dtype=float)
        output, _ = maxpool2d_forward(input, pool_size=2)
        expected = np.array([[[6, 8], [14, 16]]], dtype=float)
        np.testing.assert_array_equal(output, expected)

    def test_backward_routes_to_max(self):
        """Gradient should only flow to the position that held the max."""
        input = np.array([[[1, 2], [3, 4]]], dtype=float)  # 1×2×2
        output, max_indices = maxpool2d_forward(input, pool_size=2)
        d_output = np.array([[[5.0]]])  # 1×1×1
        d_input = maxpool2d_backward(d_output, max_indices, input.shape, pool_size=2)
        # Max was at position (1,1) = value 4
        expected = np.array([[[0, 0], [0, 5.0]]])
        np.testing.assert_array_equal(d_input, expected)


class TestSoftmax(unittest.TestCase):
    """Test softmax output properties."""

    def test_sums_to_one(self):
        """Softmax output should sum to 1."""
        logits = np.array([2.0, 1.0, 0.1])
        probs = softmax(logits)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

    def test_largest_logit_gets_highest_prob(self):
        """The largest logit should get the highest probability."""
        logits = np.array([1.0, 5.0, 2.0])
        probs = softmax(logits)
        self.assertEqual(np.argmax(probs), 1)

    def test_numerical_stability(self):
        """Should handle large logits without overflow."""
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = softmax(logits)
        self.assertFalse(np.any(np.isnan(probs)))
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)


class TestCrossEntropy(unittest.TestCase):
    """Test cross-entropy loss."""

    def test_perfect_prediction(self):
        """Loss should be near zero for a perfect prediction."""
        preds = np.array([0.0, 0.0, 1.0, 0.0])
        loss = cross_entropy_loss(preds, 2)
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_wrong_prediction_high_loss(self):
        """Loss should be high when the correct class has low probability."""
        preds = np.array([0.9, 0.05, 0.03, 0.02])
        loss = cross_entropy_loss(preds, 2)
        self.assertGreater(loss, 2.0)


class TestCNNEndToEnd(unittest.TestCase):
    """Test the full CNN pipeline."""

    def test_forward_output_shape(self):
        """CNN forward should produce 10 class probabilities."""
        cnn = CNN(num_filters=8, num_classes=10)
        image = np.random.randn(1, 28, 28)
        probs = cnn.forward(image)
        self.assertEqual(probs.shape, (10,))
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)

    def test_backward_produces_gradients(self):
        """Backward should return gradients for all parameters."""
        cnn = CNN(num_filters=8, num_classes=10)
        image = np.random.randn(1, 28, 28)
        cnn.forward(image)
        grads = cnn.backward(3)
        self.assertIn('conv_kernels', grads)
        self.assertIn('fc_weights', grads)
        self.assertEqual(grads['conv_kernels'].shape, (8, 1, 3, 3))

    def test_training_reduces_loss(self):
        """A few training steps should reduce the loss."""
        np.random.seed(42)
        cnn = CNN(num_filters=8, num_classes=10)
        images, labels = generate_digit_patterns(num_samples_per_class=5, seed=42)

        # Compute initial loss
        initial_loss = 0
        for i in range(10):
            probs = cnn.forward(images[i])
            initial_loss += cross_entropy_loss(probs, labels[i])

        # Train for 10 steps
        for i in range(10):
            probs = cnn.forward(images[i])
            grads = cnn.backward(labels[i])
            cnn.update(grads, lr=0.005)

        # Compute final loss on same samples
        final_loss = 0
        for i in range(10):
            probs = cnn.forward(images[i])
            final_loss += cross_entropy_loss(probs, labels[i])

        self.assertLess(final_loss, initial_loss)


if __name__ == '__main__':
    unittest.main()
