"""
Test suite for Day 059: Transformer Attention from Scratch

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    softmax,
    scaled_dot_product_attention,
    MultiHeadAttention,
    LayerNorm,
    FeedForward,
    positional_encoding,
    TransformerEncoderBlock,
    create_causal_mask,
)


class TestSoftmax(unittest.TestCase):
    def test_basic_softmax(self):
        """Softmax should produce valid probability distribution."""
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x)
        self.assertTrue(np.allclose(result.sum(axis=-1), 1.0))
        # Higher input should get higher probability
        self.assertGreater(result[0, 2], result[0, 1])
        self.assertGreater(result[0, 1], result[0, 0])

    def test_numerical_stability(self):
        """Softmax should handle very large values without NaN/Inf."""
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = softmax(x)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        self.assertTrue(np.allclose(result.sum(axis=-1), 1.0))

    def test_softmax_axis(self):
        """Softmax should work along specified axis."""
        x = np.random.randn(3, 4, 5)
        result = softmax(x, axis=-1)
        self.assertTrue(np.allclose(result.sum(axis=-1), 1.0))
        result_ax1 = softmax(x, axis=1)
        self.assertTrue(np.allclose(result_ax1.sum(axis=1), 1.0))


class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.Q = np.random.randn(2, 4, 8)
        self.K = np.random.randn(2, 4, 8)
        self.V = np.random.randn(2, 4, 8)

    def test_output_shape(self):
        """Output should match (batch, seq_len_q, d_v)."""
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(weights.shape, (2, 4, 4))

    def test_weights_sum_to_one(self):
        """Attention weights should sum to 1 across key dimension."""
        _, weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        self.assertTrue(np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6))

    def test_causal_mask_blocks_future(self):
        """With causal mask, tokens should not attend to future positions."""
        mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
        _, weights = scaled_dot_product_attention(self.Q, self.K, self.V, mask=mask)
        # Upper triangle (future positions) should be ~0
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertAlmostEqual(weights[0, i, j], 0.0, places=5)

    def test_scaling_effect(self):
        """Scaling should produce softer distributions than unscaled."""
        Q_big = np.random.randn(1, 4, 512)  # Large d_k
        K_big = np.random.randn(1, 4, 512)
        V_big = np.random.randn(1, 4, 512)
        _, weights = scaled_dot_product_attention(Q_big, K_big, V_big)
        # With proper scaling, weights should not be too peaked
        max_weight = weights.max()
        self.assertLess(max_weight, 0.99, "Weights too peaked — scaling may be missing")


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.d_model = 64
        self.num_heads = 8
        self.mha = MultiHeadAttention(self.d_model, self.num_heads, seed=42)
        self.X = np.random.randn(2, 6, self.d_model)

    def test_output_shape(self):
        """MHA output should match input shape."""
        output, weights = self.mha.forward(self.X, self.X, self.X)
        self.assertEqual(output.shape, (2, 6, self.d_model))
        self.assertEqual(weights.shape, (2, self.num_heads, 6, 6))

    def test_different_heads_different_patterns(self):
        """Different attention heads should produce different weight patterns."""
        _, weights = self.mha.forward(self.X, self.X, self.X)
        head_0 = weights[0, 0]
        head_1 = weights[0, 1]
        # Heads should not be identical
        self.assertFalse(np.allclose(head_0, head_1, atol=1e-3))


class TestLayerNorm(unittest.TestCase):
    def test_output_statistics(self):
        """After LayerNorm, each token should have ~0 mean and ~1 std."""
        np.random.seed(42)
        ln = LayerNorm(64)
        x = np.random.randn(2, 6, 64) * 5 + 3  # Non-zero mean, large variance
        out = ln.forward(x)
        means = out.mean(axis=-1)
        stds = out.std(axis=-1)
        self.assertTrue(np.allclose(means, 0.0, atol=1e-5))
        self.assertTrue(np.allclose(stds, 1.0, atol=0.1))

    def test_shape_preserved(self):
        """LayerNorm should not change the shape."""
        ln = LayerNorm(32)
        x = np.random.randn(3, 4, 32)
        self.assertEqual(ln.forward(x).shape, x.shape)


class TestFeedForward(unittest.TestCase):
    def test_output_shape(self):
        """FFN should preserve input shape."""
        ff = FeedForward(64, 256, seed=42)
        x = np.random.randn(2, 6, 64)
        self.assertEqual(ff.forward(x).shape, (2, 6, 64))

    def test_nonlinearity(self):
        """FFN with ReLU should not be a pure linear transformation."""
        ff = FeedForward(16, 64, seed=42)
        x1 = np.random.randn(1, 4, 16)
        x2 = x1 * 2
        out1 = ff.forward(x1)
        out2 = ff.forward(x2)
        # If linear, out2 would equal 2*out1. With ReLU, it shouldn't.
        self.assertFalse(np.allclose(out2, 2 * out1, atol=1e-3))


class TestPositionalEncoding(unittest.TestCase):
    def test_shape(self):
        pe = positional_encoding(100, 64)
        self.assertEqual(pe.shape, (100, 64))

    def test_different_positions(self):
        """Different positions should have different encodings."""
        pe = positional_encoding(10, 64)
        self.assertFalse(np.allclose(pe[0], pe[1]))
        self.assertFalse(np.allclose(pe[0], pe[9]))

    def test_bounded_values(self):
        """Sinusoidal encoding values should be in [-1, 1]."""
        pe = positional_encoding(100, 128)
        self.assertTrue(np.all(pe >= -1.0 - 1e-8))
        self.assertTrue(np.all(pe <= 1.0 + 1e-8))


class TestTransformerEncoderBlock(unittest.TestCase):
    def test_output_shape(self):
        """Encoder block should preserve input shape."""
        np.random.seed(42)
        encoder = TransformerEncoderBlock(64, 8, 256, seed=42)
        x = np.random.randn(2, 6, 64)
        output, attn = encoder.forward(x)
        self.assertEqual(output.shape, (2, 6, 64))
        self.assertEqual(attn.shape, (2, 8, 6, 6))

    def test_residual_connection(self):
        """Output should be correlated with input due to residual connections."""
        np.random.seed(42)
        encoder = TransformerEncoderBlock(64, 8, 256, seed=42)
        x = np.random.randn(1, 4, 64)
        output, _ = encoder.forward(x)
        # Cosine similarity should be positive (residual preserves information)
        cos_sim = np.mean([
            np.dot(x[0, i], output[0, i])
            / (np.linalg.norm(x[0, i]) * np.linalg.norm(output[0, i]) + 1e-8)
            for i in range(4)
        ])
        self.assertGreater(cos_sim, 0.0, "Residual connection not working")


class TestCausalMask(unittest.TestCase):
    def test_mask_shape(self):
        mask = create_causal_mask(5)
        self.assertEqual(mask.shape, (5, 5))

    def test_mask_structure(self):
        """Diagonal and below should be False, above should be True."""
        mask = create_causal_mask(4)
        # Main diagonal should be False (can attend to self)
        for i in range(4):
            self.assertFalse(mask[i, i])
        # Lower triangle should be False (can attend to past)
        for i in range(4):
            for j in range(i):
                self.assertFalse(mask[i, j])
        # Upper triangle should be True (mask future)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertTrue(mask[i, j])


if __name__ == "__main__":
    unittest.main()
