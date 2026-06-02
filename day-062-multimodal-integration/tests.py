"""
Day 62: Multi-Modal Model Integration — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    softmax, relu, layer_norm, cosine_similarity,
    TextEncoder, ImageEncoder, AudioEncoder,
    EarlyFusion, LateFusion, CrossAttentionFusion,
    MultiModalSystem, contrastive_loss,
)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0, places=6)

    def test_softmax_numerical_stability(self):
        """Large values should not cause overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(np.sum(result), 1.0, places=6)

    def test_relu_positive_and_negative(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_layer_norm_zero_mean_unit_var(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = layer_norm(x)
        self.assertAlmostEqual(np.mean(result), 0.0, places=5)
        self.assertAlmostEqual(np.var(result), 1.0, delta=0.1)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=6)

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_similarity(a, a), 1.0, places=6)

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, places=6)


class TestEncoders(unittest.TestCase):
    """Test individual modality encoders."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_text_encoder_output_shape(self):
        enc = TextEncoder(vocab_size=100, embed_dim=16, hidden_dim=32, output_dim=24)
        tokens = self.rng.randint(0, 100, size=10)
        result = enc.encode(tokens)
        self.assertEqual(result.shape, (24,))

    def test_text_encoder_deterministic(self):
        enc = TextEncoder(vocab_size=50, embed_dim=8, hidden_dim=16, output_dim=12)
        tokens = np.array([1, 5, 10, 3])
        r1 = enc.encode(tokens)
        r2 = enc.encode(tokens)
        np.testing.assert_array_equal(r1, r2)

    def test_image_encoder_output_shape(self):
        enc = ImageEncoder(in_channels=3, num_filters=8, filter_size=3, output_dim=24)
        image = self.rng.randn(3, 8, 8)
        result = enc.encode(image)
        self.assertEqual(result.shape, (24,))

    def test_audio_encoder_output_shape(self):
        enc = AudioEncoder(freq_bins=16, num_filters=8, filter_size=3, output_dim=24)
        spec = self.rng.randn(16, 20)
        result = enc.encode(spec)
        self.assertEqual(result.shape, (24,))

    def test_encoders_produce_normalized_output(self):
        """Layer norm should produce roughly zero-mean output."""
        enc = TextEncoder(100, 16, 32, 32)
        tokens = self.rng.randint(0, 100, size=10)
        result = enc.encode(tokens)
        self.assertAlmostEqual(np.mean(result), 0.0, places=4)


class TestFusion(unittest.TestCase):
    """Test fusion strategies."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.emb1 = self.rng.randn(32)
        self.emb2 = self.rng.randn(32)
        self.emb3 = self.rng.randn(32)

    def test_early_fusion_output_shape(self):
        fusion = EarlyFusion([32, 32, 32], 64, 5)
        result = fusion.fuse([self.emb1, self.emb2, self.emb3])
        self.assertEqual(result.shape, (5,))

    def test_late_fusion_output_shape(self):
        fusion = LateFusion([32, 32, 32], 5)
        result = fusion.fuse([self.emb1, self.emb2, self.emb3])
        self.assertEqual(result.shape, (5,))

    def test_cross_attention_output_shape(self):
        fusion = CrossAttentionFusion(32, 32, 32, 5)
        output, attn_weights = fusion.fuse(self.emb1, self.emb2)
        self.assertEqual(output.shape, (5,))
        self.assertAlmostEqual(np.sum(attn_weights), 1.0, places=5)

    def test_cross_attention_weights_sum_to_one(self):
        """Attention weights must form a valid probability distribution."""
        fusion = CrossAttentionFusion(32, 32, 32, 5, num_kv_tokens=6)
        _, attn_weights = fusion.fuse(self.emb1, self.emb2)
        self.assertAlmostEqual(np.sum(attn_weights), 1.0, places=5)
        self.assertTrue(np.all(attn_weights >= 0))


class TestMultiModalSystem(unittest.TestCase):
    """Test the complete multi-modal system."""

    def setUp(self):
        self.config = {
            "shared_dim": 32, "num_classes": 5, "vocab_size": 100,
            "image_channels": 3, "freq_bins": 16, "modality_dropout_rate": 0.3,
        }
        self.system = MultiModalSystem(self.config)
        self.rng = np.random.RandomState(42)
        self.tokens = self.rng.randint(0, 100, size=10)
        self.image = self.rng.randn(3, 8, 8)
        self.audio = self.rng.randn(16, 20)

    def test_predict_all_modalities(self):
        result = self.system.predict(self.tokens, self.image, self.audio, fusion_strategy="early")
        self.assertIn("logits", result)
        self.assertIn("probabilities", result)
        self.assertIn("predicted_class", result)
        probs = result["probabilities"]
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)
        self.assertTrue(0 <= result["predicted_class"] < 5)

    def test_predict_missing_modality(self):
        """System should handle missing modalities gracefully."""
        result = self.system.predict(self.tokens, None, None, fusion_strategy="early")
        self.assertIn("predicted_class", result)
        self.assertTrue(0 <= result["predicted_class"] < 5)

    def test_predict_all_fusion_strategies(self):
        for strategy in ["early", "late", "cross_attention"]:
            result = self.system.predict(
                self.tokens, self.image, self.audio, fusion_strategy=strategy
            )
            self.assertEqual(result["probabilities"].shape, (5,))


class TestContrastiveLoss(unittest.TestCase):
    """Test CLIP-style contrastive loss."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_loss_is_finite(self):
        text_embs = self.rng.randn(4, 32)
        img_embs = self.rng.randn(4, 32)
        result = contrastive_loss(text_embs, img_embs)
        self.assertTrue(np.isfinite(result["loss"]))

    def test_identical_embeddings_low_loss(self):
        """When text and image embeddings are identical, loss should be low."""
        embs = self.rng.randn(4, 32)
        result_identical = contrastive_loss(embs, embs)
        result_random = contrastive_loss(self.rng.randn(4, 32), self.rng.randn(4, 32))
        self.assertLess(result_identical["loss"], result_random["loss"])

    def test_similarity_matrix_shape(self):
        text_embs = self.rng.randn(6, 32)
        img_embs = self.rng.randn(6, 32)
        result = contrastive_loss(text_embs, img_embs)
        self.assertEqual(result["similarity_matrix"].shape, (6, 6))


if __name__ == "__main__":
    unittest.main()
