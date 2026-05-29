"""
Day 049: Fine-Tuning Sentiment Classifier — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    SimpleTokenizer,
    PretrainedEmbeddings,
    AttentionPooling,
    SentimentClassifier,
    softmax,
    layer_norm,
    gelu,
    gelu_derivative,
    compute_loss,
    evaluate,
    create_sentiment_dataset,
    clip_gradients,
)


class TestSoftmax(unittest.TestCase):
    """Test the softmax function."""

    def test_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x)
        np.testing.assert_almost_equal(np.sum(result), 1.0, decimal=6)

    def test_uniform(self):
        x = np.array([[0.0, 0.0]])
        result = softmax(x)
        np.testing.assert_array_almost_equal(result, [[0.5, 0.5]], decimal=6)

    def test_numerical_stability(self):
        """Large values should not cause overflow."""
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = softmax(x)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_almost_equal(np.sum(result), 1.0, decimal=6)


class TestLayerNorm(unittest.TestCase):
    """Test layer normalization."""

    def test_zero_mean_unit_variance(self):
        x = np.random.randn(2, 10)
        gamma = np.ones(10)
        beta = np.zeros(10)
        result = layer_norm(x, gamma, beta)
        for i in range(2):
            self.assertAlmostEqual(float(np.mean(result[i])), 0.0, places=5)
            self.assertAlmostEqual(float(np.var(result[i])), 1.0, places=4)

    def test_scale_and_shift(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.ones(4) * 2.0
        beta = np.ones(4) * 3.0
        result = layer_norm(x, gamma, beta)
        self.assertAlmostEqual(float(np.mean(result)), 3.0, places=4)


class TestGelu(unittest.TestCase):
    """Test GELU activation."""

    def test_zero_input(self):
        result = gelu(np.array([0.0]))
        self.assertAlmostEqual(float(result[0]), 0.0, places=5)

    def test_positive_input(self):
        result = gelu(np.array([2.0]))
        self.assertAlmostEqual(float(result[0]), 1.9545, places=2)

    def test_negative_input(self):
        result = gelu(np.array([-2.0]))
        self.assertTrue(result[0] < 0 and result[0] > -0.1)

    def test_derivative_at_zero(self):
        result = gelu_derivative(np.array([0.0]))
        self.assertAlmostEqual(float(result[0]), 0.5, places=2)


class TestTokenizer(unittest.TestCase):
    """Test the SimpleTokenizer."""

    def setUp(self):
        self.tokenizer = SimpleTokenizer(max_vocab_size=100, max_length=10)
        self.tokenizer.build_vocab([
            "the cat sat on the mat",
            "the dog ran in the park",
            "a good day for a walk",
        ])

    def test_special_tokens_reserved(self):
        self.assertEqual(self.tokenizer.token_to_id["[PAD]"], 0)
        self.assertEqual(self.tokenizer.token_to_id["[CLS]"], 1)
        self.assertEqual(self.tokenizer.token_to_id["[SEP]"], 2)
        self.assertEqual(self.tokenizer.token_to_id["[UNK]"], 3)

    def test_encode_length(self):
        encoded = self.tokenizer.encode("the cat sat")
        self.assertEqual(len(encoded), 10)  # max_length

    def test_encode_starts_with_cls(self):
        encoded = self.tokenizer.encode("hello world")
        self.assertEqual(encoded[0], self.tokenizer.token_to_id["[CLS]"])

    def test_encode_has_sep(self):
        encoded = self.tokenizer.encode("the cat")
        sep_id = self.tokenizer.token_to_id["[SEP]"]
        self.assertIn(sep_id, encoded)

    def test_unknown_words_get_unk(self):
        encoded = self.tokenizer.encode("xyzzy flurbo")
        unk_id = self.tokenizer.token_to_id["[UNK]"]
        self.assertIn(unk_id, encoded)

    def test_decode_roundtrip(self):
        text = "the cat sat"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertIn("cat", decoded)
        self.assertIn("sat", decoded)


class TestPretrainedEmbeddings(unittest.TestCase):
    """Test pre-trained embedding initialization."""

    def test_embedding_shape(self):
        emb = PretrainedEmbeddings(vocab_size=100, embedding_dim=32)
        self.assertEqual(emb.embeddings.shape, (100, 32))

    def test_pad_token_is_zero(self):
        emb = PretrainedEmbeddings(vocab_size=100, embedding_dim=32)
        np.testing.assert_array_equal(emb.embeddings[0], np.zeros(32))

    def test_get_embeddings_shape(self):
        emb = PretrainedEmbeddings(vocab_size=100, embedding_dim=32)
        token_ids = np.array([[1, 2, 3], [4, 5, 6]])
        result = emb.get_embeddings(token_ids)
        self.assertEqual(result.shape, (2, 3, 32))


class TestAttentionPooling(unittest.TestCase):
    """Test the attention pooling layer."""

    def test_output_shape(self):
        pool = AttentionPooling(embedding_dim=32)
        x = np.random.randn(2, 8, 32)
        mask = np.ones((2, 8))
        pooled, weights = pool.forward(x, mask)
        self.assertEqual(pooled.shape, (2, 32))
        self.assertEqual(weights.shape, (2, 8))

    def test_weights_sum_to_one(self):
        pool = AttentionPooling(embedding_dim=32)
        x = np.random.randn(1, 5, 32)
        mask = np.ones((1, 5))
        _, weights = pool.forward(x, mask)
        np.testing.assert_almost_equal(float(np.sum(weights)), 1.0, decimal=5)

    def test_masked_tokens_get_zero_weight(self):
        pool = AttentionPooling(embedding_dim=16)
        x = np.random.randn(1, 6, 16)
        mask = np.array([[1, 1, 1, 0, 0, 0]], dtype=np.float64)
        _, weights = pool.forward(x, mask)
        # Masked positions should have ~0 weight
        self.assertAlmostEqual(float(np.sum(weights[0, 3:])), 0.0, places=5)


class TestSentimentClassifier(unittest.TestCase):
    """Test the full sentiment classifier."""

    def setUp(self):
        self.tokenizer = SimpleTokenizer(max_vocab_size=500, max_length=16)
        texts = [
            "good great amazing love best wonderful",
            "bad terrible awful hate worst horrible",
            "the movie was really quite good overall",
            "a terrible waste of my precious time",
        ]
        self.tokenizer.build_vocab(texts)
        self.model = SentimentClassifier(
            tokenizer=self.tokenizer,
            embedding_dim=32,
            hidden_dim=16,
            n_classes=2,
        )

    def test_forward_output_shapes(self):
        token_ids = np.array([self.tokenizer.encode("good movie")])
        logits, pooled = self.model.forward(token_ids)
        self.assertEqual(logits.shape, (1, 2))
        self.assertEqual(pooled.shape, (1, 32))

    def test_predict_probabilities_sum_to_one(self):
        token_ids = np.array([self.tokenizer.encode("great film")])
        probs = self.model.predict(token_ids)
        np.testing.assert_almost_equal(float(np.sum(probs)), 1.0, decimal=5)

    def test_predict_text_returns_label(self):
        label, confidence = self.model.predict_text("this is good")
        self.assertIn(label, ["positive", "negative"])
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_backward_returns_gradients(self):
        token_ids = np.array([self.tokenizer.encode("good movie")])
        self.model.forward(token_ids)
        grads = self.model.backward(np.array([1]))
        self.assertIn('W_out', grads)
        self.assertIn('W_hidden', grads)
        self.assertIn('embeddings', grads)


class TestComputeLoss(unittest.TestCase):
    """Test the cross-entropy loss function."""

    def test_perfect_prediction(self):
        logits = np.array([[100.0, -100.0]])
        labels = np.array([0])
        loss = compute_loss(logits, labels)
        self.assertAlmostEqual(loss, 0.0, places=3)

    def test_wrong_prediction(self):
        logits = np.array([[-100.0, 100.0]])
        labels = np.array([0])
        loss = compute_loss(logits, labels)
        self.assertGreater(loss, 10.0)

    def test_uniform_prediction(self):
        logits = np.array([[0.0, 0.0]])
        labels = np.array([0])
        loss = compute_loss(logits, labels)
        self.assertAlmostEqual(loss, 0.693, places=2)


class TestGradientClipping(unittest.TestCase):
    """Test gradient clipping."""

    def test_no_clipping_needed(self):
        grads = {'a': np.array([1.0, 0.0]), 'b': np.array([0.0, 1.0])}
        clipped = clip_gradients(grads, max_norm=5.0)
        np.testing.assert_array_almost_equal(clipped['a'], grads['a'])

    def test_clipping_applied(self):
        grads = {'a': np.array([100.0, 0.0])}
        clipped = clip_gradients(grads, max_norm=5.0)
        total_norm = np.sqrt(np.sum(clipped['a'] ** 2))
        self.assertAlmostEqual(total_norm, 5.0, places=3)


class TestDataset(unittest.TestCase):
    """Test dataset creation."""

    def test_split_sizes(self):
        train, val, test = create_sentiment_dataset()
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

    def test_labels_are_binary(self):
        train, val, test = create_sentiment_dataset()
        all_data = train + val + test
        labels = set(label for _, label in all_data)
        self.assertEqual(labels, {0, 1})


if __name__ == "__main__":
    unittest.main()
