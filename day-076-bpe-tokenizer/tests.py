"""
Tests for BPE Tokenizer implementation.

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import unittest
from my_solution import BPETokenizer, get_training_corpus


class TestBPETokenizer(unittest.TestCase):
    """Comprehensive tests for the BPE tokenizer."""

    @classmethod
    def setUpClass(cls):
        """Train a tokenizer once for use across tests."""
        cls.corpus = get_training_corpus()
        cls.tokenizer = BPETokenizer(vocab_size=350)
        cls.tokenizer.train(cls.corpus)

    def test_base_vocabulary_exists(self):
        """All 256 byte values should be in the vocabulary."""
        for i in range(256):
            self.assertIn(i, self.tokenizer.vocab)
            self.assertEqual(self.tokenizer.vocab[i], bytes([i]))

    def test_vocab_size(self):
        """Vocabulary should not exceed the target size."""
        self.assertLessEqual(len(self.tokenizer.vocab), self.tokenizer.vocab_size)
        # Should have some merged tokens beyond base 256
        self.assertGreater(len(self.tokenizer.vocab), 256)

    def test_merges_recorded(self):
        """Training should produce merge rules."""
        self.assertGreater(len(self.tokenizer.merges), 0)
        # Each merge should have a pair and a new ID >= 256
        for pair, new_id in self.tokenizer.merges:
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
            self.assertGreaterEqual(new_id, 256)

    def test_roundtrip_simple(self):
        """Encode then decode should recover the original string."""
        text = "hello world"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_roundtrip_training_corpus(self):
        """Encode then decode the entire training corpus should be lossless."""
        encoded = self.tokenizer.encode(self.corpus)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, self.corpus)

    def test_roundtrip_unseen_text(self):
        """Tokenizer should handle text not in the training corpus."""
        text = "supercalifragilisticexpialidocious"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_empty_string(self):
        """Empty string should encode to empty list and decode back."""
        encoded = self.tokenizer.encode("")
        self.assertEqual(encoded, [])
        decoded = self.tokenizer.decode([])
        self.assertEqual(decoded, "")

    def test_single_character(self):
        """Single ASCII character should encode to a single byte token."""
        encoded = self.tokenizer.encode("a")
        self.assertEqual(len(encoded), 1)
        self.assertEqual(encoded[0], ord("a"))

    def test_compression_ratio(self):
        """Compression ratio should be > 1.0 for the training corpus (BPE compresses)."""
        ratio = self.tokenizer.compression_ratio(self.corpus)
        self.assertGreater(ratio, 1.0)

    def test_encode_produces_fewer_tokens_than_bytes(self):
        """For text with common substrings, encoding should compress vs raw bytes."""
        text = "the the the the the"
        encoded = self.tokenizer.encode(text)
        byte_length = len(text.encode("utf-8"))
        self.assertLess(len(encoded), byte_length)

    def test_merge_order_matters(self):
        """Two tokenizers trained on same data should produce same merges."""
        tok1 = BPETokenizer(vocab_size=280)
        tok2 = BPETokenizer(vocab_size=280)
        tok1.train(self.corpus)
        tok2.train(self.corpus)
        # Same training data -> same merge rules
        self.assertEqual(len(tok1.merges), len(tok2.merges))
        for m1, m2 in zip(tok1.merges, tok2.merges):
            self.assertEqual(m1, m2)

    def test_larger_vocab_means_fewer_tokens(self):
        """A larger vocabulary should generally produce fewer tokens for the same text."""
        text = "The transformer architecture revolutionized natural language processing"
        small_tok = BPETokenizer(vocab_size=280)
        large_tok = BPETokenizer(vocab_size=400)
        small_tok.train(self.corpus)
        large_tok.train(self.corpus)
        small_encoded = small_tok.encode(text)
        large_encoded = large_tok.encode(text)
        self.assertGreaterEqual(len(small_encoded), len(large_encoded))


if __name__ == "__main__":
    unittest.main()
