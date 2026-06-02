"""
Tests for Day 061: Speech-to-Text Pipeline

Run with: python3 -m pytest tests.py -v
     or: python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    hz_to_mel, mel_to_hz, create_mel_filterbank, compute_mel_spectrogram,
    ctc_forward, ctc_greedy_decode, softmax, log_softmax, sigmoid,
    GRUCell, BiGRU, BLANK, VOCAB, VOCAB_SIZE, CHAR_TO_IDX,
)


class TestMelScale(unittest.TestCase):
    """Test Mel scale conversions."""

    def test_hz_to_mel_known_values(self):
        """1000 Hz should map to ~1127 Mel."""
        mel = hz_to_mel(1000.0)
        self.assertAlmostEqual(mel, 1127.01, places=0)

    def test_hz_to_mel_zero(self):
        """0 Hz should map to 0 Mel."""
        self.assertAlmostEqual(hz_to_mel(0.0), 0.0, places=5)

    def test_roundtrip(self):
        """Converting Hz → Mel → Hz should return the original value."""
        for hz in [100, 500, 1000, 4000, 8000]:
            result = mel_to_hz(hz_to_mel(hz))
            self.assertAlmostEqual(result, hz, places=2,
                                   msg=f"Roundtrip failed for {hz} Hz")

    def test_mel_is_monotonic(self):
        """Higher Hz should give higher Mel values."""
        freqs = [100, 500, 1000, 2000, 4000]
        mels = [hz_to_mel(f) for f in freqs]
        for i in range(len(mels) - 1):
            self.assertGreater(mels[i + 1], mels[i])


class TestMelFilterbank(unittest.TestCase):
    """Test Mel filterbank construction."""

    def test_shape(self):
        """Filterbank should have correct dimensions."""
        fb = create_mel_filterbank(num_filters=40, fft_size=512)
        self.assertEqual(fb.shape, (40, 257))  # 512//2 + 1 = 257

    def test_non_negative(self):
        """All filter values should be non-negative (triangular filters)."""
        fb = create_mel_filterbank(num_filters=20, fft_size=256)
        self.assertTrue(np.all(fb >= 0))

    def test_filters_have_nonzero_values(self):
        """Each filter should have at least some non-zero values."""
        fb = create_mel_filterbank(num_filters=20, fft_size=512)
        for i in range(20):
            self.assertGreater(np.sum(fb[i] > 0), 0,
                               msg=f"Filter {i} is all zeros")


class TestMelSpectrogram(unittest.TestCase):
    """Test Mel spectrogram computation."""

    def test_output_shape(self):
        """Check output dimensions are reasonable."""
        # 1 second at 16kHz with 25ms frames and 10ms stride
        # Expected frames: 1 + (16000 - 400) // 160 = ~98
        signal = np.random.randn(16000).astype(np.float32)
        mel_spec = compute_mel_spectrogram(signal, num_mel_filters=40)
        self.assertEqual(mel_spec.shape[1], 40)
        self.assertGreater(mel_spec.shape[0], 50)
        self.assertLess(mel_spec.shape[0], 200)

    def test_output_is_finite(self):
        """No NaN or Inf in output."""
        signal = np.random.randn(8000).astype(np.float32)
        mel_spec = compute_mel_spectrogram(signal)
        self.assertTrue(np.all(np.isfinite(mel_spec)))

    def test_different_signals_different_spectrograms(self):
        """Different audio inputs should produce different spectrograms."""
        t = np.linspace(0, 1, 16000)
        sig1 = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        sig2 = np.sin(2 * np.pi * 4000 * t).astype(np.float32)
        mel1 = compute_mel_spectrogram(sig1)
        mel2 = compute_mel_spectrogram(sig2)
        # Different frequencies should produce different energy patterns
        self.assertFalse(np.allclose(mel1, mel2, atol=0.1))


class TestCTCDecode(unittest.TestCase):
    """Test CTC greedy decoding."""

    def _make_log_probs(self, token_sequence):
        """Helper: create log_probs that predict the given token sequence."""
        T = len(token_sequence)
        logits = np.full((T, VOCAB_SIZE), -10.0)
        for t, tok in enumerate(token_sequence):
            logits[t, tok] = 0.0
        # Normalize
        x_max = np.max(logits, axis=-1, keepdims=True)
        log_sum = x_max + np.log(np.sum(np.exp(logits - x_max), axis=-1, keepdims=True))
        return logits - log_sum

    def test_simple_decode(self):
        """ε-a-a-ε-b-ε should decode to 'ab'."""
        tokens = [0, 1, 1, 0, 2, 0]
        log_probs = self._make_log_probs(tokens)
        self.assertEqual(ctc_greedy_decode(log_probs), 'ab')

    def test_hello(self):
        """Should decode 'hello' with blanks and repeats."""
        # h=8, e=5, l=12, o=15
        tokens = [0, 8, 0, 5, 0, 12, 0, 12, 0, 15, 0]
        log_probs = self._make_log_probs(tokens)
        self.assertEqual(ctc_greedy_decode(log_probs), 'hello')

    def test_all_blanks(self):
        """All blanks should decode to empty string."""
        tokens = [0, 0, 0, 0, 0]
        log_probs = self._make_log_probs(tokens)
        self.assertEqual(ctc_greedy_decode(log_probs), '')

    def test_consecutive_same_chars_need_blank(self):
        """'aa' requires a blank separator: a-ε-a."""
        # Without blank: a-a collapses to 'a'
        tokens_no_blank = [1, 1]
        log_probs = self._make_log_probs(tokens_no_blank)
        self.assertEqual(ctc_greedy_decode(log_probs), 'a')

        # With blank: a-ε-a gives 'aa'
        tokens_with_blank = [1, 0, 1]
        log_probs = self._make_log_probs(tokens_with_blank)
        self.assertEqual(ctc_greedy_decode(log_probs), 'aa')


class TestCTCForward(unittest.TestCase):
    """Test CTC forward algorithm (loss computation)."""

    def test_perfect_alignment_low_loss(self):
        """Loss should be low when model output matches target well."""
        T = 8
        target = [CHAR_TO_IDX['h'], CHAR_TO_IDX['i']]

        logits = np.full((T, VOCAB_SIZE), -10.0)
        # Good alignment: ε-ε-h-h-ε-i-i-ε
        for t in [0, 1, 4, 7]:
            logits[t, BLANK] = 0.0
        for t in [2, 3]:
            logits[t, CHAR_TO_IDX['h']] = 0.0
        for t in [5, 6]:
            logits[t, CHAR_TO_IDX['i']] = 0.0

        x_max = np.max(logits, axis=-1, keepdims=True)
        log_sum = x_max + np.log(np.sum(np.exp(logits - x_max), axis=-1, keepdims=True))
        log_probs = logits - log_sum

        loss = ctc_forward(log_probs, target)
        self.assertLess(loss, 1.0, "Loss should be small for good alignment")

    def test_random_output_high_loss(self):
        """Random model output should give high CTC loss."""
        np.random.seed(42)
        T = 20
        target = [CHAR_TO_IDX['a'], CHAR_TO_IDX['b'], CHAR_TO_IDX['c']]

        logits = np.random.randn(T, VOCAB_SIZE)
        x_max = np.max(logits, axis=-1, keepdims=True)
        log_sum = x_max + np.log(np.sum(np.exp(logits - x_max), axis=-1, keepdims=True))
        log_probs = logits - log_sum

        loss = ctc_forward(log_probs, target)
        self.assertGreater(loss, 1.0, "Loss should be high for random output")


class TestActivations(unittest.TestCase):
    """Test activation functions."""

    def test_softmax_sums_to_one(self):
        x = np.random.randn(5, 10)
        s = softmax(x, axis=-1)
        np.testing.assert_allclose(np.sum(s, axis=-1), np.ones(5), atol=1e-6)

    def test_softmax_non_negative(self):
        x = np.random.randn(3, 8)
        self.assertTrue(np.all(softmax(x) >= 0))

    def test_log_softmax_consistency(self):
        """log_softmax should equal log(softmax) for reasonable inputs."""
        x = np.random.randn(4, 6)
        ls = log_softmax(x)
        expected = np.log(softmax(x) + 1e-30)
        np.testing.assert_allclose(ls, expected, atol=1e-5)

    def test_sigmoid_range(self):
        x = np.array([-100, -1, 0, 1, 100])
        s = sigmoid(x)
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))
        self.assertAlmostEqual(float(s[2]), 0.5, places=5)


class TestGRU(unittest.TestCase):
    """Test GRU cell and BiGRU."""

    def test_gru_cell_output_shape(self):
        np.random.seed(0)
        cell = GRUCell(input_size=10, hidden_size=20)
        x = np.random.randn(10)
        h = np.zeros(20)
        h_new = cell.forward(x, h)
        self.assertEqual(h_new.shape, (20,))

    def test_gru_cell_output_finite(self):
        np.random.seed(0)
        cell = GRUCell(input_size=8, hidden_size=16)
        x = np.random.randn(8)
        h = np.random.randn(16)
        h_new = cell.forward(x, h)
        self.assertTrue(np.all(np.isfinite(h_new)))

    def test_bigru_output_shape(self):
        np.random.seed(0)
        bigru = BiGRU(input_size=10, hidden_size=20)
        x = np.random.randn(15, 10)  # 15 time steps
        out = bigru.forward(x)
        self.assertEqual(out.shape, (15, 40))  # 2 * hidden_size


if __name__ == '__main__':
    unittest.main()
