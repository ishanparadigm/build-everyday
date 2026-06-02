"""
Day 061: Speech-to-Text Pipeline — Your Implementation

Build a complete ASR pipeline from scratch:
  1. Mel spectrogram feature extraction
  2. CTC forward algorithm
  3. Conv + BiGRU + Linear model
  4. Greedy CTC decoder

Fill in each function below. Run `python3 tests.py` to check your work.
"""

import numpy as np
from typing import List, Tuple, Optional


# Character vocabulary: blank + 26 letters + space
BLANK = 0
VOCAB = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz ')
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


# =============================================================================
# Part 1: Audio Feature Extraction
# =============================================================================

def hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to Mel scale.

    Hint: mel = 2595 * log10(1 + hz / 700)
    """
    raise NotImplementedError("TODO: implement this")


def mel_to_hz(mel: float) -> float:
    """Convert Mel scale back to Hz.

    Hint: Inverse of hz_to_mel.
    """
    raise NotImplementedError("TODO: implement this")


def create_mel_filterbank(
    num_filters: int = 40,
    fft_size: int = 512,
    sample_rate: int = 16000,
    low_freq: float = 0.0,
    high_freq: Optional[float] = None
) -> np.ndarray:
    """Create triangular Mel-spaced filterbank.

    Steps:
    1. Convert frequency range to Mel scale
    2. Create num_filters + 2 evenly spaced Mel points
    3. Convert back to Hz, then to FFT bin indices
    4. Build triangular filters between consecutive triplets of bins

    Returns:
        Matrix of shape (num_filters, fft_size // 2 + 1).

    Hint: Each filter is a triangle with peak at center bin,
          rising from left bin, falling to right bin.
    """
    raise NotImplementedError("TODO: implement this")


def compute_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int = 16000,
    frame_size_ms: float = 25.0,
    frame_stride_ms: float = 10.0,
    num_mel_filters: int = 40,
    fft_size: int = 512
) -> np.ndarray:
    """Compute log Mel spectrogram from raw audio.

    Pipeline:
    1. Pre-emphasis (coefficient = 0.97)
    2. Frame into overlapping windows
    3. Apply Hamming window
    4. FFT → power spectrum
    5. Apply Mel filterbank
    6. Log compression

    Returns:
        Log Mel spectrogram of shape (num_frames, num_mel_filters).

    Hint: Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1]
          Hamming: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 2: CTC
# =============================================================================

def ctc_forward(log_probs: np.ndarray, target: List[int]) -> float:
    """Compute CTC loss using the forward algorithm.

    Steps:
    1. Expand target with blanks: [ε, t[0], ε, t[1], ..., ε, t[L-1], ε]
    2. Initialize DP table (alpha) in log space
    3. Fill table: at each (t, s), sum over valid predecessors
    4. Return negative log-likelihood

    Args:
        log_probs: Shape (T, vocab_size) — model output log probabilities.
        target: List of target character indices.

    Returns:
        CTC loss (negative log-likelihood).

    Hint: Valid transitions from state s:
          - Same state s (stay)
          - Previous state s-1 (advance)
          - State s-2 (skip blank, only if expanded[s] != expanded[s-2])
    """
    raise NotImplementedError("TODO: implement this")


def ctc_greedy_decode(log_probs: np.ndarray) -> str:
    """Greedy CTC decoding.

    Steps:
    1. Argmax at each time step
    2. Remove consecutive duplicates
    3. Remove blank tokens
    4. Map to characters

    Args:
        log_probs: Shape (T, vocab_size).

    Returns:
        Decoded text string.

    Hint: Track the previous token. Only emit when the token changes
          AND it's not blank.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 3: Neural Network Components
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Hint: Subtract max before exp to prevent overflow.
    """
    raise NotImplementedError("TODO: implement this")


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log-softmax: log(softmax(x)) but numerically stable.

    Hint: x - log(sum(exp(x - max(x)))) - max(x)? Think carefully about the math.
    """
    raise NotImplementedError("TODO: implement this")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation.

    Hint: Clip inputs to prevent overflow in exp().
    """
    raise NotImplementedError("TODO: implement this")


class GRUCell:
    """Single GRU cell.

    GRU has two gates:
    - Reset gate (r): how much of the past to forget
    - Update gate (z): how much of the new state to accept

    Equations:
        r = sigmoid(x @ W_ir + h @ W_hr + b_r)
        z = sigmoid(x @ W_iz + h @ W_hz + b_z)
        n = tanh(x @ W_in + (r * h) @ W_hn + b_n)
        h_new = (1 - z) * n + z * h
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        self.W_ir = np.random.randn(input_size, hidden_size) * scale
        self.W_hr = np.random.randn(hidden_size, hidden_size) * scale
        self.b_r = np.zeros(hidden_size)

        self.W_iz = np.random.randn(input_size, hidden_size) * scale
        self.W_hz = np.random.randn(hidden_size, hidden_size) * scale
        self.b_z = np.zeros(hidden_size)

        self.W_in = np.random.randn(input_size, hidden_size) * scale
        self.W_hn = np.random.randn(hidden_size, hidden_size) * scale
        self.b_n = np.zeros(hidden_size)

    def forward(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Process one time step.

        Args:
            x: Input, shape (input_size,).
            h: Previous hidden state, shape (hidden_size,).

        Returns:
            New hidden state, shape (hidden_size,).
        """
        raise NotImplementedError("TODO: implement this")


class BiGRU:
    """Bidirectional GRU — processes sequence forwards and backwards.

    Hint: Run forward GRU left-to-right, backward GRU right-to-left,
          then concatenate outputs at each time step.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.forward_cell = GRUCell(input_size, hidden_size)
        self.backward_cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Process sequence bidirectionally.

        Args:
            x: Shape (seq_len, input_size).

        Returns:
            Shape (seq_len, 2 * hidden_size).
        """
        raise NotImplementedError("TODO: implement this")


class ASRModel:
    """ASR model: Conv1D → ReLU → BiGRU → Linear → Log-Softmax.

    Hint: The conv layer treats mel bins as input channels and slides over time.
          Reshape mel_spec from (frames, mel_bins) to (1, mel_bins, frames) for conv.
    """

    def __init__(
        self,
        num_mel_filters: int = 40,
        conv_channels: int = 32,
        conv_kernel: int = 3,
        conv_stride: int = 2,
        hidden_size: int = 64,
        vocab_size: int = VOCAB_SIZE
    ):
        # You'll need to implement Conv1D or use the one from solution
        # For now, focus on the BiGRU → Linear → Log-Softmax part
        raise NotImplementedError("TODO: implement this")

    def forward(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Forward pass: mel spectrogram → log probabilities.

        Args:
            mel_spectrogram: Shape (num_frames, num_mel_filters).

        Returns:
            Log probabilities, shape (output_time_steps, vocab_size).
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 4: Pipeline
# =============================================================================

class SpeechToTextPipeline:
    """Complete pipeline: audio → mel spectrogram → model → CTC decode → text."""

    def __init__(self, sample_rate: int = 16000, num_mel_filters: int = 40, hidden_size: int = 64):
        raise NotImplementedError("TODO: implement this")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        raise NotImplementedError("TODO: implement this")


if __name__ == '__main__':
    print("Speech-to-Text Pipeline — Testing Your Implementation")
    print("=" * 55)

    # Test 1: Mel scale conversion
    print("\n[Test 1] Mel scale conversion:")
    try:
        mel_val = hz_to_mel(1000.0)
        hz_val = mel_to_hz(mel_val)
        print(f"  1000 Hz → {mel_val:.1f} Mel → {hz_val:.1f} Hz (should be ~1000)")
    except NotImplementedError:
        print("  Not implemented yet")

    # Test 2: Mel spectrogram
    print("\n[Test 2] Mel spectrogram:")
    try:
        t = np.linspace(0, 1, 16000)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        mel_spec = compute_mel_spectrogram(audio)
        print(f"  Shape: {mel_spec.shape} (expected ~(98, 40))")
    except NotImplementedError:
        print("  Not implemented yet")

    # Test 3: CTC decoding
    print("\n[Test 3] CTC greedy decode:")
    try:
        T = 10
        log_probs = np.full((T, VOCAB_SIZE), -10.0)
        # Spell "ab": blank, a, a, blank, b, blank...
        tokens = [0, 1, 1, 0, 2, 0, 0, 0, 0, 0]
        for t, tok in enumerate(tokens):
            log_probs[t, tok] = 0.0
        log_probs_norm = log_softmax(log_probs) if 'log_softmax' in dir() else log_probs
        text = ctc_greedy_decode(log_probs_norm)
        print(f"  Decoded: '{text}' (expected 'ab')")
    except NotImplementedError:
        print("  Not implemented yet")

    # Test 4: CTC loss
    print("\n[Test 4] CTC forward loss:")
    try:
        T = 6
        log_probs = np.full((T, VOCAB_SIZE), -10.0)
        for t in [0, 2, 4, 5]:
            log_probs[t, BLANK] = 0.0
        log_probs[1, CHAR_TO_IDX['a']] = 0.0
        log_probs[3, CHAR_TO_IDX['b']] = 0.0
        log_probs = log_softmax(log_probs) if 'log_softmax' in dir() else log_probs
        loss = ctc_forward(log_probs, [CHAR_TO_IDX['a'], CHAR_TO_IDX['b']])
        print(f"  CTC loss: {loss:.4f} (should be small for good alignment)")
    except NotImplementedError:
        print("  Not implemented yet")

    print("\nRun `python3 tests.py` for the full test suite.")
