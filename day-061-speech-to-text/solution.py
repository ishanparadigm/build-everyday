"""
Day 061: Speech-to-Text Pipeline

A complete ASR pipeline from raw audio waveforms to text transcriptions,
implemented from scratch using NumPy. Covers:
  1. Mel spectrogram feature extraction
  2. CTC forward algorithm for loss computation
  3. Conv + BiGRU + Linear model architecture
  4. Greedy CTC decoding

No external ML frameworks — pure NumPy to understand every component.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# =============================================================================
# Part 1: Audio Feature Extraction — Mel Spectrograms
# =============================================================================

def generate_synthetic_audio(
    duration: float = 1.0,
    sample_rate: int = 16000,
    frequencies: Optional[List[float]] = None,
    noise_level: float = 0.01
) -> np.ndarray:
    """Generate synthetic audio by combining sine waves.

    Real speech is a complex mixture of harmonics. We simulate this with
    multiple sine waves at different frequencies, which creates a signal
    with interesting spectral content for our pipeline to process.

    Args:
        duration: Length in seconds.
        sample_rate: Samples per second (16kHz is standard for speech).
        frequencies: List of frequencies to combine. If None, uses speech-like freqs.
        noise_level: Standard deviation of additive Gaussian noise.

    Returns:
        1D numpy array of audio samples in [-1, 1] range.
    """
    if frequencies is None:
        # Typical speech fundamental + harmonics (vowel-like sound)
        frequencies = [150.0, 300.0, 600.0, 1200.0, 2400.0]

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Sum sine waves with decreasing amplitude (higher harmonics are quieter)
    signal = np.zeros_like(t)
    for i, freq in enumerate(frequencies):
        amplitude = 1.0 / (i + 1)  # Harmonic series decay
        signal += amplitude * np.sin(2 * np.pi * freq * t)

    # Normalize to [-1, 1] and add noise
    signal = signal / np.max(np.abs(signal))
    signal += noise_level * np.random.randn(len(signal))

    return signal.astype(np.float32)


def hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to Mel scale.

    The Mel scale is a perceptual scale — equal distances in Mel correspond
    to equal perceived pitch differences. The formula is empirically derived
    from psychoacoustic experiments.
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert Mel scale back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    num_filters: int = 40,
    fft_size: int = 512,
    sample_rate: int = 16000,
    low_freq: float = 0.0,
    high_freq: Optional[float] = None
) -> np.ndarray:
    """Create a bank of triangular Mel-spaced filters.

    Each filter is a triangle centered on a Mel-spaced frequency. Adjacent
    filters overlap, ensuring no frequency information is lost. The triangular
    shape provides smooth interpolation between frequency bins.

    Args:
        num_filters: Number of Mel filters (40 is standard for ASR).
        fft_size: Size of FFT (determines frequency resolution).
        sample_rate: Audio sample rate.
        low_freq: Lowest frequency edge of the filterbank.
        high_freq: Highest frequency edge (default: Nyquist frequency).

    Returns:
        Matrix of shape (num_filters, fft_size // 2 + 1).
    """
    if high_freq is None:
        high_freq = sample_rate / 2.0  # Nyquist frequency

    # Convert frequency range to Mel scale
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)

    # Create num_filters + 2 evenly spaced points in Mel space
    # (+2 because we need left and right edges for the outermost filters)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert Hz points to FFT bin indices
    # Each FFT bin covers sample_rate / fft_size Hz
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    num_fft_bins = fft_size // 2 + 1
    filterbank = np.zeros((num_filters, num_fft_bins))

    for i in range(num_filters):
        # Left edge, center, and right edge of this triangular filter
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope: left to center
        for j in range(left, center):
            if center != left:  # Avoid division by zero
                filterbank[i, j] = (j - left) / (center - left)

        # Falling slope: center to right
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def compute_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int = 16000,
    frame_size_ms: float = 25.0,
    frame_stride_ms: float = 10.0,
    num_mel_filters: int = 40,
    fft_size: int = 512
) -> np.ndarray:
    """Compute Mel spectrogram from raw audio.

    This is the standard audio front-end for ASR systems. The pipeline:
    1. Pre-emphasis → boost high frequencies (speech energy drops at ~6dB/octave)
    2. Framing → split into overlapping windows
    3. Windowing → apply Hamming window to reduce spectral leakage
    4. FFT → convert each frame to frequency domain
    5. Mel filterbank → compress frequencies to perceptual scale
    6. Log → compress dynamic range

    Args:
        signal: Raw audio samples.
        sample_rate: Sample rate in Hz.
        frame_size_ms: Frame length in milliseconds.
        frame_stride_ms: Frame hop in milliseconds.
        num_mel_filters: Number of Mel filterbank channels.
        fft_size: FFT size (should be >= frame_length, zero-padded if not).

    Returns:
        Log Mel spectrogram of shape (num_frames, num_mel_filters).
    """
    # Step 1: Pre-emphasis — boost high frequencies to flatten the spectrum.
    # Speech has more energy at low frequencies; pre-emphasis compensates.
    pre_emphasis_coeff = 0.97
    emphasized = np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])

    # Step 2: Frame the signal into overlapping windows
    frame_length = int(frame_size_ms * sample_rate / 1000)
    frame_step = int(frame_stride_ms * sample_rate / 1000)
    num_samples = len(emphasized)

    # Calculate number of frames (at least 1)
    num_frames = max(1, 1 + (num_samples - frame_length) // frame_step)

    # Pad signal to fill last frame completely
    pad_length = max(num_samples, (num_frames - 1) * frame_step + frame_length)
    padded = np.zeros(pad_length)
    padded[:num_samples] = emphasized

    # Extract frames using stride tricks for efficiency
    # Each row is one frame of frame_length samples
    indices = np.arange(frame_length)[None, :] + np.arange(num_frames)[:, None] * frame_step
    frames = padded[indices]

    # Step 3: Apply Hamming window to each frame
    # The window tapers frame edges to zero, preventing discontinuities
    # that would cause spectral leakage (false high-frequency content)
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))
    frames *= hamming

    # Step 4: FFT — convert each frame from time domain to frequency domain
    # We only keep the positive frequencies (real signal → symmetric spectrum)
    fft_frames = np.fft.rfft(frames, n=fft_size)
    power_spectrum = np.abs(fft_frames) ** 2 / fft_size  # Power spectral density

    # Step 5: Apply Mel filterbank
    mel_filterbank = create_mel_filterbank(
        num_filters=num_mel_filters,
        fft_size=fft_size,
        sample_rate=sample_rate
    )
    mel_energies = power_spectrum @ mel_filterbank.T  # (num_frames, num_mel_filters)

    # Step 6: Log compression — matches human loudness perception
    # Add epsilon to avoid log(0)
    log_mel = np.log(mel_energies + 1e-10)

    return log_mel.astype(np.float32)


# =============================================================================
# Part 2: CTC (Connectionist Temporal Classification)
# =============================================================================

# Character vocabulary: 26 letters + space + blank token
# Blank is always index 0 by convention in CTC
BLANK = 0
VOCAB = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz ')
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)  # 28: blank + 26 letters + space


def ctc_forward(
    log_probs: np.ndarray,
    target: List[int]
) -> float:
    """Compute CTC loss using the forward algorithm.

    The CTC forward algorithm computes P(target | input) by summing over
    all possible alignments using dynamic programming. This is analogous
    to the forward algorithm in HMMs.

    Key idea: We expand the target by inserting blanks between every character
    and at the start/end. For target "ab", the expanded target is "εaεbε"
    (where ε = blank). The DP table tracks the probability of being at each
    position in this expanded target at each time step.

    Transitions allowed:
    - Stay at same position (repeat character or stay on blank)
    - Move to next position
    - Skip one position (but only over a blank, and only if the character
      before and after the blank are different)

    Args:
        log_probs: Log probabilities from model, shape (T, vocab_size).
        target: List of target character indices (without blanks).

    Returns:
        Negative log-likelihood (CTC loss) for this target.
    """
    T = log_probs.shape[0]  # Number of time steps
    L = len(target)

    # Expand target with blanks: [blank, t[0], blank, t[1], ..., blank, t[L-1], blank]
    S = 2 * L + 1  # Length of expanded target
    expanded = np.zeros(S, dtype=int)
    for i in range(L):
        expanded[2 * i + 1] = target[i]
    # Even indices are already 0 (blank)

    # DP table in log space to avoid numerical underflow
    # alpha[t, s] = log P(target[:s] aligned to input[:t])
    NEG_INF = -1e30
    alpha = np.full((T, S), NEG_INF)

    # Initialization: at t=0, we can only be at the first blank or first char
    alpha[0, 0] = log_probs[0, expanded[0]]  # Start with blank
    if S > 1:
        alpha[0, 1] = log_probs[0, expanded[1]]  # Start with first char

    # Helper for log-space addition: log(exp(a) + exp(b))
    def log_add(a: float, b: float) -> float:
        if a == NEG_INF:
            return b
        if b == NEG_INF:
            return a
        if a > b:
            return a + np.log1p(np.exp(b - a))
        return b + np.log1p(np.exp(a - b))

    # Fill DP table
    for t in range(1, T):
        for s in range(S):
            # Option 1: Stay at same position
            prev = alpha[t - 1, s]

            # Option 2: Move from previous position
            if s > 0:
                prev = log_add(prev, alpha[t - 1, s - 1])

            # Option 3: Skip a blank (only if current != previous non-blank)
            # This allows transitions like: ...char_a, blank, char_b → skip blank
            if s > 1 and expanded[s] != expanded[s - 2]:
                prev = log_add(prev, alpha[t - 1, s - 2])

            # Add log prob of emitting current label at this time step
            alpha[t, s] = prev + log_probs[t, expanded[s]]

    # Total probability: can end at last blank or last character
    log_prob = log_add(alpha[T - 1, S - 1], alpha[T - 1, S - 2])

    return -log_prob  # Return negative log-likelihood (loss)


# =============================================================================
# Part 3: Neural Network Components (NumPy only)
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation — simple, effective, and computationally cheap."""
    return np.maximum(0, x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Subtracting the max prevents overflow in exp() while giving
    identical results (exp(x-c) / sum(exp(x-c)) = exp(x) / sum(exp(x))).
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log-softmax: more numerically stable than log(softmax(x))."""
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return x - log_sum_exp


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid with clipping to prevent overflow."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class Conv1D:
    """1D convolution layer for processing spectral features.

    In ASR, conv layers serve two purposes:
    1. Extract local patterns (like phoneme-level features)
    2. Downsample the time axis (via stride > 1), reducing sequence length
       before the RNN — crucial for keeping CTC tractable.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        # Xavier/Glorot initialization for stable training
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = np.random.randn(out_channels, in_channels, kernel_size) * scale
        self.bias = np.zeros(out_channels)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: Input of shape (batch=1, in_channels, time_steps).

        Returns:
            Output of shape (batch=1, out_channels, new_time_steps).
        """
        _, in_ch, T = x.shape
        out_ch = self.weight.shape[0]

        # Output length after convolution with given stride
        out_T = (T - self.kernel_size) // self.stride + 1

        output = np.zeros((1, out_ch, out_T))

        for t in range(out_T):
            t_start = t * self.stride
            t_end = t_start + self.kernel_size
            # Extract patch and compute weighted sum across all input channels
            patch = x[0, :, t_start:t_end]  # (in_ch, kernel_size)
            # Einstein summation: output[c] = sum over (in_ch, kernel) of weight[c] * patch
            for c in range(out_ch):
                output[0, c, t] = np.sum(self.weight[c] * patch) + self.bias[c]

        return output


class GRUCell:
    """Single GRU (Gated Recurrent Unit) cell.

    GRU is simpler than LSTM (2 gates vs 3) with comparable performance.
    The two gates:
    - Reset gate (r): controls how much past state to forget
    - Update gate (z): controls how much new state to accept

    This balance of simplicity and expressiveness makes GRU popular in ASR.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Gate weights: concatenated for efficiency [reset, update, new]
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
            x: Input at this time step, shape (input_size,).
            h: Previous hidden state, shape (hidden_size,).

        Returns:
            New hidden state, shape (hidden_size,).
        """
        # Reset gate: decides what part of history to forget
        r = sigmoid(x @ self.W_ir + h @ self.W_hr + self.b_r)

        # Update gate: decides how much to update vs keep
        z = sigmoid(x @ self.W_iz + h @ self.W_hz + self.b_z)

        # Candidate new state: computed with reset gate applied to history
        n = np.tanh(x @ self.W_in + (r * h) @ self.W_hn + self.b_n)

        # Final state: interpolation between old state and candidate
        h_new = (1 - z) * n + z * h

        return h_new


class BiGRU:
    """Bidirectional GRU layer.

    Processes the sequence in both directions and concatenates outputs.
    This gives each time step access to both past and future context —
    critical for ASR because a phoneme's identity often depends on
    what comes after it (coarticulation effects).

    Trade-off: bidirectional models can't do real-time streaming, since
    they need the full utterance. For streaming ASR, you'd use unidirectional.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.forward_cell = GRUCell(input_size, hidden_size)
        self.backward_cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Process a sequence bidirectionally.

        Args:
            x: Input sequence, shape (seq_len, input_size).

        Returns:
            Concatenated forward+backward outputs, shape (seq_len, 2*hidden_size).
        """
        seq_len = x.shape[0]

        # Forward pass: left to right
        h_fwd = np.zeros(self.hidden_size)
        fwd_outputs = []
        for t in range(seq_len):
            h_fwd = self.forward_cell.forward(x[t], h_fwd)
            fwd_outputs.append(h_fwd.copy())

        # Backward pass: right to left
        h_bwd = np.zeros(self.hidden_size)
        bwd_outputs = [None] * seq_len
        for t in range(seq_len - 1, -1, -1):
            h_bwd = self.backward_cell.forward(x[t], h_bwd)
            bwd_outputs[t] = h_bwd.copy()

        # Concatenate forward and backward at each time step
        outputs = np.array([
            np.concatenate([fwd_outputs[t], bwd_outputs[t]])
            for t in range(seq_len)
        ])

        return outputs


class ASRModel:
    """Complete ASR model: Conv1D → BiGRU → Linear → Log-Softmax.

    Architecture choices:
    - Conv with stride=2 reduces sequence length by 2x, making the RNN faster
    - BiGRU captures bidirectional context
    - Linear projects to vocabulary size for CTC

    In production, you'd stack multiple BiGRU layers and use larger hidden
    sizes. This minimal version demonstrates the architecture pattern.
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
        self.conv = Conv1D(num_mel_filters, conv_channels, conv_kernel, conv_stride)
        self.bigru = BiGRU(conv_channels, hidden_size)

        # Linear projection: 2*hidden_size (bidirectional) → vocab_size
        proj_input = 2 * hidden_size
        scale = np.sqrt(2.0 / proj_input)
        self.W_out = np.random.randn(proj_input, vocab_size) * scale
        self.b_out = np.zeros(vocab_size)

    def forward(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Full forward pass from Mel spectrogram to log probabilities.

        Args:
            mel_spectrogram: Shape (num_frames, num_mel_filters).

        Returns:
            Log probabilities, shape (output_time_steps, vocab_size).
        """
        # Reshape for conv: (1, channels=num_mel_filters, time=num_frames)
        # We treat mel bins as "channels" — the conv slides over time
        x = mel_spectrogram.T[np.newaxis, :, :]  # (1, mel_filters, time)

        # Conv1D: extract local features and downsample time
        x = self.conv.forward(x)  # (1, conv_channels, time/stride)
        x = relu(x)

        # Reshape for RNN: (time_steps, conv_channels)
        x = x[0].T  # (time_steps, conv_channels)

        # BiGRU: capture temporal dependencies
        x = self.bigru.forward(x)  # (time_steps, 2*hidden_size)

        # Linear projection to vocabulary
        logits = x @ self.W_out + self.b_out  # (time_steps, vocab_size)

        # Log-softmax for CTC
        log_probs = log_softmax(logits, axis=-1)

        return log_probs


# =============================================================================
# Part 4: CTC Greedy Decoder
# =============================================================================

def ctc_greedy_decode(log_probs: np.ndarray) -> str:
    """Greedy CTC decoding: argmax at each step, then collapse.

    Algorithm:
    1. At each time step, pick the character with highest probability.
    2. Remove consecutive duplicates (key CTC rule).
    3. Remove blank tokens.
    4. Map remaining indices to characters.

    This is O(T) and simple but can make errors that beam search would avoid.
    For example, if the model briefly dips below threshold for a character
    in the middle, greedy decoding might split it into two characters.

    Args:
        log_probs: Model output, shape (T, vocab_size).

    Returns:
        Decoded text string.
    """
    # Step 1: Argmax at each time step
    best_path = np.argmax(log_probs, axis=1)

    # Step 2 & 3: Collapse consecutive duplicates and remove blanks
    decoded = []
    prev = -1  # Track previous token to detect duplicates
    for token in best_path:
        if token != prev:  # Only emit on change (collapse duplicates)
            if token != BLANK:  # Skip blank tokens
                decoded.append(token)
        prev = token

    # Step 4: Map indices to characters
    text = ''.join(IDX_TO_CHAR[idx] for idx in decoded)

    return text


def ctc_greedy_decode_with_detail(log_probs: np.ndarray) -> Tuple[str, List[dict]]:
    """Greedy decode with detailed step-by-step output for visualization.

    Returns both the decoded text and a trace of the decoding process,
    showing what the model predicted at each time step and how the
    collapsing rule was applied.
    """
    best_path = np.argmax(log_probs, axis=1)
    confidences = np.max(np.exp(log_probs), axis=1)  # Convert log-prob to prob

    trace = []
    decoded = []
    prev = -1

    for t, (token, conf) in enumerate(zip(best_path, confidences)):
        action = ""
        if token == prev:
            action = "duplicate (collapsed)"
        elif token == BLANK:
            action = "blank (removed)"
        else:
            action = "emitted"
            decoded.append(token)

        trace.append({
            'step': t,
            'token_idx': int(token),
            'token': VOCAB[token],
            'confidence': float(conf),
            'action': action
        })
        prev = token

    text = ''.join(IDX_TO_CHAR[idx] for idx in decoded)
    return text, trace


# =============================================================================
# Part 5: End-to-End Pipeline
# =============================================================================

class SpeechToTextPipeline:
    """Complete speech-to-text pipeline.

    Wires together all components:
    audio → Mel spectrogram → ASR model → CTC decoder → text
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_mel_filters: int = 40,
        hidden_size: int = 64
    ):
        self.sample_rate = sample_rate
        self.num_mel_filters = num_mel_filters
        self.model = ASRModel(
            num_mel_filters=num_mel_filters,
            hidden_size=hidden_size
        )

    def transcribe(self, audio: np.ndarray, verbose: bool = False) -> str:
        """Transcribe audio to text.

        Args:
            audio: Raw audio samples.
            verbose: If True, print intermediate steps.

        Returns:
            Transcribed text.
        """
        # Step 1: Extract features
        mel_spec = compute_mel_spectrogram(
            audio,
            sample_rate=self.sample_rate,
            num_mel_filters=self.num_mel_filters
        )
        if verbose:
            print(f"  Mel spectrogram shape: {mel_spec.shape}")
            print(f"  (= {mel_spec.shape[0]} frames x {mel_spec.shape[1]} mel bins)")

        # Step 2: Run model
        log_probs = self.model.forward(mel_spec)
        if verbose:
            print(f"  Model output shape: {log_probs.shape}")
            print(f"  (= {log_probs.shape[0]} time steps x {log_probs.shape[1]} vocab size)")

        # Step 3: Decode
        if verbose:
            text, trace = ctc_greedy_decode_with_detail(log_probs)
            print(f"\n  Decoding trace (first 20 steps):")
            for entry in trace[:20]:
                print(f"    t={entry['step']:3d}: '{entry['token']:>7s}' "
                      f"(conf={entry['confidence']:.3f}) → {entry['action']}")
            if len(trace) > 20:
                print(f"    ... ({len(trace) - 20} more steps)")
        else:
            text = ctc_greedy_decode(log_probs)

        return text


def demonstrate_mel_spectrogram():
    """Show the Mel spectrogram extraction process step by step."""
    print("=" * 70)
    print("STEP 1: MEL SPECTROGRAM EXTRACTION")
    print("=" * 70)

    # Generate a synthetic audio signal
    audio = generate_synthetic_audio(duration=0.5, frequencies=[200, 400, 800])
    print(f"\nSynthetic audio: {len(audio)} samples at 16kHz = {len(audio)/16000:.3f}s")
    print(f"Signal range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Show Mel scale conversion
    print(f"\nMel scale examples:")
    for hz in [100, 500, 1000, 2000, 4000, 8000]:
        mel = hz_to_mel(hz)
        print(f"  {hz:5d} Hz → {mel:7.1f} Mel")

    # Compute spectrogram
    mel_spec = compute_mel_spectrogram(audio, num_mel_filters=40)
    print(f"\nMel spectrogram shape: {mel_spec.shape}")
    print(f"  {mel_spec.shape[0]} frames (time) x {mel_spec.shape[1]} Mel bins (frequency)")
    print(f"  Value range: [{mel_spec.min():.2f}, {mel_spec.max():.2f}] (log scale)")

    # Show a slice
    print(f"\nFirst frame (40 Mel bins):")
    print(f"  {mel_spec[0, :10]}...")

    return mel_spec


def demonstrate_ctc():
    """Show how CTC loss and decoding work."""
    print("\n" + "=" * 70)
    print("STEP 2: CTC LOSS AND DECODING")
    print("=" * 70)

    # Create a simple example with known outputs
    T = 12  # 12 time steps
    target_text = "hi"
    target = [CHAR_TO_IDX[c] for c in target_text]

    print(f"\nTarget text: '{target_text}'")
    print(f"Target indices: {target} (h={CHAR_TO_IDX['h']}, i={CHAR_TO_IDX['i']})")
    print(f"Time steps: {T}")

    # Create "perfect" model output that should decode to "hi"
    # The model strongly predicts blank everywhere except the right spots
    log_probs = np.full((T, VOCAB_SIZE), -10.0)  # Very low prob for everything

    # Make it predict: blank, blank, h, h, blank, blank, i, i, blank, blank, blank, blank
    for t in [0, 1, 4, 5, 8, 9, 10, 11]:
        log_probs[t, BLANK] = -0.01  # High prob for blank
    for t in [2, 3]:
        log_probs[t, CHAR_TO_IDX['h']] = -0.01  # High prob for 'h'
    for t in [6, 7]:
        log_probs[t, CHAR_TO_IDX['i']] = -0.01  # High prob for 'i'

    # Normalize to proper log-probs
    log_probs = log_softmax(log_probs)

    # Compute CTC loss
    loss = ctc_forward(log_probs, target)
    print(f"\nCTC loss for perfect alignment: {loss:.4f}")
    print(f"(Lower is better — this should be close to 0)")

    # Decode
    decoded, trace = ctc_greedy_decode_with_detail(log_probs)
    print(f"\nGreedy decoding result: '{decoded}'")
    print(f"\nStep-by-step decoding:")
    for entry in trace:
        print(f"  t={entry['step']:2d}: predict '{entry['token']}' "
              f"(conf={entry['confidence']:.4f}) → {entry['action']}")

    # Show what happens with a BAD alignment
    print(f"\n--- Comparison: random model output ---")
    np.random.seed(42)
    random_logits = np.random.randn(T, VOCAB_SIZE)
    random_log_probs = log_softmax(random_logits)
    random_loss = ctc_forward(random_log_probs, target)
    random_decoded = ctc_greedy_decode(random_log_probs)
    print(f"CTC loss for random output: {random_loss:.4f}")
    print(f"Decoded text from random: '{random_decoded}'")

    return log_probs


def demonstrate_model():
    """Show the full model architecture and forward pass."""
    print("\n" + "=" * 70)
    print("STEP 3: ASR MODEL FORWARD PASS")
    print("=" * 70)

    np.random.seed(123)

    # Generate audio and extract features
    audio = generate_synthetic_audio(duration=1.0)
    mel_spec = compute_mel_spectrogram(audio, num_mel_filters=40)

    print(f"\nInput: Mel spectrogram of shape {mel_spec.shape}")

    # Build model
    model = ASRModel(num_mel_filters=40, hidden_size=64)

    # Forward pass with timing
    log_probs = model.forward(mel_spec)

    print(f"\nModel architecture:")
    print(f"  Conv1D: {model.conv.weight.shape[1]} → {model.conv.weight.shape[0]} channels, "
          f"kernel={model.conv.kernel_size}, stride={model.conv.stride}")
    print(f"  BiGRU: hidden_size={model.bigru.hidden_size} (output = {2*model.bigru.hidden_size})")
    print(f"  Linear: {model.W_out.shape[0]} → {model.W_out.shape[1]} (vocab size)")
    print(f"\nOutput: log probabilities of shape {log_probs.shape}")
    print(f"  {log_probs.shape[0]} time steps x {log_probs.shape[1]} characters")

    # Decode the (untrained) model's output
    decoded = ctc_greedy_decode(log_probs)
    print(f"\nDecoded output (untrained model): '{decoded}'")
    print(f"(Random gibberish expected — model hasn't been trained!)")

    # Show probability distribution at a few time steps
    probs = np.exp(log_probs)
    print(f"\nTop-3 predictions at first 5 time steps:")
    for t in range(min(5, log_probs.shape[0])):
        top3_idx = np.argsort(probs[t])[-3:][::-1]
        top3 = [(VOCAB[i], probs[t, i]) for i in top3_idx]
        print(f"  t={t}: {top3[0][0]}({top3[0][1]:.3f}), "
              f"{top3[1][0]}({top3[1][1]:.3f}), "
              f"{top3[2][0]}({top3[2][1]:.3f})")


def demonstrate_pipeline():
    """Show the full end-to-end pipeline."""
    print("\n" + "=" * 70)
    print("STEP 4: END-TO-END PIPELINE")
    print("=" * 70)

    np.random.seed(42)

    pipeline = SpeechToTextPipeline(hidden_size=64)

    # Generate different "utterances" (synthetic audio with different characteristics)
    durations = [0.5, 1.0, 2.0]
    freq_sets = [
        [200, 400],
        [150, 300, 600, 1200],
        [100, 250, 500, 1000, 2000, 4000],
    ]

    for dur, freqs in zip(durations, freq_sets):
        print(f"\n--- Audio: {dur}s, frequencies={freqs} ---")
        audio = generate_synthetic_audio(duration=dur, frequencies=freqs)
        text = pipeline.transcribe(audio, verbose=True)
        print(f"\n  Final transcription: '{text}'")

    # Demonstrate CTC decoding mechanics on a crafted example
    print("\n" + "=" * 70)
    print("STEP 5: CTC DECODING MECHANICS")
    print("=" * 70)

    print("\nShowing how CTC collapsing works on crafted sequences:")

    examples = [
        # (raw_path_description, raw_tokens)
        ("ε-ε-h-h-h-ε-e-e-ε-l-l-l-ε-l-l-ε-o-o-ε",
         [0,0,8,8,8,0,5,5,0,12,12,12,0,12,12,0,15,15,0]),
        ("h-ε-i-i-ε",
         [8,0,9,9,0]),
        ("ε-a-a-a-ε-b-ε-ε-c-ε",
         [0,1,1,1,0,2,0,0,3,0]),
    ]

    for desc, tokens in examples:
        # Build fake log_probs where the given token has highest prob at each step
        T = len(tokens)
        log_probs = np.full((T, VOCAB_SIZE), -10.0)
        for t, tok in enumerate(tokens):
            log_probs[t, tok] = 0.0
        log_probs = log_softmax(log_probs)

        decoded = ctc_greedy_decode(log_probs)
        print(f"\n  Raw path: {desc}")
        print(f"  Tokens:   {tokens}")
        print(f"  Decoded:  '{decoded}'")


if __name__ == '__main__':
    print("Day 061: Speech-to-Text Pipeline")
    print("================================\n")
    print("Building a complete ASR system from scratch using NumPy.")
    print("Components: Mel spectrogram → Conv+BiGRU model → CTC decoder\n")

    demonstrate_mel_spectrogram()
    demonstrate_ctc()
    demonstrate_model()
    demonstrate_pipeline()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key takeaways:
1. Mel spectrograms compress audio into a perceptually-meaningful 2D representation
2. CTC solves the alignment problem — we don't need to label every audio frame
3. The CTC forward algorithm uses DP to sum over all valid alignments
4. Greedy decoding (argmax + collapse) is simple but effective
5. An untrained model produces random text — training requires gradient descent
   on the CTC loss (which we computed but didn't backpropagate here)

In production:
- Use beam search decoding with a language model for much better accuracy
- Stack multiple BiGRU layers with residual connections
- Train on thousands of hours of labeled speech (e.g., LibriSpeech)
- Add SpecAugment data augmentation during training
- Consider attention-based architectures (Whisper-style) for best quality
""")
