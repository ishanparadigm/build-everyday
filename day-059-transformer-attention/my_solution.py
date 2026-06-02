"""
Day 059: Transformer Attention from Scratch — Your Implementation

Implement the core Transformer building blocks from scratch using NumPy:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Layer Normalization
4. Position-Wise Feed-Forward Network
5. Positional Encoding
6. Full Transformer Encoder Block

Key reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute numerically stable softmax along the given axis.

    Hint: Subtract max(x) before exp() to prevent overflow.
    softmax is shift-invariant, so this doesn't change the result.

    Args:
        x: Input array
        axis: Axis to compute softmax over

    Returns:
        Softmax probabilities (same shape as x)
    """
    raise NotImplementedError("TODO: implement numerically stable softmax")


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V.

    Hint: The scaling factor sqrt(d_k) is crucial — without it, large d_k
    causes dot products to grow, pushing softmax into saturation.

    For masking: set masked positions to a very large negative number
    before softmax so they get ~0 probability.

    Args:
        Q: Queries, shape (..., seq_len_q, d_k)
        K: Keys, shape (..., seq_len_k, d_k)
        V: Values, shape (..., seq_len_k, d_v)
        mask: Boolean mask, True = masked out. Shape broadcastable to (..., seq_q, seq_k)

    Returns:
        output: Shape (..., seq_len_q, d_v)
        attention_weights: Shape (..., seq_len_q, seq_len_k)
    """
    raise NotImplementedError("TODO: implement scaled dot-product attention")


class MultiHeadAttention:
    """
    Multi-Head Attention: h parallel attention heads with learned projections.

    Hint: The key trick is reshaping (batch, seq, d_model) into
    (batch, num_heads, seq, d_k) rather than creating separate matrices.
    d_k = d_model // num_heads.
    """

    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        """
        Initialize projection matrices W_Q, W_K, W_V, W_O.

        Hint: Use Xavier-like initialization (scale by 1/sqrt(d_model)).
        All four matrices are shape (d_model, d_model).
        """
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        raise NotImplementedError("TODO: initialize W_Q, W_K, W_V, W_O matrices")

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Reshape (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k).

        Hint: First reshape last dim to (num_heads, d_k), then transpose
        to put num_heads before seq_len.
        """
        raise NotImplementedError("TODO: split the last dimension into multiple heads")

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Reverse of _split_heads: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model).
        """
        raise NotImplementedError("TODO: combine heads back into d_model dimension")

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: project -> split heads -> attention -> combine -> project.

        Args:
            Q, K, V: Shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Shape (batch, seq_len, d_model)
            attention_weights: Shape (batch, num_heads, seq_len, seq_len)
        """
        raise NotImplementedError("TODO: implement multi-head attention forward pass")


class LayerNorm:
    """
    Layer Normalization: normalize across the feature (last) dimension.

    Hint: Unlike BatchNorm, LayerNorm normalizes each token independently.
    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Initialize gamma=1, beta=0.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.eps = eps
        raise NotImplementedError("TODO: initialize gamma and beta parameters")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Shape (..., d_model)
        Returns:
            Normalized tensor, same shape
        """
        raise NotImplementedError("TODO: implement layer normalization")


class FeedForward:
    """
    Position-wise FFN: FFN(x) = max(0, xW1 + b1)W2 + b2

    Hint: Inner dimension d_ff is typically 4x d_model.
    The ReLU non-linearity is essential — without it, two linear layers
    collapse to one.
    """

    def __init__(self, d_model: int, d_ff: int, seed: int = 42):
        raise NotImplementedError("TODO: initialize W1, b1, W2, b2")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Shape (..., d_model)
        Returns:
            Shape (..., d_model)
        """
        raise NotImplementedError("TODO: implement feed-forward forward pass")


def positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Hint: Compute the division term in log space for numerical stability:
    exp(2i * -log(10000) / d_model) = 1 / 10000^(2i/d_model)

    Args:
        max_seq_len: Maximum sequence length
        d_model: Model dimensionality

    Returns:
        Shape (max_seq_len, d_model)
    """
    raise NotImplementedError("TODO: implement sinusoidal positional encoding")


class TransformerEncoderBlock:
    """
    Full encoder block: Self-Attention -> Add&Norm -> FFN -> Add&Norm

    Hint: The residual connections (x + sublayer(x)) are critical for
    training deep networks. Don't forget them!
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, seed: int = 42):
        raise NotImplementedError("TODO: initialize attention, norms, and FFN")

    def forward(
        self, x: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: Shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Shape (batch, seq_len, d_model)
            attention_weights: Shape (batch, num_heads, seq_len, seq_len)
        """
        raise NotImplementedError("TODO: implement encoder block forward pass")


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create upper-triangular causal mask for autoregressive attention.

    Hint: Token i should only attend to tokens 0..i (not future tokens).
    True values will be masked (set to -inf before softmax).

    Args:
        seq_len: Sequence length

    Returns:
        Boolean mask, shape (seq_len, seq_len)
    """
    raise NotImplementedError("TODO: implement causal mask")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 8
    d_ff = 256

    print("Testing your Transformer implementation...")
    print("=" * 50)

    # Test softmax
    test_input = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    sm = softmax(test_input)
    print(f"Softmax test: {sm[0].round(4)}")
    assert np.allclose(sm.sum(axis=-1), 1.0), "Softmax rows must sum to 1"
    assert not np.any(np.isnan(sm)), "Softmax should handle large values"
    print("  PASSED")

    # Test scaled dot-product attention
    Q = np.random.randn(batch_size, seq_len, 8)
    K = np.random.randn(batch_size, seq_len, 8)
    V = np.random.randn(batch_size, seq_len, 8)
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (batch_size, seq_len, 8), f"Wrong output shape: {out.shape}"
    assert np.allclose(weights.sum(axis=-1), 1.0), "Attention weights must sum to 1"
    print(f"Scaled dot-product attention: output shape {out.shape} — PASSED")

    # Test with causal mask
    mask = create_causal_mask(seq_len)
    out_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert weights_masked[0, -1, -1] > 0, "Last token should attend to itself"
    print(f"Causal masking — PASSED")

    # Test multi-head attention
    X = np.random.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    mha_out, mha_weights = mha.forward(X, X, X)
    assert mha_out.shape == (batch_size, seq_len, d_model)
    print(f"Multi-head attention: output shape {mha_out.shape} — PASSED")

    # Test layer norm
    ln = LayerNorm(d_model)
    ln_out = ln.forward(X)
    assert ln_out.shape == X.shape
    # After layer norm, each token should have ~0 mean and ~1 std
    means = ln_out.mean(axis=-1)
    assert np.allclose(means, 0, atol=1e-5), f"LayerNorm mean should be ~0: {means.mean()}"
    print(f"Layer normalization — PASSED")

    # Test feed-forward
    ff = FeedForward(d_model, d_ff)
    ff_out = ff.forward(X)
    assert ff_out.shape == X.shape
    print(f"Feed-forward network: output shape {ff_out.shape} — PASSED")

    # Test positional encoding
    pe = positional_encoding(seq_len, d_model)
    assert pe.shape == (seq_len, d_model)
    print(f"Positional encoding: shape {pe.shape} — PASSED")

    # Test full encoder block
    encoder = TransformerEncoderBlock(d_model, num_heads, d_ff)
    enc_out, enc_attn = encoder.forward(X)
    assert enc_out.shape == X.shape
    print(f"Transformer encoder block: output shape {enc_out.shape} — PASSED")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
