"""
Day 059: Transformer Attention from Scratch

A complete implementation of the Transformer encoder architecture using only NumPy.
Covers scaled dot-product attention, multi-head attention, layer normalization,
position-wise feed-forward networks, positional encoding, and the full encoder block.

Building on Day 015 (neural network forward pass) — attention is a dynamic,
input-dependent weighted sum, not a fixed weight matrix.
"""

import numpy as np
from typing import Optional, Tuple


# =============================================================================
# Utility: Numerically Stable Softmax
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax along the given axis with numerical stability.

    The trick: subtract max(x) before exponentiating. Since softmax is
    shift-invariant (softmax(x) == softmax(x - c) for any scalar c),
    this doesn't change the result but prevents exp() from overflowing.

    Without this, exp(1000) = inf, and inf/inf = NaN.
    With this, the largest exponent is exp(0) = 1, and everything works.
    """
    # Subtract max for numerical stability — keepdims so broadcasting works
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# Scaled Dot-Product Attention
# =============================================================================

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V.

    Args:
        Q: Query matrix, shape (..., seq_len_q, d_k)
        K: Key matrix, shape (..., seq_len_k, d_k)
        V: Value matrix, shape (..., seq_len_k, d_v)
        mask: Optional boolean mask, shape broadcastable to (..., seq_len_q, seq_len_k).
              True values are MASKED (set to -inf before softmax).

    Returns:
        output: Weighted sum of values, shape (..., seq_len_q, d_v)
        attention_weights: Softmax weights, shape (..., seq_len_q, seq_len_k)

    Why scale by sqrt(d_k)?
    -----------------------
    If Q and K entries have mean=0, var=1, then each entry of Q @ K^T is a sum
    of d_k products of unit-variance terms, giving variance = d_k. For d_k=512,
    dot products reach ~22 in magnitude, pushing softmax into near-one-hot
    distributions where gradients vanish. Dividing by sqrt(512) ≈ 22.6 brings
    variance back to ~1, keeping softmax in a well-behaved regime.
    """
    d_k = Q.shape[-1]

    # Step 1: Compute raw attention scores
    # Q @ K^T: each query's similarity to each key
    # Shape: (..., seq_len_q, seq_len_k)
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)

    # Step 2: Apply mask (e.g., causal mask to prevent attending to future tokens)
    # We set masked positions to -inf so softmax assigns them probability ≈ 0
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Step 3: Softmax over the key dimension — convert scores to probabilities
    attention_weights = softmax(scores, axis=-1)

    # Step 4: Weighted sum of values
    # Each query's output is a mixture of all values, weighted by attention
    output = attention_weights @ V

    return output, attention_weights


# =============================================================================
# Multi-Head Attention
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention: run h parallel attention functions, each on a
    different learned linear projection of Q, K, V.

    Why multiple heads?
    -------------------
    A single attention head computes one set of attention weights — one way
    of deciding which tokens are relevant to which. But language has many
    simultaneous types of relationships: syntactic (subject-verb), semantic
    (coreference), positional (adjacent tokens), etc. Multiple heads let
    the model capture all of these in parallel.

    Parameter efficiency: with h heads and d_model total dimensions, each
    head operates on d_model/h dimensions. Total parameter count is roughly
    the same as single-head attention with full d_model, but representation
    power is much richer.
    """

    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        """
        Args:
            d_model: Total model dimensionality (must be divisible by num_heads)
            num_heads: Number of parallel attention heads
            seed: Random seed for reproducible weight initialization
        """
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        rng = np.random.RandomState(seed)

        # Xavier/Glorot initialization: scale by 1/sqrt(d_model) to keep
        # activations from exploding or vanishing through layers
        scale = 1.0 / np.sqrt(d_model)

        # Projection matrices for Q, K, V — each projects from d_model to d_model
        # (which gets split across heads: d_model = num_heads * d_k)
        self.W_Q = rng.randn(d_model, d_model) * scale
        self.W_K = rng.randn(d_model, d_model) * scale
        self.W_V = rng.randn(d_model, d_model) * scale

        # Output projection: concatenated heads (d_model) -> d_model
        self.W_O = rng.randn(d_model, d_model) * scale

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Reshape (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k).

        This is the key trick: instead of creating num_heads separate matrices,
        we reshape the last dimension. For d_model=512 and num_heads=8:
        (batch, seq_len, 512) -> (batch, seq_len, 8, 64) -> (batch, 8, seq_len, 64)

        The transpose puts heads in dim 1 so we can batch attention across heads.
        """
        batch_size, seq_len, _ = x.shape
        # Split d_model into (num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch, heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Reverse of _split_heads: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model).

        After attention, we need to recombine the heads back into a single
        d_model-dimensional representation.
        """
        batch_size, _, seq_len, _ = x.shape
        # Transpose back: (batch, heads, seq_len, d_k) -> (batch, seq_len, heads, d_k)
        x = x.transpose(0, 2, 1, 3)
        # Merge heads: (batch, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.

        Args:
            Q, K, V: Input tensors, shape (batch, seq_len, d_model)
            mask: Optional mask, shape broadcastable to (batch, num_heads, seq_q, seq_k)

        Returns:
            output: Shape (batch, seq_len, d_model)
            attention_weights: Shape (batch, num_heads, seq_len, seq_len)
        """
        # Step 1: Linear projections
        # Each projection learns what to query/key/value for attention
        Q_proj = Q @ self.W_Q  # (batch, seq_len, d_model)
        K_proj = K @ self.W_K
        V_proj = V @ self.W_V

        # Step 2: Split into heads
        Q_heads = self._split_heads(Q_proj)  # (batch, heads, seq_len, d_k)
        K_heads = self._split_heads(K_proj)
        V_heads = self._split_heads(V_proj)

        # Step 3: Apply attention to all heads in parallel
        # Thanks to the batched shape, this one call handles all heads
        attn_output, attn_weights = scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, mask
        )

        # Step 4: Combine heads back
        combined = self._combine_heads(attn_output)  # (batch, seq_len, d_model)

        # Step 5: Final linear projection
        # This lets the model learn how to combine information from different heads
        output = combined @ self.W_O

        return output, attn_weights


# =============================================================================
# Layer Normalization
# =============================================================================

class LayerNorm:
    """
    Layer Normalization: normalize across the feature dimension (last dim).

    Unlike Batch Normalization (which normalizes across the batch dimension),
    Layer Norm normalizes each token independently. This is important because:
    1. Sequence lengths vary — batch stats are unstable
    2. At inference, we process one sequence at a time — no batch to compute stats over
    3. Layer norm makes each token's representation well-conditioned independently

    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    where gamma (scale) and beta (shift) are learned parameters.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)   # Learned scale, initialized to 1
        self.beta = np.zeros(d_model)   # Learned shift, initialized to 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input tensor, shape (..., d_model)
        Returns:
            Normalized tensor, same shape
        """
        # Compute mean and variance across the last dimension (feature dim)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Normalize: center and scale to unit variance
        # eps prevents division by zero when variance is tiny
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Apply learned affine transformation
        # gamma and beta let the model "undo" normalization if that's optimal
        return self.gamma * x_norm + self.beta


# =============================================================================
# Position-Wise Feed-Forward Network
# =============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network: FFN(x) = max(0, xW1 + b1)W2 + b2

    Applied identically to each position (token) independently. This is where
    the model does its "thinking" — attention gathers information from other
    positions, and the FFN processes it.

    The inner dimension (d_ff) is typically 4x d_model. This expansion-then-
    compression creates a higher-dimensional space where the model can learn
    more complex functions before projecting back down.
    """

    def __init__(self, d_model: int, d_ff: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale1 = 1.0 / np.sqrt(d_model)
        scale2 = 1.0 / np.sqrt(d_ff)

        self.W1 = rng.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Shape (..., d_model)
        Returns:
            Shape (..., d_model)
        """
        # First linear + ReLU activation
        # ReLU introduces non-linearity — without it, two linear layers
        # collapse to a single linear transformation (matrix multiplication)
        hidden = np.maximum(0, x @ self.W1 + self.b1)

        # Second linear projects back to d_model
        return hidden @ self.W2 + self.b2


# =============================================================================
# Positional Encoding
# =============================================================================

def positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.

    Attention is permutation-invariant: Attention({A, B, C}) == Attention({C, A, B}).
    But word order matters! Positional encoding injects position information.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sinusoidal?
    - Each dimension has a different frequency, creating unique position "fingerprints"
    - The model can learn relative positions: PE(pos+k) is a linear function of PE(pos)
    - Generalizes to longer sequences than seen during training (unlike learned embeddings)

    Args:
        max_seq_len: Maximum sequence length
        d_model: Model dimensionality

    Returns:
        Positional encoding matrix, shape (max_seq_len, d_model)
    """
    pe = np.zeros((max_seq_len, d_model))
    position = np.arange(max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)

    # Compute the division term: 10000^(2i/d_model)
    # We compute in log space for numerical stability:
    # exp(2i * -log(10000) / d_model) = 1 / 10000^(2i/d_model)
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )

    # Even indices: sin, Odd indices: cos
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


# =============================================================================
# Transformer Encoder Block
# =============================================================================

class TransformerEncoderBlock:
    """
    A single Transformer encoder block:

        x -> [Multi-Head Self-Attention] -> [Add & Norm] -> [FFN] -> [Add & Norm] -> output

    The residual connections (Add) are critical:
    - They allow gradients to bypass the attention/FFN layers during backprop
    - Without them, deep transformers (12+ layers) cannot train
    - They let each layer learn a "delta" to add to the representation,
      rather than having to learn the entire transformation
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, seed: int = 42):
        self.attention = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 1)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self, x: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: Input, shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Shape (batch, seq_len, d_model)
            attention_weights: Shape (batch, num_heads, seq_len, seq_len)
        """
        # Sub-layer 1: Multi-Head Self-Attention + Residual + LayerNorm
        # "Self" attention: Q, K, V all come from the same input
        attn_output, attn_weights = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)  # Residual connection, then normalize

        # Sub-layer 2: Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)

        return x, attn_weights


# =============================================================================
# Causal (Autoregressive) Mask
# =============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create an upper-triangular mask for autoregressive (decoder) attention.

    In language generation, token at position i should only attend to tokens
    at positions <= i (can't look at the future). This mask sets future
    positions to True (meaning "mask this out").

    For seq_len=4:
        [[False, True,  True,  True ],    # token 0 sees only itself
         [False, False, True,  True ],    # token 1 sees 0, 1
         [False, False, False, True ],    # token 2 sees 0, 1, 2
         [False, False, False, False]]    # token 3 sees all
    """
    # np.triu with k=1 gives upper triangle above the main diagonal
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


# =============================================================================
# Demo: Putting It All Together
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # === Configuration ===
    batch_size = 2
    seq_len = 6       # 6 tokens per sequence
    d_model = 64      # Model dimensionality (small for demo; GPT-3 uses 12288)
    num_heads = 8     # 8 attention heads (d_k = 64/8 = 8 per head)
    d_ff = 256        # Feed-forward inner dimension (4x d_model)

    print("=" * 70)
    print("TRANSFORMER ATTENTION FROM SCRATCH")
    print("=" * 70)

    # === Step 1: Create fake input embeddings ===
    # In a real model, these would come from an embedding layer (token -> vector)
    X = np.random.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {X.shape}")
    print(f"  (batch_size={batch_size}, seq_len={seq_len}, d_model={d_model})")

    # === Step 2: Add Positional Encoding ===
    pe = positional_encoding(seq_len, d_model)
    print(f"\nPositional encoding shape: {pe.shape}")
    print(f"  First position, first 8 dims: {pe[0, :8].round(4)}")
    print(f"  Last position, first 8 dims:  {pe[-1, :8].round(4)}")

    X_with_pos = X + pe[np.newaxis, :, :]  # Broadcast: (1, seq_len, d_model)
    print(f"\nInput + positional encoding applied (shape unchanged: {X_with_pos.shape})")

    # === Step 3: Scaled Dot-Product Attention (single head) ===
    print("\n" + "=" * 70)
    print("STEP 3: Scaled Dot-Product Attention (Single Head)")
    print("=" * 70)

    Q = X_with_pos[:, :, :8]  # Use first 8 dims as Q (simulating d_k=8)
    K = X_with_pos[:, :, 8:16]
    V = X_with_pos[:, :, 16:24]

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"\n  Q shape: {Q.shape}  (batch, seq_len, d_k)")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {weights.shape}")
    print(f"\n  Attention weights for batch 0 (each row sums to 1.0):")
    for i in range(seq_len):
        row = weights[0, i]
        print(f"    Token {i} attends to: [{', '.join(f'{w:.3f}' for w in row)}]  sum={row.sum():.4f}")

    # === Step 4: Causal Masking ===
    print("\n" + "=" * 70)
    print("STEP 4: Causal (Autoregressive) Masking")
    print("=" * 70)

    causal_mask = create_causal_mask(seq_len)
    print(f"\n  Causal mask (True = masked out):")
    for i in range(seq_len):
        print(f"    Token {i}: {causal_mask[i].astype(int)}")

    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print(f"\n  Causal attention weights for batch 0:")
    for i in range(seq_len):
        row = weights_causal[0, i]
        print(f"    Token {i} attends to: [{', '.join(f'{w:.3f}' for w in row)}]  sum={row.sum():.4f}")
    print("  (Notice: each token only attends to itself and earlier tokens)")

    # === Step 5: Multi-Head Attention ===
    print("\n" + "=" * 70)
    print("STEP 5: Multi-Head Attention")
    print("=" * 70)

    mha = MultiHeadAttention(d_model, num_heads, seed=42)
    mha_output, mha_weights = mha.forward(X_with_pos, X_with_pos, X_with_pos)

    print(f"\n  d_model={d_model}, num_heads={num_heads}, d_k={d_model // num_heads}")
    print(f"  Input shape:  {X_with_pos.shape}")
    print(f"  Output shape: {mha_output.shape}")
    print(f"  Attention weights shape: {mha_weights.shape}")
    print(f"    (batch, num_heads, seq_len_q, seq_len_k)")

    print(f"\n  Attention patterns differ across heads (batch 0, token 0):")
    for h in range(min(4, num_heads)):
        row = mha_weights[0, h, 0]
        max_idx = np.argmax(row)
        print(f"    Head {h}: [{', '.join(f'{w:.3f}' for w in row)}]  max_attn -> token {max_idx}")

    # === Step 6: Full Transformer Encoder Block ===
    print("\n" + "=" * 70)
    print("STEP 6: Full Transformer Encoder Block")
    print("=" * 70)

    encoder = TransformerEncoderBlock(d_model, num_heads, d_ff, seed=42)
    enc_output, enc_attn = encoder.forward(X_with_pos)

    print(f"\n  Input shape:  {X_with_pos.shape}")
    print(f"  Output shape: {enc_output.shape}")

    # Show how the representation changes through the block
    input_norm = np.linalg.norm(X_with_pos, axis=-1).mean()
    output_norm = np.linalg.norm(enc_output, axis=-1).mean()
    print(f"\n  Mean L2 norm — input: {input_norm:.4f}, output: {output_norm:.4f}")
    print(f"  (LayerNorm keeps representations well-conditioned)")

    # Verify residual connections work: output should be correlated with input
    # (the identity path ensures information isn't lost)
    cosine_sim = np.mean([
        np.dot(X_with_pos[0, i], enc_output[0, i])
        / (np.linalg.norm(X_with_pos[0, i]) * np.linalg.norm(enc_output[0, i]) + 1e-8)
        for i in range(seq_len)
    ])
    print(f"  Mean cosine similarity (input vs output): {cosine_sim:.4f}")
    print(f"  (Residual connections preserve input information)")

    # === Step 7: Stacking Multiple Encoder Blocks ===
    print("\n" + "=" * 70)
    print("STEP 7: Stacking 3 Encoder Blocks (like a mini-BERT)")
    print("=" * 70)

    num_layers = 3
    layers = [
        TransformerEncoderBlock(d_model, num_heads, d_ff, seed=42 + i)
        for i in range(num_layers)
    ]

    h = X_with_pos
    all_attn = []
    for i, layer in enumerate(layers):
        h, attn = layer.forward(h)
        norm = np.linalg.norm(h, axis=-1).mean()
        print(f"  Layer {i+1} output norm: {norm:.4f}")
        all_attn.append(attn)

    print(f"\n  Final output shape: {h.shape}")
    print(f"  Each layer refines the representation, with LayerNorm")
    print(f"  keeping norms stable across depth.")

    # === Summary Statistics ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_params = (
        d_model * d_model * 4  # W_Q, W_K, W_V, W_O in MHA
        + d_model * d_ff + d_ff  # W1, b1 in FFN
        + d_ff * d_model + d_model  # W2, b2 in FFN
        + d_model * 2 * 2  # gamma, beta for 2 LayerNorms
    )
    print(f"\n  Parameters per encoder block: {total_params:,}")
    print(f"  Parameters for {num_layers} blocks:  {total_params * num_layers:,}")
    print(f"\n  For comparison:")
    print(f"    BERT-base:  110M params (12 layers, d_model=768, 12 heads)")
    print(f"    GPT-3:     175B params (96 layers, d_model=12288, 96 heads)")

    print(f"\n  Key insight: the same simple building block —")
    print(f"  attention + FFN + residual + norm — scales from this toy example")
    print(f"  to the largest models in the world. The architecture is elegant;")
    print(f"  the magic is in scale, data, and training.")

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 70)
