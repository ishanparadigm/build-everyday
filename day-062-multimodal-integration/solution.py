"""
Day 62: Multi-Modal Model Integration

A complete multi-modal system that fuses text, image, and audio representations
using three fusion strategies (early, late, cross-attention) with contrastive
pre-alignment and modality dropout for robustness.

We use numpy only — no deep learning frameworks — to expose every operation.
"""

import numpy as np
from typing import Optional


# =============================================================================
# Utility Functions
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    # Subtract max for numerical stability — prevents exp overflow
    # This doesn't change the output because softmax(x) = softmax(x - c)
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization — stabilizes training by normalizing activations.

    Unlike batch norm, layer norm normalizes across features (not batch),
    making it suitable for variable-length sequences and small batches.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: [-1, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# Individual Modality Encoders
# =============================================================================

class TextEncoder:
    """Encodes text (as token indices) into a fixed-size embedding.

    Architecture: Embedding lookup → mean pooling → MLP projection.

    Why mean pooling? It's simple and surprisingly effective for short texts.
    For longer sequences, you'd want attention-based pooling (like [CLS] token
    in BERT), but mean pooling captures the "average meaning" well.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Xavier initialization — keeps variance stable across layers
        # Scale = sqrt(2 / (fan_in + fan_out))
        self.embeddings = rng.randn(vocab_size, embed_dim) * np.sqrt(2.0 / (vocab_size + embed_dim))
        self.W1 = rng.randn(embed_dim, hidden_dim) * np.sqrt(2.0 / (embed_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros(output_dim)

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """Encode a sequence of token IDs into a single vector.

        Args:
            token_ids: shape (seq_len,) — integer token indices

        Returns:
            shape (output_dim,) — the text embedding
        """
        # Look up embeddings for each token
        embedded = self.embeddings[token_ids]  # (seq_len, embed_dim)
        # Mean pool across sequence — collapses variable-length input to fixed size
        pooled = np.mean(embedded, axis=0)  # (embed_dim,)
        # Two-layer MLP with ReLU — learns nonlinear transformation
        hidden = relu(pooled @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        return layer_norm(output)


class ImageEncoder:
    """Encodes a small image into a fixed-size embedding.

    Architecture: Conv layer → ReLU → Global average pooling → Linear projection.

    We implement a single conv layer manually. In production you'd use a
    pre-trained ResNet or ViT, but the principle is identical: extract spatial
    features, then pool to a fixed-size vector.
    """

    def __init__(self, in_channels: int, num_filters: int, filter_size: int, output_dim: int, seed: int = 43):
        rng = np.random.RandomState(seed)
        # Conv filters — each filter detects a different pattern (edges, textures, etc.)
        # Shape: (num_filters, in_channels, filter_size, filter_size)
        self.filters = rng.randn(num_filters, in_channels, filter_size, filter_size) * np.sqrt(
            2.0 / (in_channels * filter_size * filter_size)
        )
        self.conv_bias = np.zeros(num_filters)
        # Projection from pooled features to output dim
        self.W_proj = rng.randn(num_filters, output_dim) * np.sqrt(2.0 / (num_filters + output_dim))
        self.b_proj = np.zeros(output_dim)
        self.filter_size = filter_size

    def _conv2d(self, image: np.ndarray) -> np.ndarray:
        """Apply convolution filters to an image.

        Args:
            image: shape (channels, height, width)

        Returns:
            shape (num_filters, out_height, out_width) — feature maps
        """
        C, H, W = image.shape
        num_filters = self.filters.shape[0]
        fs = self.filter_size
        out_h = H - fs + 1
        out_w = W - fs + 1

        output = np.zeros((num_filters, out_h, out_w))
        for f in range(num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    # Extract the patch and compute dot product with filter
                    # This is the core conv operation: sliding window + element-wise multiply + sum
                    patch = image[:, i:i+fs, j:j+fs]
                    output[f, i, j] = np.sum(patch * self.filters[f]) + self.conv_bias[f]
        return output

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode an image into a single vector.

        Args:
            image: shape (channels, height, width)

        Returns:
            shape (output_dim,) — the image embedding
        """
        # Apply conv filters to get feature maps
        feature_maps = self._conv2d(image)  # (num_filters, H', W')
        # ReLU activation — introduces nonlinearity
        feature_maps = relu(feature_maps)
        # Global average pooling — take mean of each feature map
        # This makes the representation invariant to spatial position
        pooled = np.mean(feature_maps, axis=(1, 2))  # (num_filters,)
        # Project to output dimension
        output = pooled @ self.W_proj + self.b_proj
        return layer_norm(output)


class AudioEncoder:
    """Encodes audio features (spectrogram) into a fixed-size embedding.

    Architecture: 1D conv over time frames → ReLU → Global average pooling → Linear.

    Audio is typically represented as a spectrogram: a 2D matrix of
    (frequency_bins × time_frames). We treat frequency bins as "channels"
    and convolve over time, capturing temporal patterns in the audio.
    """

    def __init__(self, freq_bins: int, num_filters: int, filter_size: int, output_dim: int, seed: int = 44):
        rng = np.random.RandomState(seed)
        # 1D conv filters — convolve over time axis
        # Shape: (num_filters, freq_bins, filter_size)
        self.filters = rng.randn(num_filters, freq_bins, filter_size) * np.sqrt(
            2.0 / (freq_bins * filter_size)
        )
        self.conv_bias = np.zeros(num_filters)
        self.W_proj = rng.randn(num_filters, output_dim) * np.sqrt(2.0 / (num_filters + output_dim))
        self.b_proj = np.zeros(output_dim)
        self.filter_size = filter_size

    def _conv1d(self, spectrogram: np.ndarray) -> np.ndarray:
        """Apply 1D convolution over time axis.

        Args:
            spectrogram: shape (freq_bins, time_frames)

        Returns:
            shape (num_filters, out_time) — temporal feature maps
        """
        freq_bins, T = spectrogram.shape
        num_filters = self.filters.shape[0]
        fs = self.filter_size
        out_t = T - fs + 1

        output = np.zeros((num_filters, out_t))
        for f in range(num_filters):
            for t in range(out_t):
                patch = spectrogram[:, t:t+fs]
                output[f, t] = np.sum(patch * self.filters[f]) + self.conv_bias[f]
        return output

    def encode(self, spectrogram: np.ndarray) -> np.ndarray:
        """Encode a spectrogram into a single vector.

        Args:
            spectrogram: shape (freq_bins, time_frames)

        Returns:
            shape (output_dim,) — the audio embedding
        """
        temporal_features = self._conv1d(spectrogram)  # (num_filters, T')
        temporal_features = relu(temporal_features)
        pooled = np.mean(temporal_features, axis=1)  # (num_filters,)
        output = pooled @ self.W_proj + self.b_proj
        return layer_norm(output)


# =============================================================================
# Fusion Strategies
# =============================================================================

class EarlyFusion:
    """Concatenate embeddings from all modalities, then process jointly.

    The simplest approach: just stack everything and let a single network
    sort it out. The MLP after concatenation can learn arbitrary interactions
    between modalities.

    Tradeoff: Maximum expressiveness, but the network must learn cross-modal
    relationships from scratch (no inductive bias about what each modality means).
    """

    def __init__(self, input_dims: list[int], hidden_dim: int, output_dim: int, seed: int = 50):
        rng = np.random.RandomState(seed)
        total_input = sum(input_dims)
        self.W1 = rng.randn(total_input, hidden_dim) * np.sqrt(2.0 / (total_input + hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros(output_dim)

    def fuse(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Fuse multiple modality embeddings by concatenation + MLP.

        Args:
            embeddings: list of 1D arrays, one per modality

        Returns:
            shape (output_dim,) — fused representation
        """
        # Concatenate all embeddings into one long vector
        concat = np.concatenate(embeddings)
        # Two-layer MLP learns cross-modal interactions
        hidden = relu(concat @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class LateFusion:
    """Process each modality independently, combine only the final predictions.

    Each modality gets its own classifier head. The final prediction is a
    weighted combination of per-modality predictions.

    Tradeoff: Most modular — you can swap out any encoder without retraining
    others. But misses fine-grained cross-modal interactions (e.g., "the text
    says 'cat' but the image shows a dog" requires early interaction).
    """

    def __init__(self, modality_dims: list[int], output_dim: int, seed: int = 51):
        rng = np.random.RandomState(seed)
        self.heads = []
        for dim in modality_dims:
            W = rng.randn(dim, output_dim) * np.sqrt(2.0 / (dim + output_dim))
            b = np.zeros(output_dim)
            self.heads.append((W, b))
        # Learnable weights for combining modality predictions
        # Initialized uniformly — each modality starts equally important
        self.modality_weights = np.ones(len(modality_dims)) / len(modality_dims)

    def fuse(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Fuse by weighted average of per-modality predictions.

        Args:
            embeddings: list of 1D arrays, one per modality

        Returns:
            shape (output_dim,) — fused prediction logits
        """
        predictions = []
        for emb, (W, b) in zip(embeddings, self.heads):
            pred = emb @ W + b
            predictions.append(pred)

        # Weighted combination — weights could be learned via gradient descent
        # or set based on modality reliability
        result = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.modality_weights):
            result += weight * pred
        return result


class CrossAttentionFusion:
    """Use attention to let modalities interact at the representation level.

    The "query" modality attends to "key/value" modalities. This is exactly
    the encoder-decoder attention from transformers (Day 59), but here the
    encoder and decoder are different modalities instead of different languages.

    Example: text queries attending to image keys — for each word, the model
    learns which image region is most relevant.

    Tradeoff: Most powerful for capturing cross-modal dependencies, but
    quadratic complexity in sequence length and harder to train.
    """

    def __init__(self, query_dim: int, kv_dim: int, hidden_dim: int, output_dim: int,
                 num_kv_tokens: int = 4, seed: int = 52):
        rng = np.random.RandomState(seed)

        # Project query and key/value into the same space for attention
        self.W_q = rng.randn(query_dim, hidden_dim) * np.sqrt(2.0 / (query_dim + hidden_dim))
        self.W_k = rng.randn(kv_dim, hidden_dim) * np.sqrt(2.0 / (kv_dim + hidden_dim))
        self.W_v = rng.randn(kv_dim, hidden_dim) * np.sqrt(2.0 / (kv_dim + hidden_dim))

        # Final projection after attention
        self.W_out = rng.randn(hidden_dim + query_dim, output_dim) * np.sqrt(
            2.0 / (hidden_dim + query_dim + output_dim)
        )
        self.b_out = np.zeros(output_dim)
        self.hidden_dim = hidden_dim

        # We'll split the KV embedding into multiple "tokens" to give attention
        # something to attend over. In practice, these come from spatial positions
        # (image patches) or temporal positions (audio frames).
        self.num_kv_tokens = num_kv_tokens
        self.W_split = rng.randn(kv_dim, num_kv_tokens * kv_dim) * np.sqrt(2.0 / (kv_dim + num_kv_tokens * kv_dim))
        self.kv_dim = kv_dim

    def fuse(self, query_emb: np.ndarray, kv_emb: np.ndarray) -> np.ndarray:
        """Fuse query modality with key/value modality via cross-attention.

        Args:
            query_emb: shape (query_dim,) — the modality asking questions
            kv_emb: shape (kv_dim,) — the modality being attended to

        Returns:
            shape (output_dim,) — fused representation
        """
        # Create multiple KV tokens by projecting and reshaping
        kv_tokens = (kv_emb @ self.W_split).reshape(self.num_kv_tokens, self.kv_dim)

        # Project query and KV tokens into attention space
        Q = query_emb @ self.W_q  # (hidden_dim,)
        K = kv_tokens @ self.W_k  # (num_kv_tokens, hidden_dim)
        V = kv_tokens @ self.W_v  # (num_kv_tokens, hidden_dim)

        # Scaled dot-product attention
        # Scale by sqrt(d_k) to prevent dot products from growing too large,
        # which would push softmax into saturation (near-zero gradients)
        scale = np.sqrt(self.hidden_dim)
        attn_scores = (Q @ K.T) / scale  # (num_kv_tokens,)
        attn_weights = softmax(attn_scores)  # (num_kv_tokens,)

        # Weighted sum of values
        attended = attn_weights @ V  # (hidden_dim,)

        # Concatenate query with attended representation, then project
        # This residual-like connection preserves the original query information
        combined = np.concatenate([query_emb, attended])
        output = combined @ self.W_out + self.b_out
        return output, attn_weights


# =============================================================================
# Multi-Modal System
# =============================================================================

class MultiModalSystem:
    """Complete multi-modal system combining encoders, fusion, and classification.

    This is the main class that ties everything together. It:
    1. Encodes each modality with its specialized encoder
    2. Fuses representations using the chosen strategy
    3. Classifies the fused representation
    4. Supports modality dropout for robustness
    """

    def __init__(self, config: dict):
        # Shared embedding dimension — all modalities project to this size
        self.shared_dim = config.get("shared_dim", 32)
        num_classes = config.get("num_classes", 5)

        # Initialize encoders
        self.text_encoder = TextEncoder(
            vocab_size=config.get("vocab_size", 100),
            embed_dim=16,
            hidden_dim=32,
            output_dim=self.shared_dim,
        )
        self.image_encoder = ImageEncoder(
            in_channels=config.get("image_channels", 3),
            num_filters=8,
            filter_size=3,
            output_dim=self.shared_dim,
        )
        self.audio_encoder = AudioEncoder(
            freq_bins=config.get("freq_bins", 16),
            num_filters=8,
            filter_size=3,
            output_dim=self.shared_dim,
        )

        # Initialize all three fusion strategies
        self.early_fusion = EarlyFusion(
            input_dims=[self.shared_dim] * 3,
            hidden_dim=64,
            output_dim=num_classes,
        )
        self.late_fusion = LateFusion(
            modality_dims=[self.shared_dim] * 3,
            output_dim=num_classes,
        )
        self.cross_attention = CrossAttentionFusion(
            query_dim=self.shared_dim,
            kv_dim=self.shared_dim,
            hidden_dim=self.shared_dim,
            output_dim=num_classes,
        )

        self.modality_dropout_rate = config.get("modality_dropout_rate", 0.2)

    def encode_all(
        self,
        text_tokens: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        audio_spectrogram: Optional[np.ndarray] = None,
    ) -> dict[str, Optional[np.ndarray]]:
        """Encode all available modalities into the shared embedding space.

        Returns a dict of embeddings, with None for missing modalities.
        """
        embeddings = {}
        embeddings["text"] = self.text_encoder.encode(text_tokens) if text_tokens is not None else None
        embeddings["image"] = self.image_encoder.encode(image) if image is not None else None
        embeddings["audio"] = self.audio_encoder.encode(audio_spectrogram) if audio_spectrogram is not None else None
        return embeddings

    def apply_modality_dropout(
        self, embeddings: dict[str, Optional[np.ndarray]], training: bool = True
    ) -> dict[str, Optional[np.ndarray]]:
        """Randomly zero out entire modalities during training.

        This forces the model to not rely on any single modality and handles
        missing inputs gracefully at inference time.

        Key constraint: never drop ALL modalities — at least one must remain.
        """
        if not training:
            return embeddings

        result = {}
        available = [k for k, v in embeddings.items() if v is not None]

        if len(available) <= 1:
            return embeddings

        dropped = []
        for key in embeddings:
            if embeddings[key] is not None and np.random.random() < self.modality_dropout_rate:
                dropped.append(key)

        # Ensure at least one modality survives
        if len(dropped) == len(available):
            dropped.pop()  # Keep one random modality

        for key in embeddings:
            if key in dropped:
                result[key] = np.zeros_like(embeddings[key])
            else:
                result[key] = embeddings[key]

        return result

    def predict(
        self,
        text_tokens: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        audio_spectrogram: Optional[np.ndarray] = None,
        fusion_strategy: str = "early",
        training: bool = False,
    ) -> dict:
        """Full forward pass: encode → dropout → fuse → classify.

        Args:
            text_tokens: Token IDs for text input
            image: Image array (channels, height, width)
            audio_spectrogram: Spectrogram (freq_bins, time_frames)
            fusion_strategy: "early", "late", or "cross_attention"
            training: Whether to apply modality dropout

        Returns:
            Dict with logits, probabilities, predicted class, and per-modality embeddings
        """
        # Step 1: Encode each modality
        embeddings = self.encode_all(text_tokens, image, audio_spectrogram)

        # Step 2: Apply modality dropout during training
        embeddings = self.apply_modality_dropout(embeddings, training=training)

        # Replace None embeddings with zero vectors for fusion
        emb_list = []
        for key in ["text", "image", "audio"]:
            if embeddings[key] is not None:
                emb_list.append(embeddings[key])
            else:
                emb_list.append(np.zeros(self.shared_dim))

        # Step 3: Fuse and classify
        attn_weights = None
        if fusion_strategy == "early":
            logits = self.early_fusion.fuse(emb_list)
        elif fusion_strategy == "late":
            logits = self.late_fusion.fuse(emb_list)
        elif fusion_strategy == "cross_attention":
            # Text attends to image (most common setup, like image captioning)
            logits, attn_weights = self.cross_attention.fuse(emb_list[0], emb_list[1])
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        probs = softmax(logits)
        predicted_class = int(np.argmax(probs))

        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": predicted_class,
            "embeddings": embeddings,
            "attention_weights": attn_weights,
        }


# =============================================================================
# Contrastive Learning (CLIP-style alignment)
# =============================================================================

def contrastive_loss(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    temperature: float = 0.07,
) -> dict:
    """Compute CLIP-style symmetric contrastive loss.

    This loss pulls matching (text, image) pairs together in embedding space
    and pushes non-matching pairs apart.

    Args:
        text_embeddings: shape (batch_size, embed_dim) — L2 normalized
        image_embeddings: shape (batch_size, embed_dim) — L2 normalized
        temperature: controls sharpness of the softmax distribution.
            Lower τ → more confident matching → sharper gradients.
            Too low → training instability. Too high → weak signal.
            0.07 is CLIP's default, tuned via hyperparameter search.

    Returns:
        Dict with loss value, similarity matrix, and per-sample losses
    """
    batch_size = text_embeddings.shape[0]

    # L2 normalize embeddings — cosine similarity = dot product of unit vectors
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    image_norm = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8)

    # Similarity matrix: (batch_size, batch_size)
    # sim[i, j] = cosine_similarity(text_i, image_j) / temperature
    # Diagonal entries are matching pairs, off-diagonal are negatives
    similarity = (text_norm @ image_norm.T) / temperature

    # Labels: the diagonal — text_i should match image_i
    labels = np.arange(batch_size)

    # Image-to-text loss: for each image, which text matches?
    # This is cross-entropy where the "correct class" is the diagonal
    log_probs_i2t = similarity - np.log(np.sum(np.exp(similarity), axis=1, keepdims=True) + 1e-8)
    loss_i2t = -np.mean([log_probs_i2t[i, labels[i]] for i in range(batch_size)])

    # Text-to-image loss: for each text, which image matches?
    log_probs_t2i = similarity.T - np.log(np.sum(np.exp(similarity.T), axis=1, keepdims=True) + 1e-8)
    loss_t2i = -np.mean([log_probs_t2i[i, labels[i]] for i in range(batch_size)])

    # Symmetric loss — both directions matter equally
    total_loss = (loss_i2t + loss_t2i) / 2.0

    return {
        "loss": total_loss,
        "loss_i2t": loss_i2t,
        "loss_t2i": loss_t2i,
        "similarity_matrix": similarity * temperature,  # Return un-scaled for interpretability
    }


# =============================================================================
# Demonstration
# =============================================================================

def generate_synthetic_data(rng: np.random.RandomState, num_samples: int = 8) -> list[dict]:
    """Generate synthetic multi-modal data for demonstration.

    Creates aligned (text, image, audio) samples where each "class" has
    distinct patterns in each modality. This simulates the real scenario
    where different modalities carry correlated information about the same concept.
    """
    samples = []
    num_classes = 5
    vocab_size = 100

    for i in range(num_samples):
        label = i % num_classes

        # Text: token IDs clustered by class — tokens 20*label to 20*(label+1)
        # This simulates vocabulary that's associated with specific concepts
        text_tokens = rng.randint(20 * label, 20 * (label + 1), size=10)

        # Image: 3-channel 8x8 image with class-dependent mean intensity
        # Simulates visual features that correlate with the class
        image = rng.randn(3, 8, 8) * 0.5 + label * 0.3

        # Audio: spectrogram with class-dependent frequency pattern
        # Different classes have energy in different frequency bands
        spectrogram = rng.randn(16, 20) * 0.3
        spectrogram[label * 3:(label + 1) * 3, :] += 2.0  # Class-specific frequency band

        samples.append({
            "text_tokens": text_tokens,
            "image": image,
            "audio": spectrogram,
            "label": label,
        })

    return samples


def main():
    np.random.seed(42)
    rng = np.random.RandomState(42)

    print("=" * 70)
    print("DAY 62: MULTI-MODAL MODEL INTEGRATION")
    print("=" * 70)

    # =========================================================================
    # 1. Initialize the system
    # =========================================================================
    print("\n[1] Initializing multi-modal system...")
    config = {
        "shared_dim": 32,
        "num_classes": 5,
        "vocab_size": 100,
        "image_channels": 3,
        "freq_bins": 16,
        "modality_dropout_rate": 0.3,
    }
    system = MultiModalSystem(config)
    print(f"    Shared embedding dimension: {config['shared_dim']}")
    print(f"    Number of classes: {config['num_classes']}")
    print(f"    Modality dropout rate: {config['modality_dropout_rate']}")

    # =========================================================================
    # 2. Generate synthetic data
    # =========================================================================
    print("\n[2] Generating synthetic multi-modal data...")
    samples = generate_synthetic_data(rng, num_samples=8)
    print(f"    Generated {len(samples)} samples across {config['num_classes']} classes")
    print(f"    Text: {samples[0]['text_tokens'].shape} token IDs per sample")
    print(f"    Image: {samples[0]['image'].shape} (C, H, W)")
    print(f"    Audio: {samples[0]['audio'].shape} (freq_bins, time_frames)")

    # =========================================================================
    # 3. Encode and compare modalities
    # =========================================================================
    print("\n[3] Encoding samples and computing cross-modal similarities...")

    text_embeddings = []
    image_embeddings = []
    for s in samples:
        embs = system.encode_all(s["text_tokens"], s["image"], s["audio"])
        text_embeddings.append(embs["text"])
        image_embeddings.append(embs["image"])

    text_embeddings = np.array(text_embeddings)
    image_embeddings = np.array(image_embeddings)

    # Show cross-modal similarity for first few samples
    print("\n    Cross-modal cosine similarity (text ↔ image):")
    print("    " + "".join(f"{'Img'+str(j):>8}" for j in range(4)))
    for i in range(4):
        sims = [cosine_similarity(text_embeddings[i], image_embeddings[j]) for j in range(4)]
        row = "".join(f"{s:8.3f}" for s in sims)
        print(f"    Txt{i} {row}")

    # =========================================================================
    # 4. Compare fusion strategies
    # =========================================================================
    print("\n[4] Comparing fusion strategies on sample 0...")
    sample = samples[0]

    for strategy in ["early", "late", "cross_attention"]:
        result = system.predict(
            text_tokens=sample["text_tokens"],
            image=sample["image"],
            audio_spectrogram=sample["audio"],
            fusion_strategy=strategy,
        )
        print(f"\n    {strategy.upper()} fusion:")
        print(f"      Predicted class: {result['predicted_class']} (true: {sample['label']})")
        print(f"      Probabilities: {np.array2string(result['probabilities'], precision=3)}")

        if result["attention_weights"] is not None:
            print(f"      Attention weights (text→image): {np.array2string(result['attention_weights'], precision=3)}")

    # =========================================================================
    # 5. Contrastive loss computation
    # =========================================================================
    print("\n[5] Computing contrastive loss (CLIP-style alignment)...")

    cl_result = contrastive_loss(text_embeddings, image_embeddings, temperature=0.07)
    print(f"    Symmetric contrastive loss: {cl_result['loss']:.4f}")
    print(f"    Image→Text loss: {cl_result['loss_i2t']:.4f}")
    print(f"    Text→Image loss: {cl_result['loss_t2i']:.4f}")

    print("\n    Similarity matrix (text × image):")
    sim_matrix = cl_result["similarity_matrix"]
    print("    " + "".join(f"{'Img'+str(j):>7}" for j in range(min(6, len(samples)))))
    for i in range(min(6, len(samples))):
        row = "".join(f"{sim_matrix[i, j]:7.3f}" for j in range(min(6, len(samples))))
        print(f"    Txt{i} {row}")
    print("    (Diagonal = matching pairs, should ideally be highest in each row/col)")

    # =========================================================================
    # 6. Modality dropout robustness test
    # =========================================================================
    print("\n[6] Testing modality dropout robustness...")

    sample = samples[0]

    # Full input
    full_result = system.predict(
        sample["text_tokens"], sample["image"], sample["audio"],
        fusion_strategy="early",
    )

    # Missing image
    no_image = system.predict(
        sample["text_tokens"], None, sample["audio"],
        fusion_strategy="early",
    )

    # Missing audio
    no_audio = system.predict(
        sample["text_tokens"], sample["image"], None,
        fusion_strategy="early",
    )

    # Only text
    only_text = system.predict(
        sample["text_tokens"], None, None,
        fusion_strategy="early",
    )

    print(f"    All modalities → class {full_result['predicted_class']}, "
          f"probs: {np.array2string(full_result['probabilities'], precision=3)}")
    print(f"    No image       → class {no_image['predicted_class']}, "
          f"probs: {np.array2string(no_image['probabilities'], precision=3)}")
    print(f"    No audio       → class {no_audio['predicted_class']}, "
          f"probs: {np.array2string(no_audio['probabilities'], precision=3)}")
    print(f"    Text only      → class {only_text['predicted_class']}, "
          f"probs: {np.array2string(only_text['probabilities'], precision=3)}")

    # =========================================================================
    # 7. Cross-attention visualization
    # =========================================================================
    print("\n[7] Cross-attention analysis (text attending to image regions)...")

    for i in range(3):
        result = system.predict(
            samples[i]["text_tokens"], samples[i]["image"], samples[i]["audio"],
            fusion_strategy="cross_attention",
        )
        attn = result["attention_weights"]
        top_region = np.argmax(attn)
        print(f"    Sample {i} (class {samples[i]['label']}): "
              f"attention weights = {np.array2string(attn, precision=3)}, "
              f"peak region = {top_region}")

    # =========================================================================
    # 8. Embedding space analysis
    # =========================================================================
    print("\n[8] Embedding space analysis...")

    # Compute intra-class and inter-class similarities
    all_embeddings = []
    all_labels = []
    for s in samples:
        embs = system.encode_all(s["text_tokens"], s["image"], s["audio"])
        # Combine all modality embeddings for a unified representation
        combined = np.concatenate([embs["text"], embs["image"], embs["audio"]])
        all_embeddings.append(combined)
        all_labels.append(s["label"])

    intra_sims = []
    inter_sims = []
    for i in range(len(all_embeddings)):
        for j in range(i + 1, len(all_embeddings)):
            sim = cosine_similarity(all_embeddings[i], all_embeddings[j])
            if all_labels[i] == all_labels[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    if intra_sims:
        print(f"    Intra-class similarity (same class): {np.mean(intra_sims):.4f} "
              f"(±{np.std(intra_sims):.4f})")
    print(f"    Inter-class similarity (diff class):  {np.mean(inter_sims):.4f} "
          f"(±{np.std(inter_sims):.4f})")
    if intra_sims:
        print(f"    Separation gap: {np.mean(intra_sims) - np.mean(inter_sims):.4f}")
        print("    (Positive gap = classes are separable in embedding space)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key takeaways:
1. Each modality encoder maps raw input → fixed-size embedding in shared space
2. Early fusion is simplest but most rigid; late fusion is modular;
   cross-attention is most expressive but most complex
3. Contrastive loss aligns modalities so matching pairs cluster together
4. Modality dropout makes the system robust to missing inputs
5. In production, replace toy encoders with pre-trained models (CLIP, Whisper)
   but the fusion architecture stays the same
""")


if __name__ == "__main__":
    main()
