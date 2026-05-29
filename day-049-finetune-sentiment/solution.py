"""
Day 049: Fine-Tuning a Sentiment Classifier

A complete implementation of fine-tuning a pre-trained model for binary
sentiment classification. Built from scratch with numpy — no PyTorch/
TensorFlow/HuggingFace required.

This demonstrates the core mechanics of fine-tuning:
1. Pre-trained embeddings that capture language understanding
2. A classification head added on top
3. End-to-end training with analytical gradients
4. Learning rate sensitivity — too small (no learning), too large (catastrophic forgetting)
5. Proper evaluation with train/val/test splits

Architecture: Pre-trained Embeddings → Attention Pooling → Hidden Layer → Classifier
This mirrors the structure of real fine-tuning (BERT [CLS] → Linear) while being
fully trainable with numpy and demonstrating every concept clearly.
"""

import numpy as np
import warnings
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import re
from collections import Counter
import copy

# Suppress numpy warnings from padding token operations (all-zero embeddings
# can produce NaN in intermediate matmul results; these are masked out by
# the attention mask before they affect the final output)
np.seterr(all='ignore')


# =============================================================================
# Part 1: Tokenizer — Word-level with special tokens
# =============================================================================

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens, mimicking the interface
    of real subword tokenizers (BERT's WordPiece, GPT's BPE).

    In production, you'd use a pre-trained tokenizer that exactly matches
    the model's vocabulary. Here we build one from the training corpus.
    """

    def __init__(self, max_vocab_size: int = 5000, max_length: int = 64):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        # Special tokens — these have reserved IDs
        self.pad_token = "[PAD]"    # Padding for fixed-length sequences
        self.cls_token = "[CLS]"    # Classification token (sequence summary)
        self.sep_token = "[SEP]"    # Separator / end of sequence
        self.unk_token = "[UNK]"    # Unknown / out-of-vocabulary words

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from corpus. Most frequent words get lowest IDs."""
        special_tokens = [self.pad_token, self.cls_token, self.sep_token, self.unk_token]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        word_counts: Counter = Counter()
        for text in texts:
            words = self._tokenize_text(text)
            word_counts.update(words)

        most_common = word_counts.most_common(self.max_vocab_size - len(special_tokens))
        for word, _ in most_common:
            idx = len(self.token_to_id)
            self.token_to_id[word] = idx
            self.id_to_token[idx] = word

        self.vocab_size = len(self.token_to_id)

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        text = text.lower().strip()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs: [CLS] words... [SEP] [PAD]...

        This mirrors BERT's input format. The [CLS] token's final hidden
        state becomes the sequence representation for classification.
        """
        words = self._tokenize_text(text)
        unk_id = self.token_to_id[self.unk_token]
        word_ids = [self.token_to_id.get(w, unk_id) for w in words]

        cls_id = self.token_to_id[self.cls_token]
        sep_id = self.token_to_id[self.sep_token]
        token_ids = [cls_id] + word_ids[:self.max_length - 2] + [sep_id]

        pad_id = self.token_to_id[self.pad_token]
        while len(token_ids) < self.max_length:
            token_ids.append(pad_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text (for debugging)."""
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, self.unk_token)
            if token in (self.pad_token, self.cls_token, self.sep_token):
                continue
            tokens.append(token)
        return " ".join(tokens)


# =============================================================================
# Part 2: Pre-trained Embeddings — Simulating transfer learning
# =============================================================================

class PretrainedEmbeddings:
    """
    Simulates pre-trained word embeddings that capture semantic relationships.

    In real fine-tuning, these come from BERT/GPT pre-training on billions of
    words. Here we create embeddings that encode sentiment-relevant features
    to demonstrate how pre-training gives the model a head start.

    The key insight: pre-trained embeddings already "know" that "great" and
    "excellent" are similar, and "terrible" and "awful" are similar. Fine-tuning
    just needs to learn that the first cluster → positive, second → negative.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        rng = np.random.RandomState(seed)
        # Initialize with small random values. The variance is set so that
        # downstream dot products are O(1), preventing gradient explosion.
        scale = 1.0 / np.sqrt(embedding_dim)
        self.embeddings = rng.randn(vocab_size, embedding_dim) * scale

        # Pad token should be all zeros (carries no information)
        self.embeddings[0] = 0.0

    def inject_sentiment_knowledge(self, tokenizer: 'SimpleTokenizer') -> None:
        """
        Simulate pre-training by encoding sentiment structure into embeddings.

        We add a "sentiment direction" to the embedding space — positive words
        get pushed in one direction, negative words in the opposite direction.
        We also add similarity structure so related words cluster together.
        """
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "loved", "best", "perfect", "beautiful", "enjoy", "enjoyed",
            "brilliant", "outstanding", "superb", "impressive", "delightful",
            "happy", "pleased", "recommend", "favorite", "fun", "exciting",
            "nice", "fine", "pleasant", "satisfying", "remarkable", "like",
            "incredible", "spectacular", "restored", "faith", "exceeded",
            "entertaining", "sharp", "honors", "masterpiece", "feel",
        }

        negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "poor", "boring",
            "hate", "hated", "waste", "stupid", "ugly", "disappointing",
            "dull", "mediocre", "annoying", "painful", "weak", "fails",
            "sad", "angry", "avoid", "disaster", "rubbish", "trash",
            "worse", "broken", "useless", "pathetic", "dreadful", "dislike",
            "miserable", "failed", "catastrophic", "garbage", "bored",
            "frustrated", "ruins", "laughable", "pointless", "lifeless",
            "lazy", "poorly", "disappointed",
        }

        # Create multiple sentiment-related axes in embedding space
        # This is richer than a single direction — real pre-training creates
        # complex semantic manifolds, not just one axis
        rng = np.random.RandomState(99)
        n_sentiment_dims = min(8, self.embedding_dim)
        sentiment_directions = rng.randn(n_sentiment_dims, self.embedding_dim)
        # Normalize each direction
        for i in range(n_sentiment_dims):
            sentiment_directions[i] /= np.linalg.norm(sentiment_directions[i])

        strength = 0.3

        for word in positive_words:
            if word in tokenizer.token_to_id:
                idx = tokenizer.token_to_id[word]
                for d in range(n_sentiment_dims):
                    self.embeddings[idx] += strength * sentiment_directions[d]

        for word in negative_words:
            if word in tokenizer.token_to_id:
                idx = tokenizer.token_to_id[word]
                for d in range(n_sentiment_dims):
                    self.embeddings[idx] -= strength * sentiment_directions[d]

    def get_embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up embeddings: (batch, seq_len) → (batch, seq_len, dim)."""
        return self.embeddings[token_ids].copy()


# =============================================================================
# Part 3: Building Blocks — Activations and Normalization
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax. Subtract max to prevent overflow."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Layer normalization — normalizes across feature dimension.
    Used in transformers because it works with variable sequence lengths
    and is independent of batch size (unlike BatchNorm).
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation — the standard for transformers (BERT, GPT).
    Smoother than ReLU, allows small negative values through.
    GELU(x) ≈ x * Phi(x) where Phi is the Gaussian CDF.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of GELU for backpropagation."""
    # Using the approximation: d/dx GELU(x) ≈ Phi(x) + x * phi(x)
    # where Phi is CDF and phi is PDF of standard normal
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    # Approximate derivative of tanh part
    inner = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
    tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    sech2 = 1 - np.tanh(tanh_arg) ** 2
    return 0.5 * (1 + np.tanh(tanh_arg)) + 0.5 * x * sech2 * inner


# =============================================================================
# Part 4: Self-Attention Pooling — How the model reads a sentence
# =============================================================================

class AttentionPooling:
    """
    Learned attention pooling over token embeddings.

    Instead of simple mean pooling, this learns WHICH tokens matter most
    for the classification task. This is analogous to how BERT's [CLS] token
    learns to aggregate information through self-attention layers.

    The attention mechanism: each token embedding is scored by a learned
    query vector, then we compute a weighted average. Tokens that are
    more relevant to the task get higher weights.

    attn_score_i = tanh(x_i @ W_attn) @ v_attn
    weights = softmax(scores) * mask
    output = sum(weights_i * x_i)
    """

    def __init__(self, embedding_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.d = embedding_dim

        # Attention parameters — a small network that scores each token
        # W_attn: project embedding to attention space
        # v_attn: score vector that produces a scalar from the projection
        scale = 1.0 / np.sqrt(embedding_dim)
        self.W_attn = rng.randn(embedding_dim, embedding_dim) * scale
        self.v_attn = rng.randn(embedding_dim) * scale

    def forward(self, x: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pool a sequence of embeddings into a single vector using attention.

        Args:
            x: (batch, seq_len, dim) — token embeddings
            mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            pooled: (batch, dim) — weighted average of token embeddings
            attn_weights: (batch, seq_len) — attention weights (for inspection)
        """
        # Project each token, then score
        projected = np.tanh(x @ self.W_attn)            # (B, T, D)
        scores = projected @ self.v_attn                  # (B, T)

        # Mask out padding tokens (set their scores to -inf before softmax)
        scores = scores + (1 - mask) * (-1e9)

        attn_weights = softmax(scores, axis=-1)           # (B, T)
        attn_weights = attn_weights * mask                # Zero out padding

        # Weighted sum of embeddings
        pooled = np.einsum('bt,btd->bd', attn_weights, x)  # (B, D)

        return pooled, attn_weights


# =============================================================================
# Part 5: The Sentiment Classifier
# =============================================================================

class SentimentClassifier:
    """
    A sentiment classifier that demonstrates fine-tuning mechanics.

    Architecture:
        Pre-trained Embeddings → Attention Pooling → Hidden Layer → Classifier

    The model has two conceptual parts:
    1. "Pre-trained backbone" (embeddings + attention pooling) — initialized with
       weights that capture language understanding. Updated with a SMALL lr.
    2. "Classification head" (hidden + output layers) — randomly initialized.
       Learns to map pooled representation to sentiment labels. Can use larger lr.

    During fine-tuning, BOTH parts are updated, but with different rates.
    This is called "discriminative fine-tuning" — the key insight that makes
    transfer learning work in practice.
    """

    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        embedding_dim: int = 64,
        hidden_dim: int = 32,
        n_classes: int = 2,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        rng = np.random.RandomState(seed)

        # --- Pre-trained backbone ---
        self.embeddings = PretrainedEmbeddings(
            tokenizer.vocab_size, embedding_dim, seed=seed
        )
        self.embeddings.inject_sentiment_knowledge(tokenizer)

        # Attention pooling (part of backbone)
        self.attention = AttentionPooling(embedding_dim, seed=seed)

        # Layer norm before classification head
        self.ln_gamma = np.ones(embedding_dim)
        self.ln_beta = np.zeros(embedding_dim)

        # --- Classification head (randomly initialized) ---
        # Hidden layer: embedding_dim → hidden_dim
        scale_h = np.sqrt(2.0 / (embedding_dim + hidden_dim))
        self.W_hidden = rng.randn(embedding_dim, hidden_dim) * scale_h
        self.b_hidden = np.zeros(hidden_dim)

        # Output layer: hidden_dim → n_classes
        scale_o = np.sqrt(2.0 / (hidden_dim + n_classes))
        self.W_out = rng.randn(hidden_dim, n_classes) * scale_o
        self.b_out = np.zeros(n_classes)

        # Cache for backward pass
        self._cache: Dict = {}

    def forward(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: token_ids → (logits, pooled_representation).

        We cache all intermediate values for the backward pass.
        This is what autograd does automatically in PyTorch — here we do
        it manually to make the gradient flow explicit and educational.
        """
        B, T = token_ids.shape
        pad_id = self.tokenizer.token_to_id[self.tokenizer.pad_token]
        mask = (token_ids != pad_id).astype(np.float64)  # (B, T)

        # Step 1: Look up embeddings
        x = self.embeddings.get_embeddings(token_ids)     # (B, T, D)

        # Step 2: Attention pooling → single vector per sequence
        pooled, attn_weights = self.attention.forward(x, mask)  # (B, D)

        # Step 3: Layer normalization
        pooled_normed = layer_norm(pooled, self.ln_gamma, self.ln_beta)

        # Step 4: Hidden layer with GELU activation
        hidden_pre = pooled_normed @ self.W_hidden + self.b_hidden  # (B, H)
        hidden = gelu(hidden_pre)                                    # (B, H)

        # Step 5: Output layer → logits
        logits = hidden @ self.W_out + self.b_out  # (B, C)

        # Cache everything for backward pass
        self._cache = {
            'token_ids': token_ids,
            'mask': mask,
            'x': x,
            'pooled': pooled,
            'attn_weights': attn_weights,
            'pooled_normed': pooled_normed,
            'hidden_pre': hidden_pre,
            'hidden': hidden,
            'logits': logits,
        }

        return logits, pooled_normed

    def backward(self, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass: compute gradients for ALL parameters analytically.

        This is the heart of fine-tuning — the gradient flows from the loss
        back through the classification head AND into the pre-trained embeddings.
        Understanding this gradient flow is key to understanding why fine-tuning
        works and why learning rate matters so much.

        The chain rule gives us:
        dL/dW_out = hidden^T @ d_logits
        dL/dW_hidden = pooled^T @ (d_logits @ W_out^T * gelu'(hidden_pre))
        dL/d_embeddings = ... (flows back through attention and hidden layers)
        """
        cache = self._cache
        B = labels.shape[0]

        # --- Step 1: Gradient of cross-entropy loss w.r.t. logits ---
        # For CE loss with softmax: dL/d_logits = probs - one_hot_labels
        # This elegant formula is why softmax + CE is so popular
        probs = softmax(cache['logits'], axis=-1)       # (B, C)
        d_logits = probs.copy()
        d_logits[np.arange(B), labels] -= 1.0
        d_logits /= B  # Average over batch

        # --- Step 2: Gradient through output layer ---
        # logits = hidden @ W_out + b_out
        # dL/dW_out = hidden^T @ d_logits
        d_W_out = cache['hidden'].T @ d_logits           # (H, C)
        d_b_out = np.sum(d_logits, axis=0)               # (C,)
        d_hidden = d_logits @ self.W_out.T               # (B, H)

        # --- Step 3: Gradient through GELU activation ---
        # hidden = gelu(hidden_pre)
        # dL/d_hidden_pre = dL/d_hidden * gelu'(hidden_pre)
        d_hidden_pre = d_hidden * gelu_derivative(cache['hidden_pre'])  # (B, H)

        # --- Step 4: Gradient through hidden layer ---
        # hidden_pre = pooled_normed @ W_hidden + b_hidden
        d_W_hidden = cache['pooled_normed'].T @ d_hidden_pre  # (D, H)
        d_b_hidden = np.sum(d_hidden_pre, axis=0)             # (H,)
        d_pooled = d_hidden_pre @ self.W_hidden.T             # (B, D)

        # --- Step 5: Gradient through attention pooling into embeddings ---
        # pooled = sum(attn_weights * x, axis=seq)
        # The gradient flows through the weighted sum into each token embedding
        # Tokens with higher attention get more gradient (they contributed more)
        attn_w = cache['attn_weights']  # (B, T)
        token_ids = cache['token_ids']
        x = cache['x']

        # d_pooled is (B, D), attn_weights is (B, T)
        # dx_i = attn_weight_i * d_pooled  (gradient to each token embedding)
        d_x = np.einsum('bt,bd->btd', attn_w, d_pooled)  # (B, T, D)

        # Accumulate gradients into embedding matrix
        # This is the key part: we're updating pre-trained embeddings
        d_embeddings = np.zeros_like(self.embeddings.embeddings)
        for b in range(B):
            for t in range(token_ids.shape[1]):
                tid = token_ids[b, t]
                if cache['mask'][b, t] > 0:
                    d_embeddings[tid] += d_x[b, t]

        # --- Step 6: Gradient for attention parameters ---
        # We also update the attention pooling weights
        # This is a simplified gradient (ignoring the tanh derivative for clarity)
        d_W_attn = np.zeros_like(self.attention.W_attn)
        d_v_attn = np.zeros_like(self.attention.v_attn)

        # The full gradient through attention is complex (involves softmax Jacobian)
        # We compute a simplified but effective version:
        # The attention should focus on tokens whose embeddings point in the
        # direction that reduces the loss
        for b in range(B):
            for t in range(token_ids.shape[1]):
                if cache['mask'][b, t] > 0:
                    # How much does increasing this token's attention help?
                    benefit = x[b, t] @ d_pooled[b]
                    # Update attention to focus on beneficial tokens
                    proj = np.tanh(x[b, t] @ self.attention.W_attn)
                    d_v_attn += benefit * proj / B
                    # Gradient for W_attn (outer product, simplified)
                    d_W_attn += np.outer(x[b, t], benefit * self.attention.v_attn * (1 - proj**2)) / B

        return {
            'W_out': d_W_out,
            'b_out': d_b_out,
            'W_hidden': d_W_hidden,
            'b_hidden': d_b_hidden,
            'embeddings': d_embeddings,
            'W_attn': d_W_attn,
            'v_attn': d_v_attn,
        }

    def predict(self, token_ids: np.ndarray) -> np.ndarray:
        """Get predicted class probabilities."""
        logits, _ = self.forward(token_ids)
        return softmax(logits, axis=-1)

    def predict_text(self, text: str) -> Tuple[str, float]:
        """Classify a single text string."""
        token_ids = np.array([self.tokenizer.encode(text)])
        probs = self.predict(token_ids)[0]
        label = "positive" if np.argmax(probs) == 1 else "negative"
        confidence = float(np.max(probs))
        return label, confidence


# =============================================================================
# Part 6: Training — The fine-tuning loop with discriminative learning rates
# =============================================================================

def compute_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Cross-entropy loss — the standard objective for classification.

    CE(y, y_hat) = -mean(log(p_correct_class))

    We clip probabilities to avoid log(0) = -inf.
    """
    probs = softmax(logits, axis=-1)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    batch_size = labels.shape[0]
    correct_probs = probs[np.arange(batch_size), labels]
    return float(-np.mean(np.log(correct_probs)))


def train_epoch(
    model: SentimentClassifier,
    train_data: List[Tuple[str, int]],
    lr_head: float = 1e-2,
    lr_backbone: float = 1e-3,
    batch_size: int = 16,
) -> float:
    """
    Train for one epoch with discriminative learning rates.

    KEY INSIGHT: The classification head (randomly initialized) needs a
    LARGER learning rate to learn quickly. The pre-trained backbone needs
    a SMALLER learning rate to preserve useful representations.

    This is "discriminative fine-tuning" — different learning rates for
    different parts of the model. It's crucial for effective fine-tuning.

    Returns average loss for the epoch.
    """
    indices = np.random.permutation(len(train_data))
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(train_data), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_texts = [train_data[i][0] for i in batch_idx]
        batch_labels = np.array([train_data[i][1] for i in batch_idx])

        token_ids = np.array([model.tokenizer.encode(t) for t in batch_texts])

        # Forward pass
        logits, _ = model.forward(token_ids)
        loss = compute_loss(logits, batch_labels)
        total_loss += loss
        n_batches += 1

        # Backward pass — compute analytical gradients
        grads = model.backward(batch_labels)
        grads = clip_gradients(grads, max_norm=5.0)

        # Update classification head with LARGER learning rate
        model.W_out -= lr_head * grads['W_out']
        model.b_out -= lr_head * grads['b_out']
        model.W_hidden -= lr_head * grads['W_hidden']
        model.b_hidden -= lr_head * grads['b_hidden']

        # Update backbone with SMALLER learning rate
        # This is the essence of fine-tuning: gentle updates to pre-trained weights
        model.embeddings.embeddings -= lr_backbone * grads['embeddings']
        model.attention.W_attn -= lr_backbone * grads['W_attn']
        model.attention.v_attn -= lr_backbone * grads['v_attn']

        # Keep pad embedding at zero
        model.embeddings.embeddings[0] = 0.0

    return total_loss / max(n_batches, 1)


def evaluate(
    model: SentimentClassifier,
    data: List[Tuple[str, int]],
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    Returns accuracy, loss, precision, recall, and F1 score.
    """
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        texts = [t[0] for t in batch]
        labels = np.array([t[1] for t in batch])

        token_ids = np.array([model.tokenizer.encode(t) for t in texts])
        logits, _ = model.forward(token_ids)

        loss = compute_loss(logits, labels)
        total_loss += loss
        n_batches += 1

        preds = np.argmax(logits, axis=-1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float(np.mean(all_preds == all_labels))

    tp = int(np.sum((all_preds == 1) & (all_labels == 1)))
    fp = int(np.sum((all_preds == 1) & (all_labels == 0)))
    fn = int(np.sum((all_preds == 0) & (all_labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "loss": total_loss / max(n_batches, 1),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# =============================================================================
# Part 7: Dataset
# =============================================================================

def create_sentiment_dataset() -> Tuple[
    List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]
]:
    """
    Create a sentiment classification dataset with train/val/test splits.
    Label 0 = negative, Label 1 = positive.
    """
    positive_examples = [
        "This movie was absolutely brilliant and I loved every minute",
        "The acting was superb and the story was deeply moving",
        "I would highly recommend this to everyone I know",
        "What a fantastic film with amazing performances throughout",
        "The best movie I have seen in years truly outstanding",
        "Beautiful cinematography and a wonderful emotional journey",
        "I enjoyed this so much and the ending was perfect",
        "An excellent production with great attention to detail",
        "This was fun exciting and kept me engaged the whole time",
        "The director did an impressive job with this masterpiece",
        "I loved the characters and the story was very satisfying",
        "A delightful experience from beginning to end",
        "The music was beautiful and enhanced every scene perfectly",
        "One of the best films of the year highly recommended",
        "A remarkable achievement in storytelling and visual art",
        "The performances were outstanding and the writing was sharp",
        "I was pleasantly surprised by how good this turned out",
        "A great movie that the whole family can enjoy together",
        "The plot was exciting and full of wonderful surprises",
        "This film restored my faith in good cinema absolutely loved it",
        "Incredible acting and a story that stays with you",
        "A feel good movie with excellent production values",
        "The humor was perfect and the dramatic moments hit hard",
        "This is exactly the kind of movie I love watching",
        "A spectacular achievement that deserves all the praise",
        "The script was brilliant and every actor delivered perfectly",
        "I was thoroughly impressed and entertained throughout",
        "A wonderful adaptation that honors the source material",
        "The best thing I have watched this year hands down",
        "A truly enjoyable experience that exceeded my expectations",
    ]

    negative_examples = [
        "This was the worst movie I have ever had to sit through",
        "Terrible acting and a plot that made absolutely no sense",
        "I would not recommend this to anyone complete waste of time",
        "What a horrible film with no redeeming qualities at all",
        "The worst movie of the year by far truly dreadful",
        "Boring slow and completely pointless from start to finish",
        "I hated every single minute of this awful experience",
        "A terrible production with lazy writing and bad direction",
        "This was dull lifeless and a complete disappointment",
        "The director clearly had no idea what they were doing",
        "I disliked the characters and the story was weak",
        "A painful experience that I would rather forget",
        "The music was annoying and distracted from the scenes",
        "One of the worst films ever made avoid at all costs",
        "A pathetic attempt at filmmaking that fails on every level",
        "The performances were terrible and the writing was awful",
        "I was deeply disappointed by how bad this turned out",
        "A horrible movie that wastes the talent of its cast",
        "The plot was boring and full of stupid decisions",
        "This film made me angry at how poorly it was executed",
        "Mediocre acting and a story that goes absolutely nowhere",
        "A disaster of a movie with terrible production values",
        "The humor fell flat and the drama was laughable",
        "This is exactly the kind of movie I hate watching",
        "A catastrophic failure that deserves all the criticism",
        "The script was garbage and every actor looked bored",
        "I was thoroughly bored and frustrated throughout",
        "A terrible adaptation that ruins the source material",
        "The worst thing I have watched this year without question",
        "A truly miserable experience that failed all expectations",
    ]

    data = [(text, 1) for text in positive_examples] + [(text, 0) for text in negative_examples]

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(data))
    data = [data[i] for i in indices]

    n = len(data)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    return data[:train_end], data[train_end:val_end], data[val_end:]


# =============================================================================
# Part 8: Main — Full fine-tuning pipeline with analysis
# =============================================================================

def clip_gradients(grads: Dict[str, np.ndarray], max_norm: float = 5.0) -> Dict[str, np.ndarray]:
    """
    Gradient clipping — prevents exploding gradients during training.

    If the total gradient norm exceeds max_norm, scale all gradients down
    proportionally. This is essential for stable fine-tuning because:
    - Some batches produce unexpectedly large gradients
    - Without clipping, a single bad batch can destroy the model
    """
    total_norm = 0.0
    for key, grad in grads.items():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        return {key: grad * scale for key, grad in grads.items()}
    return grads


def main():
    np.random.seed(42)

    print("=" * 70)
    print("Fine-Tuning a Sentiment Classifier")
    print("=" * 70)

    # Step 1: Create dataset
    print("\n--- Step 1: Preparing Dataset ---")
    train_data, val_data, test_data = create_sentiment_dataset()
    print(f"Train: {len(train_data)} examples")
    print(f"Val:   {len(val_data)} examples")
    print(f"Test:  {len(test_data)} examples")
    print(f"\nSample training examples:")
    for text, label in train_data[:3]:
        print(f"  [{['negative', 'positive'][label]}] {text[:60]}...")

    # Step 2: Build tokenizer
    print("\n--- Step 2: Building Tokenizer ---")
    tokenizer = SimpleTokenizer(max_vocab_size=2000, max_length=32)
    all_texts = [t[0] for t in train_data + val_data + test_data]
    tokenizer.build_vocab(all_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    sample_text = train_data[0][0]
    encoded = tokenizer.encode(sample_text)
    print(f"\nTokenization example:")
    print(f"  Text:   '{sample_text[:50]}...'")
    print(f"  Tokens: {encoded[:10]}... (length {len(encoded)})")
    print(f"  Decoded: '{tokenizer.decode(encoded[:10])}...'")

    # Step 3: Initialize model
    print("\n--- Step 3: Initializing Model ---")
    model = SentimentClassifier(
        tokenizer=tokenizer, embedding_dim=64, hidden_dim=32, n_classes=2, seed=42,
    )

    n_emb = model.embeddings.embeddings.size
    n_attn = model.attention.W_attn.size + model.attention.v_attn.size
    n_head = (model.W_hidden.size + model.b_hidden.size +
              model.W_out.size + model.b_out.size)
    print(f"Total parameters: {n_emb + n_attn + n_head:,}")
    print(f"  Pre-trained backbone: {n_emb + n_attn:,} (embeddings + attention)")
    print(f"  Classification head:  {n_head:,} (hidden + output)")

    # Step 4: Evaluate BEFORE fine-tuning
    print("\n--- Step 4: Pre-Fine-Tuning Baseline ---")
    baseline = evaluate(model, test_data)
    print(f"Test accuracy BEFORE fine-tuning: {baseline['accuracy']:.1%}")
    print(f"  (Random baseline would be ~50%)")

    print(f"\nPre-training predictions on test set:")
    for text, true_label in test_data[:6]:
        pred_label, confidence = model.predict_text(text)
        correct = "+" if (pred_label == "positive") == (true_label == 1) else "-"
        print(f"  {correct} [{pred_label:>8s} {confidence:.0%}] {text[:55]}...")

    # Step 5: Fine-tune!
    print("\n--- Step 5: Fine-Tuning ---")
    print("Using discriminative learning rates:")
    print("  Classification head lr: 0.05  (needs to learn from scratch)")
    print("  Backbone lr:           0.005  (preserve pre-trained knowledge)")

    n_epochs = 15
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(n_epochs):
        train_loss = train_epoch(
            model, train_data,
            lr_head=0.05,      # Classification head learns fast
            lr_backbone=0.005,  # Backbone updates gently
            batch_size=8,
        )

        val_metrics = evaluate(model, val_data)

        marker = ""
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            # Save best model (deep copy)
            best_model_state = {
                'W_out': model.W_out.copy(),
                'b_out': model.b_out.copy(),
                'W_hidden': model.W_hidden.copy(),
                'b_hidden': model.b_hidden.copy(),
                'embeddings': model.embeddings.embeddings.copy(),
                'W_attn': model.attention.W_attn.copy(),
                'v_attn': model.attention.v_attn.copy(),
            }
            marker = " *best*"

        if epoch < 5 or (epoch + 1) % 3 == 0 or marker:
            print(f"  Epoch {epoch+1:2d}: loss={train_loss:.4f}  "
                  f"val_acc={val_metrics['accuracy']:.1%}  "
                  f"val_f1={val_metrics['f1']:.3f}{marker}")

    # Restore best model
    if best_model_state:
        model.W_out = best_model_state['W_out']
        model.b_out = best_model_state['b_out']
        model.W_hidden = best_model_state['W_hidden']
        model.b_hidden = best_model_state['b_hidden']
        model.embeddings.embeddings = best_model_state['embeddings']
        model.attention.W_attn = best_model_state['W_attn']
        model.attention.v_attn = best_model_state['v_attn']

    # Step 6: Final evaluation
    print(f"\n--- Step 6: Final Test Evaluation ---")
    test_metrics = evaluate(model, test_data)
    print(f"Test Accuracy:  {test_metrics['accuracy']:.1%}")
    print(f"Test Precision: {test_metrics['precision']:.3f}")
    print(f"Test Recall:    {test_metrics['recall']:.3f}")
    print(f"Test F1:        {test_metrics['f1']:.3f}")
    print(f"Test Loss:      {test_metrics['loss']:.4f}")

    improvement = test_metrics['accuracy'] - baseline['accuracy']
    print(f"\nImprovement from fine-tuning: {improvement:+.1%}")

    # Step 7: Inference on new examples
    print("\n--- Step 7: Inference on New Examples ---")
    test_sentences = [
        "This product is absolutely amazing and works perfectly",
        "Terrible quality and broke after one day of use",
        "Not bad but nothing special either just average",
        "I am so happy with this purchase best decision ever",
        "Complete garbage do not waste your money on this",
        "The service was excellent and the staff were very helpful",
        "Worst experience of my life never coming back here",
    ]

    print(f"\nClassifying new sentences:")
    for text in test_sentences:
        label, confidence = model.predict_text(text)
        print(f"  [{label:>8s} {confidence:.0%}] {text}")

    # Step 8: Attention analysis — what is the model looking at?
    print("\n--- Step 8: Attention Analysis ---")
    print("Which words does the model attend to for classification?")

    analysis_texts = [
        "The movie was absolutely brilliant and wonderful",
        "A terrible waste of time completely boring and awful",
    ]

    for text in analysis_texts:
        token_ids = np.array([tokenizer.encode(text)])
        model.forward(token_ids)  # Populates cache
        attn_w = model._cache['attn_weights'][0]  # (T,)

        words = tokenizer._tokenize_text(text)
        print(f"\n  '{text}'")
        print(f"  Token attention weights:")
        for i, word in enumerate(words[:10]):
            # +1 because of [CLS] token at position 0
            weight = attn_w[i + 1] if (i + 1) < len(attn_w) else 0
            bar = "#" * int(weight * 50)
            print(f"    {word:>15s}: {weight:.3f} {bar}")

    # Step 9: Learning rate sensitivity
    print("\n--- Step 9: Learning Rate Sensitivity ---")
    print("Comparing head-only vs full fine-tuning after just 3 epochs:")
    print("(Each starts from a fresh pre-trained model)\n")

    for name, lr_h, lr_b, epochs in [
        ("no training (baseline)", 0.0, 0.0, 0),
        ("head only (backbone frozen)", 0.05, 0.0, 3),
        ("full fine-tune (gentle backbone)", 0.05, 0.005, 3),
        ("full fine-tune (15 epochs)", 0.05, 0.005, 15),
    ]:
        test_model = SentimentClassifier(
            tokenizer=tokenizer, embedding_dim=64, hidden_dim=32, seed=42,
        )
        for _ in range(epochs):
            train_epoch(test_model, train_data, lr_head=lr_h, lr_backbone=lr_b, batch_size=8)
        train_m = evaluate(test_model, train_data)
        val_m = evaluate(test_model, val_data)
        print(f"  {name:>35s}: train={train_m['accuracy']:.1%}  val={val_m['accuracy']:.1%}")

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("  1. Pre-trained embeddings encode language knowledge (positive/negative clusters)")
    print("  2. Fine-tuning the classification head alone works, but updating the")
    print("     backbone too (gently!) gives the best results")
    print("  3. Discriminative learning rates are key: head learns fast, backbone adapts slowly")
    print("  4. Too aggressive backbone updates destroy pre-trained knowledge (catastrophic forgetting)")
    print("  5. Attention reveals WHICH words the model relies on for classification")
    print("=" * 70)


if __name__ == "__main__":
    main()
