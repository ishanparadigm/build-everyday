"""
Day 049: Fine-Tuning a Sentiment Classifier — Your Implementation

Implement a sentiment classifier that demonstrates fine-tuning mechanics:
1. A tokenizer with special tokens ([CLS], [SEP], [PAD])
2. Pre-trained embeddings with injected sentiment knowledge
3. Attention pooling over token embeddings
4. A classification head (hidden + output layers)
5. Analytical gradient computation (backward pass)
6. Training with discriminative learning rates
7. Evaluation metrics (accuracy, precision, recall, F1)

Test your implementation: python3 -m pytest tests.py
"""

import numpy as np
import warnings
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import re
from collections import Counter

np.seterr(all='ignore')


# =============================================================================
# Part 1: Tokenizer
# =============================================================================

class SimpleTokenizer:
    """
    Word-level tokenizer with special tokens [PAD], [CLS], [SEP], [UNK].

    Hint: The encode() method should produce:
    [CLS] token1 token2 ... tokenN [SEP] [PAD] [PAD] ...
    padded to max_length.
    """

    def __init__(self, max_vocab_size: int = 5000, max_length: int = 64):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.

        Hint: Reserve IDs 0-3 for special tokens, then assign IDs
        to the most frequent words in the corpus.
        """
        raise NotImplementedError("TODO: implement this")

    def _tokenize_text(self, text: str) -> List[str]:
        """Split text into lowercase word tokens using regex."""
        raise NotImplementedError("TODO: implement this")

    def encode(self, text: str) -> List[int]:
        """
        Convert text to fixed-length token ID sequence.

        Hint: Format is [CLS] words... [SEP] [PAD]...
        Truncate words to fit max_length - 2 (for CLS and SEP).
        Unknown words get the UNK token ID.
        """
        raise NotImplementedError("TODO: implement this")

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text, skipping special tokens."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 2: Pre-trained Embeddings
# =============================================================================

class PretrainedEmbeddings:
    """
    Simulated pre-trained word embeddings.

    Hint: Initialize with small random values (scale = 1/sqrt(dim)).
    Pad token (ID 0) should be all zeros.
    inject_sentiment_knowledge() pushes positive/negative words apart
    in embedding space to simulate what pre-training learns.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        raise NotImplementedError("TODO: initialize self.embeddings (vocab_size x embedding_dim)")

    def inject_sentiment_knowledge(self, tokenizer: 'SimpleTokenizer') -> None:
        """
        Add sentiment signal to embeddings.

        Hint: Create sentiment direction(s) in embedding space.
        Push positive words in that direction, negative words opposite.
        Use multiple directions for richer representation.
        """
        raise NotImplementedError("TODO: implement this")

    def get_embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up embeddings: (batch, seq_len) -> (batch, seq_len, dim)."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 3: Building Blocks
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax. Hint: subtract max before exp."""
    raise NotImplementedError("TODO: implement this")


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer norm: normalize, then scale by gamma and shift by beta."""
    raise NotImplementedError("TODO: implement this")


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    raise NotImplementedError("TODO: implement this")


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of GELU for backpropagation.

    Hint: d/dx [0.5 * x * (1 + tanh(f(x)))] where f(x) = sqrt(2/pi) * (x + 0.044715x^3)
    Use product rule: 0.5 * (1 + tanh(f)) + 0.5 * x * sech^2(f) * f'(x)
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 4: Attention Pooling
# =============================================================================

class AttentionPooling:
    """
    Learned attention pooling: scores each token, then computes weighted average.

    Hint:
    1. Project each token: projected = tanh(x @ W_attn)
    2. Score each token: scores = projected @ v_attn
    3. Mask padding tokens (set scores to -1e9)
    4. Apply softmax to get weights
    5. Compute weighted sum: output = sum(weights * embeddings)
    """

    def __init__(self, embedding_dim: int, seed: int = 42):
        self.d = embedding_dim
        raise NotImplementedError("TODO: initialize W_attn and v_attn")

    def forward(self, x: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pool sequence into single vector.

        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            pooled: (batch, dim)
            attn_weights: (batch, seq_len)
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 5: Sentiment Classifier
# =============================================================================

class SentimentClassifier:
    """
    Architecture: Embeddings -> Attention Pool -> Hidden Layer -> Output

    Hint: Forward pass:
    1. Look up token embeddings
    2. Attention pooling to get single vector
    3. Layer norm
    4. Hidden layer with GELU: hidden = gelu(pooled @ W_hidden + b_hidden)
    5. Output: logits = hidden @ W_out + b_out

    Backward pass: chain rule through each layer.
    Cache all intermediates in forward() for backward().
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
        self._cache: Dict = {}
        raise NotImplementedError("TODO: initialize embeddings, attention, layer norm, hidden layer, output layer")

    def forward(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: token_ids -> (logits, pooled_representation).
        Cache intermediates for backward().
        """
        raise NotImplementedError("TODO: implement this")

    def backward(self, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass: compute gradients for all parameters.

        Hint: Chain rule from loss back through each layer:
        1. dL/d_logits = softmax(logits) - one_hot(labels)  (divided by batch_size)
        2. dL/dW_out = hidden^T @ d_logits
        3. d_hidden = d_logits @ W_out^T
        4. d_hidden_pre = d_hidden * gelu_derivative(hidden_pre)
        5. dL/dW_hidden = pooled^T @ d_hidden_pre
        6. d_pooled = d_hidden_pre @ W_hidden^T
        7. Propagate through attention into embeddings

        Return dict with keys: W_out, b_out, W_hidden, b_hidden, embeddings, W_attn, v_attn
        """
        raise NotImplementedError("TODO: implement this")

    def predict(self, token_ids: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        raise NotImplementedError("TODO: implement this")

    def predict_text(self, text: str) -> Tuple[str, float]:
        """Classify text. Returns (label_string, confidence)."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 6: Training
# =============================================================================

def compute_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Cross-entropy loss: CE = -mean(log(p_correct_class)).
    Hint: Clip probabilities to avoid log(0).
    """
    raise NotImplementedError("TODO: implement this")


def clip_gradients(grads: Dict[str, np.ndarray], max_norm: float = 5.0) -> Dict[str, np.ndarray]:
    """Scale all gradients down if total norm exceeds max_norm."""
    raise NotImplementedError("TODO: implement this")


def train_epoch(
    model: SentimentClassifier,
    train_data: List[Tuple[str, int]],
    lr_head: float = 1e-2,
    lr_backbone: float = 1e-3,
    batch_size: int = 16,
) -> float:
    """
    Train for one epoch with discriminative learning rates.

    Hint: The head (W_hidden, b_hidden, W_out, b_out) uses lr_head.
    The backbone (embeddings, attention) uses lr_backbone.
    Shuffle data, iterate in batches, forward, backward, update.
    Keep pad embedding at zero after updates.
    """
    raise NotImplementedError("TODO: implement this")


def evaluate(
    model: SentimentClassifier,
    data: List[Tuple[str, int]],
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate model. Returns dict with accuracy, loss, precision, recall, f1."""
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Part 7: Dataset
# =============================================================================

def create_sentiment_dataset() -> Tuple[
    List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]
]:
    """Create train/val/test splits. Label 0=negative, 1=positive."""
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Main — test your implementation
# =============================================================================

if __name__ == "__main__":
    print("Building dataset and tokenizer...")
    train_data, val_data, test_data = create_sentiment_dataset()
    print(f"Dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    tokenizer = SimpleTokenizer(max_vocab_size=2000, max_length=32)
    all_texts = [t[0] for t in train_data + val_data + test_data]
    tokenizer.build_vocab(all_texts)
    print(f"Vocab size: {tokenizer.vocab_size}")

    print("\nInitializing model...")
    model = SentimentClassifier(tokenizer=tokenizer, embedding_dim=64, hidden_dim=32)

    print("Evaluating before fine-tuning...")
    baseline = evaluate(model, test_data)
    print(f"Baseline accuracy: {baseline['accuracy']:.1%}")

    print("\nFine-tuning for 5 epochs...")
    for epoch in range(5):
        loss = train_epoch(model, train_data, lr_head=0.05, lr_backbone=0.005, batch_size=8)
        val_metrics = evaluate(model, val_data)
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, val_acc={val_metrics['accuracy']:.1%}")

    print("\nFinal test evaluation...")
    test_metrics = evaluate(model, test_data)
    print(f"Test accuracy: {test_metrics['accuracy']:.1%}")
    print(f"Test F1: {test_metrics['f1']:.3f}")

    print("\nInference on new examples:")
    examples = [
        "This is absolutely amazing I love it",
        "Terrible product worst purchase ever",
        "It was okay nothing special",
    ]
    for text in examples:
        label, conf = model.predict_text(text)
        print(f"  [{label} {conf:.0%}] {text}")
