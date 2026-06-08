"""
Day 068: Graph Neural Network (GCN) from Scratch — Your Implementation

Implement a Graph Convolutional Network for node classification.
The key idea: each node updates its representation by aggregating
information from its neighbors (message passing).

Run tests with: python3 -m pytest tests.py -v
"""

import numpy as np
from typing import Tuple, List, Optional


class Graph:
    """
    Represents an undirected graph with node features and optional labels.

    You need to implement adjacency normalization:
    Â = D̃^(-1/2) · Ã · D̃^(-1/2) where Ã = A + I (self-loops added)

    Hint: D̃ is the degree matrix of Ã. Think about why we need
    both the self-loops AND the symmetric normalization.
    """

    def __init__(self, adj: np.ndarray, features: np.ndarray,
                 labels: Optional[np.ndarray] = None):
        assert adj.shape[0] == adj.shape[1], "Adjacency must be square"
        assert adj.shape[0] == features.shape[0], "Node count mismatch"

        self.num_nodes = adj.shape[0]
        self.adj = adj.astype(np.float64)
        self.features = features.astype(np.float64)
        self.labels = labels

        self.adj_norm = self._normalize_adjacency(self.adj)

    def _normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
        """
        Compute normalized adjacency: Â = D̃^(-1/2) · Ã · D̃^(-1/2)

        Steps:
          1. Add self-loops: Ã = A + I
          2. Compute degree vector from Ã
          3. Compute D̃^(-1/2) (handle zero-degree nodes!)
          4. Apply symmetric normalization

        Hint: You can avoid building full diagonal matrices by using
        broadcasting: (d[:, None] * A) * d[None, :]
        """
        raise NotImplementedError("TODO: implement adjacency normalization")


class GCNLayer:
    """
    Single Graph Convolutional layer: H' = activation(Â · H · W + b)

    The forward pass is: aggregate neighbors (Â·H), transform (·W+b), activate.
    The backward pass reverses this with the chain rule.

    Hint: For Xavier initialization, scale = sqrt(2 / (fan_in + fan_out))
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = 'relu', seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (in_features + out_features))

        self.W = rng.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.activation = activation

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = {}

    def forward(self, A_hat: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Forward pass: H' = activation(Â · H · W + b)

        Three stages:
          1. Message passing: AH = Â @ H
          2. Linear transform: Z = AH @ W + b
          3. Activation: relu or none

        Cache all intermediates for backward pass.

        Hint: Think about what Â @ H does — it replaces each node's
        features with a weighted sum of its neighborhood's features.
        """
        raise NotImplementedError("TODO: implement forward pass")

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass through the GCN layer.

        Chain rule through: activation → linear → message passing

        Key gradients:
          - dW = (AH)^T @ d_Z
          - db = sum(d_Z, axis=0)
          - d_AH = d_Z @ W^T
          - d_H = A_hat^T @ d_AH

        IMPORTANT: Use in-place assignment (self.dW[:] = ...) not plain
        assignment (self.dW = ...) so the optimizer's reference stays valid.

        Hint: Since A_hat is symmetric, A_hat^T = A_hat. This is one
        of the nice properties of symmetric normalization.
        """
        raise NotImplementedError("TODO: implement backward pass")


class GCN:
    """
    2-layer GCN for node classification.

    Architecture: features → [GCNLayer+ReLU] → dropout → [GCNLayer] → logits

    Hint: Use inverted dropout (scale by 1/(1-p) during training)
    so you don't need to scale at test time.
    """

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int,
                 dropout: float = 0.5, seed: int = 42):
        self.layer1 = GCNLayer(in_features, hidden_dim, activation='relu', seed=seed)
        self.layer2 = GCNLayer(hidden_dim, num_classes, activation='none', seed=seed + 1)
        self.dropout = dropout
        self._dropout_mask = None
        self._training = True

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def forward(self, graph: Graph) -> np.ndarray:
        """
        Forward pass through both layers.

        Returns logits (N x num_classes) — no softmax here.

        Hint: Apply dropout between layers during training only.
        """
        raise NotImplementedError("TODO: implement forward pass")

    def backward(self, d_logits: np.ndarray):
        """
        Backward pass through both layers.

        Don't forget to pass gradients through dropout!
        """
        raise NotImplementedError("TODO: implement backward pass")

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return list of (param, grad) tuples for optimizer."""
        return [
            (self.layer1.W, self.layer1.dW),
            (self.layer1.b, self.layer1.db),
            (self.layer2.W, self.layer2.dW),
            (self.layer2.b, self.layer2.db),
        ]


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray,
                       mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Masked softmax cross-entropy loss.

    Only nodes where mask[i] > 0 contribute to the loss.

    Steps:
      1. Stable softmax: subtract max, exp, normalize
      2. Cross-entropy: -log(prob of correct class) for masked nodes
      3. Gradient: probs - one_hot(labels), masked and averaged

    Hint: The gradient of softmax + cross-entropy simplifies to
    (predicted_probs - target_one_hot), which is beautifully simple.

    Returns:
        loss: scalar
        d_logits: gradient w.r.t. logits (N x C)
    """
    raise NotImplementedError("TODO: implement cross-entropy loss")


class Adam:
    """
    Adam optimizer with weight decay.

    Hint: Don't forget bias correction for the first few steps,
    and add weight decay to gradients before computing moments.
    """

    def __init__(self, params: List[Tuple[np.ndarray, np.ndarray]],
                 lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 5e-4):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]

    def step(self):
        """Perform one optimization step."""
        raise NotImplementedError("TODO: implement Adam step")

    def zero_grad(self):
        """Reset all gradients to zero."""
        for _, grad in self.params:
            grad[:] = 0


def generate_citation_network(
    num_nodes: int = 200,
    num_classes: int = 4,
    feature_dim: int = 16,
    edge_prob_same: float = 0.15,
    edge_prob_diff: float = 0.01,
    seed: int = 42
) -> Tuple[Graph, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic citation network with community structure.

    Hint: Use a stochastic block model — same-class nodes connect
    with higher probability than different-class nodes.
    """
    raise NotImplementedError("TODO: implement dataset generation")


# =============================================================================
# Main — test your implementation
# =============================================================================

if __name__ == '__main__':
    print("Graph Neural Network (GCN) — Your Implementation")
    print("=" * 50)

    # Step 1: Generate data
    print("\n[1] Generating citation network...")
    graph, train_mask, val_mask, test_mask = generate_citation_network()
    print(f"    Nodes: {graph.num_nodes}, Features: {graph.features.shape[1]}")
    print(f"    Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    # Step 2: Train
    print("\n[2] Training GCN...")
    num_classes = len(np.unique(graph.labels))
    model = GCN(graph.features.shape[1], 32, num_classes, dropout=0.5, seed=42)
    optimizer = Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        logits = model.forward(graph)
        loss, d_logits = cross_entropy_loss(logits, graph.labels, train_mask.astype(np.float64))
        optimizer.zero_grad()
        model.backward(d_logits)
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            eval_logits = model.forward(graph)
            preds = eval_logits.argmax(axis=1)
            val_acc = (preds[val_mask] == graph.labels[val_mask]).mean()
            print(f"    Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.3f}")

    # Step 3: Evaluate
    print("\n[3] Final evaluation:")
    model.eval()
    logits = model.forward(graph)
    preds = logits.argmax(axis=1)
    test_acc = (preds[test_mask] == graph.labels[test_mask]).mean()
    print(f"    Test accuracy: {test_acc:.3f}")
