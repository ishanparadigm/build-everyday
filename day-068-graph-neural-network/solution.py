"""
Day 068: Graph Neural Network (GCN) from Scratch

A complete implementation of a Graph Convolutional Network following
Kipf & Welling (2017), with manual forward/backward passes, trained
on a synthetic citation-network-style node classification task.

No PyTorch/TensorFlow — just NumPy, to understand every gradient.
"""

import numpy as np
import warnings
from typing import Tuple, List, Optional

# Suppress numpy overflow warnings in matmul — these occur during the
# over-smoothing demo when features grow unbounded after many hops.
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*matmul.*')


# =============================================================================
# Graph Data Structures
# =============================================================================

class Graph:
    """
    Represents an undirected graph with node features and optional labels.

    Stores:
      - adj: raw adjacency matrix (N x N)
      - features: node feature matrix (N x F)
      - labels: integer class labels per node (N,), or None
      - adj_norm: precomputed normalized adjacency Â = D̃^(-1/2) Ã D̃^(-1/2)
    """

    def __init__(self, adj: np.ndarray, features: np.ndarray,
                 labels: Optional[np.ndarray] = None):
        assert adj.shape[0] == adj.shape[1], "Adjacency must be square"
        assert adj.shape[0] == features.shape[0], "Node count mismatch"

        self.num_nodes = adj.shape[0]
        self.adj = adj.astype(np.float64)
        self.features = features.astype(np.float64)
        self.labels = labels

        # Precompute normalized adjacency — this is the key preprocessing step.
        # We do it once because the graph structure doesn't change during training.
        self.adj_norm = self._normalize_adjacency(self.adj)

    def _normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
        """
        Compute Â = D̃^(-1/2) · Ã · D̃^(-1/2)

        Step 1: Add self-loops → Ã = A + I
          Without self-loops, a node's own features are excluded from aggregation.

        Step 2: Compute degree matrix D̃ where D̃[i,i] = Σ_j Ã[i,j]

        Step 3: Symmetric normalization D̃^(-1/2) · Ã · D̃^(-1/2)
          This scales each edge (i,j) by 1/√(deg(i) · deg(j)), preventing
          high-degree nodes from dominating the aggregation.
        """
        # Step 1: Add self-loops
        A_tilde = A + np.eye(self.num_nodes)

        # Step 2: Degree matrix (sum of each row, since self-loop adds 1)
        degrees = A_tilde.sum(axis=1)  # Shape: (N,)

        # Step 3: D^(-1/2) — avoid division by zero for isolated nodes
        d_inv_sqrt = np.zeros_like(degrees)
        nonzero = degrees > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

        # D^(-1/2) is diagonal, so we can use element-wise multiplication:
        # D^(-1/2) · Ã · D^(-1/2) = diag(d_inv_sqrt) @ Ã @ diag(d_inv_sqrt)
        # Efficiently: (d_inv_sqrt[:, None] * A_tilde) * d_inv_sqrt[None, :]
        A_hat = (d_inv_sqrt[:, None] * A_tilde) * d_inv_sqrt[None, :]

        return A_hat


# =============================================================================
# GCN Layer
# =============================================================================

class GCNLayer:
    """
    A single Graph Convolutional layer: H' = σ(Â · H · W + b)

    The forward pass has three stages:
      1. Aggregate: Â · H — replace each node's features with a weighted
         sum of its neighbors' features (message passing)
      2. Transform: (Â · H) · W + b — apply a learned linear map
      3. Activate: σ(·) — apply nonlinearity

    We store intermediate values for the backward pass.
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = 'relu', seed: int = 42):
        """
        Initialize weights using Xavier/Glorot initialization.

        Xavier init sets variance = 2/(fan_in + fan_out), which keeps
        signal magnitudes stable across layers. This is crucial for GCNs
        because the adjacency multiplication already changes the scale.
        """
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (in_features + out_features))

        self.W = rng.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.activation = activation

        # Gradients (accumulated during backward)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache for backward pass
        self._cache = {}

    def forward(self, A_hat: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Forward pass: H' = activation(Â · H · W + b)

        Args:
            A_hat: Normalized adjacency matrix (N x N)
            H: Input node features (N x F_in)
        Returns:
            Output node features (N x F_out)
        """
        # Stage 1: Message passing — aggregate neighbor features
        # AH[i] = weighted sum of features from node i's neighborhood
        AH = A_hat @ H  # (N x F_in)

        # Stage 2: Linear transformation
        Z = AH @ self.W + self.b  # (N x F_out)

        # Stage 3: Activation
        if self.activation == 'relu':
            out = np.maximum(0, Z)
        elif self.activation == 'none':
            out = Z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Cache everything needed for backward pass
        self._cache = {
            'A_hat': A_hat,
            'H': H,        # Input to this layer
            'AH': AH,      # After message passing
            'Z': Z,         # Before activation
            'out': out,     # After activation
        }

        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass through the GCN layer.

        Given gradient of loss w.r.t. output (d_out), compute:
          - dW, db: gradients for this layer's parameters
          - d_H: gradient w.r.t. input H (to propagate to previous layer)

        Chain rule through: activation → linear → message passing
        """
        A_hat = self._cache['A_hat']
        H = self._cache['H']
        AH = self._cache['AH']
        Z = self._cache['Z']

        # Step 1: Gradient through activation
        if self.activation == 'relu':
            # ReLU gradient: 1 where Z > 0, 0 elsewhere
            d_Z = d_out * (Z > 0).astype(np.float64)
        elif self.activation == 'none':
            d_Z = d_out
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Step 2: Gradient through linear transform Z = AH @ W + b
        # dW = (AH)^T @ d_Z — each column of dW sums contributions from all nodes
        # IMPORTANT: use in-place assignment ([:] =) so the optimizer's reference
        # to our gradient arrays stays valid. Plain = would rebind to a new array.
        self.dW[:] = AH.T @ d_Z
        # db = sum over all nodes (each node contributes to the bias gradient)
        self.db[:] = d_Z.sum(axis=0)

        # d_AH = d_Z @ W^T — gradient flows back through the weight matrix
        d_AH = d_Z @ self.W.T

        # Step 3: Gradient through message passing AH = A_hat @ H
        # d_H = A_hat^T @ d_AH
        # Key insight: A_hat is symmetric (because of symmetric normalization),
        # so A_hat^T = A_hat. This is why symmetric normalization is elegant —
        # the same matrix works for both forward and backward message passing.
        d_H = A_hat.T @ d_AH

        return d_H


# =============================================================================
# Full GCN Model
# =============================================================================

class GCN:
    """
    A 2-layer Graph Convolutional Network for node classification.

    Architecture:
      Layer 1: features → hidden (ReLU activation + dropout)
      Layer 2: hidden → num_classes (no activation, raw logits)

    The softmax and cross-entropy are computed in the loss function,
    not in the model, for numerical stability (log-sum-exp trick).
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
        Forward pass through the 2-layer GCN.

        Returns raw logits (N x num_classes) — softmax is applied in the loss.
        """
        A_hat = graph.adj_norm

        # Layer 1: aggregate + transform + ReLU
        h = self.layer1.forward(A_hat, graph.features)

        # Dropout between layers (only during training)
        # Dropout randomly zeros out neurons to prevent co-adaptation.
        # We scale surviving neurons by 1/(1-p) so expected values stay the same
        # ("inverted dropout"). At test time, we use all neurons with no scaling.
        if self._training and self.dropout > 0:
            self._dropout_mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float64)
            h = h * self._dropout_mask / (1.0 - self.dropout)
        else:
            self._dropout_mask = None

        # Layer 2: aggregate + transform (no activation — raw logits)
        logits = self.layer2.forward(A_hat, h)

        return logits

    def backward(self, d_logits: np.ndarray):
        """
        Backward pass through both layers.

        d_logits: gradient of loss w.r.t. logits (N x num_classes)
        """
        # Backward through layer 2
        d_h = self.layer2.backward(d_logits)

        # Backward through dropout (same mask, same scaling)
        if self._dropout_mask is not None:
            d_h = d_h * self._dropout_mask / (1.0 - self.dropout)

        # Backward through layer 1
        self.layer1.backward(d_h)

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return list of (param, grad) tuples for optimizer."""
        return [
            (self.layer1.W, self.layer1.dW),
            (self.layer1.b, self.layer1.db),
            (self.layer2.W, self.layer2.dW),
            (self.layer2.b, self.layer2.db),
        ]


# =============================================================================
# Loss Function
# =============================================================================

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray,
                       mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute masked cross-entropy loss with softmax.

    Only nodes where mask[i] == True contribute to the loss.
    This is the semi-supervised part: we train on labeled nodes
    but the message passing uses ALL nodes' features.

    Uses the log-sum-exp trick for numerical stability:
      softmax(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))

    Args:
        logits: Raw model output (N x C)
        labels: Integer class labels (N,)
        mask: Boolean mask — True for nodes to include in loss (N,)

    Returns:
        loss: Scalar cross-entropy loss (averaged over masked nodes)
        d_logits: Gradient of loss w.r.t. logits (N x C)
    """
    N, C = logits.shape

    # Numerically stable softmax using log-sum-exp trick
    # Subtract max per row to prevent exp() overflow
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Cross-entropy: -log(prob of correct class), only for masked nodes
    num_masked = mask.sum()
    if num_masked == 0:
        return 0.0, np.zeros_like(logits)

    # Gather probabilities of correct classes
    correct_probs = probs[np.arange(N), labels]
    # Clip to avoid log(0)
    correct_probs = np.clip(correct_probs, 1e-15, 1.0)

    # Loss: average -log(p_correct) over masked nodes
    loss = -np.sum(np.log(correct_probs) * mask) / num_masked

    # Gradient of cross-entropy w.r.t. logits:
    # d_logits[i,j] = probs[i,j] - 1(j == labels[i])  (for masked nodes)
    # This elegant gradient is why softmax + cross-entropy are paired:
    # the gradient is simply (predicted - target), same as MSE for regression.
    d_logits = probs.copy()
    d_logits[np.arange(N), labels] -= 1.0

    # Zero out gradient for non-masked nodes (they don't contribute to loss)
    d_logits *= mask[:, None]
    d_logits /= num_masked

    return loss, d_logits


# =============================================================================
# Optimizer: Adam
# =============================================================================

class Adam:
    """
    Adam optimizer — adaptive learning rates per parameter.

    Adam combines two ideas:
    1. Momentum (first moment): smooth out gradient noise by using
       an exponential moving average of past gradients
    2. RMSProp (second moment): adapt learning rate per parameter
       by dividing by the EMA of squared gradients

    Plus bias correction for the initial steps when the EMAs are
    still warming up (initialized at zero, they'd be too small).
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

        # Initialize moment estimates
        self.m = [np.zeros_like(p) for p, _ in params]  # First moment (mean)
        self.v = [np.zeros_like(p) for p, _ in params]  # Second moment (variance)

    def step(self):
        """Perform one optimization step."""
        self.t += 1

        for i, (param, grad) in enumerate(self.params):
            # Clip gradients to prevent numerical explosion — important
            # for GCNs where message passing can amplify gradient magnitudes
            grad = np.clip(grad, -5.0, 5.0)

            # L2 regularization (weight decay)
            # This penalizes large weights, acting as a prior that weights
            # should be small. Crucial for GCNs to prevent overfitting
            # when we only have a few labeled training nodes.
            g = grad + self.weight_decay * param

            # Update first moment estimate (gradient mean)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # Update second moment estimate (gradient variance)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction — without this, early updates would be too small
            # because m and v are initialized at zero
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Parameter update: step in the direction of corrected momentum,
            # scaled inversely by the corrected variance (adaptive LR)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset all gradients to zero before next backward pass."""
        for _, grad in self.params:
            grad[:] = 0


# =============================================================================
# Synthetic Dataset: Citation Network
# =============================================================================

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

    Creates a graph where nodes in the same class are more likely to be
    connected (homophily), mimicking how papers in the same field tend
    to cite each other. Each node gets features correlated with its class
    plus noise.

    Args:
        num_nodes: Total number of nodes
        num_classes: Number of classes (communities)
        feature_dim: Dimension of node features
        edge_prob_same: Probability of edge between same-class nodes
        edge_prob_diff: Probability of edge between different-class nodes
        seed: Random seed

    Returns:
        graph: The Graph object
        train_mask: Boolean mask for training nodes (few labels)
        val_mask: Boolean mask for validation nodes
        test_mask: Boolean mask for test nodes
    """
    rng = np.random.RandomState(seed)

    # Assign class labels (balanced classes)
    labels = np.array([i % num_classes for i in range(num_nodes)])
    rng.shuffle(labels)

    # Generate features: class-correlated signal + noise
    # Each class has a "prototype" feature vector; nodes get that + noise.
    # This gives the GCN something to work with beyond just graph structure.
    # Weak feature signal + strong noise — this makes the task hard enough
    # that graph structure (neighbor labels) actually helps. If features were
    # strong, a simple MLP would suffice and the GNN would add nothing.
    class_prototypes = rng.randn(num_classes, feature_dim) * 1.0
    features = np.zeros((num_nodes, feature_dim))
    for i in range(num_nodes):
        features[i] = class_prototypes[labels[i]] + rng.randn(feature_dim) * 1.5

    # Generate edges with community structure (stochastic block model)
    # Same-class nodes connect with higher probability → homophily
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if labels[i] == labels[j]:
                p = edge_prob_same
            else:
                p = edge_prob_diff
            if rng.rand() < p:
                adj[i, j] = 1
                adj[j, i] = 1  # Undirected

    # Create train/val/test splits
    # Semi-supervised: only ~10% of nodes are labeled for training
    # This is realistic — in most graph ML settings, labels are expensive
    indices = rng.permutation(num_nodes)
    n_train = num_nodes // 10  # 10% for training
    n_val = num_nodes // 5     # 20% for validation

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train + n_val]] = True
    test_mask[indices[n_train + n_val:]] = True

    graph = Graph(adj, features, labels)

    return graph, train_mask, val_mask, test_mask


# =============================================================================
# Training Loop
# =============================================================================

def train_gcn(
    graph: Graph,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    hidden_dim: int = 32,
    num_epochs: int = 200,
    lr: float = 0.01,
    dropout: float = 0.5,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[GCN, dict]:
    """
    Train a GCN model for node classification.

    Returns the trained model and training history.
    """
    num_classes = len(np.unique(graph.labels))
    in_features = graph.features.shape[1]

    # Initialize model
    model = GCN(in_features, hidden_dim, num_classes, dropout=dropout, seed=seed)
    optimizer = Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # === Forward pass (training mode) ===
        model.train()
        logits = model.forward(graph)

        # === Compute loss (only on training nodes) ===
        loss, d_logits = cross_entropy_loss(logits, graph.labels, train_mask.astype(np.float64))

        # === Backward pass ===
        optimizer.zero_grad()
        model.backward(d_logits)

        # === Update parameters ===
        optimizer.step()

        # === Evaluate ===
        model.eval()
        eval_logits = model.forward(graph)
        predictions = eval_logits.argmax(axis=1)

        train_acc = (predictions[train_mask] == graph.labels[train_mask]).mean()
        val_acc = (predictions[val_mask] == graph.labels[val_mask]).mean()

        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if verbose and (epoch % 20 == 0 or epoch == num_epochs - 1):
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    return model, history


def evaluate(model: GCN, graph: Graph, mask: np.ndarray) -> float:
    """Evaluate model accuracy on masked nodes."""
    model.eval()
    logits = model.forward(graph)
    predictions = logits.argmax(axis=1)
    accuracy = (predictions[mask] == graph.labels[mask]).mean()
    return accuracy


# =============================================================================
# Visualization: ASCII Graph and Embeddings
# =============================================================================

def visualize_predictions(graph: Graph, model: GCN, test_mask: np.ndarray):
    """Print a summary of predictions vs ground truth on test nodes."""
    model.eval()
    logits = model.forward(graph)
    predictions = logits.argmax(axis=1)

    # Get confidence (max softmax probability)
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)

    test_indices = np.where(test_mask)[0][:20]  # Show first 20 test nodes

    print("\n  Predictions on test nodes (first 20):")
    print(f"  {'Node':>6} | {'True':>5} | {'Pred':>5} | {'Conf':>6} | {'Correct':>7}")
    print(f"  {'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*7}")

    correct = 0
    for idx in test_indices:
        is_correct = predictions[idx] == graph.labels[idx]
        correct += is_correct
        marker = "  Y" if is_correct else "  N"
        print(f"  {idx:6d} | {graph.labels[idx]:5d} | {predictions[idx]:5d} | "
              f"{confidence[idx]:6.3f} | {marker}")

    print(f"\n  Shown: {correct}/{len(test_indices)} correct")


def print_graph_stats(graph: Graph, train_mask: np.ndarray,
                      val_mask: np.ndarray, test_mask: np.ndarray):
    """Print summary statistics about the graph."""
    num_edges = int(graph.adj.sum()) // 2  # Undirected, so divide by 2
    avg_degree = graph.adj.sum(axis=1).mean()
    num_classes = len(np.unique(graph.labels))

    print(f"\n  Graph Statistics:")
    print(f"    Nodes: {graph.num_nodes}")
    print(f"    Edges: {num_edges}")
    print(f"    Avg degree: {avg_degree:.1f}")
    print(f"    Features per node: {graph.features.shape[1]}")
    print(f"    Classes: {num_classes}")
    print(f"    Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    # Class distribution
    print(f"    Class distribution: ", end="")
    for c in range(num_classes):
        count = (graph.labels == c).sum()
        print(f"C{c}={count} ", end="")
    print()

    # Homophily ratio: fraction of edges connecting same-class nodes
    same_class_edges = 0
    total_edges = 0
    for i in range(graph.num_nodes):
        for j in range(i + 1, graph.num_nodes):
            if graph.adj[i, j] > 0:
                total_edges += 1
                if graph.labels[i] == graph.labels[j]:
                    same_class_edges += 1

    homophily = same_class_edges / max(total_edges, 1)
    print(f"    Homophily ratio: {homophily:.3f} (1.0 = perfect community structure)")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Day 068: Graph Neural Network (GCN) from Scratch")
    print("=" * 60)

    # --- Step 1: Generate synthetic citation network ---
    print("\n[1] Generating synthetic citation network...")
    graph, train_mask, val_mask, test_mask = generate_citation_network(
        num_nodes=200,
        num_classes=4,
        feature_dim=16,
        edge_prob_same=0.15,
        edge_prob_diff=0.01,
        seed=42
    )
    print_graph_stats(graph, train_mask, val_mask, test_mask)

    # --- Step 2: Inspect the normalized adjacency ---
    print(f"\n[2] Adjacency matrix normalization:")
    print(f"    Raw adjacency - nonzero entries: {(graph.adj > 0).sum()}")
    print(f"    Normalized adjacency - shape: {graph.adj_norm.shape}")
    print(f"    Normalized adjacency - row sums (should be ~1):")
    row_sums = graph.adj_norm.sum(axis=1)
    print(f"      Mean: {row_sums.mean():.4f}, Std: {row_sums.std():.4f}")
    print(f"      Min: {row_sums.min():.4f}, Max: {row_sums.max():.4f}")

    # --- Step 3: Train the GCN ---
    print(f"\n[3] Training 2-layer GCN (features→32→4 classes)...")
    print(f"    Using only {train_mask.sum()} labeled nodes for training "
          f"({train_mask.sum()/graph.num_nodes*100:.0f}% of graph)\n")

    np.random.seed(42)
    model, history = train_gcn(
        graph, train_mask, val_mask,
        hidden_dim=32, num_epochs=200, lr=0.01,
        dropout=0.3, seed=42, verbose=True
    )

    # --- Step 4: Final evaluation ---
    print(f"\n[4] Final Evaluation:")
    train_acc = evaluate(model, graph, train_mask)
    val_acc = evaluate(model, graph, val_mask)
    test_acc = evaluate(model, graph, test_mask)
    print(f"    Train accuracy: {train_acc:.3f}")
    print(f"    Val accuracy:   {val_acc:.3f}")
    print(f"    Test accuracy:  {test_acc:.3f}")

    # --- Step 5: Show predictions ---
    visualize_predictions(graph, model, test_mask)

    # --- Step 6: Compare with baseline (no graph structure) ---
    print(f"\n[5] Baseline comparison: What if we ignore graph structure?")
    print(f"    Training a GCN with identity adjacency (no message passing)...")

    # Create graph with no edges — just self-loops after normalization
    no_edge_graph = Graph(
        np.zeros_like(graph.adj),  # No edges
        graph.features,
        graph.labels
    )
    np.random.seed(123)
    baseline_model, baseline_history = train_gcn(
        no_edge_graph, train_mask, val_mask,
        hidden_dim=32, num_epochs=200, lr=0.01,
        dropout=0.3, seed=42, verbose=False
    )
    baseline_test_acc = evaluate(baseline_model, no_edge_graph, test_mask)
    print(f"    Baseline test accuracy (no graph): {baseline_test_acc:.3f}")
    print(f"    GCN test accuracy (with graph):    {test_acc:.3f}")
    improvement = (test_acc - baseline_test_acc) * 100
    print(f"    Graph structure adds: {improvement:+.1f} percentage points")

    # --- Step 7: Demonstrate over-smoothing ---
    print(f"\n[6] Over-smoothing demonstration:")
    print(f"    Measuring feature similarity after k layers of message passing...")

    H = graph.features.copy()
    for k in range(6):
        # Measure how similar node features are across different classes
        class_centers = []
        for c in range(4):
            class_mask = graph.labels == c
            class_centers.append(H[class_mask].mean(axis=0))

        # Inter-class distance: how distinguishable are the classes?
        inter_class_dist = 0
        count = 0
        for i in range(4):
            for j in range(i + 1, 4):
                inter_class_dist += np.linalg.norm(class_centers[i] - class_centers[j])
                count += 1
        inter_class_dist /= count

        print(f"    After {k} hops: inter-class distance = {inter_class_dist:.4f}"
              + (" ← features start converging!" if k >= 4 else ""))

        # One more round of message passing (without learned transform)
        H = graph.adj_norm @ H

    print(f"\n    This is why GCNs typically use only 2-3 layers!")

    print("\n" + "=" * 60)
    print("  Complete! The GCN learns to classify nodes by combining")
    print("  local features with graph structure via message passing.")
    print("=" * 60)
