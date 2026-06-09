"""
Day 069: Mixture of Experts (MoE) — Your Implementation

Build a Mixture of Experts model from scratch. The key insight: instead of
routing every input through one big network, use a gating network to select
which expert sub-networks process each input.

Hints:
- Start with a single Expert MLP — it's just a standard 2-layer network
- The gating network is a linear layer + softmax + top-K selection
- Load balancing loss prevents expert collapse (the #1 failure mode)
- Think about WHY we renormalize after top-K selection

Run tests: python3 -m pytest tests.py -v
Run solution: python3 my_solution.py
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


# ============================================================================
# Activation Functions
# ============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    raise NotImplementedError("TODO: implement this")


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where x > 0, else 0."""
    raise NotImplementedError("TODO: implement this")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the given axis.

    Hint: subtract max before exp() to prevent overflow.
    """
    raise NotImplementedError("TODO: implement this")


def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus: smooth approximation to ReLU, always positive.
    softplus(x) = log(1 + exp(x))

    Hint: clip x to avoid overflow in exp().
    """
    raise NotImplementedError("TODO: implement this")


# ============================================================================
# Expert Network (Single MLP)
# ============================================================================

class Expert:
    """A single expert: a 2-layer MLP with ReLU activation.

    Architecture: input_dim -> hidden_dim -> output_dim

    Hint: Use He initialization (scale = sqrt(2/fan_in)) for ReLU networks.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, expert_id: int):
        self.expert_id = expert_id
        # TODO: Initialize W1, b1, W2, b2 with He initialization
        raise NotImplementedError("TODO: implement this")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x -> linear -> ReLU -> linear -> output.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)

        Hint: Cache intermediate values (x, z1, h1) for backprop.
        """
        raise NotImplementedError("TODO: implement this")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the expert MLP.

        Args:
            grad_output: Gradient from downstream, shape (batch_size, output_dim)
        Returns:
            Gradient w.r.t. input x, shape (batch_size, input_dim)

        Hint: Compute gradients for W2, b2, then backprop through ReLU, then W1, b1.
        """
        raise NotImplementedError("TODO: implement this")

    def update(self, lr: float):
        """SGD parameter update.

        Hint: W -= lr * grad_W for each parameter.
        """
        raise NotImplementedError("TODO: implement this")


# ============================================================================
# Gating Network (Router)
# ============================================================================

class GatingNetwork:
    """The gating network routes inputs to experts.

    For input x, it produces a probability distribution over N experts,
    then selects the top-K experts.

    Hint: The core idea is simple:
    1. logits = x @ W_gate
    2. probs = softmax(logits + noise)
    3. Keep only top-K, zero the rest
    4. Renormalize so selected weights sum to 1
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2, noise_std: float = 1.0):
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.training = True
        # TODO: Initialize W_gate and W_noise
        raise NotImplementedError("TODO: implement this")

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gating weights with optional noisy top-K routing.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            gate_weights: Sparse gate weights, shape (batch_size, num_experts)
                          Only top-K entries non-zero per row, sum to 1.
            full_probs: Full softmax probabilities (for load balancing loss)

        Hint: During training, add noise = randn * softplus(x @ W_noise) * noise_std
        """
        raise NotImplementedError("TODO: implement this")

    def backward(self, grad_gate: np.ndarray) -> np.ndarray:
        """Backward pass through the gating network.

        Hint: Gradient through softmax is: p * (grad - sum(grad * p))
        """
        raise NotImplementedError("TODO: implement this")

    def update(self, lr: float):
        """SGD parameter update."""
        raise NotImplementedError("TODO: implement this")


# ============================================================================
# Mixture of Experts Layer
# ============================================================================

class MoELayer:
    """The complete Mixture of Experts layer.

    For each input:
    1. Gate produces weights over experts
    2. Top-K experts are selected
    3. Selected experts compute their outputs
    4. Outputs are combined: y = sum(gate_weight_i * expert_i(x))
    5. Load balancing loss is computed

    Hint: The load balancing loss is:
    L = alpha * N * sum(f_i * p_i)
    where f_i = fraction of tokens routed to expert i
    and p_i = mean gate probability for expert i
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_experts: int = 8, top_k: int = 2, balance_coeff: float = 0.01):
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_coeff = balance_coeff
        # TODO: Create list of Expert networks and a GatingNetwork
        # TODO: Initialize expert_counts and total_tokens for utilization tracking
        raise NotImplementedError("TODO: implement this")

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward pass through the MoE layer.

        Args:
            x: Input tensor, shape (batch_size, input_dim)
        Returns:
            output: Combined expert outputs, shape (batch_size, output_dim)
            balance_loss: Auxiliary load balancing loss (scalar)

        Hint: The output is gate_weights[:,:,None] * expert_outputs summed over experts.
        """
        raise NotImplementedError("TODO: implement this")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the MoE layer.

        Hint: grad for gate = sum(grad_output * expert_output) over output dim
              grad for each expert = gate_weight * grad_output
        """
        raise NotImplementedError("TODO: implement this")

    def update(self, lr: float):
        """Update all parameters."""
        raise NotImplementedError("TODO: implement this")

    def get_utilization(self) -> np.ndarray:
        """Get per-expert utilization as fractions."""
        if self.total_tokens == 0:
            return np.zeros(self.num_experts)
        return self.expert_counts / self.total_tokens

    def reset_utilization(self):
        """Reset utilization counters."""
        self.expert_counts = np.zeros(self.num_experts)
        self.total_tokens = 0


# ============================================================================
# Full MoE Classification Model
# ============================================================================

class MoEClassifier:
    """A complete classification model using MoE.

    Architecture: Input -> MoE Layer -> Output projection -> Softmax
    """

    def __init__(self, input_dim: int, num_classes: int, moe_hidden: int = 32,
                 moe_output: int = 16, num_experts: int = 8, top_k: int = 2,
                 balance_coeff: float = 0.01):
        # TODO: Create MoE layer and output projection (W_out, b_out)
        raise NotImplementedError("TODO: implement this")

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward pass: input -> MoE -> logits.

        Returns:
            logits: Raw class scores, shape (batch_size, num_classes)
            balance_loss: Auxiliary load balancing loss
        """
        raise NotImplementedError("TODO: implement this")

    def backward(self, grad_logits: np.ndarray):
        """Backward pass through the full model."""
        raise NotImplementedError("TODO: implement this")

    def update(self, lr: float):
        """Update all parameters."""
        raise NotImplementedError("TODO: implement this")

    def set_training(self, mode: bool):
        """Toggle training mode (affects noisy gating)."""
        raise NotImplementedError("TODO: implement this")


# ============================================================================
# Loss Function
# ============================================================================

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Cross-entropy loss with softmax.

    Args:
        logits: Raw scores, shape (batch_size, num_classes)
        labels: Integer labels, shape (batch_size,)
    Returns:
        loss: Scalar cross-entropy loss
        grad: Gradient w.r.t. logits, shape (batch_size, num_classes)

    Hint: The gradient of cross-entropy w.r.t. logits is simply: softmax(logits) - one_hot(labels)
    """
    raise NotImplementedError("TODO: implement this")


# ============================================================================
# Synthetic Dataset
# ============================================================================

def generate_multi_region_data(n_samples: int = 2000, n_classes: int = 6,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic multi-region classification dataset.

    Each class is a Gaussian cluster centered at a point on a circle.
    Perfect for MoE — different experts can specialize on different regions.
    """
    np.random.seed(seed)

    X_list, y_list = [], []
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    radius = 3.0

    for i, angle in enumerate(angles):
        center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        n = n_samples // n_classes
        spread = 0.8 + 0.4 * (i % 3)
        X_class = np.random.randn(n, 2) * spread + center
        X_list.append(X_class)
        y_list.append(np.full(n, i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


# ============================================================================
# Main — Test your implementation
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MIXTURE OF EXPERTS (MoE) — YOUR IMPLEMENTATION")
    print("=" * 60)

    # Test activations
    print("\n--- Testing activation functions ---")
    test_x = np.array([-2, -1, 0, 1, 2], dtype=float)
    print(f"ReLU({test_x}) = {relu(test_x)}")
    print(f"Softmax({test_x}) = {softmax(test_x)}")
    print(f"Softplus({test_x}) = {softplus(test_x)}")

    # Generate data
    print("\n--- Generating data ---")
    X, y = generate_multi_region_data(n_samples=1200, n_classes=6)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Classes: {len(np.unique(y))}")

    # Test single expert
    print("\n--- Testing single Expert ---")
    expert = Expert(input_dim=2, hidden_dim=16, output_dim=8, expert_id=0)
    out = expert.forward(X_train[:5])
    print(f"Expert output shape: {out.shape}")  # Should be (5, 8)

    # Test gating network
    print("\n--- Testing GatingNetwork ---")
    gate = GatingNetwork(input_dim=2, num_experts=8, top_k=2)
    weights, probs = gate.forward(X_train[:5])
    print(f"Gate weights shape: {weights.shape}")  # Should be (5, 8)
    print(f"Non-zero per row: {np.sum(weights > 0, axis=1)}")  # Should be [2,2,2,2,2]
    print(f"Row sums: {np.sum(weights, axis=1)}")  # Should be ~[1,1,1,1,1]

    # Test full model
    print("\n--- Training MoE Classifier ---")
    np.random.seed(123)
    model = MoEClassifier(
        input_dim=2, num_classes=6,
        moe_hidden=32, moe_output=16,
        num_experts=8, top_k=2,
        balance_coeff=0.01
    )

    # Quick training loop
    for epoch in range(30):
        model.set_training(True)
        model.moe.reset_utilization()

        perm = np.random.permutation(len(X_train))
        for start in range(0, len(X_train), 64):
            end = min(start + 64, len(X_train))
            xb = X_train[perm[start:end]]
            yb = y_train[perm[start:end]]

            logits, bl = model.forward(xb)
            loss, grad = cross_entropy_loss(logits, yb)
            model.backward(grad)
            model.update(0.05)

        if epoch % 10 == 0:
            model.set_training(False)
            val_logits, _ = model.forward(X_val)
            val_preds = np.argmax(val_logits, axis=1)
            acc = np.mean(val_preds == y_val)
            util = model.moe.get_utilization()
            print(f"Epoch {epoch:3d} | Val Acc: {acc:.3f} | Utilization: {np.array2string(util, precision=2)}")

    print("\nDone! Check expert utilization — all experts should be active.")
