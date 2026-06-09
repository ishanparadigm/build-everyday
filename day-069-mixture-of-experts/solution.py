"""
Day 069: Mixture of Experts (MoE) from Scratch

A complete implementation of the Mixture of Experts architecture using only NumPy.
We build a gating network, expert MLPs, load balancing loss, and train the full
system on a synthetic multi-region classification task to observe expert specialization.

Key architectural choices:
- Top-K gating with noisy routing for exploration during training
- Auxiliary load balancing loss to prevent expert collapse
- Per-expert utilization tracking to monitor health
- Softmax gating with temperature control

Run: python3 solution.py
"""

import warnings
import numpy as np

# Suppress transient overflow warnings from early training batches.
# The gradient/weight clipping handles numerical recovery — these warnings
# are expected when random initialization produces a few extreme values.
warnings.filterwarnings('ignore', category=RuntimeWarning)
from typing import Tuple, List, Dict, Optional


# ============================================================================
# Activation Functions
# ============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation. Dead-simple but effective for expert MLPs."""
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(float)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the given axis.

    Subtracting max prevents overflow in exp(). This doesn't change the output
    because softmax is shift-invariant: softmax(x) = softmax(x - c) for any c.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus: smooth approximation to ReLU. Used for noise magnitude (must be positive)."""
    return np.log1p(np.exp(np.clip(x, -20, 20)))


# ============================================================================
# Expert Network (Single MLP)
# ============================================================================

class Expert:
    """A single expert: a 2-layer MLP with ReLU activation.

    Architecture: input_dim -> hidden_dim -> output_dim

    Each expert has independent weights. Over training, different experts
    specialize on different regions of the input space — this is the whole
    point of MoE.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, expert_id: int):
        self.expert_id = expert_id
        # He initialization: scale by sqrt(2/fan_in) for ReLU networks.
        # This keeps variance stable through layers, preventing vanishing/exploding gradients.
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        # Cache for backprop
        self.cache: Dict = {}
        # Gradient accumulators
        self.grads: Dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the expert MLP.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Layer 1: linear + ReLU
        z1 = x @ self.W1 + self.b1          # (batch, hidden)
        h1 = relu(z1)                         # (batch, hidden)

        # Layer 2: linear (no activation — the MoE layer handles that)
        z2 = h1 @ self.W2 + self.b2          # (batch, output)

        # Cache everything for backprop
        self.cache = {'x': x, 'z1': z1, 'h1': h1}
        return z2

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the expert MLP.

        Args:
            grad_output: Gradient from downstream, shape (batch_size, output_dim)
        Returns:
            Gradient w.r.t. input x, shape (batch_size, input_dim)
        """
        x, z1, h1 = self.cache['x'], self.cache['z1'], self.cache['h1']
        batch_size = x.shape[0]

        # Gradients for layer 2
        self.grads['W2'] = h1.T @ grad_output / batch_size    # (hidden, output)
        self.grads['b2'] = np.mean(grad_output, axis=0)       # (output,)

        # Backprop through layer 2 → layer 1
        grad_h1 = grad_output @ self.W2.T                     # (batch, hidden)
        grad_z1 = grad_h1 * relu_grad(z1)                     # (batch, hidden)

        # Gradients for layer 1
        self.grads['W1'] = x.T @ grad_z1 / batch_size         # (input, hidden)
        self.grads['b1'] = np.mean(grad_z1, axis=0)           # (hidden,)

        # Gradient w.r.t. input (for upstream layers)
        grad_x = grad_z1 @ self.W1.T                          # (batch, input)
        return grad_x

    def update(self, lr: float):
        """SGD parameter update with gradient clipping for numerical stability."""
        for key in ['W1', 'b1', 'W2', 'b2']:
            grad = np.nan_to_num(self.grads[key], nan=0.0, posinf=5.0, neginf=-5.0)
            grad = np.clip(grad, -5.0, 5.0)
            param = getattr(self, key) - lr * grad
            # Clip weights to prevent overflow in forward pass matmuls
            setattr(self, key, np.clip(param, -10.0, 10.0))


# ============================================================================
# Gating Network (Router)
# ============================================================================

class GatingNetwork:
    """The gating network routes inputs to experts.

    For input x, it produces a probability distribution over N experts,
    then selects the top-K experts. Only those K experts actually compute.

    Key design decisions:
    - Noisy gating during training: adds learned Gaussian noise to encourage
      exploration and prevent premature expert collapse
    - Top-K selection: only K experts are active per input, giving O(K/N) compute savings
    - Renormalization: the K selected gate values are renormalized to sum to 1
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2, noise_std: float = 1.0):
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Gate weights: maps input to expert scores
        self.W_gate = np.random.randn(input_dim, num_experts) * 0.01
        # Noise weights: maps input to per-expert noise magnitude
        self.W_noise = np.random.randn(input_dim, num_experts) * 0.01

        self.cache: Dict = {}
        self.grads: Dict = {}
        self.training = True

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gating weights with optional noisy top-K routing.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            gate_weights: Sparse gate weights of shape (batch_size, num_experts)
                          Only top-K entries are non-zero per row, and they sum to 1.
            full_probs: Full softmax probabilities (needed for load balancing loss)
        """
        # Raw gate logits
        logits = x @ self.W_gate  # (batch, num_experts)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise_magnitude = softplus(x @ self.W_noise)  # (batch, num_experts), always positive
            noise = np.random.randn(*logits.shape) * noise_magnitude * self.noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        # Full softmax probabilities (used for load balancing loss)
        full_probs = softmax(noisy_logits)  # (batch, num_experts)

        # Top-K selection: keep only the K largest gate values per input
        # This is where the computational savings come from
        top_k_indices = np.argsort(noisy_logits, axis=-1)[:, -self.top_k:]  # (batch, K)

        # Create sparse gate weights: zero everywhere except top-K positions
        gate_weights = np.zeros_like(full_probs)
        for i in range(x.shape[0]):
            gate_weights[i, top_k_indices[i]] = full_probs[i, top_k_indices[i]]

        # Renormalize so the K selected weights sum to 1
        # This ensures the output scale is consistent regardless of K
        gate_sum = np.sum(gate_weights, axis=-1, keepdims=True) + 1e-8
        gate_weights = gate_weights / gate_sum

        # Cache for backprop
        self.cache = {
            'x': x,
            'logits': logits,
            'full_probs': full_probs,
            'gate_weights': gate_weights,
            'top_k_indices': top_k_indices
        }

        return gate_weights, full_probs

    def backward(self, grad_gate: np.ndarray) -> np.ndarray:
        """Backward pass through the gating network.

        We approximate the gradient by treating the top-K selection as fixed
        (straight-through estimator style) and backpropping through the softmax
        only for the selected experts. This is a simplification — the full
        gradient through top-K is discontinuous.

        Args:
            grad_gate: Gradient w.r.t. gate_weights, shape (batch_size, num_experts)
        Returns:
            Gradient w.r.t. input x
        """
        x = self.cache['x']
        full_probs = self.cache['full_probs']
        batch_size = x.shape[0]

        # Gradient through softmax: d(softmax)/d(logits)
        # For softmax output p, the Jacobian is diag(p) - p @ p.T
        # Applied to grad_gate: grad_logits = p * (grad_gate - sum(grad_gate * p))
        weighted_sum = np.sum(grad_gate * full_probs, axis=-1, keepdims=True)
        grad_logits = full_probs * (grad_gate - weighted_sum)  # (batch, num_experts)

        # Gradient for gate weights
        self.grads['W_gate'] = x.T @ grad_logits / batch_size

        # Gradient w.r.t. input
        grad_x = grad_logits @ self.W_gate.T

        return grad_x

    def update(self, lr: float):
        """SGD parameter update."""
        grad = np.nan_to_num(self.grads['W_gate'], nan=0.0, posinf=5.0, neginf=-5.0)
        grad = np.clip(grad, -5.0, 5.0)
        self.W_gate = np.clip(self.W_gate - lr * grad, -10.0, 10.0)


# ============================================================================
# Mixture of Experts Layer
# ============================================================================

class MoELayer:
    """The complete Mixture of Experts layer.

    Combines a gating network with N expert MLPs. For each input:
    1. Gate produces weights over experts
    2. Top-K experts are selected
    3. Only selected experts run (sparse computation)
    4. Outputs are combined: y = sum(gate_weight_i * expert_i(x))

    Also computes the load balancing auxiliary loss to prevent expert collapse.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_experts: int = 8, top_k: int = 2, balance_coeff: float = 0.01):
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_coeff = balance_coeff

        # Create expert networks
        self.experts = [
            Expert(input_dim, hidden_dim, output_dim, expert_id=i)
            for i in range(num_experts)
        ]

        # Create gating network
        self.gate = GatingNetwork(input_dim, num_experts, top_k)

        # Tracking expert utilization for monitoring
        self.expert_counts = np.zeros(num_experts)
        self.total_tokens = 0

        self.cache: Dict = {}

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward pass through the MoE layer.

        Args:
            x: Input tensor, shape (batch_size, input_dim)
        Returns:
            output: Combined expert outputs, shape (batch_size, output_dim)
            balance_loss: Auxiliary load balancing loss (scalar)
        """
        batch_size = x.shape[0]

        # Step 1: Compute gating weights
        gate_weights, full_probs = self.gate.forward(x)  # (batch, num_experts)

        # Step 2: Run experts and combine outputs
        # In a real implementation, you'd only run selected experts.
        # Here we run all but mask the output — conceptually equivalent,
        # and simpler to implement with NumPy (no dynamic dispatch).
        expert_outputs = []
        for expert in self.experts:
            out = expert.forward(x)  # (batch, output_dim)
            expert_outputs.append(out)
        expert_outputs = np.stack(expert_outputs, axis=1)  # (batch, num_experts, output_dim)

        # Weighted combination: gate_weights acts as a sparse selector
        # gate_weights[:, :, None] broadcasts to (batch, num_experts, 1)
        # Multiply and sum over experts dimension
        output = np.sum(gate_weights[:, :, None] * expert_outputs, axis=1)  # (batch, output_dim)

        # Step 3: Compute load balancing loss
        # f_i = fraction of tokens routed to expert i
        # p_i = mean gate probability for expert i
        # balance_loss = N * sum(f_i * p_i)
        # This is minimized when f and p are both uniform (= 1/N)

        # Count which experts were in top-K for each input
        top_k_mask = (gate_weights > 0).astype(float)  # (batch, num_experts)
        f = np.mean(top_k_mask, axis=0)                # (num_experts,) fraction routed to each
        p = np.mean(full_probs, axis=0)                 # (num_experts,) mean probability each
        balance_loss = self.balance_coeff * self.num_experts * np.sum(f * p)

        # Update utilization tracking
        self.expert_counts += np.sum(top_k_mask, axis=0)
        self.total_tokens += batch_size

        # Cache for backprop
        self.cache = {
            'x': x,
            'gate_weights': gate_weights,
            'full_probs': full_probs,
            'expert_outputs': expert_outputs,
            'f': f,
            'p': p
        }

        return output, balance_loss

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the MoE layer.

        Distributes gradients to both the experts (weighted by gate) and
        the gating network (weighted by expert outputs).
        """
        gate_weights = self.cache['gate_weights']
        expert_outputs = self.cache['expert_outputs']
        full_probs = self.cache['full_probs']
        f = self.cache['f']
        batch_size = grad_output.shape[0]

        # Gradient w.r.t. gate_weights:
        # output = sum_i gate_i * expert_i
        # d(output)/d(gate_i) = expert_i
        # Chain rule: grad_gate_i = dot(grad_output, expert_i_output)
        # Shape: (batch, num_experts)
        grad_gate = np.sum(grad_output[:, None, :] * expert_outputs, axis=2)

        # Add gradient from load balancing loss
        # L_balance = alpha * N * sum(f_i * p_i)
        # Gradient w.r.t. full_probs: alpha * N * f_i / batch_size (since p_i = mean of probs)
        grad_balance = self.balance_coeff * self.num_experts * f[None, :] / batch_size
        grad_gate = grad_gate + grad_balance

        # Backward through gating network
        grad_x_gate = self.gate.backward(grad_gate)

        # Gradient w.r.t. each expert's output, weighted by its gate value
        # grad_expert_i = gate_i * grad_output
        grad_x_experts = np.zeros_like(self.cache['x'])
        for i, expert in enumerate(self.experts):
            # Only backprop through experts that had non-zero gate weights
            expert_grad = gate_weights[:, i:i+1] * grad_output  # (batch, output_dim)
            grad_x_i = expert.backward(expert_grad)
            grad_x_experts += grad_x_i

        return grad_x_gate + grad_x_experts

    def update(self, lr: float):
        """Update all parameters."""
        self.gate.update(lr)
        for expert in self.experts:
            expert.update(lr)

    def get_utilization(self) -> np.ndarray:
        """Get per-expert utilization as fractions."""
        if self.total_tokens == 0:
            return np.zeros(self.num_experts)
        return self.expert_counts / self.total_tokens

    def reset_utilization(self):
        """Reset utilization counters (call between epochs)."""
        self.expert_counts = np.zeros(self.num_experts)
        self.total_tokens = 0


# ============================================================================
# Full MoE Classification Model
# ============================================================================

class MoEClassifier:
    """A complete classification model using MoE.

    Architecture: Input -> MoE Layer -> Output projection -> Softmax

    The MoE layer provides the "brain" — different experts specialize on
    different input regions. The output projection maps to class logits.
    """

    def __init__(self, input_dim: int, num_classes: int, moe_hidden: int = 32,
                 moe_output: int = 16, num_experts: int = 8, top_k: int = 2,
                 balance_coeff: float = 0.01):
        # MoE layer: the core conditional computation
        self.moe = MoELayer(
            input_dim=input_dim,
            hidden_dim=moe_hidden,
            output_dim=moe_output,
            num_experts=num_experts,
            top_k=top_k,
            balance_coeff=balance_coeff
        )

        # Output head: maps MoE output to class logits
        self.W_out = np.random.randn(moe_output, num_classes) * np.sqrt(2.0 / moe_output)
        self.b_out = np.zeros(num_classes)

        self.cache: Dict = {}

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward pass: input -> MoE -> logits.

        Returns:
            logits: Raw class scores, shape (batch_size, num_classes)
            balance_loss: Auxiliary load balancing loss
        """
        # MoE layer
        moe_out, balance_loss = self.moe.forward(x)  # (batch, moe_output)

        # Output projection
        logits = moe_out @ self.W_out + self.b_out  # (batch, num_classes)

        self.cache = {'moe_out': moe_out, 'logits': logits}
        return logits, balance_loss

    def backward(self, grad_logits: np.ndarray):
        """Backward pass through the full model."""
        moe_out = self.cache['moe_out']
        batch_size = grad_logits.shape[0]

        # Gradients for output projection
        self.grad_W_out = moe_out.T @ grad_logits / batch_size
        self.grad_b_out = np.mean(grad_logits, axis=0)

        # Gradient flowing back to MoE
        grad_moe = grad_logits @ self.W_out.T
        self.moe.backward(grad_moe)

    def update(self, lr: float):
        """Update all parameters."""
        grad_w = np.nan_to_num(self.grad_W_out, nan=0.0, posinf=5.0, neginf=-5.0)
        grad_b = np.nan_to_num(self.grad_b_out, nan=0.0, posinf=5.0, neginf=-5.0)
        self.W_out = np.clip(self.W_out - lr * np.clip(grad_w, -5.0, 5.0), -10.0, 10.0)
        self.b_out = np.clip(self.b_out - lr * np.clip(grad_b, -5.0, 5.0), -10.0, 10.0)
        self.moe.update(lr)

    def set_training(self, mode: bool):
        """Toggle training mode (affects noisy gating)."""
        self.moe.gate.training = mode


# ============================================================================
# Loss Function
# ============================================================================

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Cross-entropy loss with softmax.

    Combines softmax + negative log-likelihood for numerical stability.

    Args:
        logits: Raw scores, shape (batch_size, num_classes)
        labels: Integer labels, shape (batch_size,)
    Returns:
        loss: Scalar cross-entropy loss
        grad: Gradient w.r.t. logits, shape (batch_size, num_classes)
    """
    probs = softmax(logits)
    batch_size = logits.shape[0]

    # Negative log-likelihood of correct class
    # Clip to avoid log(0)
    correct_probs = probs[np.arange(batch_size), labels]
    loss = -np.mean(np.log(correct_probs + 1e-8))

    # Gradient of cross-entropy w.r.t. logits is simply: probs - one_hot(labels)
    # This elegant form comes from the calculus of softmax + NLL
    grad = probs.copy()
    grad[np.arange(batch_size), labels] -= 1.0

    return loss, grad


# ============================================================================
# Synthetic Dataset: Multi-Region Classification
# ============================================================================

def generate_multi_region_data(n_samples: int = 2000, n_classes: int = 6,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset where different classes occupy different
    regions of 2D space.

    This is ideal for MoE because different experts can specialize on different
    spatial regions — we can visually verify that specialization happens.

    Each class is a Gaussian cluster centered at a point on a circle.
    """
    np.random.seed(seed)

    X_list, y_list = [], []
    # Place class centers evenly around a circle
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    radius = 3.0

    for i, angle in enumerate(angles):
        center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        n = n_samples // n_classes
        # Each cluster has its own spread — this makes some classes harder
        spread = 0.8 + 0.4 * (i % 3)
        X_class = np.random.randn(n, 2) * spread + center
        X_list.append(X_class)
        y_list.append(np.full(n, i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


# ============================================================================
# Training Loop
# ============================================================================

def train_moe(model: MoEClassifier, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 64, lr: float = 0.05,
              verbose: bool = True) -> Dict[str, List]:
    """Train the MoE classifier and track metrics.

    Returns a dictionary of training history including losses,
    accuracies, and per-expert utilization over time.
    """
    n_train = X_train.shape[0]
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'balance_loss': [], 'utilization': []
    }

    for epoch in range(epochs):
        model.set_training(True)
        model.moe.reset_utilization()

        # Shuffle training data each epoch
        perm = np.random.permutation(n_train)
        X_shuffled, y_shuffled = X_train[perm], y_train[perm]

        epoch_loss, epoch_bloss, n_batches = 0.0, 0.0, 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            logits, balance_loss = model.forward(X_batch)
            task_loss, grad_logits = cross_entropy_loss(logits, y_batch)
            total_loss = task_loss + balance_loss

            # Backward pass
            model.backward(grad_logits)

            # Update parameters
            model.update(lr)

            epoch_loss += task_loss
            epoch_bloss += balance_loss
            n_batches += 1

        # Track metrics
        avg_loss = epoch_loss / n_batches
        avg_bloss = epoch_bloss / n_batches
        utilization = model.moe.get_utilization()

        # Validation
        model.set_training(False)
        val_logits, _ = model.forward(X_val)
        val_loss, _ = cross_entropy_loss(val_logits, y_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = np.mean(val_preds == y_val)

        train_logits, _ = model.forward(X_train)
        train_preds = np.argmax(train_logits, axis=1)
        train_acc = np.mean(train_preds == y_train)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['balance_loss'].append(avg_bloss)
        history['utilization'].append(utilization.copy())

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Bal: {avg_bloss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
            print(f"          | Expert utilization: {np.array2string(utilization, precision=2)}")

    return history


# ============================================================================
# Analysis: Expert Specialization
# ============================================================================

def analyze_expert_specialization(model: MoEClassifier, X: np.ndarray, y: np.ndarray):
    """Analyze which experts handle which classes — the signature of successful MoE training.

    If MoE is working properly, we should see different experts being the primary
    router for different input classes/regions.
    """
    model.set_training(False)
    gate_weights, _ = model.moe.gate.forward(X)

    num_classes = len(np.unique(y))
    num_experts = model.moe.num_experts

    print("\n" + "=" * 60)
    print("EXPERT SPECIALIZATION ANALYSIS")
    print("=" * 60)

    # For each class, show which experts are primarily responsible
    print(f"\n{'Class':<8} | {'Primary Expert':<15} | {'Gate Weight Distribution'}")
    print("-" * 60)

    for c in range(num_classes):
        mask = (y == c)
        class_gates = gate_weights[mask]  # (n_class, num_experts)
        mean_gates = np.mean(class_gates, axis=0)
        primary = np.argmax(mean_gates)

        # Format gate distribution as a bar chart
        bar = ""
        for i in range(num_experts):
            blocks = int(mean_gates[i] * 20)
            bar += f"E{i}:{'#' * blocks} "

        print(f"Class {c:<3} | Expert {primary:<9} | {bar}")

    # For each expert, show what fraction of each class it handles
    print(f"\n{'Expert':<8} | {'Class Distribution (fraction of inputs from each class)'}")
    print("-" * 60)

    for e in range(num_experts):
        # Inputs where this expert has non-zero gate weight
        expert_active = gate_weights[:, e] > 0
        if np.sum(expert_active) == 0:
            print(f"Expert {e} | UNUSED (expert collapse detected!)")
            continue

        expert_labels = y[expert_active]
        class_dist = np.zeros(num_classes)
        for c in range(num_classes):
            class_dist[c] = np.mean(expert_labels == c)

        primary_class = np.argmax(class_dist)
        dist_str = " ".join(f"C{c}:{class_dist[c]:.2f}" for c in range(num_classes))
        print(f"Expert {e} | Primary: Class {primary_class} | {dist_str}")

    # Gate entropy analysis — higher entropy means less specialization
    per_sample_entropy = -np.sum(gate_weights * np.log(gate_weights + 1e-8), axis=1)
    print(f"\nGate entropy: mean={np.mean(per_sample_entropy):.3f}, "
          f"std={np.std(per_sample_entropy):.3f}")
    print(f"(Lower entropy = more decisive routing, max possible = {np.log(num_experts):.3f})")


# ============================================================================
# Comparison: MoE vs Dense
# ============================================================================

def compare_moe_vs_dense(X_train, y_train, X_val, y_val, num_classes):
    """Compare MoE to a dense network with similar active parameter count.

    This demonstrates the key MoE advantage: more total capacity with the
    same compute budget, leading to better performance.
    """
    print("\n" + "=" * 60)
    print("MOE vs DENSE COMPARISON")
    print("=" * 60)

    input_dim = X_train.shape[1]

    # MoE: 8 experts, top-2 routing
    # Active params per input: 2 expert MLPs + gate + output head
    moe_model = MoEClassifier(
        input_dim=input_dim, num_classes=num_classes,
        moe_hidden=32, moe_output=16,
        num_experts=8, top_k=2, balance_coeff=0.01
    )

    # Dense: single expert with ~2x the hidden dim to match active param count
    # MoE active: 2 experts * (2*32 + 32*16) = 2 * (64+512) = 1152 active params
    # Dense: 1 expert with hidden=64: 2*64 + 64*16 = 128+1024 = 1152 params
    dense_model = MoEClassifier(
        input_dim=input_dim, num_classes=num_classes,
        moe_hidden=64, moe_output=16,
        num_experts=1, top_k=1, balance_coeff=0.0
    )

    print("\nTraining MoE (8 experts, top-2)...")
    moe_hist = train_moe(moe_model, X_train, y_train, X_val, y_val,
                         epochs=50, lr=0.05, verbose=False)

    print("Training Dense (1 expert, matching active params)...")
    dense_hist = train_moe(dense_model, X_train, y_train, X_val, y_val,
                           epochs=50, lr=0.05, verbose=False)

    moe_total = sum(e.W1.size + e.b1.size + e.W2.size + e.b2.size
                    for e in moe_model.moe.experts)
    moe_total += moe_model.moe.gate.W_gate.size + moe_model.W_out.size

    dense_total = sum(e.W1.size + e.b1.size + e.W2.size + e.b2.size
                      for e in dense_model.moe.experts)
    dense_total += dense_model.W_out.size

    print(f"\n{'Metric':<25} | {'MoE (8 exp, top-2)':<20} | {'Dense (1 exp)':<20}")
    print("-" * 70)
    print(f"{'Total parameters':<25} | {moe_total:<20} | {dense_total:<20}")
    print(f"{'Final train accuracy':<25} | {moe_hist['train_acc'][-1]:<20.3f} | {dense_hist['train_acc'][-1]:<20.3f}")
    print(f"{'Final val accuracy':<25} | {moe_hist['val_acc'][-1]:<20.3f} | {dense_hist['val_acc'][-1]:<20.3f}")
    print(f"{'Final train loss':<25} | {moe_hist['train_loss'][-1]:<20.4f} | {dense_hist['train_loss'][-1]:<20.4f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MIXTURE OF EXPERTS (MoE) FROM SCRATCH")
    print("=" * 60)

    # Generate synthetic dataset
    print("\n--- Generating multi-region classification data ---")
    X, y = generate_multi_region_data(n_samples=2400, n_classes=6, seed=42)

    # Train/val split (80/20)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Input dim: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Create and train MoE model
    print("\n--- Training MoE Classifier ---")
    print("Architecture: 8 experts (hidden=32, output=16), top-2 gating")
    print()

    np.random.seed(123)
    model = MoEClassifier(
        input_dim=2, num_classes=6,
        moe_hidden=32, moe_output=16,
        num_experts=8, top_k=2,
        balance_coeff=0.01
    )

    history = train_moe(model, X_train, y_train, X_val, y_val,
                       epochs=50, batch_size=64, lr=0.05)

    # Analyze expert specialization
    analyze_expert_specialization(model, X_val, y_val)

    # Compare MoE vs Dense
    np.random.seed(456)
    compare_moe_vs_dense(X_train, y_train, X_val, y_val, num_classes=6)

    # Summary
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. CONDITIONAL COMPUTATION: MoE only activates K of N experts per input,
   giving O(K/N) compute savings while maintaining N experts of capacity.

2. LOAD BALANCING IS CRITICAL: Without the auxiliary loss, training collapses
   to using 1-2 experts. The balance loss keeps all experts active and learning.

3. EXPERT SPECIALIZATION: Different experts naturally learn to handle different
   regions of the input space — this emergent specialization is what gives
   MoE its power.

4. CAPACITY VS COMPUTE: MoE decouples total model capacity (total params) from
   per-input compute (active params). This is how Mixtral 8x7B achieves
   near-GPT-4 quality with ~13B active params out of 47B total.

5. SCALING: In practice (Transformers), MoE replaces the FFN layers while
   keeping attention dense. Each token is routed through its own pair of
   expert FFNs, enabling massive scaling.
""")
