"""
Day 058: Policy Gradient Methods — REINFORCE Algorithm (Your Implementation)

Implement the REINFORCE policy gradient algorithm from scratch using NumPy.
You'll build a neural network policy, collect episodes, compute returns,
and update the policy to solve CartPole.

Hint: The key equation is ∇_θ J(θ) = E[ ∇_θ log π(a|s) · Gₜ ]
This says: make good actions more probable, bad actions less probable.

Run tests: python3 -m pytest tests.py
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


# =============================================================================
# Neural Network Building Blocks
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    raise NotImplementedError("TODO: implement ReLU activation")


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where x > 0, 0 elsewhere."""
    raise NotImplementedError("TODO: implement ReLU derivative")


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert raw logits to probabilities.

    Hint: Subtract max(logits) before exp() for numerical stability.
    The subtraction cancels out mathematically but prevents overflow.
    """
    raise NotImplementedError("TODO: implement softmax")


# =============================================================================
# Policy Network
# =============================================================================

class PolicyNetwork:
    """
    A 2-layer MLP: state → hidden (ReLU) → action logits (softmax).

    Architecture: state(4) → hidden(32) → ReLU → output(2) → softmax

    Hint: Use Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
    Hint: Cache intermediate values in forward() — you need them for backward()
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, seed: Optional[int] = None):
        """
        Initialize weights with Xavier/Glorot initialization.

        Hint: W1 shape is (state_dim, hidden_dim), b1 shape is (hidden_dim,)
              W2 shape is (hidden_dim, action_dim), b2 shape is (action_dim,)
        """
        if seed is not None:
            np.random.seed(seed)

        raise NotImplementedError("TODO: initialize W1, b1, W2, b2 and self.cache")

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass: state → action probabilities.

        Hint: z1 = state @ W1 + b1, then ReLU, then z2 = a1 @ W2 + b2, then softmax.
        Don't forget to cache state, z1, a1, z2, probs for the backward pass.
        """
        raise NotImplementedError("TODO: implement forward pass")

    def backward(self, action: int, advantage: float) -> Dict[str, np.ndarray]:
        """
        Compute gradients of -log π(a|s) · advantage w.r.t. all parameters.

        Hint: The softmax + cross-entropy gradient w.r.t. logits z2 is:
              d_z2 = (probs - one_hot(action)) * advantage

        Then backprop through layer 2 (get dW2, db2),
        through ReLU (multiply by relu_derivative),
        through layer 1 (get dW1, db1).

        Return dict with keys 'W1', 'b1', 'W2', 'b2'.
        """
        raise NotImplementedError("TODO: implement backward pass")

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return a copy of all parameters."""
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy()
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Load parameters."""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()


# =============================================================================
# REINFORCE Agent
# =============================================================================

class REINFORCEAgent:
    """
    REINFORCE policy gradient agent.

    Algorithm:
    1. Collect full episode using current policy
    2. Compute return-to-go Gₜ at each timestep
    3. Normalize returns (simple baseline)
    4. Update: θ -= lr * ∇_θ(-log π(a|s) · advantage) for each timestep

    Hint: This is on-policy — you must collect fresh episodes each time.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        seed: Optional[int] = None
    ):
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim, seed)
        self.lr = learning_rate
        self.gamma = gamma
        self.action_dim = action_dim

        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Sample an action from the policy's probability distribution.

        Hint: Use self.policy.forward(state) to get probabilities,
              then np.random.choice with p=probs to sample.
        """
        raise NotImplementedError("TODO: implement action selection")

    def store_transition(self, state: np.ndarray, action: int, reward: float) -> None:
        """Store one timestep of experience."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_returns(self) -> np.ndarray:
        """
        Compute discounted return-to-go for each timestep.

        Gₜ = rₜ + γ·G_{t+1}

        Hint: Work backwards from the last timestep. Start with G_T = r_T,
              then G_{T-1} = r_{T-1} + γ·G_T, etc.
        """
        raise NotImplementedError("TODO: implement return computation")

    def normalize_returns(self, returns: np.ndarray) -> np.ndarray:
        """
        Normalize returns to zero mean and unit variance.

        Hint: (returns - mean) / (std + epsilon)
        Handle the edge case where std ≈ 0 (all returns identical).
        """
        raise NotImplementedError("TODO: implement return normalization")

    def update(self) -> Dict[str, float]:
        """
        Perform one REINFORCE update using the stored episode.

        Steps:
        1. Compute returns-to-go
        2. Normalize them (baseline)
        3. For each timestep: forward pass, backward pass, accumulate gradients
        4. Apply accumulated gradients: param -= lr * avg_gradient

        Hint: Average the gradients over timesteps before applying the update.
        Don't forget to clear the episode memory at the end!
        """
        raise NotImplementedError("TODO: implement REINFORCE update")


# =============================================================================
# CartPole Environment
# =============================================================================

class CartPoleEnv:
    """
    CartPole-v1 environment (from scratch, no gym dependency).

    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 (left), 1 (right)
    Reward: +1 per timestep

    Hint: The physics is just Newton's laws applied to an inverted pendulum
    on a cart. The key equations couple the cart and pole accelerations.
    """

    def __init__(self, seed: Optional[int] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        self.x_threshold = 2.4
        self.theta_threshold = 12 * np.pi / 180

        self.max_steps = 500
        self.rng = np.random.RandomState(seed)
        self.state: Optional[np.ndarray] = None
        self.steps = 0

    def reset(self) -> np.ndarray:
        """Reset to a random state near equilibrium."""
        raise NotImplementedError("TODO: implement reset")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Apply action and advance physics by one timestep.

        Hint: The coupled equations are:
          temp = (force + polemass_length * theta_dot² * sin(θ)) / total_mass
          θ_acc = (g·sin(θ) - cos(θ)·temp) / (length * (4/3 - m_pole·cos²(θ)/total_mass))
          x_acc = temp - polemass_length * θ_acc * cos(θ) / total_mass

        Then Euler integration: x += tau * x_dot, x_dot += tau * x_acc, etc.

        Returns: (next_state, reward=1.0, done)
        """
        raise NotImplementedError("TODO: implement step")


# =============================================================================
# Training
# =============================================================================

def train_reinforce(
    num_episodes: int = 1000,
    hidden_dim: int = 64,
    learning_rate: float = 0.005,
    gamma: float = 0.99,
    seed: int = 42,
    print_every: int = 50,
    solved_threshold: float = 195.0,
    solved_window: int = 100
) -> Tuple[REINFORCEAgent, List[float]]:
    """
    Train a REINFORCE agent on CartPole.

    Hint: The loop is simple:
      for each episode:
        reset env, collect full episode, call agent.update()
    """
    raise NotImplementedError("TODO: implement training loop")


def evaluate_agent(agent: REINFORCEAgent, num_episodes: int = 10, seed: int = 123) -> List[float]:
    """Evaluate a trained agent over multiple episodes."""
    raise NotImplementedError("TODO: implement evaluation")


if __name__ == '__main__':
    print("Day 058: Policy Gradient Methods — REINFORCE")
    print("=" * 50)

    # Test your building blocks
    print("\n1. Test softmax:")
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    print(f"   softmax([1, 2, 3]) = {probs}")

    print("\n2. Test policy network:")
    policy = PolicyNetwork(state_dim=4, hidden_dim=32, action_dim=2, seed=42)
    state = np.array([0.01, -0.02, 0.03, -0.01])
    probs = policy.forward(state)
    print(f"   P(actions | state) = {probs}")

    print("\n3. Test return computation:")
    agent = REINFORCEAgent(state_dim=4, action_dim=2, seed=42)
    for _ in range(5):
        agent.store_transition(np.zeros(4), 0, 1.0)
    returns = agent.compute_returns()
    print(f"   Returns for [1,1,1,1,1] with γ=0.99: {returns}")
    agent.states, agent.actions, agent.rewards = [], [], []

    print("\n4. Train on CartPole:")
    trained_agent, rewards = train_reinforce(num_episodes=800, seed=42, print_every=100)

    print("\n5. Evaluate:")
    eval_rewards = evaluate_agent(trained_agent, num_episodes=10, seed=123)
    print(f"   Average reward: {np.mean(eval_rewards):.1f}")
