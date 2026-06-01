"""
Day 058: Policy Gradient Methods — REINFORCE Algorithm

A complete from-scratch implementation of the REINFORCE policy gradient algorithm
using only NumPy. We train a neural network policy to solve CartPole-v1.

Key ideas:
- The policy is a neural network that maps states → action probabilities
- We collect full episodes, compute returns-to-go, and update the policy
  to increase the probability of actions that led to high returns
- This is the foundation of all modern policy optimization (PPO, SAC, A3C)

No PyTorch, no TensorFlow — just NumPy and manual backpropagation so you
can see exactly how every gradient flows.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


# =============================================================================
# Neural Network Building Blocks
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x). Simple, effective, and has nice gradient properties."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where x > 0, 0 elsewhere. The discontinuity at 0 doesn't matter in practice."""
    return (x > 0).astype(float)


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert raw logits to probabilities: exp(x_i) / sum(exp(x_j)).

    We subtract max(logits) for numerical stability — this doesn't change the
    output (it cancels in numerator/denominator) but prevents overflow when
    logits are large.
    """
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


# =============================================================================
# Policy Network
# =============================================================================

class PolicyNetwork:
    """
    A 2-layer MLP that maps states to action probabilities.

    Architecture: state(4) → hidden(32) → ReLU → output(2) → softmax

    Why this architecture?
    - CartPole has 4 state dimensions and 2 actions — a tiny problem
    - 32 hidden units is plenty for this; larger networks would overfit
    - ReLU is standard; softmax converts logits to valid probabilities

    We store weights and implement forward/backward passes manually.
    This is the same math that PyTorch autograd does for you, made explicit.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, seed: Optional[int] = None):
        """
        Initialize with Xavier/Glorot initialization.

        Xavier init sets weight scale to sqrt(2 / (fan_in + fan_out)), which keeps
        the variance of activations roughly constant across layers. Without this,
        signals either explode or vanish as they propagate through the network.
        """
        if seed is not None:
            np.random.seed(seed)

        # Xavier initialization for stable training
        # W1: (state_dim, hidden_dim), b1: (hidden_dim,)
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / (state_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        # W2: (hidden_dim, action_dim), b2: (action_dim,)
        self.W2 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / (hidden_dim + action_dim))
        self.b2 = np.zeros(action_dim)

        # Cache for backward pass — we need these intermediate values for gradients
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass: state → probabilities over actions.

        We cache intermediate values (pre-activation, post-activation) because
        backpropagation needs them. This is the fundamental reason neural network
        training is memory-intensive — you must store O(params) intermediate values
        per sample.
        """
        # Layer 1: linear transformation + ReLU
        z1 = state @ self.W1 + self.b1          # (hidden_dim,)
        a1 = relu(z1)                             # (hidden_dim,)

        # Layer 2: linear transformation + softmax
        z2 = a1 @ self.W2 + self.b2              # (action_dim,)
        probs = softmax(z2)                        # (action_dim,) — valid probability distribution

        # Cache everything needed for backprop
        self.cache = {
            'state': state,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'probs': probs
        }

        return probs

    def backward(self, action: int, advantage: float) -> Dict[str, np.ndarray]:
        """
        Compute gradients of the policy gradient loss.

        The "loss" for policy gradients is: -log π(a|s) · advantage
        (negative because we want gradient ASCENT but optimizers do descent)

        For the chosen action a with probability p_a:
            ∂(-log p_a)/∂z2_i = p_i - 1{i == a}

        This is the same softmax-cross-entropy gradient you'd see in classification,
        but weighted by the advantage (how much better/worse than average this action was).

        Args:
            action: which action was taken (index)
            advantage: Gₜ - baseline, the signal for how good this action was

        Returns:
            Dictionary of parameter gradients {name: gradient_array}
        """
        state = self.cache['state']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        probs = self.cache['probs']

        # Gradient of -log π(a|s) w.r.t. logits z2
        # For softmax + cross-entropy: d_z2 = probs - one_hot(action)
        d_z2 = probs.copy()                       # (action_dim,)
        d_z2[action] -= 1.0
        # Scale by advantage — this is what makes it a POLICY GRADIENT update
        # rather than a supervised learning update
        d_z2 *= advantage                          # positive advantage → decrease loss → increase prob

        # Gradients for W2, b2 (layer 2)
        # z2 = a1 @ W2 + b2, so:
        dW2 = np.outer(a1, d_z2)                   # (hidden_dim, action_dim)
        db2 = d_z2                                  # (action_dim,)

        # Backprop through layer 2 to layer 1
        d_a1 = d_z2 @ self.W2.T                    # (hidden_dim,)

        # Backprop through ReLU
        d_z1 = d_a1 * relu_derivative(z1)          # (hidden_dim,) — gradient is zero where ReLU was inactive

        # Gradients for W1, b1 (layer 1)
        dW1 = np.outer(state, d_z1)                # (state_dim, hidden_dim)
        db1 = d_z1                                  # (hidden_dim,)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return a copy of all parameters (useful for inspection/saving)."""
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy()
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Load parameters (useful for restoring a saved policy)."""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()


# =============================================================================
# REINFORCE Agent
# =============================================================================

class REINFORCEAgent:
    """
    REINFORCE (Williams, 1992) policy gradient agent.

    The simplest policy gradient algorithm:
    1. Run the current policy to collect a full episode
    2. Compute return-to-go Gₜ at each timestep
    3. Normalize returns (simple baseline)
    4. Update policy: θ += α · ∇_θ log π(aₜ|sₜ) · (Gₜ - baseline)

    Why REINFORCE specifically?
    - It's the purest form of policy gradient — no value function approximation,
      no bootstrapping, no replay buffer
    - Every modern policy gradient method is a variance-reduced, stabilized
      version of REINFORCE
    - Understanding REINFORCE deeply means understanding the core of PPO, A3C, etc.
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

        # Episode memory — cleared after each update
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Sample an action from the policy distribution.

        This is stochastic — we SAMPLE from the distribution rather than taking
        the argmax. This is essential for exploration: if we always took the most
        probable action, we'd never discover that less-likely actions might be better.

        Returns:
            action: the sampled action index
            probs: the full probability distribution (for logging/debugging)
        """
        probs = self.policy.forward(state)

        # Sample from the categorical distribution
        # np.random.choice with p= does weighted random sampling
        action = np.random.choice(self.action_dim, p=probs)

        return action, probs

    def store_transition(self, state: np.ndarray, action: int, reward: float) -> None:
        """Store one timestep of experience. We need the full episode before updating."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_returns(self) -> np.ndarray:
        """
        Compute discounted return-to-go for each timestep.

        Gₜ = rₜ + γ·r_{t+1} + γ²·r_{t+2} + ... = rₜ + γ·G_{t+1}

        We compute this backwards for O(T) efficiency instead of O(T²).
        Working backwards: start from the last reward and accumulate.

        The discount factor γ < 1 serves two purposes:
        1. Mathematical: ensures the sum converges for infinite-horizon problems
        2. Practical: makes the agent prioritize near-term rewards, which have
           less uncertainty and are easier to credit to recent actions
        """
        T = len(self.rewards)
        returns = np.zeros(T)

        # Bootstrap: last return is just the last reward
        returns[T - 1] = self.rewards[T - 1]

        # Work backwards: Gₜ = rₜ + γ · G_{t+1}
        for t in range(T - 2, -1, -1):
            returns[t] = self.rewards[t] + self.gamma * returns[t + 1]

        return returns

    def normalize_returns(self, returns: np.ndarray) -> np.ndarray:
        """
        Normalize returns to have zero mean and unit variance.

        This is a simple but effective baseline technique:
        - Returns > mean → positive advantage → increase action probability
        - Returns < mean → negative advantage → decrease action probability

        Without normalization, if all returns are positive (common in CartPole
        where every step gives +1), ALL actions get reinforced. Normalization
        ensures roughly half get reinforced and half get discouraged.

        This is not a learned baseline (that would be Actor-Critic), but it
        captures the same intuition: compare to average performance.
        """
        mean = np.mean(returns)
        std = np.std(returns)
        if std < 1e-8:
            # If all returns are identical, no signal — return zeros
            return np.zeros_like(returns)
        return (returns - mean) / (std + 1e-8)

    def update(self) -> Dict[str, float]:
        """
        Perform one REINFORCE update using the stored episode.

        This is the heart of the algorithm:
        1. Compute returns-to-go (how good was the outcome from each timestep?)
        2. Normalize them (baseline subtraction)
        3. For each timestep, compute the policy gradient
        4. Accumulate gradients and apply a single parameter update

        Returns:
            Dictionary with training metrics (loss, episode_return, episode_length)
        """
        if len(self.rewards) == 0:
            return {'loss': 0.0, 'episode_return': 0.0, 'episode_length': 0}

        # Step 1: Compute discounted returns-to-go
        returns = self.compute_returns()
        episode_return = returns[0]  # G₀ is the total discounted return

        # Step 2: Normalize (simple baseline)
        advantages = self.normalize_returns(returns)

        # Step 3: Accumulate policy gradients across all timesteps
        grad_accum = {
            'W1': np.zeros_like(self.policy.W1),
            'b1': np.zeros_like(self.policy.b1),
            'W2': np.zeros_like(self.policy.W2),
            'b2': np.zeros_like(self.policy.b2),
        }

        total_loss = 0.0
        T = len(self.rewards)

        for t in range(T):
            # Re-run forward pass to populate cache for this timestep
            probs = self.policy.forward(self.states[t])
            action = self.actions[t]
            advantage = advantages[t]

            # Policy gradient loss: -log π(a|s) · advantage
            log_prob = np.log(probs[action] + 1e-10)
            total_loss += -log_prob * advantage

            # Backpropagate to get parameter gradients for this timestep
            grads = self.policy.backward(action, advantage)

            # Accumulate — we'll apply one update at the end
            # (could also update per-timestep, but batching is more stable)
            for key in grad_accum:
                grad_accum[key] += grads[key]

        # Step 4: Apply accumulated gradients (gradient descent on the loss = gradient ascent on expected return)
        # We use simple SGD here. Adam would be better for stability but adds complexity.
        # Note: we sum (not average) over timesteps — each timestep contributes its
        # own gradient signal. The learning rate controls the step size.
        for key in grad_accum:
            param = getattr(self.policy, key)
            param -= self.lr * grad_accum[key]

        # Clear episode memory for next episode
        episode_length = len(self.rewards)
        self.states = []
        self.actions = []
        self.rewards = []

        return {
            'loss': total_loss / T,
            'episode_return': episode_return,
            'episode_length': episode_length
        }


# =============================================================================
# CartPole Environment (simplified, no gym dependency)
# =============================================================================

class CartPoleEnv:
    """
    CartPole-v1 environment implemented from scratch.

    A pole is attached to a cart on a frictionless track. The agent applies
    a force of +1 or -1 to the cart at each timestep. The goal is to keep
    the pole balanced (upright) for as long as possible.

    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 (push left), 1 (push right)
    Reward: +1 for every timestep the pole stays upright

    This is the "hello world" of RL — simple enough to solve quickly but
    complex enough to require real learning.

    Physics: We use the exact same dynamics as OpenAI Gym's CartPole-v1,
    based on Barto, Sutton, and Anderson (1983). The equations come from
    Euler integration of the inverted pendulum equations of motion.
    """

    def __init__(self, seed: Optional[int] = None):
        # Physical constants (matching OpenAI Gym exactly)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # time step (seconds)

        # Termination thresholds
        self.x_threshold = 2.4          # cart position limit
        self.theta_threshold = 12 * np.pi / 180  # pole angle limit (12 degrees in radians)

        self.max_steps = 500  # CartPole-v1 has a 500-step limit
        self.rng = np.random.RandomState(seed)
        self.state: Optional[np.ndarray] = None
        self.steps = 0

    def reset(self) -> np.ndarray:
        """
        Reset to a random initial state near equilibrium.

        Small random perturbations ensure the agent sees diverse starting
        conditions, preventing it from memorizing a single trajectory.
        """
        # Random state in [-0.05, 0.05] for each dimension
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Apply an action and advance the physics by one timestep.

        The dynamics equations come from Newton's laws applied to the
        cart-pole system. The key insight is that the pole's angular
        acceleration depends on both the applied force AND the cart's
        acceleration (they're coupled), requiring simultaneous solution.

        Returns: (next_state, reward, done)
        """
        assert self.state is not None, "Must call reset() before step()"

        x, x_dot, theta, theta_dot = self.state

        # Applied force: left (-10N) or right (+10N)
        force = self.force_mag if action == 1 else -self.force_mag

        # Physics: coupled differential equations for cart-pole system
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Intermediate calculation (appears in both acceleration equations)
        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass

        # Angular acceleration of the pole
        # Derived from torque = I·α, accounting for the constraint that
        # the pole pivot moves with the cart
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos_theta**2 / self.total_mass)
        )

        # Linear acceleration of the cart
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        # Euler integration (simple but introduces small errors; RK4 would be better
        # but Gym uses Euler so we match it for compatibility)
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        # Episode terminates if:
        # 1. Cart goes off screen (|x| > 2.4)
        # 2. Pole falls too far (|θ| > 12°)
        # 3. Reached max steps (success!)
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
            or self.steps >= self.max_steps
        )

        # Reward: +1 for every step survived (incentivizes balancing)
        reward = 1.0

        return self.state.copy(), reward, done


# =============================================================================
# Training Loop
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

    The training loop is simple because REINFORCE is an on-policy, episodic algorithm:
    - Collect one episode → update → discard data → repeat
    - No replay buffer, no target network, no experience reuse

    This simplicity is both a strength (easy to implement correctly) and weakness
    (sample inefficient — each transition is used exactly once).

    Args:
        num_episodes: total episodes to train
        hidden_dim: neurons in hidden layer
        learning_rate: step size for gradient updates
        gamma: discount factor (0.99 means we care about ~100 future steps)
        seed: for reproducibility
        print_every: logging frequency
        solved_threshold: CartPole is "solved" at 195+ avg reward
        solved_window: number of episodes to average over

    Returns:
        Trained agent and list of episode rewards
    """
    env = CartPoleEnv(seed=seed)
    agent = REINFORCEAgent(
        state_dim=4,
        action_dim=2,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed
    )

    all_rewards: List[float] = []
    solved = False

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        # Collect a full episode
        while not done:
            action, probs = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            episode_reward += reward

        # Update policy using the complete episode
        metrics = agent.update()
        all_rewards.append(episode_reward)

        # Check if solved (average reward over last N episodes)
        if len(all_rewards) >= solved_window:
            avg_reward = np.mean(all_rewards[-solved_window:])
            if avg_reward >= solved_threshold and not solved:
                print(f"\n*** SOLVED at episode {episode}! "
                      f"Average reward: {avg_reward:.1f} over last {solved_window} episodes ***\n")
                solved = True

        # Logging
        if (episode + 1) % print_every == 0:
            recent_avg = np.mean(all_rewards[-print_every:])
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg(last {print_every}): {recent_avg:6.1f} | "
                  f"Loss: {metrics['loss']:.4f}")

    return agent, all_rewards


def evaluate_agent(agent: REINFORCEAgent, num_episodes: int = 10, seed: int = 123) -> List[float]:
    """
    Evaluate a trained agent (no exploration noise, but we keep stochastic
    policy since that's what was trained).

    In practice you might switch to the greedy (argmax) policy for evaluation,
    but for CartPole the stochastic policy works fine — a well-trained policy
    puts >99% probability on the correct action.
    """
    env = CartPoleEnv(seed=seed)
    rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, probs = agent.select_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return rewards


# =============================================================================
# Main — Demonstration
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Day 058: Policy Gradient Methods — REINFORCE on CartPole")
    print("=" * 70)

    # --- Demonstrate core components first ---
    print("\n--- 1. Policy Network Forward Pass ---")
    policy = PolicyNetwork(state_dim=4, hidden_dim=32, action_dim=2, seed=42)
    test_state = np.array([0.01, -0.02, 0.03, -0.01])  # near-equilibrium state
    probs = policy.forward(test_state)
    print(f"State: {test_state}")
    print(f"Action probabilities: left={probs[0]:.4f}, right={probs[1]:.4f}")
    print(f"Sum of probabilities: {np.sum(probs):.6f} (should be 1.0)")

    print("\n--- 2. Softmax Properties ---")
    print("Softmax converts arbitrary logits to a valid probability distribution.")
    for logits in [np.array([0, 0]), np.array([2, -2]), np.array([100, 100])]:
        p = softmax(logits)
        print(f"  logits={logits} → probs={p} (sum={np.sum(p):.6f})")

    print("\n--- 3. Return Computation ---")
    agent = REINFORCEAgent(state_dim=4, action_dim=2, seed=42)
    # Simulate a short episode: 5 steps with reward=1 each
    dummy_state = np.zeros(4)
    for _ in range(5):
        agent.store_transition(dummy_state, 0, 1.0)
    returns = agent.compute_returns()
    print(f"Rewards:  [1.0, 1.0, 1.0, 1.0, 1.0]  (γ=0.99)")
    print(f"Returns:  {returns}")
    print(f"G₀ = 1 + 0.99 + 0.99² + 0.99³ + 0.99⁴ = {1 + 0.99 + 0.99**2 + 0.99**3 + 0.99**4:.4f}")
    print(f"Computed: {returns[0]:.4f}")
    agent.states, agent.actions, agent.rewards = [], [], []  # reset

    print("\n--- 4. Training REINFORCE on CartPole ---")
    print("Training for 800 episodes (this is pure NumPy, no GPU)...\n")

    trained_agent, rewards = train_reinforce(
        num_episodes=800,
        hidden_dim=64,
        learning_rate=0.005,
        gamma=0.99,
        seed=42,
        print_every=100
    )

    print(f"\n--- 5. Training Summary ---")
    # Show learning progress in windows
    for start in range(0, len(rewards), 200):
        end = min(start + 200, len(rewards))
        window_avg = np.mean(rewards[start:end])
        print(f"Episodes {start+1:4d}-{end:4d}: avg reward = {window_avg:.1f}")

    print(f"\n--- 6. Evaluation (10 episodes) ---")
    eval_rewards = evaluate_agent(trained_agent, num_episodes=10, seed=123)
    for i, r in enumerate(eval_rewards):
        print(f"  Episode {i+1}: reward = {r:.0f}")
    print(f"  Average: {np.mean(eval_rewards):.1f}")
    print(f"  Min: {np.min(eval_rewards):.0f}, Max: {np.max(eval_rewards):.0f}")

    print("\n--- 7. Key Takeaways ---")
    print("• REINFORCE learns a POLICY directly: state → P(action)")
    print("• The gradient ∇log π(a|s) · G tells us: 'make good actions more likely'")
    print("• Normalizing returns acts as a baseline, reducing variance")
    print("• On-policy: each episode is used once, then discarded (sample inefficient)")
    print("• This is the foundation — PPO, A3C, SAC all build on this idea")

    print("\n" + "=" * 70)
    print("Done! Next: explore Actor-Critic (learned baseline) and PPO (clipped updates)")
    print("=" * 70)
