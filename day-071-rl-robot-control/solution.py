"""
Day 71: Reinforcement Learning for Robot Control
=================================================

A DQN agent learns to control a 2-link planar robot arm to reach target positions.

This combines:
- RL fundamentals (Day 57-58): Q-learning, neural function approximation
- Robotics (Day 8, 42): forward kinematics, trajectory planning
- The key insight: the agent discovers control policies through trial and error

Architecture:
- Environment: 2-link arm with simplified rigid-body dynamics
- Agent: DQN with experience replay and target network
- State: [θ₁, θ₂, θ̇₁, θ̇₂, x_target, y_target] (6D continuous)
- Action: discretized torques {-1, 0, +1} per joint (9 actions)
"""

import numpy as np
import math
import random
from collections import deque
from typing import Tuple, List, Optional, NamedTuple


# ---------------------------------------------------------------------------
# 1. ROBOT ARM ENVIRONMENT
# ---------------------------------------------------------------------------

class RobotArmEnv:
    """
    2-link planar robot arm environment (gym-like interface).

    Physics: simplified rigid-body dynamics using Euler integration.
    Each link is a uniform rod with configurable length and mass.
    Gravity is ignored (horizontal plane) to keep the learning problem tractable —
    adding gravity makes exploration much harder without changing the core concepts.
    """

    def __init__(
        self,
        link1_length: float = 1.0,
        link2_length: float = 1.0,
        link1_mass: float = 1.0,
        link2_mass: float = 1.0,
        dt: float = 0.05,
        max_steps: int = 200,
        target_threshold: float = 0.15,
        damping: float = 0.2,
        max_angular_vel: float = 3.0,
        torque_scale: float = 2.0,
    ):
        self.l1 = link1_length
        self.l2 = link2_length
        self.m1 = link1_mass
        self.m2 = link2_mass
        self.dt = dt
        self.max_steps = max_steps
        self.target_threshold = target_threshold
        # Damping prevents infinite acceleration — models real joint friction
        self.damping = damping
        self.max_angular_vel = max_angular_vel
        self.torque_scale = torque_scale

        # State: [θ₁, θ₂, θ̇₁, θ̇₂]
        self.theta = np.zeros(2)
        self.theta_dot = np.zeros(2)
        self.target = np.zeros(2)
        self.steps = 0

        # Discretized action space: {-1, 0, +1} for each of 2 joints = 9 actions
        # This is a simplification — real systems use continuous torques (DDPG/SAC)
        self.action_map = []
        for t1 in [-1, 0, 1]:
            for t2 in [-1, 0, 1]:
                self.action_map.append((t1, t2))
        self.n_actions = len(self.action_map)  # 9

        # State dimension: θ₁, θ₂, θ̇₁, θ̇₂, x_target, y_target
        self.state_dim = 6

    def forward_kinematics(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute end-effector position from joint angles.
        Same math as Day 8 — the foundation of all robot control.

        For a 2-link planar arm:
            x = l1*cos(θ₁) + l2*cos(θ₁ + θ₂)
            y = l1*sin(θ₁) + l2*sin(θ₁ + θ₂)
        """
        x = self.l1 * math.cos(theta[0]) + self.l2 * math.cos(theta[0] + theta[1])
        y = self.l1 * math.sin(theta[0]) + self.l2 * math.sin(theta[0] + theta[1])
        return np.array([x, y])

    def _sample_reachable_target(self) -> np.ndarray:
        """
        Sample a target within the arm's reachable workspace.
        The workspace is an annulus: r ∈ [|l1 - l2|, l1 + l2].
        We sample slightly inside to avoid singularities at the boundaries.
        """
        r_min = abs(self.l1 - self.l2) + 0.2
        r_max = (self.l1 + self.l2) - 0.2
        r = random.uniform(r_min, r_max)
        angle = random.uniform(-math.pi, math.pi)
        return np.array([r * math.cos(angle), r * math.sin(angle)])

    def reset(self) -> np.ndarray:
        """Reset environment to a random initial state with a new target."""
        # Random initial joint angles — diverse starting configurations help exploration
        self.theta = np.array([
            random.uniform(-math.pi, math.pi),
            random.uniform(-math.pi, math.pi),
        ])
        self.theta_dot = np.zeros(2)
        self.target = self._sample_reachable_target()
        self.steps = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Construct the observation vector.
        We normalize angles to [-π, π] for stable learning — without this,
        the network sees discontinuities at ±π that confuse gradient descent.
        """
        # Normalize angles to [-π, π]
        theta_norm = np.array([
            math.atan2(math.sin(self.theta[0]), math.cos(self.theta[0])),
            math.atan2(math.sin(self.theta[1]), math.cos(self.theta[1])),
        ])
        return np.concatenate([
            theta_norm,
            self.theta_dot / self.max_angular_vel,  # Normalize velocities to ~[-1, 1]
            self.target,  # Target position (already in reasonable range)
        ])

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action, advance physics, compute reward.

        Physics model: simplified Newton-Euler dynamics.
        For a real 2-link arm, the mass matrix M(θ) is configuration-dependent
        (Coriolis and centrifugal terms). We simplify to independent joints with
        damping — good enough to learn the RL concepts without getting bogged
        down in dynamics derivation.
        """
        torques = np.array(self.action_map[action_idx], dtype=np.float64)
        torques *= self.torque_scale

        # Simplified dynamics: τ = I*α + damping*ω
        # Real dynamics would use the full manipulator equation M(θ)θ̈ + C(θ,θ̇)θ̇ = τ
        # The simplification means our "robot" is easier to control than a real one,
        # but the RL algorithm and training procedure are identical.
        alpha = torques - self.damping * self.theta_dot  # angular acceleration

        # Euler integration (simple but introduces energy drift at large dt)
        self.theta_dot += alpha * self.dt
        # Clamp angular velocity — prevents unrealistic speeds
        self.theta_dot = np.clip(self.theta_dot, -self.max_angular_vel, self.max_angular_vel)
        self.theta += self.theta_dot * self.dt

        self.steps += 1

        # Compute reward
        ee_pos = self.forward_kinematics(self.theta)
        distance = np.linalg.norm(ee_pos - self.target)
        reached = distance < self.target_threshold

        # Reward shaping: continuous signal guides exploration
        # Without shaping, the agent almost never accidentally reaches the target
        reward = -distance  # Primary signal: get closer
        reward -= 0.01 * np.sum(np.abs(torques))  # Energy penalty — prefer efficient motion
        reward -= 0.05 * np.sum(np.abs(self.theta_dot))  # Smoothness — penalize jerky motion

        if reached:
            reward += 100.0  # Large bonus for success

        done = reached or self.steps >= self.max_steps

        info = {
            "distance": distance,
            "reached": reached,
            "ee_pos": ee_pos.copy(),
            "steps": self.steps,
        }

        return self._get_state(), reward, done, info


# ---------------------------------------------------------------------------
# 2. EXPERIENCE REPLAY BUFFER
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """A single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size FIFO buffer that stores transitions and samples random mini-batches.

    Why replay is essential:
    1. Breaks temporal correlation — consecutive transitions are highly correlated,
       which violates the i.i.d. assumption of SGD. Random sampling decorrelates.
    2. Data efficiency — each transition is used for multiple gradient updates.
    3. Prevents catastrophic forgetting — without replay, the network overwrites
       what it learned about old states as it trains on new ones.

    Buffer size tradeoff:
    - Too small (< 1000): rapid forgetting of old experiences
    - Too large (> 1M): stale data slows adaptation to improved policy
    - Sweet spot depends on environment complexity; 10K-100K is typical for simple tasks
    """

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 3. NEURAL NETWORK (from scratch — no PyTorch dependency)
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    Simple feedforward network with ReLU activations.

    Architecture: input(6) → hidden(128) → hidden(128) → output(9)

    Built from scratch to show what's under the hood. In production,
    you'd use PyTorch/JAX for GPU acceleration and automatic differentiation.

    Weight initialization: He initialization (scale by sqrt(2/fan_in)) for ReLU.
    This prevents vanishing/exploding gradients in deep networks.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 1e-3):
        self.lr = lr

        # Xavier/Glorot initialization — balances signal magnitude across layers
        # He init (sqrt(2/fan_in)) can cause overflow in from-scratch networks
        # without framework-level numerics (float32 accumulators, fused ops).
        # Xavier (sqrt(1/fan_in)) is more conservative and sufficient here.
        self.W1 = np.random.randn(input_dim, hidden_dim) * math.sqrt(1.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * math.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * math.sqrt(1.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Stores intermediates for backprop."""
        self._x = x
        # Clip activations to prevent overflow — the from-scratch matmul can produce
        # extreme values that NaN-propagate through subsequent layers.
        # Framework networks (PyTorch/JAX) handle this via float32 accumulators.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            self._z1 = np.clip(np.nan_to_num(x @ self.W1 + self.b1), -50, 50)
            self._a1 = np.maximum(0, self._z1)  # ReLU
            self._z2 = np.clip(np.nan_to_num(self._a1 @ self.W2 + self.b2), -50, 50)
            self._a2 = np.maximum(0, self._z2)  # ReLU
            self._out = np.clip(np.nan_to_num(self._a2 @ self.W3 + self.b3), -500, 500)
        return self._out

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backpropagation with gradient descent.

        The chain rule gives us gradients for each layer:
            ∂L/∂W3 = a2ᵀ · ∂L/∂out
            ∂L/∂W2 = a1ᵀ · (∂L/∂z2)
            ∂L/∂W1 = xᵀ · (∂L/∂z1)

        ReLU gradient: 1 if z > 0, else 0 (kills gradient for inactive neurons).
        """
        batch_size = grad_output.shape[0]

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            # Layer 3 gradients
            dW3 = self._a2.T @ grad_output / batch_size
            db3 = np.mean(grad_output, axis=0)

            # Backprop through layer 3 → layer 2
            d_a2 = grad_output @ self.W3.T
            d_z2 = d_a2 * (self._z2 > 0)  # ReLU gradient

            dW2 = self._a1.T @ d_z2 / batch_size
            db2 = np.mean(d_z2, axis=0)

            # Backprop through layer 2 → layer 1
            d_a1 = d_z2 @ self.W2.T
            d_z1 = d_a1 * (self._z1 > 0)  # ReLU gradient

            dW1 = self._x.T @ d_z1 / batch_size
            db1 = np.mean(d_z1, axis=0)

        # Gradient clipping — prevents exploding gradients that destabilize training
        # Without this, a single bad batch can blow up the weights
        def clip(g: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
            # Replace NaN/Inf with zeros — prevents weight corruption from overflow
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            norm = np.linalg.norm(g)
            if norm > max_norm:
                return g * (max_norm / norm)
            return g

        dW1, dW2, dW3 = clip(dW1), clip(dW2), clip(dW3)
        db1, db2, db3 = clip(db1), clip(db2), clip(db3)

        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def copy_weights_from(self, other: 'NeuralNetwork') -> None:
        """Copy weights from another network (for target network updates)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


# ---------------------------------------------------------------------------
# 4. DQN AGENT
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent for robot arm control.

    Key components and why each exists:
    1. Q-network: approximates Q*(s,a) — the expected return of taking action a in state s
    2. Target network: stabilizes training by providing a slowly-changing TD target
    3. Replay buffer: breaks correlation and enables data reuse
    4. ε-greedy: balances exploration (try new things) vs exploitation (do what works)
    """

    def __init__(
        self,
        state_dim: int = 6,
        n_actions: int = 9,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.997,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 20,
    ):
        self.n_actions = n_actions
        self.gamma = gamma  # Discount factor — how much to value future rewards
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Two networks: online (trained every step) and target (updated periodically)
        # Without the target network, the TD target y = r + γ max Q(s', a') changes
        # every gradient step, creating a moving target that prevents convergence.
        self.q_network = NeuralNetwork(state_dim, hidden_dim, n_actions, lr)
        self.target_network = NeuralNetwork(state_dim, hidden_dim, n_actions, lr)
        self.target_network.copy_weights_from(self.q_network)

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.train_step_count = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        ε-greedy action selection.

        During training: with probability ε, take a random action (explore).
        During evaluation: always take the greedy action (exploit).

        The exploration-exploitation tradeoff is fundamental:
        - Too much exploration → wastes time on random actions
        - Too little → gets stuck in local optima, never finds better strategies
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # Greedy: pick action with highest Q-value
        q_values = self.q_network.forward(state.reshape(1, -1))
        # Guard against NaN from numerical overflow — fall back to random action
        if np.any(np.isnan(q_values)):
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(q_values[0]))

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient step.

        The DQN loss is:
            L = E[(y - Q(s, a; θ))²]
        where:
            y = r + γ * max_a' Q_target(s', a'; θ⁻)   (if not terminal)
            y = r                                        (if terminal)

        This is just MSE regression — predict the discounted future reward.
        The "deep" part is using a neural network instead of a table.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        # Vectorize the batch for efficient computation
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float64)

        # Clip rewards to prevent extreme values from destabilizing training
        # This is standard practice — rewards outside [-10, 110] are outliers
        rewards = np.clip(rewards, -10.0, 110.0)

        # Current Q-values for all actions
        current_q = self.q_network.forward(states)

        # Target Q-values using the TARGET network (not the online network!)
        # This is the key stability trick in DQN
        next_q = self.target_network.forward(next_states)
        max_next_q = np.max(next_q, axis=1)

        # Guard against NaN propagation — if network outputs NaN, skip this batch
        if np.any(np.isnan(current_q)) or np.any(np.isnan(next_q)):
            return None

        # Bellman target: y = r + γ * max Q_target(s', a') * (1 - done)
        # The (1 - done) term zeroes out future reward for terminal states
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        # Compute gradient only for the taken actions
        # For actions not taken, gradient is zero (we don't update those Q-values)
        target_q = current_q.copy()
        for i in range(self.batch_size):
            target_q[i, actions[i]] = targets[i]

        # MSE gradient: ∂L/∂Q = 2 * (Q - target) / batch_size
        grad = 2.0 * (current_q - target_q)
        # Clip gradient to prevent exploding updates
        grad = np.clip(grad, -1.0, 1.0)
        loss = np.mean((current_q[np.arange(self.batch_size), actions] - targets) ** 2)

        self.q_network.backward(grad)

        # Periodically sync target network
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)

        return loss

    def decay_epsilon(self) -> None:
        """
        Decay exploration rate after each episode.
        Exponential decay: ε *= decay_rate, clamped to ε_end.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------------------------

def train_agent(
    n_episodes: int = 500,
    print_every: int = 50,
    eval_episodes: int = 10,
) -> Tuple[DQNAgent, RobotArmEnv, dict]:
    """
    Train the DQN agent on the robot arm environment.

    Returns the trained agent, environment, and training metrics.
    """
    env = RobotArmEnv()
    agent = DQNAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=64,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=20,
    )

    # Metrics tracking
    episode_rewards = []
    episode_successes = []
    losses = []
    metrics = {"rewards": [], "successes": [], "avg_losses": []}

    print("=" * 70)
    print("Training DQN Agent for 2-Link Robot Arm Control")
    print("=" * 70)
    print(f"State dim: {env.state_dim}, Actions: {env.n_actions}")
    print(f"Target threshold: {env.target_threshold}")
    print(f"Max steps/episode: {env.max_steps}")
    print()

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_loss = []

        for step in range(env.max_steps):
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        episode_successes.append(1.0 if info["reached"] else 0.0)

        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        losses.append(avg_loss)

        # Print progress at intervals
        if (episode + 1) % print_every == 0:
            recent_rewards = episode_rewards[-print_every:]
            recent_success = episode_successes[-print_every:]
            recent_losses = losses[-print_every:]

            avg_reward = np.mean(recent_rewards)
            success_rate = np.mean(recent_success) * 100
            avg_l = np.mean(recent_losses)

            metrics["rewards"].append(avg_reward)
            metrics["successes"].append(success_rate)
            metrics["avg_losses"].append(avg_l)

            print(f"Episode {episode + 1:4d}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:8.1f} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Avg Loss: {avg_l:8.2f} | "
                  f"ε: {agent.epsilon:.3f}")

    return agent, env, metrics


# ---------------------------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------------------------

def evaluate_agent(agent: DQNAgent, env: RobotArmEnv, n_episodes: int = 20) -> dict:
    """
    Evaluate the trained agent with no exploration (ε = 0).

    Tracks success rate, average distance to target, and path efficiency.
    """
    successes = 0
    total_steps = 0
    total_distance = 0.0
    trajectories = []

    print("\n" + "=" * 70)
    print("Evaluating Trained Agent (ε = 0, pure exploitation)")
    print("=" * 70)

    for ep in range(n_episodes):
        state = env.reset()
        trajectory = [env.forward_kinematics(env.theta).copy()]
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, _, done, info = env.step(action)
            trajectory.append(info["ee_pos"].copy())

        successes += int(info["reached"])
        total_steps += info["steps"]
        total_distance += info["distance"]
        trajectories.append(trajectory)

        status = "REACHED" if info["reached"] else "MISSED"
        print(f"  Episode {ep + 1:2d}: {status} | "
              f"Final dist: {info['distance']:.3f} | "
              f"Steps: {info['steps']:3d} | "
              f"Target: ({env.target[0]:.2f}, {env.target[1]:.2f})")

    success_rate = successes / n_episodes * 100
    avg_steps = total_steps / n_episodes
    avg_dist = total_distance / n_episodes

    print(f"\n{'Results':=^70}")
    print(f"  Success rate: {success_rate:.1f}% ({successes}/{n_episodes})")
    print(f"  Avg final distance: {avg_dist:.4f}")
    print(f"  Avg steps: {avg_steps:.1f}")

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_distance": avg_dist,
        "trajectories": trajectories,
    }


# ---------------------------------------------------------------------------
# 7. ANALYSIS: Understanding what the agent learned
# ---------------------------------------------------------------------------

def analyze_q_values(agent: DQNAgent, env: RobotArmEnv) -> None:
    """
    Analyze Q-values to understand the learned policy.

    The Q-values tell us what the agent "thinks" about different situations.
    High Q-value = the agent expects high future reward from this state-action pair.
    """
    print("\n" + "=" * 70)
    print("Q-Value Analysis: What Did the Agent Learn?")
    print("=" * 70)

    # Test case: arm pointing right (θ=0,0), target in different positions
    test_configs = [
        ("Target ahead (easy)", np.array([0.0, 0.0, 0.0, 0.0, 1.5, 0.0])),
        ("Target behind (hard)", np.array([0.0, 0.0, 0.0, 0.0, -1.5, 0.0])),
        ("Target above", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5])),
        ("Already at target", np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])),
    ]

    action_names = [
        "(-1,-1)", "(-1, 0)", "(-1,+1)",
        "( 0,-1)", "( 0, 0)", "( 0,+1)",
        "(+1,-1)", "(+1, 0)", "(+1,+1)",
    ]

    for name, state in test_configs:
        q_values = agent.q_network.forward(state.reshape(1, -1))[0]
        best_action = np.argmax(q_values)
        print(f"\n  {name}:")
        print(f"    Best action: {action_names[best_action]} (torques for joint1, joint2)")
        print(f"    Q-values: {', '.join(f'{q:.1f}' for q in q_values)}")
        print(f"    Max Q: {q_values[best_action]:.2f}, Min Q: {np.min(q_values):.2f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Train the agent
    agent, env, metrics = train_agent(n_episodes=500, print_every=50)

    # Evaluate the trained policy
    eval_results = evaluate_agent(agent, env, n_episodes=20)

    # Analyze what the agent learned
    analyze_q_values(agent, env)

    # Print training progression summary
    print("\n" + "=" * 70)
    print("Training Progression Summary")
    print("=" * 70)
    if metrics["rewards"]:
        print(f"  Initial avg reward:  {metrics['rewards'][0]:8.1f}")
        print(f"  Final avg reward:    {metrics['rewards'][-1]:8.1f}")
        print(f"  Initial success:     {metrics['successes'][0]:5.1f}%")
        print(f"  Final success:       {metrics['successes'][-1]:5.1f}%")
        print(f"  Reward improvement:  {metrics['rewards'][-1] - metrics['rewards'][0]:+.1f}")

    print("\n" + "=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("""
  1. REWARD SHAPING is critical — without distance-based reward, the agent
     would almost never reach the target by chance.

  2. EXPERIENCE REPLAY breaks temporal correlation. Without it, the network
     overfits to recent experiences and forgets old ones.

  3. TARGET NETWORK stabilizes training. Without it, the TD target moves
     every gradient step, preventing convergence.

  4. EXPLORATION SCHEDULE matters — too fast decay and the agent misses
     good strategies; too slow and it wastes time on random actions.

  5. This is the SAME architecture used in production robotics RL (with
     continuous actions via DDPG/SAC and better simulators like MuJoCo).
""")
