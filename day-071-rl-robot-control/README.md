# Day 71: Reinforcement Learning for Robot Control

## Overview

Build a reinforcement learning agent that learns to control a simulated 2D robot arm to reach target positions. This bridges two domains you've already explored — RL fundamentals (Day 57-58: Q-learning and policy gradients) and robotics (Day 8: kinematics, Day 42: trajectory planning) — by having an agent *discover* control policies through trial and error rather than hand-coding them.

**Why this matters:** Classical robot control requires precise mathematical models of the robot and its environment. When those models are inaccurate, brittle, or too complex to derive (think a 30-DOF humanoid hand manipulating deformable objects), RL lets the robot learn through experience. This is how real systems like OpenAI's Dactyl learned to manipulate a Rubik's cube, and how DeepMind's robots learn locomotion in simulation before transferring to hardware.

## Core Concepts

### 1. The Robot Control MDP

Every RL problem is a Markov Decision Process (MDP). For robot control:

- **State s:** The robot's configuration — joint angles θ₁, θ₂, angular velocities θ̇₁, θ̇₂, and the target position (x_target, y_target). That's a 6D continuous state space.
- **Action a:** Torques applied to each joint. We discretize into {-1, 0, +1} per joint, giving 9 possible actions. (Continuous action spaces use policy gradients; discrete works for learning the concepts.)
- **Transition T(s'|s,a):** Physics simulation — how torques change angular velocities and positions. We'll use simple rigid-body dynamics.
- **Reward R(s,a,s'):** Shaped to guide learning. The key design choice in any RL robotics problem.

### 2. Reward Shaping for Robotics

Naive reward: +1 when the end-effector reaches the target, 0 otherwise. This is *sparse* — the agent almost never stumbles into success randomly, so it can't learn.

**Shaped reward** provides continuous signal:

```
R = -distance(end_effector, target)          # Get closer = better
    - 0.01 * sum(|torques|)                   # Penalize energy use
    + 100.0 * (distance < threshold)          # Big bonus for reaching target
    - 0.1 * sum(|angular_velocities|)         # Penalize jerky motion
```

The math: reward shaping is provably safe (preserves optimal policy) if the shaping function is a *potential-based* function Φ(s). Our distance-based reward approximates this — it's a function of state only, not the action.

**Tradeoff:** Too much shaping → the agent exploits the reward signal instead of solving the task. Too little → intractable exploration. This tension is fundamental to applied RL.

### 3. Deep Q-Network (DQN) for Continuous States

Day 57 covered tabular Q-learning, which discretizes states into bins. With 6 continuous dimensions, the state space explodes (even 10 bins per dimension = 10⁶ states). DQN replaces the Q-table with a neural network:

```
Q(s, a; θ) ≈ Q*(s, a)
```

The network takes state s as input and outputs Q-values for all 9 actions. Key ingredients:

- **Experience replay buffer:** Stores (s, a, r, s', done) transitions. Training samples random mini-batches, breaking temporal correlation that destabilizes SGD. Buffer size matters: too small → catastrophic forgetting, too large → stale data.
- **Target network:** A slowly-updated copy of the Q-network used to compute TD targets: `y = r + γ * max_a' Q_target(s', a')`. Without this, the target shifts every gradient step, causing divergence. Updated every C steps by copying weights.
- **ε-greedy exploration:** With probability ε, take a random action. ε decays from 1.0 → 0.01 over training. The decay schedule is crucial — too fast means the agent never explores enough, too slow means it wastes time on random actions.

**The Bellman backup (what makes it work):**

```
L(θ) = E[(r + γ max_a' Q_target(s', a'; θ⁻) - Q(s, a; θ))²]
```

This is a regression problem: predict the discounted future reward. The "bootstrap" — using Q_target to estimate future reward — is what lets the agent plan ahead without explicit search.

### 4. Sim-to-Real: Why Simulation Matters

We simulate physics because:
- Real robots are expensive and break
- You need millions of episodes (at ~1000 steps each, that's billions of physics steps)
- You can parallelize simulation but not reality

The **sim-to-real gap** is the central challenge: policies trained in simulation often fail on hardware because the simulation doesn't perfectly match reality (friction, motor delays, sensor noise). Solutions include domain randomization (randomize physics parameters during training) and system identification (carefully calibrate the simulator).

Our simulator uses simplified 2D rigid-body dynamics with configurable noise — a stepping stone toward production simulators like MuJoCo or Isaac Gym.

## Step-by-Step Breakdown

### Step 1: Build the 2D Robot Arm Environment
Create a gym-like environment with a 2-link planar arm. Forward kinematics (from Day 8) computes end-effector position from joint angles. Simple Euler integration for dynamics. The environment handles state representation, action application, reward computation, and episode termination.

### Step 2: Implement the DQN Agent
Neural network with 2 hidden layers (128 units each, ReLU activation). The network maps 6D state → 9 action Q-values. Implement experience replay buffer, target network synchronization, and ε-greedy action selection.

### Step 3: Training Loop
Run episodes, collect transitions, train the network on mini-batches from the replay buffer. Track metrics: episode reward, success rate, average steps to reach target. Implement the exploration schedule.

### Step 4: Evaluation and Analysis
Test the learned policy (ε=0) on new random targets. Measure success rate, path efficiency, and smoothness. Compare early vs. late training behavior to verify learning.

## Learning Objectives

- **Connect RL theory to robotics practice:** See how MDPs, value functions, and policy learning apply to physical control
- **Understand reward engineering:** Experience the tradeoffs in reward shaping for continuous control
- **Implement DQN from scratch:** Build all components — replay buffer, target network, ε-schedule — understanding *why* each exists
- **Appreciate sim-to-real challenges:** Understand why simulation fidelity and domain randomization matter

## Going Deeper

- **Continuous actions:** Replace DQN with DDPG or SAC for continuous torque control (no discretization)
- **Hindsight Experience Replay (HER):** Relabel failed episodes with achieved goals — dramatically improves sample efficiency for goal-conditioned tasks
- **Curriculum learning:** Start with targets near the arm, gradually increase difficulty
- **Multi-joint control:** Scale to 3+ links and observe how exploration difficulty grows exponentially with DOF
- **Domain randomization:** Randomize link lengths, masses, and friction during training to improve robustness
- **Connects to production:** This architecture (DQN/SAC + sim environment) is the same pattern used in real robotics labs, just scaled up with better simulators (MuJoCo, Isaac Gym) and distributed training
