# Day 057: Reinforcement Learning — Q-Learning from Scratch

## What You're Building

A complete Q-learning agent that learns to navigate a grid world environment through trial and error. No labeled data, no supervision — just rewards and consequences. The agent starts knowing nothing about the world and, through thousands of episodes of exploration, discovers the optimal policy for reaching a goal while avoiding traps.

**Why it matters:** Q-learning is the foundation of modern RL (Deep Q-Networks, AlphaGo's early training, robotics control). Understanding tabular Q-learning deeply is essential before touching deep RL — it teaches you the core dynamics of exploration vs. exploitation, temporal credit assignment, and value-based decision making without neural network complexity obscuring the ideas.

## Core Concepts

### The Reinforcement Learning Framework

Unlike supervised learning (where you have input-output pairs) or unsupervised learning (where you find structure in data), RL learns from **interaction**. An agent takes actions in an environment, observes state transitions and rewards, and tries to maximize cumulative future reward.

Formally, this is a **Markov Decision Process (MDP)**:
- **S**: Set of states (grid cells in our case)
- **A**: Set of actions (up, down, left, right)
- **P(s'|s,a)**: Transition probability — probability of reaching state s' from state s taking action a
- **R(s,a,s')**: Reward function — immediate reward for a transition
- **γ (gamma)**: Discount factor ∈ [0,1] — how much we value future rewards vs. immediate ones

The **Markov property** means the future depends only on the current state, not on how we got there. This is what makes the problem tractable.

### The Q-Function: What Makes Q-Learning Work

The Q-function Q(s,a) represents the **expected cumulative discounted reward** of taking action `a` in state `s` and then following the optimal policy thereafter:

```
Q*(s,a) = E[R(t) + γR(t+1) + γ²R(t+2) + ... | s_t = s, a_t = a]
```

If we knew Q* perfectly, the optimal policy would be trivial: in every state, pick the action with the highest Q-value. The challenge is *learning* Q* from experience.

### The Bellman Equation: The Heart of Q-Learning

The key insight (Bellman, 1957) is that Q* satisfies a recursive relationship:

```
Q*(s,a) = E[r + γ · max_a'(Q*(s',a'))]
```

In English: the value of taking action `a` in state `s` equals the immediate reward `r` plus the discounted value of the best action in the next state `s'`. This is the **Bellman optimality equation**.

### The Q-Learning Update Rule

Since we don't know Q* upfront, we start with arbitrary Q-values and iteratively improve them using experience:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a'(Q(s',a')) - Q(s,a)]
```

Breaking this down:
- `r + γ · max_a'(Q(s',a'))` is the **TD target** — our current best estimate of what Q(s,a) should be
- `Q(s,a)` is our current estimate
- The difference `[target - current]` is the **TD error** (temporal difference error)
- `α` (alpha) is the **learning rate** — how much we adjust toward the new estimate

**Why this works:** Each update nudges Q(s,a) closer to the true value. With enough exploration and a decaying learning rate, Q-values provably converge to Q* (Watkins & Dayan, 1992).

**Critical subtlety:** Q-learning is **off-policy** — it updates toward the *max* action (greedy) regardless of what action the agent actually took. This separates the *behavior policy* (how we explore) from the *target policy* (what we're learning). This is what distinguishes Q-learning from SARSA (which is on-policy).

### Exploration vs. Exploitation: The ε-Greedy Strategy

If the agent always picks the best-known action (exploitation), it might miss better options it hasn't tried. If it always picks randomly (exploration), it never uses what it's learned.

**ε-greedy** balances this:
- With probability ε: take a random action (explore)
- With probability 1-ε: take the best-known action (exploit)

**ε decay** is crucial: start with high ε (lots of exploration) and gradually reduce it as the agent learns. Common schedule: `ε = max(ε_min, ε * decay_rate)` after each episode.

### The Discount Factor γ: Present vs. Future

- γ = 0: Agent is completely myopic, only cares about immediate reward
- γ = 1: Agent values all future rewards equally (can cause instability)
- γ = 0.9-0.99: Typical range — agent plans ahead but prefers sooner rewards

The choice of γ encodes your problem's time horizon. A robot navigating a room needs γ ≈ 0.95. A financial agent over years might need γ ≈ 0.999.

## Step-by-Step Breakdown

### Step 1: Build the Grid World Environment
Create a grid with a start position, goal position, walls, and traps. The environment needs `reset()` (return to start) and `step(action)` (take action, return next_state, reward, done). Rewards: -1 per step (encourages efficiency), -10 for traps, +100 for goal.

Without the step penalty, the agent has no incentive to find short paths. Without large trap penalties, it won't learn to avoid dangerous states.

### Step 2: Initialize the Q-Table
Create a 2D table Q[state][action] initialized to zeros. For an NxN grid with 4 actions, this is N² × 4 values. Zero initialization is a reasonable default — optimistic initialization (positive values) can encourage exploration but complicates analysis.

### Step 3: Implement ε-Greedy Action Selection
The agent needs a method that returns a random action with probability ε, or the Q-maximizing action otherwise. Handle ties randomly to avoid bias.

### Step 4: Implement the Q-Learning Update
After each (state, action, reward, next_state) experience, apply the Bellman update. For terminal states, there's no future value: the target is just `r`.

### Step 5: Run Training Episodes
Each episode: reset environment, loop (select action, take step, update Q, transition to next state) until terminal. Track cumulative rewards per episode to monitor learning.

### Step 6: Extract and Visualize the Policy
After training, the learned policy is `π(s) = argmax_a Q(s,a)` — just pick the best action in each state. Visualize this as arrows on the grid to verify it makes sense.

## Learning Objectives
- Understand the MDP framework and how it models sequential decision-making
- Implement tabular Q-learning with the Bellman update rule
- Build intuition for exploration vs. exploitation tradeoffs
- See how hyperparameters (α, γ, ε) shape learning dynamics
- Visualize policy convergence and interpret Q-values
- Understand the difference between on-policy (SARSA) and off-policy (Q-learning) methods

## Going Deeper
- **Deep Q-Networks (DQN):** Replace the Q-table with a neural network to handle continuous/high-dimensional state spaces. Key innovations: experience replay buffer and target networks (Mnih et al., 2015).
- **Double Q-Learning:** Q-learning overestimates Q-values because it uses max for both selection and evaluation. Double Q-learning uses two Q-tables to decouple these, reducing overestimation bias.
- **SARSA comparison:** Change the update to use the *actual* next action instead of max. This makes the agent more conservative — it learns to avoid states where its ε-greedy behavior might lead to traps.
- **Function approximation dangers:** The "deadly triad" of off-policy + function approximation + bootstrapping can cause divergence. Understanding tabular Q-learning helps you diagnose these issues.
- **Production RL:** Real systems use prioritized experience replay, dueling architectures, distributional RL, and careful reward shaping. But they all build on the same Bellman equation you implement today.
