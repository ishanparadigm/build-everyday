# Day 058: Policy Gradient Methods — REINFORCE Algorithm

## Overview

Yesterday you built a Q-learning agent that learned a value for every (state, action) pair and picked the best one. That works beautifully for small, discrete environments. But what happens when your action space is continuous (steering angles, joint torques) or your state space is enormous (raw pixels)? Storing a Q-table becomes impossible.

**Policy gradient methods** flip the script: instead of learning a value function and deriving a policy from it, they learn the policy directly. The agent maintains a parameterized policy π_θ(a|s) — a function that maps states to a probability distribution over actions — and optimizes the parameters θ by gradient ascent on expected return.

This is the foundation of every modern RL algorithm used in production: PPO (ChatGPT's RLHF), SAC (robotic manipulation), A3C (game playing), and TRPO all descend from the policy gradient theorem.

## Core Concepts

### Why Learn a Policy Directly?

Value-based methods (Q-learning, Day 057) have three limitations that policy gradients overcome:

1. **Continuous actions**: Q-learning requires `max_a Q(s, a)` — you'd need to search over all possible continuous actions. A policy network just outputs the action distribution directly.

2. **Stochastic policies**: Sometimes the optimal policy is stochastic (e.g., rock-paper-scissors). Value-based methods produce deterministic policies; policy gradients naturally represent distributions.

3. **Smoother optimization**: Small changes to θ produce small changes to the policy. In Q-learning, a small change to Q-values can flip which action is selected, causing unstable learning.

### The Policy Gradient Theorem

We want to maximize the expected total return:

```
J(θ) = E_τ~π_θ [R(τ)]
```

where τ = (s₀, a₀, r₁, s₁, a₁, r₂, ...) is a trajectory and R(τ) = Σ_t γ^t r_t is the discounted return.

The key insight (the **policy gradient theorem**) is:

```
∇_θ J(θ) = E_τ~π_θ [ Σ_t ∇_θ log π_θ(aₜ|sₜ) · Gₜ ]
```

where Gₜ = Σ_{k=t}^{T} γ^(k-t) r_k is the **return-to-go** from timestep t.

**Intuition**: This formula says "increase the probability of actions that led to high returns, decrease the probability of actions that led to low returns." The log-probability gradient tells us how to nudge θ to make action aₜ more likely, and Gₜ tells us how good the outcome was.

**Derivation sketch**: The trick is the log-derivative identity: ∇_θ π_θ = π_θ · ∇_θ log π_θ. This lets us rewrite the gradient of an expectation as an expectation of a gradient — something we can estimate with Monte Carlo samples.

### REINFORCE Algorithm

REINFORCE (Williams, 1992) is the simplest policy gradient algorithm:

1. Run the policy to collect a complete episode trajectory
2. For each timestep t, compute the return-to-go Gₜ
3. Compute the policy gradient: ∇_θ log π_θ(aₜ|sₜ) · Gₜ
4. Update parameters: θ ← θ + α · gradient

This is a **Monte Carlo** method — it uses complete episode returns, not bootstrapped estimates.

### The Variance Problem and Baselines

Raw REINFORCE has high variance. Imagine all returns are positive (e.g., all between 50 and 100). The gradient pushes up the probability of *every* action, just pushing some up more than others. This is correct in expectation but very noisy.

**Solution**: Subtract a **baseline** b(s) from the return:

```
∇_θ J(θ) = E [ Σ_t ∇_θ log π_θ(aₜ|sₜ) · (Gₜ - b(sₜ)) ]
```

This doesn't change the expected gradient (provably unbiased) but dramatically reduces variance. The most common baseline is the **value function** V(s), estimated by a separate neural network. When Gₜ - V(sₜ) > 0, the action was better than average → increase its probability. When < 0, it was worse → decrease it.

### Softmax Policy for Discrete Actions

For discrete action spaces, we parameterize the policy with a softmax:

```
π_θ(a|s) = exp(h(s,a,θ)) / Σ_{a'} exp(h(s,a',θ))
```

where h(s,a,θ) are learned **action preferences** (logits). This ensures probabilities are valid (positive, sum to 1) and differentiable.

### Discount Factor in Policy Gradients

The discount factor γ appears in two places:
- In the return Gₜ (how much we care about future rewards)
- Optionally as a weighting factor γ^t on each timestep's gradient contribution

Using γ^t weighting (called the "discounted gradient") makes the agent care less about optimizing behavior far into the future, which can stabilize learning.

## Step-by-Step Breakdown

### Step 1: Define the Policy Network
A small neural network maps states → action logits. For CartPole (4-dim state, 2 actions), a simple 2-layer MLP suffices. We implement this from scratch using NumPy — no PyTorch — to see every gradient computation.

### Step 2: Forward Pass — Action Selection
Given a state, compute logits via the network, apply softmax to get probabilities, then sample an action from the distribution. We must store the log-probability of the chosen action for the gradient update.

### Step 3: Collect Full Episode
Run the policy until the episode terminates. Store (state, action, reward, log_prob) at each step. We need the complete trajectory because REINFORCE is a Monte Carlo method.

### Step 4: Compute Returns-to-Go
Work backwards through the episode: Gₜ = rₜ + γ · G_{t+1}. This is more efficient than computing each Gₜ from scratch.

### Step 5: Normalize Returns (Baseline Approximation)
Subtract the mean and divide by standard deviation of returns across the episode. This acts as a simple baseline, centering the returns so that roughly half are positive (reinforced) and half negative (discouraged).

### Step 6: Compute and Apply Gradients
For each timestep: compute ∇_θ log π_θ(aₜ|sₜ) · (normalized Gₜ). Sum over the episode, then update θ with gradient ascent. We implement backpropagation through the softmax and network layers manually.

### Step 7: Train Over Many Episodes
Repeat for hundreds of episodes. Track average reward to monitor learning. CartPole is "solved" at average reward ≥ 195 over 100 episodes.

## Learning Objectives

- Understand **why** policy gradients exist and when they're preferred over value-based methods
- Derive and implement the **REINFORCE** algorithm from scratch
- Implement a neural network policy with manual backpropagation through softmax
- Understand the **variance problem** and how baselines reduce it
- See the connection to modern algorithms (PPO, A3C) that build on these foundations
- Gain intuition for the **credit assignment** problem (which actions caused the reward?)

## Going Deeper

- **Actor-Critic**: Replace the return-to-go with a learned value function (the "critic") to reduce variance further while adding some bias — the bias-variance tradeoff at the heart of RL
- **PPO (Proximal Policy Optimization)**: Clips the policy ratio to prevent destructive large updates — the algorithm behind RLHF in ChatGPT
- **Continuous actions**: Replace softmax with a Gaussian policy: the network outputs μ and σ, actions are sampled from N(μ, σ²)
- **Natural policy gradients**: Use the Fisher information matrix to take steps of constant KL-divergence in policy space, rather than constant Euclidean distance in parameter space
- **Connection to Day 057**: Q-learning is off-policy (can learn from old data); REINFORCE is on-policy (must use fresh trajectories). This makes REINFORCE less sample-efficient but more stable.
