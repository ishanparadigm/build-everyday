# Day 069: Mixture of Experts (MoE) from Scratch

## Overview

Build a Mixture of Experts model from scratch — the architecture behind modern LLMs like GPT-4 and Mixtral. Instead of routing every input through a single monolithic network, MoE uses a **gating network** to dynamically select which "expert" sub-networks process each input. This means you can scale model capacity (total parameters) without proportionally scaling compute (active parameters per input).

**Why it matters:** MoE is how frontier AI labs build models with hundreds of billions of parameters that remain practical to run. Mixtral 8x7B has 47B total parameters but only activates ~13B per token — achieving near-GPT-4 quality at a fraction of the inference cost. Understanding MoE is essential to understanding modern AI scaling.

## Core Concepts

### 1. The Scaling Problem MoE Solves

Dense neural networks have a fundamental tension: **capacity vs. compute**. A 100B parameter dense model needs 100B parameters of compute for every single input. But not every input needs the same processing — a math question and a poetry question might benefit from very different internal representations.

MoE breaks this coupling. You can have N expert networks, each specialized, but only activate K of them per input (where K << N). Total parameters = N × expert_size, but active parameters = K × expert_size.

### 2. The Gating Network (Router)

The gating network is the brain of MoE. Given input x, it produces a probability distribution over N experts:

```
G(x) = Softmax(W_g · x + noise)
```

Where:
- `W_g` is a learnable weight matrix of shape (input_dim, num_experts)
- `noise` is optional Gaussian noise added during training for exploration (more on this below)
- The softmax output tells us how much to "trust" each expert for this particular input

**Top-K routing:** In practice, we don't use all experts. We select the top-K experts by gate value and zero out the rest:

```
TopK_G(x) = TopK(Softmax(W_g · x))   # Keep only K largest values, zero the rest
            → Renormalize so the K values sum to 1
```

For most modern systems, K=1 or K=2. Mixtral uses K=2 out of 8 experts.

### 3. Expert Networks

Each expert is a standard feed-forward network (typically an MLP):

```
Expert_i(x) = W2_i · ReLU(W1_i · x + b1_i) + b2_i
```

All experts share the same architecture but have **independent weights**. Over training, they specialize — one might become good at syntax, another at factual recall, another at reasoning.

### 4. The Combined Output

The final output is a weighted sum of the active experts' outputs:

```
y = Σ_i  G(x)_i · Expert_i(x)
```

But crucially, for the experts where G(x)_i = 0 (not in top-K), we **don't compute** Expert_i(x) at all. This is where the computational savings come from.

### 5. The Load Balancing Problem

Without intervention, MoE training collapses: the gating network learns to send everything to one or two "favorite" experts, leaving the rest untrained. This is called **expert collapse** and it wastes most of your model capacity.

**Why it happens:** Early in training, one expert might be slightly better by random initialization. The gate routes more data to it, it improves faster, the gate routes even more data to it — a positive feedback loop.

**The fix — auxiliary load balancing loss:**

```
L_balance = α · N · Σ_i (f_i · p_i)
```

Where:
- `f_i` = fraction of tokens routed to expert i (in this batch)
- `p_i` = average gate probability for expert i (in this batch)
- `α` = balancing coefficient (typically 0.01)
- `N` = number of experts

This loss is minimized when all experts get equal traffic. It's added to the main task loss during training.

**Intuition:** If expert 3 is getting 50% of the traffic (f_3 = 0.5) while the ideal is 12.5% (1/8), the balancing loss penalizes this proportionally. The `f_i · p_i` product ensures we penalize both the routing decision AND the gate's probability, giving the optimizer a smooth gradient to fix the imbalance.

### 6. Noisy Gating for Exploration

During training, we add tunable Gaussian noise before the softmax:

```
H(x) = W_g · x + StandardNormal() · Softplus(W_noise · x)
G(x) = Softmax(H(x))
```

The noise magnitude is input-dependent (learned via W_noise). This encourages exploration — inputs that the gate is uncertain about will sometimes be routed to different experts, giving all experts a chance to learn.

## Step-by-Step Breakdown

### Step 1: Build a Single Expert MLP
Create a simple feed-forward network with one hidden layer. This is the building block — each expert will be one of these. Use ReLU activation and standard weight initialization.

### Step 2: Build the Gating Network
Implement a linear layer that maps input_dim → num_experts, followed by optional noise injection and softmax. Add top-K selection that zeros out non-selected experts and renormalizes.

### Step 3: Assemble the MoE Layer
Combine the gating network with N expert MLPs. For each input, compute gate values, select top-K experts, run only those experts, and combine their outputs with the gate weights.

### Step 4: Implement Load Balancing Loss
Track the fraction of inputs routed to each expert and the average gate probability per expert. Compute the auxiliary loss and add it to the task loss.

### Step 5: Build a Full MoE Model
Stack the MoE layer into a complete model for a classification task. Add an input layer and output head around the MoE layer.

### Step 6: Train and Analyze
Train on a synthetic dataset. Monitor: task loss, balancing loss, per-expert utilization, and gate entropy. Visualize which experts specialize on which input regions.

## Learning Objectives

- Understand **conditional computation** and why it enables scaling
- Implement **top-K gating** with differentiable routing
- Build the **load balancing loss** that prevents expert collapse
- See how **expert specialization** emerges from training
- Understand the compute/capacity tradeoff that makes MoE practical
- Connect this to real architectures: Switch Transformer, Mixtral, GShard

## Going Deeper

- **Switch Transformer (K=1):** Google showed that using just 1 expert per token (instead of 2) works surprisingly well and simplifies routing. The key insight: with proper load balancing, even K=1 learns good specialization.
- **Expert capacity factor:** In distributed settings, each expert has a fixed buffer size. Tokens that exceed an expert's capacity are dropped or sent to a fallback. This introduces an interesting tradeoff between load balance and information loss.
- **Sparse vs. Dense training:** MoE models are harder to train than dense models of the same active parameter count. They need more data, are sensitive to hyperparameters, and can suffer from training instabilities.
- **Routing strategies beyond top-K:** Expert Choice routing (experts choose tokens instead of tokens choosing experts), hash-based routing, and learned routing with reinforcement learning.
- **Connection to Day 059 (Transformer Attention):** In practice, MoE replaces the FFN layers in a Transformer. The attention layers remain dense — only the FFN computation is sparse. This is how Mixtral works: standard attention + MoE FFN.
