# Day 059: Transformer Attention from Scratch

## Overview

Build the core attention mechanism that powers every modern large language model — from GPT to Claude to Gemini. You'll implement **Scaled Dot-Product Attention** and **Multi-Head Attention** from scratch using only NumPy, then stack them into a complete Transformer encoder block.

This matters because attention is *the* breakthrough that replaced recurrence in sequence modeling. Understanding it deeply — not just the API calls — is what separates engineers who can debug, optimize, and extend these systems from those who treat them as black boxes.

## Core Concepts

### 1. The Attention Intuition

Traditional sequence models (RNNs, LSTMs) process tokens one at a time, building up a hidden state. This creates a bottleneck: information from early tokens must survive through every subsequent step. Attention sidesteps this entirely by allowing every token to directly "look at" every other token.

Think of it like a database query: each token creates a **Query** ("what am I looking for?"), a **Key** ("what do I contain?"), and a **Value** ("what information should I pass along?"). The attention score between two tokens is just the dot product of one token's Query with another token's Key.

### 2. Scaled Dot-Product Attention — The Math

Given input sequences packed into matrices:
- **Q** (queries): shape `(seq_len, d_k)`
- **K** (keys): shape `(seq_len, d_k)`
- **V** (values): shape `(seq_len, d_v)`

The attention function is:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**Why scale by sqrt(d_k)?** Without scaling, when `d_k` is large, the dot products grow in magnitude. If Q and K have entries with mean 0 and variance 1, then `Q @ K^T` has entries with variance `d_k`. Large values push softmax into regions where gradients are vanishingly small (saturation). Dividing by `sqrt(d_k)` restores unit variance, keeping gradients healthy.

**Why softmax?** We need attention weights that are (a) non-negative and (b) sum to 1 across the key dimension — i.e., a probability distribution. Softmax gives us exactly that, plus it's differentiable.

### 3. Multi-Head Attention

A single attention function can only capture one type of relationship. Multi-head attention runs `h` attention functions in parallel, each with its own learned projections:

```
head_i = Attention(X @ W_Q_i, X @ W_K_i, X @ W_V_i)
MultiHead(X) = Concat(head_1, ..., head_h) @ W_O
```

Each head has reduced dimensionality: `d_k = d_v = d_model / h`. This means multi-head attention has roughly the same parameter count as single-head attention with full dimensionality, but captures richer patterns.

**Why multiple heads?** Different heads learn different attention patterns: some capture syntactic dependencies (subject-verb agreement), others capture semantic relationships (coreference), positional patterns (attend to previous token), etc.

### 4. The Transformer Encoder Block

A complete encoder block combines:
1. **Multi-Head Self-Attention** — each token attends to all tokens
2. **Add & Norm** — residual connection + layer normalization
3. **Feed-Forward Network** — position-wise MLP (two linear layers with ReLU/GELU)
4. **Add & Norm** — another residual connection + layer normalization

The residual connections are critical: they allow gradients to flow directly through the network, enabling training of very deep models. Layer normalization stabilizes training by normalizing activations.

### 5. Positional Encoding

Attention is permutation-invariant — it doesn't inherently know token order. Positional encodings inject position information using sinusoidal functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why sinusoidal?** Each dimension oscillates at a different frequency, creating a unique "fingerprint" for each position. The model can learn to attend to relative positions because `PE(pos+k)` can be expressed as a linear function of `PE(pos)`.

## Step-by-Step Breakdown

### Step 1: Implement Softmax
Numerically stable softmax: subtract the max before exponentiating to prevent overflow. This doesn't change the result (softmax is shift-invariant) but prevents `exp(large_number) = inf`.

### Step 2: Implement Scaled Dot-Product Attention
Matrix multiply Q with K^T, scale, optionally apply a mask (for causal/padding), then softmax, then multiply by V. The mask sets certain positions to -inf before softmax, which zeros them out.

### Step 3: Implement Multi-Head Attention
Project Q, K, V through h sets of learned weight matrices. Run attention on each head. Concatenate and project through output matrix. The key implementation detail: reshape to split the head dimension rather than literally creating h separate matrices.

### Step 4: Implement Layer Normalization
Normalize across the feature dimension (not the batch dimension like batch norm). Compute mean and variance per-token, normalize, then apply learned scale (gamma) and shift (beta).

### Step 5: Implement Position-Wise Feed-Forward Network
Two linear transformations with a ReLU activation in between: `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`. The inner dimension is typically 4x the model dimension.

### Step 6: Assemble the Transformer Encoder Block
Stack self-attention → add & norm → FFN → add & norm. Apply to a sample input and visualize the attention weights.

### Step 7: Positional Encoding
Generate the sinusoidal encoding matrix and add it to the input embeddings.

## Learning Objectives

- Understand scaled dot-product attention mathematically and implement it from first principles
- Build multi-head attention with proper dimension splitting and concatenation
- Implement layer normalization and understand why it's preferred over batch norm in transformers
- Assemble a complete transformer encoder block with residual connections
- Visualize attention patterns to build intuition about what the model "sees"
- Connect this to production systems: how this basic block scales to GPT, BERT, and beyond

## Going Deeper

- **Causal (autoregressive) masking**: For decoder-style models (GPT), mask future tokens so the model can only attend to past context. This is a simple upper-triangular mask applied before softmax.
- **KV-cache**: In autoregressive generation, keys and values from previous tokens don't change. Caching them avoids O(n^2) recomputation, enabling efficient inference.
- **Flash Attention**: The standard attention algorithm is memory-bound (materializes the full n×n attention matrix). Flash Attention tiles the computation to stay in SRAM, achieving 2-4x speedup with exact results.
- **Rotary Position Embeddings (RoPE)**: Modern models encode relative position by rotating Q and K vectors, which is more flexible than additive sinusoidal encodings.
- **Grouped Query Attention (GQA)**: Share K/V heads across multiple Q heads to reduce memory during inference. Used in Llama 2/3 and many production models.
- **Connection to Day 015**: The forward pass of neural networks you built earlier is a building block here — attention is a dynamic, input-dependent version of a weighted sum.
