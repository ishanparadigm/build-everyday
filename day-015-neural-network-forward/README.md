# Day 015: Simple Neural Network (Forward Pass)

## Overview

Build a neural network from scratch — no TensorFlow, no PyTorch, just NumPy and your understanding of linear algebra. Today we implement the **forward pass**: the process of transforming inputs into predictions through layers of weighted connections and nonlinear activations.

This is the foundational computation behind every deep learning model in production — from GPT to AlphaFold to self-driving cars. Understanding the forward pass means understanding how a neural network actually *thinks*: matrix multiplications followed by nonlinearities, stacked to learn increasingly abstract representations.

## Core Concepts

### Why Neural Networks Work: Universal Approximation

A single hidden layer with enough neurons can approximate *any* continuous function (the Universal Approximation Theorem). But why?

Consider a simple function you want to learn. A single neuron computes:

```
output = activation(w1*x1 + w2*x2 + ... + wn*xn + b)
```

This is just a weighted sum passed through a nonlinearity. One neuron draws a single decision boundary. But stack enough of them in a layer, and their combined outputs can carve up the input space into arbitrary regions. Add more layers, and the network can compose simple features into complex abstractions.

### The Forward Pass: Matrix Form

For a network with L layers, the forward pass is:

```
Layer 1: z1 = X @ W1 + b1,  a1 = activation(z1)
Layer 2: z2 = a1 @ W2 + b2, a2 = activation(z2)
...
Layer L: zL = a(L-1) @ WL + bL, output = final_activation(zL)
```

Where:
- **X** is the input matrix (batch_size x n_features)
- **Wi** is the weight matrix for layer i (n_prev x n_current)
- **bi** is the bias vector for layer i (1 x n_current)
- **zi** is the pre-activation (linear combination)
- **ai** is the post-activation output

This is the key insight: **the entire forward pass is just alternating matrix multiplications and element-wise nonlinearities**. That's why GPUs (built for matrix math) revolutionized deep learning.

### Activation Functions: Why Nonlinearity Matters

Without activation functions, stacking layers is pointless. Two linear transformations compose into one:

```
(X @ W1) @ W2 = X @ (W1 @ W2) = X @ W_combined
```

You'd just have a single linear layer. Nonlinear activations break this, allowing the network to learn nonlinear decision boundaries.

**ReLU: f(x) = max(0, x)**
- Gradient is 1 for positive inputs, 0 for negative — fast to compute
- Suffers from "dying ReLU": neurons that always output 0 can never recover
- Most popular for hidden layers due to simplicity and effectiveness

**Sigmoid: f(x) = 1 / (1 + e^(-x))**
- Squashes output to (0, 1) — natural for probabilities
- Gradient vanishes for large |x| — problematic in deep networks
- Used in output layer for binary classification

**Softmax: softmax(x_i) = e^(x_i) / sum(e^(x_j))**
- Converts a vector of scores into a probability distribution
- Outputs sum to 1 — used for multi-class classification
- Numerically unstable without the max-subtraction trick

### Weight Initialization: Getting It Right

Bad initialization → bad gradients → no learning. If weights are too large, activations saturate. Too small, signals vanish.

**Xavier/Glorot initialization**: `W ~ N(0, sqrt(2 / (n_in + n_out)))`
- Keeps variance roughly constant across layers
- Designed for sigmoid/tanh activations

**He initialization**: `W ~ N(0, sqrt(2 / n_in))`
- Accounts for ReLU killing half the neurons (hence the factor of 2)
- Standard choice for ReLU networks

### Loss Functions: Measuring Error

**Cross-entropy loss** for classification:
```
L = -sum(y_true * log(y_pred))
```

This penalizes confident wrong predictions heavily (log of a small number is very negative). It's the standard loss for classification because it directly optimizes the predicted probability of the correct class.

**Mean squared error** for regression:
```
L = mean((y_true - y_pred)^2)
```

## Step-by-Step Breakdown

1. **Define the architecture**: Specify layer sizes (e.g., [784, 128, 64, 10] for MNIST). Each pair of consecutive sizes defines a weight matrix.

2. **Initialize parameters**: For each layer, create a weight matrix W (n_prev x n_current) and bias vector b (1 x n_current). Use He initialization for ReLU layers.

3. **Implement activation functions**: Code ReLU, sigmoid, and softmax. For softmax, subtract the max for numerical stability: `exp(x - max(x))`.

4. **Forward pass through each layer**: For each layer, compute z = a_prev @ W + b, then a = activation(z). Store all intermediate values (needed for backpropagation later).

5. **Compute loss**: Compare the final output to the true labels using cross-entropy.

6. **Predict**: For classification, take argmax of the output layer to get the predicted class.

## Learning Objectives

- Implement a feedforward neural network using only NumPy
- Understand the forward pass as a sequence of linear transformations + nonlinearities
- Implement ReLU, sigmoid, and softmax activation functions with numerical stability
- Apply proper weight initialization (He, Xavier)
- Compute cross-entropy loss for multi-class classification
- Process data in mini-batches using matrix operations
- Build intuition for how network depth and width affect representational capacity

## Going Deeper

- **Batch normalization**: Normalize activations between layers to stabilize training. This is essentially standardizing z values before applying the activation — reduces internal covariate shift.
- **Dropout**: Randomly zero out neurons during training to prevent co-adaptation. Forces the network to learn redundant representations.
- **Skip connections (ResNets)**: Add the input of a block to its output: `a = activation(z + x)`. This gives gradients a direct path through the network, enabling training of very deep networks (100+ layers).
- **Why depth > width**: Deeper networks can learn hierarchical features (edges → textures → objects) more efficiently than wide shallow ones. Each layer abstracts over the previous one.
- **Connection to backpropagation**: Tomorrow's challenge. The forward pass stores intermediate values specifically so backprop can compute gradients efficiently using the chain rule.
