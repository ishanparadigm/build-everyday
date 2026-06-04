# Day 064: Generative Adversarial Network (GAN) from Scratch

## Overview

Build a complete GAN training loop using only NumPy — a generator that creates synthetic data and a discriminator that tries to distinguish real from fake. GANs are the foundational framework behind image synthesis, style transfer, data augmentation, and more. Understanding the adversarial training dynamic from first principles reveals why generative AI works — and why it's so hard to stabilize.

## Core Concepts

### The Minimax Game

A GAN is two neural networks locked in a zero-sum game:

- **Generator G(z)**: Takes random noise z ~ N(0, I) and maps it to data space. Its goal is to produce samples indistinguishable from real data.
- **Discriminator D(x)**: Takes a data point x and outputs the probability it came from the real distribution. Its goal is to correctly classify real vs. fake.

The training objective is the minimax value function:

```
min_G max_D  V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

Where:
- E[log D(x)] — discriminator's log-probability on real data (D wants this high)
- E[log(1 - D(G(z)))] — discriminator's log-probability of correctly rejecting fakes (D wants this high, G wants it low)

**Intuition**: The discriminator is a detective trying to spot counterfeits. The generator is a forger trying to fool the detective. As they train together, both improve — the forger produces better fakes, and the detective develops a sharper eye.

### Nash Equilibrium

At the theoretical optimum, G perfectly replicates the data distribution (p_g = p_data) and D outputs 0.5 everywhere — it literally cannot tell real from fake. In practice, we never reach this equilibrium, but we want to get close.

### Why Not Just Use Maximum Likelihood?

Maximum likelihood estimation (MLE) minimizes KL(p_data || p_model), which penalizes the model heavily for placing zero probability on real data points. This leads to "mode covering" — the model smears probability mass across all modes, producing blurry outputs. GANs implicitly minimize a different divergence (Jensen-Shannon), which allows sharper outputs at the cost of potential "mode collapse" — the generator may only capture some modes of the data.

### Training Dynamics and Instability

GAN training is notoriously unstable because:

1. **Mode collapse**: G finds one output that fools D and keeps producing it, ignoring the diversity of real data.
2. **Vanishing gradients**: If D becomes too good, D(G(z)) → 0 and log(1 - D(G(z))) → 0 with near-zero gradient. G gets no learning signal.
3. **Oscillation**: Rather than converging, G and D chase each other in parameter space.

**Practical fix for vanishing gradients**: Instead of minimizing log(1 - D(G(z))), the generator maximizes log(D(G(z))). This provides stronger gradients early in training when D easily rejects fakes. Mathematically different, but same fixed point.

### Generator and Discriminator Architecture

For this challenge, we use simple MLPs (multi-layer perceptrons):

- **Generator**: z (latent dim) → hidden → hidden → output (data dim). Uses ReLU activations in hidden layers, tanh in the output layer to bound outputs to [-1, 1].
- **Discriminator**: x (data dim) → hidden → hidden → sigmoid. Uses LeakyReLU activations (slope 0.2 for negative inputs) to prevent dying ReLU, critical for discriminator gradient flow.

### Binary Cross-Entropy Loss

Both networks use BCE loss:
- D on real: -log(D(x))
- D on fake: -log(1 - D(G(z)))
- G: -log(D(G(z)))  (non-saturating version)

## Step-by-Step Breakdown

### Step 1: Define the Real Data Distribution
We generate 2D data from a mixture of Gaussians — multiple clusters that the generator must learn to reproduce. This is a clean, visualizable target distribution that reveals mode collapse (if the generator only captures some clusters).

### Step 2: Build the Generator Network
Forward pass: noise → linear → ReLU → linear → ReLU → linear → tanh. Each layer is initialized with He initialization (scaled by sqrt(2/fan_in)) for ReLU layers. The tanh output ensures generated points are bounded.

### Step 3: Build the Discriminator Network
Forward pass: data → linear → LeakyReLU → linear → LeakyReLU → linear → sigmoid. LeakyReLU(x) = x if x > 0, else 0.2x. This prevents dead neurons when inputs are negative, which is critical because the discriminator sees data centered around zero.

### Step 4: Implement Backpropagation
Manual backprop through each layer for both networks. Key derivatives:
- sigmoid: σ(x)(1 - σ(x))
- tanh: 1 - tanh²(x)
- LeakyReLU: 1 if x > 0, else 0.2
- BCE loss gradient: -(y/p - (1-y)/(1-p))

### Step 5: Training Loop
Each iteration:
1. Sample real data batch from the mixture of Gaussians
2. Sample noise z ~ N(0, 1) and generate fake data via G(z)
3. Train D: forward pass on real (label=1) and fake (label=0), compute BCE loss, backprop, update weights
4. Sample fresh noise, generate new fakes
5. Train G: forward pass through G then D, compute BCE loss with label=1 (trick D), backprop through D (frozen) then G, update G's weights

### Step 6: Evaluation Metrics
- **Discriminator accuracy**: Should hover around 50% when training is balanced
- **Generator loss**: Should decrease as fakes improve
- **Mode coverage**: Check if generated samples cover all clusters of the real distribution
- **Wasserstein estimate**: Approximate Earth Mover's distance between real and generated distributions

## Learning Objectives

- Understand adversarial training as a minimax optimization
- Implement forward and backward passes for generator and discriminator MLPs
- Experience and diagnose GAN training instabilities (mode collapse, vanishing gradients)
- Apply the non-saturating generator loss trick and understand why it helps
- Analyze training dynamics through discriminator accuracy and loss curves

## Going Deeper

- **Wasserstein GAN (WGAN)**: Replaces BCE with Wasserstein distance, uses weight clipping or gradient penalty. Much more stable training. The key insight: Wasserstein distance provides useful gradients even when distributions don't overlap.
- **Conditional GAN (cGAN)**: Both G and D receive class labels as additional input, allowing controlled generation.
- **Progressive GAN**: Starts with low-resolution generation and progressively adds layers for higher resolution — key technique behind early photorealistic face generation.
- **Spectral Normalization**: Constrains the Lipschitz constant of D by normalizing weight matrices by their spectral norm — stabilizes training without gradient penalty.
- **Production use**: GANs power data augmentation (medical imaging), super-resolution (ESRGAN), style transfer (CycleGAN), and were the precursors to modern diffusion models. Understanding GAN training dynamics helps debug diffusion model training too.
