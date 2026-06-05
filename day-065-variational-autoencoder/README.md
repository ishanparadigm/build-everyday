# Day 65: Variational Autoencoder (VAE) from Scratch

## Overview

A Variational Autoencoder (VAE) is a generative model that learns to encode data into a structured latent space and then decode samples from that space back into realistic data. Unlike a standard autoencoder that learns a deterministic mapping, a VAE learns a *probability distribution* over the latent space — making it possible to generate entirely new data by sampling from that distribution.

**Why it matters:** VAEs are foundational in modern generative AI. They power drug discovery (generating novel molecular structures), image synthesis, anomaly detection (anything far from the learned distribution is anomalous), and data augmentation. Unlike GANs (Day 64), VAEs provide a principled probabilistic framework with a well-defined loss function — no adversarial training instability. Understanding VAEs gives you the mathematical foundation for diffusion models, which are essentially hierarchical VAEs.

## Core Concepts

### 1. Autoencoders: The Starting Point

A standard autoencoder has two parts:
- **Encoder** f(x) → z: compresses input x into a low-dimensional latent code z
- **Decoder** g(z) → x̂: reconstructs the input from the latent code

The network is trained to minimize reconstruction error: L = ||x - x̂||². The problem? The latent space has no structure. Points between two encoded examples might decode to garbage. You can't sample random z vectors and get meaningful outputs.

### 2. The VAE Idea: Encode to Distributions, Not Points

Instead of mapping x → z (a point), the encoder maps x → (μ, σ²) — the parameters of a Gaussian distribution. The latent code z is then *sampled* from N(μ, σ²).

This changes everything:
- The latent space becomes continuous and structured
- Nearby points in latent space decode to similar outputs
- You can sample z ~ N(0, I) and decode it to generate new data

### 3. The Reparameterization Trick

**The problem:** We need to backpropagate through a sampling operation z ~ N(μ, σ²), but sampling is stochastic and non-differentiable.

**The trick:** Instead of sampling z directly, we compute:

    z = μ + σ * ε,  where ε ~ N(0, I)

This moves the randomness to ε (which doesn't depend on parameters), making z a deterministic, differentiable function of μ and σ. Gradients flow through μ and σ normally.

This is one of the most elegant tricks in deep learning — it lets us optimize stochastic models with standard backpropagation.

### 4. The ELBO Loss Function

The VAE loss has two terms:

    L = Reconstruction Loss + KL Divergence

**Reconstruction Loss:** How well can the decoder reconstruct the input from the sampled z? For continuous data, this is MSE. For binary data (like MNIST pixels normalized to [0,1]), it's Binary Cross-Entropy:

    L_recon = -Σ [x_i * log(x̂_i) + (1 - x_i) * log(1 - x̂_i)]

**KL Divergence:** How far is the learned distribution q(z|x) = N(μ, σ²) from the prior p(z) = N(0, I)?

    KL(q(z|x) || p(z)) = -0.5 * Σ (1 + log(σ²) - μ² - σ²)

This has a beautiful closed-form solution for two Gaussians. It acts as a regularizer: it pushes the encoder to produce distributions close to the standard normal, ensuring the latent space is well-organized.

Together, these form the Evidence Lower Bound (ELBO) — a lower bound on the log-likelihood of the data. Maximizing the ELBO simultaneously improves reconstruction quality AND latent space structure.

### 5. The Reconstruction-Regularization Tradeoff

The two loss terms compete:
- **Low reconstruction loss** → the encoder wants to spread latent codes apart (precise encoding)
- **Low KL divergence** → the encoder wants to collapse everything to N(0, I) (good sampling)

The β-VAE variant adds a weight β to the KL term: L = L_recon + β * KL. β > 1 encourages more disentangled representations (each latent dimension captures one factor of variation). β < 1 prioritizes reconstruction quality.

## Step-by-Step Approach

### Step 1: Data Preparation
Load MNIST digits. Normalize pixel values to [0, 1] (important for BCE loss). Flatten 28x28 images to 784-dimensional vectors.

### Step 2: Build the Encoder
MLP that maps 784 → 512 → 256 → (μ, log_σ²). We predict log(σ²) instead of σ² directly for numerical stability — it can be any real number, while σ² must be positive.

### Step 3: Build the Decoder
MLP that maps z → 256 → 512 → 784 with sigmoid activation on the output (pixel values must be in [0, 1]).

### Step 4: Implement Reparameterization
Sample ε ~ N(0, I), compute z = μ + exp(0.5 * log_σ²) * ε. Note: exp(0.5 * log_σ²) = σ.

### Step 5: Implement the Loss
Combine BCE reconstruction loss with KL divergence. Both terms should be summed over dimensions and averaged over the batch.

### Step 6: Training Loop
Standard gradient descent. Monitor both loss components separately — if KL goes to 0, you have "posterior collapse" (the model ignores the latent code). If reconstruction loss stays high, the latent space is too constrained.

### Step 7: Generate New Samples
Sample z ~ N(0, I) and pass through the decoder. Also explore latent space interpolation: encode two images, linearly interpolate their μ vectors, decode each interpolation step.

## Learning Objectives

- Understand the mathematical foundation of variational inference and the ELBO
- Implement the reparameterization trick and understand why it's necessary
- Build and train a generative model with a principled probabilistic loss
- Analyze the tradeoff between reconstruction quality and latent space regularization
- Compare VAE generation quality and training stability with GANs (Day 64)
- Explore latent space structure through interpolation and sampling

## Going Deeper

- **Posterior collapse:** When the decoder is too powerful (e.g., autoregressive), it ignores z entirely and KL → 0. Solutions: KL annealing (gradually increase β from 0 to 1), free bits (minimum KL per dimension).
- **Convolutional VAE:** Replace MLPs with conv/deconv layers for better image quality.
- **VQ-VAE:** Replace continuous latent space with discrete codes — the foundation of modern image tokenizers (DALL-E, Stable Diffusion's latent space).
- **Conditional VAE (CVAE):** Condition generation on a label — generate specific digits, specific styles.
- **Disentanglement:** β-VAE, FactorVAE, DIP-VAE — techniques to make each latent dimension correspond to one interpretable factor.
- **Connection to diffusion models:** A diffusion model can be viewed as a hierarchical VAE with many latent layers and a fixed (non-learned) encoder. Understanding VAEs deeply is prerequisite to understanding diffusion.
