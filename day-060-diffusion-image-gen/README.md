# Day 060: Image Generation with Diffusion

## Overview

Build a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch using only NumPy. You'll implement the complete forward noising process, learn the reverse denoising process with a small neural network, and generate images by iteratively removing noise — the same foundational mechanism behind Stable Diffusion, DALL-E, and Imagen.

**Why this matters:** Diffusion models have become the dominant paradigm for image generation, surpassing GANs in both quality and training stability. Understanding the math behind the forward/reverse process — how you systematically destroy information and then learn to reconstruct it — is essential for anyone working with generative AI in production.

## Core Concepts

### The Diffusion Process: Destroying and Rebuilding Information

The key insight of diffusion models is strikingly simple: if you can learn to reverse a gradual noising process, you can generate data from pure noise.

**Forward process (adding noise):** Given a clean image x₀, we progressively add Gaussian noise over T timesteps:

```
q(xₜ | xₜ₋₁) = N(xₜ; √(1 - βₜ) · xₜ₋₁, βₜ · I)
```

where βₜ is a noise schedule that controls how much noise to add at each step. The beauty is that we can skip directly to any timestep t using the **closed-form formula**:

```
q(xₜ | x₀) = N(xₜ; √ᾱₜ · x₀, (1 - ᾱₜ) · I)
```

where αₜ = 1 - βₜ and ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ (cumulative product of alphas).

This means: `xₜ = √ᾱₜ · x₀ + √(1 - ᾱₜ) · ε` where ε ~ N(0, I)

**Intuition:** At t=0, we have a clean image. At t=T, we have pure Gaussian noise. The forward process defines a smooth path between data and noise.

### The Noise Schedule

The noise schedule β₁, β₂, ..., βₜ controls the rate of information destruction:

- **Linear schedule:** βₜ increases linearly from β₁ to βₜ. Simple but the image is destroyed too quickly in early steps.
- **Cosine schedule:** βₜ = 1 - ᾱₜ/ᾱₜ₋₁ where ᾱₜ = cos²((t/T + s)/(1+s) · π/2). Preserves more information in early steps, leading to better quality.

The schedule directly affects generation quality — too aggressive and details are lost before the model can learn them; too gentle and training is slow.

### The Reverse Process: Learning to Denoise

The reverse process learns to undo each noising step:

```
p_θ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μ_θ(xₜ, t), σₜ² · I)
```

The model predicts either the noise ε or the clean image x₀. The **noise prediction** formulation is standard because it leads to a simpler loss:

```
L = E[‖ε - ε_θ(xₜ, t)‖²]
```

This is just MSE between the actual noise added and the model's prediction of that noise.

Given the predicted noise ε_θ, we compute the mean of the reverse step:

```
μ_θ(xₜ, t) = (1/√αₜ) · (xₜ - (βₜ/√(1 - ᾱₜ)) · ε_θ(xₜ, t))
```

### Timestep Conditioning

The model must know *which* timestep it's denoising — removing heavy noise (large t) is a very different task than removing faint noise (small t). We encode the timestep using **sinusoidal embeddings** (same idea as positional encodings in transformers), which the network receives as additional input.

### Sampling (Generation)

Starting from pure noise xₜ ~ N(0, I), we iterate backward:

```
xₜ₋₁ = μ_θ(xₜ, t) + σₜ · z,  where z ~ N(0, I) for t > 1, z = 0 for t = 1
```

Each step slightly denoises the image. After T steps, we have a generated sample.

## Step-by-Step Breakdown

1. **Define the noise schedule** — Compute βₜ, αₜ, and ᾱₜ for all timesteps. These are precomputed once and used throughout training and sampling.

2. **Implement the forward process** — Given a clean image and a timestep, compute the noisy version using the closed-form formula. This is how we create training data.

3. **Build the denoising network** — A small MLP that takes a flattened noisy image + timestep embedding and predicts the noise that was added. For real images you'd use a U-Net, but an MLP suffices for learning the concept.

4. **Implement sinusoidal timestep embedding** — Encode the integer timestep into a continuous vector the network can use.

5. **Training loop** — Sample a clean image, sample a random timestep, add noise, predict the noise, compute MSE loss, backpropagate.

6. **Sampling loop** — Start from noise, iteratively apply the learned reverse process to generate an image.

7. **Evaluate quality** — Compare generated samples against the training distribution visually and statistically.

## Learning Objectives

- Understand the mathematical framework of diffusion: forward process, reverse process, and their connection through Bayes' rule
- Implement noise schedules and analyze their effect on information preservation
- Build a timestep-conditioned neural network for noise prediction
- Implement the DDPM sampling algorithm from pure noise to generated images
- Connect diffusion models to score matching and denoising score matching
- Understand why diffusion models train more stably than GANs (no adversarial dynamics)

## Going Deeper

- **DDIM sampling:** Deterministic sampling that allows fewer steps (50 instead of 1000) with comparable quality
- **Classifier-free guidance:** Mixing conditional and unconditional predictions for controllable generation
- **Latent diffusion:** Running diffusion in a compressed latent space (the "Stable" in Stable Diffusion) for efficiency
- **Score-based formulation:** Diffusion models are equivalent to learning the score function ∇ₓ log p(x), connecting to score matching theory
- **Noise prediction vs x₀ prediction vs v-prediction:** Different parameterizations of the same objective, each with practical tradeoffs
- **Progressive distillation:** Training a student model to take fewer denoising steps
