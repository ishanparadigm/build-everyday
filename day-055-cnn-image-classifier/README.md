# Day 55: Image Classifier with CNN

## Overview

Build a Convolutional Neural Network (CNN) from scratch — no PyTorch, no TensorFlow, just NumPy. You'll implement convolution layers, pooling, ReLU activations, and a fully connected output layer, then train it to classify handwritten digits. This is the foundation of every modern computer vision system, from self-driving cars to medical imaging.

## Why This Matters

CNNs revolutionized computer vision because they exploit **spatial structure** in data. Before CNNs, image classifiers treated each pixel independently (like the KNN and naive Bayes classifiers from earlier days). A CNN learns hierarchical features — edges → textures → shapes → objects — by sharing weights across spatial positions. This makes them dramatically more parameter-efficient and translation-invariant than fully connected networks.

## Core Concepts

### 1. Convolution Operation

The convolution layer slides a small **kernel** (filter) across the input image and computes element-wise products at each position:

```
output[i,j] = Σ_m Σ_n input[i+m, j+n] × kernel[m, n]
```

For a 3×3 kernel on a 28×28 image, the kernel visits 26×26 positions (no padding), producing a 26×26 **feature map**. Each kernel detects one type of pattern (horizontal edge, vertical edge, corner, etc.). Using K kernels produces K feature maps — the **depth** of the output.

**Why convolution works**: A horizontal-edge detector has the same weights whether the edge is in the top-left or bottom-right of the image. This **weight sharing** means a CNN with a 3×3 kernel has only 9 parameters per filter, versus 784×hidden for a fully connected layer on MNIST. Fewer parameters = less overfitting, faster training, and built-in translation invariance.

**Cross-correlation vs. convolution**: In math, convolution flips the kernel before sliding. In deep learning, we skip the flip (technically computing cross-correlation). It doesn't matter because the kernel weights are learned — flipping a learned kernel gives you another valid kernel.

### 2. ReLU Activation

```
ReLU(x) = max(0, x)
```

Applied element-wise after convolution. It introduces non-linearity (without it, stacking convolution layers just produces another linear operation). ReLU is preferred over sigmoid/tanh because:
- **No vanishing gradient**: gradient is exactly 1 for positive inputs
- **Sparse activation**: negative values become zero, creating sparse representations
- **Cheap to compute**: just a threshold comparison

The gradient is: `dReLU/dx = 1 if x > 0, else 0`. Dead neurons (always negative) can be a problem — but in practice, ReLU works well for most architectures.

### 3. Max Pooling

Downsamples each feature map by taking the maximum value in each non-overlapping window (typically 2×2):

```
Input:  [1 3]    → Output: 4
        [2 4]
```

**Why max pooling**:
- Reduces spatial dimensions by 2× in each direction (4× fewer values)
- Provides a small amount of **translation invariance** — shifting an edge by 1 pixel doesn't change the max in a 2×2 window
- Reduces computation for subsequent layers

**Backprop through max pooling**: The gradient flows only to the position that had the max value. All other positions get zero gradient. You need to remember which position was the max during the forward pass.

### 4. Softmax Output

Converts raw logits into probabilities:

```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

For numerical stability, subtract `max(z)` before exponentiating: `exp(z_i - max(z))`. Without this, `exp(z)` overflows for large values.

### 5. Cross-Entropy Loss

```
L = -Σ_i y_i × log(p_i)
```

For one-hot labels (only one class is correct), this simplifies to `L = -log(p_correct)`. The gradient with respect to the softmax input is beautifully simple: `dL/dz_i = p_i - y_i`. This is the same elegant result as logistic regression (Day 3).

### 6. Backpropagation Through Convolution

This is the trickiest part. During backprop through a conv layer:
- **Gradient w.r.t. kernels**: Convolve the input with the upstream gradient (this accumulates how much each kernel weight contributed to the loss)
- **Gradient w.r.t. input**: Convolve the upstream gradient with the **flipped** kernel (this propagates the error signal backwards through the layer)

The math is the same as the forward pass, just with different operands.

## Architecture

```
Input (1×28×28)
  → Conv Layer (8 filters, 3×3) → ReLU → Output: 8×26×26
  → Max Pool (2×2) → Output: 8×13×13
  → Flatten → Output: 1352
  → Fully Connected (1352 → 10) → Softmax
  → Cross-Entropy Loss
```

This is minimal but effective — enough to reach ~95%+ accuracy on MNIST in just a few epochs.

## Step-by-Step Breakdown

### Step 1: Data Loading
Load MNIST digits (or generate synthetic digit-like data). Normalize pixel values to [0, 1] by dividing by 255. This keeps activations in a reasonable range and helps gradient flow.

### Step 2: Initialize Parameters
- Conv kernels: 8 filters of size 3×3, initialized with He initialization: `N(0, sqrt(2/fan_in))` where fan_in = 3×3 = 9. He init accounts for ReLU killing half the values.
- FC weights: shape (1352, 10), also He initialized.
- FC biases: zeros.

### Step 3: Forward Pass
Run input through conv → ReLU → pool → flatten → FC → softmax. Cache intermediate values needed for backprop (pre-ReLU activations, max pool indices, etc.).

### Step 4: Compute Loss
Cross-entropy between predicted probabilities and true one-hot labels.

### Step 5: Backward Pass
Compute gradients in reverse order:
1. Softmax + CE gradient: `dL/dz = predictions - one_hot_labels`
2. FC layer gradient: standard matmul backprop
3. Unflatten gradient back to pooled shape
4. Max pool backward: route gradients to max positions
5. ReLU backward: zero out gradients where input was ≤ 0
6. Conv backward: compute kernel gradients and input gradients

### Step 6: Update Parameters
SGD: `param -= learning_rate × gradient`. Simple but effective for this architecture.

### Step 7: Training Loop
Iterate over mini-batches for multiple epochs. Track loss and accuracy to verify learning.

## Learning Objectives

- Implement 2D convolution as a sliding-window operation from scratch
- Understand how weight sharing in CNNs reduces parameters vs. fully connected networks
- Implement backpropagation through convolution, pooling, and fully connected layers
- Build intuition for feature hierarchies (edges → patterns → shapes)
- Understand max pooling's role in spatial invariance and dimensionality reduction
- See how He initialization and ReLU work together to enable deep network training

## Going Deeper

- **Multiple conv layers**: Stack conv→ReLU→pool blocks. Each layer learns higher-level features. Modern architectures (ResNet, VGG) use dozens of layers.
- **Padding**: Add zeros around the input so output has the same spatial dimensions. Important for very deep networks where you'd otherwise shrink to 1×1.
- **Stride**: Skip positions during convolution to downsample without pooling.
- **Batch normalization**: Normalize activations between layers. Dramatically speeds up training and allows higher learning rates.
- **Data augmentation**: Random rotations, shifts, and scaling create virtually infinite training data.
- **Depthwise separable convolutions**: Factor the convolution into spatial and channel-wise operations (used in MobileNet). Reduces parameters by ~9× for a 3×3 kernel.
- **Production systems**: Real image classifiers use GPU-accelerated frameworks with automatic differentiation. The from-scratch implementation here teaches you what those frameworks do under the hood.
