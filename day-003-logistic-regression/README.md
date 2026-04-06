# Day 003: Logistic Regression from Scratch

## Overview

Logistic regression is the foundational algorithm for **binary classification** — predicting whether something belongs to class 0 or class 1. Spam or not spam. Tumor malignant or benign. Transaction fraudulent or legitimate.

Despite its name, logistic regression is a **classification** algorithm, not a regression one. The name comes from the fact that it uses a regression framework (linear combination of features) but wraps it in a logistic function to produce probabilities. It's the bridge between Day 001's linear regression and the neural networks we'll build later — in fact, a single neuron with a sigmoid activation *is* logistic regression.

Understanding logistic regression deeply matters because:
- It's the building block of neural networks (each neuron is essentially a logistic unit)
- It introduces **maximum likelihood estimation**, the dominant paradigm in modern ML
- It teaches you the **cross-entropy loss**, which powers everything from GPT to image classifiers
- It's still widely used in production for interpretable, high-stakes decisions (credit scoring, medical diagnosis)

## Core Concepts

### Why Not Just Use Linear Regression for Classification?

Linear regression predicts unbounded continuous values: y = wᵀx + b can output anything from -∞ to +∞. But for classification, we need probabilities in [0, 1]. If we threshold a linear regression at 0.5, it "works" on simple cases but breaks badly:

1. **Outliers distort the decision boundary.** A single extreme data point pulls the regression line, moving the threshold for *all* predictions.
2. **Outputs aren't calibrated probabilities.** Linear regression might predict 1.7 or -0.3 — these aren't meaningful as probabilities.
3. **The loss landscape is wrong.** Mean squared error on classification creates a non-convex optimization problem with multiple local minima.

### The Sigmoid (Logistic) Function

The sigmoid function maps any real number to (0, 1):

```
σ(z) = 1 / (1 + e^(-z))
```

Key properties that make it perfect for classification:
- **Output range (0, 1):** Directly interpretable as probability
- **Monotonic:** Higher z → higher probability (preserves ordering from the linear model)
- **Symmetric:** σ(-z) = 1 - σ(z), so P(class=0) = 1 - P(class=1)
- **Beautiful derivative:** σ'(z) = σ(z) · (1 - σ(z)), which makes gradient computation elegant

The model becomes: **P(y=1|x) = σ(wᵀx + b)**

### Maximum Likelihood Estimation

In linear regression (Day 001), we minimized squared error. For classification, we use a fundamentally different approach: **maximize the likelihood of the observed data**.

Given data points (xᵢ, yᵢ) where yᵢ ∈ {0, 1}, the likelihood is:

```
L(w, b) = ∏ᵢ p(yᵢ|xᵢ; w, b)
```

For a single point with predicted probability p̂:
- If yᵢ = 1: we want p̂ to be high → contribute p̂ to likelihood
- If yᵢ = 0: we want p̂ to be low → contribute (1 - p̂) to likelihood

Compact form: `p(yᵢ|xᵢ) = p̂ᵢ^yᵢ · (1 - p̂ᵢ)^(1-yᵢ)`

### Cross-Entropy Loss (Log Loss)

Taking the negative log of the likelihood (because products → sums, and we minimize by convention):

```
J(w, b) = -1/m Σᵢ [yᵢ · log(p̂ᵢ) + (1 - yᵢ) · log(1 - p̂ᵢ)]
```

Why this loss function is brilliant:
- **Convex** in the parameters (w, b) — guaranteed to find the global minimum
- **Penalizes confident wrong predictions harshly:** If yᵢ = 1 but p̂ᵢ → 0, then -log(p̂ᵢ) → ∞
- **Rewards calibrated probabilities:** The minimum is achieved when p̂ᵢ equals the true conditional probability
- **Information-theoretic meaning:** It measures the "surprise" or information lost when using p̂ to approximate the true distribution

### Gradient Derivation

The gradient of the cross-entropy loss w.r.t. weights turns out to be elegantly simple:

```
∂J/∂w = 1/m · Xᵀ(p̂ - y)
∂J/∂b = 1/m · Σ(p̂ᵢ - yᵢ)
```

This is *identical in form* to the linear regression gradient from Day 001! The only difference is that p̂ = σ(Xw + b) instead of p̂ = Xw + b. This is not a coincidence — it's a deep property of the exponential family of distributions.

The derivation:
1. ∂J/∂z = p̂ - y  (using σ'(z) = σ(z)(1-σ(z)), the terms cancel beautifully)
2. ∂z/∂w = X  (since z = Xw + b)
3. Chain rule: ∂J/∂w = Xᵀ · (∂J/∂z) / m

### Decision Boundary

Logistic regression produces a **linear decision boundary** — the set of points where P(y=1|x) = 0.5, i.e., where wᵀx + b = 0. This is a hyperplane in feature space:
- In 2D: a line
- In 3D: a plane
- In nD: a hyperplane

Points on one side are classified as 1, the other as 0. The distance from the boundary relates to prediction confidence.

## Step-by-Step Approach

### Step 1: Feature Scaling
Just like Day 001, gradient descent converges faster with standardized features (zero mean, unit variance). Without scaling, features on different scales create elongated contours, making gradient descent zigzag slowly.

### Step 2: Initialize Parameters
Start with weights = 0 and bias = 0. Unlike neural networks, logistic regression is convex, so initialization doesn't matter for correctness (but affects convergence speed).

### Step 3: Forward Pass
Compute z = Xw + b, then apply sigmoid to get predicted probabilities p̂ = σ(z). We must handle numerical stability — when z is very large/negative, exp(-z) can overflow.

### Step 4: Compute Loss
Calculate cross-entropy loss. Add a small epsilon inside log() to avoid log(0) = -∞.

### Step 5: Compute Gradients
Calculate ∂J/∂w and ∂J/∂b using the formulas derived above.

### Step 6: Update Parameters
w ← w - α · ∂J/∂w and b ← b - α · ∂J/∂b, where α is the learning rate.

### Step 7: Iterate and Evaluate
Repeat steps 3-6. Monitor loss convergence. Evaluate using accuracy, precision, recall, and the confusion matrix.

## Learning Objectives

- Understand why linear regression fails for classification and how the sigmoid function fixes it
- Derive and implement cross-entropy loss from maximum likelihood principles
- Implement gradient descent for logistic regression, connecting it to Day 001's approach
- Build numerical stability techniques (log-sum-exp, epsilon clipping)
- Evaluate classification models with accuracy, precision, recall, and F1-score
- Understand the decision boundary as a geometric object in feature space

## Going Deeper

### Regularization
Adding L2 regularization (ridge) to logistic regression: J_reg = J + λ/2m · ||w||². This prevents overfitting and is mathematically equivalent to placing a Gaussian prior on the weights (Bayesian interpretation). L1 regularization produces sparse weights — useful for feature selection.

### Multi-class Extension
Logistic regression extends to K classes via **softmax regression**: P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx). The cross-entropy loss generalizes naturally, and this is exactly what the output layer of a classification neural network does.

### Connection to Neural Networks
A single-layer neural network with sigmoid activation and binary cross-entropy loss *is* logistic regression. Deep learning stacks many of these units, but understanding one unit deeply is essential before scaling up. We'll build on this in Day 007 (forward pass) and Day 008 (backpropagation).

### Newton's Method
Gradient descent is first-order (uses only the gradient). Newton's method uses the second derivative (Hessian) to converge in fewer iterations. For logistic regression, this is called **Iteratively Reweighted Least Squares (IRLS)** and converges quadratically near the optimum, but each iteration is O(d³) vs O(md) for gradient descent.
