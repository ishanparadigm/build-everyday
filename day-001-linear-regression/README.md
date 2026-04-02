# Day 001: Linear Regression from Scratch

## What You're Building

A complete linear regression engine — no scikit-learn, no PyTorch, just NumPy and raw math. You'll implement both the closed-form (Normal Equation) and iterative (Gradient Descent) solutions, understand exactly when and why each breaks down, and build the evaluation machinery to prove your model works.

Linear regression is the foundation of modern machine learning. Every neural network, at its core, is performing linear transformations followed by nonlinearities. Understanding regression deeply means understanding the optimization landscape that all of ML lives on.

## Core Concepts

### The Model

Linear regression models the relationship between input features **X** and a continuous target **y** as:

```
y = Xw + b
```

Where **w** is a weight vector and **b** is a bias (intercept). We can absorb the bias into **w** by prepending a column of ones to **X**, giving us:

```
y = Xw
```

This isn't just a modeling trick — it's a core idea in linear algebra. By augmenting the feature matrix, we're embedding our data in a higher-dimensional space where the intercept becomes just another weight. This same idea (adding dimensions to make problems linear) recurs throughout ML, from kernel methods to positional encodings.

### The Loss Function

We minimize **Mean Squared Error (MSE)**:

```
L(w) = (1/n) * ||Xw - y||^2
```

Why squared error? Three reasons:
1. **Differentiability** — absolute error has a kink at zero; squared error is smooth everywhere
2. **Convexity** — the loss surface is a paraboloid with a single global minimum (no local minima traps)
3. **Probabilistic interpretation** — minimizing MSE is equivalent to Maximum Likelihood Estimation when errors are Gaussian. If you assume y = Xw + ε where ε ~ N(0, σ²), the MLE for w is exactly the MSE minimizer.

### Solution 1: The Normal Equation (Closed-Form)

Taking the gradient of L(w) and setting it to zero:

```
∇L = (2/n) * X^T(Xw - y) = 0
X^T X w = X^T y
w* = (X^T X)^{-1} X^T y
```

This gives us the exact solution in one shot. The matrix X^T X is called the **Gram matrix** — it captures the correlations between all pairs of features.

**When it breaks:**
- **Computational cost**: Inverting X^T X is O(d³) where d = number of features. For d = 100,000 features, this is intractable.
- **Numerical instability**: If features are highly correlated, X^T X becomes ill-conditioned (near-singular). Small floating-point errors get amplified catastrophically.
- **Memory**: Storing X^T X requires O(d²) memory.

**Practical note**: In production, nobody actually computes the inverse. They use the **pseudoinverse** via SVD or QR decomposition, which is numerically stable. We'll implement the naive version first, then discuss why.

### Solution 2: Gradient Descent (Iterative)

Instead of solving analytically, we iteratively walk downhill on the loss surface:

```
w_{t+1} = w_t - α * ∇L(w_t)
w_{t+1} = w_t - α * (2/n) * X^T(Xw_t - y)
```

Where α is the **learning rate** — the step size.

**Why this matters more than the Normal Equation:**
- Scales to millions of features and billions of data points (via mini-batching)
- Generalizes to ANY differentiable loss function — logistic regression, neural networks, transformers all use the same core loop
- The gradient descent update rule is the single most important equation in modern ML

**The learning rate tradeoff:**
- Too small: convergence takes forever, may get stuck in flat regions
- Too large: overshoots the minimum, oscillates, or diverges entirely
- Just right: smooth convergence in reasonable iterations

**Feature scaling is critical for gradient descent.** If feature 1 ranges [0, 1] and feature 2 ranges [0, 1000000], the loss surface becomes an elongated ellipsoid. Gradient descent zigzags across the narrow dimension instead of heading straight to the minimum. Standardization (zero mean, unit variance) makes the surface spherical and descent efficient.

### Evaluation: R² Score

```
R² = 1 - (SS_res / SS_tot)
R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
```

Interpretation:
- R² = 1.0: perfect prediction
- R² = 0.0: model is no better than predicting the mean
- R² < 0.0: model is worse than predicting the mean (yes, this is possible — it means your model is actively harmful)

## Step-by-Step Approach

### Step 1: Generate synthetic data
We create data where y = 3x₁ + 7x₂ + 2 + noise, so we know the ground truth. This lets us verify our implementation is correct — if we can't recover known coefficients, something is wrong.

### Step 2: Feature augmentation
Prepend a column of ones to X to absorb the bias term. This turns an (n, d) matrix into (n, d+1) and lets us treat bias as just another weight.

### Step 3: Normal Equation solution
Compute w* = (X^T X)^{-1} X^T y directly. Compare recovered weights against ground truth.

### Step 4: Gradient Descent solution
Initialize weights to zero (or random). Iterate: compute predictions, compute gradient, update weights. Track loss at each step to verify convergence.

### Step 5: Feature scaling experiment
Run gradient descent with and without standardization. Show that unscaled features require tiny learning rates and many more iterations.

### Step 6: Evaluation
Compute MSE and R² on held-out test data for both methods. They should match (both find the same optimum for this convex problem).

## Learning Objectives

- Derive and implement the Normal Equation from the MSE loss function
- Implement batch gradient descent with convergence tracking
- Understand why feature scaling is essential for gradient-based optimization
- Build intuition for the loss landscape geometry of convex problems
- Implement MSE and R² evaluation metrics from scratch
- See that closed-form and iterative methods converge to the same solution

## Going Deeper

- **Regularization**: Add L2 penalty (Ridge) to the Normal Equation: w* = (X^T X + λI)^{-1} X^T y. This shrinks weights and fixes the ill-conditioning problem. L1 penalty (Lasso) induces sparsity but has no closed-form solution.
- **Stochastic Gradient Descent (SGD)**: Use a random subset of data at each step. Noisier gradients but vastly faster per iteration. The noise actually helps escape saddle points in non-convex problems (neural nets).
- **Learning rate schedules**: Start with a large α and decay over time. Combines fast initial progress with fine-grained convergence.
- **Connection to neural networks**: A neural network with no hidden layers and MSE loss IS linear regression. Adding layers and nonlinearities is what makes it "deep" — but the optimization loop is the same gradient descent you'll implement today.
- **Polynomial regression**: Feed x, x², x³ as features to a linear regression model. It's still "linear" in the parameters — the model is linear in **w**, even though the relationship to the original features is nonlinear.
