# Day 004: Decision Tree Classifier from Scratch

## What You're Building

A decision tree classifier — the algorithm that learns to ask a sequence of yes/no questions about your data to arrive at a prediction. Decision trees are foundational to some of the most powerful models in production today (Random Forests, XGBoost, LightGBM), and understanding how they work from the inside out is essential before using those ensemble methods.

Unlike the linear models we built on Days 001 and 003, decision trees can capture **non-linear decision boundaries** — they can learn rules like "if age > 30 AND income < 50k, then class B" without you having to manually engineer interaction features. This flexibility is their superpower and their curse (overfitting).

## Core Concepts

### Information Theory: Entropy and Information Gain

The central question in building a decision tree is: **at each node, which feature should we split on?** We want the split that gives us the most "information" — that is, the split that best separates the classes.

**Entropy** measures the impurity (or uncertainty) of a set of labels:

```
H(S) = -Σ p_i * log2(p_i)
```

where `p_i` is the proportion of class `i` in set `S`.

Intuition:
- If all samples belong to one class: H = 0 (no uncertainty — we know the answer)
- If samples are evenly split between 2 classes: H = 1 (maximum uncertainty for binary)
- Entropy is maximized when the distribution is uniform

**Information Gain** measures how much entropy decreases after splitting on a feature:

```
IG(S, feature) = H(S) - Σ (|S_v| / |S|) * H(S_v)
```

where `S_v` is the subset of samples where the feature takes value `v`. We're computing the weighted average entropy of the children and subtracting from the parent entropy. The feature with the highest IG wins.

### Gini Impurity: An Alternative Splitting Criterion

While entropy uses logarithms, **Gini impurity** is computationally cheaper:

```
Gini(S) = 1 - Σ p_i²
```

Intuition: Gini measures the probability that a randomly chosen sample would be misclassified if labeled according to the distribution. Like entropy, Gini = 0 for a pure node.

In practice, Gini and entropy produce very similar trees. Gini is slightly faster to compute (no logarithm), which is why scikit-learn defaults to it.

### The Recursive Splitting Algorithm (ID3/CART)

Building a decision tree is a **greedy, recursive** process:

1. **Base cases**: Stop if (a) all samples have the same label (pure node), (b) no features left to split on, (c) we've hit max depth, or (d) too few samples remain.
2. **Find best split**: For each feature, compute information gain (or Gini reduction). Pick the one that maximizes it. For continuous features, we must also find the best threshold — try all midpoints between sorted unique values.
3. **Split**: Partition the data on the best feature/threshold.
4. **Recurse**: Build left and right subtrees on the partitioned data.
5. **Predict**: At a leaf node, return the majority class.

### Why Greedy? The Tradeoff

Finding the globally optimal decision tree is NP-complete. The greedy approach (pick the locally best split at each node) is a practical approximation. This means decision trees can miss globally better structures, but in practice they work remarkably well — especially when ensembled.

### Overfitting and Regularization

A fully-grown decision tree will memorize the training data (100% training accuracy) by creating one leaf per sample. This is classic overfitting. Controls include:

- **Max depth**: Limit how deep the tree can grow
- **Min samples per leaf**: Don't split if a child would have too few samples
- **Min samples to split**: Require a minimum number of samples at a node to consider splitting
- **Pruning**: Grow the full tree, then remove branches that don't improve validation accuracy (post-pruning). We'll implement pre-pruning via max_depth and min_samples.

### Continuous vs. Categorical Features

For **categorical features**: consider each possible value as a branch.
For **continuous features**: find the best binary threshold. Sort the values, test splits at each midpoint between consecutive distinct values, and pick the threshold with the highest information gain.

Our implementation handles continuous features, which is more general and matches how CART (Classification and Regression Trees) works.

## Step-by-Step Breakdown

### Step 1: Compute Entropy / Gini
Calculate the impurity of a label set. This is the building block for evaluating splits.

### Step 2: Find the Best Split
For each feature, iterate over possible thresholds (midpoints of sorted unique values). Compute the information gain for each candidate split. Track the best feature + threshold combination.

**What would go wrong without this?** Random splits would produce trees that are no better than random guessing. The information-theoretic criterion is what gives the tree its predictive power.

### Step 3: Recursive Tree Construction
At each node, find the best split and recurse. Apply stopping criteria (max depth, min samples, pure node) to prevent overfitting.

**What would go wrong without stopping criteria?** The tree would grow until every leaf contains a single sample — perfect training accuracy, terrible generalization.

### Step 4: Prediction
Traverse the tree from root to leaf for each test sample. At each internal node, go left if `feature_value <= threshold`, else go right. Return the leaf's majority class.

### Step 5: Evaluation
Compute accuracy and visualize the tree structure to verify it learned sensible rules.

## Learning Objectives

- Understand information theory (entropy, information gain) and why it's the right framework for feature selection
- Implement the CART algorithm from scratch with continuous feature support
- Build intuition for the bias-variance tradeoff through max_depth experimentation
- See how a non-linear classifier differs from the linear models in Days 001 and 003
- Understand recursive data structures (the tree itself) and recursive algorithms (building and traversing)

## Going Deeper

- **Random Forests**: Build many decision trees on random subsets of data and features, then vote. This reduces variance dramatically. Each tree overfits differently, so the ensemble generalizes.
- **Gradient Boosting (XGBoost, LightGBM)**: Build trees sequentially, where each new tree corrects the errors of the previous ensemble. The dominant algorithm for tabular data in production.
- **Pruning strategies**: Cost-complexity pruning (CART's approach) grows the full tree then prunes back using a complexity parameter α. More principled than just setting max_depth.
- **Regression trees**: Instead of majority vote at leaves, predict the mean target value. Split criterion becomes variance reduction instead of information gain.
- **Feature importance**: Count how much each feature reduces impurity across all splits — a powerful interpretability tool that comes free with tree models.
