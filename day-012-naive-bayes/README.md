# Day 012: Naive Bayes Classifier

## What You're Building

A Naive Bayes text classifier from scratch — no sklearn, no libraries for the core algorithm. You'll build a spam detector that learns the probabilistic "fingerprint" of spam vs. ham messages, then classifies new messages by computing which category they most likely belong to.

Naive Bayes is one of the most practically important algorithms in ML history. It powered the first generation of email spam filters, remains competitive for text classification despite being "naive," and runs fast enough for real-time classification. Understanding it deeply teaches you Bayesian reasoning — the foundation of probabilistic ML.

## Core Concepts

### Bayes' Theorem — The Engine

Everything starts with Bayes' theorem:

```
P(class | document) = P(document | class) * P(class) / P(document)
```

In English: "The probability a document belongs to a class, given the words we see, equals how likely those words are in that class, times how common that class is, divided by how likely those words are overall."

We don't need P(document) for classification — it's the same for all classes, so it cancels out when comparing. We only need the **numerator**:

```
P(class | document) ∝ P(document | class) * P(class)
```

- **P(class)** = the **prior**. How common is spam vs. ham in our training data? If 30% of emails are spam, P(spam) = 0.3.
- **P(document | class)** = the **likelihood**. How probable is this exact combination of words, given the class?

### The "Naive" Assumption — Why It Works Anyway

Computing P(document | class) for a full document is intractable — you'd need to estimate the joint probability of every possible word combination. The "naive" assumption is **conditional independence**: given the class, each word's presence is independent of every other word.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

This is obviously wrong — "Nigerian" and "prince" are not independent in spam. But it works remarkably well in practice because:

1. **Classification doesn't need accurate probabilities** — it only needs the correct *ranking* of classes.
2. **Errors in independence assumptions tend to cancel out** across many features.
3. **The model has very few parameters** to estimate, so it doesn't overfit even with small datasets.

### Log-Space Arithmetic — Avoiding Numerical Underflow

Multiplying many small probabilities (e.g., P(word|spam) = 0.001) quickly underflows to zero in floating point. The fix: work in **log space**.

```
log P(class | doc) = log P(class) + Σ log P(wi | class)
```

Multiplication becomes addition. Products of tiny numbers become sums of negative numbers. This is numerically stable and actually faster.

### Laplace Smoothing — Handling Unseen Words

What if a word appears in test data but never appeared in training? P(word | class) = 0, which zeros out the entire product — one unseen word kills the prediction. 

**Laplace smoothing** (additive smoothing) fixes this by adding a pseudocount α (typically 1) to every word count:

```
P(word | class) = (count(word, class) + α) / (total_words_in_class + α * vocabulary_size)
```

This ensures no probability is ever zero. α = 1 corresponds to a uniform Dirichlet prior — you're saying "before seeing any data, every word is equally likely." Larger α means stronger smoothing (more uniform), smaller α means less.

### Bag of Words — The Document Representation

We represent documents as **bags of words** — unordered counts of word occurrences. "the cat sat on the mat" becomes {the: 2, cat: 1, sat: 1, on: 1, mat: 1}. Word order is discarded.

This is a lossy representation, but it captures the key signal for classification: which words appear and how often. Combined with the naive independence assumption, it makes the math tractable.

## Step-by-Step Breakdown

### Step 1: Text Preprocessing
Tokenize text into words, lowercase everything, remove punctuation. This normalizes the input so "Free" and "free" are treated as the same feature. Without this, you'd split your evidence across multiple representations of the same word, weakening each one.

### Step 2: Compute Prior Probabilities
Count documents in each class. P(spam) = num_spam / num_total. These priors capture the base rate — if 90% of emails are ham, a message starts with a strong presumption of being ham before we even look at the words.

### Step 3: Build Word Likelihood Tables
For each class, count how often each word appears across all documents in that class. Apply Laplace smoothing. Store log probabilities for numerical stability. This is the "training" step — you're building a probabilistic vocabulary fingerprint for each class.

### Step 4: Classify New Documents
For a new document, compute the log-posterior for each class: log P(class) + sum of log P(word | class) for each word. The class with the highest log-posterior wins. Words not in the vocabulary are ignored (they contribute equally to all classes).

### Step 5: Evaluate
Compute accuracy, precision, recall, and F1 score. For spam detection, precision matters (don't flag legitimate email as spam) but recall also matters (don't let spam through). The F1 score balances both.

## Learning Objectives

- Implement Bayesian classification from first principles
- Understand why the "naive" independence assumption works in practice
- Handle numerical underflow with log-space arithmetic
- Apply Laplace smoothing to avoid zero-probability problems
- Build a complete text classification pipeline: tokenization, training, prediction, evaluation
- Connect to prior days: compare the decision boundary approach (logistic regression, decision trees) vs. the generative model approach (Naive Bayes)

## Going Deeper

- **Multinomial vs. Bernoulli vs. Gaussian NB**: We implement Multinomial (word counts). Bernoulli uses binary word presence. Gaussian handles continuous features. Each makes different assumptions about the data.
- **TF-IDF weighting**: Raw word counts overweight common words. TF-IDF downweights words that appear in many documents ("the", "is") and upweights distinctive words.
- **Feature selection**: Not all words are equally informative. Mutual information or chi-squared tests can select the most discriminative features.
- **Complement Naive Bayes**: When classes are imbalanced, estimating P(word | class) from the *complement* of each class can improve performance.
- **Production systems**: In production, Naive Bayes is often used as a fast first-pass filter (e.g., spam detection), with more expensive models (deep learning) as a second pass for uncertain cases.
- **Connection to Day 001 (Linear Regression) and Day 003 (Logistic Regression)**: Naive Bayes is a *generative* classifier — it models P(X|Y) and uses Bayes' rule. Logistic regression is *discriminative* — it models P(Y|X) directly. When the naive assumption holds, NB converges faster with less data. When it doesn't, logistic regression typically wins with enough data.
