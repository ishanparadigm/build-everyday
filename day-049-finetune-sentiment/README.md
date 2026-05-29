# Day 049: Fine-Tuning a Sentiment Classifier

## Overview

You're building a sentiment classifier by fine-tuning a pre-trained language model on labeled text data. This is arguably the most important practical skill in modern NLP — taking a powerful general-purpose model and specializing it for your domain with a small amount of labeled data.

**Why this matters:** Pre-trained models like BERT understand language structure (syntax, semantics, context) from being trained on massive corpora. But they don't know *your* task. Fine-tuning bridges that gap — you take the model's general language understanding and teach it to map text to your specific labels (positive/negative, spam/not-spam, etc.) using just hundreds or thousands of examples. This is how most production NLP systems work today.

## Core Concepts

### Transfer Learning: Standing on the Shoulders of Giants

The key insight behind fine-tuning is **transfer learning**: knowledge learned on one task transfers to another.

A pre-trained language model has learned:
- Word meanings and relationships (embeddings)
- Syntactic patterns (attention heads)
- Contextual disambiguation ("bank" near "river" vs "bank" near "money")
- Long-range dependencies

These capabilities are **task-agnostic** — they help with *any* NLP task. Fine-tuning adds a thin task-specific layer on top and adjusts the entire model slightly to optimize for your objective.

**The math:** Given a pre-trained model with parameters θ_pretrained, fine-tuning minimizes:

```
L(θ) = Σ CrossEntropy(f(x_i; θ), y_i)
```

where f is the model with a classification head, x_i are input texts, y_i are labels, and θ is initialized from θ_pretrained. The crucial difference from training from scratch: θ starts in a *good region* of parameter space, so we need far fewer examples and epochs.

### Why Not Just Train From Scratch?

| Approach | Data needed | Compute | Quality |
|----------|-------------|---------|---------|
| Train from scratch | ~100K+ examples | GPU-days | Moderate |
| Fine-tune pre-trained | ~1K examples | GPU-minutes | High |
| Zero-shot prompting | 0 examples | Per-request | Variable |

Fine-tuning hits the sweet spot: you get high quality with modest data and compute.

### The Architecture: BERT + Classification Head

```
Input: "This movie was absolutely brilliant"
  ↓
Tokenizer → [CLS] This movie was absolutely brilliant [SEP]
  ↓
BERT Encoder (12 transformer layers)
  ↓
[CLS] token embedding (768-dim vector)
  ↓
Linear layer (768 → num_classes)
  ↓
Softmax → [0.02, 0.98]  (negative, positive)
```

The **[CLS] token** is special — BERT is pre-trained to make this token's representation a summary of the entire sequence. We attach our classification head here.

### Learning Rate: The Most Critical Hyperparameter

Fine-tuning uses a **much smaller learning rate** than training from scratch (typically 2e-5 to 5e-5 vs 1e-3). Why?

- Pre-trained weights are already in a good region of loss landscape
- Large updates would destroy the useful representations (called **catastrophic forgetting**)
- We want to *nudge* the model toward our task, not overwrite its knowledge

### Tokenization: Subword Units

Modern models use **subword tokenization** (BPE or WordPiece). The word "unhappiness" becomes ["un", "##happiness"] or ["un", "happi", "ness"]. This means:
- No out-of-vocabulary words (any word can be decomposed)
- The model learns morphological patterns (prefixes, suffixes)
- Vocabulary stays manageable (~30K tokens vs millions of words)

## Step-by-Step Breakdown

### Step 1: Prepare the Dataset
Load labeled text data, split into train/validation/test sets. Tokenize using the pre-trained model's tokenizer (this must match — you can't use BERT's tokenizer with GPT's model). Handle padding and truncation to fixed sequence length.

### Step 2: Load Pre-Trained Model
Initialize from pre-trained weights. Add a classification head (linear layer) on top. The classification head is randomly initialized — this is the part that needs the most learning.

### Step 3: Configure Training
Set learning rate (2e-5 is a good starting point), batch size, number of epochs (2-4 is usually enough — more risks overfitting). Use a learning rate scheduler (linear warmup + decay) for stability.

### Step 4: Training Loop
For each epoch: forward pass through model, compute cross-entropy loss, backpropagate, update weights. Track training loss and validation accuracy to detect overfitting.

### Step 5: Evaluation
Compute accuracy, precision, recall, F1 on held-out test set. Look at confusion matrix to understand error patterns. Examine misclassified examples — they often reveal label noise or ambiguous cases.

### Step 6: Inference
Use the fine-tuned model to classify new, unseen text. Compare against baseline (e.g., keyword matching, zero-shot) to quantify improvement.

## Learning Objectives

- Understand transfer learning and why pre-trained models are so effective
- Implement a complete fine-tuning pipeline: data prep → training → evaluation → inference
- Learn the critical hyperparameters and their effects (learning rate, epochs, batch size)
- Build intuition for overfitting detection in fine-tuning
- Understand tokenization and sequence handling for transformer models

## Going Deeper

- **Layer freezing:** Freeze early BERT layers (which capture syntax) and only fine-tune later layers (which capture semantics) — useful with very small datasets
- **Learning rate scheduling:** Discriminative fine-tuning uses different learning rates for different layers (lower for early layers, higher for the classification head)
- **Data augmentation:** Back-translation, synonym replacement, or mixup to expand small datasets
- **Multi-task fine-tuning:** Train on multiple related tasks simultaneously to improve generalization
- **Distillation:** After fine-tuning a large model, distill it into a smaller model for production deployment
- **LoRA/QLoRA:** Parameter-efficient fine-tuning that only updates a small number of adapter parameters — dramatically reduces compute and memory requirements
- **Evaluation beyond accuracy:** In production, you care about calibration (are confidence scores meaningful?), latency, and handling of distribution shift
