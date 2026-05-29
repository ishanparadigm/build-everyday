# Day 47: Embeddings and Vector Search

## Overview

Build a complete vector search engine from scratch — from generating text embeddings to performing efficient similarity search using multiple distance metrics and indexing strategies.

**Why this matters:** Vector search is the backbone of modern AI systems. Every RAG pipeline, recommendation engine, semantic search tool, and retrieval-augmented agent depends on the ability to convert text (or images, audio, etc.) into dense vectors and find the most similar ones fast. Understanding how embeddings work and how to search them efficiently is essential for building production AI systems. Yesterday's RAG pipeline (Day 46) used embeddings as a black box — today we open that box.

## Core Concepts

### What Are Embeddings?

An embedding is a mapping from a discrete, high-dimensional space (like the space of all possible sentences) to a continuous, low-dimensional vector space (like R^384). The key property: **semantically similar inputs land close together in vector space**.

Formally, an embedding function `f: X → R^d` maps inputs to d-dimensional real vectors such that:
- `sim(x₁, x₂) ≈ distance(f(x₁), f(x₂))`

Where `sim` is some notion of semantic similarity and `distance` is a geometric distance metric.

**How are embeddings trained?** Models like Word2Vec, GloVe, and sentence transformers learn embeddings by training on large text corpora. The key insight: words/sentences that appear in similar contexts should have similar representations. Contrastive learning pushes similar pairs closer and dissimilar pairs apart in the embedding space.

### Distance Metrics

Given two vectors **a** and **b** in R^d:

**Cosine Similarity:**
```
cos(a, b) = (a · b) / (||a|| × ||b||)
```
- Range: [-1, 1] (1 = identical direction, 0 = orthogonal, -1 = opposite)
- Ignores magnitude, focuses on direction — good when vector lengths vary (e.g., different document lengths)
- Most common metric for text embeddings

**Euclidean Distance (L2):**
```
d(a, b) = √(Σ(aᵢ - bᵢ)²)
```
- Range: [0, ∞)
- Sensitive to magnitude — a long document vector might be far from a short one even if semantically similar
- Better when magnitude carries meaning (e.g., TF-IDF vectors)

**Dot Product:**
```
a · b = Σ(aᵢ × bᵢ)
```
- Range: (-∞, ∞)
- Equivalent to cosine similarity when vectors are normalized
- Computationally cheapest — no square root or normalization needed
- Used by many production systems (FAISS, Pinecone) after normalizing vectors at index time

**When to use which?** If your embeddings are normalized (unit length), all three metrics give equivalent rankings. Most embedding models (like `all-MiniLM-L6-v2`) output normalized vectors, so cosine similarity is standard. For raw TF-IDF or count-based vectors, Euclidean may be more appropriate.

### The Curse of Dimensionality

In high dimensions, distances between points converge — the ratio of the nearest to the farthest neighbor approaches 1. This means brute-force search degrades as dimensions increase.

**Practical impact:** For 100K documents with 384-dimensional embeddings, brute force requires 100K × 384 floating-point operations per query. That's ~38M FLOPs — fast enough. But at 10M documents, you need indexing.

### Approximate Nearest Neighbor (ANN) Search

Exact k-NN search is O(n×d) per query. ANN methods trade a small accuracy loss for massive speedup:

**Inverted File Index (IVF):**
1. Cluster all vectors into `nlist` clusters using k-means
2. At query time, only search the `nprobe` closest clusters
3. Speedup: ~nlist/nprobe × (with some overhead for cluster assignment)
4. Tradeoff: higher `nprobe` = more accurate but slower

**Locality-Sensitive Hashing (LSH):**
1. Generate random hyperplanes in the embedding space
2. Each hyperplane divides space in two — assign 0 or 1 based on which side a vector falls
3. Concatenate bits to form a hash — similar vectors get similar hashes
4. At query time, only compare vectors with matching hash buckets
5. Tradeoff: more hash bits = fewer false positives but more false negatives

**Product Quantization (PQ):**
1. Split each d-dimensional vector into m sub-vectors
2. Cluster each sub-space independently into k centroids
3. Replace each sub-vector with its centroid ID (lossy compression)
4. Distance computation uses precomputed centroid-to-centroid distances
5. Reduces memory from d×4 bytes to m×log₂(k) bits per vector

### TF-IDF as a Baseline Embedding

Before neural embeddings, TF-IDF (Term Frequency × Inverse Document Frequency) was the standard text representation:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
TF(t, d) = count(t in d) / |d|
IDF(t) = log(N / df(t))
```

Where N = total documents, df(t) = documents containing term t. TF-IDF vectors are sparse and high-dimensional (one dimension per vocabulary word) — a useful contrast to dense neural embeddings.

## Step-by-Step Breakdown

### Step 1: Build a TF-IDF Vectorizer
Create a from-scratch TF-IDF implementation. This grounds the concept of "turning text into vectors" before we use neural models. We tokenize, compute term frequencies, compute IDF weights, and multiply.

### Step 2: Implement Distance Metrics
Code cosine similarity, Euclidean distance, and dot product from scratch using only NumPy. Understanding these geometrically is essential — they're not interchangeable.

### Step 3: Build a Brute-Force Vector Index
Store vectors in a matrix and implement exact k-NN search. This is the baseline everything else improves upon.

### Step 4: Generate Dense Embeddings
Use a pre-trained sentence transformer to generate dense embeddings. Compare the quality of TF-IDF vs. dense embeddings on semantic similarity tasks.

### Step 5: Implement LSH Indexing
Build a Locality-Sensitive Hashing index from scratch. Random hyperplanes partition the space, and hash collisions identify candidate neighbors. This demonstrates the core ANN concept.

### Step 6: Build a Complete Search Engine
Combine everything into a search engine that indexes documents and answers natural language queries. Compare brute-force vs. LSH on speed and recall.

## Learning Objectives

- Understand what embeddings represent geometrically and how they encode semantic meaning
- Implement cosine similarity, Euclidean distance, and dot product from scratch
- Build TF-IDF vectorization to understand sparse text representations
- Use sentence transformers to generate dense embeddings
- Implement LSH (Locality-Sensitive Hashing) for approximate nearest neighbor search
- Measure and compare search quality (recall@k) vs. speed tradeoffs
- Connect these foundations to production vector databases (FAISS, Pinecone, Weaviate)

## Going Deeper

- **HNSW (Hierarchical Navigable Small World):** The dominant ANN algorithm in production (used by FAISS, Qdrant, Weaviate). Builds a multi-layer graph where each node connects to nearby neighbors. Navigation starts at the top layer (sparse, long-range connections) and descends to the bottom (dense, short-range). O(log n) query time.
- **Matryoshka Representation Learning:** Train embeddings where the first k dimensions are themselves a valid (lower-quality) embedding. Allows trading dimension for speed at query time without retraining.
- **Quantization in production:** FAISS uses Product Quantization + IVF to search billions of vectors on a single machine. Memory drops from 1.5 TB (4B × 384 × 4 bytes) to ~15 GB with PQ.
- **Hybrid search:** Combine dense vector search with sparse keyword search (BM25) for better recall. Many production systems (Vespa, Elasticsearch) support this natively.
- **Multi-vector representations:** ColBERT represents each document as multiple vectors (one per token), enabling more fine-grained matching at the cost of storage.
- **Reranking:** Use a cross-encoder (which sees query + document together) to re-score the top-k results from a bi-encoder retrieval. More accurate but too slow for first-stage retrieval.
