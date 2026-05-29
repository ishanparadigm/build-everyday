# Day 046: Build a RAG Pipeline

## Overview

Retrieval-Augmented Generation (RAG) is the pattern that makes LLMs useful with private or up-to-date data. Instead of stuffing everything into the prompt or fine-tuning on your corpus, you **retrieve** the most relevant chunks of text at query time and **augment** the LLM's prompt with them. The LLM then **generates** an answer grounded in real source material.

This matters in production because:
- LLMs have knowledge cutoffs and hallucinate about specifics
- Fine-tuning is expensive and slow to iterate
- RAG lets you update knowledge by updating the index — no retraining needed
- You get citations/provenance for free (you know which chunks contributed)

Today you'll build a complete RAG pipeline from scratch: document chunking, embedding generation, vector storage with similarity search, and answer generation with source attribution.

## Core Concepts

### 1. Document Chunking

Raw documents are too long to embed as a single vector (embeddings lose fidelity beyond ~512 tokens) and too long to fit many into a prompt. We split documents into overlapping chunks.

**Why overlapping?** A fact might straddle a chunk boundary. If chunk A ends with "The reactor operates at" and chunk B starts with "450 degrees Celsius", neither chunk alone captures the complete fact. Overlap (typically 10-20% of chunk size) ensures boundary facts appear in at least one chunk intact.

**Chunk size tradeoffs:**
- Too small (50 tokens): loses context, retrieves fragments
- Too large (2000 tokens): dilutes the embedding signal, wastes prompt space
- Sweet spot: 200-500 tokens for most use cases

### 2. Embeddings and Vector Similarity

An embedding maps text to a dense vector (e.g., 384 or 1536 dimensions) where **semantic similarity corresponds to geometric proximity**. Two passages about the same topic will have vectors that are close together, even if they share few words.

**Cosine similarity** is the standard metric:

```
cos(A, B) = (A . B) / (||A|| * ||B||)
```

This measures the angle between vectors, ignoring magnitude. A value of 1.0 means identical direction (same meaning), 0.0 means orthogonal (unrelated), -1.0 means opposite.

**Why cosine over Euclidean distance?** Embedding models don't guarantee consistent vector magnitudes. A longer passage might produce a larger-magnitude vector without being "more similar" to anything. Cosine normalizes this out.

We use `sentence-transformers/all-MiniLM-L6-v2` — a small (80MB) model that produces 384-dimensional embeddings. In production you'd use OpenAI's `text-embedding-3-small` or Cohere's embeddings, but the concepts are identical.

### 3. Vector Store (In-Memory FAISS-like Index)

Production RAG uses vector databases (Pinecone, Weaviate, ChromaDB, pgvector). Today we build a simple in-memory index to understand what they do under the hood:

1. Store all chunk embeddings in a matrix
2. On query: embed the query, compute cosine similarity against all chunks
3. Return the top-k most similar chunks

This is brute-force O(n) search. Production systems use approximate nearest neighbor (ANN) algorithms like HNSW or IVF to get sub-linear search time, but the interface is the same.

### 4. Prompt Construction and Generation

The retrieved chunks become context in a carefully structured prompt:

```
Given the following context, answer the question.
If the context doesn't contain enough information, say so.

Context:
[chunk 1]
[chunk 2]
[chunk 3]

Question: {user_query}
```

Key design decisions:
- **Number of chunks (k)**: More context = more information but more noise and cost. Typically k=3-5.
- **Ordering**: Most relevant first (the LLM pays more attention to the start)
- **Instruction to say "I don't know"**: Reduces hallucination when retrieved context is irrelevant

Since we're building this without API keys, we simulate the generation step with a template-based approach that demonstrates the mechanics.

## Step-by-Step Breakdown

### Step 1: Document Loading and Preprocessing
Load text documents, clean whitespace, normalize encoding. This is the "garbage in, garbage out" step — noisy input means noisy embeddings.

### Step 2: Chunking with Overlap
Split each document into fixed-size chunks with configurable overlap. Track which document each chunk came from (for citation).

### Step 3: Embedding Generation
Run each chunk through a sentence transformer model to get dense vectors. We batch this for efficiency.

### Step 4: Index Building
Store embeddings in a numpy matrix. Build metadata mapping from chunk index to source document and position.

### Step 5: Query Processing
Embed the user's question with the same model, compute cosine similarity against all chunks, retrieve top-k.

### Step 6: Answer Generation
Construct an augmented prompt with retrieved context and generate a response (simulated locally, but the prompt is production-ready).

### Step 7: Source Attribution
Return which documents and chunk positions contributed to the answer — this is what makes RAG trustworthy.

## Learning Objectives

- Understand why RAG exists and when to use it vs. fine-tuning vs. long-context
- Implement text chunking with overlap and understand the tradeoffs
- Generate and use text embeddings for semantic search
- Build a vector similarity search from scratch
- Construct effective RAG prompts with retrieved context
- Add source attribution for answer provenance

## Going Deeper

- **Hybrid search**: Combine dense vector search with sparse BM25 (keyword) search for better recall. Dense search misses exact keyword matches; BM25 misses semantic paraphrases. Together they cover both.
- **Re-ranking**: After retrieving top-20 candidates, use a cross-encoder model to re-rank them. Cross-encoders are more accurate than bi-encoders but too slow for the initial search.
- **Chunking strategies**: Instead of fixed-size, chunk by paragraph, by section header, or by semantic boundaries (detect topic shifts). Recursive character splitting (used by LangChain) tries multiple separators in order.
- **Metadata filtering**: In production, chunks have metadata (date, author, category). Filter before or after vector search to scope results.
- **Evaluation**: Measure retrieval quality with recall@k and MRR. Measure generation quality with faithfulness (does the answer stick to the context?) and relevance.
- **Connections to Day 033 (Prompt Chaining)**: RAG is essentially a two-step chain — retrieve then generate. You could extend this to multi-hop RAG where the first retrieval informs a refined query for a second retrieval.
