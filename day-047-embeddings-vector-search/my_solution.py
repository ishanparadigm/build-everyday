"""
Day 47: Embeddings and Vector Search — Your Implementation

Build a vector search engine from scratch:
1. TF-IDF vectorization
2. Distance metrics (cosine, euclidean, dot product)
3. Brute-force k-NN search
4. LSH approximate nearest neighbor search

Dependencies: numpy
"""

import math
import re
from collections import Counter
from typing import Optional

import numpy as np


# =============================================================================
# Step 1: TF-IDF Vectorizer
# =============================================================================

class TFIDFVectorizer:
    """
    Converts text documents into TF-IDF vectors.

    Hint: TF-IDF = Term Frequency × Inverse Document Frequency
    - TF(t, d) = count of term t in document d / total terms in d
    - IDF(t) = log(N / number of docs containing t) + 1
    - The +1 prevents zero IDF for terms in all documents
    """

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.vocab_size: int = 0

    def _tokenize(self, text: str) -> list[str]:
        """Split text into lowercase tokens. Use re.findall(r'[a-z0-9]+', text.lower())."""
        raise NotImplementedError("TODO: implement this")

    def fit(self, documents: list[str]) -> 'TFIDFVectorizer':
        """
        Learn vocabulary and IDF weights from a corpus.

        Steps:
        1. Tokenize each document
        2. Build vocabulary (unique words → integer indices)
        3. Count document frequency for each term
        4. Compute IDF = log(N / df) + 1

        Hint: Use a Counter for document frequencies. For each document,
        find the SET of unique tokens (not the list — avoid double-counting).
        """
        raise NotImplementedError("TODO: implement this")

    def transform(self, documents: list[str]) -> np.ndarray:
        """
        Convert documents to TF-IDF vectors.

        Returns: np.ndarray of shape (n_documents, vocab_size)

        Hint: For each document, compute TF for each token, then
        multiply by the corresponding IDF weight.
        """
        raise NotImplementedError("TODO: implement this")

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        """Fit vocabulary and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


# =============================================================================
# Step 2: Distance Metrics
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity: cos(a, b) = (a · b) / (||a|| × ||b||)

    Range: [-1, 1] where 1 = identical direction.
    Handle zero vectors by returning 0.0.

    Hint: Use np.dot for dot product, np.linalg.norm for magnitude.
    """
    raise NotImplementedError("TODO: implement this")


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance: d(a, b) = √(Σ(aᵢ - bᵢ)²)

    Range: [0, ∞) where 0 = identical vectors.

    Hint: Compute the difference vector, then its L2 norm.
    """
    raise NotImplementedError("TODO: implement this")


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot product: a · b = Σ(aᵢ × bᵢ)

    Equivalent to cosine similarity when vectors are L2-normalized.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Step 3: Brute-Force Vector Index
# =============================================================================

class BruteForceIndex:
    """
    Exact k-NN search by computing distances to all indexed vectors.

    Hint: Store vectors in a matrix. For search, compute the chosen
    distance metric against ALL vectors, then return the top-k.
    Use np.argpartition for O(n) top-k selection instead of O(n log n) sort.
    """

    def __init__(self, metric: str = "cosine") -> None:
        self.vectors: Optional[np.ndarray] = None
        self.metric = metric
        self.n_vectors = 0
        self.dim = 0

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index. Shape: (n, d).

        Hint: Use np.vstack to append to existing vectors.
        """
        raise NotImplementedError("TODO: implement this")

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Find the k most similar vectors to the query.

        Returns: list of (index, score) tuples sorted by relevance.
        - For cosine/dot_product: higher score = more similar
        - For euclidean: lower score = more similar

        Hint: Compute scores/distances vectorized (matrix @ vector),
        then use np.argpartition for efficient top-k.
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Step 4: LSH Index
# =============================================================================

class LSHIndex:
    """
    Approximate nearest neighbor search using Locality-Sensitive Hashing.

    Core idea:
    - Random hyperplanes partition vector space
    - sign(hyperplane · vector) gives one hash bit
    - Concatenate bits → hash key
    - Similar vectors likely share hash keys (same bucket)

    Multiple hash tables increase recall.

    Hint: Think of each hyperplane as a coin flip that's biased by the
    vector's direction. Similar vectors get the same "flip" more often.
    """

    def __init__(
        self,
        dim: int,
        n_bits: int = 8,
        n_tables: int = 4,
        seed: int = 42
    ) -> None:
        self.dim = dim
        self.n_bits = n_bits
        self.n_tables = n_tables

        rng = np.random.RandomState(seed)
        # Random hyperplanes: shape (n_tables, n_bits, dim)
        self.hyperplanes = rng.randn(n_tables, n_bits, dim)

        self.tables: list[dict[str, list[int]]] = [
            {} for _ in range(n_tables)
        ]
        self.vectors: Optional[np.ndarray] = None
        self.n_vectors = 0

    def _hash(self, vector: np.ndarray, table_idx: int) -> str:
        """
        Compute LSH hash for a vector in a specific table.

        Steps:
        1. Dot product with each hyperplane: self.hyperplanes[table_idx] @ vector
        2. Take sign of each projection (>= 0 → '1', < 0 → '0')
        3. Join into a binary string

        Returns: hash key string like "10110011"
        """
        raise NotImplementedError("TODO: implement this")

    def add(self, vectors: np.ndarray) -> None:
        """
        Index a batch of vectors.

        For each vector, hash it into each table and store its index
        in the corresponding bucket.

        Hint: Keep track of global indices if adding multiple batches.
        """
        raise NotImplementedError("TODO: implement this")

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Find approximate k nearest neighbors.

        Steps:
        1. Hash query in each table
        2. Collect all candidate indices from matching buckets (use a set!)
        3. Compute exact cosine similarity for candidates only
        4. Return top-k by similarity

        Hint: The speedup is in step 3 — you only check candidates,
        not all vectors. Return empty list if no candidates found.
        """
        raise NotImplementedError("TODO: implement this")

    def stats(self) -> dict:
        """Return index statistics (n_vectors, bucket sizes, etc.)."""
        bucket_sizes = []
        for t in range(self.n_tables):
            for bucket in self.tables[t].values():
                bucket_sizes.append(len(bucket))

        return {
            "n_vectors": self.n_vectors,
            "n_tables": self.n_tables,
            "n_bits": self.n_bits,
            "total_buckets": sum(len(t) for t in self.tables),
            "avg_bucket_size": np.mean(bucket_sizes) if bucket_sizes else 0,
        }


# =============================================================================
# Evaluation Helper
# =============================================================================

def compute_recall_at_k(
    exact_results: list[tuple[int, float]],
    approx_results: list[tuple[int, float]],
    k: int
) -> float:
    """
    Recall@k = |exact_top_k ∩ approx_top_k| / k

    Measures what fraction of true nearest neighbors the approximate
    method found. Target ≥ 0.95 for production systems.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    documents = [
        "Neural networks learn by adjusting weights through backpropagation",
        "Gradient descent minimizes the loss function iteratively",
        "Convolutional neural networks excel at image recognition tasks",
        "Bitcoin uses proof of work consensus to validate transactions",
        "Ethereum smart contracts enable decentralized applications",
        "PID controllers maintain stability in robotic motion systems",
        "SLAM algorithms help robots build maps while navigating",
        "Kalman filters estimate state from noisy sensor measurements",
    ]

    # Test TF-IDF
    print("--- Testing TF-IDF ---")
    tfidf = TFIDFVectorizer()
    vectors = tfidf.fit_transform(documents)
    print(f"Matrix shape: {vectors.shape}")
    print(f"Vocabulary size: {tfidf.vocab_size}")

    # Test distance metrics
    print("\n--- Testing Distance Metrics ---")
    v1, v2, v3 = vectors[0], vectors[1], vectors[3]
    print(f"ML vs ML (cosine):     {cosine_similarity(v1, v2):.4f}")
    print(f"ML vs Crypto (cosine): {cosine_similarity(v1, v3):.4f}")
    print(f"ML vs ML (euclidean):  {euclidean_distance(v1, v2):.4f}")
    print(f"Dot product:           {dot_product(v1, v2):.4f}")

    # Test brute-force search
    print("\n--- Testing Brute-Force Search ---")
    bf = BruteForceIndex(metric="cosine")
    bf.add(vectors)
    query = tfidf.transform(["deep learning neural network"])[0]
    results = bf.search(query, k=3)
    for idx, score in results:
        print(f"  [{score:.4f}] {documents[idx][:60]}")

    # Test LSH search
    print("\n--- Testing LSH Search ---")
    lsh = LSHIndex(dim=vectors.shape[1], n_bits=4, n_tables=6)
    lsh.add(vectors)
    lsh_results = lsh.search(query, k=3)
    for idx, score in lsh_results:
        print(f"  [{score:.4f}] {documents[idx][:60]}")

    # Test recall
    print("\n--- Recall Comparison ---")
    recall = compute_recall_at_k(results, lsh_results, k=3)
    print(f"Recall@3: {recall:.2f}")

    print(f"\nLSH stats: {lsh.stats()}")
    print("\nAll tests passed! Your vector search engine works.")
