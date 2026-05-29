"""
Day 47: Embeddings and Vector Search — Complete Implementation

Build a vector search engine from scratch, covering:
1. TF-IDF vectorization (sparse embeddings)
2. Distance metrics (cosine, euclidean, dot product)
3. Brute-force k-NN search
4. Dense embeddings via sentence transformers
5. LSH (Locality-Sensitive Hashing) for approximate nearest neighbor
6. Full search engine with recall@k evaluation

Dependencies: numpy, sentence-transformers (optional — falls back to TF-IDF if unavailable)
"""

import math
import time
import re
from collections import Counter
from typing import Optional

import numpy as np


# =============================================================================
# Step 1: TF-IDF Vectorizer from Scratch
# =============================================================================

class TFIDFVectorizer:
    """
    Converts text documents into TF-IDF vectors.

    TF-IDF captures word importance: a word is important if it appears
    frequently in a document (high TF) but rarely across documents (high IDF).
    This naturally downweights common words like "the" and "is".

    The resulting vectors are sparse (mostly zeros) and high-dimensional
    (one dimension per unique word in the vocabulary).
    """

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}  # word -> index mapping
        self.idf: np.ndarray = np.array([])   # inverse document frequency per term
        self.vocab_size: int = 0

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple whitespace + punctuation tokenizer.
        In production you'd use a proper tokenizer (spaCy, NLTK, etc.)
        that handles stemming, lemmatization, and stop words.
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return tokens

    def fit(self, documents: list[str]) -> 'TFIDFVectorizer':
        """
        Learn vocabulary and IDF weights from a corpus.

        IDF(t) = log(N / df(t)) + 1

        We add 1 to avoid zero IDF for terms that appear in all documents.
        This is the "smooth" IDF variant used by scikit-learn.
        """
        n_docs = len(documents)

        # Build vocabulary: assign each unique word an integer index
        word_set: set[str] = set()
        doc_freq: Counter = Counter()  # How many documents contain each word

        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            word_set.update(unique_tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        # Sort vocabulary for deterministic ordering
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(word_set))}
        self.vocab_size = len(self.vocabulary)

        # Compute IDF for each term
        # IDF = log(N / df) + 1 (smooth variant — prevents zero IDF)
        self.idf = np.zeros(self.vocab_size)
        for word, idx in self.vocabulary.items():
            df = doc_freq.get(word, 0)
            self.idf[idx] = math.log(n_docs / (df + 1e-10)) + 1

        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        """
        Convert documents to TF-IDF vectors.

        TF(t, d) = count(t in d) / len(d)   [normalized term frequency]
        TF-IDF(t, d) = TF(t, d) × IDF(t)

        Returns an (n_documents, vocab_size) matrix.
        """
        n_docs = len(documents)
        # We use a dense matrix here for simplicity. In production with
        # large vocabularies (100K+ words), you'd use scipy.sparse.
        matrix = np.zeros((n_docs, self.vocab_size))

        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            if not tokens:
                continue

            # Count occurrences of each token
            token_counts = Counter(tokens)
            doc_len = len(tokens)

            for token, count in token_counts.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    # TF: normalized by document length
                    tf = count / doc_len
                    matrix[i, idx] = tf * self.idf[idx]

        return matrix

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        """Fit vocabulary and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


# =============================================================================
# Step 2: Distance Metrics from Scratch
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    cos(a, b) = (a · b) / (||a|| × ||b||)

    Measures the angle between vectors, ignoring magnitude.
    Range: [-1, 1] where 1 = identical direction.

    This is the most common metric for text embeddings because
    document length shouldn't affect similarity — a short sentence
    and a long paragraph about the same topic should be similar.
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Guard against zero vectors (e.g., empty documents)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean (L2) distance between two vectors.

    d(a, b) = √(Σ(aᵢ - bᵢ)²)

    Measures straight-line distance in vector space.
    Range: [0, ∞) where 0 = identical vectors.

    Sensitive to magnitude — two vectors pointing in the same direction
    but with different lengths will have nonzero distance. This makes
    it less ideal for text embeddings (where magnitude ≈ document length)
    but useful for normalized embeddings or geometric applications.
    """
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot product between two vectors.

    a · b = Σ(aᵢ × bᵢ) = ||a|| × ||b|| × cos(θ)

    When vectors are L2-normalized (unit length), dot product equals
    cosine similarity. Many production systems (FAISS, Pinecone)
    normalize at index time and use dot product for speed — it avoids
    the division in cosine similarity.
    """
    return float(np.dot(a, b))


# =============================================================================
# Step 3: Brute-Force Vector Index
# =============================================================================

class BruteForceIndex:
    """
    Exact k-NN search by computing distances to all indexed vectors.

    Time complexity: O(n × d) per query, where n = number of vectors, d = dimensions.
    Space complexity: O(n × d) for the index.

    This is the baseline — correct but doesn't scale. At 10K vectors with
    384 dimensions, a query takes ~1ms. At 10M vectors, ~1 second.
    Production systems need approximate methods for > 100K vectors.
    """

    def __init__(self, metric: str = "cosine") -> None:
        """
        Args:
            metric: "cosine", "euclidean", or "dot_product"
        """
        self.vectors: Optional[np.ndarray] = None
        self.metric = metric
        self.n_vectors = 0
        self.dim = 0

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index. Shape: (n, d).

        We store everything in a single matrix for cache-friendly
        access — iterating over rows of a contiguous array is much
        faster than iterating over a list of separate arrays.
        """
        if self.vectors is None:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.n_vectors = self.vectors.shape[0]
        self.dim = self.vectors.shape[1]

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Find the k most similar vectors to the query.

        Returns list of (index, score) tuples sorted by relevance.
        For cosine/dot_product: higher = more similar.
        For euclidean: lower = more similar.

        Implementation note: We compute ALL distances at once using
        matrix operations rather than looping — NumPy's BLAS backend
        makes this 10-100x faster than a Python loop.
        """
        if self.vectors is None or self.n_vectors == 0:
            return []

        if self.metric == "cosine":
            # Vectorized cosine similarity against all indexed vectors
            # cos(q, v) = (q · v) / (||q|| × ||v||)
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return [(i, 0.0) for i in range(min(k, self.n_vectors))]

            # Compute dot products with all vectors at once: shape (n,)
            dots = self.vectors @ query
            # Compute norms of all vectors: shape (n,)
            norms = np.linalg.norm(self.vectors, axis=1)
            # Avoid division by zero
            norms = np.maximum(norms, 1e-10)
            scores = dots / (norms * query_norm)

            # Get top-k indices (highest similarity first)
            # argpartition is O(n) vs argsort's O(n log n)
            if k < self.n_vectors:
                top_k_idx = np.argpartition(scores, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            else:
                top_k_idx = np.argsort(scores)[::-1][:k]

            return [(int(idx), float(scores[idx])) for idx in top_k_idx]

        elif self.metric == "euclidean":
            # Vectorized L2 distance: ||q - v||₂ for all v
            # Expand: ||q - v||² = ||q||² + ||v||² - 2(q · v)
            # This avoids materializing (n, d) difference matrix
            q_sq = np.dot(query, query)
            v_sq = np.sum(self.vectors ** 2, axis=1)
            dots = self.vectors @ query
            distances = np.sqrt(np.maximum(q_sq + v_sq - 2 * dots, 0))

            # Get top-k indices (smallest distance first)
            if k < self.n_vectors:
                top_k_idx = np.argpartition(distances, k)[:k]
                top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]
            else:
                top_k_idx = np.argsort(distances)[:k]

            return [(int(idx), float(distances[idx])) for idx in top_k_idx]

        elif self.metric == "dot_product":
            dots = self.vectors @ query
            if k < self.n_vectors:
                top_k_idx = np.argpartition(dots, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(dots[top_k_idx])[::-1]]
            else:
                top_k_idx = np.argsort(dots)[::-1][:k]

            return [(int(idx), float(dots[idx])) for idx in top_k_idx]

        else:
            raise ValueError(f"Unknown metric: {self.metric}")


# =============================================================================
# Step 4: Dense Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    """
    Generates dense vector embeddings from text.

    Uses sentence-transformers if available, otherwise falls back
    to normalized TF-IDF vectors. The key difference:

    - TF-IDF: Sparse, high-dimensional (~10K dims), bag-of-words (no word order),
      can't capture synonyms or paraphrases
    - Dense (transformer): Dense, low-dimensional (384 dims), captures semantic
      meaning, handles synonyms, paraphrases, and context

    Example: "The cat sat on the mat" and "A feline rested on the rug"
    - TF-IDF: Low similarity (different words)
    - Dense: High similarity (same meaning)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.use_transformer = False
        self._tfidf: Optional[TFIDFVectorizer] = None
        self._tfidf_fitted = False

        # Try to load sentence-transformers; fall back gracefully
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.use_transformer = True
            print(f"[EmbeddingGenerator] Using sentence-transformer: {model_name}")
        except ImportError:
            print("[EmbeddingGenerator] sentence-transformers not installed.")
            print("  Falling back to normalized TF-IDF embeddings.")
            print("  Install with: pip install sentence-transformers")
            self._tfidf = TFIDFVectorizer()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of texts to embedding vectors.

        Returns: np.ndarray of shape (len(texts), embedding_dim)

        For sentence-transformers: embedding_dim = 384 (for MiniLM)
        For TF-IDF fallback: embedding_dim = vocabulary size

        All vectors are L2-normalized so cosine similarity = dot product.
        """
        if self.use_transformer:
            # sentence-transformers returns normalized embeddings by default
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        else:
            # TF-IDF fallback: fit on first call, transform on subsequent
            if not self._tfidf_fitted:
                vectors = self._tfidf.fit_transform(texts)
                self._tfidf_fitted = True
            else:
                vectors = self._tfidf.transform(texts)

            # L2 normalize so cosine similarity = dot product
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # avoid division by zero
            return vectors / norms

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        if self.use_transformer:
            return self.model.get_sentence_embedding_dimension()
        else:
            return self._tfidf.vocab_size if self._tfidf else 0


# =============================================================================
# Step 5: LSH (Locality-Sensitive Hashing) Index
# =============================================================================

class LSHIndex:
    """
    Approximate nearest neighbor search using Locality-Sensitive Hashing.

    Core idea: Random hyperplanes partition the vector space. Each hyperplane
    divides space into two halves — a vector is on the "positive" side (bit=1)
    or "negative" side (bit=0). A hash = concatenation of bits from multiple
    hyperplanes. Similar vectors are likely to get the same hash.

    Why it works (math):
    For cosine similarity, the probability that two vectors a, b get the
    same bit from a random hyperplane is:
        P(same bit) = 1 - θ(a,b)/π
    where θ is the angle between them. Small angle → high probability.

    With n_bits random hyperplanes, P(same hash) = (1 - θ/π)^n_bits
    - Similar vectors (small θ): high collision probability → same bucket
    - Dissimilar vectors (large θ): low collision probability → different buckets

    Multiple hash tables increase recall at the cost of memory and query time.

    Tradeoffs:
    - More bits → fewer false positives, more false negatives (buckets too specific)
    - More tables → higher recall, more memory
    - Sweet spot depends on dataset size and accuracy requirements
    """

    def __init__(
        self,
        dim: int,
        n_bits: int = 8,
        n_tables: int = 4,
        seed: int = 42
    ) -> None:
        """
        Args:
            dim: Dimensionality of input vectors
            n_bits: Number of hash bits per table (controls bucket granularity)
            n_tables: Number of independent hash tables (controls recall)
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.n_bits = n_bits
        self.n_tables = n_tables

        rng = np.random.RandomState(seed)

        # Generate random hyperplanes for each hash table
        # Each hyperplane is a d-dimensional vector; the sign of (hyperplane · input)
        # determines the hash bit
        # Shape: (n_tables, n_bits, dim)
        self.hyperplanes = rng.randn(n_tables, n_bits, dim)

        # Hash tables: table_idx -> {hash_key -> list of vector indices}
        self.tables: list[dict[str, list[int]]] = [
            {} for _ in range(n_tables)
        ]

        # Store all vectors for distance computation after candidate retrieval
        self.vectors: Optional[np.ndarray] = None
        self.n_vectors = 0

    def _hash(self, vector: np.ndarray, table_idx: int) -> str:
        """
        Compute the LSH hash for a vector in a specific table.

        For each hyperplane h, compute sign(h · v):
        - Positive → bit = 1
        - Negative → bit = 0

        Concatenate all bits into a binary string hash key.
        """
        # Dot product with all hyperplanes at once: shape (n_bits,)
        projections = self.hyperplanes[table_idx] @ vector
        # Convert to binary string: "10110..."
        bits = ''.join(['1' if p >= 0 else '0' for p in projections])
        return bits

    def add(self, vectors: np.ndarray) -> None:
        """
        Index a batch of vectors.

        Each vector is hashed into each table and stored in the
        corresponding bucket. A vector appears in n_tables buckets total.
        """
        start_idx = self.n_vectors

        if self.vectors is None:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        # Hash each vector into each table
        for i in range(vectors.shape[0]):
            global_idx = start_idx + i
            for t in range(self.n_tables):
                hash_key = self._hash(vectors[i], t)
                if hash_key not in self.tables[t]:
                    self.tables[t][hash_key] = []
                self.tables[t][hash_key].append(global_idx)

        self.n_vectors = self.vectors.shape[0]

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Find approximate k nearest neighbors using LSH.

        Algorithm:
        1. Hash the query in each table
        2. Collect all candidate indices from matching buckets
        3. Compute exact cosine similarity only for candidates
        4. Return top-k by similarity

        The speedup comes from step 3: we only compute distances for
        candidates (typically << n_vectors) instead of all vectors.

        Failure mode: If the query's true nearest neighbor happens to
        land in a different bucket in ALL tables, we miss it. This is
        why we use multiple tables — it reduces the miss probability
        exponentially.
        """
        if self.vectors is None:
            return []

        # Step 1-2: Collect candidates from all tables
        candidate_set: set[int] = set()
        for t in range(self.n_tables):
            hash_key = self._hash(query, t)
            if hash_key in self.tables[t]:
                candidate_set.update(self.tables[t][hash_key])

        if not candidate_set:
            # No candidates found — fall back to random sample
            # This happens when n_bits is too high (buckets too specific)
            return []

        # Step 3: Compute exact cosine similarity for candidates only
        candidates = list(candidate_set)
        candidate_vectors = self.vectors[candidates]

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return [(c, 0.0) for c in candidates[:k]]

        dots = candidate_vectors @ query
        norms = np.linalg.norm(candidate_vectors, axis=1)
        norms = np.maximum(norms, 1e-10)
        similarities = dots / (norms * query_norm)

        # Step 4: Sort and return top-k
        sorted_indices = np.argsort(similarities)[::-1][:k]

        return [
            (candidates[int(idx)], float(similarities[idx]))
            for idx in sorted_indices
        ]

    def stats(self) -> dict:
        """Return statistics about the index for debugging."""
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
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "min_bucket_size": min(bucket_sizes) if bucket_sizes else 0,
        }


# =============================================================================
# Step 6: Complete Search Engine
# =============================================================================

class VectorSearchEngine:
    """
    A complete vector search engine that:
    1. Accepts raw text documents
    2. Generates embeddings (dense or sparse)
    3. Indexes them for fast retrieval
    4. Answers natural language queries

    Supports both brute-force (exact) and LSH (approximate) search.
    """

    def __init__(
        self,
        use_lsh: bool = False,
        n_bits: int = 8,
        n_tables: int = 4,
        metric: str = "cosine"
    ) -> None:
        self.embedder = EmbeddingGenerator()
        self.documents: list[str] = []
        self.use_lsh = use_lsh
        self.metric = metric
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.index = None

    def index_documents(self, documents: list[str]) -> None:
        """
        Index a list of documents for search.

        Steps:
        1. Store raw documents (for returning results)
        2. Generate embeddings
        3. Build search index
        """
        self.documents = documents
        print(f"  Generating embeddings for {len(documents)} documents...")

        embeddings = self.embedder.embed(documents)
        dim = embeddings.shape[1]

        print(f"  Embedding dimension: {dim}")

        if self.use_lsh:
            self.index = LSHIndex(dim=dim, n_bits=self.n_bits, n_tables=self.n_tables)
        else:
            self.index = BruteForceIndex(metric=self.metric)

        self.index.add(embeddings)
        print(f"  Indexed {len(documents)} documents.")

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Search for documents similar to the query.

        Returns list of (document_text, similarity_score) tuples.
        """
        # Embed the query using the same model
        query_embedding = self.embedder.embed([query])[0]

        # Search the index
        results = self.index.search(query_embedding, k=k)

        # Map indices back to documents
        return [
            (self.documents[idx], score)
            for idx, score in results
            if idx < len(self.documents)
        ]


# =============================================================================
# Evaluation: Recall@K
# =============================================================================

def compute_recall_at_k(
    exact_results: list[tuple[int, float]],
    approx_results: list[tuple[int, float]],
    k: int
) -> float:
    """
    Measure how many of the true top-k results the approximate method found.

    recall@k = |exact_top_k ∩ approx_top_k| / k

    This is THE metric for evaluating ANN methods. A recall@10 of 0.9 means
    the approximate method found 9 out of 10 true nearest neighbors.

    Production systems typically target recall@10 ≥ 0.95.
    """
    exact_ids = set(idx for idx, _ in exact_results[:k])
    approx_ids = set(idx for idx, _ in approx_results[:k])

    if not exact_ids:
        return 1.0  # Both empty = perfect recall

    return len(exact_ids & approx_ids) / len(exact_ids)


# =============================================================================
# Main: Demonstrate Everything
# =============================================================================

if __name__ == "__main__":
    # --- Sample corpus: mix of topics for testing semantic search ---
    documents = [
        # Machine learning
        "Neural networks learn by adjusting weights through backpropagation",
        "Gradient descent minimizes the loss function iteratively",
        "Convolutional neural networks excel at image recognition tasks",
        "Random forests combine multiple decision trees for better predictions",
        "Support vector machines find optimal hyperplanes for classification",
        # Blockchain / crypto
        "Bitcoin uses proof of work consensus to validate transactions",
        "Ethereum smart contracts enable decentralized applications",
        "Merkle trees provide efficient verification of blockchain data integrity",
        "Zero knowledge proofs allow verification without revealing information",
        "DeFi protocols enable lending and borrowing without intermediaries",
        # Robotics
        "PID controllers maintain stability in robotic motion systems",
        "SLAM algorithms help robots build maps while navigating unknown environments",
        "Kalman filters estimate state from noisy sensor measurements",
        "Inverse kinematics calculates joint angles for desired end effector position",
        "Swarm robotics uses simple local rules to achieve complex collective behavior",
        # General / mixed
        "Python is widely used for data science and machine learning",
        "Cloud computing provides scalable infrastructure for AI workloads",
        "Edge computing brings AI inference closer to IoT sensor devices",
        "Reinforcement learning trains agents through reward signals",
        "Transfer learning leverages pre-trained models for new tasks with less data",
    ]

    print("=" * 70)
    print("EMBEDDINGS AND VECTOR SEARCH — COMPLETE DEMONSTRATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Part 1: TF-IDF Vectorization
    # -------------------------------------------------------------------------
    print("\n--- Part 1: TF-IDF Vectorization ---")

    tfidf = TFIDFVectorizer()
    tfidf_vectors = tfidf.fit_transform(documents)

    print(f"Vocabulary size: {tfidf.vocab_size}")
    print(f"Document matrix shape: {tfidf_vectors.shape}")
    print(f"Sparsity: {(tfidf_vectors == 0).sum() / tfidf_vectors.size:.1%}")

    # Show a sample: TF-IDF weights for the first document
    doc0_nonzero = [(word, tfidf_vectors[0, idx])
                     for word, idx in tfidf.vocabulary.items()
                     if tfidf_vectors[0, idx] > 0]
    doc0_nonzero.sort(key=lambda x: -x[1])
    print(f"\nTop TF-IDF terms for: '{documents[0][:50]}...'")
    for word, weight in doc0_nonzero[:5]:
        print(f"  {word}: {weight:.4f}")

    # -------------------------------------------------------------------------
    # Part 2: Distance Metrics
    # -------------------------------------------------------------------------
    print("\n--- Part 2: Distance Metrics ---")

    # Compare first two documents (both about ML) vs first and sixth (ML vs crypto)
    v_ml1 = tfidf_vectors[0]   # Neural networks / backpropagation
    v_ml2 = tfidf_vectors[1]   # Gradient descent / loss function
    v_crypto = tfidf_vectors[5]  # Bitcoin / proof of work

    print(f"ML doc 1 vs ML doc 2 (related):")
    print(f"  Cosine similarity:    {cosine_similarity(v_ml1, v_ml2):.4f}")
    print(f"  Euclidean distance:   {euclidean_distance(v_ml1, v_ml2):.4f}")
    print(f"  Dot product:          {dot_product(v_ml1, v_ml2):.4f}")

    print(f"ML doc 1 vs Crypto doc (unrelated):")
    print(f"  Cosine similarity:    {cosine_similarity(v_ml1, v_crypto):.4f}")
    print(f"  Euclidean distance:   {euclidean_distance(v_ml1, v_crypto):.4f}")
    print(f"  Dot product:          {dot_product(v_ml1, v_crypto):.4f}")

    # -------------------------------------------------------------------------
    # Part 3: Brute-Force Search
    # -------------------------------------------------------------------------
    print("\n--- Part 3: Brute-Force Search ---")

    bf_index = BruteForceIndex(metric="cosine")
    bf_index.add(tfidf_vectors)

    query_text = "How do neural networks learn?"
    query_vec = tfidf.transform([query_text])[0]

    print(f"Query: '{query_text}'")
    print("Top 5 results (brute force, TF-IDF):")
    bf_results = bf_index.search(query_vec, k=5)
    for rank, (idx, score) in enumerate(bf_results, 1):
        print(f"  {rank}. [{score:.4f}] {documents[idx][:70]}...")

    # -------------------------------------------------------------------------
    # Part 4: Dense Embeddings
    # -------------------------------------------------------------------------
    print("\n--- Part 4: Dense Embeddings ---")

    embedder = EmbeddingGenerator()
    dense_vectors = embedder.embed(documents)

    print(f"Dense embedding shape: {dense_vectors.shape}")
    print(f"Vector norm (should be ~1.0): {np.linalg.norm(dense_vectors[0]):.4f}")

    # Semantic similarity comparison
    print("\nSemantic similarity test (dense embeddings):")
    # These two are semantically similar but share few words
    s1 = "The cat sat on the mat"
    s2 = "A feline rested on the rug"
    s3 = "Bitcoin mining consumes electricity"

    v1, v2, v3 = embedder.embed([s1, s2, s3])
    print(f"  '{s1}' vs '{s2}': {cosine_similarity(v1, v2):.4f}")
    print(f"  '{s1}' vs '{s3}': {cosine_similarity(v1, v3):.4f}")
    print(f"  (Dense embeddings capture meaning, not just word overlap)")

    # -------------------------------------------------------------------------
    # Part 5: LSH Index
    # -------------------------------------------------------------------------
    print("\n--- Part 5: LSH Approximate Search ---")

    dim = dense_vectors.shape[1]
    lsh = LSHIndex(dim=dim, n_bits=6, n_tables=8)
    lsh.add(dense_vectors)

    stats = lsh.stats()
    print(f"LSH index stats:")
    print(f"  Vectors: {stats['n_vectors']}")
    print(f"  Tables: {stats['n_tables']}, Bits per hash: {stats['n_bits']}")
    print(f"  Total buckets: {stats['total_buckets']}")
    print(f"  Avg bucket size: {stats['avg_bucket_size']:.1f}")

    # -------------------------------------------------------------------------
    # Part 6: Full Search Engine Comparison
    # -------------------------------------------------------------------------
    print("\n--- Part 6: Search Engine Comparison ---")

    queries = [
        "How does blockchain verify data?",
        "Robot navigation in unknown environments",
        "Training deep learning models",
        "Decentralized finance lending",
        "Sensor data processing for robots",
    ]

    # Build both engines
    print("\nBuilding brute-force search engine...")
    engine_bf = VectorSearchEngine(use_lsh=False)
    engine_bf.index_documents(documents)

    print("\nBuilding LSH search engine...")
    engine_lsh = VectorSearchEngine(use_lsh=True, n_bits=6, n_tables=8)
    engine_lsh.index_documents(documents)

    print("\n" + "=" * 70)
    print("SEARCH RESULTS COMPARISON")
    print("=" * 70)

    total_recall = 0.0
    k = 5

    for query in queries:
        print(f"\nQuery: '{query}'")

        # Brute-force results (ground truth)
        bf_results = engine_bf.search(query, k=k)

        # Time the brute-force search
        t0 = time.perf_counter()
        for _ in range(100):
            engine_bf.search(query, k=k)
        bf_time = (time.perf_counter() - t0) / 100

        # LSH results
        lsh_results = engine_lsh.search(query, k=k)

        # Time the LSH search
        t0 = time.perf_counter()
        for _ in range(100):
            engine_lsh.search(query, k=k)
        lsh_time = (time.perf_counter() - t0) / 100

        # Compute recall
        bf_ids = [(i, s) for i, (doc, s) in enumerate(bf_results)]
        lsh_ids = []
        for doc, s in lsh_results:
            try:
                idx = documents.index(doc)
                lsh_ids.append((idx, s))
            except ValueError:
                pass

        recall = compute_recall_at_k(
            [(documents.index(doc), s) for doc, s in bf_results],
            [(documents.index(doc), s) for doc, s in lsh_results],
            k=k
        )
        total_recall += recall

        print(f"  Brute force ({bf_time*1000:.2f}ms):")
        for rank, (doc, score) in enumerate(bf_results[:3], 1):
            print(f"    {rank}. [{score:.4f}] {doc[:60]}...")

        print(f"  LSH ({lsh_time*1000:.2f}ms):")
        for rank, (doc, score) in enumerate(lsh_results[:3], 1):
            print(f"    {rank}. [{score:.4f}] {doc[:60]}...")

        print(f"  Recall@{k}: {recall:.2f}")

    avg_recall = total_recall / len(queries)
    print(f"\n{'=' * 70}")
    print(f"Average Recall@{k}: {avg_recall:.2f}")
    print(f"{'=' * 70}")

    # -------------------------------------------------------------------------
    # Bonus: Show how distance metrics relate for normalized vectors
    # -------------------------------------------------------------------------
    print("\n--- Bonus: Metric Equivalence for Normalized Vectors ---")
    v_a = dense_vectors[0]  # Already normalized by sentence-transformers
    v_b = dense_vectors[1]

    cos_sim = cosine_similarity(v_a, v_b)
    dot_prod = dot_product(v_a, v_b)
    euc_dist = euclidean_distance(v_a, v_b)

    # For unit vectors: euclidean² = 2(1 - cos_sim)
    euc_from_cos = math.sqrt(2 * (1 - cos_sim))

    print(f"Vector norms: ||a|| = {np.linalg.norm(v_a):.4f}, ||b|| = {np.linalg.norm(v_b):.4f}")
    print(f"Cosine similarity:  {cos_sim:.6f}")
    print(f"Dot product:        {dot_prod:.6f}  (= cosine sim for unit vectors)")
    print(f"Euclidean distance: {euc_dist:.6f}")
    print(f"Euclidean from cos: {euc_from_cos:.6f}  (√(2(1-cos)) — should match)")
    print(f"\nWhen vectors are normalized, all three metrics give equivalent rankings!")
