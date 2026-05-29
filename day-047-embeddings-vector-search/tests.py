"""
Day 47: Embeddings and Vector Search — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import math
import numpy as np

from my_solution import (
    TFIDFVectorizer,
    cosine_similarity,
    euclidean_distance,
    dot_product,
    BruteForceIndex,
    LSHIndex,
    compute_recall_at_k,
)


class TestTFIDFVectorizer(unittest.TestCase):
    """Tests for TF-IDF vectorization."""

    def setUp(self):
        self.docs = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "cats and dogs are pets",
        ]
        self.tfidf = TFIDFVectorizer()
        self.vectors = self.tfidf.fit_transform(self.docs)

    def test_output_shape(self):
        """TF-IDF matrix should be (n_docs, vocab_size)."""
        self.assertEqual(self.vectors.shape[0], 3)
        self.assertEqual(self.vectors.shape[1], self.tfidf.vocab_size)

    def test_vocabulary_built(self):
        """Vocabulary should contain all unique tokens."""
        self.assertGreater(self.tfidf.vocab_size, 0)
        self.assertIn("cat", self.tfidf.vocabulary)
        self.assertIn("dog", self.tfidf.vocabulary)

    def test_nonzero_values(self):
        """TF-IDF vectors should have nonzero entries for words in the document."""
        cat_idx = self.tfidf.vocabulary["cat"]
        # "cat" appears in doc 0 but not doc 1
        self.assertGreater(self.vectors[0, cat_idx], 0)
        self.assertEqual(self.vectors[1, cat_idx], 0)

    def test_idf_weighting(self):
        """Words appearing in fewer documents should have higher IDF."""
        # "cat" appears in 1 doc, "the" appears in 2 docs
        cat_idx = self.tfidf.vocabulary["cat"]
        the_idx = self.tfidf.vocabulary["the"]
        self.assertGreater(self.tfidf.idf[cat_idx], self.tfidf.idf[the_idx])

    def test_transform_new_document(self):
        """Transform should work on documents not seen during fit."""
        new_vec = self.tfidf.transform(["the cat and the dog"])
        self.assertEqual(new_vec.shape, (1, self.tfidf.vocab_size))
        # Should have nonzero entries for known words
        cat_idx = self.tfidf.vocabulary["cat"]
        self.assertGreater(new_vec[0, cat_idx], 0)


class TestDistanceMetrics(unittest.TestCase):
    """Tests for cosine similarity, euclidean distance, and dot product."""

    def test_cosine_identical_vectors(self):
        """Identical vectors should have cosine similarity of 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)

    def test_cosine_orthogonal_vectors(self):
        """Orthogonal vectors should have cosine similarity of 0.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0, places=5)

    def test_cosine_opposite_vectors(self):
        """Opposite vectors should have cosine similarity of -1.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v1, v2), -1.0, places=5)

    def test_cosine_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        v = np.array([1.0, 2.0])
        z = np.array([0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v, z), 0.0, places=5)

    def test_euclidean_identical(self):
        """Identical vectors should have distance 0."""
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(euclidean_distance(v, v), 0.0, places=5)

    def test_euclidean_known_value(self):
        """Test against a known distance: (0,0) to (3,4) = 5."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        self.assertAlmostEqual(euclidean_distance(v1, v2), 5.0, places=5)

    def test_dot_product_known(self):
        """Test dot product against known value."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        # 1*4 + 2*5 + 3*6 = 32
        self.assertAlmostEqual(dot_product(v1, v2), 32.0, places=5)

    def test_cosine_equals_dot_for_normalized(self):
        """For unit vectors, cosine similarity should equal dot product."""
        v1 = np.array([3.0, 4.0])
        v2 = np.array([1.0, 2.0])
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        cos = cosine_similarity(v1_norm, v2_norm)
        dot = dot_product(v1_norm, v2_norm)
        self.assertAlmostEqual(cos, dot, places=5)


class TestBruteForceIndex(unittest.TestCase):
    """Tests for brute-force k-NN search."""

    def setUp(self):
        # Create simple test vectors in 2D for easy verification
        self.vectors = np.array([
            [1.0, 0.0],   # 0: right
            [0.0, 1.0],   # 1: up
            [-1.0, 0.0],  # 2: left
            [0.7, 0.7],   # 3: upper-right
            [0.9, 0.1],   # 4: mostly right
        ])
        self.index = BruteForceIndex(metric="cosine")
        self.index.add(self.vectors)

    def test_exact_match(self):
        """Querying with an indexed vector should return it as top result."""
        query = np.array([1.0, 0.0])
        results = self.index.search(query, k=1)
        self.assertEqual(results[0][0], 0)  # Index 0 is [1, 0]
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_k_results(self):
        """Should return exactly k results."""
        query = np.array([1.0, 0.0])
        results = self.index.search(query, k=3)
        self.assertEqual(len(results), 3)

    def test_ordering(self):
        """Results should be sorted by decreasing similarity."""
        query = np.array([1.0, 0.0])
        results = self.index.search(query, k=5)
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_nearest_neighbor_correctness(self):
        """Query [0.8, 0.2] should be closest to [1, 0] or [0.9, 0.1]."""
        query = np.array([0.8, 0.2])
        results = self.index.search(query, k=2)
        top_indices = {idx for idx, _ in results}
        # Should include index 4 ([0.9, 0.1]) and index 0 ([1, 0])
        self.assertTrue(top_indices & {0, 4})


class TestLSHIndex(unittest.TestCase):
    """Tests for LSH approximate nearest neighbor search."""

    def setUp(self):
        np.random.seed(42)
        # Generate clustered data: 3 clusters of 10 vectors each
        self.dim = 20
        cluster1 = np.random.randn(10, self.dim) + np.array([5] * self.dim)
        cluster2 = np.random.randn(10, self.dim) + np.array([-5] * self.dim)
        cluster3 = np.random.randn(10, self.dim)
        self.vectors = np.vstack([cluster1, cluster2, cluster3])

        self.lsh = LSHIndex(dim=self.dim, n_bits=6, n_tables=8, seed=42)
        self.lsh.add(self.vectors)

    def test_returns_results(self):
        """LSH search should return results."""
        query = self.vectors[0]  # Query a known vector
        results = self.lsh.search(query, k=5)
        self.assertGreater(len(results), 0)

    def test_self_retrieval(self):
        """Querying an indexed vector should find itself (high probability)."""
        query = self.vectors[0]
        results = self.lsh.search(query, k=5)
        found_indices = {idx for idx, _ in results}
        self.assertIn(0, found_indices)

    def test_cluster_coherence(self):
        """Results for a cluster-1 query should mostly be from cluster 1."""
        query = self.vectors[0]  # From cluster 1 (indices 0-9)
        results = self.lsh.search(query, k=5)
        cluster1_count = sum(1 for idx, _ in results if idx < 10)
        # At least 3 of top 5 should be from same cluster
        self.assertGreaterEqual(cluster1_count, 3)

    def test_stats(self):
        """Stats should report correct number of vectors."""
        stats = self.lsh.stats()
        self.assertEqual(stats["n_vectors"], 30)
        self.assertEqual(stats["n_tables"], 8)
        self.assertEqual(stats["n_bits"], 6)


class TestRecallAtK(unittest.TestCase):
    """Tests for recall@k metric."""

    def test_perfect_recall(self):
        """Identical results should give recall of 1.0."""
        exact = [(0, 0.9), (1, 0.8), (2, 0.7)]
        approx = [(0, 0.9), (1, 0.8), (2, 0.7)]
        self.assertAlmostEqual(compute_recall_at_k(exact, approx, k=3), 1.0)

    def test_zero_recall(self):
        """Completely different results should give recall of 0.0."""
        exact = [(0, 0.9), (1, 0.8), (2, 0.7)]
        approx = [(3, 0.9), (4, 0.8), (5, 0.7)]
        self.assertAlmostEqual(compute_recall_at_k(exact, approx, k=3), 0.0)

    def test_partial_recall(self):
        """Overlapping results should give proportional recall."""
        exact = [(0, 0.9), (1, 0.8), (2, 0.7)]
        approx = [(0, 0.9), (3, 0.8), (2, 0.7)]
        # 2 out of 3 match
        self.assertAlmostEqual(compute_recall_at_k(exact, approx, k=3), 2 / 3)


if __name__ == "__main__":
    unittest.main()
