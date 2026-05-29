"""
Day 046: RAG Pipeline — Test Suite

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import unittest
from my_solution import (
    Document, Chunk, SearchResult, RAGResponse,
    chunk_document, TFIDFEmbedder, VectorStore, RAGPipeline
)


class TestChunking(unittest.TestCase):
    """Test document chunking with overlap."""

    def test_basic_chunking(self):
        """Document should be split into multiple chunks."""
        doc = Document(text="A" * 600, source="test.txt")
        chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_chunk_overlap(self):
        """Consecutive chunks should have overlapping regions."""
        doc = Document(text="word " * 200, source="test.txt")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
        if len(chunks) >= 2:
            # The end of chunk 0 should overlap with the start of chunk 1
            end_of_first = chunks[0].text[-20:]
            self.assertIn(end_of_first[:10], chunks[1].text)

    def test_chunk_provenance(self):
        """Each chunk should track its source document."""
        doc = Document(text="Hello world. " * 50, source="my_file.txt")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
        for chunk in chunks:
            self.assertEqual(chunk.source, "my_file.txt")

    def test_empty_document(self):
        """Empty documents should produce no chunks."""
        doc = Document(text="", source="empty.txt")
        chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
        self.assertEqual(len(chunks), 0)

    def test_small_document(self):
        """A document smaller than chunk_size should produce exactly one chunk."""
        doc = Document(text="Short text.", source="small.txt")
        chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short text.")

    def test_chunk_indices_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        doc = Document(text="Hello. " * 100, source="test.txt")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)


class TestTFIDFEmbedder(unittest.TestCase):
    """Test the TF-IDF embedding system."""

    def setUp(self):
        self.embedder = TFIDFEmbedder()
        self.corpus = [
            "the cat sat on the mat",
            "the dog played in the park",
            "a robot navigates using sensors",
        ]
        self.embedder.fit(self.corpus)

    def test_fit_builds_vocabulary(self):
        """Fitting should create a non-empty vocabulary."""
        self.assertGreater(len(self.embedder.vocabulary), 0)
        self.assertTrue(self.embedder.fitted)

    def test_embed_returns_correct_dimensions(self):
        """Embedding dimension should match vocabulary size."""
        vec = self.embedder.embed("the cat sat on the mat")
        self.assertEqual(len(vec), len(self.embedder.vocabulary))

    def test_embed_is_normalized(self):
        """Embedding vectors should be L2-normalized (magnitude ≈ 1)."""
        vec = self.embedder.embed("the cat sat on the mat")
        import math
        magnitude = math.sqrt(sum(v * v for v in vec))
        self.assertAlmostEqual(magnitude, 1.0, places=5)

    def test_similar_texts_higher_similarity(self):
        """Semantically similar texts should have higher cosine similarity."""
        vec_cat = self.embedder.embed("the cat sat on the mat")
        vec_dog = self.embedder.embed("the dog played in the park")
        vec_robot = self.embedder.embed("a robot navigates using sensors")

        # cat/dog share words like "the", should be more similar than cat/robot
        sim_cat_dog = sum(a * b for a, b in zip(vec_cat, vec_dog))
        sim_cat_robot = sum(a * b for a, b in zip(vec_cat, vec_robot))
        self.assertGreater(sim_cat_dog, sim_cat_robot)

    def test_embed_before_fit_raises(self):
        """Embedding before fitting should raise an error."""
        fresh_embedder = TFIDFEmbedder()
        with self.assertRaises(RuntimeError):
            fresh_embedder.embed("hello world")


class TestVectorStore(unittest.TestCase):
    """Test the vector store and similarity search."""

    def setUp(self):
        self.store = VectorStore()
        self.chunks = [
            Chunk(text="about cats", source="a.txt", chunk_index=0,
                  start_char=0, end_char=10),
            Chunk(text="about dogs", source="b.txt", chunk_index=0,
                  start_char=0, end_char=10),
            Chunk(text="about robots", source="c.txt", chunk_index=0,
                  start_char=0, end_char=12),
        ]
        # Simple 3D embeddings for testing
        self.embeddings = [
            [1.0, 0.0, 0.0],   # cats -> x axis
            [0.9, 0.1, 0.0],   # dogs -> near x axis
            [0.0, 0.0, 1.0],   # robots -> z axis
        ]
        self.store.add(self.chunks, self.embeddings)

    def test_add_and_length(self):
        """Store should track the number of chunks."""
        self.assertEqual(len(self.store), 3)

    def test_search_returns_correct_count(self):
        """Search should return exactly top_k results."""
        query = [1.0, 0.0, 0.0]  # closest to cats
        results = self.store.search(query, top_k=2)
        self.assertEqual(len(results), 2)

    def test_search_finds_most_similar(self):
        """Search should return the most similar chunk first."""
        query = [1.0, 0.0, 0.0]  # closest to cats
        results = self.store.search(query, top_k=1)
        self.assertEqual(results[0].chunk.text, "about cats")

    def test_search_scores_descending(self):
        """Results should be sorted by similarity descending."""
        query = [0.5, 0.5, 0.5]
        results = self.store.search(query, top_k=3)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_empty_store(self):
        """Searching an empty store should return empty list."""
        empty_store = VectorStore()
        results = empty_store.search([1.0, 0.0, 0.0], top_k=3)
        self.assertEqual(len(results), 0)


class TestRAGPipeline(unittest.TestCase):
    """Test the full RAG pipeline end-to-end."""

    def setUp(self):
        self.docs = [
            Document(
                text=(
                    "Bitcoin uses proof of work consensus. "
                    "Miners solve cryptographic puzzles to validate transactions. "
                    "The total supply is capped at 21 million coins."
                ),
                source="bitcoin.txt"
            ),
            Document(
                text=(
                    "Kalman filters estimate state from noisy measurements. "
                    "They use a predict-update cycle. "
                    "The Kalman gain balances measurement and prediction trust."
                ),
                source="kalman.txt"
            ),
        ]
        self.rag = RAGPipeline(chunk_size=200, chunk_overlap=30, top_k=2)
        self.rag.ingest(self.docs)

    def test_ingest_returns_chunk_count(self):
        """Ingest should return the number of chunks created."""
        rag = RAGPipeline(chunk_size=200, chunk_overlap=30, top_k=2)
        count = rag.ingest(self.docs)
        self.assertGreater(count, 0)

    def test_query_returns_rag_response(self):
        """Query should return a RAGResponse with answer and sources."""
        response = self.rag.query("What is Bitcoin?")
        self.assertIsInstance(response, RAGResponse)
        self.assertIsInstance(response.answer, str)
        self.assertGreater(len(response.answer), 0)
        self.assertGreater(len(response.sources), 0)

    def test_query_retrieves_relevant_sources(self):
        """Bitcoin question should retrieve bitcoin source, not kalman."""
        response = self.rag.query("How does Bitcoin mining work?")
        source_files = [s.chunk.source for s in response.sources]
        self.assertIn("bitcoin.txt", source_files)

    def test_prompt_contains_context(self):
        """The RAG prompt should include retrieved context."""
        response = self.rag.query("What is Bitcoin?")
        self.assertIn("Context", response.prompt_used)
        self.assertIn("Question", response.prompt_used)


if __name__ == "__main__":
    unittest.main()
