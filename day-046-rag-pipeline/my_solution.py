"""
Day 046: Build a RAG (Retrieval-Augmented Generation) Pipeline

YOUR TASK: Implement a complete RAG system from scratch.

The pipeline has 4 main components:
1. Document chunking with overlap
2. TF-IDF embedding (text -> vector)
3. Vector store with cosine similarity search
4. Query pipeline: embed question -> search -> build prompt -> generate answer

Hints are provided as comments. Run this file to test your implementation.
"""

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Data structures (provided — do not modify)
# =============================================================================

@dataclass
class Document:
    """A source document with text content and metadata."""
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a document, ready for embedding."""
    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    """The final response with answer and source attribution."""
    answer: str
    sources: list[SearchResult]
    prompt_used: str


# =============================================================================
# Step 1: Document Chunking — implement this
# =============================================================================

def chunk_document(
    doc: Document,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> list[Chunk]:
    """
    Split a document into overlapping chunks of approximately chunk_size characters.

    Args:
        doc: The source document to chunk
        chunk_size: Target size in characters for each chunk
        chunk_overlap: Number of characters to overlap between consecutive chunks

    Returns:
        List of Chunk objects with provenance information.

    Hints:
        - Walk through the text with a sliding window
        - Try to break at sentence boundaries (., !, ?) for better embeddings
        - Track start_char and end_char for each chunk
        - Remember to handle the overlap: next chunk starts at (end - overlap)
        - Ensure forward progress to avoid infinite loops
    """
    raise NotImplementedError("TODO: implement document chunking")


# =============================================================================
# Step 2: TF-IDF Embedder — implement this
# =============================================================================

class TFIDFEmbedder:
    """
    A TF-IDF vectorizer that produces embedding vectors for text.

    Hints:
        - TF(word) = count(word) / total_words_in_doc
        - IDF(word) = log(N / (1 + docs_containing_word))
        - The embedding vector has one dimension per vocabulary word
        - L2-normalize the vector so cosine similarity = dot product
    """

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.fitted: bool = False

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text: lowercase, remove punctuation, split on whitespace.

        Hints:
            - Use regex to strip punctuation
            - Filter out very short tokens (length <= 1)
        """
        raise NotImplementedError("TODO: implement tokenization")

    def fit(self, texts: list[str]) -> None:
        """
        Build vocabulary and compute IDF scores from a corpus.

        Hints:
            - Count document frequency: for each text, which unique words appear?
            - Build vocabulary: assign each unique word an index
            - IDF = log(N / (1 + df)) for each word
            - Set self.fitted = True when done
        """
        raise NotImplementedError("TODO: implement fit")

    def embed(self, text: str) -> list[float]:
        """
        Convert text to a TF-IDF vector.

        Hints:
            - Tokenize, compute term frequencies
            - Multiply TF * IDF for each word in vocabulary
            - L2-normalize: divide each element by sqrt(sum of squares)
            - Return a list of floats with length = vocabulary size
        """
        raise NotImplementedError("TODO: implement embed")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [self.embed(text) for text in texts]


# =============================================================================
# Step 3: Vector Store — implement this
# =============================================================================

class VectorStore:
    """
    In-memory vector store with brute-force cosine similarity search.

    Hints:
        - Store chunks and their embeddings in parallel lists
        - For search: compute dot product (= cosine sim for normalized vectors)
        - Sort by similarity descending, return top-k
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks and their embeddings to the store."""
        raise NotImplementedError("TODO: implement add")

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        """
        Find the top_k most similar chunks to the query.

        Hints:
            - Compute dot product of query_embedding with each stored embedding
            - Dot product = cosine similarity when vectors are L2-normalized
            - Sort by score descending, take top_k
        """
        raise NotImplementedError("TODO: implement search")

    def __len__(self) -> int:
        return len(self.chunks)


# =============================================================================
# Step 4: RAG Pipeline — implement this
# =============================================================================

class RAGPipeline:
    """
    The complete RAG pipeline: ingest documents, index them, answer questions.
    """

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        top_k: int = 3
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedder = TFIDFEmbedder()
        self.vector_store = VectorStore()
        self.all_chunks: list[Chunk] = []

    def ingest(self, documents: list[Document]) -> int:
        """
        Ingest documents: chunk them, fit embedder, embed chunks, store in vector store.

        Hints:
            - Chunk each document using chunk_document()
            - Fit the embedder on ALL chunk texts
            - Embed all chunks and add to vector store
            - Return total number of chunks
        """
        raise NotImplementedError("TODO: implement ingest")

    def _build_prompt(self, question: str, context_chunks: list[SearchResult]) -> str:
        """
        Construct the augmented prompt with retrieved context.

        Hints:
            - Include system instruction (answer from context only)
            - Include each retrieved chunk with source attribution
            - Include the user's question
            - Tell the model to say "I don't know" if context is insufficient
        """
        raise NotImplementedError("TODO: implement prompt construction")

    def _generate_answer(self, prompt: str, sources: list[SearchResult]) -> str:
        """
        Generate an answer from the prompt.

        For this exercise, implement a simple extractive approach:
            - Find sentences in the retrieved chunks most relevant to the question
            - Return the top 3 with source citations

        Hints:
            - Extract the question from the prompt
            - Split chunk text into sentences
            - Score each sentence by word overlap with the question
            - Return top-scoring sentences with [Source N] citations
        """
        raise NotImplementedError("TODO: implement answer generation")

    def query(self, question: str) -> RAGResponse:
        """
        Answer a question using the RAG pipeline.

        Hints:
            - Embed the question
            - Search for top-k similar chunks
            - Build the augmented prompt
            - Generate the answer
            - Return a RAGResponse with answer, sources, and prompt
        """
        raise NotImplementedError("TODO: implement query")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    # Sample documents to test with
    docs = [
        Document(
            text=(
                "Bitcoin is a decentralized digital currency created in 2009 by Satoshi Nakamoto. "
                "It uses a proof-of-work consensus mechanism where miners compete to solve "
                "cryptographic puzzles. Bitcoin's total supply is capped at 21 million coins."
            ),
            source="bitcoin.txt"
        ),
        Document(
            text=(
                "Ethereum is a blockchain platform that supports smart contracts. "
                "It transitioned from proof-of-work to proof-of-stake in 2022. "
                "Gas fees are paid in ETH to compensate validators for computation."
            ),
            source="ethereum.txt"
        ),
        Document(
            text=(
                "A Kalman filter estimates unknown variables from noisy measurements. "
                "It works in two steps: predict and update. The Kalman gain determines "
                "how much to trust the measurement versus the prediction."
            ),
            source="kalman.txt"
        ),
    ]

    # Initialize and ingest
    rag = RAGPipeline(chunk_size=200, chunk_overlap=30, top_k=2)
    num_chunks = rag.ingest(docs)
    print(f"Ingested {len(docs)} documents into {num_chunks} chunks")

    # Test queries
    questions = [
        "How does Bitcoin achieve consensus?",
        "What is a Kalman filter?",
        "How does Ethereum handle fees?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        response = rag.query(q)
        print(f"A: {response.answer}")
        print(f"Sources: {[s.chunk.source for s in response.sources]}")
