"""
Day 046: Build a RAG (Retrieval-Augmented Generation) Pipeline

A complete RAG system from scratch: document chunking, embedding, vector search,
and answer generation with source attribution. Uses numpy for embeddings and
a simple TF-IDF approach so it runs without external API keys or large model downloads.

For production, swap the TF-IDF embedder with sentence-transformers or OpenAI embeddings,
and swap the template generator with an actual LLM call. The architecture stays the same.
"""

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Document:
    """A source document with text content and metadata."""
    text: str
    source: str  # filename, URL, or identifier
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a document, ready for embedding."""
    text: str
    source: str
    chunk_index: int       # position within the source document
    start_char: int        # character offset in original document
    end_char: int          # character offset end
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float           # cosine similarity, higher = more relevant


@dataclass
class RAGResponse:
    """The final response with answer and source attribution."""
    answer: str
    sources: list[SearchResult]
    prompt_used: str       # the full prompt sent to the LLM (for debugging)


# =============================================================================
# Step 1: Document Chunking
# =============================================================================

def chunk_document(
    doc: Document,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> list[Chunk]:
    """
    Split a document into overlapping chunks of approximately chunk_size characters.

    Why overlapping chunks? Consider a sentence that spans two chunks:
    "The protocol uses [chunk boundary] 256-bit encryption."
    Without overlap, neither chunk captures the complete fact. Overlap ensures
    boundary-spanning information appears intact in at least one chunk.

    Args:
        doc: The source document to chunk
        chunk_size: Target size in characters for each chunk.
                    Smaller = more precise retrieval but less context per chunk.
                    Larger = more context but diluted embedding signal.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
                       Should be 10-20% of chunk_size.

    Returns:
        List of Chunk objects with provenance information.
    """
    text = doc.text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end of the document, try to break at a sentence
        # boundary to avoid cutting mid-sentence. This improves embedding quality
        # because complete sentences embed better than fragments.
        if end < len(text):
            # Look for the last sentence-ending punctuation within the chunk
            last_period = text.rfind('.', start, end)
            last_question = text.rfind('?', start, end)
            last_exclaim = text.rfind('!', start, end)
            best_break = max(last_period, last_question, last_exclaim)

            # Only use the sentence boundary if it's in the latter half of the chunk
            # (otherwise the chunk would be too small)
            if best_break > start + chunk_size // 2:
                end = best_break + 1  # include the punctuation

        # Clamp to document length
        end = min(end, len(text))

        chunk_text = text[start:end].strip()
        if chunk_text:  # skip empty chunks
            chunks.append(Chunk(
                text=chunk_text,
                source=doc.source,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
                metadata=doc.metadata.copy()
            ))
            chunk_index += 1

        # Advance by (chunk_size - overlap) to create the overlap region.
        # The next chunk will re-read the last `overlap` characters.
        start = end - chunk_overlap if end < len(text) else end

        # Safety: ensure we always make forward progress
        if start <= chunks[-1].start_char if chunks else start <= 0:
            start = end

    return chunks


# =============================================================================
# Step 2: TF-IDF Embedding (lightweight, no external dependencies)
# =============================================================================

class TFIDFEmbedder:
    """
    A TF-IDF vectorizer that produces dense-ish embeddings for text.

    In production you'd use sentence-transformers or OpenAI embeddings, which
    capture semantic similarity (e.g., "car" ≈ "automobile"). TF-IDF only captures
    lexical overlap, but it demonstrates the same pipeline architecture.

    TF-IDF = Term Frequency × Inverse Document Frequency
    - TF: how often a word appears in THIS document (local importance)
    - IDF: log(N / df) where df = how many documents contain this word (global rarity)
    - Words that are frequent locally but rare globally get high scores
    - Stop words like "the" get low IDF (they appear everywhere)
    """

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}   # word -> index
        self.idf: dict[str, float] = {}        # word -> IDF score
        self.fitted: bool = False

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenizer: lowercase, remove punctuation, split on whitespace.
        Production systems use subword tokenizers (BPE, WordPiece) that handle
        morphology and out-of-vocabulary words better.
        """
        text = text.lower()
        # Remove punctuation except hyphens within words
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = text.strip().split()
        # Remove very short tokens (usually noise)
        return [t for t in tokens if len(t) > 1]

    def fit(self, texts: list[str]) -> None:
        """
        Build vocabulary and compute IDF scores from a corpus.

        IDF(word) = log(N / (1 + df(word)))
        where N = total documents, df = documents containing word.
        The +1 in the denominator prevents division by zero and provides
        mild smoothing.
        """
        n_docs = len(texts)
        doc_freq: Counter = Counter()  # how many docs contain each word
        all_words: set[str] = set()

        for text in texts:
            tokens = set(self._tokenize(text))  # set to count each word once per doc
            for token in tokens:
                doc_freq[token] += 1
            all_words.update(tokens)

        # Build vocabulary: assign an index to each unique word
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}

        # Compute IDF for each word
        self.idf = {
            word: math.log(n_docs / (1 + df))
            for word, df in doc_freq.items()
        }

        self.fitted = True

    def embed(self, text: str) -> list[float]:
        """
        Convert text to a TF-IDF vector.

        Returns a vector of length |vocabulary| where each dimension is
        TF(word) * IDF(word). The vector is L2-normalized so cosine similarity
        reduces to a dot product.
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before embed()")

        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total_tokens = len(tokens) if tokens else 1

        # Build the TF-IDF vector
        vector = [0.0] * len(self.vocabulary)
        for word, count in tf.items():
            if word in self.vocabulary:
                # TF is normalized by document length to prevent bias toward longer chunks
                tf_score = count / total_tokens
                idf_score = self.idf.get(word, 0.0)
                vector[self.vocabulary[word]] = tf_score * idf_score

        # L2 normalize: this makes cosine similarity = dot product,
        # which is computationally cheaper
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. In production, this would batch GPU inference."""
        return [self.embed(text) for text in texts]


# =============================================================================
# Step 3: Vector Store with Similarity Search
# =============================================================================

class VectorStore:
    """
    In-memory vector store with brute-force cosine similarity search.

    This is what ChromaDB, Pinecone, or pgvector do under the hood, but they add:
    - Approximate Nearest Neighbor (ANN) for sub-linear search (HNSW, IVF)
    - Persistence to disk
    - Metadata filtering
    - Distributed scaling

    For <10K chunks, brute force is fast enough and exact (no approximation error).
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks and their embeddings to the store."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length")
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        """
        Find the top_k most similar chunks to the query.

        Since our embeddings are L2-normalized, cosine similarity = dot product:
            cos(a, b) = (a · b) / (||a|| * ||b||)
        With ||a|| = ||b|| = 1, this simplifies to just a · b.

        Time complexity: O(n * d) where n = number of chunks, d = embedding dimension.
        Space complexity: O(n) for storing scores.
        """
        if not self.embeddings:
            return []

        scores: list[tuple[int, float]] = []
        for idx, stored_emb in enumerate(self.embeddings):
            # Dot product = cosine similarity for normalized vectors
            similarity = sum(a * b for a, b in zip(query_embedding, stored_emb))
            scores.append((idx, similarity))

        # Sort by similarity descending, take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]

        return [
            SearchResult(chunk=self.chunks[idx], score=score)
            for idx, score in top_scores
        ]

    def __len__(self) -> int:
        return len(self.chunks)


# =============================================================================
# Step 4: RAG Pipeline
# =============================================================================

class RAGPipeline:
    """
    The complete RAG pipeline: ingest documents, index them, answer questions.

    Architecture:
        Ingest:  Documents -> Chunking -> Embedding -> Vector Store
        Query:   Question -> Embedding -> Search -> Prompt Construction -> Generation
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
        Ingest documents into the RAG pipeline.

        This is the "offline" phase — done once when your knowledge base changes.
        In production, this runs as a background job triggered by document updates.

        Returns the total number of chunks indexed.
        """
        # Step 1: Chunk all documents
        for doc in documents:
            chunks = chunk_document(doc, self.chunk_size, self.chunk_overlap)
            self.all_chunks.extend(chunks)

        if not self.all_chunks:
            return 0

        # Step 2: Fit the embedder on the entire corpus
        # TF-IDF needs to see all documents to compute IDF scores.
        # Transformer-based embedders skip this step (they're pre-trained).
        all_texts = [c.text for c in self.all_chunks]
        self.embedder.fit(all_texts)

        # Step 3: Embed all chunks and add to vector store
        embeddings = self.embedder.embed_batch(all_texts)
        self.vector_store.add(self.all_chunks, embeddings)

        return len(self.all_chunks)

    def _build_prompt(self, question: str, context_chunks: list[SearchResult]) -> str:
        """
        Construct the augmented prompt with retrieved context.

        Prompt engineering for RAG is critical:
        1. System instruction: sets the role and constraints
        2. Context: retrieved chunks, most relevant first
        3. Question: the user's actual query
        4. Output instruction: format requirements

        Key: "If the context doesn't contain the answer, say so."
        This reduces hallucination — without it, the LLM will make up answers
        using its parametric knowledge, defeating the purpose of RAG.
        """
        context_parts = []
        for i, result in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source {i}: {result.chunk.source}, "
                f"chunk {result.chunk.chunk_index}]\n"
                f"{result.chunk.text}"
            )

        context_str = "\n\n".join(context_parts)

        prompt = (
            "You are a helpful assistant that answers questions based on the provided context.\n"
            "Use ONLY the information in the context below. If the context does not contain\n"
            "enough information to answer the question, say \"I don't have enough information\n"
            "to answer this question based on the available context.\"\n"
            "Cite your sources using [Source N] notation.\n"
            "\n"
            "Context:\n"
            f"{context_str}\n"
            "\n"
            f"Question: {question}\n"
            "\n"
            "Answer:"
        )
        return prompt

    def _generate_answer(self, prompt: str, sources: list[SearchResult]) -> str:
        """
        Generate an answer from the prompt.

        In production, this calls an LLM API:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        Here we simulate generation by extracting key sentences from the
        retrieved context that are most relevant to the question. This
        demonstrates the mechanics without requiring an API key.
        """
        # Extract the question from the prompt
        question_match = re.search(r'Question: (.+)', prompt)
        question = question_match.group(1) if question_match else ""
        question_words = set(question.lower().split())

        # Score sentences from the context by relevance to the question
        # (a very simple extractive approach — real RAG uses generative LLMs)
        scored_sentences: list[tuple[str, float, int]] = []
        for i, source in enumerate(sources):
            sentences = re.split(r'[.!?]+', source.chunk.text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # skip fragments
                    continue
                words = set(sentence.lower().split())
                overlap = len(question_words & words)
                scored_sentences.append((sentence, overlap + source.score, i + 1))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Take top 3 sentences and format as an answer with citations
        answer_parts = []
        used_sources = set()
        for sentence, score, source_idx in scored_sentences[:3]:
            answer_parts.append(f"{sentence}. [Source {source_idx}]")
            used_sources.add(source_idx)

        if answer_parts:
            return " ".join(answer_parts)
        else:
            return "I don't have enough information to answer this question based on the available context."

    def query(self, question: str) -> RAGResponse:
        """
        Answer a question using the RAG pipeline.

        This is the "online" phase — runs on every user query.
        Latency budget in production:
            - Embed query: ~10ms (transformer) or <1ms (TF-IDF)
            - Vector search: ~5ms (ANN) or O(n) brute force
            - LLM generation: 500-2000ms (dominates latency)
        """
        # Step 1: Embed the question with the same embedder used for chunks
        # Critical: query and chunks MUST use the same embedding model/space
        query_embedding = self.embedder.embed(question)

        # Step 2: Retrieve top-k relevant chunks
        results = self.vector_store.search(query_embedding, self.top_k)

        # Step 3: Build the augmented prompt
        prompt = self._build_prompt(question, results)

        # Step 4: Generate answer
        answer = self._generate_answer(prompt, results)

        return RAGResponse(
            answer=answer,
            sources=results,
            prompt_used=prompt
        )


# =============================================================================
# Sample knowledge base for demonstration
# =============================================================================

SAMPLE_DOCUMENTS = [
    Document(
        text=(
            "Bitcoin is a decentralized digital currency created in 2009 by Satoshi Nakamoto. "
            "It uses a proof-of-work consensus mechanism where miners compete to solve "
            "cryptographic puzzles. The Bitcoin network processes about 7 transactions per second. "
            "Bitcoin's total supply is capped at 21 million coins, creating digital scarcity. "
            "The block reward halves approximately every 4 years in an event called the halving. "
            "Bitcoin uses SHA-256 as its hashing algorithm for proof of work. "
            "Transactions are grouped into blocks, and each block references the previous block's hash, "
            "forming an immutable chain. The average block time is about 10 minutes."
        ),
        source="bitcoin_overview.txt",
        metadata={"topic": "cryptocurrency", "subtopic": "bitcoin"}
    ),
    Document(
        text=(
            "Ethereum is a blockchain platform that supports smart contracts and decentralized "
            "applications. It was proposed by Vitalik Buterin in 2013 and launched in 2015. "
            "Ethereum transitioned from proof-of-work to proof-of-stake in September 2022 "
            "during an event called The Merge. This reduced energy consumption by approximately 99.95%. "
            "The Ethereum Virtual Machine (EVM) executes smart contract bytecode. "
            "Gas fees are paid in ETH to compensate validators for computation. "
            "EIP-1559 introduced a base fee that is burned and a priority fee that goes to validators. "
            "Ethereum processes about 15-30 transactions per second on the base layer, "
            "with Layer 2 rollups like Optimism and Arbitrum handling thousands more."
        ),
        source="ethereum_overview.txt",
        metadata={"topic": "cryptocurrency", "subtopic": "ethereum"}
    ),
    Document(
        text=(
            "Reinforcement learning is a type of machine learning where an agent learns to make "
            "decisions by interacting with an environment. The agent receives rewards or penalties "
            "for its actions and learns a policy that maximizes cumulative reward. "
            "Key concepts include the state space, action space, reward function, and discount factor. "
            "Q-learning is a model-free RL algorithm that learns action-value functions. "
            "Deep Q-Networks (DQN) combine Q-learning with neural networks to handle large state spaces. "
            "Policy gradient methods directly optimize the policy using gradient ascent on expected reward. "
            "The exploration-exploitation tradeoff is fundamental: the agent must balance "
            "trying new actions (exploration) with using known good actions (exploitation). "
            "Epsilon-greedy is a simple strategy: with probability epsilon, take a random action; "
            "otherwise, take the best known action."
        ),
        source="reinforcement_learning.txt",
        metadata={"topic": "ai", "subtopic": "reinforcement_learning"}
    ),
    Document(
        text=(
            "A Kalman filter is an algorithm that uses a series of measurements observed over time "
            "to estimate unknown variables. It operates recursively on streams of noisy input data "
            "to produce a statistically optimal estimate of the underlying system state. "
            "The algorithm works in two steps: predict and update. In the predict step, "
            "the filter projects the current state estimate forward using a state transition model. "
            "In the update step, it incorporates a new measurement, weighting the prediction "
            "and measurement by their respective uncertainties. "
            "The Kalman gain determines how much to trust the measurement versus the prediction. "
            "When measurement noise is high, the gain is low (trust the prediction more). "
            "When process noise is high, the gain is high (trust the measurement more). "
            "Kalman filters are widely used in robotics for sensor fusion, GPS navigation, "
            "and autonomous vehicle state estimation."
        ),
        source="kalman_filter.txt",
        metadata={"topic": "robotics", "subtopic": "state_estimation"}
    ),
    Document(
        text=(
            "SLAM stands for Simultaneous Localization and Mapping. It is the problem of "
            "building a map of an unknown environment while simultaneously keeping track of "
            "the robot's location within it. This is a chicken-and-egg problem: you need a map "
            "to localize, but you need to know your location to build a map. "
            "EKF-SLAM uses an Extended Kalman Filter to jointly estimate the robot pose and "
            "landmark positions. The state vector contains the robot's position and orientation "
            "plus the positions of all observed landmarks. "
            "Particle filter SLAM (FastSLAM) represents the robot's pose distribution as a set "
            "of particles, each with its own map estimate. "
            "Graph-based SLAM formulates the problem as a pose graph optimization, where nodes "
            "are robot poses and edges are spatial constraints from odometry and observations. "
            "Modern visual SLAM systems like ORB-SLAM use camera features for real-time operation."
        ),
        source="slam_overview.txt",
        metadata={"topic": "robotics", "subtopic": "slam"}
    ),
]


# =============================================================================
# Main: demonstrate the complete pipeline
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RAG Pipeline — Full Demonstration")
    print("=" * 70)

    # --- Initialize the pipeline ---
    print("\n[1] Initializing RAG pipeline...")
    rag = RAGPipeline(chunk_size=300, chunk_overlap=50, top_k=3)

    # --- Ingest documents ---
    print(f"\n[2] Ingesting {len(SAMPLE_DOCUMENTS)} documents...")
    num_chunks = rag.ingest(SAMPLE_DOCUMENTS)
    print(f"    Created {num_chunks} chunks from {len(SAMPLE_DOCUMENTS)} documents")
    print(f"    Vocabulary size: {len(rag.embedder.vocabulary)} unique terms")

    # Show the chunks created from the first document
    print("\n    Example chunks from 'bitcoin_overview.txt':")
    btc_chunks = [c for c in rag.all_chunks if c.source == "bitcoin_overview.txt"]
    for chunk in btc_chunks:
        preview = chunk.text[:80].replace('\n', ' ')
        print(f"    Chunk {chunk.chunk_index}: [{chunk.start_char}:{chunk.end_char}] \"{preview}...\"")

    # --- Query 1: Specific factual question ---
    print("\n" + "=" * 70)
    print("[3] Query 1: 'How does Ethereum handle transaction fees?'")
    print("=" * 70)
    response = rag.query("How does Ethereum handle transaction fees?")

    print(f"\n    Answer: {response.answer}")
    print(f"\n    Sources retrieved ({len(response.sources)}):")
    for i, src in enumerate(response.sources, 1):
        print(f"    {i}. [{src.chunk.source}] chunk {src.chunk.chunk_index} "
              f"(similarity: {src.score:.4f})")
        print(f"       \"{src.chunk.text[:100]}...\"")

    # --- Query 2: Cross-document question ---
    print("\n" + "=" * 70)
    print("[4] Query 2: 'What is the difference between proof of work and proof of stake?'")
    print("=" * 70)
    response2 = rag.query("What is the difference between proof of work and proof of stake?")

    print(f"\n    Answer: {response2.answer}")
    print(f"\n    Sources retrieved ({len(response2.sources)}):")
    for i, src in enumerate(response2.sources, 1):
        print(f"    {i}. [{src.chunk.source}] chunk {src.chunk.chunk_index} "
              f"(similarity: {src.score:.4f})")

    # --- Query 3: Robotics question ---
    print("\n" + "=" * 70)
    print("[5] Query 3: 'How does a Kalman filter decide whether to trust the sensor or the prediction?'")
    print("=" * 70)
    response3 = rag.query("How does a Kalman filter decide whether to trust the sensor or the prediction?")

    print(f"\n    Answer: {response3.answer}")
    print(f"\n    Sources retrieved ({len(response3.sources)}):")
    for i, src in enumerate(response3.sources, 1):
        print(f"    {i}. [{src.chunk.source}] chunk {src.chunk.chunk_index} "
              f"(similarity: {src.score:.4f})")

    # --- Query 4: Out-of-scope question ---
    print("\n" + "=" * 70)
    print("[6] Query 4: 'What is the capital of France?' (should have low relevance)")
    print("=" * 70)
    response4 = rag.query("What is the capital of France?")

    print(f"\n    Answer: {response4.answer}")
    print(f"\n    Top similarity score: {response4.sources[0].score:.4f}")
    print(f"    (Low score indicates the knowledge base doesn't cover this topic)")

    # --- Show the full RAG prompt for debugging ---
    print("\n" + "=" * 70)
    print("[7] Example RAG prompt (what gets sent to the LLM):")
    print("=" * 70)
    print(response.prompt_used)

    # --- Pipeline statistics ---
    print("\n" + "=" * 70)
    print("[8] Pipeline Statistics")
    print("=" * 70)
    print(f"    Documents indexed: {len(SAMPLE_DOCUMENTS)}")
    print(f"    Total chunks: {len(rag.vector_store)}")
    print(f"    Vocabulary size: {len(rag.embedder.vocabulary)}")
    print(f"    Embedding dimensions: {len(rag.embedder.vocabulary)}")
    print(f"    Chunk size: {rag.chunk_size} chars, overlap: {rag.chunk_overlap} chars")
    print(f"    Top-k retrieval: {rag.top_k}")

    avg_chunk_len = sum(len(c.text) for c in rag.all_chunks) / len(rag.all_chunks)
    print(f"    Average chunk length: {avg_chunk_len:.0f} chars")

    print("\n    Done! In production, swap TFIDFEmbedder with sentence-transformers")
    print("    and _generate_answer with an actual LLM API call.")
