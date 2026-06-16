"""
Day 76: Byte-Pair Encoding (BPE) Tokenizer from Scratch — Your Implementation

Build a BPE tokenizer that:
1. Trains on a text corpus by iteratively merging the most frequent adjacent pairs
2. Encodes arbitrary text into token ID sequences
3. Decodes token ID sequences back to the original text (losslessly)

Key concepts to keep in mind:
- The base vocabulary is 256 individual bytes (0-255)
- Each merge creates ONE new token and adds it to the vocabulary
- Merge order matters during encoding — apply them in training order
- Pre-tokenization (splitting on spaces) prevents cross-word merges
"""

from typing import Optional


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer that learns subword units from training data.

    Attributes:
        vocab_size: Target vocabulary size (256 base bytes + num_merges)
        merges: Ordered list of merge rules as (pair, new_token_id) tuples
        vocab: Mapping from token_id -> bytes representation
    """

    def __init__(self, vocab_size: int = 512) -> None:
        """
        Initialize the tokenizer.

        Args:
            vocab_size: Desired vocabulary size. Must be > 256.
                       Number of BPE merges = vocab_size - 256.
        """
        assert vocab_size > 256, "Vocab size must exceed 256 (the base byte vocabulary)"
        self.vocab_size = vocab_size
        self.merges: list[tuple[tuple[int, int], int]] = []
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def _get_pair_counts(self, token_sequences: list[list[int]]) -> dict[tuple[int, int], int]:
        """
        Count frequency of every adjacent token pair across all sequences.

        Hint: Iterate through each sequence, look at consecutive pairs.
        Use a dictionary to accumulate counts.

        Args:
            token_sequences: List of token sequences

        Returns:
            Dictionary mapping (token_a, token_b) -> count
        """
        raise NotImplementedError("TODO: implement this")

    def _merge_pair(
        self, token_sequences: list[list[int]], pair: tuple[int, int], new_id: int
    ) -> list[list[int]]:
        """
        Replace all occurrences of `pair` with `new_id` in all sequences.

        Hint: Scan each sequence left-to-right. When you find the pair,
        append new_id and skip ahead by 2. Otherwise append current token
        and advance by 1.

        Watch out for overlapping patterns like [a, a, a] with pair (a, a).

        Args:
            token_sequences: Current corpus as list of token sequences
            pair: The (token_a, token_b) pair to merge
            new_id: The new token ID

        Returns:
            Updated token sequences
        """
        raise NotImplementedError("TODO: implement this")

    def train(self, text: str, verbose: bool = False) -> None:
        """
        Train the BPE tokenizer on a text corpus.

        Steps:
        1. Split text into words (pre-tokenization on whitespace)
        2. Convert words to byte sequences (prepend space to all but first)
        3. Loop vocab_size - 256 times:
           a. Count all adjacent pairs
           b. Find the most frequent pair
           c. Create new token (id = 256 + merge_index)
           d. Record merge rule and update vocab
           e. Apply merge to all sequences

        Hint: self.vocab[new_id] should be the concatenation of
        self.vocab[pair[0]] + self.vocab[pair[1]]

        Args:
            text: Raw training text
            verbose: If True, print progress
        """
        raise NotImplementedError("TODO: implement this")

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a sequence of token IDs.

        Steps:
        1. Convert text to list of byte values
        2. For each merge rule (in training order!), apply it to the sequence
        3. Return the final token sequence

        Hint: Reuse the merge logic but for a single sequence.
        The ORDER of applying merges matters — use training order.

        Args:
            text: String to encode

        Returns:
            List of token IDs
        """
        raise NotImplementedError("TODO: implement this")

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a string.

        Hint: Look up each token_id in self.vocab to get its bytes,
        concatenate all bytes, then decode as UTF-8.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded string
        """
        raise NotImplementedError("TODO: implement this")

    def get_token_string(self, token_id: int) -> str:
        """Get the string representation of a single token."""
        return self.vocab[token_id].decode("utf-8", errors="replace")

    def get_vocab_tokens(self) -> list[tuple[int, str]]:
        """Return all tokens in the vocabulary as (id, string) pairs, sorted by ID."""
        return [
            (tid, self.vocab[tid].decode("utf-8", errors="replace"))
            for tid in sorted(self.vocab.keys())
        ]

    def compression_ratio(self, text: str) -> float:
        """
        Calculate compression ratio: original_bytes / num_tokens.
        Higher is better.

        Hint: Get byte length of text, get length of encoded token list,
        return the ratio.
        """
        raise NotImplementedError("TODO: implement this")

    def tokenize_verbose(self, text: str) -> list[tuple[int, str]]:
        """Encode text and return tokens with their string representations."""
        token_ids = self.encode(text)
        return [(tid, self.get_token_string(tid)) for tid in token_ids]


def get_training_corpus() -> str:
    """Return a small training corpus for demonstration."""
    return """
    The quick brown fox jumps over the lazy dog. The dog barked at the fox.
    Machine learning is a subset of artificial intelligence. Deep learning uses
    neural networks with many layers. Natural language processing enables
    computers to understand human language. The transformer architecture
    revolutionized natural language processing. Attention mechanisms allow
    models to focus on relevant parts of the input. Tokenization is the first
    step in any natural language processing pipeline. Byte pair encoding is
    a popular tokenization algorithm. The vocabulary size affects both the
    model size and the sequence length. Larger vocabularies lead to shorter
    sequences but require more parameters. The tradeoff between vocabulary
    size and sequence length is important for model efficiency.

    Bitcoin uses a proof of work consensus mechanism. Ethereum supports smart
    contracts and decentralized applications. Blockchain technology provides
    a distributed ledger for recording transactions. Cryptographic hash
    functions ensure data integrity. Digital signatures verify the authenticity
    of messages. Merkle trees efficiently verify large datasets.

    Robots use sensors to perceive their environment. Path planning algorithms
    help robots navigate from one point to another. PID controllers maintain
    stable system behavior. Reinforcement learning enables robots to learn
    from experience. Computer vision allows robots to see and interpret
    their surroundings. The fusion of AI and robotics creates autonomous
    systems capable of complex decision making.

    The cat sat on the mat. The mat was on the floor. The floor was in the
    room. The room was in the house. The house was on the street. The street
    was in the city. The city was in the country. The country was on the
    continent. The continent was on the planet.
    """


if __name__ == "__main__":
    print("Testing your BPE Tokenizer implementation...")

    corpus = get_training_corpus()
    tokenizer = BPETokenizer(vocab_size=350)

    # Train
    print("\n1. Training tokenizer...")
    tokenizer.train(corpus, verbose=True)
    print(f"   Vocabulary size: {len(tokenizer.vocab)}")

    # Encode/decode
    print("\n2. Testing encode/decode...")
    test_strings = [
        "The quick brown fox",
        "natural language processing",
        "blockchain technology",
        "tokenization algorithm",
    ]

    for text in test_strings:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"   '{text}' -> {len(encoded)} tokens -> '{decoded}'")
        assert decoded == text, f"Round-trip FAILED for '{text}'"

    # Compression
    print("\n3. Testing compression ratio...")
    ratio = tokenizer.compression_ratio(corpus)
    print(f"   Corpus compression: {ratio:.2f}x")

    print("\nAll tests passed!")
