"""
Day 76: Byte-Pair Encoding (BPE) Tokenizer from Scratch

A complete implementation of the BPE algorithm used by GPT, Claude, and modern LLMs.
Trains a subword vocabulary from raw text, then encodes/decodes arbitrary strings.

The algorithm:
1. Start with individual bytes as the vocabulary
2. Repeatedly find the most frequent adjacent pair
3. Merge that pair into a new token
4. Record the merge rule
5. Stop at desired vocabulary size

This gives us a vocabulary of subword units that balances between character-level
(tiny vocab, long sequences) and word-level (huge vocab, OOV problems) approaches.
"""

from typing import Optional


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer that learns subword units from training data.

    The tokenizer operates on raw bytes, making it language-agnostic — any valid
    byte sequence can be tokenized and perfectly reconstructed.

    Attributes:
        vocab_size: Target vocabulary size (256 base bytes + num_merges)
        merges: Ordered list of merge rules as (pair, new_token_id) tuples
        vocab: Mapping from token_id -> bytes representation
        inverse_vocab: Mapping from bytes -> token_id (for encoding)
    """

    def __init__(self, vocab_size: int = 512) -> None:
        """
        Initialize the tokenizer.

        Args:
            vocab_size: Desired vocabulary size. Must be > 256 since the base
                       vocabulary (individual bytes) takes 256 slots. The number
                       of BPE merges performed = vocab_size - 256.
        """
        assert vocab_size > 256, "Vocab size must exceed 256 (the base byte vocabulary)"
        self.vocab_size = vocab_size
        # merges stores the ordered merge rules: list of ((id_a, id_b), new_id)
        # The ORDER matters — during encoding, we apply merges in this exact order
        self.merges: list[tuple[tuple[int, int], int]] = []
        # vocab maps token_id -> the bytes that token represents
        # Initialize with the 256 individual bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def _get_pair_counts(self, token_sequences: list[list[int]]) -> dict[tuple[int, int], int]:
        """
        Count frequency of every adjacent token pair across all sequences.

        This is the core statistical operation in BPE — we need to find which
        pair of adjacent tokens appears most often so we can merge it next.

        Args:
            token_sequences: List of token sequences (each sequence is a list of ints)

        Returns:
            Dictionary mapping (token_a, token_b) -> frequency count

        Why we scan ALL sequences: BPE is a corpus-level algorithm. A pair that's
        common across many words should be prioritized over one that's frequent
        in just one word. This gives us globally useful subword units.
        """
        counts: dict[tuple[int, int], int] = {}
        for seq in token_sequences:
            # Iterate through adjacent pairs in this sequence
            # For sequence [a, b, c, d], pairs are (a,b), (b,c), (c,d)
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(
        self, token_sequences: list[list[int]], pair: tuple[int, int], new_id: int
    ) -> list[list[int]]:
        """
        Replace all occurrences of `pair` with `new_id` in all sequences.

        This modifies the corpus representation after we decide to merge a pair.
        We scan left-to-right and greedily replace — if we see the pair, we merge it.

        Args:
            token_sequences: Current corpus as list of token sequences
            pair: The (token_a, token_b) pair to merge
            new_id: The new token ID that replaces the pair

        Returns:
            Updated token sequences with all occurrences of pair replaced

        Subtle behavior: Merging can be non-trivial with overlapping patterns.
        Consider sequence [a, a, a] with pair (a, a):
        - Left-to-right greedy gives [aa, a] (merge first occurrence)
        - We do NOT get [a, aa] — greediness means first match wins
        """
        new_sequences = []
        for seq in token_sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                # Check if we're at a position where the pair starts
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(new_id)
                    i += 2  # Skip both tokens in the pair
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences.append(new_seq)
        return new_sequences

    def train(self, text: str, verbose: bool = False) -> None:
        """
        Train the BPE tokenizer on a text corpus.

        The training loop:
        1. Convert text to bytes (our initial tokens)
        2. Split into word-level sequences (pre-tokenization)
        3. Repeatedly find and merge the most frequent pair
        4. Stop when we reach desired vocab size

        Pre-tokenization (splitting on spaces/punctuation) is important because
        it prevents merges from spanning word boundaries. Without it, BPE might
        learn a token like "e t" (the 'e' at end of one word + space + 't' at
        start of next), which wastes vocabulary slots on non-meaningful units.

        Args:
            text: Raw training text
            verbose: If True, print progress during training
        """
        # Step 1: Pre-tokenize by splitting on whitespace
        # Each word (including its trailing space if any) becomes a separate sequence
        # This is a simplified version — production tokenizers use regex patterns
        # like GPT-2's: r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+"""
        words = text.split()

        # Step 2: Convert each word to a sequence of byte values
        # We prepend a space to each word except the first (to capture word boundaries)
        # This means the tokenizer learns space-prefixed tokens like " the" as a single unit
        token_sequences: list[list[int]] = []
        for i, word in enumerate(words):
            prefix = " " if i > 0 else ""
            word_bytes = (prefix + word).encode("utf-8")
            token_sequences.append(list(word_bytes))

        num_merges = self.vocab_size - 256

        if verbose:
            print(f"Training BPE tokenizer: {num_merges} merges on {len(token_sequences)} words")
            print(f"Initial corpus size: {sum(len(s) for s in token_sequences)} tokens")

        # Step 3: Main training loop — greedily merge most frequent pair
        for merge_idx in range(num_merges):
            # Count all adjacent pairs across the entire corpus
            pair_counts = self._get_pair_counts(token_sequences)

            if not pair_counts:
                # No more pairs to merge (corpus is fully compressed)
                if verbose:
                    print(f"  No more pairs after {merge_idx} merges. Stopping early.")
                break

            # Find the most frequent pair
            # Ties are broken arbitrarily (by dict ordering), which is fine
            best_pair = max(pair_counts, key=pair_counts.get)  # type: ignore[arg-type]
            best_count = pair_counts[best_pair]

            # Assign the next available ID to the new merged token
            new_id = 256 + merge_idx

            # Record the merge rule
            self.merges.append((best_pair, new_id))

            # Update the vocabulary: new token = concatenation of the pair's bytes
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply the merge across all sequences
            token_sequences = self._merge_pair(token_sequences, best_pair, new_id)

            if verbose and (merge_idx < 10 or (merge_idx + 1) % 50 == 0):
                token_repr = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"  Merge {merge_idx + 1:4d}: "
                    f"({best_pair[0]:4d}, {best_pair[1]:4d}) -> {new_id:4d}  "
                    f"'{token_repr}' (count={best_count})"
                )

        if verbose:
            final_size = sum(len(s) for s in token_sequences)
            print(f"Final corpus size: {final_size} tokens")
            print(f"Vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a sequence of token IDs.

        The encoding algorithm:
        1. Convert text to bytes
        2. Start with each byte as its own token
        3. Apply merge rules in training order (this is crucial!)
        4. Each merge rule: scan sequence, replace all occurrences of the pair

        Applying merges in order guarantees we get the same segmentation that
        training would produce. The first merge is the most common pair in training
        data, so it gets applied first, potentially creating new pairs that later
        merges will handle.

        Args:
            text: String to encode

        Returns:
            List of token IDs

        Complexity: O(n * m) where n = text length in bytes, m = number of merges.
        Production tokenizers optimize this, but the naive approach is correct.
        """
        # Start with individual byte values
        tokens = list(text.encode("utf-8"))

        # Apply each merge rule in order
        # Order matters! Earlier merges have higher priority (more frequent pairs)
        for pair, new_id in self.merges:
            tokens = self._apply_single_merge(tokens, pair, new_id)

        return tokens

    def _apply_single_merge(
        self, tokens: list[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        """Apply one merge rule to a token sequence. Same logic as _merge_pair but for a single sequence."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a string.

        This is the easy direction — just look up the bytes for each token
        and concatenate. BPE is lossless, so decode(encode(text)) == text
        for any valid UTF-8 string that only uses characters seen during training.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded string

        Note: We use errors="replace" for robustness, but in practice BPE on
        well-formed UTF-8 input always produces valid UTF-8 output.
        """
        byte_sequence = b"".join(self.vocab[token_id] for token_id in token_ids)
        return byte_sequence.decode("utf-8", errors="replace")

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
        Calculate how much BPE compresses the text vs raw bytes.

        Ratio = original_bytes / num_tokens. Higher is better — means each
        token represents more bytes on average. A ratio of 3.5 means each
        token represents ~3.5 bytes.

        This is a practical measure of tokenizer quality for a given text.
        """
        original_length = len(text.encode("utf-8"))
        encoded_length = len(self.encode(text))
        return original_length / encoded_length if encoded_length > 0 else 0.0

    def tokenize_verbose(self, text: str) -> list[tuple[int, str]]:
        """
        Encode text and return tokens with their string representations.
        Useful for debugging and understanding how the tokenizer segments text.
        """
        token_ids = self.encode(text)
        return [(tid, self.get_token_string(tid)) for tid in token_ids]


def get_training_corpus() -> str:
    """
    Return a small but diverse training corpus for demonstration.

    In production, BPE is trained on billions of tokens. Here we use a small
    corpus that still demonstrates the algorithm's behavior — common words
    become single tokens, rare words get split into subword units.
    """
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
    print("=" * 70)
    print("BPE Tokenizer from Scratch")
    print("=" * 70)

    # Train the tokenizer
    corpus = get_training_corpus()
    tokenizer = BPETokenizer(vocab_size=350)  # 256 base + 94 merges

    print("\n--- Training Phase ---")
    tokenizer.train(corpus, verbose=True)

    # Show some learned vocabulary
    print("\n--- Learned Vocabulary (merged tokens only) ---")
    merged_tokens = [
        (tid, text) for tid, text in tokenizer.get_vocab_tokens() if tid >= 256
    ]
    print(f"Total merged tokens: {len(merged_tokens)}")
    print("First 20 merged tokens:")
    for tid, text in merged_tokens[:20]:
        print(f"  ID {tid:4d}: '{text}' ({len(tokenizer.vocab[tid])} bytes)")

    # Encode and decode examples
    print("\n--- Encoding/Decoding Examples ---")
    test_strings = [
        "The quick brown fox",
        "natural language processing",
        "blockchain technology",
        "reinforcement learning",
        "tokenization algorithm",
        "xyzzy",  # Rare word — should stay mostly as bytes
    ]

    for text in test_strings:
        tokens = tokenizer.tokenize_verbose(text)
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        ratio = tokenizer.compression_ratio(text)

        print(f"\n  Input:    '{text}'")
        print(f"  Tokens:   {[t[1] for t in tokens]}")
        print(f"  IDs:      {token_ids}")
        print(f"  Decoded:  '{decoded}'")
        print(f"  Ratio:    {ratio:.2f}x compression")
        assert decoded == text, f"Round-trip failed! '{text}' != '{decoded}'"

    print("\n--- Round-trip Verification ---")
    # Verify lossless encoding on the entire training corpus
    encoded = tokenizer.encode(corpus)
    decoded = tokenizer.decode(encoded)
    assert decoded == corpus, "Round-trip failed on training corpus!"
    print(f"Training corpus: {len(corpus.encode('utf-8'))} bytes -> {len(encoded)} tokens")
    print(f"Compression ratio: {tokenizer.compression_ratio(corpus):.2f}x")
    print("Round-trip verification: PASSED")

    # Show how different vocab sizes affect compression
    print("\n--- Vocab Size vs Compression ---")
    test_text = "The transformer architecture revolutionized natural language processing"
    for vs in [270, 300, 350, 400, 500]:
        tok = BPETokenizer(vocab_size=vs)
        tok.train(corpus)
        ratio = tok.compression_ratio(test_text)
        n_tokens = len(tok.encode(test_text))
        print(f"  Vocab={vs:4d}: {n_tokens:3d} tokens, {ratio:.2f}x compression")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
