# Day 76: Byte-Pair Encoding (BPE) Tokenizer from Scratch

## Overview

Build a complete BPE tokenizer from the ground up — the same algorithm that powers tokenization in GPT, Claude, and virtually every modern language model. You'll train a vocabulary from raw text, encode strings into token sequences, and decode them back. This is the invisible but critical first step in every LLM pipeline: before attention heads or feed-forward layers ever see your input, the tokenizer decides how to chop it up.

**Why it matters:** Tokenization shapes everything downstream. A bad tokenizer wastes context window on redundant tokens, struggles with multilingual text, and creates bizarre failure modes (like GPT-3's infamous inability to count letters in words). Understanding BPE deeply lets you reason about model behavior, context limits, and the fundamental tradeoff between vocabulary size and sequence length.

## Core Concepts

### The Problem: From Characters to Tokens

Neural networks need fixed-size vocabulary inputs. The naive approaches fail:

- **Character-level:** Vocabulary is tiny (~256 bytes), but sequences become extremely long. "tokenization" = 12 tokens. Attention is O(n²), so this kills performance.
- **Word-level:** Vocabulary explodes (English alone has 170,000+ words), out-of-vocabulary words are unhandled, and morphological relationships are lost ("run", "running", "runner" share no representation).

We need something in between: a vocabulary of **subword units** that balances vocabulary size against sequence length.

### Byte-Pair Encoding: The Algorithm

BPE was originally a data compression algorithm (Gage, 1994), repurposed for NLP by Sennrich et al. (2016). The core idea is elegant:

1. **Start with bytes.** Every string is a sequence of individual bytes (or characters). Your initial vocabulary is just the 256 possible byte values.

2. **Count pairs.** Scan the entire training corpus and count every adjacent pair of tokens. For example, in "low lower lowest", the pair ('l', 'o') appears 3 times.

3. **Merge the most frequent pair.** Take the pair that appears most often and create a new token by concatenating them. Replace all occurrences of that pair in the corpus with the new token.

4. **Repeat.** Go back to step 2 with the updated corpus. Each iteration adds one new token to the vocabulary. Stop when you reach your desired vocabulary size.

**The math is simple but the intuition is key:** BPE discovers a greedy, bottom-up hierarchy of subword units. Common words like "the" quickly become single tokens. Rare words stay decomposed into meaningful subparts. The word "unhappiness" might become ["un", "happiness"] or ["un", "happ", "iness"] depending on training data — but it's never an unknown token.

### Encoding: Greedy Left-to-Right Matching

Once you have a trained vocabulary with merge rules, encoding new text works by:

1. Split text into individual bytes/characters
2. Apply merge rules in priority order (the order they were learned during training)
3. Each merge rule says "if you see token A followed by token B, combine them into AB"
4. Keep applying until no more merges can be made

This is a **greedy** algorithm — it always applies the highest-priority merge first. This isn't optimal in theory (there might be a better global segmentation), but it's fast and works well in practice.

### Decoding: Trivial Concatenation

Decoding is the easy direction — just concatenate the byte representations of each token. This is why BPE is lossless: every possible byte sequence can be encoded and decoded back to the original.

### Vocabulary Size Tradeoffs

| Vocab Size | Pros | Cons |
|-----------|------|------|
| Small (1K) | Tiny embedding table, generalizes well to rare text | Very long sequences, slow inference |
| Medium (32K) | Good balance, used by many models | Some rare words still split oddly |
| Large (100K+) | Short sequences, common words = single tokens | Huge embedding table, rare tokens poorly trained |

GPT-4 uses ~100K tokens. Claude uses ~100K. Most open models use 32K-64K. The sweet spot depends on your training data size and compute budget.

### Connection to Day 59 (Transformer Attention)

Remember the transformer architecture? The tokenizer is what generates the input sequence that feeds into the embedding layer. Token boundaries directly affect:
- **Positional encodings:** Each token gets one position. A word split into 3 tokens occupies 3 positions.
- **Attention patterns:** Self-attention operates over tokens. Subword splits change what the model can "see" in one step.
- **Context window efficiency:** A 4K context window means 4K tokens, not 4K words. Better tokenization = more content per window.

## Step-by-Step Breakdown

### Step 1: Build the Training Corpus Representation

Convert training text into a list of token sequences (initially individual bytes). We need an efficient data structure because we'll be scanning for pairs millions of times.

**Key decision:** Use a list of lists, where each inner list represents one word/segment. This avoids merging across word boundaries (we don't want "the_cat" to create a "e_c" token).

### Step 2: Count All Adjacent Pairs

Iterate through every token sequence and count the frequency of every adjacent pair. Store in a dictionary mapping (token_a, token_b) → count.

**Why this matters:** The most frequent pair represents the biggest compression opportunity. By always merging the most frequent pair, BPE achieves a greedy approximation of optimal compression.

### Step 3: Merge the Top Pair

Find the pair with the highest count. Create a new token (the concatenation). Scan through all sequences and replace every occurrence of the pair with the new token. Record this merge rule.

**Subtle point:** After merging, new pairs are created. If we merge ('a', 'b') in the sequence ['a', 'b', 'c'], the new sequence ['ab', 'c'] creates the new pair ('ab', 'c') and destroys the old pair ('b', 'c').

### Step 4: Build the Merge Table

After training, you have an ordered list of merge rules: [(pair1, merged_token1), (pair2, merged_token2), ...]. This is your tokenizer's "model" — it's all you need to encode new text.

### Step 5: Implement Encoding

Given new text, split into bytes, then apply merges in order. For each merge rule, scan the token sequence and apply it wherever the pair appears. This is O(n * m) where n is sequence length and m is number of merges — acceptable for most uses.

### Step 6: Implement Decoding

Map each token ID back to its byte string and concatenate. Handle UTF-8 decoding at the end.

## Learning Objectives

- Understand why subword tokenization exists and the vocabulary-sequence length tradeoff
- Implement the BPE training algorithm: pair counting, merging, vocabulary building
- Implement encoding (text → token IDs) and decoding (token IDs → text)
- Reason about how tokenization affects downstream model behavior
- Analyze vocabulary composition and compression ratios

## Going Deeper

- **Regex-based pre-tokenization:** GPT-2/GPT-4 use regex patterns to split text before BPE (separating numbers, punctuation, etc.). This prevents merges across categories.
- **Unigram/SentencePiece:** An alternative to BPE that starts with a large vocabulary and prunes. Used by T5 and LLaMA. Enables proper probabilistic segmentation.
- **Tiktoken:** OpenAI's fast BPE implementation in Rust. Compare your output against it.
- **Special tokens:** How models handle `<|endoftext|>`, `<|im_start|>`, and other control tokens.
- **Multilingual tokenization:** BPE on bytes handles any language, but some languages get "tokenizer tax" — the same meaning takes more tokens in some languages than others.
- **Tokenizer attacks:** Adversarial strings designed to maximize token count and waste context window.
