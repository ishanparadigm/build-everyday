"""
Day 002: SHA-256 Hash Implementation from Scratch

A complete implementation of the SHA-256 cryptographic hash function using only
basic Python operations — no hashlib, no cryptography libraries. Every step is
annotated to explain not just what happens, but why.

Reference: NIST FIPS 180-4 (Secure Hash Standard)
"""

from typing import List


# =============================================================================
# Constants
# =============================================================================

# Initial hash values: first 32 bits of the fractional parts of the square roots
# of the first 8 primes (2, 3, 5, 7, 11, 13, 17, 19).
#
# Why primes? Using irrational numbers derived from primes guarantees no hidden
# algebraic structure. This is a "nothing up my sleeve" design — anyone can verify
# these constants independently.
#
# Example: sqrt(2) = 1.4142135... → fractional part = 0.4142135...
# 0.4142135... × 2^32 = 0x6a09e667

H_INITIAL: List[int] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

# Round constants: first 32 bits of the fractional parts of the cube roots
# of the first 64 primes (2, 3, 5, ..., 311).
# Same rationale — verifiable, no backdoor.

K: List[int] = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]


# =============================================================================
# Bitwise helper functions
# =============================================================================

def rotr(n: int, x: int, w: int = 32) -> int:
    """
    Right-rotate x by n positions within a w-bit word.

    Unlike a right shift (which loses bits off the end), rotation wraps them
    around to the top. This is critical for diffusion — it spreads bit influence
    across the entire word without losing information.

    Example: rotr(2, 0b11010011, 8) = 0b11110100
    The bottom 2 bits (11) wrap to the top.
    """
    return ((x >> n) | (x << (w - n))) & ((1 << w) - 1)


def shr(n: int, x: int) -> int:
    """
    Right-shift x by n positions (standard logical shift).

    Unlike rotation, this discards bits — it's a lossy operation that adds
    non-linearity when combined with XOR.
    """
    return x >> n


# SHA-256 defines six logical functions. The "big sigma" (Σ) functions are used
# on the state variables; the "small sigma" (σ) functions are used in the
# message schedule. The rotation amounts were chosen to maximize diffusion
# across the 32-bit word.

def sigma0(x: int) -> int:
    """σ0 — used in message schedule expansion (W[i] computation)."""
    return rotr(7, x) ^ rotr(18, x) ^ shr(3, x)


def sigma1(x: int) -> int:
    """σ1 — used in message schedule expansion (W[i] computation)."""
    return rotr(17, x) ^ rotr(19, x) ^ shr(10, x)


def big_sigma0(x: int) -> int:
    """Σ0 — used in the compression function on variable 'a'."""
    return rotr(2, x) ^ rotr(13, x) ^ rotr(22, x)


def big_sigma1(x: int) -> int:
    """Σ1 — used in the compression function on variable 'e'."""
    return rotr(6, x) ^ rotr(11, x) ^ rotr(25, x)


def ch(e: int, f: int, g: int) -> int:
    """
    Choice function: for each bit position, if e=1, pick f's bit; else pick g's bit.

    This is a 2-to-1 multiplexer controlled by e. It introduces non-linearity
    because the output depends on three inputs in a data-dependent way.
    Equivalent to: (e & f) ^ (~e & g)
    """
    return (e & f) ^ ((~e) & g)


def maj(a: int, b: int, c: int) -> int:
    """
    Majority function: for each bit position, output the majority vote.

    If 2 or 3 of the inputs have a 1, output 1; otherwise 0.
    This is a threshold function — another source of non-linearity.
    Equivalent to: (a & b) ^ (a & c) ^ (b & c)
    """
    return (a & b) ^ (a & c) ^ (b & c)


# Mask for 32-bit arithmetic — all additions in SHA-256 are mod 2^32
MOD32 = 0xFFFFFFFF


def add32(*args: int) -> int:
    """Add multiple values with 32-bit wrapping (mod 2^32)."""
    return sum(args) & MOD32


# =============================================================================
# Preprocessing
# =============================================================================

def pad_message(message: bytes) -> bytes:
    """
    Pad the message according to SHA-256 spec (FIPS 180-4 Section 5.1.1).

    The padding scheme:
    1. Append bit '1' (0x80 byte since we work in bytes)
    2. Append zeros until message length ≡ 448 mod 512 (in bits)
       equivalently: byte length ≡ 56 mod 64
    3. Append the original message length as a 64-bit big-endian integer

    Why 448? Because 448 + 64 (length field) = 512, completing the final block.

    Why include the length? Without it, messages like "abc" and "abc\\x00" could
    be manipulated to produce the same padded output. The length encoding makes
    the padding unambiguous — this is called Merkle-Damgård strengthening.
    """
    msg_len_bits = len(message) * 8

    # Step 1: append the '1' bit (as 0x80 byte)
    message += b'\x80'

    # Step 2: pad with zeros until length ≡ 56 mod 64 bytes
    # We need (56 - current_length) mod 64 zero bytes
    padding_needed = (56 - len(message) % 64) % 64
    message += b'\x00' * padding_needed

    # Step 3: append original length as 64-bit big-endian
    message += msg_len_bits.to_bytes(8, byteorder='big')

    assert len(message) % 64 == 0, "Padded message must be multiple of 512 bits"
    return message


def parse_blocks(padded: bytes) -> List[List[int]]:
    """
    Split padded message into 512-bit (64-byte) blocks,
    each parsed as sixteen 32-bit big-endian words.
    """
    blocks = []
    for i in range(0, len(padded), 64):
        block = []
        for j in range(0, 64, 4):
            # Pack 4 bytes into a 32-bit word (big-endian)
            word = int.from_bytes(padded[i + j:i + j + 4], byteorder='big')
            block.append(word)
        blocks.append(block)
    return blocks


# =============================================================================
# Core SHA-256
# =============================================================================

def sha256(message: bytes, verbose: bool = False) -> str:
    """
    Compute the SHA-256 hash of a message.

    Args:
        message: Input bytes to hash
        verbose: If True, print intermediate state for educational purposes

    Returns:
        64-character hexadecimal string (256 bits)
    """
    # --- Preprocessing ---
    padded = pad_message(message)
    blocks = parse_blocks(padded)

    if verbose:
        print(f"Input: {message!r}")
        print(f"Input length: {len(message)} bytes ({len(message) * 8} bits)")
        print(f"Padded length: {len(padded)} bytes ({len(padded) * 8} bits)")
        print(f"Number of 512-bit blocks: {len(blocks)}")
        print()

    # --- Initialize hash state ---
    # These get updated after processing each block
    H = H_INITIAL.copy()

    # --- Process each block ---
    for block_idx, block in enumerate(blocks):
        if verbose:
            print(f"--- Processing block {block_idx + 1}/{len(blocks)} ---")

        # Step 1: Prepare the message schedule W[0..63]
        # W[0..15] come directly from the block
        W = block.copy()

        # W[16..63] are derived — this is where input bits get mixed extensively.
        # Each new W[i] depends on four previous W values through σ0 and σ1,
        # ensuring that by round 64, every input bit has influenced every state bit.
        for i in range(16, 64):
            W.append(add32(sigma1(W[i - 2]), W[i - 7], sigma0(W[i - 15]), W[i - 16]))

        # Step 2: Initialize working variables from current hash state
        a, b, c, d, e, f, g, h = H

        if verbose and block_idx == 0:
            print(f"Initial state: a={a:08x} e={e:08x}")

        # Step 3: 64 rounds of compression
        # This is the heart of SHA-256. Each round:
        # - Computes T1 using the 'e' group (Σ1, Ch) + round constant + schedule word
        # - Computes T2 using the 'a' group (Σ0, Maj)
        # - Shifts all variables down and injects T1+T2 at the top
        #
        # The shift pattern means each variable passes through all 8 positions
        # over 8 rounds, getting mixed differently at each position.
        for i in range(64):
            T1 = add32(h, big_sigma1(e), ch(e, f, g), K[i], W[i])
            T2 = add32(big_sigma0(a), maj(a, b, c))

            h = g
            g = f
            f = e
            e = add32(d, T1)  # T1 feeds into the 'e' position
            d = c
            c = b
            b = a
            a = add32(T1, T2)  # T1+T2 feeds into the 'a' position

            if verbose and i < 4:
                print(f"  Round {i:2d}: a={a:08x} e={e:08x} T1={T1:08x} T2={T2:08x}")

        if verbose:
            print(f"  ... (rounds 4-63 omitted)")

        # Step 4: Add compressed chunk to hash state (mod 2^32)
        # This addition is crucial: it makes the compression function one-way.
        # Without it, you could simply reverse the 64 rounds. The addition of
        # the original H values means you'd need to know both the input and
        # output of the compression to reverse it.
        H[0] = add32(a, H[0])
        H[1] = add32(b, H[1])
        H[2] = add32(c, H[2])
        H[3] = add32(d, H[3])
        H[4] = add32(e, H[4])
        H[5] = add32(f, H[5])
        H[6] = add32(g, H[6])
        H[7] = add32(h, H[7])

        if verbose:
            print(f"  Hash state: {H[0]:08x} {H[1]:08x} ... {H[7]:08x}")
            print()

    # --- Produce final digest ---
    # Concatenate all eight 32-bit words as big-endian bytes
    digest = ''.join(f'{word:08x}' for word in H)
    return digest


# =============================================================================
# Avalanche effect analysis
# =============================================================================

def avalanche_analysis(msg1: bytes, msg2: bytes) -> None:
    """
    Demonstrate the avalanche effect: a small input change causes ~50% of
    output bits to flip. This is a key property of cryptographic hashes.
    """
    hash1 = sha256(msg1)
    hash2 = sha256(msg2)

    # Convert hex strings to integers for XOR comparison
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    diff = int1 ^ int2

    # Count differing bits
    flipped = bin(diff).count('1')
    total_bits = 256

    print(f"Input 1:  {msg1!r}")
    print(f"Hash 1:   {hash1}")
    print(f"Input 2:  {msg2!r}")
    print(f"Hash 2:   {hash2}")
    print(f"XOR diff: {diff:064x}")
    print(f"Bits flipped: {flipped}/{total_bits} ({flipped/total_bits*100:.1f}%)")
    print(f"Expected: ~128/256 (50.0%) for a good hash function")


# =============================================================================
# Verification against known test vectors
# =============================================================================

def verify() -> bool:
    """
    Verify our implementation against NIST test vectors.
    These are the official test cases from FIPS 180-4.
    """
    test_vectors = [
        # (input_bytes, expected_sha256_hex)
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        ),
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        ),
        (
            b"The quick brown fox jumps over the lazy dog",
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        ),
    ]

    all_passed = True
    for msg, expected in test_vectors:
        result = sha256(msg)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        # Truncate message display for readability
        msg_display = repr(msg) if len(msg) <= 40 else repr(msg[:37]) + "..."
        print(f"  [{status}] SHA-256({msg_display})")
        if result != expected:
            print(f"         Got:      {result}")
            print(f"         Expected: {expected}")

    return all_passed


# =============================================================================
# Main — demonstration and verification
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SHA-256 Implementation from Scratch")
    print("=" * 70)

    # --- Step 1: Verification against NIST test vectors ---
    print("\n1. NIST Test Vector Verification")
    print("-" * 40)
    all_passed = verify()
    print(f"\n{'All tests passed!' if all_passed else 'SOME TESTS FAILED!'}\n")

    # --- Step 2: Verbose walkthrough of a simple hash ---
    print("2. Verbose Hash Computation")
    print("-" * 40)
    result = sha256(b"abc", verbose=True)
    print(f"Final SHA-256 digest: {result}\n")

    # --- Step 3: Avalanche effect demonstration ---
    print("3. Avalanche Effect Analysis")
    print("-" * 40)

    # Change just one character
    print("\nTest A: Single character change")
    avalanche_analysis(b"hello", b"hallo")

    # Change just one bit (lowercase 'a' = 0x61, 'b' = 0x62, differ by 1 bit)
    print("\nTest B: Single bit flip")
    avalanche_analysis(b"a", b"b")

    # Near-identical long messages
    print("\nTest C: One byte difference in longer message")
    avalanche_analysis(
        b"The quick brown fox jumps over the lazy dog",
        b"The quick brown fox jumps over the lazy cog"  # dog → cog
    )

    # --- Step 4: Performance characteristics ---
    print("\n4. Block Count vs Message Length")
    print("-" * 40)
    for length in [0, 1, 55, 56, 64, 100, 1000]:
        msg = b'x' * length
        padded = pad_message(msg)
        n_blocks = len(padded) // 64
        print(f"  {length:4d} bytes → {n_blocks} block(s)")
        # Note: 55 bytes fits in 1 block (55 + 1 + 0 + 8 = 64)
        #        56 bytes needs 2 blocks (56 + 1 = 57, can't fit 8-byte length in same block)

    print("\n" + "=" * 70)
    print("Key takeaway: SHA-256 is built from simple, auditable operations")
    print("(rotate, shift, XOR, AND, add mod 2^32) composed in a way that")
    print("makes the output completely unpredictable without running the")
    print("full computation. No shortcuts exist — that's the whole point.")
    print("=" * 70)
