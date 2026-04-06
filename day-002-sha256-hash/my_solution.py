"""
Day 002: SHA-256 Hash Implementation from Scratch

Implement the SHA-256 cryptographic hash function using only basic Python
operations -- no hashlib, no cryptography libraries.

Reference: NIST FIPS 180-4 (Secure Hash Standard)
"""

from typing import List


# =============================================================================
# Constants (provided -- do NOT modify)
# =============================================================================

# Initial hash values: first 32 bits of the fractional parts of the square
# roots of the first 8 primes (2, 3, 5, 7, 11, 13, 17, 19).
H_INITIAL: List[int] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

# Round constants: first 32 bits of the fractional parts of the cube roots
# of the first 64 primes.
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

# Mask for 32-bit arithmetic
MOD32 = 0xFFFFFFFF


# =============================================================================
# Bitwise Helper Functions
# =============================================================================

def rotr(n: int, x: int, w: int = 32) -> int:
    """
    Right-rotate x by n positions within a w-bit word.

    Unlike a right shift, rotation wraps bits around to the top.
    Example: rotr(2, 0b11010011, 8) = 0b11110100
    """
    # Hint: combine a right shift and a left shift, then mask to w bits
    raise NotImplementedError("TODO: implement this")


def shr(n: int, x: int) -> int:
    """
    Right-shift x by n positions (standard logical shift).

    Unlike rotation, this discards bits -- it's a lossy operation.
    """
    # Hint: this is just the >> operator
    raise NotImplementedError("TODO: implement this")


def sigma0(x: int) -> int:
    """Small sigma 0 -- used in message schedule expansion."""
    # Hint: rotr(7, x) ^ rotr(18, x) ^ shr(3, x)
    raise NotImplementedError("TODO: implement this")


def sigma1(x: int) -> int:
    """Small sigma 1 -- used in message schedule expansion."""
    # Hint: rotr(17, x) ^ rotr(19, x) ^ shr(10, x)
    raise NotImplementedError("TODO: implement this")


def big_sigma0(x: int) -> int:
    """Big Sigma 0 -- used in the compression function on variable 'a'."""
    # Hint: rotr(2, x) ^ rotr(13, x) ^ rotr(22, x)
    raise NotImplementedError("TODO: implement this")


def big_sigma1(x: int) -> int:
    """Big Sigma 1 -- used in the compression function on variable 'e'."""
    # Hint: rotr(6, x) ^ rotr(11, x) ^ rotr(25, x)
    raise NotImplementedError("TODO: implement this")


def ch(e: int, f: int, g: int) -> int:
    """
    Choice function: for each bit, if e=1 pick f's bit, else pick g's bit.

    This is a 2-to-1 multiplexer controlled by e.
    """
    # Hint: (e & f) ^ (~e & g)
    raise NotImplementedError("TODO: implement this")


def maj(a: int, b: int, c: int) -> int:
    """
    Majority function: for each bit position, output the majority vote.

    If 2 or 3 of the inputs have a 1, output 1; otherwise 0.
    """
    # Hint: (a & b) ^ (a & c) ^ (b & c)
    raise NotImplementedError("TODO: implement this")


def add32(*args: int) -> int:
    """Add multiple values with 32-bit wrapping (mod 2^32)."""
    # Hint: sum all args and mask with MOD32
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Preprocessing
# =============================================================================

def pad_message(message: bytes) -> bytes:
    """
    Pad the message according to SHA-256 spec (FIPS 180-4 Section 5.1.1).

    Steps:
    1. Append bit '1' (0x80 byte)
    2. Append zeros until length == 56 mod 64 (in bytes)
    3. Append original message length as 64-bit big-endian integer

    The result length must be a multiple of 64 bytes (512 bits).
    """
    # Hint: track original length in BITS before modifying the message
    # Hint: use int.to_bytes(8, byteorder='big') for the length field
    raise NotImplementedError("TODO: implement this")


def parse_blocks(padded: bytes) -> List[List[int]]:
    """
    Split padded message into 512-bit (64-byte) blocks, each parsed
    as sixteen 32-bit big-endian words.

    Returns:
        List of blocks, where each block is a list of 16 ints.
    """
    # Hint: iterate in chunks of 64 bytes
    # Hint: use int.from_bytes(..., byteorder='big') to parse 4-byte words
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Core SHA-256
# =============================================================================

def sha256(message: bytes, verbose: bool = False) -> str:
    """
    Compute the SHA-256 hash of a message.

    Algorithm overview:
    1. Pad and parse the message into 512-bit blocks
    2. For each block:
       a. Expand 16 message words into 64-word schedule using sigma0/sigma1
       b. Run 64 rounds of compression using Ch, Maj, big_sigma0/1, K constants
       c. Add compressed state back to running hash (Merkle-Damgard)
    3. Concatenate the 8 hash words into a hex string

    Args:
        message: Input bytes to hash
        verbose: If True, print intermediate state

    Returns:
        64-character hexadecimal string (256 bits)
    """
    # Hint: start with H = H_INITIAL.copy()
    # Hint: message schedule W has 64 entries: W[0..15] from block, W[16..63] derived
    # Hint: 64 rounds update working variables a..h using T1, T2
    # Hint: after each block, add working variables back to H (mod 2^32)
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SHA-256 Implementation from Scratch")
    print("=" * 70)

    # Test against known NIST vectors
    test_vectors = [
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ]

    for msg, expected in test_vectors:
        result = sha256(msg)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] SHA-256({msg!r})")
        print(f"  Got:      {result}")
        print(f"  Expected: {expected}")
