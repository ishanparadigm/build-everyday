"""
Day 002: SHA-256 -- Test Suite

Run with:
    python3 -m pytest tests.py -v
    python3 tests.py
"""

import unittest
from my_solution import (
    rotr,
    shr,
    sigma0,
    sigma1,
    big_sigma0,
    big_sigma1,
    ch,
    maj,
    add32,
    pad_message,
    parse_blocks,
    sha256,
)


class TestBitwiseHelpers(unittest.TestCase):
    """Tests for low-level bitwise operations."""

    def test_rotr_basic(self):
        """Right-rotate known value."""
        # rotr(2, 0b11010011, 8) should wrap bottom 2 bits to top
        result = rotr(2, 0b11010011, 8)
        self.assertEqual(result, 0b11110100)

    def test_rotr_full_rotation(self):
        """Rotating by the word size returns the original value."""
        val = 0xDEADBEEF
        self.assertEqual(rotr(32, val, 32), val)

    def test_rotr_zero_rotation(self):
        val = 0x12345678
        self.assertEqual(rotr(0, val, 32), val)

    def test_shr_basic(self):
        self.assertEqual(shr(4, 0xFF), 0x0F)
        self.assertEqual(shr(1, 0b110), 0b011)

    def test_ch_function(self):
        # When e bit is 1, pick f; when e bit is 0, pick g
        # e=0xFF, f=0xAA, g=0x55 -> all e bits are 1, so result = f = 0xAA
        self.assertEqual(ch(0xFF, 0xAA, 0x55), 0xAA)
        # e=0x00 -> all e bits are 0, so result = g
        self.assertEqual(ch(0x00, 0xAA, 0x55), 0x55)

    def test_maj_function(self):
        # All same -> that value
        self.assertEqual(maj(0xFF, 0xFF, 0xFF), 0xFF)
        self.assertEqual(maj(0x00, 0x00, 0x00), 0x00)
        # Two of three are 0xFF -> majority is 0xFF
        self.assertEqual(maj(0xFF, 0xFF, 0x00), 0xFF)

    def test_add32_wraps_at_2_32(self):
        """Addition should wrap at 2^32."""
        self.assertEqual(add32(0xFFFFFFFF, 1), 0)
        self.assertEqual(add32(0xFFFFFFFF, 2), 1)

    def test_add32_multiple_args(self):
        self.assertEqual(add32(1, 2, 3), 6)


class TestPreprocessing(unittest.TestCase):
    """Tests for message padding and block parsing."""

    def test_pad_message_length_multiple_of_64(self):
        """Padded message must be a multiple of 64 bytes."""
        for length in [0, 1, 55, 56, 64, 100]:
            msg = b'x' * length
            padded = pad_message(msg)
            self.assertEqual(len(padded) % 64, 0,
                             f"Failed for input length {length}: padded length {len(padded)}")

    def test_pad_message_short(self):
        """Empty message should pad to exactly 64 bytes."""
        padded = pad_message(b"")
        self.assertEqual(len(padded), 64)

    def test_pad_message_boundary(self):
        """55 bytes should fit in one block; 56 bytes needs two blocks."""
        self.assertEqual(len(pad_message(b'x' * 55)), 64)
        self.assertEqual(len(pad_message(b'x' * 56)), 128)

    def test_parse_blocks_count(self):
        """Number of blocks should match padded length / 64."""
        padded = pad_message(b"abc")
        blocks = parse_blocks(padded)
        self.assertEqual(len(blocks), len(padded) // 64)

    def test_parse_blocks_word_count(self):
        """Each block should have 16 words."""
        padded = pad_message(b"abc")
        blocks = parse_blocks(padded)
        for block in blocks:
            self.assertEqual(len(block), 16)


class TestSHA256(unittest.TestCase):
    """Tests for the full SHA-256 hash function."""

    def test_nist_vector_abc(self):
        """NIST test vector: SHA-256('abc')."""
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        self.assertEqual(sha256(b"abc"), expected)

    def test_nist_vector_empty(self):
        """NIST test vector: SHA-256('')."""
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        self.assertEqual(sha256(b""), expected)

    def test_nist_vector_448bit(self):
        """NIST test vector: two-block message."""
        msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        expected = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        self.assertEqual(sha256(msg), expected)

    def test_nist_vector_fox(self):
        msg = b"The quick brown fox jumps over the lazy dog"
        expected = "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        self.assertEqual(sha256(msg), expected)

    def test_avalanche_property(self):
        """A single-character change should flip roughly 50% of output bits."""
        hash1 = sha256(b"hello")
        hash2 = sha256(b"hallo")

        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        diff = int1 ^ int2
        flipped = bin(diff).count('1')

        # Expect roughly 128/256 bits to flip; accept a wide range
        self.assertGreater(flipped, 90, "Too few bits flipped -- weak diffusion")
        self.assertLess(flipped, 170, "Too many bits flipped -- suspicious")

    def test_deterministic(self):
        """Same input should always produce the same output."""
        self.assertEqual(sha256(b"test"), sha256(b"test"))


if __name__ == "__main__":
    unittest.main()
