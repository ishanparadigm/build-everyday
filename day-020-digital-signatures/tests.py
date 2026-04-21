"""
Day 020: ECDSA Digital Signatures — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import secrets
import unittest

from my_solution import (
    A, B, G, GX, GY, INFINITY, N, P,
    Point, generate_keypair, hash_message, mod_inverse,
    nonce_reuse_attack, point_add, scalar_multiply, sign, verify,
)


class TestModInverse(unittest.TestCase):
    """Test modular multiplicative inverse."""

    def test_small_values(self):
        """a * a⁻¹ ≡ 1 (mod m) for small values."""
        for a, m in [(3, 7), (5, 11), (7, 13), (10, 17)]:
            inv = mod_inverse(a, m)
            self.assertEqual((a * inv) % m, 1)

    def test_large_prime_field(self):
        """Works with secp256k1's prime P."""
        a = 12345678901234567890
        inv = mod_inverse(a, P)
        self.assertEqual((a * inv) % P, 1)

    def test_negative_input(self):
        """Handles negative values correctly."""
        inv = mod_inverse(-3, 7)
        self.assertEqual((-3 * inv) % 7, 1)


class TestPointArithmetic(unittest.TestCase):
    """Test elliptic curve point operations."""

    def test_generator_on_curve(self):
        """G must satisfy y² = x³ + 7 (mod P)."""
        self.assertEqual(pow(GY, 2, P), (pow(GX, 3, P) + B) % P)

    def test_add_infinity(self):
        """P + O = P (identity element)."""
        self.assertEqual(point_add(G, INFINITY), G)
        self.assertEqual(point_add(INFINITY, G), G)

    def test_add_inverse(self):
        """P + (-P) = O."""
        neg_g = Point(GX, (-GY) % P)
        result = point_add(G, neg_g)
        self.assertTrue(result.is_infinity)

    def test_doubling_on_curve(self):
        """2G should be on the curve."""
        two_g = point_add(G, G)
        self.assertFalse(two_g.is_infinity)
        lhs = pow(two_g.y, 2, P)
        rhs = (pow(two_g.x, 3, P) + B) % P
        self.assertEqual(lhs, rhs)

    def test_scalar_multiply_identity(self):
        """1 * G = G."""
        self.assertEqual(scalar_multiply(1, G), G)

    def test_scalar_multiply_matches_addition(self):
        """k*G computed via scalar_multiply should match iterated addition."""
        two_g_add = point_add(G, G)
        two_g_mul = scalar_multiply(2, G)
        self.assertEqual(two_g_add, two_g_mul)

        three_g = point_add(two_g_add, G)
        self.assertEqual(scalar_multiply(3, G), three_g)

    def test_scalar_multiply_order(self):
        """N * G = O (generator has order N)."""
        result = scalar_multiply(N, G)
        self.assertTrue(result.is_infinity)


class TestECDSA(unittest.TestCase):
    """Test the full ECDSA sign/verify flow."""

    def setUp(self):
        self.private_key, self.public_key = generate_keypair()

    def test_sign_verify_valid(self):
        """A valid signature should verify."""
        msg = "Test message"
        sig = sign(msg, self.private_key)
        self.assertTrue(verify(msg, sig, self.public_key))

    def test_tampered_message_rejected(self):
        """A signature should not verify against a different message."""
        sig = sign("original", self.private_key)
        self.assertFalse(verify("tampered", sig, self.public_key))

    def test_wrong_key_rejected(self):
        """A signature should not verify with a different public key."""
        sig = sign("test", self.private_key)
        _, other_pub = generate_keypair()
        self.assertFalse(verify("test", sig, other_pub))

    def test_empty_message(self):
        """Empty strings should be signable and verifiable."""
        sig = sign("", self.private_key)
        self.assertTrue(verify("", sig, self.public_key))

    def test_signature_components_in_range(self):
        """r and s must be in [1, N-1]."""
        sig = sign("range check", self.private_key)
        r, s = sig
        self.assertGreaterEqual(r, 1)
        self.assertLess(r, N)
        self.assertGreaterEqual(s, 1)
        self.assertLess(s, N)

    def test_invalid_signature_range(self):
        """Signatures with r or s outside [1, N-1] should be rejected."""
        self.assertFalse(verify("test", (0, 1), self.public_key))
        self.assertFalse(verify("test", (1, 0), self.public_key))
        self.assertFalse(verify("test", (N, 1), self.public_key))


class TestNonceReuseAttack(unittest.TestCase):
    """Test that nonce reuse leaks the private key."""

    def test_recover_private_key(self):
        """Reusing a nonce across two signatures should leak the private key."""
        priv, _ = generate_keypair()
        k = secrets.randbelow(N - 1) + 1

        msg1, msg2 = "message one", "message two"
        z1, z2 = hash_message(msg1), hash_message(msg2)

        R = scalar_multiply(k, G)
        r = R.x % N
        k_inv = mod_inverse(k, N)
        s1 = (k_inv * (z1 + r * priv)) % N
        s2 = (k_inv * (z2 + r * priv)) % N

        recovered = nonce_reuse_attack(msg1, (r, s1), msg2, (r, s2))
        self.assertEqual(recovered, priv)


if __name__ == "__main__":
    unittest.main()
