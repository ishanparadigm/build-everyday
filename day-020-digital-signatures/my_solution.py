"""
Day 020: Digital Signatures (ECDSA) — Your Implementation

Implement the Elliptic Curve Digital Signature Algorithm from scratch.
This is the cryptographic primitive securing Bitcoin and Ethereum transactions.

Hints:
- All arithmetic must be done modulo P (for point ops) or N (for signature math)
- "Division" in modular arithmetic = multiplication by the modular inverse
- The double-and-add algorithm is the key to efficient scalar multiplication
- The point at infinity is the identity element (like 0 for addition)
"""

import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional, Tuple


# secp256k1 curve parameters (same as Bitcoin/Ethereum)
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
A = 0
B = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def mod_inverse(a: int, m: int) -> int:
    """
    Compute the modular multiplicative inverse of a modulo m.
    Find x such that a*x ≡ 1 (mod m).

    Hint: Use the extended Euclidean algorithm.
    Alternative: Fermat's little theorem gives a^(m-2) mod m when m is prime.
    """
    raise NotImplementedError("TODO: implement this")


@dataclass
class Point:
    """
    A point on the secp256k1 elliptic curve.
    x=None, y=None represents the point at infinity (identity element).
    """
    x: Optional[int]
    y: Optional[int]

    @property
    def is_infinity(self) -> bool:
        return self.x is None and self.y is None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y


INFINITY = Point(None, None)
G = Point(GX, GY)


def point_add(p1: Point, p2: Point) -> Point:
    """
    Add two points on the secp256k1 curve.

    Handle these cases:
    1. Either point is infinity → return the other
    2. Same x, different y → return infinity (they're inverses)
    3. Same point (doubling) → use tangent slope: (3x²+a) / (2y)
    4. Different points → use secant slope: (y2-y1) / (x2-x1)

    Hint: "Division" here means multiplying by mod_inverse.
    All results should be reduced mod P.
    """
    raise NotImplementedError("TODO: implement this")


def scalar_multiply(k: int, point: Point) -> Point:
    """
    Compute k * point using the double-and-add algorithm.

    Hint: Think of k in binary. Scan bits from LSB to MSB.
    For each bit: if it's 1, add the current addend to the result.
    Then double the addend for the next bit position.

    Don't forget to reduce k modulo N first.
    """
    raise NotImplementedError("TODO: implement this")


def hash_message(message: str) -> int:
    """
    Hash a message using SHA-256 and return as an integer.

    Hint: hashlib.sha256(...).digest() gives bytes.
    int.from_bytes(digest, 'big') converts to integer.
    Truncate to the bit-length of N if needed.
    """
    raise NotImplementedError("TODO: implement this")


def generate_keypair() -> Tuple[int, Point]:
    """
    Generate an ECDSA key pair.

    Private key: random integer d in [1, N-1]
    Public key: Q = d * G

    Hint: Use secrets.randbelow() for cryptographic randomness.
    """
    raise NotImplementedError("TODO: implement this")


def sign(message: str, private_key: int) -> Tuple[int, int]:
    """
    Sign a message using ECDSA.

    Steps:
    1. z = hash_message(message)
    2. Pick random nonce k in [1, N-1]
    3. R = k * G, r = R.x mod N (retry if r == 0)
    4. s = k⁻¹ * (z + r * private_key) mod N (retry if s == 0)
    5. Return (r, s)

    Hint: The nonce k MUST be random and unique per signature.
    """
    raise NotImplementedError("TODO: implement this")


def verify(message: str, signature: Tuple[int, int], public_key: Point) -> bool:
    """
    Verify an ECDSA signature.

    Steps:
    1. Check r, s in [1, N-1]
    2. z = hash_message(message)
    3. w = s⁻¹ mod N
    4. u1 = z*w mod N, u2 = r*w mod N
    5. R' = u1*G + u2*public_key
    6. Valid iff R'.x mod N == r

    Hint: The algebra works because substituting s = k⁻¹(z+rd)
    makes u1*G + u2*Q simplify back to k*G = R.
    """
    raise NotImplementedError("TODO: implement this")


def nonce_reuse_attack(
    msg1: str, sig1: Tuple[int, int],
    msg2: str, sig2: Tuple[int, int]
) -> Optional[int]:
    """
    Extract the private key from two signatures that reused a nonce.

    Given s₁ = k⁻¹(z₁ + r·d) and s₂ = k⁻¹(z₂ + r·d):
    1. k = (z₁ - z₂) · (s₁ - s₂)⁻¹ mod N
    2. d = (s₁·k - z₁) · r⁻¹ mod N

    Hint: The signatures must share the same r value (same nonce → same R).
    """
    raise NotImplementedError("TODO: implement this")


if __name__ == "__main__":
    # Test your implementation step by step

    # 1. Test mod_inverse
    print("Testing mod_inverse...")
    assert (mod_inverse(3, 7) * 3) % 7 == 1
    print("  mod_inverse works!")

    # 2. Test point operations — verify G is on the curve
    print("\nVerifying generator point G is on secp256k1...")
    lhs = pow(GY, 2, P)
    rhs = (pow(GX, 3, P) + B) % P
    assert lhs == rhs, "G is not on the curve!"
    print("  G is on the curve!")

    # 3. Test point addition — G + G should give 2G
    print("\nTesting point doubling (G + G)...")
    two_g = point_add(G, G)
    assert not two_g.is_infinity
    # Verify 2G is on the curve
    lhs = pow(two_g.y, 2, P)
    rhs = (pow(two_g.x, 3, P) + B) % P
    assert lhs == rhs, "2G is not on the curve!"
    print(f"  2G is on the curve!")

    # 4. Test scalar multiplication
    print("\nTesting scalar multiplication...")
    also_two_g = scalar_multiply(2, G)
    assert two_g == also_two_g, "2*G should equal G+G"
    print("  scalar_multiply(2, G) == point_add(G, G) ✓")

    # 5. Test key generation, signing, and verification
    print("\nTesting full ECDSA flow...")
    priv, pub = generate_keypair()
    msg = "Hello, ECDSA!"
    sig = sign(msg, priv)
    assert verify(msg, sig, pub), "Valid signature should verify!"
    print("  Sign + Verify: ✓")

    assert not verify("Tampered!", sig, pub), "Tampered message should NOT verify!"
    print("  Tampered message rejected: ✓")

    # 6. Test nonce reuse attack
    print("\nTesting nonce reuse attack...")
    k = secrets.randbelow(N - 1) + 1
    z1 = hash_message("msg1")
    z2 = hash_message("msg2")
    R = scalar_multiply(k, G)
    r = R.x % N
    k_inv = mod_inverse(k, N)
    s1 = (k_inv * (z1 + r * priv)) % N
    s2 = (k_inv * (z2 + r * priv)) % N
    recovered = nonce_reuse_attack("msg1", (r, s1), "msg2", (r, s2))
    assert recovered == priv, "Should recover the private key!"
    print("  Private key recovered from nonce reuse: ✓")

    print("\nAll tests passed!")
