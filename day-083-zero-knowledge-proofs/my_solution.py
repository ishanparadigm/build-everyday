"""
Day 83: Zero-Knowledge Proofs — Your Implementation

Implement ZK proof systems from scratch:
1. Schnorr's Interactive Protocol
2. Fiat-Shamir Non-Interactive Proofs
3. OR-Composition (prove one of two secrets)
4. Arithmetic Circuit R1CS Verification
5. Range Proofs

Run tests with: python3 -m pytest tests.py -v
"""

import hashlib
import secrets
from typing import Tuple, List, Optional
from dataclasses import dataclass


# =============================================================================
# Mathematical Utilities
# =============================================================================

def is_prime(n: int) -> bool:
    """
    Miller-Rabin primality test.

    Hint: Write n-1 as 2^r * d, then test multiple witnesses.
    If any witness proves compositeness, return False.
    """
    raise NotImplementedError("TODO: implement Miller-Rabin primality test")


def generate_safe_prime(bits: int = 256) -> int:
    """
    Generate a safe prime p where q = (p-1)/2 is also prime.

    Hint: Generate random primes q, then check if p = 2q + 1 is also prime.
    Safe primes prevent small-subgroup attacks.
    """
    raise NotImplementedError("TODO: implement safe prime generation")


def find_generator(p: int) -> int:
    """
    Find a generator of the order-q subgroup of Z*_p where q = (p-1)/2.

    Hint: Pick random h, compute g = h^2 mod p. Check g ≠ 1 and g ≠ p-1.
    Squaring ensures g is in the order-q subgroup.
    """
    raise NotImplementedError("TODO: implement generator finding")


def hash_to_int(*args: int, modulus: int) -> int:
    """
    Hash multiple integers to a single integer mod modulus using SHA-256.

    Hint: Convert each int to bytes, feed to SHA-256, convert digest back to int.
    This is the core of the Fiat-Shamir heuristic.
    """
    raise NotImplementedError("TODO: implement hash-to-integer")


# =============================================================================
# Schnorr's Protocol Data Structures
# =============================================================================

@dataclass
class SchnorrParams:
    """Public parameters for Schnorr's protocol."""
    p: int  # Safe prime
    q: int  # Order of the subgroup: q = (p-1)/2
    g: int  # Generator of the order-q subgroup


@dataclass
class SchnorrCommitment:
    """Prover's first message: the commitment."""
    t: int  # t = g^r mod p


@dataclass
class SchnorrResponse:
    """Prover's response to the verifier's challenge."""
    s: int  # s = r + c*x mod q


@dataclass
class NonInteractiveProof:
    """A non-interactive ZK proof using the Fiat-Shamir heuristic."""
    commitment: int  # t = g^r mod p
    challenge: int   # c = H(g, p, y, t)
    response: int    # s = r + c*x mod q


# =============================================================================
# Schnorr Interactive Protocol
# =============================================================================

class SchnorrProver:
    """
    Prover in Schnorr's protocol. Knows secret x where y = g^x mod p.

    Hint: The 3-step protocol is commit → (receive challenge) → respond.
    CRITICAL: Never reuse the random nonce r across different proofs!
    """

    def __init__(self, params: SchnorrParams, secret: int):
        self.params = params
        self.x = secret
        self.y = pow(params.g, secret, params.p)
        self._r: Optional[int] = None

    def commit(self) -> SchnorrCommitment:
        """
        Step 1: Pick random r, compute t = g^r mod p, return commitment.

        Hint: r should be random in [1, q-1]. Store it for the respond step.
        """
        raise NotImplementedError("TODO: implement commitment")

    def respond(self, challenge: int) -> SchnorrResponse:
        """
        Step 3: Compute s = r + c*x mod q.

        Hint: This linearly encodes the secret x, but the random r masks it.
        Clear r after use to prevent nonce reuse.
        """
        raise NotImplementedError("TODO: implement response")


class SchnorrVerifier:
    """
    Verifier in Schnorr's protocol. Knows public key y = g^x mod p.
    """

    def __init__(self, params: SchnorrParams, public_key: int):
        self.params = params
        self.y = public_key

    def challenge(self) -> int:
        """
        Step 2: Generate random challenge c in [1, q-1].
        """
        raise NotImplementedError("TODO: implement challenge generation")

    def verify(self, commitment: SchnorrCommitment, challenge: int,
               response: SchnorrResponse) -> bool:
        """
        Step 4: Check g^s ≡ t * y^c (mod p).

        Hint: LHS = g^s mod p, RHS = (t * y^c) mod p. They should be equal.
        """
        raise NotImplementedError("TODO: implement verification")


# =============================================================================
# Fiat-Shamir Non-Interactive Proof
# =============================================================================

def fiat_shamir_prove(params: SchnorrParams, secret: int) -> NonInteractiveProof:
    """
    Create a non-interactive proof using the Fiat-Shamir transform.

    Hint: Same as interactive Schnorr, but compute c = H(g, p, y, t) instead
    of receiving it from the verifier. This makes the proof non-interactive.
    """
    raise NotImplementedError("TODO: implement Fiat-Shamir prove")


def fiat_shamir_verify(params: SchnorrParams, public_key: int,
                       proof: NonInteractiveProof) -> bool:
    """
    Verify a non-interactive Schnorr proof.

    Hint: Recompute c = H(g, p, y, t) and check it matches proof.challenge.
    Then verify the standard Schnorr equation g^s = t * y^c.
    """
    raise NotImplementedError("TODO: implement Fiat-Shamir verify")


# =============================================================================
# OR-Composition
# =============================================================================

def prove_or(params: SchnorrParams, secret: int, y1: int, y2: int,
             which: int) -> Tuple[NonInteractiveProof, NonInteractiveProof]:
    """
    Prove knowledge of x1 OR x2 without revealing which.

    Hint: For the KNOWN secret, run the real protocol.
    For the UNKNOWN one, SIMULATE a valid-looking proof:
    - Pick random s_fake, c_fake
    - Compute t_fake = g^s_fake * y_fake^(-c_fake)
    Then bind with: c_real + c_fake = H(t1, t2) mod q
    """
    raise NotImplementedError("TODO: implement OR-proof")


def verify_or(params: SchnorrParams, y1: int, y2: int,
              proof1: NonInteractiveProof, proof2: NonInteractiveProof) -> bool:
    """
    Verify an OR-composed proof.

    Hint: Check c1 + c2 = H(g, p, y1, y2, t1, t2) mod q,
    then verify both Schnorr equations individually.
    """
    raise NotImplementedError("TODO: implement OR-proof verification")


# =============================================================================
# Arithmetic Circuit R1CS
# =============================================================================

@dataclass
class R1CSConstraint:
    """A single R1CS constraint: (a · w) * (b · w) = (c · w)"""
    a: List[Tuple[int, int]]  # Sparse vector a: list of (index, coefficient)
    b: List[Tuple[int, int]]  # Sparse vector b
    c: List[Tuple[int, int]]  # Sparse vector c


class ArithmeticCircuit:
    """
    Arithmetic circuit as R1CS (Rank-1 Constraint System).

    Hint: Each constraint is (a·w) * (b·w) = (c·w) where w is the witness vector.
    The witness contains [1, public_outputs..., private_inputs..., intermediates...].
    """

    def __init__(self, field_prime: int, num_public: int, num_private: int):
        self.p = field_prime
        self.num_public = num_public
        self.num_private = num_private
        self.witness_size = 1 + num_public + num_private
        self.constraints: List[R1CSConstraint] = []

    def add_constraint(self, a: List[Tuple[int, int]], b: List[Tuple[int, int]],
                       c: List[Tuple[int, int]]) -> None:
        """Add a constraint (a·w) * (b·w) = (c·w) mod p."""
        self.constraints.append(R1CSConstraint(a=a, b=b, c=c))

    def _dot(self, sparse_vec: List[Tuple[int, int]], witness: List[int]) -> int:
        """
        Compute sparse dot product with witness mod p.

        Hint: Sum coeff * witness[idx] for each (idx, coeff) in sparse_vec.
        """
        raise NotImplementedError("TODO: implement sparse dot product")

    def verify_witness(self, witness: List[int]) -> bool:
        """
        Verify that a witness satisfies ALL constraints.

        Hint: For each constraint, compute a·w, b·w, c·w and check
        (a·w) * (b·w) ≡ (c·w) (mod p).
        """
        raise NotImplementedError("TODO: implement witness verification")


def create_cubic_circuit(field_prime: int) -> ArithmeticCircuit:
    """
    Create R1CS for x³ + x + 5 = output.

    Witness layout: [1, output, x, v1, v2]
    Where: v1 = x*x, v2 = v1*x, output = v2 + x + 5

    Hint: You need 3 constraints:
    1. x * x = v1
    2. v1 * x = v2
    3. (v2 + x + 5) * 1 = output
    """
    raise NotImplementedError("TODO: implement cubic circuit")


def create_witness_cubic(x: int, field_prime: int) -> List[int]:
    """
    Compute the full witness for x³ + x + 5.

    Hint: Compute each intermediate value and pack into
    [1, output, x, v1, v2].
    """
    raise NotImplementedError("TODO: implement witness generation")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    print("Testing your ZK proof implementations...\n")

    # Test 1: Primality and safe prime generation
    print("1. Testing safe prime generation...")
    try:
        p = generate_safe_prime(bits=32)
        q = (p - 1) // 2
        assert is_prime(p), "p should be prime"
        assert is_prime(q), "q should be prime"
        print(f"   Safe prime: p={p}, q={q} ✓")
    except NotImplementedError:
        print("   Not yet implemented")

    # Test 2: Schnorr interactive proof
    print("\n2. Testing Schnorr interactive proof...")
    try:
        p = generate_safe_prime(bits=64)
        params = SchnorrParams(p=p, q=(p-1)//2, g=find_generator(p))
        secret = secrets.randbelow(params.q - 1) + 1
        prover = SchnorrProver(params, secret)
        verifier = SchnorrVerifier(params, prover.y)

        commitment = prover.commit()
        c = verifier.challenge()
        response = prover.respond(c)
        assert verifier.verify(commitment, c, response), "Valid proof should verify"
        print(f"   Interactive proof verified ✓")
    except NotImplementedError:
        print("   Not yet implemented")

    # Test 3: Fiat-Shamir non-interactive proof
    print("\n3. Testing Fiat-Shamir non-interactive proof...")
    try:
        proof = fiat_shamir_prove(params, secret)
        public_key = pow(params.g, secret, params.p)
        assert fiat_shamir_verify(params, public_key, proof), "Proof should verify"
        print(f"   Non-interactive proof verified ✓")
    except NotImplementedError:
        print("   Not yet implemented")

    # Test 4: OR-proof
    print("\n4. Testing OR-composition proof...")
    try:
        x1 = secrets.randbelow(params.q - 1) + 1
        y1 = pow(params.g, x1, params.p)
        y2 = pow(params.g, secrets.randbelow(params.q - 1) + 1, params.p)
        proof1, proof2 = prove_or(params, x1, y1, y2, which=0)
        assert verify_or(params, y1, y2, proof1, proof2), "OR-proof should verify"
        print(f"   OR-proof verified ✓")
    except NotImplementedError:
        print("   Not yet implemented")

    # Test 5: R1CS circuit
    print("\n5. Testing R1CS arithmetic circuit...")
    try:
        field_prime = 2**61 - 1
        circuit = create_cubic_circuit(field_prime)
        witness = create_witness_cubic(3, field_prime)
        assert circuit.verify_witness(witness), "Valid witness should verify"
        assert witness[1] == 35, "3³ + 3 + 5 = 35"
        print(f"   R1CS verification passed ✓")
    except NotImplementedError:
        print("   Not yet implemented")

    print("\nAll implemented tests passed!")
