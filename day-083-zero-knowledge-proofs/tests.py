"""
Day 83: Zero-Knowledge Proofs — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import secrets
from my_solution import (
    is_prime, generate_safe_prime, find_generator, hash_to_int,
    SchnorrParams, SchnorrCommitment, SchnorrResponse,
    SchnorrProver, SchnorrVerifier,
    NonInteractiveProof, fiat_shamir_prove, fiat_shamir_verify,
    prove_or, verify_or,
    ArithmeticCircuit, R1CSConstraint,
    create_cubic_circuit, create_witness_cubic,
)


class TestPrimality(unittest.TestCase):
    """Test mathematical utilities."""

    def test_known_primes(self):
        """is_prime should correctly identify known primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 101, 127]
        for p in primes:
            self.assertTrue(is_prime(p), f"{p} should be prime")

    def test_known_composites(self):
        """is_prime should reject composites."""
        composites = [0, 1, 4, 6, 8, 9, 10, 15, 21, 25, 100, 121]
        for n in composites:
            self.assertFalse(is_prime(n), f"{n} should not be prime")

    def test_safe_prime_generation(self):
        """generate_safe_prime should return p where (p-1)/2 is also prime."""
        p = generate_safe_prime(bits=32)
        q = (p - 1) // 2
        self.assertTrue(is_prime(p), "p should be prime")
        self.assertTrue(is_prime(q), "q = (p-1)/2 should be prime")
        self.assertEqual(p, 2 * q + 1, "p should equal 2q + 1")

    def test_generator(self):
        """find_generator should return a valid generator of order-q subgroup."""
        p = generate_safe_prime(bits=32)
        q = (p - 1) // 2
        g = find_generator(p)
        self.assertNotEqual(g, 1, "Generator should not be 1")
        self.assertNotEqual(g, p - 1, "Generator should not be p-1")
        self.assertEqual(pow(g, q, p), 1, "g^q should be 1 mod p")


class TestHashToInt(unittest.TestCase):
    """Test the hash function used in Fiat-Shamir."""

    def test_deterministic(self):
        """Same inputs should produce same output."""
        h1 = hash_to_int(42, 100, 7, modulus=1000)
        h2 = hash_to_int(42, 100, 7, modulus=1000)
        self.assertEqual(h1, h2)

    def test_different_inputs(self):
        """Different inputs should (almost certainly) produce different outputs."""
        h1 = hash_to_int(1, 2, 3, modulus=2**128)
        h2 = hash_to_int(1, 2, 4, modulus=2**128)
        self.assertNotEqual(h1, h2)

    def test_modulus_bound(self):
        """Output should be less than modulus."""
        for _ in range(20):
            val = secrets.randbelow(10**6)
            h = hash_to_int(val, modulus=100)
            self.assertLess(h, 100)


class TestSchnorrInteractive(unittest.TestCase):
    """Test Schnorr's interactive ZK proof protocol."""

    def setUp(self):
        """Generate parameters for tests."""
        p = generate_safe_prime(bits=64)
        q = (p - 1) // 2
        g = find_generator(p)
        self.params = SchnorrParams(p=p, q=q, g=g)
        self.secret = secrets.randbelow(q - 1) + 1

    def test_honest_proof_verifies(self):
        """A proof with the correct secret should always verify."""
        prover = SchnorrProver(self.params, self.secret)
        verifier = SchnorrVerifier(self.params, prover.y)

        commitment = prover.commit()
        c = verifier.challenge()
        response = prover.respond(c)
        self.assertTrue(verifier.verify(commitment, c, response))

    def test_wrong_secret_fails(self):
        """A proof with a wrong secret should fail verification."""
        wrong_secret = (self.secret + 1) % self.params.q
        prover = SchnorrProver(self.params, wrong_secret)
        public_key = pow(self.params.g, self.secret, self.params.p)  # Original PK
        verifier = SchnorrVerifier(self.params, public_key)

        commitment = prover.commit()
        c = verifier.challenge()
        response = prover.respond(c)
        self.assertFalse(verifier.verify(commitment, c, response))

    def test_multiple_rounds(self):
        """Multiple rounds should all verify for an honest prover."""
        prover = SchnorrProver(self.params, self.secret)
        verifier = SchnorrVerifier(self.params, prover.y)

        for _ in range(5):
            commitment = prover.commit()
            c = verifier.challenge()
            response = prover.respond(c)
            self.assertTrue(verifier.verify(commitment, c, response))


class TestFiatShamir(unittest.TestCase):
    """Test non-interactive Fiat-Shamir proofs."""

    def setUp(self):
        p = generate_safe_prime(bits=64)
        q = (p - 1) // 2
        g = find_generator(p)
        self.params = SchnorrParams(p=p, q=q, g=g)
        self.secret = secrets.randbelow(q - 1) + 1
        self.public_key = pow(g, self.secret, p)

    def test_valid_proof(self):
        """A valid non-interactive proof should verify."""
        proof = fiat_shamir_prove(self.params, self.secret)
        self.assertTrue(fiat_shamir_verify(self.params, self.public_key, proof))

    def test_tampered_response_fails(self):
        """Modifying the response should break the proof."""
        proof = fiat_shamir_prove(self.params, self.secret)
        tampered = NonInteractiveProof(
            commitment=proof.commitment,
            challenge=proof.challenge,
            response=(proof.response + 1) % self.params.q
        )
        self.assertFalse(fiat_shamir_verify(self.params, self.public_key, tampered))

    def test_tampered_commitment_fails(self):
        """Modifying the commitment should break the proof (challenge won't match)."""
        proof = fiat_shamir_prove(self.params, self.secret)
        tampered = NonInteractiveProof(
            commitment=(proof.commitment + 1) % self.params.p,
            challenge=proof.challenge,
            response=proof.response
        )
        self.assertFalse(fiat_shamir_verify(self.params, self.public_key, tampered))


class TestORProof(unittest.TestCase):
    """Test OR-composition proofs."""

    def setUp(self):
        p = generate_safe_prime(bits=64)
        q = (p - 1) // 2
        g = find_generator(p)
        self.params = SchnorrParams(p=p, q=q, g=g)
        self.x1 = secrets.randbelow(q - 1) + 1
        self.x2 = secrets.randbelow(q - 1) + 1
        self.y1 = pow(g, self.x1, p)
        self.y2 = pow(g, self.x2, p)

    def test_or_proof_knowing_first(self):
        """OR-proof should verify when prover knows first secret."""
        proof1, proof2 = prove_or(self.params, self.x1, self.y1, self.y2, which=0)
        self.assertTrue(verify_or(self.params, self.y1, self.y2, proof1, proof2))

    def test_or_proof_knowing_second(self):
        """OR-proof should verify when prover knows second secret."""
        proof1, proof2 = prove_or(self.params, self.x2, self.y1, self.y2, which=1)
        self.assertTrue(verify_or(self.params, self.y1, self.y2, proof1, proof2))


class TestR1CS(unittest.TestCase):
    """Test arithmetic circuit R1CS verification."""

    def test_cubic_circuit_valid_witness(self):
        """Valid witness for x³ + x + 5 should verify."""
        field_prime = 2**61 - 1
        circuit = create_cubic_circuit(field_prime)
        witness = create_witness_cubic(3, field_prime)
        self.assertTrue(circuit.verify_witness(witness))
        self.assertEqual(witness[1], 35)  # 27 + 3 + 5

    def test_cubic_circuit_different_x(self):
        """Circuit should work for different values of x."""
        field_prime = 2**61 - 1
        circuit = create_cubic_circuit(field_prime)
        for x in [0, 1, 2, 5, 10, 100]:
            witness = create_witness_cubic(x, field_prime)
            self.assertTrue(circuit.verify_witness(witness),
                            f"Should verify for x={x}")
            expected_output = (x**3 + x + 5) % field_prime
            self.assertEqual(witness[1], expected_output)

    def test_cubic_circuit_wrong_witness_fails(self):
        """A witness with wrong intermediate values should fail."""
        field_prime = 2**61 - 1
        circuit = create_cubic_circuit(field_prime)
        # Wrong: claiming v1 = 10 when x = 3 (should be 9)
        wrong_witness = [1, 35, 3, 10, 27]
        self.assertFalse(circuit.verify_witness(wrong_witness))

    def test_cubic_circuit_wrong_output_fails(self):
        """A witness with wrong output should fail."""
        field_prime = 2**61 - 1
        circuit = create_cubic_circuit(field_prime)
        wrong_witness = [1, 36, 3, 9, 27]  # output should be 35
        self.assertFalse(circuit.verify_witness(wrong_witness))

    def test_witness_size_check(self):
        """Witness of wrong size should be rejected."""
        field_prime = 2**61 - 1
        circuit = create_cubic_circuit(field_prime)
        self.assertFalse(circuit.verify_witness([1, 35, 3]))  # Too short
        self.assertFalse(circuit.verify_witness([1, 35, 3, 9, 27, 99]))  # Too long


if __name__ == "__main__":
    unittest.main()
