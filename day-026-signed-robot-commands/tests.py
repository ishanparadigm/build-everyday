"""
Day 026: Tests for Cryptographically Signed Robot Command Protocol

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import math
import time
import unittest

from my_solution import (
    ECPoint,
    CommandType,
    MerkleBatchCommand,
    Operator,
    PIDController,
    Robot,
    RobotCommand,
    RobotState,
    SignedCommand,
    build_merkle_tree,
    ecdsa_sign,
    ecdsa_verify,
    generate_keypair,
    get_merkle_proof,
    merkle_hash_internal,
    merkle_hash_leaf,
    modinv,
    sha256,
    verify_merkle_proof,
    G, N,
)


class TestECDSA(unittest.TestCase):
    """Test elliptic curve cryptography primitives."""

    def test_key_generation(self):
        """Keys generated from the same seed must be identical."""
        priv1, pub1 = generate_keypair(b"test-seed")
        priv2, pub2 = generate_keypair(b"test-seed")
        self.assertEqual(priv1, priv2)
        self.assertEqual(pub1, pub2)

    def test_different_seeds_different_keys(self):
        """Different seeds produce different key pairs."""
        priv1, _ = generate_keypair(b"seed-a")
        priv2, _ = generate_keypair(b"seed-b")
        self.assertNotEqual(priv1, priv2)

    def test_sign_and_verify(self):
        """A valid signature must verify correctly."""
        priv, pub = generate_keypair(b"sign-test")
        msg = sha256(b"hello world")
        r, s = ecdsa_sign(priv, msg, b"k-seed")
        self.assertTrue(ecdsa_verify(pub, msg, r, s))

    def test_wrong_key_rejects(self):
        """Signature verified with wrong key must fail."""
        priv1, pub1 = generate_keypair(b"signer")
        _, pub2 = generate_keypair(b"imposter")
        msg = sha256(b"message")
        r, s = ecdsa_sign(priv1, msg, b"k")
        self.assertFalse(ecdsa_verify(pub2, msg, r, s))

    def test_tampered_message_rejects(self):
        """Signature on tampered message must fail."""
        priv, pub = generate_keypair(b"tamper-test")
        msg = sha256(b"original")
        r, s = ecdsa_sign(priv, msg, b"k")
        tampered = sha256(b"tampered")
        self.assertFalse(ecdsa_verify(pub, tampered, r, s))


class TestMerkleTree(unittest.TestCase):
    """Test Merkle tree construction and verification."""

    def test_single_leaf(self):
        """Tree with one leaf should still produce a valid root and proof."""
        root, layers = build_merkle_tree([b"only-leaf"])
        self.assertIsInstance(root, bytes)
        self.assertEqual(len(root), 32)
        proof = get_merkle_proof(layers, 0)
        self.assertTrue(verify_merkle_proof(b"only-leaf", proof, root))

    def test_multiple_leaves(self):
        """Each leaf should have a valid inclusion proof."""
        leaves = [f"leaf-{i}".encode() for i in range(5)]
        root, layers = build_merkle_tree(leaves)
        for i, leaf in enumerate(leaves):
            proof = get_merkle_proof(layers, i)
            self.assertTrue(verify_merkle_proof(leaf, proof, root))

    def test_wrong_leaf_rejects(self):
        """Proof for a different leaf must fail verification."""
        leaves = [b"a", b"b", b"c", b"d"]
        root, layers = build_merkle_tree(leaves)
        proof = get_merkle_proof(layers, 0)
        self.assertFalse(verify_merkle_proof(b"FAKE", proof, root))

    def test_tampered_root_rejects(self):
        """Proof against a tampered root must fail."""
        leaves = [b"x", b"y"]
        root, layers = build_merkle_tree(leaves)
        proof = get_merkle_proof(layers, 0)
        fake_root = sha256(b"fake-root")
        self.assertFalse(verify_merkle_proof(b"x", proof, fake_root))


class TestCommandProtocol(unittest.TestCase):
    """Test command signing, verification, and attack resistance."""

    def setUp(self):
        self.alice = Operator("alice", seed=b"alice-test")
        self.bob = Operator("bob", seed=b"bob-test")
        self.eve = Operator("eve", seed=b"eve-test")

        self.robot = Robot("TestBot")
        self.robot.register_operator("alice", self.alice.public_key)
        self.robot.register_operator("bob", self.bob.public_key)

    def test_authenticated_command_accepted(self):
        """Valid signed command from registered operator must be accepted."""
        cmd = self.alice.create_signed_command(
            CommandType.MOVE, {"distance": 1.0, "speed": 0.5}
        )
        self.assertTrue(self.robot.receive_command(cmd))

    def test_unauthorized_operator_rejected(self):
        """Command from unregistered operator must be rejected."""
        cmd = self.eve.create_signed_command(
            CommandType.MOVE, {"distance": 1.0, "speed": 0.5}
        )
        self.assertFalse(self.robot.receive_command(cmd))

    def test_replay_attack_rejected(self):
        """Replaying the same command must be rejected (nonce check)."""
        cmd = self.alice.create_signed_command(
            CommandType.MOVE, {"distance": 1.0, "speed": 0.5}
        )
        self.assertTrue(self.robot.receive_command(cmd))
        self.assertFalse(self.robot.receive_command(cmd))  # Replay

    def test_tampered_command_rejected(self):
        """Command with modified parameters but original signature must fail."""
        signed = self.alice.create_signed_command(
            CommandType.MOVE, {"distance": 1.0, "speed": 0.5}
        )
        tampered = SignedCommand(
            command=RobotCommand(
                cmd_type=CommandType.MOVE,
                params={"distance": 100.0, "speed": 1.0},
                timestamp=signed.command.timestamp,
                nonce=signed.command.nonce,
                operator_id=signed.command.operator_id,
            ),
            signature_r=signed.signature_r,
            signature_s=signed.signature_s,
        )
        self.assertFalse(self.robot.receive_command(tampered))

    def test_batch_commands_accepted(self):
        """All commands in a valid Merkle batch should be accepted."""
        batch = self.alice.create_batch([
            (CommandType.MOVE, {"distance": 1.0, "speed": 0.5}),
            (CommandType.ROTATE, {"angle": math.pi / 4}),
            (CommandType.STOP, {}),
        ])
        for i in range(len(batch.commands)):
            cmd, proof = batch.get_command_with_proof(i)
            self.assertTrue(self.robot.receive_batch_command(batch, i, proof))

    def test_emergency_stop_halts_robot(self):
        """Emergency stop should transition robot to HALTED state."""
        cmd = self.alice.create_signed_command(CommandType.EMERGENCY_STOP, {})
        self.robot.receive_command(cmd)
        self.assertEqual(self.robot.state, RobotState.HALTED)

    def test_halted_robot_rejects_move(self):
        """A halted robot should reject MOVE commands."""
        estop = self.alice.create_signed_command(CommandType.EMERGENCY_STOP, {})
        self.robot.receive_command(estop)
        move = self.alice.create_signed_command(
            CommandType.MOVE, {"distance": 1.0, "speed": 0.5}
        )
        self.assertFalse(self.robot.receive_command(move))

    def test_halted_robot_recovers_with_stop(self):
        """STOP command should recover a HALTED robot back to IDLE."""
        estop = self.alice.create_signed_command(CommandType.EMERGENCY_STOP, {})
        self.robot.receive_command(estop)
        self.assertEqual(self.robot.state, RobotState.HALTED)
        reset = self.alice.create_signed_command(CommandType.STOP, {})
        self.robot.receive_command(reset)
        self.assertEqual(self.robot.state, RobotState.IDLE)


class TestPIDController(unittest.TestCase):
    """Test PID controller convergence."""

    def test_converges_to_zero_error(self):
        """PID should drive error close to zero over time."""
        pid = PIDController(kp=2.0, ki=0.1, kd=0.5)
        error = 10.0
        dt = 0.1
        for _ in range(100):
            control = pid.compute(error, dt)
            error -= control * dt
        self.assertAlmostEqual(error, 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
