"""
Day 026: Cryptographically Signed Robot Command Protocol — YOUR IMPLEMENTATION

Build a secure robot command system integrating:
- ECDSA digital signatures for command authentication
- Merkle trees for batch command verification
- State machine for safe command execution flow
- PID controller for physical command execution

Run tests with: python3 -m pytest tests.py -v
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# =============================================================================
# PART 1: ECDSA on secp256k1
# Hint: Review Day 020 for point arithmetic and signing/verification.
# Key insight: signing needs the private key, verification only needs the public key.
# =============================================================================

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def modinv(a: int, m: int) -> int:
    """Modular inverse using extended Euclidean algorithm."""
    raise NotImplementedError("TODO: implement modular inverse")


class ECPoint:
    """Point on the secp256k1 elliptic curve.

    Hint: You need point addition and scalar multiplication.
    Point at infinity is represented by x=None, y=None.
    """

    def __init__(self, x: Optional[int], y: Optional[int]):
        self.x = x
        self.y = y

    @property
    def is_infinity(self) -> bool:
        return self.x is None and self.y is None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ECPoint):
            return False
        return self.x == other.x and self.y == other.y

    def __add__(self, other: ECPoint) -> ECPoint:
        """Point addition on the elliptic curve.
        Hint: Handle three cases — identity, doubling, and general addition.
        Use the slope of the line through the two points."""
        raise NotImplementedError("TODO: implement point addition")

    def __rmul__(self, scalar: int) -> ECPoint:
        """Scalar multiplication using double-and-add.
        Hint: Process bits of the scalar from LSB to MSB."""
        raise NotImplementedError("TODO: implement scalar multiplication")


G = ECPoint(GX, GY)


def generate_keypair(seed: bytes) -> tuple[int, ECPoint]:
    """Generate a deterministic key pair from a seed.
    Hint: private_key = HMAC-SHA256(seed, "ecdsa-keygen") mod (N-1) + 1
    public_key = private_key * G"""
    raise NotImplementedError("TODO: implement key generation")


def ecdsa_sign(private_key: int, message_hash: bytes, k_seed: bytes) -> tuple[int, int]:
    """Sign a message hash using ECDSA.
    Hint: Compute deterministic k, then R = k*G, r = R.x mod N,
    s = k_inv * (z + r*d) mod N."""
    raise NotImplementedError("TODO: implement ECDSA signing")


def ecdsa_verify(public_key: ECPoint, message_hash: bytes, r: int, s: int) -> bool:
    """Verify an ECDSA signature.
    Hint: Compute u1 = z*s_inv, u2 = r*s_inv, check (u1*G + u2*Q).x == r."""
    raise NotImplementedError("TODO: implement ECDSA verification")


# =============================================================================
# PART 2: Merkle Tree for Batch Verification
# Hint: Review Day 013. Use domain separation to prevent second preimage attacks.
# =============================================================================

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def merkle_hash_leaf(data: bytes) -> bytes:
    """Domain-separated leaf hash. Prefix with b'\\x00'."""
    raise NotImplementedError("TODO: implement leaf hashing")


def merkle_hash_internal(left: bytes, right: bytes) -> bytes:
    """Domain-separated internal node hash. Prefix with b'\\x01'."""
    raise NotImplementedError("TODO: implement internal node hashing")


def build_merkle_tree(leaves: list[bytes]) -> tuple[bytes, list[list[bytes]]]:
    """Build a Merkle tree from leaf data.
    Returns (root_hash, tree_layers) where tree_layers[0] = leaf hashes.
    Hint: Pair up nodes at each level, hashing pairs together.
    Duplicate the last node if a layer has an odd number of elements."""
    raise NotImplementedError("TODO: implement Merkle tree construction")


def get_merkle_proof(layers: list[list[bytes]], index: int) -> list[tuple[bytes, str]]:
    """Get a Merkle inclusion proof for the leaf at `index`.
    Returns list of (sibling_hash, direction) pairs.
    Hint: At each layer, find the sibling and record whether it's left or right."""
    raise NotImplementedError("TODO: implement Merkle proof generation")


def verify_merkle_proof(leaf_data: bytes, proof: list[tuple[bytes, str]], root: bytes) -> bool:
    """Verify that leaf_data is included in the tree with the given root.
    Hint: Hash the leaf, then iteratively combine with proof siblings."""
    raise NotImplementedError("TODO: implement Merkle proof verification")


# =============================================================================
# PART 3: Command Protocol
# Hint: Commands must serialize deterministically (sorted JSON).
# Nonces prevent replay attacks. Timestamps add freshness.
# =============================================================================

class CommandType(Enum):
    MOVE = auto()
    ROTATE = auto()
    STOP = auto()
    EMERGENCY_STOP = auto()


@dataclass
class RobotCommand:
    """A single robot command with authentication metadata."""
    cmd_type: CommandType
    params: dict
    timestamp: float
    nonce: int
    operator_id: str

    def serialize(self) -> bytes:
        """Deterministic serialization for signing.
        Hint: Use sorted JSON with no whitespace."""
        raise NotImplementedError("TODO: implement command serialization")

    def hash(self) -> bytes:
        """SHA-256 hash of the serialized command."""
        raise NotImplementedError("TODO: implement command hashing")


@dataclass
class SignedCommand:
    """A command bundled with its ECDSA signature."""
    command: RobotCommand
    signature_r: int
    signature_s: int

    def verify(self, public_key: ECPoint) -> bool:
        """Verify the signature against the given public key."""
        raise NotImplementedError("TODO: implement signed command verification")


@dataclass
class MerkleBatchCommand:
    """A batch of commands in a Merkle tree with one root signature."""
    commands: list[RobotCommand]
    merkle_root: bytes
    root_signature_r: int
    root_signature_s: int
    merkle_layers: list[list[bytes]]

    def get_command_with_proof(self, index: int) -> tuple[RobotCommand, list[tuple[bytes, str]]]:
        """Get a command and its Merkle inclusion proof."""
        raise NotImplementedError("TODO: implement batch command proof retrieval")


# =============================================================================
# PART 4: Operator (Command Issuer)
# Hint: The operator holds the private key and signs commands.
# The robot only needs the public key.
# =============================================================================

class Operator:
    """An authorized operator who can sign commands for a robot."""

    def __init__(self, name: str, seed: Optional[bytes] = None):
        self.name = name
        seed = seed or f"operator-{name}".encode()
        self.private_key, self.public_key = generate_keypair(seed)
        self._nonce = 0

    def _next_nonce(self) -> int:
        self._nonce += 1
        return self._nonce

    def create_command(self, cmd_type: CommandType, params: dict) -> RobotCommand:
        """Create a command with automatic timestamp and nonce."""
        raise NotImplementedError("TODO: implement command creation")

    def sign_command(self, cmd: RobotCommand) -> SignedCommand:
        """Sign a single command with ECDSA."""
        raise NotImplementedError("TODO: implement command signing")

    def create_signed_command(self, cmd_type: CommandType, params: dict) -> SignedCommand:
        """Convenience: create and sign in one step."""
        raise NotImplementedError("TODO: implement create + sign")

    def create_batch(self, command_specs: list[tuple[CommandType, dict]]) -> MerkleBatchCommand:
        """Create a Merkle batch of commands with one signature on the root.
        Hint: Build Merkle tree from serialized commands, sign the root."""
        raise NotImplementedError("TODO: implement batch creation")


# =============================================================================
# PART 5: Robot State Machine
# Hint: States are IDLE, EXECUTING, HALTED.
# Security invariant: NO action without cryptographic verification.
# =============================================================================

class RobotState(Enum):
    IDLE = auto()
    EXECUTING = auto()
    HALTED = auto()


@dataclass
class SecurityLog:
    """Immutable log entry for security auditing."""
    timestamp: float
    event: str
    operator: str
    command_type: str
    nonce: int
    verified: bool
    details: str = ""


# =============================================================================
# PART 6: PID Controller
# Hint: Review Day 006. u(t) = Kp*e + Ki*∫e + Kd*de/dt
# Don't forget anti-windup for the integral term.
# =============================================================================

class PIDController:
    """PID controller for smooth command execution."""

    def __init__(self, kp: float, ki: float, kd: float, output_limits: tuple[float, float] = (-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        """Compute PID output for the given error and timestep."""
        raise NotImplementedError("TODO: implement PID computation")


# =============================================================================
# PART 7: Robot (Command Executor)
# This is the integration point — wire together verification, state machine,
# and PID control. The robot should NEVER execute an unverified command.
# =============================================================================

class Robot:
    """A simulated robot that only executes authenticated commands."""

    COMMAND_TTL = 30.0

    def __init__(self, name: str):
        self.name = name
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.velocity = 0.0
        self.state = RobotState.IDLE
        self.authorized_keys: dict[str, ECPoint] = {}
        self.last_nonce: dict[str, int] = {}
        self.security_log: list[SecurityLog] = []
        self.position_pid = PIDController(kp=2.0, ki=0.1, kd=0.5, output_limits=(-1.0, 1.0))
        self.heading_pid = PIDController(kp=3.0, ki=0.05, kd=0.3, output_limits=(-math.pi, math.pi))

    def register_operator(self, operator_id: str, public_key: ECPoint):
        """Register an operator's public key."""
        raise NotImplementedError("TODO: implement operator registration")

    def receive_command(self, signed_cmd: SignedCommand) -> bool:
        """Process a signed command through the verification pipeline.
        Hint: Check state machine, verify signature, check nonce + timestamp,
        then execute. Log everything."""
        raise NotImplementedError("TODO: implement command reception and verification")

    def receive_batch_command(
        self,
        batch: MerkleBatchCommand,
        index: int,
        proof: list[tuple[bytes, str]],
    ) -> bool:
        """Process a command from a Merkle batch.
        Hint: Verify Merkle proof first, then verify root signature,
        then check nonce, then execute."""
        raise NotImplementedError("TODO: implement batch command reception")

    def _execute_command(self, cmd: RobotCommand) -> bool:
        """Execute an authenticated command using PID control.
        Hint: Handle each CommandType differently.
        MOVE uses position PID, ROTATE uses heading PID,
        EMERGENCY_STOP immediately halts."""
        raise NotImplementedError("TODO: implement command execution with PID")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Day 026: Cryptographically Signed Robot Command Protocol")
    print("=" * 70)

    # Setup
    alice = Operator("alice", seed=b"alice-secret-seed")
    bob = Operator("bob", seed=b"bob-secret-seed")
    eve = Operator("eve", seed=b"eve-evil-seed")

    robot = Robot("Atlas")
    robot.register_operator("alice", alice.public_key)
    robot.register_operator("bob", bob.public_key)

    # Test 1: Authenticated command
    print("\n[Test 1] Alice sends MOVE command...")
    cmd = alice.create_signed_command(CommandType.MOVE, {"distance": 3.0, "speed": 0.8})
    result = robot.receive_command(cmd)
    print(f"  Accepted: {result} (should be True)")
    print(f"  Position: ({robot.x:.2f}, {robot.y:.2f})")

    # Test 2: Unauthorized operator
    print("\n[Test 2] Eve (unauthorized) sends command...")
    evil = eve.create_signed_command(CommandType.MOVE, {"distance": 1.0, "speed": 0.5})
    result = robot.receive_command(evil)
    print(f"  Accepted: {result} (should be False)")

    # Test 3: Replay attack
    print("\n[Test 3] Replaying Alice's first command...")
    result = robot.receive_command(cmd)
    print(f"  Accepted: {result} (should be False — replay detected)")

    # Test 4: Batch commands
    print("\n[Test 4] Alice sends waypoint batch...")
    batch = alice.create_batch([
        (CommandType.MOVE, {"distance": 2.0, "speed": 0.6}),
        (CommandType.ROTATE, {"angle": math.pi / 4}),
        (CommandType.STOP, {}),
    ])
    for i in range(len(batch.commands)):
        bcmd, proof = batch.get_command_with_proof(i)
        result = robot.receive_batch_command(batch, i, proof)
        print(f"  Waypoint {i}: accepted={result}")

    print(f"\n  Final position: ({robot.x:.2f}, {robot.y:.2f})")
    print("  Done!")
