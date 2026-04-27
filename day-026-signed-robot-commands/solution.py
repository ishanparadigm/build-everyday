"""
Day 026: Cryptographically Signed Robot Command Protocol

A secure robot command system integrating:
- ECDSA digital signatures for command authentication
- Merkle trees for batch command verification
- State machine for safe command execution flow
- PID controller for physical command execution

This solution uses Python's built-in ecdsa-equivalent via hashlib + hmac for
hashing, and a simplified ECDSA implementation for educational clarity.
For production, use the `cryptography` library.
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
# PART 1: ECDSA on secp256k1 (simplified from Day 020)
# =============================================================================

# secp256k1 curve parameters — the same curve Bitcoin uses.
# y² = x³ + 7 over the prime field F_p
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def modinv(a: int, m: int) -> int:
    """Modular inverse using extended Euclidean algorithm."""
    if a < 0:
        a = a % m
    g, x, _ = _extended_gcd(a, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return x % m


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


class ECPoint:
    """Point on the secp256k1 elliptic curve."""

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
        """Point addition on the curve. This is the core operation that makes
        elliptic curve cryptography work — it's a group operation that's easy
        to compute forward but computationally infeasible to reverse (the
        discrete logarithm problem)."""
        if self.is_infinity:
            return other
        if other.is_infinity:
            return self
        if self.x == other.x and self.y != other.y:
            return ECPoint(None, None)  # Point at infinity

        if self == other:
            # Point doubling: tangent line slope
            lam = (3 * self.x * self.x + A) * modinv(2 * self.y, P) % P
        else:
            # Point addition: secant line slope
            lam = (other.y - self.y) * modinv(other.x - self.x, P) % P

        x3 = (lam * lam - self.x - other.x) % P
        y3 = (lam * (self.x - x3) - self.y) % P
        return ECPoint(x3, y3)

    def __rmul__(self, scalar: int) -> ECPoint:
        """Scalar multiplication using double-and-add.
        Computes scalar * Point in O(log scalar) time."""
        result = ECPoint(None, None)
        addend = self
        k = scalar % N
        while k:
            if k & 1:
                result = result + addend
            addend = addend + addend
            k >>= 1
        return result


# Generator point
G = ECPoint(GX, GY)


def generate_keypair(seed: bytes) -> tuple[int, ECPoint]:
    """Generate a deterministic key pair from a seed.
    In production, use a CSPRNG. Here we use HMAC-SHA256 for reproducibility."""
    # Derive private key from seed via HMAC (deterministic for testing)
    private_key = int.from_bytes(
        hmac.new(seed, b"ecdsa-keygen", hashlib.sha256).digest(), "big"
    ) % (N - 1) + 1  # Ensure 1 <= d < N
    public_key = private_key * G
    return private_key, public_key


def ecdsa_sign(private_key: int, message_hash: bytes, k_seed: bytes) -> tuple[int, int]:
    """Sign a message hash using ECDSA.
    k_seed provides deterministic nonce generation (RFC 6979 style)."""
    z = int.from_bytes(message_hash, "big")
    # Deterministic k from private key + message (simplified RFC 6979)
    k_bytes = hmac.new(
        private_key.to_bytes(32, "big") + k_seed,
        message_hash,
        hashlib.sha256,
    ).digest()
    k = int.from_bytes(k_bytes, "big") % (N - 1) + 1

    R = k * G
    r = R.x % N
    s = (modinv(k, N) * (z + r * private_key)) % N

    if r == 0 or s == 0:
        raise ValueError("Degenerate signature, retry with different k")
    return r, s


def ecdsa_verify(public_key: ECPoint, message_hash: bytes, r: int, s: int) -> bool:
    """Verify an ECDSA signature. Returns True iff the signature is valid."""
    if not (1 <= r < N and 1 <= s < N):
        return False
    z = int.from_bytes(message_hash, "big")
    s_inv = modinv(s, N)
    u1 = (z * s_inv) % N
    u2 = (r * s_inv) % N
    point = u1 * G + u2 * public_key
    if point.is_infinity:
        return False
    return point.x % N == r


# =============================================================================
# PART 2: Merkle Tree for Batch Verification (from Day 013)
# =============================================================================

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def merkle_hash_leaf(data: bytes) -> bytes:
    """Domain-separated leaf hash to prevent second preimage attacks.
    Without domain separation, an attacker could create an internal node
    that looks like a leaf, breaking the proof."""
    return sha256(b"\x00" + data)


def merkle_hash_internal(left: bytes, right: bytes) -> bytes:
    """Domain-separated internal node hash."""
    return sha256(b"\x01" + left + right)


def build_merkle_tree(leaves: list[bytes]) -> tuple[bytes, list[list[bytes]]]:
    """Build a Merkle tree from leaf data.
    Returns (root_hash, tree_layers) where tree_layers[0] = leaf hashes."""
    if not leaves:
        return sha256(b""), [[]]

    # Hash leaves with domain separation
    current_layer = [merkle_hash_leaf(leaf) for leaf in leaves]
    layers = [current_layer[:]]

    while len(current_layer) > 1:
        next_layer = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            # If odd number of nodes, duplicate the last one
            right = current_layer[i + 1] if i + 1 < len(current_layer) else current_layer[i]
            next_layer.append(merkle_hash_internal(left, right))
        current_layer = next_layer
        layers.append(current_layer[:])

    return current_layer[0], layers


def get_merkle_proof(layers: list[list[bytes]], index: int) -> list[tuple[bytes, str]]:
    """Get a Merkle inclusion proof for the leaf at `index`.
    Returns list of (sibling_hash, direction) pairs."""
    proof = []
    for layer in layers[:-1]:  # Skip root layer
        if index % 2 == 0:
            sibling_idx = index + 1
            direction = "right"
        else:
            sibling_idx = index - 1
            direction = "left"
        # Handle odd-length layers
        if sibling_idx >= len(layer):
            sibling_idx = index
            direction = "right"
        proof.append((layer[sibling_idx], direction))
        index //= 2
    return proof


def verify_merkle_proof(leaf_data: bytes, proof: list[tuple[bytes, str]], root: bytes) -> bool:
    """Verify that leaf_data is included in the tree with the given root."""
    current = merkle_hash_leaf(leaf_data)
    for sibling_hash, direction in proof:
        if direction == "right":
            current = merkle_hash_internal(current, sibling_hash)
        else:
            current = merkle_hash_internal(sibling_hash, current)
    return current == root


# =============================================================================
# PART 3: Command Protocol
# =============================================================================

class CommandType(Enum):
    """Robot command types. Each maps to a specific physical action."""
    MOVE = auto()           # Linear movement to a target position
    ROTATE = auto()         # Rotation to a target heading
    STOP = auto()           # Graceful deceleration to zero velocity
    EMERGENCY_STOP = auto() # Immediate halt (bypasses PID smoothing)


@dataclass
class RobotCommand:
    """A single robot command with all metadata needed for authentication.

    Fields:
    - cmd_type: What action to perform
    - params: Command-specific parameters (distance, angle, speed, etc.)
    - timestamp: When the command was created (Unix epoch)
    - nonce: Monotonically increasing counter — prevents replay attacks
    - operator_id: Identifier for the signing operator
    """
    cmd_type: CommandType
    params: dict
    timestamp: float
    nonce: int
    operator_id: str

    def serialize(self) -> bytes:
        """Deterministic serialization for signing.
        We use sorted JSON to ensure identical commands always produce
        identical byte strings — critical for signature verification."""
        data = {
            "type": self.cmd_type.name,
            "params": self.params,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "operator_id": self.operator_id,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

    def hash(self) -> bytes:
        """SHA-256 hash of the serialized command."""
        return sha256(self.serialize())


@dataclass
class SignedCommand:
    """A command bundled with its ECDSA signature."""
    command: RobotCommand
    signature_r: int
    signature_s: int

    def verify(self, public_key: ECPoint) -> bool:
        """Verify the signature against the given public key."""
        return ecdsa_verify(public_key, self.command.hash(), self.signature_r, self.signature_s)


@dataclass
class MerkleBatchCommand:
    """A batch of commands organized in a Merkle tree.
    The operator signs only the Merkle root — one signature authenticates
    the entire batch. Individual commands carry inclusion proofs."""
    commands: list[RobotCommand]
    merkle_root: bytes
    root_signature_r: int
    root_signature_s: int
    merkle_layers: list[list[bytes]]  # For generating proofs

    def get_command_with_proof(self, index: int) -> tuple[RobotCommand, list[tuple[bytes, str]]]:
        """Get a command and its Merkle inclusion proof."""
        proof = get_merkle_proof(self.merkle_layers, index)
        return self.commands[index], proof


# =============================================================================
# PART 4: Operator (Command Issuer)
# =============================================================================

class Operator:
    """An authorized operator who can sign commands for a robot.

    The operator holds a private key and generates signed commands.
    The robot only needs the operator's public key to verify."""

    def __init__(self, name: str, seed: Optional[bytes] = None):
        self.name = name
        seed = seed or f"operator-{name}".encode()
        self.private_key, self.public_key = generate_keypair(seed)
        self._nonce = 0  # Tracks the next nonce to use

    def _next_nonce(self) -> int:
        self._nonce += 1
        return self._nonce

    def create_command(self, cmd_type: CommandType, params: dict) -> RobotCommand:
        """Create a command with automatic timestamp and nonce."""
        return RobotCommand(
            cmd_type=cmd_type,
            params=params,
            timestamp=time.time(),
            nonce=self._next_nonce(),
            operator_id=self.name,
        )

    def sign_command(self, cmd: RobotCommand) -> SignedCommand:
        """Sign a single command with ECDSA."""
        msg_hash = cmd.hash()
        r, s = ecdsa_sign(self.private_key, msg_hash, b"cmd-sign")
        return SignedCommand(command=cmd, signature_r=r, signature_s=s)

    def create_signed_command(self, cmd_type: CommandType, params: dict) -> SignedCommand:
        """Convenience: create and sign in one step."""
        cmd = self.create_command(cmd_type, params)
        return self.sign_command(cmd)

    def create_batch(self, command_specs: list[tuple[CommandType, dict]]) -> MerkleBatchCommand:
        """Create a batch of commands organized in a Merkle tree.
        One signature covers the entire batch via the Merkle root."""
        commands = [self.create_command(ct, params) for ct, params in command_specs]
        leaves = [cmd.serialize() for cmd in commands]
        root, layers = build_merkle_tree(leaves)

        # Sign the Merkle root — this one signature authenticates all commands
        r, s = ecdsa_sign(self.private_key, root, b"batch-sign")

        return MerkleBatchCommand(
            commands=commands,
            merkle_root=root,
            root_signature_r=r,
            root_signature_s=s,
            merkle_layers=layers,
        )


# =============================================================================
# PART 5: Robot State Machine
# =============================================================================

class RobotState(Enum):
    """Robot operational states. Transitions require cryptographic authorization."""
    IDLE = auto()       # Waiting for commands
    EXECUTING = auto()  # Running a command (PID active)
    HALTED = auto()     # Emergency stop — only accepts authenticated reset


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
# PART 6: PID Controller (from Day 006)
# =============================================================================

class PIDController:
    """PID controller for smooth command execution.

    The three terms work together:
    - P (proportional): Corrects based on current error — fast but overshoots
    - I (integral): Corrects accumulated past error — eliminates steady-state error
    - D (derivative): Dampens based on error rate of change — reduces overshoot

    Combined: u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt
    """

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
        self.integral += error * dt
        # Anti-windup: clamp integral to prevent accumulation during saturation
        max_integral = self.output_limits[1] / (self.ki if self.ki != 0 else 1)
        self.integral = max(-max_integral, min(max_integral, self.integral))

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.output_limits[0], min(self.output_limits[1], output))


# =============================================================================
# PART 7: Robot (Command Executor)
# =============================================================================

class Robot:
    """A simulated robot that only executes cryptographically authenticated commands.

    Security invariant: NO physical action occurs without verified authorization.

    The robot maintains:
    - Position (x, y) and heading (theta) in 2D space
    - A set of authorized operator public keys
    - Nonce tracking per operator (replay protection)
    - A security audit log
    """

    COMMAND_TTL = 30.0  # Commands older than 30 seconds are rejected

    def __init__(self, name: str):
        self.name = name
        # Physical state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # Heading in radians
        self.velocity = 0.0
        # Security state
        self.state = RobotState.IDLE
        self.authorized_keys: dict[str, ECPoint] = {}  # operator_id -> public_key
        self.last_nonce: dict[str, int] = {}  # operator_id -> last accepted nonce
        self.security_log: list[SecurityLog] = []
        # Control
        self.position_pid = PIDController(kp=2.0, ki=0.1, kd=0.5, output_limits=(-1.0, 1.0))
        self.heading_pid = PIDController(kp=3.0, ki=0.05, kd=0.3, output_limits=(-math.pi, math.pi))

    def register_operator(self, operator_id: str, public_key: ECPoint):
        """Register an operator's public key. In production, this would use
        a certificate chain rooted in a trusted CA."""
        self.authorized_keys[operator_id] = public_key
        self.last_nonce[operator_id] = 0
        print(f"  [{self.name}] Registered operator '{operator_id}'")

    def _log(self, event: str, cmd: RobotCommand, verified: bool, details: str = ""):
        self.security_log.append(SecurityLog(
            timestamp=time.time(),
            event=event,
            operator=cmd.operator_id,
            command_type=cmd.cmd_type.name,
            nonce=cmd.nonce,
            verified=verified,
            details=details,
        ))

    def _verify_command(self, cmd: RobotCommand, signature_r: int, signature_s: int) -> tuple[bool, str]:
        """Multi-layer command verification.
        Returns (is_valid, rejection_reason)."""

        # Layer 1: Check operator is authorized
        if cmd.operator_id not in self.authorized_keys:
            return False, f"Unknown operator: {cmd.operator_id}"

        # Layer 2: Check nonce freshness (replay protection)
        last = self.last_nonce.get(cmd.operator_id, 0)
        if cmd.nonce <= last:
            return False, f"Stale nonce {cmd.nonce} <= last accepted {last}"

        # Layer 3: Check timestamp freshness
        age = time.time() - cmd.timestamp
        if age > self.COMMAND_TTL:
            return False, f"Command too old: {age:.1f}s > TTL {self.COMMAND_TTL}s"

        # Layer 4: Verify ECDSA signature
        pub_key = self.authorized_keys[cmd.operator_id]
        if not ecdsa_verify(pub_key, cmd.hash(), signature_r, signature_s):
            return False, "Invalid ECDSA signature"

        return True, "OK"

    def receive_command(self, signed_cmd: SignedCommand) -> bool:
        """Process a signed command through the verification pipeline.
        Returns True if the command was accepted and executed."""
        cmd = signed_cmd.command
        print(f"\n  [{self.name}] Received {cmd.cmd_type.name} from '{cmd.operator_id}' (nonce={cmd.nonce})")

        # State machine check: HALTED only accepts EMERGENCY_STOP (as reset)
        if self.state == RobotState.HALTED and cmd.cmd_type != CommandType.STOP:
            reason = "Robot is HALTED — only STOP commands accepted for reset"
            self._log("REJECTED", cmd, False, reason)
            print(f"  [{self.name}] REJECTED: {reason}")
            return False

        # Verify command authenticity
        valid, reason = self._verify_command(cmd, signed_cmd.signature_r, signed_cmd.signature_s)

        if not valid:
            self._log("REJECTED", cmd, False, reason)
            print(f"  [{self.name}] REJECTED: {reason}")
            return False

        # Command is authentic — update nonce and execute
        self.last_nonce[cmd.operator_id] = cmd.nonce
        self._log("ACCEPTED", cmd, True)
        print(f"  [{self.name}] VERIFIED ✓ — executing {cmd.cmd_type.name}")

        return self._execute_command(cmd)

    def receive_batch_command(
        self,
        batch: MerkleBatchCommand,
        index: int,
        proof: list[tuple[bytes, str]],
    ) -> bool:
        """Process a single command from a Merkle batch.
        Verifies both the Merkle inclusion proof and the root signature."""
        cmd = batch.commands[index]
        print(f"\n  [{self.name}] Received batch cmd [{index}] {cmd.cmd_type.name} from '{cmd.operator_id}'")

        # Verify Merkle inclusion proof
        if not verify_merkle_proof(cmd.serialize(), proof, batch.merkle_root):
            reason = "Merkle inclusion proof failed"
            self._log("REJECTED", cmd, False, reason)
            print(f"  [{self.name}] REJECTED: {reason}")
            return False

        # Verify the Merkle root signature
        pub_key = self.authorized_keys.get(cmd.operator_id)
        if pub_key is None:
            reason = f"Unknown operator: {cmd.operator_id}"
            self._log("REJECTED", cmd, False, reason)
            print(f"  [{self.name}] REJECTED: {reason}")
            return False

        if not ecdsa_verify(pub_key, batch.merkle_root, batch.root_signature_r, batch.root_signature_s):
            reason = "Batch root signature invalid"
            self._log("REJECTED", cmd, False, reason)
            print(f"  [{self.name}] REJECTED: {reason}")
            return False

        # Check nonce freshness
        last = self.last_nonce.get(cmd.operator_id, 0)
        if cmd.nonce <= last:
            reason = f"Stale nonce {cmd.nonce} <= last accepted {last}"
            self._log("REJECTED", cmd, False, reason)
            print(f"  [{self.name}] REJECTED: {reason}")
            return False

        # All checks passed
        self.last_nonce[cmd.operator_id] = cmd.nonce
        self._log("ACCEPTED_BATCH", cmd, True, f"Merkle proof verified, index={index}")
        print(f"  [{self.name}] VERIFIED ✓ (Merkle proof + signature) — executing")

        return self._execute_command(cmd)

    def _execute_command(self, cmd: RobotCommand) -> bool:
        """Execute an authenticated command using PID control.
        Simulates the physical execution in discrete timesteps."""

        if cmd.cmd_type == CommandType.EMERGENCY_STOP:
            self.state = RobotState.HALTED
            self.velocity = 0.0
            print(f"  [{self.name}] EMERGENCY STOP — velocity zeroed, state=HALTED")
            return True

        if cmd.cmd_type == CommandType.STOP:
            if self.state == RobotState.HALTED:
                # Reset from HALTED state
                self.state = RobotState.IDLE
                print(f"  [{self.name}] Reset from HALTED to IDLE")
                return True
            self.state = RobotState.EXECUTING
            self.position_pid.reset()
            # Decelerate to zero
            steps = 10
            dt = 0.1
            for i in range(steps):
                control = self.position_pid.compute(-self.velocity, dt)
                self.velocity += control * dt
                if abs(self.velocity) < 0.01:
                    self.velocity = 0.0
                    break
            self.state = RobotState.IDLE
            print(f"  [{self.name}] Stopped. velocity={self.velocity:.3f}")
            return True

        if cmd.cmd_type == CommandType.MOVE:
            self.state = RobotState.EXECUTING
            distance = cmd.params.get("distance", 1.0)
            speed = cmd.params.get("speed", 0.5)
            self.position_pid.reset()

            # Simulate movement with PID control
            traveled = 0.0
            dt = 0.05
            max_steps = 200  # Safety limit
            for step in range(max_steps):
                error = distance - traveled
                if abs(error) < 0.01:
                    break
                control = self.position_pid.compute(error, dt)
                # Scale control by desired speed
                self.velocity = control * speed
                # Update position along current heading
                dx = self.velocity * math.cos(self.theta) * dt
                dy = self.velocity * math.sin(self.theta) * dt
                self.x += dx
                self.y += dy
                traveled += abs(self.velocity) * dt

            self.velocity = 0.0
            self.state = RobotState.IDLE
            print(f"  [{self.name}] Moved {traveled:.2f}m → pos=({self.x:.2f}, {self.y:.2f})")
            return True

        if cmd.cmd_type == CommandType.ROTATE:
            self.state = RobotState.EXECUTING
            target_angle = cmd.params.get("angle", 0.0)  # Radians
            self.heading_pid.reset()

            dt = 0.05
            max_steps = 200
            for step in range(max_steps):
                error = target_angle - self.theta
                # Normalize to [-pi, pi]
                error = (error + math.pi) % (2 * math.pi) - math.pi
                if abs(error) < 0.01:
                    break
                control = self.heading_pid.compute(error, dt)
                self.theta += control * dt

            # Normalize final heading
            self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
            self.state = RobotState.IDLE
            print(f"  [{self.name}] Rotated → heading={math.degrees(self.theta):.1f}°")
            return True

        return False

    def print_security_log(self):
        """Print the full security audit trail."""
        print(f"\n{'='*70}")
        print(f"  Security Audit Log for '{self.name}' ({len(self.security_log)} entries)")
        print(f"{'='*70}")
        for entry in self.security_log:
            status = "✓" if entry.verified else "✗"
            details = f" — {entry.details}" if entry.details else ""
            print(f"  [{status}] {entry.event:15s} | {entry.command_type:15s} | "
                  f"operator={entry.operator:10s} | nonce={entry.nonce}{details}")


# =============================================================================
# MAIN: End-to-End Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Day 026: Cryptographically Signed Robot Command Protocol")
    print("=" * 70)

    # --- Setup ---
    print("\n[1] SETUP: Creating operators and robot")
    alice = Operator("alice", seed=b"alice-secret-seed")
    bob = Operator("bob", seed=b"bob-secret-seed")
    eve = Operator("eve", seed=b"eve-evil-seed")  # Unauthorized!

    robot = Robot("Atlas")
    robot.register_operator("alice", alice.public_key)
    robot.register_operator("bob", bob.public_key)
    # Note: eve is NOT registered — her commands should be rejected

    # --- Single Command Verification ---
    print("\n" + "=" * 70)
    print("[2] SINGLE COMMAND: Alice sends MOVE command")
    print("=" * 70)
    move_cmd = alice.create_signed_command(
        CommandType.MOVE, {"distance": 3.0, "speed": 0.8}
    )
    robot.receive_command(move_cmd)

    print(f"\n  Robot position: ({robot.x:.2f}, {robot.y:.2f}), heading: {math.degrees(robot.theta):.1f}°")

    # --- Rotate and Move ---
    print("\n" + "=" * 70)
    print("[3] SEQUENTIAL COMMANDS: Bob rotates then moves")
    print("=" * 70)
    rotate_cmd = bob.create_signed_command(
        CommandType.ROTATE, {"angle": math.pi / 2}  # 90 degrees
    )
    robot.receive_command(rotate_cmd)

    move_cmd2 = bob.create_signed_command(
        CommandType.MOVE, {"distance": 2.0, "speed": 0.5}
    )
    robot.receive_command(move_cmd2)
    print(f"\n  Robot position: ({robot.x:.2f}, {robot.y:.2f}), heading: {math.degrees(robot.theta):.1f}°")

    # --- Attack 1: Unauthorized Operator ---
    print("\n" + "=" * 70)
    print("[4] ATTACK: Eve (unauthorized) tries to send EMERGENCY_STOP")
    print("=" * 70)
    evil_cmd = eve.create_signed_command(CommandType.EMERGENCY_STOP, {})
    robot.receive_command(evil_cmd)

    # --- Attack 2: Replay Attack ---
    print("\n" + "=" * 70)
    print("[5] ATTACK: Replaying Alice's first MOVE command")
    print("=" * 70)
    print("  (Attacker captured the signed command and retransmits it)")
    robot.receive_command(move_cmd)  # Same command as step 2

    # --- Attack 3: Tampered Command ---
    print("\n" + "=" * 70)
    print("[6] ATTACK: Tampering with a signed command")
    print("=" * 70)
    legit_cmd = alice.create_signed_command(
        CommandType.MOVE, {"distance": 1.0, "speed": 0.3}
    )
    # Attacker modifies the distance after signing
    tampered = SignedCommand(
        command=RobotCommand(
            cmd_type=CommandType.MOVE,
            params={"distance": 100.0, "speed": 1.0},  # Tampered!
            timestamp=legit_cmd.command.timestamp,
            nonce=legit_cmd.command.nonce,
            operator_id=legit_cmd.command.operator_id,
        ),
        signature_r=legit_cmd.signature_r,
        signature_s=legit_cmd.signature_s,
    )
    robot.receive_command(tampered)

    # --- Merkle Batch Commands ---
    print("\n" + "=" * 70)
    print("[7] BATCH: Alice sends a waypoint mission via Merkle batch")
    print("=" * 70)

    waypoints = [
        (CommandType.MOVE, {"distance": 2.0, "speed": 0.6}),
        (CommandType.ROTATE, {"angle": math.pi / 4}),
        (CommandType.MOVE, {"distance": 1.5, "speed": 0.4}),
        (CommandType.ROTATE, {"angle": 0.0}),
        (CommandType.STOP, {}),
    ]

    batch = alice.create_batch(waypoints)
    print(f"  Merkle root: {batch.merkle_root.hex()[:32]}...")
    print(f"  Batch size: {len(batch.commands)} commands, 1 signature")

    for i in range(len(batch.commands)):
        cmd, proof = batch.get_command_with_proof(i)
        proof_hashes = [h.hex()[:8] + "..." for h, _ in proof]
        print(f"\n  --- Waypoint {i} (proof depth: {len(proof)}, nodes: {proof_hashes}) ---")
        robot.receive_batch_command(batch, i, proof)

    print(f"\n  Final position: ({robot.x:.2f}, {robot.y:.2f}), heading: {math.degrees(robot.theta):.1f}°")

    # --- Attack 4: Tamper with batch command ---
    print("\n" + "=" * 70)
    print("[8] ATTACK: Tampering with a command inside a Merkle batch")
    print("=" * 70)

    # Create a new batch, then try to substitute a command
    batch2 = alice.create_batch([
        (CommandType.MOVE, {"distance": 1.0, "speed": 0.3}),
        (CommandType.STOP, {}),
    ])

    # Get the proof for command 0, but use a different command
    _, proof0 = batch2.get_command_with_proof(0)
    fake_cmd = alice.create_command(CommandType.MOVE, {"distance": 999.0, "speed": 1.0})
    # Manually put the fake command in the batch for the receive call
    original_cmd = batch2.commands[0]
    batch2.commands[0] = fake_cmd
    robot.receive_batch_command(batch2, 0, proof0)
    batch2.commands[0] = original_cmd  # Restore

    # --- Emergency Stop and Recovery ---
    print("\n" + "=" * 70)
    print("[9] EMERGENCY: Alice triggers emergency stop, then recovers")
    print("=" * 70)
    estop = alice.create_signed_command(CommandType.EMERGENCY_STOP, {})
    robot.receive_command(estop)
    print(f"  Robot state: {robot.state.name}")

    # Try to move while halted (should fail even if authenticated)
    move_while_halted = alice.create_signed_command(
        CommandType.MOVE, {"distance": 1.0, "speed": 0.5}
    )
    robot.receive_command(move_while_halted)

    # Reset with STOP command
    reset_cmd = alice.create_signed_command(CommandType.STOP, {})
    robot.receive_command(reset_cmd)
    print(f"  Robot state: {robot.state.name}")

    # --- Security Audit Log ---
    robot.print_security_log()

    # --- Summary Statistics ---
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    total = len(robot.security_log)
    accepted = sum(1 for e in robot.security_log if e.verified)
    rejected = total - accepted
    print(f"  Total commands processed: {total}")
    print(f"  Accepted (verified):      {accepted}")
    print(f"  Rejected (attacks):       {rejected}")
    print(f"  Final position:           ({robot.x:.2f}, {robot.y:.2f})")
    print(f"  Final heading:            {math.degrees(robot.theta):.1f}°")
    print(f"  Robot state:              {robot.state.name}")
    print()
