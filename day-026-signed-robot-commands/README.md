# Day 026: Cryptographically Signed Robot Command Protocol

## Overview

Build a secure robot command protocol where every movement command is digitally signed using ECDSA, commands are batched into Merkle trees for efficient verification, and a simulated robot executes only authenticated commands using a state machine and PID controller.

**Why this matters:** In real-world robotics — from factory floors to autonomous drones — unauthorized commands can cause physical damage, safety hazards, or sabotage. Production robot systems (ROS2 SROS2, DDS Security) use cryptographic authentication to ensure commands come from authorized operators. This challenge builds that security layer from the ground up, integrating cryptography, robotics control, and data structures you've built in previous days.

## Core Concepts

### 1. Command Authentication Chain

The fundamental problem: how does a robot know a command is legitimate?

We solve this with a **three-layer authentication stack**:

```
Layer 3: Merkle Tree Batch Verification
         └── Efficiently prove a command belongs to an authorized batch
Layer 2: ECDSA Digital Signatures
         └── Prove the command came from a specific operator
Layer 1: Command Structure
         └── Type-safe, timestamped, nonce-protected commands
```

**Nonce protection** prevents replay attacks. Without it, an attacker could record a valid "move forward" command and replay it indefinitely. Each command carries a monotonically increasing nonce; the robot rejects any nonce ≤ the last accepted nonce.

**Timestamps** add a freshness window. Even with nonces, a command from 10 minutes ago shouldn't execute — the environment has changed. We enforce a configurable TTL (time-to-live) window.

### 2. ECDSA for Command Signing (Review from Day 020)

ECDSA operates over elliptic curves. For a command `m`:

1. **Sign**: Operator with private key `d` computes signature `(r, s)` where:
   - Pick random `k`, compute point `R = k·G`, set `r = R.x mod n`
   - `s = k⁻¹(hash(m) + r·d) mod n`

2. **Verify**: Robot with public key `Q = d·G` checks:
   - Compute `u₁ = hash(m)·s⁻¹ mod n`, `u₂ = r·s⁻¹ mod n`
   - Check that `(u₁·G + u₂·Q).x mod n == r`

The key insight: signing requires the private key (only the operator has it), but verification only needs the public key (the robot stores it). Compromise of the robot doesn't compromise signing authority.

### 3. Merkle Batch Verification (Review from Day 013)

When an operator sends a batch of commands (e.g., a waypoint sequence), we organize them into a Merkle tree. This gives us:

- **O(log n) inclusion proofs**: The robot can verify any single command belongs to the batch without storing the entire batch
- **Batch integrity**: If any command in the batch is tampered with, the root hash changes
- **Selective execution**: The robot can execute commands one at a time, verifying each against the Merkle root

The Merkle root itself is signed, so one signature authenticates an entire batch of commands.

### 4. Robot State Machine for Command Execution

The robot operates as a finite state machine:

```
IDLE ──[authenticated cmd]──> EXECUTING
  ^                              │
  │                              v
  └──[cmd complete]──── EXECUTING
  
IDLE ──[invalid cmd]──> REJECTED (log & stay IDLE)
EXECUTING ──[emergency stop]──> HALTED
HALTED ──[authenticated reset]──> IDLE
```

Each state transition requires cryptographic verification. The robot never executes an unauthenticated command — this is the **security invariant** of the system.

### 5. PID Control for Command Execution

Once a command is authenticated, the robot uses PID control to execute it physically:

- **Move commands**: PID controls velocity to reach target position
- **Rotate commands**: PID controls angular velocity to reach target heading
- **Stop commands**: PID brings velocity to zero smoothly

The PID controller is the bridge between the digital (authenticated command) and physical (motor control) domains.

## Step-by-Step Breakdown

### Step 1: Define the Command Protocol

Create a typed command structure with fields for command type (MOVE, ROTATE, STOP, EMERGENCY_STOP), parameters (distance, angle, speed), timestamp, nonce, and operator ID. Without structured commands, you can't hash them deterministically for signing.

### Step 2: Implement Key Management

Create an Operator class that holds an ECDSA key pair and can sign commands. Create a Robot class that stores a list of authorized operator public keys. This separation of concerns mirrors real PKI — the robot never needs to know private keys.

### Step 3: Build Command Signing and Verification

The operator signs the SHA-256 hash of the serialized command. The robot verifies by checking the signature against the operator's registered public key, then checks nonce freshness and timestamp TTL. Verification must be **all-or-nothing** — if any check fails, the entire command is rejected.

### Step 4: Add Merkle Batch Commands

For waypoint sequences, the operator builds a Merkle tree of commands, signs the root, and sends commands with their Merkle proofs. The robot verifies each command's inclusion proof against the signed root before execution.

### Step 5: Integrate the State Machine

Wire the verification pipeline into a state machine. IDLE accepts new commands, EXECUTING runs PID loops, HALTED only accepts authenticated reset commands. Every state transition is logged with the cryptographic proof that authorized it.

### Step 6: Simulate End-to-End

Run a full scenario: operator creates a waypoint mission, signs the batch, robot verifies and executes each waypoint using PID control, with attack simulations (tampered commands, replay attacks, unauthorized operators) to prove the security properties hold.

## Learning Objectives

- **Security architecture**: Design a multi-layer authentication system for physical systems
- **Protocol design**: Build a command protocol with replay protection and freshness guarantees
- **System integration**: Combine cryptographic primitives (ECDSA, Merkle trees, SHA-256) with robotics control (state machines, PID)
- **Attack modeling**: Understand and defend against replay attacks, command tampering, and unauthorized access
- **Trust boundaries**: Separate signing authority from execution authority

## Going Deeper

- **Certificate chains**: In production, operator keys are signed by a Certificate Authority. The robot trusts the CA, which trusts operators — this is how ROS2 SROS2 works.
- **Hardware Security Modules (HSMs)**: Private keys should live in tamper-resistant hardware, not software. TPM chips on robots store verification keys.
- **Rate limiting**: Even authenticated commands should be rate-limited to prevent a compromised operator from sending dangerous rapid-fire commands.
- **Encrypted commands**: We authenticate but don't encrypt. In adversarial environments (military, competitive), command encryption prevents eavesdropping on mission plans.
- **Multi-signature requirements**: Critical commands (self-destruct, enter hazardous area) could require M-of-N operator signatures.
- **Connection to Day 020 (ECDSA)**, **Day 013 (Merkle Trees)**, **Day 007 (State Machines)**, **Day 006 (PID Control)**: This challenge directly builds on all four, showing how isolated concepts compose into a real system.
