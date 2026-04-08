# Day 008: Forward and Inverse Kinematics

## Overview

Build a complete kinematics engine for a 2D robotic arm. Given joint angles, compute where the end-effector lands (forward kinematics). Given a target position, compute what joint angles reach it (inverse kinematics). This is the foundational math behind every robotic arm — from factory welding robots to surgical manipulators to your desk lamp's adjustable arm.

**Why it matters:** Every time a robot arm picks up an object, it solves an inverse kinematics problem. Forward kinematics tells the robot "given my current joint configuration, where is my hand?" Inverse kinematics answers the harder question: "I want my hand *there* — what should my joints do?" This duality is at the heart of all robotic manipulation.

## Core Concepts

### Forward Kinematics: From Joints to End-Effector

A robotic arm is a chain of rigid links connected by joints. Each joint has an angle, and each link has a length. Forward kinematics computes the position of the tip (end-effector) by chaining together rotations and translations.

For a 2D arm with `n` joints, the end-effector position is:

```
x = L1*cos(theta1) + L2*cos(theta1+theta2) + ... + Ln*cos(theta1+...+thetan)
y = L1*sin(theta1) + L2*sin(theta1+theta2) + ... + Ln*sin(theta1+...+thetan)
```

Each joint angle accumulates — joint 2 rotates relative to link 1's orientation, not the world frame. This is why we sum angles: `theta1 + theta2` gives joint 2's absolute angle in the world frame.

**Homogeneous transformation matrices** formalize this. Each joint-link pair is a rotation followed by a translation:

```
T_i = | cos(theta_i)  -sin(theta_i)  L_i*cos(theta_i) |
      | sin(theta_i)   cos(theta_i)  L_i*sin(theta_i) |
      |     0               0              1           |
```

The full transform is `T = T1 * T2 * ... * Tn`. The end-effector position is extracted from the final column.

### Inverse Kinematics: From Target to Joints

Inverse kinematics (IK) is harder because:
1. **It may have no solution** — the target is out of reach
2. **It may have multiple solutions** — a 2-link arm can reach most points with "elbow up" or "elbow down"
3. **For >2 links, it's underdetermined** — infinitely many joint configurations reach the same point

#### Analytical IK (2-link arm)

For a 2-link arm with lengths `L1, L2` targeting point `(tx, ty)`:

1. **Check reachability:** `|L1 - L2| <= sqrt(tx^2 + ty^2) <= L1 + L2`
2. **Use the law of cosines** to find theta2:
   ```
   cos(theta2) = (tx^2 + ty^2 - L1^2 - L2^2) / (2 * L1 * L2)
   theta2 = atan2(+/-sqrt(1 - cos^2(theta2)), cos(theta2))
   ```
   The +/- gives elbow-up vs elbow-down solutions.
3. **Find theta1** using the geometry:
   ```
   theta1 = atan2(ty, tx) - atan2(L2*sin(theta2), L1 + L2*cos(theta2))
   ```

#### Numerical IK (n-link arms): The Jacobian Method

For arms with 3+ links, we use iterative numerical methods. The **Jacobian** relates small joint angle changes to small end-effector position changes:

```
dx = J * d_theta
```

where `J` is the Jacobian matrix (2 x n for 2D):
```
J[0][i] = -sum(L_k * sin(sum(theta_1..k))) for k = i..n
J[1][i] =  sum(L_k * cos(sum(theta_1..k))) for k = i..n
```

To move toward a target, we compute:
```
d_theta = J_pseudo_inverse * (target - current_position)
```

The **pseudo-inverse** `J+ = J^T (J J^T)^(-1)` gives the minimum-norm joint angle change. We iterate, taking small steps, until the end-effector reaches the target (or we detect it's unreachable).

**Why pseudo-inverse?** The system is underdetermined (more joints than position DOF), so there are infinitely many solutions. The pseudo-inverse picks the one with the smallest joint angle change — a natural, smooth motion.

### Workspace Analysis

The **workspace** is the set of all points the end-effector can reach. For a 2D arm:
- **Outer boundary:** a circle of radius `L1 + L2 + ... + Ln` (all links extended)
- **Inner boundary:** depends on link lengths — if one link is longer than all others combined, there's a "hole" in the middle

Understanding the workspace helps you know before attempting IK whether a target is even reachable.

## Step-by-Step Breakdown

1. **Define the arm** — a list of link lengths. Each link connects two joints (or a joint to the end-effector).

2. **Forward kinematics** — accumulate angles, compute (x, y) of each joint and the end-effector. This is straightforward trig, but getting the angle accumulation right is critical.

3. **Analytical IK for 2 links** — implement the law-of-cosines closed-form solution. Handle the edge cases: target exactly at max reach (one solution), target unreachable (no solution), and the elbow-up/down ambiguity.

4. **Jacobian computation** — for each joint, compute how moving that joint affects the end-effector. This requires understanding partial derivatives geometrically.

5. **Numerical IK via Jacobian pseudo-inverse** — iteratively adjust joint angles to reduce the error between current end-effector position and target. Key decisions: step size (damping), convergence threshold, max iterations, and how to detect unreachable targets.

6. **Workspace visualization** — sample many joint configurations, plot all reachable end-effector positions, and overlay IK solutions to verify correctness.

## Learning Objectives

- Implement forward kinematics using cumulative angle sums and homogeneous transforms
- Derive and code analytical inverse kinematics for a 2-link planar arm
- Compute the Jacobian matrix and understand its geometric meaning
- Implement iterative IK using the Jacobian pseudo-inverse method
- Analyze workspace boundaries and reachability constraints
- Handle IK failure cases: unreachable targets, singularities, and multiple solutions

## Going Deeper

- **Singularities:** When the arm is fully extended or folded, the Jacobian loses rank and the pseudo-inverse blows up. Damped least squares (Levenberg-Marquardt) adds a regularization term to handle this gracefully.
- **3D kinematics:** Extends to 3D with rotation matrices (Euler angles, quaternions) and the Denavit-Hartenberg convention for parameterizing joint-link chains.
- **Cyclic Coordinate Descent (CCD):** An alternative IK method that adjusts one joint at a time — simpler to implement, good for real-time applications like game animation.
- **Builds on Day 006 (PID):** In a real robot, PID controllers (Day 006) would drive each joint motor to the angles computed by IK. The kinematics layer says "where to go," PID says "how to get there."
- **Builds on Day 007 (State Machine):** A pick-and-place robot might use a state machine (Day 007) to sequence: IDLE -> MOVE_TO_OBJECT -> GRASP -> MOVE_TO_TARGET -> RELEASE, with IK solving the arm configuration at each state.
