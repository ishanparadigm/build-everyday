"""
Day 008: Forward and Inverse Kinematics — Your Implementation

Implement forward and inverse kinematics for 2D robotic arms.
Run tests with: python3 -m pytest tests.py -v

Hints:
- Forward kinematics: accumulate angles as you traverse the chain
- Analytical IK: law of cosines gives theta2, then geometry gives theta1
- Jacobian: each column tells you how one joint affects the end-effector
- Numerical IK: iterate d_theta = J_pseudoinverse * error until converged
"""

import math
from typing import List, Tuple, Optional


class RobotArm:
    """A planar (2D) robotic arm defined by link lengths."""

    def __init__(self, link_lengths: List[float]):
        """
        Args:
            link_lengths: Length of each link, from base outward.
        """
        if not link_lengths or any(l <= 0 for l in link_lengths):
            raise ValueError("All link lengths must be positive")
        self.link_lengths = list(link_lengths)
        self.n_joints = len(link_lengths)

    @property
    def max_reach(self) -> float:
        """Maximum distance the end-effector can reach."""
        return sum(self.link_lengths)

    @property
    def min_reach(self) -> float:
        """Minimum distance from base the end-effector must be at."""
        longest = max(self.link_lengths)
        rest = sum(self.link_lengths) - longest
        return max(0.0, longest - rest)


def forward_kinematics(
    arm: RobotArm,
    joint_angles: List[float]
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """Compute all joint positions and end-effector position.

    Args:
        arm: The robot arm definition.
        joint_angles: Angle (radians) for each joint.

    Returns:
        (joint_positions, end_effector):
            joint_positions: list of (x, y) for each joint including base at (0,0)
            end_effector: (x, y) of the tip

    Hint: Each joint angle is relative to the previous link.
          The absolute angle is the cumulative sum of all preceding angles.
    """
    raise NotImplementedError("TODO: implement forward kinematics")


def inverse_kinematics_2link(
    arm: RobotArm,
    target: Tuple[float, float],
    elbow_up: bool = True
) -> Optional[Tuple[float, float]]:
    """Closed-form IK for a 2-link planar arm.

    Args:
        arm: Must have exactly 2 links.
        target: Desired (x, y) position for end-effector.
        elbow_up: If True, return the elbow-up solution; else elbow-down.

    Returns:
        (theta1, theta2) in radians, or None if unreachable.

    Hint: Use law of cosines to find theta2, then geometry for theta1.
          Check reachability first: |L1-L2| <= dist <= L1+L2
    """
    raise NotImplementedError("TODO: implement analytical IK")


def compute_jacobian(
    arm: RobotArm,
    joint_angles: List[float]
) -> List[List[float]]:
    """Compute the 2 x n Jacobian matrix for a planar arm.

    Args:
        arm: The robot arm.
        joint_angles: Current joint angles (radians).

    Returns:
        2 x n Jacobian matrix as list of lists.

    Hint: J[0][i] = partial(x)/partial(theta_i) — how does moving
          joint i affect the x-position of the end-effector?
          Think about which links are "downstream" of joint i.
    """
    raise NotImplementedError("TODO: implement Jacobian computation")


def inverse_kinematics_numerical(
    arm: RobotArm,
    target: Tuple[float, float],
    initial_angles: Optional[List[float]] = None,
    max_iterations: int = 200,
    tolerance: float = 1e-4,
    damping: float = 1e-3,
    step_scale: float = 1.0
) -> Optional[Tuple[List[float], int, float]]:
    """Iterative IK using the damped Jacobian pseudo-inverse.

    Args:
        arm: The robot arm.
        target: Desired (x, y) position.
        initial_angles: Starting joint angles (defaults to zeros).
        max_iterations: Max iterations before giving up.
        tolerance: Position error threshold for convergence.
        damping: Damping factor for pseudo-inverse.
        step_scale: Step size multiplier.

    Returns:
        (joint_angles, iterations, final_error) or None if unreachable.

    Hint: Each iteration: FK -> error -> Jacobian -> pseudo-inverse -> update.
          Pseudo-inverse of 2xn J: J+ = J^T (J J^T + lambda^2 I)^(-1)
          You only need to invert a 2x2 matrix!
    """
    raise NotImplementedError("TODO: implement numerical IK")


def compute_workspace(
    arm: RobotArm,
    samples_per_joint: int = 20
) -> List[Tuple[float, float]]:
    """Sample the reachable workspace by sweeping joint angles.

    Args:
        arm: The robot arm.
        samples_per_joint: Number of angle samples per joint.

    Returns:
        List of (x, y) end-effector positions.

    Hint: For each combination of angles in [-pi, pi], run FK
          and collect the end-effector position.
    """
    raise NotImplementedError("TODO: implement workspace sampling")


if __name__ == "__main__":
    # Test your implementation here

    # --- Forward Kinematics ---
    arm = RobotArm([1.0, 0.8])
    angles = [math.radians(45), math.radians(-30)]
    positions, end = forward_kinematics(arm, angles)
    print(f"FK: angles={[round(math.degrees(a)) for a in angles]} -> end={end}")

    # --- Analytical IK ---
    target = (1.2, 0.6)
    result = inverse_kinematics_2link(arm, target)
    if result:
        t1, t2 = result
        print(f"IK: target={target} -> angles=({math.degrees(t1):.1f}, {math.degrees(t2):.1f}) deg")
        # Verify round-trip
        _, verify = forward_kinematics(arm, [t1, t2])
        print(f"  Verification: {verify}")

    # --- Numerical IK ---
    arm3 = RobotArm([1.0, 0.7, 0.5])
    result = inverse_kinematics_numerical(arm3, (1.5, 0.8))
    if result:
        angles, iters, err = result
        print(f"Numerical IK: {iters} iters, error={err:.2e}")
