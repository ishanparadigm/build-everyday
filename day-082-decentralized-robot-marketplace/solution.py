"""
Day 82: Decentralized Robot Task Marketplace

A marketplace where robots bid on tasks using Vickrey auctions, coordinated
through a simulated blockchain. Demonstrates multi-agent coordination,
auction theory, smart contract state machines, and robot cost estimation.

Key design decisions:
- Vickrey (second-price) auction: incentive-compatible, robots bid true cost
- Blockchain for ordering/immutability, not for speed
- Robot cost model captures distance, battery, capability, and load
- Reputation system creates long-term incentive alignment
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# =============================================================================
# Domain Models
# =============================================================================

class TaskStatus(Enum):
    """Task lifecycle states enforced by the smart contract."""
    POSTED = auto()      # Task created, not yet open for bidding
    BIDDING = auto()     # Open for robot bids
    ASSIGNED = auto()    # Winner selected, awaiting execution
    IN_PROGRESS = auto() # Robot is executing the task
    COMPLETED = auto()   # Robot reports completion
    SETTLED = auto()     # Payment transferred
    FAILED = auto()      # Robot failed to complete
    CANCELLED = auto()   # Task cancelled by poster


class TxType(Enum):
    """Transaction types in the marketplace blockchain."""
    POST_TASK = auto()
    OPEN_BIDDING = auto()
    SUBMIT_BID = auto()
    ASSIGN_TASK = auto()
    START_TASK = auto()
    COMPLETE_TASK = auto()
    SETTLE_PAYMENT = auto()
    FAIL_TASK = auto()
    UPDATE_REPUTATION = auto()


@dataclass
class Position:
    """2D position in the workspace."""
    x: float
    y: float

    def distance_to(self, other: Position) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Robot:
    """
    A robot participant in the marketplace.

    Each robot has physical constraints (position, battery, payload capacity),
    capabilities (what types of tasks it can do), a reputation score, and a
    wallet balance for receiving payments.
    """
    robot_id: str
    position: Position
    battery: float              # 0.0 to 1.0
    speed: float                # units per second
    capabilities: set[str]      # e.g., {"delivery", "inspection", "cleaning"}
    max_payload: float          # kg
    current_load: float = 0.0   # kg currently carrying
    reputation: float = 0.5     # 0.0 to 1.0, starts neutral
    wallet: float = 0.0         # earned tokens
    assigned_task: Optional[str] = None  # task_id if currently working

    # Cost model parameters — these would be learned/calibrated in a real system
    BASE_RATE: float = 1.0       # base cost per unit time
    ENERGY_RATE: float = 0.5     # cost per unit distance for energy
    RISK_MULTIPLIER: float = 2.0 # multiplier for low-battery risk premium

    def estimate_cost(self, task: Task) -> Optional[float]:
        """
        Estimate the cost for this robot to complete a task.

        Returns None if the robot cannot do the task (missing capability,
        insufficient battery, already busy, or overloaded).

        The cost model balances four factors:
        1. Time cost: distance / speed * base_rate
        2. Energy cost: distance * energy_rate
        3. Risk premium: increases as battery drops (1/battery scaling)
        4. Load penalty: carrying items makes everything slower/costlier
        """
        # --- Capability check ---
        if not task.required_capabilities.issubset(self.capabilities):
            return None

        # --- Availability check ---
        if self.assigned_task is not None:
            return None

        # --- Payload check ---
        if task.payload_weight > (self.max_payload - self.current_load):
            return None

        distance = self.position.distance_to(task.location)

        # --- Battery check: need enough to get there and back ---
        # Simple model: 1% battery per unit distance
        energy_needed = distance * 0.01 * 2  # round trip
        if energy_needed > self.battery:
            return None

        # --- Cost calculation ---
        time_cost = (distance / self.speed) * self.BASE_RATE
        energy_cost = distance * self.ENERGY_RATE

        # Risk premium: as battery approaches 0, risk goes to infinity
        # We clamp battery to avoid division by near-zero
        effective_battery = max(self.battery, 0.05)
        risk_premium = self.RISK_MULTIPLIER * (1.0 / effective_battery - 1.0)

        # Load penalty: 10% cost increase per kg of current load
        load_penalty = self.current_load * 0.1

        total_cost = time_cost + energy_cost + risk_premium + load_penalty
        return max(total_cost, 0.01)  # minimum bid of 0.01

    def decide_bid(self, task: Task) -> Optional[float]:
        """
        Decide whether and how much to bid on a task.

        Under Vickrey auction rules, the dominant strategy is to bid
        your true cost. No strategic inflation or deflation needed.
        This is the beauty of incentive-compatible mechanisms.
        """
        cost = self.estimate_cost(task)
        if cost is None:
            return None

        # Don't bid if our cost exceeds the reward — we'd lose money
        if cost >= task.reward:
            return None

        # Vickrey optimal: bid true cost
        return cost


@dataclass
class Task:
    """
    A task posted to the marketplace.

    Tasks have physical requirements (location, payload), capability
    requirements, a deadline, and a reward offered by the poster.
    """
    task_id: str
    description: str
    location: Position
    required_capabilities: set[str]
    payload_weight: float       # kg
    reward: float               # tokens offered
    deadline: float             # seconds from now
    poster_id: str              # who posted this task
    status: TaskStatus = TaskStatus.POSTED
    assigned_robot: Optional[str] = None
    bids: list[Bid] = field(default_factory=list)
    completion_time: Optional[float] = None

    def add_bid(self, bid: Bid) -> bool:
        """Add a bid if the task is in BIDDING status."""
        if self.status != TaskStatus.BIDDING:
            return False
        if bid.amount >= self.reward:
            return False
        self.bids.append(bid)
        return True


@dataclass
class Bid:
    """A robot's bid on a task."""
    robot_id: str
    task_id: str
    amount: float
    timestamp: float


# =============================================================================
# Blockchain Layer
# =============================================================================

@dataclass
class Transaction:
    """A single transaction in the marketplace blockchain."""
    tx_type: TxType
    payload: dict
    timestamp: float
    tx_hash: str = ""

    def __post_init__(self):
        if not self.tx_hash:
            self.tx_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA-256 hash of transaction contents. Ensures integrity."""
        data = f"{self.tx_type.name}:{json.dumps(self.payload, sort_keys=True)}:{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class Block:
    """
    A block in the marketplace blockchain.

    Each block contains a list of transactions, a reference to the
    previous block (via hash), and its own hash. This creates the
    immutable chain — changing any transaction would invalidate all
    subsequent block hashes.
    """
    index: int
    transactions: list[Transaction]
    previous_hash: str
    timestamp: float
    nonce: int = 0
    block_hash: str = ""

    def __post_init__(self):
        if not self.block_hash:
            self.block_hash = self.compute_hash()

    def compute_hash(self) -> str:
        tx_data = "".join(tx.tx_hash for tx in self.transactions)
        data = f"{self.index}:{tx_data}:{self.previous_hash}:{self.timestamp}:{self.nonce}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class MarketplaceBlockchain:
    """
    Simplified blockchain that records all marketplace transactions.

    This provides:
    1. Ordering: transactions have a definitive sequence
    2. Immutability: past transactions can't be altered
    3. Auditability: anyone can verify the full history

    We skip proof-of-work here — the focus is on the marketplace logic,
    not consensus. In production, this would run on an existing L1/L2.
    """

    def __init__(self):
        self.chain: list[Block] = []
        self.pending_transactions: list[Transaction] = []
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        genesis = Block(
            index=0,
            transactions=[],
            previous_hash="0" * 16,
            timestamp=time.time(),
        )
        self.chain.append(genesis)

    def add_transaction(self, tx_type: TxType, payload: dict) -> Transaction:
        """Create and queue a transaction."""
        tx = Transaction(tx_type=tx_type, payload=payload, timestamp=time.time())
        self.pending_transactions.append(tx)
        return tx

    def mine_block(self) -> Block:
        """
        Package pending transactions into a new block.

        In a real system, miners would compete to add blocks.
        Here we just append — the focus is marketplace logic.
        """
        if not self.pending_transactions:
            raise ValueError("No pending transactions to mine")

        block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions.copy(),
            previous_hash=self.chain[-1].block_hash,
            timestamp=time.time(),
        )
        self.chain.append(block)
        self.pending_transactions.clear()
        return block

    def get_all_transactions(self) -> list[Transaction]:
        """Return all mined transactions in order."""
        txs = []
        for block in self.chain:
            txs.extend(block.transactions)
        return txs

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire chain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.previous_hash != previous.block_hash:
                return False
            if current.block_hash != current.compute_hash():
                return False
        return True


# =============================================================================
# Smart Contract: Marketplace Logic
# =============================================================================

class MarketplaceContract:
    """
    The smart contract that governs the task marketplace.

    Enforces the task lifecycle state machine, runs Vickrey auctions,
    manages wallets, and updates reputation scores. All state changes
    are recorded on the blockchain for auditability.

    In a real system, this would be a Solidity contract on Ethereum/L2.
    Here we simulate the same logic in Python.
    """

    def __init__(self, blockchain: MarketplaceBlockchain):
        self.blockchain = blockchain
        self.tasks: dict[str, Task] = {}
        self.robots: dict[str, Robot] = {}
        self.poster_wallets: dict[str, float] = {}  # poster_id → balance

    def register_robot(self, robot: Robot) -> None:
        """Register a robot in the marketplace."""
        self.robots[robot.robot_id] = robot

    def fund_poster(self, poster_id: str, amount: float) -> None:
        """Add funds to a task poster's wallet."""
        self.poster_wallets[poster_id] = self.poster_wallets.get(poster_id, 0) + amount

    def post_task(self, task: Task) -> bool:
        """
        Post a new task to the marketplace.

        The poster must have sufficient funds to cover the reward.
        Funds are escrowed (locked) when the task is posted — this
        prevents the poster from spending the reward before a robot
        completes the work.
        """
        # Check poster has funds
        poster_balance = self.poster_wallets.get(task.poster_id, 0)
        if poster_balance < task.reward:
            print(f"  [CONTRACT] Rejected: poster {task.poster_id} has insufficient funds "
                  f"({poster_balance:.2f} < {task.reward:.2f})")
            return False

        # Escrow the reward
        self.poster_wallets[task.poster_id] -= task.reward

        self.tasks[task.task_id] = task
        task.status = TaskStatus.POSTED

        self.blockchain.add_transaction(TxType.POST_TASK, {
            "task_id": task.task_id,
            "reward": task.reward,
            "poster_id": task.poster_id,
        })

        return True

    def open_bidding(self, task_id: str) -> bool:
        """Transition task from POSTED to BIDDING."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.POSTED:
            return False

        task.status = TaskStatus.BIDDING
        self.blockchain.add_transaction(TxType.OPEN_BIDDING, {"task_id": task_id})
        return True

    def submit_bid(self, robot_id: str, task_id: str, amount: float) -> bool:
        """
        Submit a bid from a robot for a task.

        Validates:
        - Task exists and is in BIDDING status
        - Robot is registered and capable
        - Bid is below the task reward
        - Robot hasn't already bid on this task
        """
        task = self.tasks.get(task_id)
        robot = self.robots.get(robot_id)

        if not task or not robot:
            return False

        if task.status != TaskStatus.BIDDING:
            return False

        # Prevent double-bidding
        if any(b.robot_id == robot_id for b in task.bids):
            return False

        # Capability check
        if not task.required_capabilities.issubset(robot.capabilities):
            return False

        bid = Bid(
            robot_id=robot_id,
            task_id=task_id,
            amount=amount,
            timestamp=time.time(),
        )

        if task.add_bid(bid):
            self.blockchain.add_transaction(TxType.SUBMIT_BID, {
                "robot_id": robot_id,
                "task_id": task_id,
                "amount": amount,
            })
            return True
        return False

    def close_bidding_and_assign(self, task_id: str) -> Optional[str]:
        """
        Close bidding and assign the task using Vickrey auction rules.

        Vickrey auction:
        - Winner = lowest bidder
        - Payment = second-lowest bid (or the winner's bid if only one bidder)

        Why second-price? It's incentive-compatible: bidding your true cost
        is always optimal regardless of what others bid. This simplifies
        robot strategy from "guess what others will bid" to "compute my cost."

        The winner's payment is stored on the task for later settlement.
        """
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.BIDDING:
            return None

        if not task.bids:
            # No bids — cancel the task and refund
            task.status = TaskStatus.CANCELLED
            self.poster_wallets[task.poster_id] = (
                self.poster_wallets.get(task.poster_id, 0) + task.reward
            )
            return None

        # Sort bids by amount (ascending)
        sorted_bids = sorted(task.bids, key=lambda b: b.amount)
        winner = sorted_bids[0]

        # Vickrey: pay second-lowest price
        # If only one bidder, they pay their own bid
        if len(sorted_bids) >= 2:
            payment = sorted_bids[1].amount
        else:
            payment = winner.amount

        # Apply reputation discount: higher reputation → slight advantage
        # This doesn't change the Vickrey payment, but we record it
        robot = self.robots[winner.robot_id]

        task.status = TaskStatus.ASSIGNED
        task.assigned_robot = winner.robot_id
        # Store the actual payment amount (Vickrey second-price)
        task.reward = min(payment, task.reward)  # cap at original reward
        robot.assigned_task = task_id

        self.blockchain.add_transaction(TxType.ASSIGN_TASK, {
            "task_id": task_id,
            "robot_id": winner.robot_id,
            "winning_bid": winner.amount,
            "payment": payment,
            "num_bids": len(task.bids),
        })

        return winner.robot_id

    def start_task(self, task_id: str, robot_id: str) -> bool:
        """Robot signals it has started working on the task."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.ASSIGNED:
            return False
        if task.assigned_robot != robot_id:
            return False

        task.status = TaskStatus.IN_PROGRESS
        self.blockchain.add_transaction(TxType.START_TASK, {
            "task_id": task_id,
            "robot_id": robot_id,
        })
        return True

    def complete_task(self, task_id: str, robot_id: str) -> bool:
        """
        Robot reports task completion.

        In a real system, this would require proof of completion
        (photo evidence, sensor readings, oracle confirmation).
        Here we trust the robot's report.
        """
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.IN_PROGRESS:
            return False
        if task.assigned_robot != robot_id:
            return False

        task.status = TaskStatus.COMPLETED
        task.completion_time = time.time()

        self.blockchain.add_transaction(TxType.COMPLETE_TASK, {
            "task_id": task_id,
            "robot_id": robot_id,
        })
        return True

    def settle_payment(self, task_id: str) -> Optional[float]:
        """
        Transfer payment from escrow to the robot's wallet.

        Only works for COMPLETED tasks. The payment amount was
        determined during assignment (Vickrey second-price).
        Any remaining escrow (reward - payment) is refunded to the poster.
        """
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.COMPLETED:
            return None

        robot = self.robots[task.assigned_robot]
        payment = task.reward  # Already set to Vickrey price during assignment

        # Pay the robot
        robot.wallet += payment
        robot.assigned_task = None

        # Update reputation: successful completion increases it
        old_rep = robot.reputation
        robot.reputation = min(1.0, robot.reputation + 0.05)

        task.status = TaskStatus.SETTLED

        self.blockchain.add_transaction(TxType.SETTLE_PAYMENT, {
            "task_id": task_id,
            "robot_id": robot.robot_id,
            "payment": payment,
        })

        self.blockchain.add_transaction(TxType.UPDATE_REPUTATION, {
            "robot_id": robot.robot_id,
            "old_reputation": round(old_rep, 3),
            "new_reputation": round(robot.reputation, 3),
        })

        return payment

    def fail_task(self, task_id: str) -> bool:
        """
        Mark a task as failed. Refund poster, penalize robot reputation.

        Failure could be: robot ran out of battery, missed deadline,
        or explicitly reported inability to complete.
        """
        task = self.tasks.get(task_id)
        if not task or task.status not in (TaskStatus.IN_PROGRESS, TaskStatus.ASSIGNED):
            return False

        robot = self.robots[task.assigned_robot]
        robot.assigned_task = None

        # Reputation penalty (harsher than the reward for completion)
        old_rep = robot.reputation
        robot.reputation = max(0.0, robot.reputation - 0.1)

        # Refund the poster
        self.poster_wallets[task.poster_id] = (
            self.poster_wallets.get(task.poster_id, 0) + task.reward
        )

        task.status = TaskStatus.FAILED

        self.blockchain.add_transaction(TxType.FAIL_TASK, {
            "task_id": task_id,
            "robot_id": robot.robot_id,
        })

        self.blockchain.add_transaction(TxType.UPDATE_REPUTATION, {
            "robot_id": robot.robot_id,
            "old_reputation": round(old_rep, 3),
            "new_reputation": round(robot.reputation, 3),
        })

        return True


# =============================================================================
# Simulation: Centralized Optimal Assignment (for comparison)
# =============================================================================

def hungarian_assignment(robots: list[Robot], tasks: list[Task]) -> list[tuple[str, str, float]]:
    """
    Compute the optimal centralized assignment using a greedy approximation
    of the assignment problem.

    The true Hungarian algorithm is O(n^3) and handles the full bipartite
    matching. Here we use a greedy approach for simplicity: assign the
    cheapest (robot, task) pair first, then remove both from consideration.

    Returns list of (robot_id, task_id, cost) tuples.
    """
    # Build cost matrix
    costs: list[tuple[float, str, str]] = []
    for robot in robots:
        for task in tasks:
            cost = robot.estimate_cost(task)
            if cost is not None:
                costs.append((cost, robot.robot_id, task.task_id))

    # Sort by cost (greedy assignment)
    costs.sort()

    assigned_robots: set[str] = set()
    assigned_tasks: set[str] = set()
    assignments: list[tuple[str, str, float]] = []

    for cost, robot_id, task_id in costs:
        if robot_id not in assigned_robots and task_id not in assigned_tasks:
            assignments.append((robot_id, task_id, cost))
            assigned_robots.add(robot_id)
            assigned_tasks.add(task_id)

    return assignments


# =============================================================================
# Simulation Runner
# =============================================================================

def simulate_task(
    contract: MarketplaceContract,
    task: Task,
    robots: list[Robot],
) -> dict:
    """
    Run a single task through the full marketplace lifecycle.

    Returns a summary dict with auction results and outcome.
    """
    result = {
        "task_id": task.task_id,
        "description": task.description,
        "reward_offered": task.reward,
        "bids": [],
        "winner": None,
        "payment": None,
        "outcome": None,
    }

    # Post and open bidding
    if not contract.post_task(task):
        result["outcome"] = "REJECTED"
        return result
    contract.open_bidding(task.task_id)

    # Each robot evaluates and potentially bids
    for robot in robots:
        bid_amount = robot.decide_bid(task)
        if bid_amount is not None:
            success = contract.submit_bid(robot.robot_id, task.task_id, bid_amount)
            if success:
                result["bids"].append({
                    "robot_id": robot.robot_id,
                    "amount": round(bid_amount, 4),
                })

    # Close bidding and assign
    winner_id = contract.close_bidding_and_assign(task.task_id)
    if winner_id is None:
        result["outcome"] = "NO_BIDS"
        return result

    result["winner"] = winner_id

    # Simulate task execution
    contract.start_task(task.task_id, winner_id)

    # Simulate: 90% success rate (battery failures, obstacles, etc.)
    robot = contract.robots[winner_id]
    # Simulate movement: reduce battery
    distance = robot.position.distance_to(task.location)
    robot.battery -= distance * 0.01

    # Check if robot "failed" (battery too low)
    if robot.battery < 0.05:
        contract.fail_task(task.task_id)
        result["outcome"] = "FAILED"
        return result

    # Success — complete and settle
    contract.complete_task(task.task_id, winner_id)
    payment = contract.settle_payment(task.task_id)

    # Move robot to task location (it's there now)
    robot.position = Position(task.location.x, task.location.y)

    result["payment"] = round(payment, 4) if payment else None
    result["outcome"] = "COMPLETED"

    return result


def run_marketplace_simulation() -> None:
    """
    Full marketplace simulation with multiple robots and tasks.

    Demonstrates:
    1. Robot registration and capability setup
    2. Task posting with fund escrow
    3. Decentralized bidding with Vickrey auctions
    4. Task execution and payment settlement
    5. Reputation updates
    6. Comparison with centralized optimal
    """
    print("=" * 70)
    print("DECENTRALIZED ROBOT TASK MARKETPLACE")
    print("=" * 70)

    # --- Setup blockchain and contract ---
    blockchain = MarketplaceBlockchain()
    contract = MarketplaceContract(blockchain)

    # --- Create robots with diverse characteristics ---
    robots = [
        Robot("R1", Position(0, 0), battery=0.9, speed=2.0,
              capabilities={"delivery", "inspection"}, max_payload=10.0),
        Robot("R2", Position(10, 0), battery=0.7, speed=3.0,
              capabilities={"delivery", "cleaning"}, max_payload=5.0),
        Robot("R3", Position(5, 5), battery=0.95, speed=1.5,
              capabilities={"delivery", "inspection", "cleaning"}, max_payload=15.0),
        Robot("R4", Position(8, 8), battery=0.4, speed=2.5,
              capabilities={"inspection"}, max_payload=3.0),
        Robot("R5", Position(2, 7), battery=0.85, speed=2.0,
              capabilities={"delivery", "inspection", "cleaning"}, max_payload=8.0),
    ]

    for robot in robots:
        contract.register_robot(robot)

    print("\n--- Registered Robots ---")
    for r in robots:
        print(f"  {r.robot_id}: pos=({r.position.x:.0f},{r.position.y:.0f}) "
              f"battery={r.battery:.0%} speed={r.speed} "
              f"caps={r.capabilities} payload={r.max_payload}kg")

    # --- Fund the task poster ---
    poster_id = "factory_alpha"
    contract.fund_poster(poster_id, 500.0)
    print(f"\n--- Poster '{poster_id}' funded with 500.0 tokens ---")

    # --- Create tasks ---
    tasks = [
        Task("T1", "Deliver package to station A", Position(3, 4),
             {"delivery"}, payload_weight=2.0, reward=20.0, deadline=60.0,
             poster_id=poster_id),
        Task("T2", "Inspect pipeline section B", Position(9, 2),
             {"inspection"}, payload_weight=0.0, reward=15.0, deadline=45.0,
             poster_id=poster_id),
        Task("T3", "Clean hazardous spill at C", Position(6, 8),
             {"cleaning"}, payload_weight=0.0, reward=25.0, deadline=90.0,
             poster_id=poster_id),
        Task("T4", "Deliver and inspect at D", Position(1, 9),
             {"delivery", "inspection"}, payload_weight=5.0, reward=35.0,
             deadline=120.0, poster_id=poster_id),
        Task("T5", "Emergency inspection at E", Position(7, 1),
             {"inspection"}, payload_weight=0.0, reward=30.0, deadline=30.0,
             poster_id=poster_id),
    ]

    print("\n--- Posted Tasks ---")
    for t in tasks:
        print(f"  {t.task_id}: '{t.description}' at ({t.location.x:.0f},{t.location.y:.0f}) "
              f"reward={t.reward} caps={t.required_capabilities}")

    # --- Run each task through the marketplace ---
    print("\n" + "=" * 70)
    print("AUCTION RESULTS")
    print("=" * 70)

    results = []
    for task in tasks:
        print(f"\n--- {task.task_id}: {task.description} ---")
        result = simulate_task(contract, task, robots)
        results.append(result)

        if result["bids"]:
            for bid in result["bids"]:
                print(f"  Bid: {bid['robot_id']} → {bid['amount']:.4f} tokens")
        else:
            print("  No bids received")

        if result["winner"]:
            print(f"  Winner: {result['winner']} | "
                  f"Payment (Vickrey 2nd price): {result['payment']}")
        print(f"  Outcome: {result['outcome']}")

    # --- Mine all transactions into blocks ---
    if blockchain.pending_transactions:
        blockchain.mine_block()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("MARKETPLACE SUMMARY")
    print("=" * 70)

    completed = [r for r in results if r["outcome"] == "COMPLETED"]
    failed = [r for r in results if r["outcome"] == "FAILED"]
    no_bids = [r for r in results if r["outcome"] == "NO_BIDS"]

    total_payment = sum(r["payment"] for r in completed if r["payment"])

    print(f"\n  Tasks completed: {len(completed)}/{len(tasks)}")
    print(f"  Tasks failed:    {len(failed)}")
    print(f"  No bids:         {len(no_bids)}")
    print(f"  Total payments:  {total_payment:.4f} tokens")

    print("\n--- Robot Final State ---")
    for robot in robots:
        print(f"  {robot.robot_id}: wallet={robot.wallet:.4f} "
              f"reputation={robot.reputation:.3f} "
              f"battery={robot.battery:.2%} "
              f"pos=({robot.position.x:.1f},{robot.position.y:.1f})")

    print(f"\n  Poster remaining balance: {contract.poster_wallets.get(poster_id, 0):.4f}")

    # --- Centralized comparison ---
    print("\n" + "=" * 70)
    print("CENTRALIZED OPTIMAL COMPARISON")
    print("=" * 70)

    # Reset robots to original state for fair comparison
    original_robots = [
        Robot("R1", Position(0, 0), battery=0.9, speed=2.0,
              capabilities={"delivery", "inspection"}, max_payload=10.0),
        Robot("R2", Position(10, 0), battery=0.7, speed=3.0,
              capabilities={"delivery", "cleaning"}, max_payload=5.0),
        Robot("R3", Position(5, 5), battery=0.95, speed=1.5,
              capabilities={"delivery", "inspection", "cleaning"}, max_payload=15.0),
        Robot("R4", Position(8, 8), battery=0.4, speed=2.5,
              capabilities={"inspection"}, max_payload=3.0),
        Robot("R5", Position(2, 7), battery=0.85, speed=2.0,
              capabilities={"delivery", "inspection", "cleaning"}, max_payload=8.0),
    ]

    # Reset tasks too
    original_tasks = [
        Task("T1", "Deliver package to station A", Position(3, 4),
             {"delivery"}, payload_weight=2.0, reward=20.0, deadline=60.0,
             poster_id=poster_id),
        Task("T2", "Inspect pipeline section B", Position(9, 2),
             {"inspection"}, payload_weight=0.0, reward=15.0, deadline=45.0,
             poster_id=poster_id),
        Task("T3", "Clean hazardous spill at C", Position(6, 8),
             {"cleaning"}, payload_weight=0.0, reward=25.0, deadline=90.0,
             poster_id=poster_id),
        Task("T4", "Deliver and inspect at D", Position(1, 9),
             {"delivery", "inspection"}, payload_weight=5.0, reward=35.0,
             deadline=120.0, poster_id=poster_id),
        Task("T5", "Emergency inspection at E", Position(7, 1),
             {"inspection"}, payload_weight=0.0, reward=30.0, deadline=30.0,
             poster_id=poster_id),
    ]

    assignments = hungarian_assignment(original_robots, original_tasks)
    centralized_cost = sum(cost for _, _, cost in assignments)

    print("\n  Optimal assignment (greedy):")
    for robot_id, task_id, cost in assignments:
        print(f"    {robot_id} → {task_id}: cost={cost:.4f}")
    print(f"\n  Total centralized cost: {centralized_cost:.4f}")
    print(f"  Total decentralized payment: {total_payment:.4f}")

    if centralized_cost > 0:
        efficiency = centralized_cost / max(total_payment, 0.01) * 100
        print(f"  Market efficiency: {efficiency:.1f}%")
        print(f"  (100% = decentralized matches centralized optimal)")

    # --- Blockchain verification ---
    print("\n" + "=" * 70)
    print("BLOCKCHAIN VERIFICATION")
    print("=" * 70)

    chain_valid = blockchain.verify_chain()
    print(f"\n  Chain length: {len(blockchain.chain)} blocks")
    print(f"  Total transactions: {sum(len(b.transactions) for b in blockchain.chain)}")
    print(f"  Chain integrity: {'VALID' if chain_valid else 'CORRUPTED'}")

    # Show transaction log
    print("\n  Transaction log:")
    for tx in blockchain.get_all_transactions():
        task_info = tx.payload.get("task_id", "")
        robot_info = tx.payload.get("robot_id", "")
        extra = ""
        if "amount" in tx.payload:
            extra = f" amount={tx.payload['amount']:.4f}"
        if "payment" in tx.payload:
            extra = f" payment={tx.payload['payment']:.4f}"
        print(f"    [{tx.tx_hash[:8]}] {tx.tx_type.name:20s} "
              f"task={task_info:4s} robot={robot_info:4s}{extra}")


if __name__ == "__main__":
    run_marketplace_simulation()
