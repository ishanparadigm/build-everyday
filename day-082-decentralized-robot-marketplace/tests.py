"""
Day 82: Decentralized Robot Task Marketplace — Test Suite

Run with: python3 -m pytest tests.py -v
     or:  python3 tests.py
"""

import unittest
import math
from my_solution import (
    Position, Robot, Task, Bid, TaskStatus, TxType,
    Transaction, Block, MarketplaceBlockchain,
    MarketplaceContract, hungarian_assignment,
)


class TestPosition(unittest.TestCase):
    def test_distance_basic(self):
        p1 = Position(0, 0)
        p2 = Position(3, 4)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0)

    def test_distance_same_point(self):
        p = Position(5, 5)
        self.assertAlmostEqual(p.distance_to(p), 0.0)

    def test_distance_symmetric(self):
        p1 = Position(1, 2)
        p2 = Position(4, 6)
        self.assertAlmostEqual(p1.distance_to(p2), p2.distance_to(p1))


class TestRobot(unittest.TestCase):
    def setUp(self):
        self.robot = Robot("R1", Position(0, 0), battery=0.9, speed=2.0,
                           capabilities={"delivery", "inspection"},
                           max_payload=10.0)

    def test_estimate_cost_basic(self):
        task = Task("T1", "Deliver", Position(3, 4), {"delivery"},
                    payload_weight=2.0, reward=50.0, deadline=60.0,
                    poster_id="p1")
        cost = self.robot.estimate_cost(task)
        self.assertIsNotNone(cost)
        self.assertGreater(cost, 0)

    def test_estimate_cost_missing_capability(self):
        task = Task("T2", "Clean", Position(3, 4), {"cleaning"},
                    payload_weight=0.0, reward=50.0, deadline=60.0,
                    poster_id="p1")
        cost = self.robot.estimate_cost(task)
        self.assertIsNone(cost)

    def test_estimate_cost_overweight(self):
        task = Task("T3", "Heavy", Position(3, 4), {"delivery"},
                    payload_weight=15.0, reward=50.0, deadline=60.0,
                    poster_id="p1")
        cost = self.robot.estimate_cost(task)
        self.assertIsNone(cost)

    def test_estimate_cost_busy_robot(self):
        self.robot.assigned_task = "T_other"
        task = Task("T4", "Deliver", Position(3, 4), {"delivery"},
                    payload_weight=2.0, reward=50.0, deadline=60.0,
                    poster_id="p1")
        cost = self.robot.estimate_cost(task)
        self.assertIsNone(cost)

    def test_decide_bid_below_reward(self):
        task = Task("T5", "Deliver", Position(3, 4), {"delivery"},
                    payload_weight=2.0, reward=50.0, deadline=60.0,
                    poster_id="p1")
        bid = self.robot.decide_bid(task)
        self.assertIsNotNone(bid)
        self.assertLess(bid, 50.0)

    def test_decide_bid_too_expensive(self):
        """Task with tiny reward that doesn't cover cost."""
        task = Task("T6", "Far delivery", Position(100, 100), {"delivery"},
                    payload_weight=2.0, reward=0.5, deadline=60.0,
                    poster_id="p1")
        bid = self.robot.decide_bid(task)
        self.assertIsNone(bid)


class TestBlockchain(unittest.TestCase):
    def test_genesis_block(self):
        bc = MarketplaceBlockchain()
        self.assertEqual(len(bc.chain), 1)
        self.assertEqual(bc.chain[0].index, 0)

    def test_add_and_mine(self):
        bc = MarketplaceBlockchain()
        bc.add_transaction(TxType.POST_TASK, {"task_id": "T1"})
        bc.add_transaction(TxType.SUBMIT_BID, {"task_id": "T1", "robot_id": "R1"})
        block = bc.mine_block()
        self.assertEqual(block.index, 1)
        self.assertEqual(len(block.transactions), 2)

    def test_chain_integrity(self):
        bc = MarketplaceBlockchain()
        bc.add_transaction(TxType.POST_TASK, {"task_id": "T1"})
        bc.mine_block()
        bc.add_transaction(TxType.SUBMIT_BID, {"task_id": "T1"})
        bc.mine_block()
        self.assertTrue(bc.verify_chain())

    def test_get_all_transactions(self):
        bc = MarketplaceBlockchain()
        bc.add_transaction(TxType.POST_TASK, {"task_id": "T1"})
        bc.mine_block()
        bc.add_transaction(TxType.SUBMIT_BID, {"task_id": "T1"})
        bc.mine_block()
        txs = bc.get_all_transactions()
        self.assertEqual(len(txs), 2)


class TestMarketplaceContract(unittest.TestCase):
    def setUp(self):
        self.bc = MarketplaceBlockchain()
        self.contract = MarketplaceContract(self.bc)

        self.r1 = Robot("R1", Position(0, 0), battery=0.9, speed=2.0,
                        capabilities={"delivery"}, max_payload=10.0)
        self.r2 = Robot("R2", Position(10, 0), battery=0.7, speed=3.0,
                        capabilities={"delivery"}, max_payload=5.0)
        self.contract.register_robot(self.r1)
        self.contract.register_robot(self.r2)
        self.contract.fund_poster("poster1", 200.0)

    def _make_task(self, task_id: str = "T1", reward: float = 30.0) -> Task:
        return Task(task_id, "Test delivery", Position(3, 4),
                    {"delivery"}, payload_weight=2.0, reward=reward,
                    deadline=60.0, poster_id="poster1")

    def test_post_task_escrow(self):
        task = self._make_task(reward=50.0)
        self.assertTrue(self.contract.post_task(task))
        self.assertAlmostEqual(self.contract.poster_wallets["poster1"], 150.0)

    def test_post_task_insufficient_funds(self):
        task = self._make_task(reward=999.0)
        self.assertFalse(self.contract.post_task(task))

    def test_vickrey_auction_two_bidders(self):
        """Winner pays second-lowest price, not their own bid."""
        task = self._make_task(reward=50.0)
        self.contract.post_task(task)
        self.contract.open_bidding("T1")

        # R1 bids 5.0, R2 bids 8.0
        self.contract.submit_bid("R1", "T1", 5.0)
        self.contract.submit_bid("R2", "T1", 8.0)

        winner = self.contract.close_bidding_and_assign("T1")
        self.assertEqual(winner, "R1")
        # Vickrey: winner pays second price = 8.0
        self.assertEqual(task.reward, 8.0)

    def test_single_bidder_pays_own_price(self):
        task = self._make_task(reward=50.0)
        self.contract.post_task(task)
        self.contract.open_bidding("T1")
        self.contract.submit_bid("R1", "T1", 5.0)
        self.contract.close_bidding_and_assign("T1")
        self.assertEqual(task.reward, 5.0)

    def test_no_bids_cancels_and_refunds(self):
        task = self._make_task(reward=50.0)
        self.contract.post_task(task)
        self.contract.open_bidding("T1")
        winner = self.contract.close_bidding_and_assign("T1")
        self.assertIsNone(winner)
        self.assertEqual(task.status, TaskStatus.CANCELLED)
        self.assertAlmostEqual(self.contract.poster_wallets["poster1"], 200.0)

    def test_full_lifecycle(self):
        task = self._make_task(reward=50.0)
        self.contract.post_task(task)
        self.contract.open_bidding("T1")
        self.contract.submit_bid("R1", "T1", 5.0)
        self.contract.submit_bid("R2", "T1", 8.0)
        self.contract.close_bidding_and_assign("T1")
        self.contract.start_task("T1", "R1")
        self.contract.complete_task("T1", "R1")
        payment = self.contract.settle_payment("T1")
        self.assertIsNotNone(payment)
        self.assertEqual(task.status, TaskStatus.SETTLED)
        self.assertGreater(self.r1.wallet, 0)
        self.assertGreater(self.r1.reputation, 0.5)

    def test_fail_task_refunds_and_penalizes(self):
        task = self._make_task(reward=50.0)
        self.contract.post_task(task)
        self.contract.open_bidding("T1")
        self.contract.submit_bid("R1", "T1", 5.0)
        self.contract.close_bidding_and_assign("T1")
        self.contract.start_task("T1", "R1")

        initial_balance = self.contract.poster_wallets["poster1"]
        self.contract.fail_task("T1")

        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertLess(self.r1.reputation, 0.5)
        self.assertGreater(self.contract.poster_wallets["poster1"], initial_balance)


class TestHungarianAssignment(unittest.TestCase):
    def test_basic_assignment(self):
        robots = [
            Robot("R1", Position(0, 0), battery=0.9, speed=2.0,
                  capabilities={"delivery"}, max_payload=10.0),
            Robot("R2", Position(10, 0), battery=0.7, speed=3.0,
                  capabilities={"delivery"}, max_payload=5.0),
        ]
        tasks = [
            Task("T1", "Near R1", Position(1, 1), {"delivery"},
                 payload_weight=1.0, reward=50.0, deadline=60.0, poster_id="p"),
            Task("T2", "Near R2", Position(9, 0), {"delivery"},
                 payload_weight=1.0, reward=50.0, deadline=60.0, poster_id="p"),
        ]
        assignments = hungarian_assignment(robots, tasks)
        self.assertEqual(len(assignments), 2)
        # R1 should get T1 (closer), R2 should get T2 (closer)
        assigned = {r: t for r, t, _ in assignments}
        self.assertEqual(assigned["R1"], "T1")
        self.assertEqual(assigned["R2"], "T2")


if __name__ == "__main__":
    unittest.main()
