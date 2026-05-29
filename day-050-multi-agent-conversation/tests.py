"""
Tests for Multi-Agent Conversation System

Run with: python3 -m pytest tests.py -v
     or: python3 tests.py
"""

import unittest
from my_solution import (
    Message, Agent, AgentSelector, Topology,
    MaxRoundsTermination, ConsensusTermination, KeywordTermination,
    ConversationManager,
)


class TestMessage(unittest.TestCase):
    """Test the Message dataclass."""

    def test_message_creation(self):
        msg = Message(sender="Alice", content="Hello")
        self.assertEqual(msg.sender, "Alice")
        self.assertEqual(msg.content, "Hello")

    def test_message_str(self):
        msg = Message(sender="Bob", content="World")
        result = str(msg)
        self.assertIn("Bob", result)
        self.assertIn("World", result)

    def test_message_metadata(self):
        msg = Message(sender="sys", content="test", metadata={"confidence": 0.9})
        self.assertEqual(msg.metadata["confidence"], 0.9)


class TestAgent(unittest.TestCase):
    """Test the Agent class."""

    def test_agent_creation(self):
        agent = Agent(name="Dev", role="Developer")
        self.assertEqual(agent.name, "Dev")
        self.assertEqual(agent.role, "Developer")

    def test_agent_responds_to_empty_history(self):
        agent = Agent(name="Dev", role="Developer")
        response = agent.respond([])
        self.assertIsNotNone(response)

    def test_agent_behavior_trigger(self):
        agent = Agent(name="Dev", role="Developer", behaviors={
            "review": "Here is my review."
        })
        history = [Message(sender="user", content="Please review this code")]
        response = agent.respond(history)
        self.assertEqual(response, "Here is my review.")

    def test_agent_no_self_response(self):
        """Agent should return None when the last message is from itself."""
        agent = Agent(name="Dev", role="Developer")
        history = [Message(sender="Dev", content="I already spoke")]
        response = agent.respond(history)
        self.assertIsNone(response)

    def test_agent_default_response(self):
        """Agent should give a default response when no behaviors match."""
        agent = Agent(name="Dev", role="Developer", behaviors={
            "specific_trigger": "Triggered response"
        })
        history = [Message(sender="Other", content="Something unrelated")]
        response = agent.respond(history)
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "Triggered response")


class TestAgentSelector(unittest.TestCase):
    """Test topology-based agent selection."""

    def setUp(self):
        self.agents = [
            Agent(name="A", role="Agent A"),
            Agent(name="B", role="Agent B"),
            Agent(name="C", role="Agent C"),
        ]

    def test_round_robin_cycles(self):
        selector = AgentSelector(self.agents, Topology.ROUND_ROBIN)
        history = [Message(sender="user", content="start")]

        selected_names = []
        for _ in range(6):
            agents = selector.select_next(history)
            self.assertEqual(len(agents), 1)
            selected_names.append(agents[0].name)

        # Should cycle: A, B, C, A, B, C
        self.assertEqual(selected_names, ["A", "B", "C", "A", "B", "C"])

    def test_broadcast_excludes_last_sender(self):
        selector = AgentSelector(self.agents, Topology.BROADCAST)
        history = [Message(sender="A", content="hello")]

        selected = selector.select_next(history)
        selected_names = [a.name for a in selected]
        self.assertNotIn("A", selected_names)
        self.assertIn("B", selected_names)
        self.assertIn("C", selected_names)

    def test_coordinator_speaks_first(self):
        coord = self.agents[0]
        selector = AgentSelector(self.agents, Topology.COORDINATOR, coordinator=coord)
        history = [Message(sender="user", content="start")]

        selected = selector.select_next(history)
        self.assertEqual(selected[0].name, "A")


class TestTermination(unittest.TestCase):
    """Test termination conditions."""

    def test_max_rounds(self):
        term = MaxRoundsTermination(3)
        history = [Message(sender="x", content="msg") for _ in range(2)]
        self.assertFalse(term.should_stop(history))
        history.append(Message(sender="x", content="msg"))
        self.assertTrue(term.should_stop(history))

    def test_consensus_detection(self):
        term = ConsensusTermination(required_agreements=2)
        history = [
            Message(sender="A", content="Some discussion"),
            Message(sender="B", content="More discussion"),
            Message(sender="A", content="I agree with this"),
            Message(sender="B", content="Looks good to me"),
        ]
        self.assertTrue(term.should_stop(history))

    def test_consensus_not_enough(self):
        term = ConsensusTermination(required_agreements=3)
        history = [
            Message(sender="A", content="I agree"),
            Message(sender="B", content="Looks good"),
        ]
        self.assertFalse(term.should_stop(history))

    def test_keyword_termination(self):
        term = KeywordTermination("FINAL ANSWER")
        history = [Message(sender="A", content="The FINAL ANSWER is 42")]
        self.assertTrue(term.should_stop(history))

    def test_keyword_not_in_last(self):
        term = KeywordTermination("DONE")
        history = [
            Message(sender="A", content="DONE with step 1"),
            Message(sender="B", content="Moving to step 2"),
        ]
        self.assertFalse(term.should_stop(history))


class TestConversationManager(unittest.TestCase):
    """Test the full conversation manager."""

    def test_conversation_runs_and_terminates(self):
        agents = [
            Agent(name="A", role="Role A", behaviors={"start": "Response from A"}),
            Agent(name="B", role="Role B", behaviors={"response": "Response from B"}),
        ]
        manager = ConversationManager(
            agents=agents,
            topology=Topology.ROUND_ROBIN,
            termination_conditions=[MaxRoundsTermination(6)],
            verbose=False,
        )
        history = manager.run("start the conversation")
        self.assertGreater(len(history), 1)
        # Should terminate (not infinite loop)
        self.assertLessEqual(len(history), 10)

    def test_conversation_summary(self):
        agents = [Agent(name="X", role="Test")]
        manager = ConversationManager(
            agents=agents,
            termination_conditions=[MaxRoundsTermination(3)],
            verbose=False,
        )
        manager.run("hello")
        summary = manager.get_summary()
        self.assertIn("X", summary)

    def test_keyword_stops_conversation(self):
        agents = [
            Agent(name="A", role="Ender", behaviors={
                "hello": "FINAL ANSWER: done"
            }),
        ]
        manager = ConversationManager(
            agents=agents,
            topology=Topology.ROUND_ROBIN,
            termination_conditions=[
                MaxRoundsTermination(20),
                KeywordTermination("FINAL ANSWER"),
            ],
            verbose=False,
        )
        history = manager.run("hello world")
        # Should stop early due to keyword, not hit max rounds
        self.assertLess(len(history), 10)


if __name__ == "__main__":
    unittest.main()
